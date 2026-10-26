"""Mutation check: do the tests notice when the code is wrong?

Makes one small change at a time to a module (a flipped comparison, + for -, a constant off by one, argmin for
argmax, min for max, True for False, a dropped ``not`` or minus sign), runs the given tests on the changed copy,
and reports every change the tests did not catch ("survivors"). A survivor is either a missing test or a change
that cannot alter behaviour; each one is looked at before the code is trusted for an expensive run.

The repo is copied to a temporary folder first (tracked and untracked-but-not-ignored files, without the
``--skip_dirs`` result folders), so the working tree is never modified, even if the check is interrupted.

    python tools/mutation_check.py noise_reachability/noise_selection.py tests/test_noise_selection.py
    python tools/mutation_check.py noise_reachability/analyze_step2.py tests/test_analyze_step2.py \
        --functions choose report_stage systematic_rate

Exit code 0 if no mutant survived, 1 otherwise.
"""
import argparse
import ast
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
SWAP_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Mult,
            ast.Pow: ast.Mult}
SWAP_NAME = {"argmin": "argmax", "argmax": "argmin", "min": "max", "max": "min", "any": "all", "all": "any"}


class Sites(ast.NodeVisitor):
    """Collects (node path, description) of every mutable spot inside the selected functions."""

    def __init__(self, functions):
        self.functions, self.stack, self.sites = functions, [], []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def active(self):
        return self.stack and (not self.functions or any(f in self.functions for f in self.stack))

    def add(self, node, what, which=0):
        if self.active():
            self.sites.append((node, f"line {node.lineno} in {self.stack[-1]}(): {what}", which))

    def visit_Compare(self, node):
        for k, op in enumerate(node.ops):
            if type(op) in SWAP_CMP:
                self.add(node, f"comparison {type(op).__name__} -> {SWAP_CMP[type(op)].__name__}", k)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in SWAP_BIN:
            self.add(node, f"operator {type(node.op).__name__} -> {SWAP_BIN[type(node.op)].__name__}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, (ast.Not, ast.USub)):
            self.add(node, f"dropped {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            self.add(node, f"constant {node.value} -> {not node.value}")
        elif isinstance(node.value, (int, float)):
            self.add(node, f"constant {node.value!r} -> {node.value + 1!r}")

    def visit_Name(self, node):
        if node.id in SWAP_NAME and isinstance(node.ctx, ast.Load):
            self.add(node, f"name {node.id} -> {SWAP_NAME[node.id]}")

    def visit_Attribute(self, node):
        if node.attr in SWAP_NAME:
            self.add(node, f"attribute .{node.attr} -> .{SWAP_NAME[node.attr]}")
        self.generic_visit(node)


def mutate(tree, index):
    """A copy of ``tree`` with mutation ``index`` (in Sites order) applied."""
    sites = Sites(FUNCTIONS)
    sites.visit(tree)
    target = sites.sites[index]
    new = copy.deepcopy(tree)
    # Locate the same node in the copy: walk both trees in the same order.
    mapping = dict(zip(map(id, ast.walk(tree)), ast.walk(new)))
    node = mapping[id(target[0])]
    if isinstance(node, ast.Compare):
        k = target[2]
        node.ops[k] = SWAP_CMP[type(node.ops[k])]()
    elif isinstance(node, ast.BinOp):
        node.op = SWAP_BIN[type(node.op)]()
    elif isinstance(node, ast.UnaryOp):
        return replace_node(new, node, node.operand), target[1]
    elif isinstance(node, ast.Constant):
        node.value = (not node.value) if isinstance(node.value, bool) else node.value + 1
    elif isinstance(node, ast.Name):
        node.id = SWAP_NAME[node.id]
    elif isinstance(node, ast.Attribute):
        node.attr = SWAP_NAME[node.attr]
    return new, target[1]


def replace_node(tree, old, new):
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, field, new)
                return tree
            if isinstance(value, list) and any(v is old for v in value):
                value[[v is old for v in value].index(True)] = new
                return tree
    raise ValueError("node not found")


def copy_repo(dest, skip_dirs):
    files = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    for rel in files:
        if rel.split("/")[0] in skip_dirs or not (REPO_ROOT / rel).is_file():
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dest / rel)


def run_tests(root, tests, timeout):
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests]
    try:
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return r.returncode == 0, r.stdout[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


FUNCTIONS = ()


def main(argv=None):
    global FUNCTIONS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module", help="repo-relative path of the module to mutate")
    p.add_argument("tests", nargs="+", help="repo-relative test files to run against each mutant")
    p.add_argument("--functions", nargs="*", default=[], help="mutate only inside these functions (default: all)")
    p.add_argument("--skip_dirs", nargs="*", default=["week1", "week2", "week3", "week4", "week5", "week6", "wandb",
                                                         "outputs", "latex", "logs"])
    p.add_argument("--timeout", type=float, default=300)
    a = p.parse_args(argv)
    FUNCTIONS = tuple(a.functions)

    source = (REPO_ROOT / a.module).read_text()
    tree = ast.parse(source)
    sites = Sites(FUNCTIONS)
    sites.visit(tree)
    n = len(sites.sites)
    print(f"{a.module}: {n} mutants; tests: {' '.join(a.tests)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="mutation_check_") as tmp:
        root = Path(tmp)
        copy_repo(root, set(a.skip_dirs))
        target = root / a.module
        target.write_text(ast.unparse(tree) + "\n")  # the unmutated module, as the mutants are written
        ok, out = run_tests(root, a.tests, a.timeout)
        if not ok:
            print("the tests fail on the unmutated code; fix that first\n" + out)
            return 2
        survivors, t0 = [], time.time()
        for k in range(n):
            mutant, what = mutate(tree, k)
            target.write_text(ast.unparse(mutant) + "\n")
            passed, _ = run_tests(root, a.tests, a.timeout)
            if passed:
                survivors.append(what)
            print(f"[{k + 1}/{n}] {'SURVIVED' if passed else 'killed  '} {what}", flush=True)
    print(f"\n{n - len(survivors)} of {n} mutants killed in {time.time() - t0:.0f} s; {len(survivors)} survived:")
    for s in survivors:
        print("  " + s)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
