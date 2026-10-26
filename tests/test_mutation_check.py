import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import mutation_check  # noqa: E402

SOURCE = '''
def pick(xs, flag=True):
    if len(xs) < 2 and not flag:
        return -1
    return xs.index(min(xs)) + 1
'''


def mutants(functions=()):
    mutation_check.FUNCTIONS = tuple(functions)
    tree = ast.parse(SOURCE)
    sites = mutation_check.Sites(mutation_check.FUNCTIONS)
    sites.visit(tree)
    out = []
    for k in range(len(sites.sites)):
        new, what = mutation_check.mutate(tree, k)
        ns = {}
        exec(compile(new, "<mutant>", "exec"), ns)
        out.append((what, ns["pick"]))
    return out


def test_every_site_gives_a_different_valid_program():
    ms = mutants()
    kinds = " | ".join(w for w, _ in ms)
    for expected in ("Lt -> LtE", "constant 2 -> 3", "dropped Not", "constant True -> False", "name min -> max",
                     "operator Add -> Sub", "constant 1 -> 2"):
        assert expected in kinds
    # The original returns the index of the minimum; each mutant differs on at least one of these inputs.
    inputs = [([3, 1, 2], True), ([5], False), ([1, 2], False), ([4, 4, 0], True), ([5],)]
    ns = {}
    exec(SOURCE, ns)
    for what, f in ms:
        assert any(f(*x) != ns["pick"](*x) for x in inputs), what


def test_original_tree_is_not_modified_and_function_filter_works():
    tree = ast.parse(SOURCE)
    before = ast.dump(tree)
    mutation_check.FUNCTIONS = ()
    mutation_check.mutate(tree, 0)
    assert ast.dump(tree) == before
    assert mutants(["other"]) == []
