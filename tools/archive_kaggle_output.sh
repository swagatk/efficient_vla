#!/usr/bin/env bash
# Download the output of one of your Kaggle notebooks (adapters, recordings, everything the results zip leaves out)
# into a new folder, and write a sha256 manifest of what arrived.
#
#   bash tools/archive_kaggle_output.sh <kernel slug>[/<version>] <archive root>
#   e.g. bash tools/archive_kaggle_output.sh noise-reachability-step-10 ~/archive/efficient_vla
#
# Without /<version>, Kaggle serves the latest version's output. The CLI is in its own venv (~/venvs/kaggle; set
# KAGGLE to another binary) and runs with PYTHONPATH cleared, so the shell's ROS paths do not leak in. Needs an API
# token in ~/.kaggle/kaggle.json (chmod 600); `kaggle kernels list --mine` prints your slugs. The CLI downloads one
# page of files per call, so this loops over pages until no next-page token is printed.
# Then check the download against the committed run: tools/check_archive.py.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <kernel slug>[/<version>] <archive root>" >&2; exit 2; }
KERNEL=$1
DEST=$2/${KERNEL//\//_v}
KAGGLE_USER=${KAGGLE_USER:-swagatk}
KAGGLE=${KAGGLE:-$HOME/venvs/kaggle/bin/kaggle}
k() { env -u PYTHONPATH PYTHONNOUSERSITE=1 "$KAGGLE" "$@"; }
[ -x "$KAGGLE" ] || { echo "no Kaggle CLI at $KAGGLE" >&2; exit 2; }
if [ -e "$DEST" ]; then echo "$DEST exists; choose a new archive root or remove it yourself" >&2; exit 2; fi
mkdir -p "$DEST/output"
k kernels status "$KAGGLE_USER/$KERNEL" | tee "$DEST/kernel_status.txt"
token="" page=0
while :; do
    page=$((page + 1))
    args=(kernels output "$KAGGLE_USER/$KERNEL" -p "$DEST/output" -q --page-size 200)
    [ -n "$token" ] && args+=(--page-token "$token")
    out=$(k "${args[@]}" 2>&1) || { echo "$out" | tee -a "$DEST/download.log" >&2; exit 1; }
    echo "$out" >> "$DEST/download.log"
    token=$(echo "$out" | sed -n 's/^Next page token: *//p' | tail -1)
    echo "page $page: $(find "$DEST/output" -type f | wc -l) files so far"
    [ -n "$token" ] || break
done
(cd "$DEST/output" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$DEST/MANIFEST.sha256"
echo "kernel $KAGGLE_USER/$KERNEL, $page pages, downloaded $(date -u +%FT%TZ)" > "$DEST/ARCHIVE.txt"
echo "$(wc -l < "$DEST/MANIFEST.sha256") files, $(du -sh "$DEST/output" | cut -f1) in $DEST/output"
