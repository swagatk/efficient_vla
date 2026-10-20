#!/usr/bin/env bash
# Run a command at low priority with few threads and record its start, end and exit status (signal
# deaths included) in logs/crash_monitor/runs.log, so a crash can be matched to monitor.log samples.
#   bash tools/logged_run.sh <label> <command> [args...]
set -u
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/logs/crash_monitor}"
mkdir -p "$LOG_DIR"
label="$1"; shift
threads="${RUN_THREADS:-4}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
echo "$(date -Is) START [$label] pid $$ threads $threads: $*" >> "$LOG_DIR/runs.log"
start=$(date +%s)
nice -n 15 "$@"
rc=$?
dur=$(( $(date +%s) - start ))
if [ $rc -gt 128 ]; then how="killed by signal $((rc - 128))"; else how="exit $rc"; fi
echo "$(date -Is) END   [$label] $how after ${dur}s" >> "$LOG_DIR/runs.log"
exit $rc
