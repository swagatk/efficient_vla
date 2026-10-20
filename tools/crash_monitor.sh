#!/usr/bin/env bash
# System health log for investigating crashes on the local machine (see the local CPU fault notes).
# Appends one sample every INTERVAL seconds to $LOG_DIR/monitor.log and flags unclean restarts.
#   nohup bash tools/crash_monitor.sh >/dev/null 2>&1 &
# Commands run through tools/logged_run.sh record their start, end and exit status in $LOG_DIR/runs.log.
set -u
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/logs/crash_monitor}"
INTERVAL="${INTERVAL:-30}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/monitor.log"; STATE="$LOG_DIR/.last_sample"; PIDF="$LOG_DIR/monitor.pid"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "crash_monitor already running (pid $(cat "$PIDF"))"; exit 0
fi
echo $$ > "$PIDF"

boot_id=$(cat /proc/sys/kernel/random/boot_id)
if [ -f "$STATE" ]; then
  read -r last_boot last_time last_clean < "$STATE"
  if [ "$last_boot" != "$boot_id" ] && [ "$last_clean" != "clean" ]; then
    echo "=== $(date -Is) UNCLEAN RESTART: kernel/WSL rebooted since last sample at $last_time (boot $last_boot) ===" >> "$LOG"
  elif [ "$last_clean" != "clean" ]; then
    echo "=== $(date -Is) monitor was killed without a clean stop; last sample $last_time ===" >> "$LOG"
  fi
fi
echo "=== $(date -Is) monitor start pid $$ boot $boot_id interval ${INTERVAL}s ===" >> "$LOG"
trap 'echo "$boot_id $(date -Is) clean" > "$STATE"; echo "=== $(date -Is) monitor stop ===" >> "$LOG"; rm -f "$PIDF"; exit 0' INT TERM

last_dmesg=$(dmesg 2>/dev/null | wc -l)
while true; do
  {
    echo "--- $(date -Is) load $(cut -d' ' -f1-4 /proc/loadavg)"
    free -m | awk 'NR==2{printf "mem used %s/%s MB avail %s MB; ", $3,$2,$7} NR==3{printf "swap used %s MB\n", $3}'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu /'
    ps -eo pid,ni,pcpu,pmem,etime,comm --sort=-pcpu | sed -n '2,6p' | sed 's/^/  /'
    now_dmesg=$(dmesg 2>/dev/null | wc -l)
    if [ "$now_dmesg" -gt "$last_dmesg" ]; then
      dmesg 2>/dev/null | tail -n $((now_dmesg - last_dmesg)) | grep -v 'dxgkio_query_adapter_info' \
        | grep -iE 'mce|machine check|segfault|oom|killed|panic|error|hung|trap' | sed 's/^/  dmesg: /'
    fi
    last_dmesg=$now_dmesg
  } >> "$LOG"
  echo "$boot_id $(date -Is) running" > "$STATE"
  sync "$LOG" 2>/dev/null
  sleep "$INTERVAL" & wait $!
done
