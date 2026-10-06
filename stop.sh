#!/usr/bin/env bash
# Stop a running ./run.sh stack: SIGTERM its process group (MOUTH shuts its engines down), wait up to 30 s, then
# SIGKILL whatever in that group is still alive. A VoxCPM2 stage worker has been seen ignoring SIGTERM and staying
# on the GPU with ~9 GB. Don't use `pkill -f <pattern>` from a shell whose own command line contains the pattern.
set -u
pids=$(pgrep -f "^bash ./run.sh$" || true)
[ -z "$pids" ] && { echo "run.sh is not running"; exit 0; }
for p in $pids; do
  g=$(ps -o pgid= -p "$p" | tr -d ' ')
  kill -TERM -- "-$g" 2>/dev/null || true
  for _ in $(seq 30); do
    pgrep -g "$g" >/dev/null || break
    sleep 1
  done
  if pgrep -g "$g" >/dev/null; then
    echo "still alive after 30 s, killing: $(pgrep -g "$g" | tr '\n' ' ')"
    kill -KILL -- "-$g" 2>/dev/null || true
  fi
done
sleep 1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/GPU still used by: /'
echo "stopped"
