#!/usr/bin/env bash
# The same measurement sequence for each stack (e.g. one responder against another), as in NOTES.md 2026-10-05.
#   scripts/bench_stack.sh TAG CONFIG STACK_LOG   (start the stack first: CONFIG=... setsid ./run.sh > STACK_LOG 2>&1 &)
# Waits for "EARS ready" in STACK_LOG, then smoke, load_test x3, --tools --fresh x3, load_test_ears 1 conv x2 and
# 4 conv x3, GPU memory idle and after load_test. Logs go to bench_runs/TAG/ (gitignored); compare tags with
# scripts/bench_summary.py TAG [TAG...]. Use a config without llm.smart unless the user OKs OpenAI calls.
set -u
TAG=$1; CFG=$(realpath "$2"); STACK=$(realpath "$3")
cd "$(dirname "$0")/.."
OUT=bench_runs/$TAG; mkdir -p "$OUT"
for _ in $(seq 240); do grep -q "EARS ready" "$STACK" && break; grep -q "exited during\|did not come up" "$STACK" && { echo "stack failed"; exit 1; }; sleep 5; done
grep -q "EARS ready" "$STACK" || { echo "stack not ready"; exit 1; }
PY=.venv/bin/python
nvidia-smi --query-gpu=memory.used,pstate,clocks.sm,clocks.mem --format=csv,noheader > "$OUT/gpu_idle.txt"
$PY scripts/smoke_test.py --config "$CFG" > "$OUT/smoke.log" 2>&1
for r in 1 2 3; do $PY scripts/load_test.py --config "$CFG" --sessions 4 --turns 6 > "$OUT/lt_plain_$r.log" 2>&1; done
for r in 1 2 3; do $PY scripts/load_test.py --config "$CFG" --sessions 4 --turns 6 --tools --fresh > "$OUT/lt_tools_$r.log" 2>&1; done
nvidia-smi --query-gpu=memory.used,pstate,clocks.sm,clocks.mem --format=csv,noheader > "$OUT/gpu_after_lt.txt"
for r in 1 2; do $PY scripts/load_test_ears.py --config "$CFG" --sessions 1 > "$OUT/ears1_$r.log" 2>&1; done
for r in 1 2 3; do $PY scripts/load_test_ears.py --config "$CFG" --sessions 4 > "$OUT/ears4_$r.log" 2>&1; done
echo "bench $TAG done"
