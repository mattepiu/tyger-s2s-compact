#!/usr/bin/env bash
# Start tyger v1 without Docker. Two processes, one venv (.venv, see requirements.txt):
#   1. MOUTH  (app.mouth): the orchestrator + BOTH models in-process on vLLM engines -- Qwen3-4B-Instruct-2507
#      (vLLM AsyncLLM) and VoxCPM2 (vLLM-Omni AsyncOmni). Chunks the reply and streams TTS audio back to EARS
#      over TCP (:8770).
#   2. EARS   (app.server): nemotron + endpoint head + the web app (:7860).
# MOUTH starts first so its engines claim their (pinned) GPU memory before EARS loads. See NOTES.md.
#
# Reach it: the web app serves on :7860 (config.yaml server.port). The mic needs a secure context, so either
#   - SSH port-forward 7860 to your localhost and open http://localhost:7860, or
#   - put it behind an HTTPS proxy (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# secrets + overrides from .env (TAVILY_API_KEY, HF_TOKEN); never printed
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
CFG_ARGS=(); [ -n "${CONFIG:-}" ] && CFG_ARGS=(--config "$CONFIG")   # optional alternate config.yaml
MOUTH_PORT="${MOUTH_PORT:-8770}"   # must match config.yaml server.mouth_port
[ -x "$PY" ] || { echo "venv not found at $VENV (see README install)"; exit 1; }

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_tcp() {  # $1 = port, $2 = label, $3 = pid that must stay alive
  echo "waiting for $2 on :$1 ..."
  for _ in $(seq 1 900); do
    kill -0 "$3" 2>/dev/null || { echo "$2 exited during start-up (see log above)"; exit 1; }
    if "$PY" - "$1" <<'PY' 2>/dev/null; then return 0; fi
import socket,sys
s=socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)
PY
    sleep 1
  done
  echo "$2 did not come up on :$1"; exit 1
}

echo "=== starting MOUTH (Qwen3-4B + VoxCPM2, in-process) ==="
"$PY" -m app.mouth "${CFG_ARGS[@]}" &
PIDS+=($!)
wait_tcp "$MOUTH_PORT" "MOUTH" "${PIDS[-1]}"

echo "=== starting EARS + web app ==="
"$PY" -m app.server "${CFG_ARGS[@]}" &
PIDS+=($!)
wait "${PIDS[-1]}"
