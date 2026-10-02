#!/usr/bin/env bash
# Start tyger v1 without Docker. Three processes, three venvs:
#   1. RESPONDER  (venv_vllm): Qwen3-4B-Instruct-2507 served by vLLM (OpenAI API on :8001). The LLM runs
#      here, not in-process -- vLLM gives ~28 ms TTFT / ~150 tok/s, the latency win over greedy decode.
#   2. MOUTH      (venv_moss, tf 5.0.0): the orchestrator + in-process MOSS-TTS-Realtime. Talks to the
#      responder over HTTP, chunks the reply into sentences, and streams TTS audio back to EARS over TCP (:8770).
#   3. EARS       (venv, tf 5.17): nemotron + endpoint head + the web app (:7860).
# Three venvs because nemotron needs transformers >=5.13, MOSS-TTS-Realtime's remote code targets tf 5.0.0,
# and vLLM brings its own torch. See NOTES.md.
#
# Reach it: the web app serves on $PORT (default 7860). The mic needs a secure context, so either
#   - SSH port-forward 7860 to your localhost and open http://localhost:7860, or
#   - put it behind an HTTPS proxy (see README.md).
set -euo pipefail
cd "$(dirname "$0")"                      # v1/

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# secrets + overrides from v1/.env (TAVILY_API_KEY, HF_TOKEN); never printed
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"

EARS_VENV="${EARS_VENV:-/workspace/venv}"        # tf 5.17 (nemotron + endpoint head)
MOUTH_VENV="${MOUTH_VENV:-/workspace/venv_moss}" # tf 5.0.0 (MOSS-TTS-Realtime + orchestrator)
VLLM_VENV="${VLLM_VENV:-/workspace/venv_vllm}"   # vLLM (torch 2.13)
PORT="${PORT:-7860}"
MOUTH_PORT="${MOUTH_PORT:-8770}"
VLLM_PORT="${VLLM_PORT:-8001}"
RESPONDER="${RESPONDER:-Qwen/Qwen3-4B-Instruct-2507}"
VLLM_GPU_FRAC="${VLLM_GPU_FRAC:-0.37}"   # MOSS-Realtime's CUDA graphs need ~12.7GB; 0.37 leaves vLLM enough
                                         # KV (0.28 starves it -> engine init fails). GPU ends ~30.5/32GB.
VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"

[ -x "$EARS_VENV/bin/python" ]  || { echo "EARS venv not found at $EARS_VENV (see README install)"; exit 1; }
[ -x "$MOUTH_VENV/bin/python" ] || { echo "MOUTH venv not found at $MOUTH_VENV (see README install)"; exit 1; }
[ -x "$VLLM_VENV/bin/vllm" ]    || { echo "vLLM venv not found at $VLLM_VENV (see README install)"; exit 1; }

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_http() {  # $1 = port, $2 = label
  echo "waiting for $2 on :$1 ..."
  for _ in $(seq 1 600); do
    if "$EARS_VENV/bin/python" - "$1" <<'PY' 2>/dev/null; then return 0; fi
import socket,sys,urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=1); sys.exit(0)
except Exception:
    sys.exit(1)
PY
    sleep 1
  done
  echo "$2 did not come up on :$1"; exit 1
}

wait_tcp() {  # $1 = port, $2 = label
  echo "waiting for $2 on :$1 ..."
  for _ in $(seq 1 600); do
    if "$EARS_VENV/bin/python" - "$1" <<'PY' 2>/dev/null; then return 0; fi
import socket,sys
s=socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)
PY
    sleep 1
  done
  echo "$2 did not come up on :$1"; exit 1
}

echo "=== starting RESPONDER (vLLM: $RESPONDER) on :$VLLM_PORT ==="
FLASHINFER_DISABLE_VERSION_CHECK=1 "$VLLM_VENV/bin/vllm" serve "$RESPONDER" \
  --port "$VLLM_PORT" --gpu-memory-utilization "$VLLM_GPU_FRAC" \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len "$VLLM_MAX_LEN" &
PIDS+=($!)
wait_http "$VLLM_PORT" "RESPONDER"

echo "=== starting MOUTH (orchestrator + MOSS-TTS-Realtime) ==="
"$MOUTH_VENV/bin/python" -m app.mouth &
PIDS+=($!)
wait_tcp "$MOUTH_PORT" "MOUTH"

echo "=== starting EARS + web app on :$PORT ==="
exec "$EARS_VENV/bin/python" -m app.server
