#!/usr/bin/env bash
# Start tyger v1 without Docker. Two processes, one venv (.venv, see requirements.txt):
#   1. MOUTH  (app.mouth): the orchestrator + BOTH models in-process on vLLM engines -- the LLM (Gemma 4 E4B,
#      vLLM AsyncLLM) and VoxCPM2 (vLLM-Omni AsyncOmni). Chunks the reply and streams TTS audio back to EARS
#      over TCP (:8770).
#   2. EARS   (app.ears): nemotron + endpoint head + the web app (:7860): the new page at /, the classic page at
#      /classic. The new page is web/'s static build (web/out/), made here first when it's missing or out of date.
#      In API mode (TYGER_PAGES=false) there are no pages and no build: only /ws and /voices (README -> "API mode").
#   With CLOUDFLARE_TUNNEL_TOKEN set, also cloudflared: a Cloudflare Tunnel to EARS (README -> "RunPod + Cloudflare").
# MOUTH starts first so its engines claim their (pinned) GPU memory before EARS loads (DEVELOPING.md -> "GPU memory").
#
# Reach it: the web app serves on :7860 (config/default.yaml server.port). The mic needs a secure context, so either
#   - SSH port-forward 7860 to your localhost and open http://localhost:7860, or
#   - put it behind an HTTPS proxy, or a Cloudflare Tunnel (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# secrets + overrides from .env (HF_TOKEN, OPENAI_API_KEY); never printed
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"
# a first start JIT-compiles GPU kernels, one compiler per CPU thread at 3-6 GB each, which ran a 64 GB box out of RAM;
# with these the peak was 32 GiB (DEVELOPING.md -> "On the GPU box"). Anything already set wins.
export MAX_JOBS="${MAX_JOBS:-4}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}" \
       TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
CFG_ARGS=(); [ -n "${CONFIG:-}" ] && CFG_ARGS=(--config "$CONFIG")   # another config (default: config/default.yaml)
MOUTH_PORT="${MOUTH_PORT:-8770}"   # must match config/default.yaml server.mouth_port
[ -x "$PY" ] || { echo "venv not found at $VENV (see README install)"; exit 1; }

# The new page: build it when web/out/ has no build or one older than the page's sources; install its packages only
# when they're missing or the lockfile changed. Without npm, or if the build fails, start anyway: EARS then serves the
# last build, or the classic page at / as well.
build_page() {
  local index=web/out/index.html served="the classic page"
  if [ -f "$index" ]; then
    local newer; newer=$(find web/app web/components web/lib web/*.json web/*.mjs -newer "$index" -print -quit || true)
    [ -z "$newer" ] && return 0
    served="the new page's last build"
  fi
  if ! command -v npm >/dev/null; then
    echo "npm not found: the new page can't be built, so / serves $served (README -> \"Run & reach it\")"; return 0
  fi
  echo "=== building the page (web/) ==="
  local installed=web/node_modules/.package-lock.json       # npm's record of what it installed
  if [ ! -f "$installed" ] || [ web/package-lock.json -nt "$installed" ]; then
    (cd web && npm ci --no-audit --no-fund) || { echo "npm ci failed: / serves $served"; return 0; }
  fi
  (cd web && npm run build) || echo "the page build failed (above): starting anyway"
}
# server.pages, with TYGER_PAGES over it (API mode: no pages, so no Node needed); a config error stops here, with its key
if [ "$("$PY" -m app.config "${CFG_ARGS[@]}" server.pages)" = "true" ]; then
  build_page
else
  echo "API mode (server.pages / TYGER_PAGES off): no pages, so no page build; EARS serves /ws and /voices only"
fi

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

echo "=== starting MOUTH (LLM + VoxCPM2, in-process) ==="
"$PY" -m app.mouth "${CFG_ARGS[@]}" &
PIDS+=($!)
wait_tcp "$MOUTH_PORT" "MOUTH" "${PIDS[-1]}"

echo "=== starting EARS + web app ==="
# expandable segments: EARS encodes windows of many lengths; without it the allocator cache fragments and grows
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "$PY" -m app.ears "${CFG_ARGS[@]}" &
PIDS+=($!)
EARS_PID=$!

# A Cloudflare Tunnel to EARS when CLOUDFLARE_TUNNEL_TOKEN is set (README -> "RunPod + Cloudflare"): cloudflared
# connects out to Cloudflare, which serves the tunnel's hostname over HTTPS and wss. It starts once EARS listens, so
# the hostname answers only when the stack is ready. The token goes in through the env, never the command line (ps
# shows that). Keep cloudflared's log level at info: at debug it logs request URLs, and so the conversations' tokens.
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
  if command -v cloudflared >/dev/null; then
    wait_tcp "$("$PY" -m app.config "${CFG_ARGS[@]}" server.port)" "EARS" "$EARS_PID"
    echo "=== starting the Cloudflare Tunnel ==="
    TUNNEL_TOKEN="$CLOUDFLARE_TUNNEL_TOKEN" cloudflared tunnel --no-autoupdate run &
    PIDS+=($!)
  else
    echo "CLOUDFLARE_TUNNEL_TOKEN is set, but cloudflared isn't installed: no tunnel (the Docker image has it)"
  fi
fi
wait "$EARS_PID"
