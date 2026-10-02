# tyger v1 — a deployable full-duplex voice agent on one GPU

Real-time spoken conversation in the browser: your mic → live turn-taking → a reasoning LLM with web tools
→ a cloned voice, all on a single GPU. The agent knows when you've finished a turn, when you're interrupting
it, and when to drop in a quick "mm-hm" — then answers in a voice cloned from a reference clip you provide.

This repo is the **inference / serving** project. The turn-taking model itself (the endpoint head and the
chain it sits on) is trained elsewhere; v1 only *runs* it. `NOTES.md` has the full design rationale and the
benchmark history behind every choice here.

```
browser mic ──ws──▶ EARS    nemotron L15 → endpoint head: endpoint / barge / backchannel decisions
                     │                                    also serves the web app on :7860
                     ▼ tcp :8770
                    MOUTH   orchestrator + two in-process vLLM engines on one asyncio loop:
                     │        • Qwen3-4B-Instruct-2507 (vLLM AsyncLLM, Hermes tool calls)
                     │        • VoxCPM2 voice clone (vLLM-Omni AsyncOmni, 48 kHz → 24 kHz)
                     ▼ ws
browser speaker ◀────┘      24 kHz audio, streamed as it decodes
```

Two processes, **one venv** (`.venv`, `requirements.txt`). **One GPU, one conversation at a time.** A second
visitor gets a "busy" page. This is a single-session demo server, not a multi-tenant service — scaling out is
the big open item (see `HANDOFF.md`).

## How it works

1. **EARS** runs the streaming ASR encoder (nemotron) and, on top of its L15 hidden state, a set of per-frame
   heads that decide, every 80 ms: has the user's turn **ended** (take the floor), is the user **barging in**
   over the agent (stop talking), and should the agent emit a **backchannel** ("right", "mm-hm") while still
   listening. EARS serves the web page and the WebSocket.
2. **MOUTH** is the orchestrator, and hosts both models in-process (no HTTP hop to either). On an endpoint it
   streams the LLM reply, cuts it into text chunks, and sends each chunk to VoxCPM2 as soon as it closes;
   audio streams back to EARS (and on to the browser) per decode step. The **first** chunk may end at a
   clause (comma, semicolon, dash) so playback doesn't wait for a long opening sentence; later chunks are
   whole sentences; no chunk is cut below 40 characters, so short replies go to the TTS whole. On a tool call
   it speaks a short lead-in ("let me check that") that overlaps the tool latency, then speaks the result.
   On a barge it stops within one audio chunk and aborts the live LLM/TTS request in the engine.

## Hardware

Built on one **NVIDIA RTX 5090** (32 GB, Blackwell sm_120), CUDA-13 driver (≥ 590) for the cu130 torch wheels.

GPU memory in use ≈ **27 GB of 32 GB** with everything loaded, warm, and under a 4-session MOUTH load: VoxCPM2
~9 GB (2 GiB pinned KV, `deploy/voxcpm2.yaml`), Qwen3-4B ~12 GB (3 GiB pinned KV, `llm.kv_cache_gib`), EARS
~5 GB. Both engines pin their KV budgets, so the total is predictable.

## Setup

### 1. Install (one venv)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=auto
```

### 2. Secrets — `.env`

Copy `.env.example` to `.env` (gitignored) and fill in:

- **`TAVILY_API_KEY`** — powers `web_search` and `get_weather`. Without it those tools return an error and the
  model says it couldn't look it up; it never invents a value.
- **`HF_TOKEN`** — pulls the **private** endpoint-head weights (`heads_repo` in `config.yaml`). If a token
  already exists at `$HF_HOME/token`, leave this blank.

Never commit `.env`.

### 3. Voices

Put one or more reference `.wav` clips in `voices/` (`voices.dir` in `config.yaml`; gitignored). The stem is the
voice name; `voices.default` picks the active one (falls back to the first clip). Optionally add
`voices/<name>.txt` with the clip's exact transcript: VoxCPM2 then clones from audio + transcript (better
quality, uses the whole clip); without it the clip is trimmed to `tts.ref_max_s` (10 s). `?debug=1` in the
browser shows a voice picker. **Reference clips and transcripts are personal data: keep them on the box, and
never commit, upload, or redistribute them.**

### 4. Configure

`config.yaml` holds model ids, engine memory budgets, endpoint thresholds, the voice directory, and runtime
flags. No secrets live here. Worth knowing:

- **Endpoint thresholds** (`ears.ep_thr` etc.) ship at the training-time defaults and have **not** been
  retuned on a live browser mic — offline thresholds have been seen to run ~2× too high on real mic signal.
  Expect to tune these live with `?debug=1`, then freeze the values back into `config.yaml`.
- **Backchannels** are always **contextual LLM acks**: a short reaction the LLM generates from the running
  ASR partial ("oh nice", "wow, really?"). When you've said too little to react to (`mouth.bc_min_words`),
  it uses a neutral "mm-hm". How *often* they fire is EARS's `ears.bc_thr` (slider under `?debug=1`).

## Run & reach it

```bash
./run.sh     # starts MOUTH (both engines + TTS warmup), waits for it, then EARS + the web app on :7860
CONFIG=/path/to/other.yaml ./run.sh   # alternate config for both processes
```

Startup from a warm cache is ~2.5 min (VoxCPM2 ~55 s, Qwen3-4B ~30 s, TTS warmup incl. batch sizes 2–4
~40 s, EARS ~15 s). The very
first start also downloads the models and pays one-time compiles (VoxCPM2's first warmup ~2 min, Qwen's
CUDA-graph capture), which are cached for later starts.

Browsers only allow the mic over **HTTPS or localhost**, so either:

- **SSH port-forward** (simplest): `ssh -L 7860:localhost:7860 <box>`, then open `http://localhost:7860`.
- **HTTPS** behind a reverse proxy (e.g. Caddy: `tyger.example.com { reverse_proxy localhost:7860 }`). The
  page upgrades the WebSocket to `wss` automatically when served over HTTPS.

`?debug=1` reveals the live tuning sliders (endpoint τ / debounce, barge, backchannel, silence fallback) and
the voice picker. Browser DSP (echo cancellation, noise suppression, auto gain) is on.

### Tests

```bash
.venv/bin/python tests/test_chunker.py          # text chunking (no GPU)
.venv/bin/python tests/test_llm_parse.py        # tool-call text splitting (no GPU)
.venv/bin/python scripts/smoke_test.py          # needs run.sh up: MOUTH tool turns + EARS speech → endpoint → audio
```

The EARS half of the smoke test uses the MOUTH's own TTS reply (held in memory) as the "user" speech, so no
test audio file is needed; pass `--wav` to use your own.

## Models & licences

| component | model | licence |
|---|---|---|
| ASR encoder + turn-taking | `nvidia/nemotron-3.5-asr-streaming-0.6b` + endpoint head (private) | NVIDIA Open Model Licence / private head |
| responder | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| TTS | `openbmb/VoxCPM2` | Apache-2.0 |
| web tools | Tavily API | commercial API (your key) |

## Privacy

No audio or transcript is written to disk or logs. Logs carry only timings and event types (endpoint source,
onset ms, barge, backchannel, tool name). Voice reference clips and their transcripts stay on the box.

## Known limits (v1)

- **One conversation at a time in EARS.** MOUTH already serves several sessions concurrently (and runs
  backchannels alongside turns); EARS's batched multi-session step is in progress (see `NOTES.md`).
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- Plain-turn onset ~500 ms, tool turn ~750 ms (smoke test, cloned voice) on a box whose GPU is stuck at half
  memory clock — expect better on a healthy GPU (see `NOTES.md`). VoxCPM2 re-encodes the reference clip on
  every request; a precomputed voice profile would cut that.
- Endpoint thresholds are not yet live-tuned (see Configure).
- The `Dockerfile` has not yet been built and run end to end.

See `HANDOFF.md` for the original work queue and `NOTES.md` for the current state.
