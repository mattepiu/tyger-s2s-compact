# tyger v1 — a deployable full-duplex voice agent on one GPU

Real-time spoken conversation in the browser: your mic → live turn-taking → a reasoning LLM with web tools
→ a cloned voice, all on a single GPU. The agent knows when you've finished a turn, when you're interrupting
it, and when to drop in a quick "mm-hm" — then answers in a voice cloned from a reference clip you provide.

This repo is the **inference / serving** project. The turn-taking model itself (the endpoint head and the
chain it sits on) is trained elsewhere; v1 only *runs* it. `NOTES.md` has the full design rationale and the
benchmark history behind every choice here.

```
browser mic ──ws──▶ EARS       nemotron L15 → endpoint head: endpoint / barge / backchannel decisions
                     │ (venv, transformers 5.17)              also serves the web app on :7860
                     ▼ tcp :8770
                    MOUTH       orchestrator + MOSS-TTS-Realtime voice clone (24 kHz), sentence-chunked
                     │ (venv_moss, transformers 5.0.0)
                     ▼ http :8001
                 RESPONDER      Qwen3-4B-Instruct-2507 + Hermes tool-calling, served by vLLM (OpenAI API)
                     │ (venv_vllm, torch 2.13)
                     ▼ ws
browser speaker ◀────┘          24 kHz audio, streamed as it decodes
```

**One GPU, one conversation at a time.** A second visitor gets a "busy" page. This is a single-session demo
server, not a multi-tenant service — scaling out is the big open item (see `HANDOFF.md`).

## How it works

1. **EARS** runs the streaming ASR encoder (nemotron) and, on top of its L15 hidden state, a set of per-frame
   heads that decide, every 80 ms: has the user's turn **ended** (take the floor), is the user **barging in**
   over the agent (stop talking), and should the agent emit a **backchannel** ("right", "mm-hm") while still
   listening. EARS serves the web page and the WebSocket.
2. **RESPONDER** is Qwen3-4B-Instruct-2507 on vLLM, exposing the standard OpenAI chat API with Hermes
   tool-calling. It's a separate process purely for latency: vLLM gives ~28 ms time-to-first-token versus the
   in-process greedy decode it replaced.
3. **MOUTH** is the orchestrator. On an endpoint it calls the responder, streams the reply **sentence by
   sentence** into MOSS-TTS-Realtime, and streams the resulting audio back to EARS (and on to the browser) as
   it decodes. On a tool call it speaks a short lead-in ("let me check that") that overlaps the tool latency,
   then speaks the result. On a barge it halts audio within one frame and cancels the generation.

### Why three processes / three virtual environments

The three models have mutually incompatible dependencies, so they can't share one environment:

| process | env | key pin | reason |
|---|---|---|---|
| EARS | `venv` | transformers **≥ 5.13** | nemotron-3.5-asr-streaming's processor isn't in older transformers |
| MOUTH | `venv_moss` | transformers **5.0.0** | MOSS-TTS-Realtime's `trust_remote_code` modeling targets it |
| RESPONDER | `venv_vllm` | torch **2.13** (vLLM) | vLLM pins its own torch |

`run.sh` starts all three and wires them over local TCP/HTTP.

## Hardware

Built and tested on one **NVIDIA RTX 5090** (32 GB, Blackwell sm_120), driver 590 / CUDA 13.1, 96-core CPU,
188 GB RAM. Needs a CUDA-13-capable driver (≥ 590) for the cu130 torch wheels.

GPU memory in use ≈ **30.5 GB of 32 GB** — MOSS-TTS-Realtime ~12.7 GB (incl. CUDA-graph pools) + vLLM-Qwen
~11.8 GB (at `--gpu-memory-utilization 0.37`) + nemotron fp32 ~5.4 GB. It fits, but with only ~2 GB of
headroom, so `VLLM_GPU_FRAC` must stay ≤ ~0.37 (0.28 starves vLLM's KV cache and the engine fails to start).
This tight fit is the main constraint to keep in mind when changing any model.

## Setup

### 1. Install (three venvs)

From the project root:

```bash
python -m venv /workspace/venv_vllm && /workspace/venv_vllm/bin/pip install -r requirements-vllm.txt
python -m venv /workspace/venv_moss && /workspace/venv_moss/bin/pip install -r requirements-moss.txt
python -m venv /workspace/venv      && /workspace/venv/bin/pip      install -r requirements-ears.txt
```

`run.sh` defaults to those three paths; override with `VLLM_VENV` / `MOUTH_VENV` / `EARS_VENV`.
`WeTextProcessing` (a MOSS dependency) sometimes fails to build from pip — see the MOSS-TTS-Realtime repo if
the MOUTH venv install errors.

### 2. Secrets — `.env`

Copy `.env.example` to `.env` (gitignored) and fill in:

- **`TAVILY_API_KEY`** — powers `web_search` and `get_weather`. Without it those tools return an error and the
  model says it couldn't look it up; it never invents a value.
- **`HF_TOKEN`** — pulls the **private** endpoint-head weights (`heads_repo` in `config.yaml`). If a token
  already exists at `$HF_HOME/token`, leave this blank.

Never commit `.env`. Confirm your repo's `.gitignore` covers `.env` and `*.wav` before the first commit.

### 3. Voices

Put one or more reference `.wav` clips in the voice directory (`voices.dir` in `config.yaml`, default
`/workspace/voices`), or point `voices.default_ref` at a single clip. Each clip is encoded once for cloning;
`?debug=1` in the browser shows a voice picker. **Reference clips are personal data: keep them on the box,
and never commit, upload, or redistribute them.**

> `config.yaml` ships with `voices.default_ref` pointing at a path from the original box — set it to your own
> clip (or rely on `voices.dir`) before first run.

### 4. Configure

`config.yaml` holds model ids, HF paths, endpoint thresholds, the voice directory, and runtime flags. No
secrets live here. Two things worth knowing:

- **Endpoint thresholds** (`ears.ep_thr` etc.) ship at the training-time defaults and have **not** been
  retuned on a live browser mic — offline thresholds have been seen to run ~2× too high on real mic signal.
  Expect to tune these live with `?debug=1`, then freeze the values back into `config.yaml`.
- **Backchannels** default to **contextual LLM acks** (`mouth.contextual_acks: true`): a short reply the LLM
  generates from the running ASR partial, so the ack reacts to what you're saying. Set it `false` for fast
  pre-rendered canned acks instead (lower latency, generic wording).

## Run & reach it

```bash
./run.sh     # starts vLLM, waits for it, then MOUTH, then EARS + the web app on :7860
```

The vLLM responder takes a few minutes on first start (engine init + CUDA-graph capture); MOUTH adds a warmup
pass to capture the TTS CUDA graphs once up front. Startup from a warm model cache is roughly: vLLM 1–5 min,
MOUTH ~2.5 min, EARS ~15 s. A cold cache adds the model downloads.

Browsers only allow the mic over **HTTPS or localhost**, so either:

- **SSH port-forward** (simplest): `ssh -L 7860:localhost:7860 <box>`, then open `http://localhost:7860`.
- **HTTPS** behind a reverse proxy (e.g. Caddy: `tyger.example.com { reverse_proxy localhost:7860 }`). The
  page upgrades the WebSocket to `wss` automatically when served over HTTPS.

`?debug=1` reveals the live tuning sliders (endpoint τ / debounce, barge, backchannel, silence fallback) and
the voice picker. Browser DSP (echo cancellation, noise suppression, auto gain) is on.

### Smoke test (no browser)

```bash
/workspace/venv/bin/python scripts/smoke_test.py   # MOUTH tool turns + EARS wav → endpoint → audio
```

## Models & licences

| component | model | licence |
|---|---|---|
| ASR encoder + turn-taking | `nvidia/nemotron-3.5-asr-streaming-0.6b` + endpoint head (private) | NVIDIA Open Model Licence / private head |
| responder | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| TTS | `OpenMOSS-Team/MOSS-TTS-Realtime` + `MOSS-Audio-Tokenizer` | Apache-2.0 |
| web tools | Tavily API | commercial API (your key) |

The MOSS-TTS-Realtime remote code is **vendored and patched** at startup (a static-cache + CUDA-graph fast
path; see `NOTES.md`). The stock code runs at RTF ~1.6–2.0 on this GPU, which is too slow to stream; the patch
brings it to ~0.55–0.7.

## Privacy

No audio or transcript is written to disk or logs. Logs carry only timings and event types (endpoint source,
onset ms, barge, backchannel, tool name). Voice reference clips stay on the box.

## Known limits (v1)

- **One speaker, one conversation at a time.** No concurrency.
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- TTS onset is **~480 ms** end-to-end on a plain turn, ~580 ms on a tool turn; steady RTF ~0.56–0.7 (streams
  ahead of playback). Lower `tts.prefill_text_len` / `tts.dec_chunk_frames` to trade pronunciation for a
  faster first audio.
- GPU headroom is ~2 GB; any larger model risks OOM.
- Endpoint thresholds are not yet live-tuned (see Configure).

See `HANDOFF.md` for the current state and what to pick up next.
