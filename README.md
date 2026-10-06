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
                     │        • Gemma 4 E4B FP8, text only, no thinking (vLLM AsyncLLM, tool calls)
                     │        • VoxCPM2 voice clone (vLLM-Omni AsyncOmni, 48 kHz → 24 kHz)
                     ▼ ws
browser speaker ◀────┘      24 kHz audio, streamed as it decodes
```

Two processes, **one venv** (`.venv`, `requirements.txt`). **One GPU, up to 4 conversations at once**
(`server.max_sessions`); the next visitor gets a "busy" page. It's a demo server with no auth (see `ISSUES.md`
I-11 before exposing it beyond an SSH tunnel).

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
   it speaks a short contextual filler ("Checking the current weather in Oslo.") the moment the call starts
   decoding, runs the tool as soon as the call parses, adds a "still looking" line if the tool is slow, then
   speaks the result. On a barge it stops within one audio chunk, aborts the live LLM/TTS requests, and keeps
   what the user heard of the reply in the conversation history.

## Hardware

Built on one **NVIDIA RTX 5090** (32 GB, Blackwell sm_120), CUDA-13 driver (≥ 590) for the cu130 torch wheels.

GPU memory in use ≈ **30.0 GB of 32 GB** with everything loaded, warm, and under a 4-conversation load: VoxCPM2
~9.8 GB (2 GiB pinned KV, `deploy/voxcpm2.yaml`), Gemma 4 E4B ~14.9 GB with FP8 weights (3 GiB pinned KV,
`llm.reply.kv_cache_gib`; bf16 weights would not fit), EARS ~7 GB. Both engines pin their KV budgets, so the total
is predictable. With the `gemma-4-e4b-nvfp4` backup (pre-quantized, int8 per-layer embeddings) the peak is ~26.7 GB.

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

- **Prompts** live in `prompts.yaml`, not in code: system prompts, few-shot turns, the small templates MOUTH
  fills in, and the tool descriptions. It is checked at MOUTH start (a bad key or placeholder stops start-up with
  the key named) and read once, so restart MOUTH after editing. `prompts/<llm instance>.yaml` (e.g.
  `prompts/gemma-4-e4b.yaml`) overrides parts of it for one model. Keep few-shot examples synthetic: the file is
  committed.

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

Startup from a warm cache is ~2.1 min (VoxCPM2 ~45 s, Gemma 4 E4B ~40 s, TTS warmup incl. batch sizes 2–4
~25 s, EARS ~15 s). The very first start also downloads the models (~25 GB) and pays one-time compiles (VoxCPM2's
first warmup ~2 min, Gemma's torch.compile + CUDA-graph capture ~1.5 min), which are cached for later starts.

Browsers only allow the mic over **HTTPS or localhost**, so either:

- **SSH port-forward** (simplest): `ssh -L 7860:localhost:7860 <box>`, then open `http://localhost:7860`.
- **HTTPS** behind a reverse proxy (e.g. Caddy: `tyger.example.com { reverse_proxy localhost:7860 }`). The
  page upgrades the WebSocket to `wss` automatically when served over HTTPS.

`?debug=1` reveals the live tuning sliders (endpoint τ / debounce, barge, backchannel, silence fallback) and
the voice picker. Browser DSP (echo cancellation, noise suppression, auto gain) is on.

### Tests

```bash
.venv/bin/python tests/test_chunker.py          # text chunking (no GPU)
.venv/bin/python tests/test_llm_parse.py        # tool-call splitting, both call formats, <think> guard (no GPU)
.venv/bin/python tests/test_mouth_session.py    # MOUTH turns/barge/tools/backchannels on the fake plugins (no GPU)
.venv/bin/python tests/test_config.py           # plugin config shape; the old shape is rejected (no GPU)
.venv/bin/python tests/test_prompts.py          # prompts.yaml loader + checks (no GPU)
.venv/bin/python tests/test_plugins.py          # plugin classes import light; off-rate TTS resampled (no GPU)
.venv/bin/python tests/test_plugin_contract.py  # plugin contract on the fakes; TYGER_CONTRACT_LOCAL=1 adds the real
                                                #   local plugins (GPU free: stop run.sh first)
.venv/bin/python tests/test_tasks.py            # background tasks + their announcement on a fake smart model (no GPU)
.venv/bin/python tests/test_ears_floor.py       # EARS floor requests: server.py's state machine, fake EARS (no GPU)
CONFIG=config.fake.yaml ./run.sh                 # the whole stack on the fake plugins: no GPU, no downloads
.venv/bin/python scripts/smoke_test.py          # needs run.sh up: MOUTH tool turns + barge-in + EARS speech → audio
.venv/bin/python scripts/load_test.py --sessions 4       # needs run.sh up: concurrent MOUTH turns + backchannels
                                                         #   --tools: tool turns (+ --fresh: empty history each)
.venv/bin/python scripts/load_test_ears.py --sessions 4  # needs run.sh up: N real-time WebSocket conversations;
                                                         #   hop lag, end of speech -> endpoint -> first audio
                                                         #   (fixed utterance and reply: runs compare)
.venv/bin/python scripts/validate_ears_batched.py        # GPU free: batched EARS == single-session reference
.venv/bin/python scripts/validate_rnnt.py                # GPU free: graphed RNN-T == nemotron generate()
.venv/bin/python scripts/validate_barge.py               # GPU free: a barge-in keeps the start of the utterance
.venv/bin/python scripts/validate_long_turn.py           # GPU free: > 20 s turns transcribed whole
.venv/bin/python scripts/bench_tts_voice.py              # GPU free: cloned-voice TTS first-audio per request
.venv/bin/python scripts/eval_escalation.py --samples 8  # GPU free: routing (local / tool / expert / background) and
                                                         #   reply language on 140 synthetic cases; local model only
.venv/bin/python scripts/bench_llm_weights.py <hf repo>  # GPU free: an LLM's GPU memory per module + single-stream tok/s
scripts/bench_stack.sh TAG CONFIG STACK_LOG              # needs run.sh up (CONFIG): smoke + load tests, logs in bench_runs/TAG
.venv/bin/python scripts/bench_summary.py TAG [TAG...]   # the numbers from bench_runs/, side by side
scripts/stop.sh                                          # stop run.sh (and any stage worker it leaves behind)
```

The EARS half of the smoke test speaks one fixed sentence with the MOUTH's TTS (held in memory) as the "user"
speech, so no test audio file is needed; pass `--wav` to use your own.

## Plugins

The LLM, the TTS and the turn-taking model are **plugins chosen in `config.yaml`** (`app/plugins/`; the design is
in `PLUGINS.md`). Each block names its `plugin:`; the rest of the block is that plugin's options.

| kind | plugin | runs | notes |
|---|---|---|---|
| LLM | `gemma-4-e4b` | local, vLLM in MOUTH | the default responder (`quantization: fp8`): thinking off, text only, Gemma tool-call tokens |
| LLM | `gemma-4-e4b-nvfp4` | local, vLLM in MOUTH | the backup: unsloth NVFP4 E4B (int8 per-layer embeddings), routes like E4B FP8, 3.6 GB less GPU, same latency; no `quantization` |
| LLM | `openai` | **hosted** | any OpenAI-compatible chat endpoint (`base_url`: OpenAI, Groq, Cerebras, a vLLM server) |
| LLM | `anthropic` | **hosted** | the Claude Messages API; default `claude-opus-5-5`, `effort: low`, server-side refusal fallback |
| TTS | `voxcpm2` | local, vLLM-Omni in MOUTH | cloned voices from `voices/` |
| EARS | `nemotron-tyger-1.0` | local, in EARS | nemotron ASR + the tyger 1.0 turn-taking heads |
| all three | `fake` | no model, CPU | scripted / energy-based; tests and `CONFIG=config.fake.yaml ./run.sh` |

Other local responders were measured against E4B and removed: their numbers are in `NOTES.md` ("Gemma 4 memory",
"NVFP4 responders", "unsloth NVFP4 Gemma 4", step 11), their plugins in git history.

**LLM roles.** `llm.reply` answers turns (and the tool phase-B answer); `llm.filler` makes backchannels, tool
lead-ins and still-looking lines. `filler: reply` (the default) shares one instance. Any role can use any LLM
plugin; two blocks with the same plugin need a `name:` each. Keys live in `.env` under the variable the block's
`api_key_env` names. Example, a hosted `reply` with the local model kept for the latency-critical `filler`:

```yaml
llm:
  reply: {plugin: openai, model: gpt-4.1-nano, api_key_env: OPENAI_API_KEY, max_tokens: 256}
  filler: {plugin: gemma-4-e4b, quantization: fp8, kv_cache_gib: 3.0, max_num_seqs: 8, temperature: 0.7}
```

Measured on the box (`NOTES.md` → step 6): a hosted `reply` adds ~0.35 s to first audio (OpenAI first text
~0.6 s against ~0.08 s local); a hosted `filler` makes backchannels ~0.55 s slower. With no local LLM at all
(`filler: reply` on a hosted plugin) MOUTH never loads vLLM for the LLM, which frees ~12 GB of GPU memory. A hosted
TTS was tried and removed: every text chunk paid a request's time to first audio (`NOTES.md` → step 8). A hosted
request that fails (rate limit, timeout, network) ends the turn with a short spoken apology; a refusal with a short
neutral line (both in `prompts.yaml` → `spoken`).

**Adding a plugin:** a module under `app/plugins/<llm|tts|ears>/` with a class that implements the protocol in
`app/plugins/base.py` (constructor `(conf, ctx)`; import heavy dependencies inside it, so the class stays light to
import), one line in `REGISTRY` (`app/plugins/__init__.py`), and a run in `tests/test_plugin_contract.py`.
Per-model prompt changes go in `prompts/<instance name>.yaml`, never in code.

## Escalation and background tasks

With an `llm.smart` block, the local `reply` model can hand work to a bigger hosted model. It decides itself, by
calling a tool, so easy turns pay nothing extra:

- **`ask_expert`** for a question that needs care now: maths, code, health / money / legal advice, comparisons,
  planning, long explanations, or "think carefully". The lead-in fits ("let me think about that"), the smart
  request starts the moment the call is read, and its answer streams straight to the voice (live: answer audio at
  ~1.0–1.3 s, under the lead-in). Follow-ups on the same topic go to it too.
- **`start_task`** for work that takes minutes: research, comparing many options, a multi-day plan, a long draft.
  The local model says it'll look into it and the conversation carries on. The task runs on the smart model in the
  background (its own web searches, `mouth.tasks.max_steps` rounds, `max_wall_s`). When it finishes, EARS gives it
  the floor once the user has been quiet for `announce_after_silence_ms`, and it speaks a short summary; the user
  can barge in as usual. "How's it going?" is answered locally from a status note; "drop it" cancels it.

```yaml
llm:
  smart: {plugin: openai, model: gpt-5.4-mini, max_tokens: 1024, extra_body: {reasoning_effort: none},
          api_key_env: OPENAI_API_KEY}
mouth:
  escalation: {context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_global: 4, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

Routing is measured by `scripts/eval_escalation.py` (local model only, no cost): with Gemma 4 E4B, 98% of 140
synthetic EN / RU / ES cases × 8 samples go the right way, 0.4% of hard questions are answered locally, no easy
one is escalated (`NOTES.md` → steps 11 and 14). Cost on ordinary turns: the longer prompt adds ~30 ms (expert) and
~90 ms (expert + tasks) to first audio. Without `llm.smart` neither tool is offered and the prompt is the plain one.
Tasks need a web-search key (`TAVILY_API_KEY`) to find anything current; without it their searches fail.

## Models & licences

| component | model | licence |
|---|---|---|
| ASR encoder + turn-taking | `nvidia/nemotron-3.5-asr-streaming-0.6b` + endpoint head (private) | NVIDIA Open Model Licence / private head |
| responder | `google/gemma-4-E4B-it`, FP8 at load (was `Qwen/Qwen3.5-4B`, before that `Qwen/Qwen3-4B-Instruct-2507`) | Apache-2.0 |
| TTS | `openbmb/VoxCPM2` | Apache-2.0 |
| web tools | Tavily API | commercial API (your key) |

## Privacy

No audio or transcript is written to disk or logs. Logs carry only timings and event types (endpoint source,
onset ms, barge, backchannel, tool name). Voice reference clips and their transcripts stay on the box.

**Hosted plugins send text off the box**, and only because the config names one: a hosted `reply` gets every
turn's transcript, the conversation history and tool results; a hosted `filler` gets the user's question (tool
lead-ins) and the running ASR partial (backchannels). A `smart` model gets only escalated turns and background
tasks: the user's words, the question or goal, and the last `context_turns` turns as plain text (tool results
left out), plus its own tool results. Audio and voice clips are never sent. The startup log names
the remote roles; request and response bodies are never logged, and MOUTH refuses to start with the SDKs' debug
logging on (`OPENAI_LOG` / `ANTHROPIC_LOG` = `debug`). Run hosted plugins only with the provider terms you accept
for that data.

## Known limits (v1)

- **Up to 4 conversations** per GPU (`server.max_sessions`); more visitors get a busy page. On a healthy 5090, 4
  conversations keep EARS real time (hop lag p50 ~90 / p90 ~200 ms); see `NOTES.md`.
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- The user waits **~1.0 s** from the end of their speech to the first reply audio at 1 conversation and
  **~1.25 s** at 4 (`load_test_ears.py`), and ~700–850 ms of that is endpoint detection. Live threshold tuning
  should cut it (targets in `NOTES.md` → Response gap). MOUTH's part is small: first audio ~265 ms after the endpoint for a plain or a tool
  turn (smoke test, cloned voice), ~410 ms p50 with 4 busy sessions.
- Endpoint thresholds are not yet live-tuned (see Configure).
- The `Dockerfile` has not yet been built and run end to end (the box used so far is itself a container, with no
  Docker daemon). The non-Docker fresh-box path above has been run end to end on a clean box.

See `NOTES.md` for the current state, `ISSUES.md` for open issues, and `HANDOFF.md` for the current work queue.
