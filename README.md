# tyger v1 — a deployable full-duplex voice agent on one GPU

Real-time spoken conversation in the browser: your mic → live turn-taking → a reasoning LLM with tools and a
hosted expert that searches the web → a cloned voice, all on a single GPU. The agent knows when you've finished a turn, when you're interrupting
it, and when to drop in a quick "mm-hm" — then answers in a voice cloned from a reference clip you provide.

This repo is the **inference / serving** project. The turn-taking model itself (the endpoint head and the
chain it sits on) is trained elsewhere; v1 only *runs* it. `CHANGELOG.md` lists what changed and why, with the
numbers; `NOTES.md` has the older, longer benchmark history.

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
   listening. EARS serves the web page and the WebSocket. To judge a barge-in, the heads hear the agent's own
   reply next to the mic (the "2nd ear"). The page plays the reply and records the mic on one audio clock and
   tags every mic block with the reply sample the user was hearing, so the 2nd ear lines up with what the user
   actually heard, whatever the network and audio latencies. The agent's turn ends when the page reports it has
   played the whole reply.
2. **MOUTH** is the orchestrator, and hosts both models in-process (no HTTP hop to either). On an endpoint it
   streams the LLM reply, cuts it into text chunks, and sends each chunk to VoxCPM2 as soon as it closes;
   audio streams back to EARS (and on to the browser) per decode step. Chunks are whole sentences: each chunk is
   its own TTS request, so the intonation restarts at a chunk boundary, and at a sentence end that sounds natural.
   (`mouth.first_chunk: clause` lets the first chunk end at a comma instead: audio ~30–95 ms sooner, but the
   intonation restarts mid-sentence.) No chunk is cut below 40 characters, so short replies go to the TTS
   whole. On a tool call
   it speaks a short contextual filler ("Checking the time in Tokyo.") the moment the call starts
   decoding, runs the tool as soon as the call parses, adds a "still looking" line if the tool is slow, then
   speaks the result. On a barge it stops within one audio chunk, aborts the live LLM/TTS requests, and keeps
   what the user heard of the reply in the conversation history.

## Hardware

Built on one **NVIDIA RTX 5090** (32 GB, Blackwell sm_120), CUDA-13 driver (≥ 590) for the cu130 torch wheels.

GPU memory in use ≈ **30.0 GB of 32 GB** with everything loaded, warm, and under a 4-conversation load: VoxCPM2
~9.8 GB (2 GiB pinned KV, `config/engines/voxcpm2.yaml`), Gemma 4 E4B ~14.9 GB with FP8 weights (3 GiB pinned KV,
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

- **`OPENAI_API_KEY`** — the `smart` model (`llm.smart` in `config/default.yaml`): escalations, background tasks,
  and live facts (weather, news, prices) through OpenAI's web search. There is no local web search or weather tool.
  Without the key, comment out `llm.smart`: the agent then says it can't check live facts.
- **`HF_TOKEN`** — pulls the **private** turn-taking heads release (`ears.heads` in `config/default.yaml`). If a token
  already exists at `$HF_HOME/token`, leave this blank.

Never commit `.env`.

### 3. Voices

Put one or more reference `.wav` clips in `voices/` (`voices.dir` in `config/default.yaml`; gitignored). The stem is the
voice name; `voices.default` picks the active one (falls back to the first clip). Optionally add
`voices/<name>.txt` with the clip's exact transcript: VoxCPM2 then clones from audio + transcript (better
quality, uses the whole clip); without it the clip is trimmed to `tts.ref_max_s` (10 s). `?debug=1` in the
browser shows a voice picker. **Reference clips and transcripts are personal data: keep them on the box, and
never commit, upload, or redistribute them.**

### 4. Configure

Everything you edit to configure the stack is in `config/` (no secrets: those are in `.env`):

```
config/
  default.yaml        # the stack: plugins and their options, thresholds, voices, MOUTH's limits (run.sh's default)
  fake.yaml           # the stack on the fake plugins: no GPU, no downloads (CONFIG=config/fake.yaml ./run.sh)
  prompts.yaml        # every prompt MOUTH sends to an LLM
  prompts/            # optional per-model overrides: <llm instance name>.yaml
  engines/
    voxcpm2.yaml      # vLLM-Omni's own engine config for VoxCPM2 (batch, KV budget), named by tts.engine_config
```

Paths inside a config are relative to the repo root. `CONFIG=<file> ./run.sh` runs another config. Worth knowing:

- **Prompts** live in `config/prompts.yaml`, not in code: system prompts, few-shot turns, the small templates MOUTH
  fills in, and the tool descriptions. It is checked at MOUTH start (a bad key or placeholder stops start-up with
  the key named) and read once, so restart MOUTH after editing. `config/prompts/<llm instance>.yaml` (e.g.
  `config/prompts/gemma-4-e4b.yaml`) overrides parts of it for one model. Keep few-shot examples synthetic: the
  file is committed.

- **Endpoint thresholds** (`ears.ep_thr` etc.) ship at the training-time defaults and have **not** been
  retuned on a live browser mic — offline thresholds have been seen to run ~2× too high on real mic signal.
  Expect to tune these live with `?debug=1`, then freeze the values back into `config/default.yaml`.
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

`?debug=1` reveals the live tuning sliders (endpoint τ / debounce, barge, backchannel) and the voice picker.
Browser DSP (echo cancellation, noise suppression, auto gain) is on.

**The page's WebSocket** (`/ws`; for a client of your own; `app/ears/playback.py` has the details). All binary
frames are little-endian, with int16 audio:
- **mic → EARS:** `[int32 reply id][int32 reply sample, or -1][int16 mic samples at 16 kHz]`. The tag is the reply
  sample (24 kHz) the user was hearing at the block's first mic sample, or -1 when no reply was playing. The page
  sends 20 ms blocks, captured with an AudioWorklet on the same AudioContext that plays the reply.
- **audio → page:** `[int32 reply id][int16 samples at 24 kHz]`. Replies and announcements are numbered from 1
  on each connection, so a client clears its per-reply counts when it connects again. Id 0 is a backchannel,
  which the page plays but never tags.
- **text → EARS:** `played:<id>:<samples>` when a reply's queue has drained; `reset`, `voice:<name>`,
  `set:<threshold>:<value>`.
- **JSON → page:** `ready`, `state`, `endpoint`, `reply`, `barge`, `turn_done`, `task_done`, and a few more.

### Tests

```bash
# tests/: no GPU, no network (each file runs on its own; all of them:)
for t in tests/test_*.py tests/*/test_*.py; do .venv/bin/python $t; done
.venv/bin/python tests/test_config.py                     # config shape + every block's Options; old shape rejected
.venv/bin/python tests/test_prompts.py                    # config/prompts.yaml loader + checks
.venv/bin/python tests/test_tools.py                      # the tool registry: order, which context answers which call
.venv/bin/python tests/mouth/test_session.py              # MOUTH turns / barge / tools / backchannels, fake plugins
.venv/bin/python tests/mouth/test_parts.py                # MOUTH's History and Speaker on their own
.venv/bin/python tests/mouth/test_tasks.py                # background tasks + their announcement, fake smart model
.venv/bin/python tests/mouth/test_barged_round.py         # a round cut in on before its answer goes on as a task
.venv/bin/python tests/mouth/test_chunker.py              # text chunking
.venv/bin/python tests/ears/test_floor.py                 # EARS floor requests: the Conversation state machine
.venv/bin/python tests/ears/test_parts.py                 # EARS's AgentPlayback and the web app (busy page, voices)
.venv/bin/python tests/ears/test_second_ear.py            # the 2nd ear from the page's playback tags; the turn
                                                          #   ends on the page's report
.venv/bin/python tests/ears/test_asr_jobs.py              # the ASR thread finishes its jobs, even on failure
.venv/bin/python tests/ears/test_rules.py                 # nemotron-tyger's turn rules: endpoint, backchannel, barge-in
.venv/bin/python tests/plugins/test_plugins.py            # base classes, light imports, off-rate TTS, thresholds
.venv/bin/python tests/plugins/test_contract.py           # the plugin contract on the fakes; with
                                                          #   TYGER_CONTRACT_LOCAL=1 the real local plugins too
                                                          #   (GPU free: stop run.sh first)
.venv/bin/python tests/plugins/test_llm_parse.py          # tool-call splitting, both call formats, <think> guard
.venv/bin/python tests/plugins/test_hosted_openai.py      # the openai plugin on stubbed streams
.venv/bin/python tests/plugins/test_hosted_anthropic.py   # the anthropic plugin on stubbed streams
(cd tests/page && npm install && node page_check.mjs)     # the page in jsdom: mic worklet, playback tags, int16
                                                          #   frames, the played report, Stop -> Start (needs node)

CONFIG=config/fake.yaml ./run.sh                          # the whole stack on the fake plugins: no GPU, no downloads
./stop.sh                                                 # stop run.sh (and any stage worker it leaves behind)

# scripts/stack/: the running stack (start run.sh first)
.venv/bin/python scripts/stack/smoke.py                   # MOUTH tool turns + barge-in + EARS speech → audio
.venv/bin/python scripts/stack/load.py --sessions 4       # concurrent MOUTH turns + backchannels
                                                          #   --tools: tool turns (+ --fresh: empty history each)
.venv/bin/python scripts/stack/load_ears.py --sessions 4  # N real-time WebSocket conversations; hop lag,
                                                          #   end of speech -> endpoint -> first audio
                                                          #   (fixed utterance and reply: runs compare);
                                                          #   --downlink-ms 500: a slow network, turns cut short
.venv/bin/python scripts/stack/barged_round.py --stack-log LOG   # cut in on the expert's lead-in: the answer
                                                          #   comes at the next pause (calls `smart`: ask first)

# scripts/validate/: GPU parity checks (GPU free: stop run.sh first)
.venv/bin/python scripts/validate/ears_batched.py         # batched EARS == single-session reference
                                                          #   (--pcm16: the mic as the page sends it, int16)
.venv/bin/python scripts/validate/rnnt.py                 # graphed RNN-T == nemotron generate()
.venv/bin/python scripts/validate/heads.py                # the heads release loads and is consistent
.venv/bin/python scripts/validate/barge.py                # a barge-in keeps the start of the utterance
.venv/bin/python scripts/validate/second_ear.py           # the 2nd ear's 24->16 kHz step, old vs new;
                                                          #   --offset-ms 0,100,200,400: false barges, cut-ins
                                                          #   and "mm-hm"s with the 2nd ear ahead (--pairing tags);
                                                          #   --echo-db=-30,-120 --barge-thr 0.4,0.6 sweep
.venv/bin/python scripts/validate/silent_wait.py          # barge-in while the agent waits in silence on a tool
.venv/bin/python scripts/validate/long_turn.py            # > 20 s turns transcribed whole

# scripts/bench/: measurements (GPU free, except stack.sh, which needs run.sh up)
.venv/bin/python scripts/bench/tts_voice.py               # cloned-voice TTS first-audio per request
.venv/bin/python scripts/bench/escalation.py --samples 8  # routing (local / tool / expert / background) and
                                                          #   reply language on 140 synthetic cases
.venv/bin/python scripts/bench/llm_weights.py <hf repo>   # an LLM's GPU memory per module + tok/s
scripts/bench/stack.sh TAG CONFIG STACK_LOG               # smoke + load tests, logs in bench_runs/TAG
.venv/bin/python scripts/bench/summary.py TAG [TAG...]    # the numbers from bench_runs/, side by side
```

The EARS half of the smoke test speaks one fixed sentence with the MOUTH's TTS (held in memory) as the "user"
speech, so no test audio file is needed; pass `--wav` to use your own.

## Plugins

The LLM, the TTS and the turn-taking model are **plugins chosen in `config/default.yaml`** (`app/plugins/`; the
design is in `PLUGINS.md`). Each block names its `plugin:`; the rest of the block is that plugin's options.

| kind | plugin | runs | notes |
|---|---|---|---|
| LLM | `gemma-4-e4b` | local, vLLM in MOUTH | the default responder (`quantization: fp8`): thinking off, text only, Gemma tool-call tokens |
| LLM | `gemma-4-e4b-nvfp4` | local, vLLM in MOUTH | the backup: unsloth NVFP4 E4B (int8 per-layer embeddings), routes like E4B FP8, 3.6 GB less GPU, same latency; no `quantization` |
| LLM | `gemma-4-26b-a4b-nvfp4` | local, vLLM in MOUTH | trial for a 48 GB card (CHANGELOG 2026-10-11): unsloth's NVFP4 Gemma 4 26B-A4B mixture of experts (FP8 attention), its thought channel kept out of speech; no `quantization` |
| LLM | `qwen3.6-35b-a3b-nvfp4` | local, vLLM in MOUTH | trial for a 48 GB card: NVIDIA's Qwen3.6-35B-A3B (Qwen XML tool calls, Qwen's sampling, `config/prompts/` override); no `quantization` |
| LLM | `qwen3.5-35b-a3b-gptq` | local, vLLM in MOUTH | trial, failed the routing gate (25% missed escalations); no `quantization` |
| LLM | `openai` | **hosted** | any OpenAI-compatible chat endpoint (`base_url`: OpenAI, Groq, Cerebras, a vLLM server) |
| LLM | `openai-responses` | **hosted** | OpenAI's Responses API with OpenAI's web search: the `smart` role (escalations, live facts, tasks) |
| LLM | `anthropic` | **hosted** | the Claude Messages API; default `claude-opus-5-5`, `effort: low`, server-side refusal fallback |
| TTS | `voxcpm2` | local, vLLM-Omni in MOUTH | cloned voices from `voices/` |
| EARS | `nemotron-tyger-1.0` | local, in EARS | nemotron ASR + the tyger 1.0 turn-taking heads |
| all three | `fake` | no model, CPU | scripted / energy-based; tests and `CONFIG=config/fake.yaml ./run.sh` |

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
neutral line (both in `config/prompts.yaml` → `spoken`).

### Adding a plugin

A plugin is a class that subclasses its kind's base in `app/plugins/base.py` (`LLMPlugin`, `TTSPlugin` or
`EarsPlugin`), declares its config block as an `Options` dataclass, and has one line in the registry. MOUTH, EARS
and the page don't change. A worked example, an LLM that answers every turn with the user's last words:

```python
# app/plugins/llm/echo.py
from __future__ import annotations
from dataclasses import dataclass

from app.plugins.base import LLMPlugin
from app.plugins.options import PluginOptions


class EchoLLM(LLMPlugin):
    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        prefix: str = "You said:"             # llm.<role>.prefix in config/default.yaml

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for word in f"{self.opts.prefix} {last}".split()[:max_new]:
            if stop_flag and stop_flag():                 # barge-in: stop within one item
                return
            yield ("text", word + " ")
```

```python
# app/plugins/__init__.py, in REGISTRY["llm"]
"echo": "app.plugins.llm.echo:EchoLLM",
```

```yaml
# config/default.yaml
llm:
  reply: {plugin: echo, prefix: "I heard"}
```

What the base gives you, and what is yours to write:
- **Options.** The constructor `(conf, ctx)` turns the block into `self.opts`, and sets `self.name` (the instance
  name) and `self.eloop` (MOUTH's asyncio loop). The config is checked against `Options` when it loads, so a typo
  stops start-up with the key named (`unknown option prefx (did you mean prefix?)`). A field without a default
  must be set. `plugin:` and `name:` belong to the config shape.
- **Constructor.** If you write one, call `super().__init__(conf, ctx)` first, and import heavy dependencies (vllm,
  an SDK) inside it, not at module level: the class must stay light to import (`tests/plugins/test_plugins.py` checks
  this).
- **Defaults.** `warmup()` and `shutdown()` do nothing unless you override them. `remote` is False; set it True if
  transcripts leave the box (the startup log then names the role).
- **LLM.** `stream()` yields the neutral events documented in `base.py`: `("text", delta)`, then for a tool call
  `("tool_start", {"name"})` and `("tool_calls", calls, native)`. It polls `stop_flag`, and raises `LLMError` for
  a failed request (MOUTH speaks an apology). Local vLLM models subclass `VLLMChatPlugin` (`vllm_base.py`) and
  override only its hooks, as `gemma4.py` does. Hosted APIs subclass `HostedLLM` (`hosted.py`: key, timeouts,
  concurrency, errors), as `openai_compat.py` and `anthropic.py` do. Validate tool arguments with
  `_toolparse.check_arguments`.
- **TTS.** Set `out_sr` and `caps`, and write `list_voices(cfg)` (a classmethod: EARS calls it without loading
  the model), `load_voice(name)` and `stream(text, voice)`, which yields float32 mono at `out_sr`. MOUTH
  resamples to the 24 kHz wire.
- **EARS.** Set `sr` and `hop_samples`, and declare the tunable thresholds as `THRESHOLDS` (`Threshold`, each
  with an optional `Slider`): the base reads their defaults from config, and the page builds its sliders from
  them. Write `open_session(thresholds)`, `close`, `feed` (one hop in, events out: FRAME / ENDPOINT / BARGE /
  BACKCHANNEL; a BARGE may say how long the user had been talking, `onset_s`, so MOUTH learns what had played
  when they started) and `full` / `active`. `nemotron_tyger/` is the full example: a package that keeps its
  encoder, heads and rules together.
- **Tests.** Add a run to `tests/plugins/test_contract.py` (`llm_suite`, `tts_suite` or `ears_suite`). The fakes in
  `app/plugins/*/fake.py` are the smallest complete plugins of each kind.

Per-model prompt changes go in `config/prompts/<instance name>.yaml`, never in code.

### Adding a tool

Tools are a registry too (`app/tools/`). Write the function in a module there (a new module goes on the import
line in `app/tools/__init__.py`, before `agent`: the order there is the order the models see), give it its JSON
schema with `@tool`, and add its description to `config/prompts.yaml` →
`tools:`; `build_tools` offers it to the models, and `execute_tool` runs it. It returns a small dict, and
`{"error": ...}` on failure:

```python
@tool("get_exchange_rate", {"type": "object", "properties": {"pair": {"type": "string"}}, "required": ["pair"]})
def get_exchange_rate(pair):
    ...
```

Tools that MOUTH answers itself (`ask_expert`, the task tools) register `handle(arguments, ctx)` with
`@agent_tool` in `app/tools/agent.py` instead.

## Escalation and background tasks

With an `llm.smart` block, the local `reply` model can hand work to a bigger hosted model. It decides itself, by
calling a tool, so easy turns pay nothing extra:

- **`ask_expert`** for a question that needs care now: maths, code, health / money / legal advice, comparisons,
  planning, long explanations, or "think carefully"; and for live facts (weather, news, prices, scores), which the
  smart model looks up with OpenAI's web search (`openai-responses`, `web_search: true`). The lead-in fits ("let me
  think about that", "let me pull up the Lisbon forecast"), the smart
  request starts the moment the call is read, and its answer streams straight to the voice (live: answer audio at
  ~1.0–1.3 s, under the lead-in). Follow-ups on the same topic go to it too.
- **`start_task`** for work that takes minutes: research, comparing many options, a multi-day plan, a long draft.
  The local model says it'll look into it and the conversation carries on. The task runs on the smart model in the
  background (its own web searches, `mouth.tasks.max_steps` rounds, `max_wall_s`). When it finishes, EARS gives it
  the floor once the user has been quiet for `announce_after_silence_ms`, and it speaks a short summary; the user
  can barge in as usual. "How's it going?" is answered locally from a status note; "drop it" cancels it.
- **A round the user cuts in on** before hearing any of its answer (during the lead-in or a still-looking line)
  isn't lost: it goes on as a background task of the conversation, counted against the task caps. The task gets
  the expert's question, or the user's words with what the round already looked up, and is announced at the next
  pause. Where no task can start (tasks off, a cap reached), history keeps the round's results instead.

```yaml
llm:
  smart: {plugin: openai-responses, model: gpt-5.4-mini, max_tokens: 1024, web_search: true, reasoning_effort: low,
          api_key_env: OPENAI_API_KEY}
mouth:
  escalation: {context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_global: 4, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

Routing is measured by `scripts/bench/escalation.py` (local model only, no cost): with Gemma 4 E4B, 98% of 140
synthetic EN / RU / ES cases × 8 samples go the right way, 0.4% of hard questions are answered locally, no easy
one is escalated (`NOTES.md` → steps 11 and 14). Cost on ordinary turns: the longer prompt adds ~30 ms (expert) and
~90 ms (expert + tasks) to first audio. Without `llm.smart` neither tool is offered and the prompt is the plain one.
Tasks and live facts need the smart model's web search (`openai-responses`); each search is billed on top of the
tokens. Without `llm.smart` the agent answers what it knows and says plainly that it can't check live facts.

## Models & licences

| component | model | licence |
|---|---|---|
| ASR encoder | `nvidia/nemotron-3.5-asr-streaming-0.6b` @ `ea30d66` (the revision the heads release pins) | OpenMDW-1.1 |
| turn-taking heads | `nineninesix/TYGER-S2S-endpointer` v0.1.0-rc1 (private; Qwen3-0.6B decoder with its LoRA merged + the heads) | Apache-2.0 (modified Qwen3-0.6B: see its NOTICE) |
| responder | `google/gemma-4-E4B-it`, FP8 at load (was `Qwen/Qwen3.5-4B`, before that `Qwen/Qwen3-4B-Instruct-2507`) | Apache-2.0 |
| TTS | `openbmb/VoxCPM2` | Apache-2.0 |
| expert, background tasks, web search | OpenAI `gpt-5.4-mini` with OpenAI's web search (`llm.smart`) | commercial API (your key) |

## Privacy

No audio or transcript is written to disk or logs. Logs carry only timings and event types (endpoint source,
onset ms, barge, backchannel, tool name). Voice reference clips and their transcripts stay on the box.

**Hosted plugins send text off the box**, and only because the config names one: a hosted `reply` gets every
turn's transcript, the conversation history and tool results; a hosted `filler` gets the user's question (tool
lead-ins) and the running ASR partial (backchannels). A `smart` model gets only escalated turns and background
tasks: the user's words, the question or goal, and the last `context_turns` turns as plain text (tool results
left out), plus its own tool results, and for a round the user cut in on, what that round looked up. Audio and
voice clips are never sent. The startup log names
the remote roles; request and response bodies are never logged, and MOUTH refuses to start with the SDKs' debug
logging on (`OPENAI_LOG` / `ANTHROPIC_LOG` = `debug`). Run hosted plugins only with the provider terms you accept
for that data.

## Known limits (v1)

- **Up to 4 conversations** per GPU (`server.max_sessions`); more visitors get a busy page. On a healthy 5090, 4
  conversations keep EARS real time (hop lag p50 ~90 / p90 ~200 ms); see `NOTES.md`.
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- The user waits **~1.0 s** from the end of their speech to the first reply audio at 1 conversation and
  **~1.25 s** at 4 (`scripts/stack/load_ears.py`), and ~700–850 ms of that is endpoint detection. Live threshold
  tuning should cut it (targets in `NOTES.md` → Response gap). MOUTH's part is small: first audio ~265 ms after the
  endpoint for a plain or a tool turn (smoke test, cloned voice), ~410 ms p50 with 4 busy sessions.
- Endpoint thresholds are not yet live-tuned (see Configure).
- The `Dockerfile` has not yet been built and run end to end (the box used so far is itself a container, with no
  Docker daemon). The non-Docker fresh-box path above has been run end to end on a clean box.

See `NOTES.md` for the current state, `ISSUES.md` for open issues, and `HANDOFF.md` for the current work queue.
