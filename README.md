# tyger v1 — a deployable full-duplex voice agent on one GPU

Real-time spoken conversation in English, in the browser: your mic → live turn-taking → a reasoning LLM with tools
and a hosted expert that searches the web → a cloned voice, all on a single GPU. The agent knows when you've finished
a turn, when you're interrupting it, and when to drop in a quick "mm-hm" — then answers in a voice cloned from a
reference clip you provide.

This repo is the **inference / serving** project. The turn-taking model itself (the endpoint head and the
chain it sits on) is trained elsewhere; v1 only *runs* it.

**The docs:**
- `README.md` (this file): what it is, and how to install, configure and run it;
- `DEVELOPING.md`: how it works inside, the rules for changing it, and how to check a change;
- `CHANGELOG.md`: what changed, in short;
- `ISSUES.md`: what is left to do, by ID;
- `HANDOFF.md`: the GPU-box agent's current round.

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
It also runs on an **RTX PRO 5000** (48 GB), with the same memory but ~20% slower LLM decoding (DEVELOPING.md →
"Benchmarks").

GPU memory in use ≈ **30 GB** with everything loaded, warm, and under a 4-conversation load: Gemma 4 E4B ~15 GB
with FP8 weights (3 GiB pinned KV, `llm.reply.kv_cache_gib`; bf16 weights would not fit), VoxCPM2 ~9.2 GB (2 GiB
pinned KV, `config/engines/voxcpm2.yaml`), EARS ~6 GB. Both engines pin their KV budgets, so the total is
predictable. DEVELOPING.md → "GPU memory" has the breakdown.

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
- **`ANTHROPIC_API_KEY`** — only if `llm.smart` is the `anthropic` plugin instead.
- **`HF_TOKEN`** — pulls the **private** turn-taking heads release (`ears.heads` in `config/default.yaml`). If a token
  already exists at `$HF_HOME/token`, leave this blank.

Never commit `.env`.

### 3. Voices

Put one or more reference `.wav` clips in `voices/` (`voices.dir` in `config/default.yaml`; gitignored). The stem is the
voice name; `voices.default` picks the active one (falls back to the first clip). Optionally add
`voices/<name>.txt` with the clip's exact transcript: VoxCPM2 then clones from audio + transcript (better
quality, uses the whole clip); without it the clip is trimmed to `tts.ref_max_s` (10 s). **Reference clips and
transcripts are personal data: keep them on the box, and never commit, upload, or redistribute them.**

**Each voice can have its own character.** A character is a few lines on who the agent is: its name, personality
and humour (`persona.characters` in `config/prompts.yaml`). Ori is the default character; Mike and Andrew are drafts
to rewrite. `voices.character` gives every voice its character, and `voices.characters` maps a voice to another
one (`mike: mike`). The page's voice picker shows for everyone, each voice labelled with its character
("Ori (ori)"). Picking another voice switches the character from the next turn, and the conversation goes on.

### 4. Configure

Everything you edit to configure the stack is in `config/` (no secrets: those are in `.env`):

```
config/
  default.yaml        # the stack: plugins and their options, thresholds, voices, MOUTH's limits (run.sh's default)
  fake.yaml           # the stack on the fake plugins: no GPU, no downloads (CONFIG=config/fake.yaml ./run.sh)
  prompts.yaml        # every prompt MOUTH sends to an LLM
  engines/
    voxcpm2.yaml      # vLLM-Omni's own engine config for VoxCPM2 (batch, KV budget), named by tts.engine_config
```

Paths inside a config are relative to the repo root. `CONFIG=<file> ./run.sh` runs another config. Worth knowing:

- **Prompts** live in `config/prompts.yaml`, not in code: system prompts, few-shot turns, the small templates MOUTH
  fills in, and the tool descriptions. It is checked at MOUTH start (a bad key or placeholder stops start-up with
  the key named) and read once, so restart MOUTH after editing. `mouth.prompts` can point at a copy for an
  experiment. Keep few-shot examples synthetic: the file is committed.
- **Endpoint thresholds** (`ears.ep_thr` etc.) ship at the training-time defaults and have **not** been
  retuned on a live browser mic — offline thresholds have been seen to run ~2× too high on real mic signal.
  Expect to tune these live with the page's sliders, then freeze the values back into `config/default.yaml`.
- **Backchannels** are always **contextual LLM acks**: a short reaction the LLM generates from the running
  ASR partial ("oh nice", "wow, really?"). When you've said too little to react to (`mouth.bc_min_words`),
  it uses a neutral "mm-hm". How *often* they fire is EARS's `ears.bc_thr` (a slider on the page).

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

The page shows the live tuning sliders (endpoint τ / debounce, barge, backchannel) while they're being tuned (O-01);
`?nodebug` hides them for a clean demo. The voice picker shows either way. Browser DSP (echo cancellation, noise
suppression, auto gain) is on.

**The new page (a trial, `web/`).** A friendlier page with:
- one mic button, and the characters as tabs;
- an orb in the middle ([OrbKit](https://github.com/zzzzshawn/orbkit)'s Ion, MIT, from the
  [Agents Kit](https://agents-ui.github.io/agents-kit/voice) registry). Its state follows the conversation (idle,
  thinking, speaking), and it reacts to your voice while you talk and to the agent's while it speaks. Each
  character has its own colours;
- the exchange as text: your words as you say them, each of the agent's backchannels beside them for 2 s, its reply
  with the sources it cited; and the whole conversation in a history.

It's the same WebSocket and the same audio code as the classic page. Run it beside the stack (node ≥ 22):

```bash
cd web && npm install && npm run dev   # http://localhost:3000; /ws and /voices pass through to EARS on :7860
```

Forward port 3000 the same way as 7860. The classic page, with the meters and the sliders, is one click away from it
("Classic view").

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
- **JSON → page:** `ready`, `state`, `partial` (the user's words so far in their turn, when they change),
  `endpoint` (with the user's `transcript`), `reply`, `links`, `backchannel` / `backchannel_done` (`aborted`:
  dropped unvoiced), `barge`, `turn_done`, `task_done`, and a few more.

## Plugins

The LLM, the TTS and the turn-taking model are **plugins chosen in `config/default.yaml`** (`app/plugins/`). Each
block names its `plugin:`; the rest of the block is that plugin's options.

| kind | plugin | runs | notes |
|---|---|---|---|
| LLM | `gemma-4-e4b` | local, vLLM in MOUTH | **the responder**, `google/gemma-4-E4B-it` with `quantization: fp8`: thinking off, text only, Gemma tool-call tokens |
| LLM | `openai-responses` | **hosted** | **the `smart` role**: OpenAI's Responses API with its web search (escalations, live facts, background tasks) |
| LLM | `anthropic` | **hosted** | the alternative `smart`: the Claude Messages API; default `claude-opus-5-5`, `effort: low`, server-side refusal fallback |
| TTS | `voxcpm2` | local, vLLM-Omni in MOUTH | cloned voices from `voices/` |
| EARS | `nemotron-tyger-1.0` | local, in EARS | nemotron ASR + the tyger 1.0 turn-taking heads |
| all three | `fake` | no model, CPU | scripted / energy-based; tests and `CONFIG=config/fake.yaml ./run.sh` |

Ten other responders were measured against Gemma 4 E4B and lost (DEVELOPING.md → "Benchmarks"); their plugins are
in git history.

**LLM roles.** `llm.reply` answers turns, and the answer after a tool round. `llm.filler` makes backchannels, tool
lead-ins and still-looking lines; `filler: reply` (the default) shares one instance. `llm.smart` (optional) answers
escalated turns and runs background tasks. Any role can use any LLM plugin; two blocks with the same plugin need a
`name:` each. Keys live in `.env`, under the variable the block's `api_key_env` names.

Hosted `reply` and `filler` roles were measured and left local: a hosted `reply` added ~0.35 s to first audio, and a
hosted `filler` made backchannels ~0.55 s slower (2026-10-04). A hosted TTS was tried and removed: every
text chunk paid a request's time to first audio. A hosted request that fails (rate limit, timeout, network) ends
the turn with a short spoken apology, and a refusal with a short neutral line (both in `config/prompts.yaml` →
`spoken`).

Adding a plugin or a tool: DEVELOPING.md.

## Escalation and background tasks

With an `llm.smart` block, the local `reply` model can hand work to a bigger hosted model. It decides itself, by
calling a tool, so easy turns pay nothing extra:

- **`ask_expert`** for a question that needs care now: maths, code, health / money / legal advice, comparisons,
  planning, long explanations, or "think carefully"; and for live facts (weather, news, prices, scores), which the
  smart model looks up with OpenAI's web search (`openai-responses`, `web_search: true`). The lead-in fits ("let me
  think about that", "let me pull up the Lisbon forecast"), the smart
  request starts the moment the call is read, and its answer streams straight to the voice (live: answer audio at
  ~1.0–1.3 s, under the lead-in). Follow-ups on the same topic go to it too. The web search's citations are cut
  out of the answer: they show as links under it on the page and are never read out.
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

Routing is measured by `scripts/bench/escalation.py` (local model only, no cost): with Gemma 4 E4B, all 119
synthetic cases × 8 samples go the right way, and every live-fact question goes to the expert (DEVELOPING.md →
"Benchmarks"). Cost on ordinary turns: the longer prompt adds ~30 ms (expert) and ~90 ms
(expert + tasks) to first audio. Without `llm.smart` neither tool is offered and the prompt is the plain one.
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
onset ms, barge, backchannel, tool name). Voice reference clips and their transcripts stay on the box. The page gets
the user's own words back (as they speak, on `partial`, and each turn's transcript, on `endpoint`) to show them.

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
  conversations keep EARS real time (hop lag p50 47–87 / p90 124–170 ms on two boxes).
- **English only:** every prompt asks for English, so the agent answers in English whatever language it hears.
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- The user waits **~0.8–1.0 s** from the end of their speech to the first reply audio at 1 conversation and
  **~1.0–1.1 s** at 4 (two RTX 5090 boxes, `scripts/stack/load_ears.py`), and ~600–770 ms of that is endpoint
  detection. Live threshold tuning should cut it (ISSUES.md, O-01). MOUTH's part is small: first audio ~175–235 ms
  after the endpoint on a plain turn (smoke test, cloned voice), 400–520 ms p50 with 4 busy sessions.
- Endpoint thresholds are not yet live-tuned (see Configure).
- The `Dockerfile` has not yet been built and run end to end (the box used so far is itself a container, with no
  Docker daemon). The non-Docker fresh-box path above has been run end to end on a clean box.
