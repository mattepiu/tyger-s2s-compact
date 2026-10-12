# Developing tyger

How tyger works inside, the rules for changing it, and how to check a change. `README.md` covers installing and
running it, `CHANGELOG.md` what changed (in short), and `ISSUES.md` what is left to do.

## Layout

```
app/
  config.py                # loads a config and checks every plugin block against its Options
  wire.py                  # the EARS <-> MOUTH framing and message kinds
  runtime/
    event_loop.py          #   EngineLoop: one asyncio loop for MOUTH's engines and API clients
    audio.py               #   StreamingAudioResampler: chunk-invariant resampling (the 24 kHz wire, the 2nd ear)
  prompts/                 # the config/prompts.yaml loader and its checks
  mouth/                   # the MOUTH process (python -m app.mouth)
    __main__.py            #   accepts EARS connections, one Session thread each; signal handling
    service.py             #   Mouth: the loaded plugins, prompts and tool lists, shared by every session
    session.py             #   Session: one conversation's state, the wire, the turn and backchannel workers
    history.py             #   History: turns trimmed whole, barged turns marked
    turns.py               #   Turn base, AnnouncementTurn, SayTurn
    reply.py               #   ReplyTurn: plain replies, tool rounds (ToolRound), escalation, barged rounds
    speech.py              #   the one path to the TTS; Speaker, one reply's audio book-keeping
    threads.py             #   Prefetch, spawn: streams on helper threads
    messages.py            #   the neutral message format and every request builder
    chunker.py             #   the streaming sentence / clause chunker
    tasks.py               #   background tasks on the smart model (TaskRunner, TaskSlots)
  ears/                    # the EARS process (python -m app.ears)
    __main__.py            #   loads the plugin, runs the web app
    web.py                 #   EarsApp: the page, the voice list, one Conversation per WebSocket
    conversation.py        #   Conversation: the turn-taking state machine and the floor queue
    playback.py            #   AgentPlayback: the reply as the page plays it, and the 2nd ear
  tools/
    registry.py            #   Tool, @tool / @agent_tool, build_tools, execute_tool
    lookups.py             #   get_current_time
    stubs.py               #   set_reminder, send_message: placeholders, not connected
    agent.py               #   the tools MOUTH answers itself: ask_expert, start_task, cancel_task, finish_task
  plugins/
    __init__.py            #   the registry; build_llms, build_tts, build_ears
    base.py                #   LLMPlugin, TTSPlugin, EarsPlugin: ABC bases with defaults; Threshold, Slider
    options.py             #   PluginOptions: a plugin's config block, typed, checked at load
    llm/
      calls.py             #   CallFormat, and what every format shares: speech split from a call at its tag, the
                           #     tool named early (tool_start), call ids, check_arguments / flag_invalid
      fake.py              #   the fake LLM, with its own call format (Hermes JSON)
      local/               #   in-process vLLM models, a folder each
        vllm_base.py       #     VLLMChatPlugin: an in-process vLLM AsyncLLM (pinned KV, abort by request id)
        gemma4/
          plugin.py        #       gemma-4-e4b, the responder
          calls.py         #       Gemma 4's call format: its tag, parsing, the repair of dropped markers
      hosted/              #   hosted APIs, a module each, with its own stream parser
        base.py            #     HostedLLM: key, timeouts, concurrency, errors, the debug-log guard
        openai_responses.py  #   openai-responses, the smart model
        anthropic.py       #     anthropic, its alternative
    tts/
      voxcpm2.py
      fake.py
    ears/
      nemotron_tyger/      #   nemotron-3.5-asr + the tyger 1.0 heads: one trained system, one package
        constants.py       #     the audio grid and the turn-rule numbers, in one place
        asr.py, rnnt.py    #     features on the GPU, the layer-15 encode, the CUDA-graphed RNN-T decoder
        heads.py           #     the heads release (loaded once)
        head_batch.py      #     the heads batched across conversations: KV ring, CUDA graphs
        rules.py           #     the per-frame turn rules: endpoint, barge-in, backchannel
        session.py, engine.py, service.py, plugin.py, release.py
      fake.py              #   an energy endpointer
config/
  default.yaml             # the stack (run.sh's default)
  fake.yaml                # the stack on the fake plugins: no GPU, no downloads
  prompts.yaml             # every prompt MOUTH sends to an LLM
  engines/voxcpm2.yaml     # vLLM-Omni's engine config for VoxCPM2
scripts/
  stack/                   # against the running stack: smoke, load tests, the simulated page, the cut-in check
  validate/                # GPU parity checks of EARS (the stack stopped); reference.py is the plain reference
  bench/                   # measurements, and the routing eval (escalation.py + escalation_set.yaml)
tests/                     # GPU-free: ears/, mouth/, plugins/, the config / prompts / tools tests, fakes.py;
                           #   page/ holds the pages' own checks (node): page_check.mjs (jsdom), web_check.mjs
static/index.html          # the page: mic capture (AudioWorklet), playback, the tags, the debug sliders
web/                       # the new page (a trial, Next.js): lib/conversation.ts = the page's audio + WebSocket code,
                           #   lib/talk.ts = what it shows (the phases); `npm run dev` on :3000 passes /ws to EARS
run.sh, stop.sh            # start and stop the stack
```

## How it works

### Two processes

- **EARS** (`app/ears`) serves the page and its WebSocket on :7860, and runs the turn-taking plugin for every
  conversation (up to `server.max_sessions`, batched on one GPU thread).
- **MOUTH** (`app/mouth`) runs the conversation: the LLM roles, the tools and the TTS. Its models are in-process:
  the LLM on a vLLM `AsyncLLM`, VoxCPM2 on a vLLM-Omni `AsyncOmni`, both on one asyncio loop (`EngineLoop`). There is
  no HTTP hop to either.
- Each conversation has its own TCP connection from EARS to MOUTH (:8770).
- `run.sh` starts MOUTH first, so its engines claim their pinned GPU memory before EARS loads. It waits for MOUTH's
  port, then starts EARS. `stop.sh` stops both, and SIGKILLs a VoxCPM2 stage worker that ignores SIGTERM.

**Why not one vLLM-Omni pipeline.** vLLM-Omni chains stages, each one's output feeding the next. Chaining the LLM
into VoxCPM2 would take MOUTH out from between them: no chunking policy, no tool calls with lead-ins, no contextual
backchannels, no barge-in control.

**Why one venv.** vLLM 0.30 brings transformers 5.14, which satisfies nemotron's ≥ 5.13. `voxcpm` is listed on its
own: the vLLM-Omni stage worker imports it, but vllm-omni doesn't depend on it.

**VoxCPM2 streams audio out, not text in.** A request needs its whole text up front, so first audio is cut on the
text side, by the chunker.

### The wire between EARS and MOUTH

A frame is 1 kind byte, a 4-byte big-endian length, then the payload (`app/wire.py`). Control payloads are JSON;
audio is float32 at 24 kHz. The page's own WebSocket is in README → "Run & reach it".

| kind | from → to | carries |
|---|---|---|
| `T` | EARS → MOUTH | a user turn: its transcript |
| `C` | EARS → MOUTH | a backchannel request, with the ASR partial |
| `B` | EARS → MOUTH | a barge: `played_s`, how much of the reply the user had heard when they started talking |
| `V` | EARS → MOUTH | the voice to use |
| `G` | EARS → MOUTH | the floor is granted to a finished task |
| `S` | test tools → MOUTH | speak this text, TTS only, never remembered (EARS never sends it) |
| `A` | MOUTH → EARS | reply audio |
| `K` | MOUTH → EARS | backchannel audio, kept apart from `A` |
| `X` | MOUTH → EARS | the text of the reply, or of a backchannel (`bc`), for the page |
| `L` | MOUTH → EARS | links the reply cites (`label`, `url`), for the page |
| `D` | MOUTH → EARS | done: a turn (`barged`), a backchannel (`bc`) or an announcement (`proactive`) |
| `F` | MOUTH → EARS | a finished task asks for the floor |

### EARS: turn-taking

- **The grid.** nemotron's encoder frame is 80 ms, and one frame is one turn-taking decision. EARS steps every
  320 ms (a hop of 4 frames) and re-encodes the last 8 s of each channel. The heads read encoder layer 15, so the
  encode stops there.
- **Two channels:** the user's mic, and the 2nd ear, what the agent was saying at that moment (next section).
- **The heads** come from the private `TYGER-S2S-endpointer` release: a Qwen3-0.6B decoder with its LoRA merged.
  Per frame they give the end of the user's turn (`p_end`), the yield (the user taking the turn) and the
  backchannel probability.
- **The rules** (`rules.py`):
  - **Listening.** ENDPOINT when `p_end` ≥ `ep_thr` for `ep_deb` frames, once the user has 350 ms of voiced speech.
    Only the head ends a turn; there is no silence rule. BACKCHANNEL when the bc head fires for `bc_deb` frames,
    then 1.2 s without one.
  - **Speaking.** BARGE when the yield ≥ `barge_thr` for `barge_deb` frames. The event carries `onset_s`, how long
    the user had been talking (by frame energy, through pauses up to 640 ms, at most 4 s). Detecting a barge takes
    a second or more, and MOUTH needs to know what had played when the user started.
  - After a reset, decisions wait 6 frames.
- **The states** (`conversation.py`):
  - **listening;**
  - **speaking**, from the turn's start until the page reports it has played the whole reply, or 2 s past the
    server's own guess (a safety net);
  - **halting**, after a barge until MOUTH's `D`. An endpoint meanwhile is held, and sent when the halt ends.
- **A barge keeps the user's first words.** While the agent speaks, EARS keeps the last 2 s of user audio, and a
  barge seeds the new turn with it.
- **Long turns** are transcribed in blocks. Past 15 s, a block is cut at its quietest frame and transcribed on the
  ASR thread, and the endpoint transcribes only the tail.
- **The ASR partial** for backchannels is re-decoded from the last 8 s at most every 0.8 s, off the critical path.
- **Batching.** One GPU thread serves every conversation.
  - A tick gathers the conversations' hops, waiting up to `gather_ms`, or `expect_ms` when one is catching up.
  - It encodes their windows in one batch, then steps the heads in one batch. Each conversation has a slot in a KV
    ring (`ring_frames`, reset at each turn), and there is one CUDA graph per attention width.
  - All-zero agent windows (while listening) are cached.
  - The RNN-T decoder is our own greedy loop: one CUDA graph of 16 steps, on its own thread and CUDA stream. It
    gives the same text as nemotron's `generate()` and is 3–5× faster.
- **CPU threads** are capped at 4 per process (`cpu_threads`). Torch's default of 48 mostly spin-waited.

### The 2nd ear

- **The yield head decides a barge from a pair:** the mic, and what the agent was saying at that same moment.
- **The page tags every mic block.** It plays the reply and records the mic on one AudioContext. Each 20 ms mic
  block carries `[reply id][reply sample]`: the sample the user was hearing at the block's first mic sample, or -1
  when nothing played.
  - The tag is corrected by the latencies the browser reports: `baseLatency` + `outputLatency` + the mic's input
    latency.
  - The page shows that total as "audio latency": 55–61 ms with built-in speakers and mic, 200 ms with EarPods.
  - A browser that doesn't report a latency counts it as 0. That matters most for Bluetooth output (~150–250 ms).
  - Nothing is set up per browser or device. On a new one, check that the "audio latency" looks plausible and that
    cut-ins feel right.
- **EARS builds the ear from the tags** (`AgentPlayback`). It keeps the current reply at 16 kHz and builds each
  agent hop from the tags, with silence exactly where nothing played. Backchannel audio (reply id 0) is played but
  never tagged, so it stays out of the 2nd ear.
- **The tags replace two guesses.** A barge's `played_s` is what the tags say the user had heard at their onset.
  The page's `played:<id>:<samples>` ends the agent's turn.
- **Why.** Until 2026-10-10 EARS paired the two channels in arrival order. Live, that put the 2nd ear 316–783 ms
  (median 566) ahead of what the user heard. With echo in the mic, the agent's own voice then looked like a cut-in.
  The log line `2nd ear by arrival order +X ms ahead` still reports that offset, as a diagnostic only.
- **Known limit (I-18).** With no echo in the mic, as with headphones, an "mm-hm" in a pause between the agent's
  sentences barges.

### MOUTH: one conversation

- **A Session** per EARS connection holds the history, the voice, recent backchannel acks and the background
  tasks. It runs two workers, turns and backchannels, so an ack is generated while a turn is. Sessions share the
  engines, and vLLM batches whatever is in flight.
- **LLM roles:**
  - `reply` answers turns;
  - `filler` makes backchannels, tool lead-ins and still-looking lines (the same instance by default);
  - `smart` (optional, hosted) answers escalated turns and runs background tasks.
- **A plain reply** streams from `reply` into the chunker. Each chunk is one VoxCPM2 request, whose audio streams
  out per decode step.
  - Chunks are whole sentences (`mouth.first_chunk: sentence`), so the intonation restarts only where a sentence
    does. No chunk is shorter than 40 characters.
  - A punctuation mark is a boundary only when whitespace follows it, so `0.50`, `3,000` and `example.com` stay
    whole. Abbreviations and initials aren't sentence ends.
- **A tool round** (`ReplyTurn`, `ToolRound`):
  1. The plugin emits `tool_start` as soon as the tool's name has decoded.
  2. A helper thread finishes decoding the call and runs the tools the moment it parses. Meanwhile the turn speaks a
     lead-in that fits the tool, made by `filler` and batched with the call.
  3. If the tools are still out 0.8 s after the last line played, one more "still looking" line, at most 2.
  4. `reply` answers from the results (phase B).
- **Escalation** (`ask_expert`, only with `smart`):
  - The smart request starts the moment the call parses, under the lead-in.
  - It gets the smart prompt, the last `context_turns` turns as plain text (tool calls and results dropped), the
    user's words and the reply model's question.
  - Its answer streams straight to the TTS. Its own tool rounds, up to `escalation.max_steps`, are covered by
    `expert_still` lines.
- **History** (`History`) keeps each turn whole: the user's words, any tool round with its call and a compacted
  result, then the reply. Calls stay in history so the model keeps calling tools on later turns (I-12). It's trimmed
  by whole turns, to `history_turns`.
- **Barge-in** stops the current turn.
  - Its live LLM and TTS requests are aborted by request id.
  - History keeps what the user heard of the reply, marked " —".
  - A call that has started decoding still decodes to the end, and its tools run: a barge stops what is said, not
    what was asked.
- **A round cut in on before any of its answer was heard** goes on as a background task of the conversation
  (`TaskRunner.continue_round`). The task gets the expert's question, or the user's words with what the round had
  looked up, and it's announced at the next pause. Where no task can start, history keeps the round's results.
- **Backchannels** are contextual.
  - `filler` reacts to the ASR partial. When the partial is shorter than `bc_min_words`, it hums instead.
  - Its recent acks are passed along as "don't repeat those".
  - The ack is one TTS request. Its audio goes out as `K`, and never into history or the 2nd ear.
  - A backchannel still being made when the turn starts is aborted.
- **Engine requests.** Every engine call has its own request id, which is what a barge aborts:

  | job | LLM requests | TTS requests |
  |---|---|---|
  | plain turn | 1 (`reply`) | 1 per chunk |
  | tool turn | the call, the lead-in (`filler`), still-looking lines, the answer (phase B) | 1 per line, 1 per answer chunk |
  | escalated turn | the call, the lead-in, the `smart` request and its own tool rounds | 1 per line, 1 per answer chunk |
  | backchannel | 1 (`filler`, at most `bc_max_new` tokens) | 1 |

### Background tasks and the floor

- **The tools.** `start_task(goal)` and `cancel_task(id)` are offered with `ask_expert` when `mouth.tasks.enabled`.
- **A task** is an agentic loop on `smart`, in its own thread. It makes its own web searches, runs for at most
  `max_steps` rounds and `max_wall_s`, and ends with `finish_task(summary, result)`.
- **Caps.** `max_concurrent` and `max_per_session` per conversation, and `max_global` across them.
- **The reply model sees its tasks** in a status note on the user's latest message, and answers "how's it going?"
  from it. It's a note, not a system message, because Gemma's template takes a system message only first.
- **Taking the floor.**
  - When a task finishes, MOUTH sends `F`.
  - EARS grants the floor (`G`) when the agent isn't speaking or halting, no user turn is held, no backchannel is
    playing, and both sides have been quiet for `announce_after_silence_ms` (1200).
  - A user endpoint always wins: the request waits.
  - On `G`, the summary is spoken like a reply, barge-in included, and ends with `D {"proactive": id}`.
- **History** gets the task's compacted result as a note, then what was heard.
- **Tasks live in memory only,** and end with the conversation.

### Plugins

- **The registry** (`app/plugins/__init__.py`) maps kind → name → `"module:Class"`, imported lazily. Third-party
  plugins aren't loaded.
- **The bases** (`base.py`) are `LLMPlugin`, `TTSPlugin` and `EarsPlugin`, with defaults for `warmup`, `shutdown`
  and `remote`. A plugin's config block is its `Options` dataclass, checked when the config loads, so a typo stops
  startup with the key named.
- **Neutral formats, owned by MOUTH:**
  - **Messages** are OpenAI-style dicts:
    - `{"role", "content"}`;
    - an assistant tool call: `{"role": "assistant", "content", "tool_calls": [{"id", "name", "arguments": dict}]}`;
    - a tool result: `{"role": "tool", "tool_call_id", "content"}`.
  - **Tool-call ids** come from the plugin and are unique per session (`call_<8 hex>`).
  - **An optional `native` field** carries a hosted plugin's raw content through a tool round (Claude's thinking
    blocks). MOUTH copies it and never reads it.
  - **LLM events:**
    - `("text", delta)`;
    - `("tool_start", {"name"})`;
    - `("tool_calls", calls, native)`, once at the end;
    - `("refusal", category)`;
    - `("links", [{"label", "url"}])`: sources the text cited, cut out of it. Only `openai-responses` sends them
      (its web search's citations, I-25). MOUTH shows them on the page and never speaks them.
- **Lazy imports.** A plugin imports vllm, vllm-omni or an SDK in its constructor, so EARS and the tests can import
  plugin classes cheaply. `test_plugins.py` checks it.
- **Call formats.** A local model writes its tool calls into its text, each model in its own way. Its plugin
  declares that format (`calls`, a `CallFormat` from the model's own `calls.py`):
  - the tag a call opens with;
  - the pattern of the tool's name right after the tag;
  - `parse(raw)`;
  - an optional repair, `normalize`.

  The shared helpers in `llm/calls.py` take the format object: speech is cut at the tag (a partial tag is held
  back), and `tool_start` names the tool as soon as its name has decoded. `VLLMChatPlugin` has no default format.
  Hosted APIs return structured calls, so they need none.
- **The local LLM** (`VLLMChatPlugin` + `local/gemma4/`) is Gemma 4 E4B:
  - FP8 at load (`quantization: fp8`);
  - text only (`language_model_only`: the vision and audio towers aren't loaded);
  - thinking off.

  Its calls are written in special tokens (`<|tool_call>call:NAME{…}<tool_call|>`), so it decodes with
  `skip_special_tokens=False`. When it drops the markers and writes a bare `set_reminder{…}`, its format's
  `normalize` makes that a call to the offered tool, and it is never spoken.
- **The hosted LLMs** share `HostedLLM`:
  - the key from the variable `api_key_env` names;
  - short timeouts, `max_retries` 1, a concurrency cap;
  - errors become `LLMError`;
  - startup is refused with the SDK's debug logging on.

  Every call's arguments are checked against the tool's schema (`check_arguments`), and a call that fails isn't
  run.
  - `openai-responses`: OpenAI's Responses API with its built-in web search, `store: false`. History maps to
    Responses items (`function_call` / `function_call_output`).
  - `anthropic`: the Claude Messages API, default `claude-opus-5-5` with `effort: low` and the server-side refusal
    fallback.
    - Opus 5.5 always thinks, so `max_tokens` covers thinking and text, and no `temperature` is sent.
    - Thinking blocks go back unchanged within a tool round (`native`), and are dropped from finished turns.
    - It's tested offline only: there's no key yet. When a key arrives, test edited history (a barged turn cut to
      what was heard) on an account created on or after 2026-08-31, where the history check is enforced, and check
      that `cache_read_input_tokens` is above zero on repeat turns.

### Prompts

- **One file.** Every piece of text MOUTH sends to an LLM, and the two lines it speaks itself, live in
  `config/prompts.yaml`: system prompts, few-shot turns, templates and tool descriptions. `mouth.prompts` can point
  at a copy for an experiment.
- **Checked at start.** Templates use `string.Template` (`$name`). The loader checks the keys, the placeholders and
  the shot shapes at MOUTH start, and a bad file stops startup with the key named. The file is read once, so restart
  MOUTH after editing.
- **Characters.** The reply system prompts (`system.voice_tools`, `escalation.system`, `tasks.reply_system`) and
  the smart model's (`smart.system`) open with a character: a few lines on who the agent is, its name, maker and
  manner (`persona.characters` in `config/prompts.yaml`). Then comes the prompt as it was (`Prompts.system`).
  - Ori is the default, from the user's Ori prompt. Mike and Andrew are drafts for the user to rewrite.
  - The expert's answers are spoken in the same voice, so the smart prompt gets the character too. The fillers
    (lead-ins, still-looking lines, backchannels) and the background task's prompt don't.
  - The user's full Ori prompt (2026-10-12: a long style base, opinions, "extremely knowledgeable") was measured
    first and failed the routing gate: 17.4% missed escalations, 12.5% with Ori-style few-shot lines. E4B then
    answered explanations, advice and code itself. A few lines before the old prompt keep routing where it was.
- **The character follows the voice.** Config `voices.character` is every voice's character, and
  `voices.characters` maps a voice to another one. A voice that isn't listed gets the default. A character that
  isn't in the prompts file stops startup with the key named (`Characters`, checked in both processes). When the
  page picks another voice (`voice:<name>` → `V`), the next turn speaks as that voice's character, and history
  stays. The picker labels each voice with its character, e.g. "Ori (ori)".
- **The few-shot turns are shared** by every character, so their style carries into each one.
- **Few-shot turns, not quoted examples.** Small models copy a quoted example word for word. Qwen3-4B said "Sure,
  let me check that for you." on every tool turn, and the first backchannel prompt's 'mm-hm' nearly every time.
- **English only.** Every prompt asks for English, and every few-shot turn is in English.
- **Examples stay synthetic:** the file is committed.

### GPU memory

- **Both engines pin their KV caches,** so the total is predictable:
  - VoxCPM2: 2 GiB (`config/engines/voxcpm2.yaml`, `max_num_seqs` 4);
  - the LLM: 3 GiB (`llm.reply.kv_cache_gib`, `max_num_seqs` 8, `max_model_len` 8192).

  `gpu_memory_utilization` is then only vLLM's start-up free-memory check.
- **The peak at 4 conversations is ~30 GB:**
  - **the LLM, ~15 GB:** 10.4 GiB of weights, of which the per-layer embedding table is about half and stays bf16;
    the 3 GiB KV; CUDA graphs. bf16 weights wouldn't fit beside the rest. Offloading the embedding table to the CPU
    was rejected as too slow;
  - **VoxCPM2, ~9.2 GB:** ~4.5 GB of bf16 weights, the 2 GiB KV, a CUDA context, and working memory for 4 streams;
  - **EARS, ~6 GB:** roughly nemotron in fp32 (2.4 GB; bf16 changed an event and wasn't faster), the heads
    (~1.2 GB), their KV ring (2048 frames × 28 layers × 4 conversations ≈ 0.9 GB), graphs and allocator headroom.
    `run.sh` sets `expandable_segments` for EARS; without it the cache fragmented past 9 GiB.
- **Headroom.** A 5090 (32.6 GB) is left with ~2.5 GB, and a 48 GB RTX PRO 5000 with ~19 GB. Measure before adding
  anything.

### VoxCPM2 settings that matter

- **`enable_unified_decode_graph: false`.** With it on, the scheduler defers every new request while any request
  decodes. Under concurrency a backchannel waited 42 s.
- **Warm-up runs every batch size** up to `max_num_seqs` before serving. Otherwise each new batch size compiles
  mid-conversation and stalls the engine for ~25 s.
- **The voice clip** is resampled once, at load. Its encoding is kept in the stage worker's speaker cache, in memory
  only, and later requests send only a reference to it. First audio per request went 146 → 78 ms.
- **Never put a tensor in a prompt's `additional_information`.** It arrives garbled and kills the stage worker. Use
  lists.
- **A script that constructs the engine** needs `if __name__ == "__main__":`, because vLLM-Omni spawns its stage
  worker.

## Extending it

### Adding a plugin

A plugin is a class that subclasses its kind's base in `app/plugins/base.py`, declares its config block as an
`Options` dataclass, and has one line in the registry. MOUTH, EARS and the page don't change. A worked example, an
LLM that answers every turn with the user's last words:

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

**A new local model** is a folder in `app/plugins/llm/local/`: its plugin, its call format, and one registry line.
The format is whatever the model writes. This one is Hermes JSON (`<tool_call>{"name": …, "arguments": …}`):

```python
# app/plugins/llm/local/mymodel/calls.py
import json, re
from app.plugins.llm.calls import CallFormat


class MyModelCalls(CallFormat):
    tag = "<tool_call>"                                          # a call opens with it
    name = re.compile(r'\s*\{\s*"name"\s*:\s*"([^"\\]+)"')      # the tool's name right after the tag

    def parse(self, raw):
        return [json.loads(m) for m in re.findall(r"<tool_call>(.*?)</tool_call>", raw, re.S)]
```

```python
# app/plugins/llm/local/mymodel/plugin.py
from app.plugins.llm.local.vllm_base import VLLMChatPlugin
from app.plugins.llm.local.mymodel.calls import MyModelCalls


class MyModel(VLLMChatPlugin):
    default_model = "org/my-model"
    calls = MyModelCalls()
```

```python
# app/plugins/__init__.py, in REGISTRY["llm"]
"my-model": "app.plugins.llm.local.mymodel.plugin:MyModel",
```

**A new hosted API** is a module in `app/plugins/llm/hosted/` that subclasses `HostedLLM`, with its own stream
parser, as `openai_responses.py` and `anthropic.py` have. It needs no call format: the API returns structured calls.

What the base gives you, and what is yours to write:
- **Options.** The constructor `(conf, ctx)` turns the block into `self.opts`. It sets `self.name`, the instance
  name, and `self.eloop`, MOUTH's asyncio loop.
  - The config is checked against `Options` when it loads: `unknown option prefx (did you mean prefix?)`.
  - A field without a default must be set.
  - `plugin:` and `name:` belong to the config shape.
- **Constructor.** If you write one, call `super().__init__(conf, ctx)` first, and import heavy dependencies inside
  it.
- **Defaults.** `warmup()` and `shutdown()` do nothing unless you override them. `remote` is False. Set it True if
  transcripts leave the box, and the startup log then names the role.
- **LLM.**
  - `stream()` yields the neutral events above.
  - It polls `stop_flag`, and raises `LLMError` for a failed request (MOUTH speaks an apology).
  - Local vLLM models subclass `VLLMChatPlugin`, declare their `calls`, and override only the hooks they need
    (`default_model`, `engine_kwargs`, `template_kwargs`, `sampling`), as `local/gemma4/plugin.py` does. A plugin
    without `calls` can't be built.
  - Hosted APIs subclass `HostedLLM`, as `hosted/openai_responses.py` and `hosted/anthropic.py` do.
- **TTS.**
  - Set `out_sr` and `caps`.
  - Write `list_voices(cfg)`, a classmethod: EARS calls it without loading the model.
  - Write `load_voice(name)`, and `stream(text, voice)`, which yields float32 mono at `out_sr`. MOUTH resamples to
    the 24 kHz wire.
- **EARS.**
  - Set `sr` and `hop_samples`.
  - Declare the tunable thresholds as `THRESHOLDS`, each a `Threshold` with an optional `Slider`. The base reads
    their defaults from config, and the page builds its sliders from them.
  - Write `open_session(thresholds)`, `close`, `feed` (one hop in, events out: FRAME, ENDPOINT, BARGE with
    `onset_s`, BACKCHANNEL), and `full` / `active`.
- **Tests.** Add a run to `tests/plugins/test_contract.py` (`llm_suite`, `tts_suite` or `ears_suite`). The fakes in
  `app/plugins/*/fake.py` are the smallest complete plugin of each kind. A new call format gets its own test file,
  like `tests/plugins/test_calls_gemma4.py`.

### Adding a tool

1. Write the function in a module in `app/tools/`. A new module goes on the import line in `app/tools/__init__.py`,
   before `agent`: the order there is the order the models see.
2. Give it its JSON schema with `@tool`.
3. Add its description to `config/prompts.yaml` → `tools:`.

`build_tools` then offers it to the models, and `execute_tool` runs it. It returns a small dict, or
`{"error": ...}` on failure:

```python
@tool("get_exchange_rate", {"type": "object", "properties": {"pair": {"type": "string"}}, "required": ["pair"]})
def get_exchange_rate(pair):
    ...
```

Tools that MOUTH answers itself (`ask_expert`, the task tools) register `handle(arguments, ctx)` with `@agent_tool`
in `app/tools/agent.py` instead. The smart model gets every tool except those (`Mouth.smart_tools`), next to its
own web search.

## Rules

- **Privacy.** No audio or transcripts on disk or in logs: timings, counts and event types only.
  - This covers new tests too: use synthetic, TTS or public LibriSpeech speech, held in memory and scored without
    printing words.
- **Voice clips, and anything derived from them, are personal data.** They stay in gitignored `voices/`, and are
  never committed, uploaded or sent to a hosted service.
- **Secrets** (`HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) live in `.env` only. Never print them, and don't
  leave copies of `.env` around, e.g. in a scratch git worktree.
- **Hosted runs.** A live run against a hosted model (`smart`, or any remote role) needs the user's OK each time.
- **The heads release is private.** This is a serving project only: no training code.
- **Prompts and tool parsing.** The agent speaks English only: prompts, few-shot turns and test cases are in English.
  Every prompt must keep `enable_thinking: false` working. Tool parsing keeps Gemma 4's format, and Hermes JSON for
  the `fake` LLM.
- **The GPU budget** is in "GPU memory" above: measure before adding anything.
- **`CHANGELOG.md` is for users** (Keep a Changelog). A user-visible change gets one short, plain line under
  Unreleased → Added / Changed / Fixed / Removed, with a key number only where it helps. Describe it in words, not
  by its issue ID alone. Something added and removed again before a release needs no line.
- **The details go in the commit message:** what changed and why, and the numbers before → after, naming the box
  and config. A re-measured baseline also gets a row in "Benchmarks" below.
- **Issues** are in `ISSUES.md` by ID. A fixed issue leaves the file, and its fix goes into the changelog.
- **Git.** The GPU-box agent commits locally on `plugins`.
  - Never push, never open PRs: the user reviews and pushes.
  - No `Co-Authored-By:` trailer.
  - One commit per issue, with the ID in the subject.
  - `agent-memory/` is gitignored: never commit it.

## Checking a change

- **GPU-free tests**, all of them:
  `for t in tests/test_*.py tests/*/test_*.py; do env -u OPENAI_API_KEY .venv/bin/python $t; done`

  ```
  tests/test_config.py                  config shape and every block's Options; old shapes rejected
  tests/test_prompts.py                 the config/prompts.yaml loader and its checks
  tests/test_tools.py                   the tool registry: order, which context answers which call
  tests/mouth/test_session.py           turns, barge-in, tools, backchannels on the fake plugins
  tests/mouth/test_parts.py             History and Speaker on their own
  tests/mouth/test_tasks.py             background tasks and their announcement, a fake smart model
  tests/mouth/test_barged_round.py      a round cut in on before its answer goes on as a task (timing-based: I-24)
  tests/mouth/test_chunker.py           text chunking
  tests/ears/test_floor.py              floor requests on the Conversation state machine
  tests/ears/test_conversation.py       halting, the held endpoint, the reset after a barge, MOUTH's D
  tests/ears/test_parts.py              AgentPlayback and the web app (busy page, voices)
  tests/ears/test_second_ear.py         the 2nd ear from the page's tags; the turn ends on the page's report
  tests/ears/test_asr_jobs.py           the ASR thread finishes its jobs, even on failure
  tests/ears/test_rules.py              the turn rules: endpoint, backchannel, barge-in
  tests/plugins/test_plugins.py         base classes, light imports, off-rate TTS, thresholds
  tests/plugins/test_contract.py        the plugin contract on the fakes (TYGER_CONTRACT_LOCAL=1: the real plugins too)
  tests/plugins/test_llm_calls.py       what every call format shares, on a toy format; the local plugin base
  tests/plugins/test_calls_gemma4.py    Gemma 4's call format and gemma-4-e4b's stream on it
  tests/plugins/test_calls_hermes.py    Hermes JSON, the fake LLM's call format
  tests/plugins/test_hosted_openai_responses.py   openai-responses on stubbed streams; citations cut from the text
  tests/plugins/test_hosted_anthropic.py          anthropic on stubbed streams
  ```
- **The page's own check** (node ≥ 18): `cd tests/page && npm install && node page_check.mjs`. It covers the mic
  worklet, the tags, the int16 frames, the played report, Stop → Start, the voice picker and a reply's links.
- **The new page's own check** (node ≥ 22.18, nothing to install): `node tests/page/web_check.mjs`. It runs the same
  cases on `web/lib/conversation.ts`, then the phases the page shows (`web/lib/talk.ts`) and a reply's links. After
  a change in `web/`, also run `cd web && npx tsc --noEmit && npm run build`.
- **The whole stack on the fakes:** `CONFIG=config/fake.yaml ./run.sh`, no GPU and no downloads.
- **GPU checks, with the stack stopped:**
  - `TYGER_CONTRACT_LOCAL=1 .venv/bin/python tests/plugins/test_contract.py`. It aborts at exit after passing (I-15).
  - `scripts/validate/`:
    - `ears_batched.py`: batched EARS = the single-session reference (`--pcm16`: the mic as int16, as the page
      sends it);
    - `rnnt.py`: the graphed RNN-T = nemotron's `generate()`;
    - `heads.py`: the heads release loads and is consistent;
    - `barge.py`: a barge keeps the start of the utterance;
    - `long_turn.py`: turns over 20 s are transcribed whole;
    - `second_ear.py --pairing tags`: false barges, cut-ins and "mm-hm"s, with echo levels and thresholds to sweep;
    - `silent_wait.py`: a barge while the agent waits in silence on a tool.
- **The stack.** Use a copy of the config without `smart`, so the load test's prompts never escalate to OpenAI.
  1. Make the copy, again after any config change (a key the code no longer reads stops startup):
     ```
     .venv/bin/python -c "import yaml; d = yaml.safe_load(open('config/default.yaml')); d['llm'].pop('smart'); d['mouth'].pop('tasks'); yaml.safe_dump(d, open('/tmp/nosmart.yaml', 'w'), sort_keys=False)"
     ```
  2. Start it: `CONFIG=/tmp/nosmart.yaml setsid ./run.sh > /tmp/stack.log 2>&1 &`, and wait for `EARS ready`.
  3. Run, each as `.venv/bin/python scripts/stack/<script>`:
     - `smoke.py`: MOUTH tool turns, barge-in, EARS speech → audio;
     - `load.py --sessions 4 --turns 6`, 3 times (`--tools --fresh` for tool turns);
     - `load_ears.py --sessions 1` and `--sessions 4`: hop lag, and end of speech → endpoint → first audio.
       `--downlink-ms 500` delays the simulated page's playback, as a slow network would.
  4. Stop it with `./stop.sh`.

  The scripts talk to EARS through a simulated page (`scripts/stack/page.py`). Compare with the last row of
  "Benchmarks" below for the same card.
- **Comparing two stacks:** `scripts/bench/stack.sh TAG CONFIG STACK_LOG` runs the same sequence each time, and
  `scripts/bench/summary.py TAG [TAG...]` lines the runs up. `llm_weights.py` gives the LLM's GPU memory per module
  and its decode speed; `tts_voice.py` the cloned voice's first audio per request.
- **Routing** (stack stopped, no hosted calls: only the reply model runs, and the reminder and message tools are
  stubbed): `.venv/bin/python scripts/bench/escalation.py --samples 8`, with and without `--no-expert`. It covers
  local, tool, expert and background routes on 119 synthetic English cases, plus spoken markup. The gate for any
  prompt or model change: at most 10% missed escalations, and no regression in tool calls or spoken markup.
  `--character NAME` routes with another character's persona (default: the config's).
- **Style** (stack stopped, no hosted calls): `.venv/bin/python scripts/bench/style.py --samples 3`, with
  `--prompts a.yaml,b.yaml` to compare prompt files and `--character NAME`. It counts, over 36 synthetic
  conversational turns, the local replies with markup (digits, symbols, emoji, parentheticals or action lines), with
  more than three sentences, and with a disfluency or filler word, and prints a few to read.
- **The expert, live:** `scripts/stack/barged_round.py` (cut in on the expert's lead-in) calls OpenAI, so it needs the
  user's OK each run. For escalation without OpenAI, give a config copy an `llm.smart` block with `plugin: fake`.
- **A fake stack next to the real one:** copy `config/fake.yaml` with ports 7861 / 8771, start it with
  `MOUTH_PORT=8771 CONFIG=... setsid ./run.sh`, and stop it by its process group, since `stop.sh` stops every
  run.sh.
- **Lint:** `uvx pyflakes app/ scripts/ tests/`.

## Benchmarks

**The stack over time.** Trend only: the rows come from different boxes and configs (noted). Every box was an RTX
5090 until 2026-10-11 (the first one throttled to half its memory clock), then an RTX PRO 5000. Compare numbers
only within one box and config.

| When | Code / config | Smoke: plain-turn first audio | 4 conversations: turn first audio p50 | End of speech → first audio p50, 1 / 4 conversations | GPU peak |
|---|---|---|---|---|---|
| 2026-10-02 | Qwen3-4B + VoxCPM2, 4 sessions (throttled box) | 503 ms | 833 ms | not measured yet | 27.0 GB |
| 2026-10-03 | `b64c772`, first healthy box | 354 ms | 539 ms | 1013 / 1022 ms | 28.4 GB (29.3 with tool turns) |
| 2026-10-03 | after the review fixes and the speaker cache, Qwen3-4B | 263–272 ms | 400–425 ms | — | 27.0 GB |
| 2026-10-04 | Qwen3.5-4B (`a7f4cf8`, the plugin-refactor baseline) | 243 ms | 482 ms | 970 / 1260 ms | 27.7 GB |
| 2026-10-04 | Gemma 4 E4B FP8 responder | 225 ms | 438 ms | 940 / ~1190 ms | 30.0 GB |
| 2026-10-06 | Gemma, production config (smart + tasks), new box | 279 ms | — | 904 / 1109 ms | 30.4 GB |
| 2026-10-06 | after the readability refactor, no `smart`, whole-sentence first chunk | 170–282 ms | 474 / 515 ms (2 runs) | ~880 / ~1090 ms | — |
| 2026-10-10 | after I-16 and I-17, no `smart`, new box | 174 ms | 397–453 ms (3 runs) | 793 / 992 ms | 30.2 GB |
| 2026-10-11 | no silence fallback, no `smart`, RTX PRO 5000 | 212 ms | 527–571 ms (3 runs) | 832 / 1093 ms | 29.9 GB |
| 2026-10-11 | `1b8e6ff`, no `smart`, a new RTX 5090 (the cleanup round's baseline) | 233 ms | 449–518 ms (3 runs) | 964 / 1105 ms | 30.2 GB |
| 2026-10-11 | after the cleanup (English only), no `smart`, same box | 204 ms | 473–517 ms (3 runs) | 935–959 / 1057–1096 ms (2 / 3 runs) | 30.1 GB |
| 2026-10-12 | `9caeca9`, no `smart`, voice `ori`, a fresh container on an RTX 5090 (this round's baseline) | 202 ms | 458–520 ms (3 runs) | 835–846 / 870–975 ms | 30.3 GB |
| 2026-10-12 | `e7b7eea`, Ori's character, same box | 215 ms | 459–514 ms (3 runs) | 836–892 / 920–974 ms (the reply varies: 2–10 texts a run) | 30.2 GB |

The voice changes the end-of-speech numbers: `load_ears.py` speaks its utterance in the default voice, and with the
`ori` clip the endpoint comes ~100 ms sooner than on the last box. Since the characters, `load_ears.py`'s "count to
ten" reply is no longer the same text every turn (2–10 distinct texts in a run, was 1), so its end-of-speech → audio
varies more.

"End of speech → first audio" is what the user hears as the response gap. About 600–850 ms of it is endpoint
detection, which live tuning of the endpoint thresholds is meant to cut (O-01).

**The two cards** (2026-10-10 and 2026-10-11, the stack without `smart`). With E4B the 5090 is the faster card:
decode runs near the bandwidth ratio, and EARS has less compute on the PRO 5000.

| | RTX 5090 (32.6 GB, 1792 GB/s) | RTX PRO 5000 (48.9 GB, 1344 GB/s) |
|---|---|---|
| Gemma E4B FP8 single-stream decode | 186 tok/s | 154 tok/s |
| `load.py` turn first audio p50, 4 sessions | 397–453 ms | 527–581 ms |
| EARS hop lag p50 / p90, 4 conversations | 47 / 124 ms | 68 / 181 ms |
| GPU peak at 4 sessions | 30.2 GB | 29.9 GB |

**Routing** (`scripts/bench/escalation.py --samples 8`, Gemma E4B, the default English prompts, the RTX 5090 of
2026-10-11): with `smart` and tasks on, 119 cases, 100.0% right and 0.0% missed escalations, every live-fact sample
to the expert; without the expert, 44 cases, 100.0% right.

**Responders measured against Gemma 4 E4B** (the gate: at most 10% of hard questions answered locally):

| Model | Weights | Missed escalations | Verdict |
|---|---|---|---|
| Gemma 4 E4B, FP8 at load | 10.4 GiB | 0.2–0.6% | **the responder** |
| Gemma 4 E4B NVFP4 (unsloth) | 7.1 GiB | 0.2% | passed; sends trivia to lookups; was the backup until 2026-10-12 |
| Gemma 4 26B-A4B NVFP4 | 15.1–17.1 GiB | 1.5–1.9% | passed; ~7 GB more, doesn't fit a 5090; skips some tool calls; thought-channel leaks |
| Qwen3.6-35B-A3B NVFP4 | 19.5 GiB | 1.3% | passed; tool turns ~200 ms slower |
| Qwen3-4B-2507, Qwen3.5-4B | 8 GiB | ~15.6% | failed: lean local |
| Qwen3-8B / 14B NVFP4 | 6.0 / 10.0 GiB | 13.8 / 18.5% | failed: lean local |
| Qwen3.5-35B-A3B GPTQ int4 | 20.3 GiB | 25.2% | failed |
| Nemotron Nano 9B v2 NVFP4 | 7.4 GiB | 20.4% | failed; narrates tools |
| Gemma 4 12B NVFP4 (unsloth) | 9.1 GiB | 14.8% | failed; leaks its thought channel |
| Gemma 4 E2B FP8 | 7.1 GiB | 42.9% | failed; leaks its thought channel |
| MiniCPM5-2B (English only) | 4.7 GiB | 17.0% | failed |

The full measurements behind these tables are in git history: CHANGELOG.md and NOTES.md at `4d4c2e3`.

## On the GPU box

- **Check the GPU first:** `nvidia-smi -q -d CLOCK,PERFORMANCE,ECC`. A healthy 5090 runs at P1, SM ~2890 MHz,
  memory 13 801 MHz. Note whether ECC is on: it costs memory and bandwidth.
- **RAM.** The first load of a model or quantization JIT-compiles FlashInfer kernels, one compiler (`cicc`) per CPU
  thread at 3–6 GB each. That twice ran a 64.6 GB container out of memory and restarted it, Claude included.
  - `run.sh` sets `MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4`, unless they're already
    set. The peak was then 32 GiB. The builds are cached in `~/.cache/flashinfer`.
  - Scripts that load a model outside `run.sh` don't get them: export them by hand before a first load in the
    validators, the local contract test, `escalation.py` or `llm_weights.py`.
  - Run long GPU jobs under a RAM watchdog that kills them before the container dies (`ramguard.sh`, in the agent's
    memory).
- **Start detached** (`setsid ./run.sh > log 2>&1 &`) and wait for `EARS ready` in the log. A warm start takes
  ~2.2 min.
- **The first start downloads ~25 GB.** On a slow link `run.sh` can give up waiting for MOUTH and leave the LLM
  engine on the GPU (I-14). Check `nvidia-smi`, kill the leftover `VLLM::EngineCore` by its process group
  (`ps -eo pid,pgid,cmd`), and start again.
- **Stop with `./stop.sh`.** A VoxCPM2 stage worker once ignored SIGTERM and kept 9 GB of GPU, so the next start
  failed its free-memory check; `stop.sh` SIGKILLs survivors. Never use `pkill -f` or `pgrep -f` with a pattern that
  appears in your own command line: it matches your shell.
- **`HF_TOKEN`** is in `.env`. Load it for a Hub call of your own (`set -a; . ./.env; set +a`) without printing it.
  The validators need it too.
- **Disk.** The boxes so far had 100 GB. Delete a model you no longer use from `$HF_HOME/hub` with
  `huggingface_hub`'s `scan_cache_dir`, since repos share blobs.
- **The box is itself a Docker container,** with no daemon, so `docker build` can't run there (O-02).
- **A fresh clone lacks the repo-local git identity.** Set it from the earlier commits' author.
- **A synthetic cut-in in the agent's own cloned voice never barges:** the heads take it for echo. Speak the user's
  lines in another voice.
- **Load-test noise.**
  - `load.py --sessions 4` turn p50 moves ±100 ms between identical runs, so use 3 runs × `--turns 6`.
  - `load_ears.py` speaks a fixed utterance (the TTS-only `S` message) and forces a "count to ten" reply, checked by
    hash. Under 4-way batching the reply still varies now and then, and the test warns: compare runs without the
    warning, or 3-run medians.
