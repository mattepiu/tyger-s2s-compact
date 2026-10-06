# tyger — readability refactor (design)

**Status:** done, 2026-10-06. Planned the same day (user: "the code is still very hard to read; use common
patterns so we can easily add plugins; the endpoint head is tied to nemotron, make them one plugin"). Implemented
step by step on branch `plugins`, one commit per step (`R-NN: …` in the subject), each checked for parity before
the next.

**Progress:** all steps done (2026-10-06): R-00 plan, R-01, R-02, R-03, R-04a, R-04b, R-05, R-06a, R-06b, R-06c,
R-07a, R-07b, R-07c, R-08, R-09, and the extra R-10a/b/c/d. Per-step results and measurements: NOTES.md → "Readability refactor". The layout
below is the code as built.

**Goal:** a reader can find where a thing happens from its name, and a new plugin is one package plus one registry
line, with no edits to MOUTH, EARS's server or the web page.

**Non-goals:** new features, model changes, latency work (L-01 / L-02 stay in HANDOFF's queue), and the wire
protocol between EARS, MOUTH and the browser (message kinds stay as they are).

## What makes it hard to read today

| where | problem |
|---|---|
| `app/mouth.py` (867 lines) | One module holds the message-format helpers (pure functions), `Session` (turn flow, tool rounds, escalation, tasks, backchannels, the wire workers) and the process `main`. Per-turn state (`_said`, `_audio_s`, `_play_end`, `_lang`) lives on the Session and is reset by hand in four places. `tasks.py` imports helpers back from `mouth.py` inside functions (a hidden import cycle). |
| `app/server.py` | `_conversation` is a 230-line closure: its state is a dict `st` with 16 keys mutated by six nested functions. It hard-codes nemotron's threshold names in the `ready` message, and the page hard-codes the same sliders. |
| `app/ears_core/` (11 modules) | The live plugin uses about half of it. Constants are spread over `labeling.py` (one constant), `live_ears.py`, `encoder_stream.py` and `model.py` (`STATE_NAMES` twice). The model id is written in four places, and `ears.nemotron` in config is never read. The head checkpoint is `torch.load`ed three times at start-up. The encoder (nemotron) and the heads (tyger 1.0) form one trained system, but the code doesn't show that boundary. |
| plugins | Protocols with no shared base: every plugin re-implements `warmup` / `shutdown` / `remote`, and reads its options with scattered `conf.get(key, default)` (a typo in config is silently ignored). The two hosted LLMs copy ~60 lines of client scaffolding and validate tool arguments differently. |
| `app/tools.py` | Schemas, Tavily, the time zone table and stubs in one file. `ask_expert` / `start_task` / `cancel_task` are special-cased by name inside `Session._run_calls`. |

## Target layout

```
app/
  config.py                # loading and checking config.yaml (every plugin block against its Options)
  wire.py                  # the EARS <-> MOUTH protocol (framing, message kinds)
  runtime/                 # what both processes' code runs on (R-10b)
    event_loop.py          #   EngineLoop: one asyncio loop for MOUTH's in-process engines and API clients
    audio.py               #   StreamingAudioResampler: chunk-invariant resampling (the 24 kHz wire, the 2nd ear)
  prompts/                 # prompts.yaml loader + checks (__init__.py) and the language guess (lang.py) (R-10a)
  mouth/                   # the MOUTH process (python -m app.mouth)
    __main__.py            #   Server: accepts EARS connections, one Session thread each; signal handling
    service.py             #   Mouth: the loaded plugins, prompts, tool lists (shared by sessions); shutdown()
    session.py             #   Session: the conversation's state, the wire, the turn and backchannel workers
    history.py             #   History: turns trimmed whole, barged turns marked
    turns.py               #   Turn base (speak a stream, fillers, remember) + AnnouncementTurn, SayTurn
    reply.py               #   ReplyTurn: plain reply, tool rounds (ToolRound), escalation
    speech.py              #   speech_audio (the one path to the TTS) + Speaker (one reply's audio book-keeping)
    threads.py             #   Prefetch, spawn: streams run on helper threads
    messages.py            #   the neutral message format + every builder (reply prefix, shots, rounds, smart, bc)
    chunker.py             #   the streaming sentence / clause chunker for the TTS
    tasks.py               #   background tasks on the smart model (TaskRunner, TaskSlots)
  ears/                    # the EARS process (python -m app.ears; run.sh starts it)
    __main__.py            #   the process: load the plugin, run the web app
    web.py                 #   EarsApp: the page, the voice list, one Conversation per WebSocket (no module globals)
    conversation.py        #   Conversation: the per-WebSocket state machine as a class with named states
    playback.py            #   AgentPlayback: the reply as the browser plays it + the 2nd ear's audio
  tools/                   # tool registry: one Tool per name, registered where it is implemented
    registry.py            #   Tool, ToolContext, @tool / @agent_tool, run_call / run_round, execute_tool, build_tools
    lookups.py             #   get_weather, web_search (Tavily), get_current_time
    stubs.py               #   set_reminder, send_message (not connected: a demo marker)
    agent.py               #   the tools MOUTH answers itself: ask_expert, start_task, cancel_task, finish_task
  plugins/
    base.py                # ABC bases (LLMPlugin, TTSPlugin, EarsPlugin) with defaults; Threshold / Slider
    options.py             # PluginOptions: each plugin's config block as a typed dataclass, checked at load
    llm/  hosted.py        #   HostedLLM base: key check, debug-log guard, concurrency, error mapping
          _toolparse.py    #   call parsing, and check_arguments: the one tool-argument check (hosted plugins)
    ears/nemotron_tyger/   # ONE package for the joint model: nemotron-3.5-asr (encoder + ASR) + tyger 1.0 heads
      constants.py         #   frame / hop / window / sample rate, state names: one place
      asr.py               #   nemotron-3.5-asr: load, features on the GPU, the L15 encode the heads read,
                           #     transcription (everything specific to that checkpoint, model id from config)
      rnnt.py              #   nemotron-3.5-asr's greedy RNN-T decoder (its LSTM decoder + joint), CUDA-graphed
      heads.py             #   TygerHeads: the TYGER-S2S-endpointer release (fuse + backbone + heads), loaded once
      head_batch.py        #   RingKV / HeadBatch: the heads batched across conversations, CUDA graphs
      rules.py             #   the per-frame turn rules (endpoint, barge-in, backchannel), shared with the reference
      session.py           #   EarsSession: one conversation's state
      engine.py            #   per-tick batching of encoder + heads, the ASR jobs
      service.py           #   the GPU thread that gathers hops across conversations
      plugin.py            #   NemotronTyger: the EarsPlugin surface + its threshold / slider spec
      release.py           #   heads_release: fetch the heads release from the Hub
scripts/                   # (R-10c)
  stack/                   # smoke.py, load.py, load_ears.py: the running stack
  validate/                # GPU parity checks (heads, rnnt, ears_batched, barge, long_turn, second_ear) and
                           #   reference.py: one conversation through the heads release, the plain way
  bench/                   # measurements (tts_voice, llm_weights, stack.sh + summary) and escalation.py + its set
  stop.sh                  # stop run.sh
tests/                     # (R-10d) fakes.py; test_config / test_prompts / test_tools; ears/, mouth/, plugins/
```

## Steps

Each step is one commit and keeps behaviour identical unless it says otherwise. "GPU-free" = every
`tests/test_*.py`. "Stack" = `scripts/stack/smoke.py` + `scripts/stack/load.py --sessions 4` (3 runs) + `scripts/stack/load_ears.py` at 1 and
4 conversations, compared against NOTES.md → "Box (2026-10-06)" on the config without `llm.smart`.

| step | what | parity check |
|---|---|---|
| R-01 | **Fix the review's correctness bugs** (the only behaviour change before R-08): ASR finals and partials in separate `try` blocks; a lost MOUTH connection ends the conversation (`error` to the page, socket closed) instead of a silent dead hop loop; `ears.nemotron` is read; an empty `say_clean` backchannel is not sent to the TTS. (Not a bug: the anthropic plugin ignoring `max_new` is deliberate, since Claude's thinking can't be turned off and `max_tokens` must cover it.) | GPU-free + new tests; stack |
| R-02 | **`app/messages.py`**: move the message builders out of `mouth.py` (no logic change); `tasks.py` imports them at the top (the cycle is gone). | GPU-free |
| R-03 | **`app/mouth/` package**: split `mouth.py` into service / session / turn / speech / `__main__`. Per-turn state becomes a `Speaker` object made per turn. `python -m app.mouth` unchanged. | GPU-free; stack |
| R-04 | **`app/ears/` package**: `Conversation` class replacing the `_conversation` closure; the 2nd-ear resampler becomes the stateful `StreamingAudioResampler` (a model-input change: measured with `scripts/validate/barge.py` and `scripts/stack/load_ears.py`). | GPU-free (`test_ears_floor.py`); stack |
| R-05 | **Plugin-owned thresholds**: an EARS plugin declares its tunable thresholds (name, type, range, step, label); the server sends that spec in `ready`, and the page builds its sliders from it. The fake plugin drops its dummy thresholds. | GPU-free; the page by hand with `?debug=1` (the user) |
| R-06 | **`nemotron_tyger` package**: the encoder and the heads become one plugin package; constants in one module; one checkpoint load (weights check included); research-only code (`LiveEars`, `EndpointEars`, `EncoderStream`, `StateEndpointModel`) moves to `scripts/reference/`. | `scripts/validate/ears_batched.py`, `scripts/validate/rnnt.py`, `scripts/validate/barge.py`, `scripts/validate/long_turn.py`; contract test; stack |
| R-07 | **Plugin bases and options**: ABC bases with the shared defaults; each plugin declares an `Options` dataclass (typed, defaulted, unknown keys rejected at config load with the key named); `HostedLLM` base for `openai` / `anthropic`; one tool-argument validator. | GPU-free (config tests for unknown keys); contract test |
| R-08 | **Tool registry**: `app/tools/` with one `Tool` per name; Tavily, time, stubs and the MOUTH-handled tools register handlers, so `Session` stops switching on tool names. | GPU-free; smoke |
| R-09 | **Docs**: README "Adding a plugin" rewritten for the new pattern (a worked example), PLUGINS.md dated note, this file marked done. | — |
| R-10 | **Extra (user, 2026-10-06): `app/` grouped by who uses each module.** R-10a: `chunker`, `messages`, `tasks` into `app/mouth/`; `prompts` + `lang` become `app/prompts/`. R-10b: `engines` + `resample` into `app/runtime/` (`event_loop.py`, `audio.py`); `weights` into the nemotron-tyger package (`release.py`). `config.py` and `wire.py` stay at the top. Pure moves (`git mv`), no aliases for the old paths. | GPU-free; lint; smoke |
| R-10c | **`scripts/` by kind**: `stack/` (smoke + load tests on the running stack), `validate/` (GPU parity checks + their reference), `bench/` (measurements + the escalation eval); the folder replaces the name prefix (`validate/heads.py`, `bench/summary.py`, `stack/load_ears.py`); `stop.sh` stays. | each script starts; validators rerun; README / HANDOFF / ISSUES commands updated |
| R-10d | **`tests/` by part**: `ears/`, `mouth/`, `plugins/` with the part prefix dropped and pytest's `test_` kept; config / prompts / tools tests stay at the top. | GPU-free suite; README / HANDOFF commands updated |

Order: R-01 first so the refactor commits are pure moves. R-02 → R-03 (MOUTH) and R-04 → R-06 (EARS) are
independent; R-07 / R-08 touch both sides and come after them.

## Rules

- No behaviour change inside a refactor step; if one is needed it gets its own commit and NOTES entry.
- The plugin contract (event shapes, the wire messages) doesn't change. Tests may move; they don't get weaker.
- Privacy rules unchanged: nothing new is logged; reference code and validators keep audio in memory.
