# tyger — readability refactor (design)

**Status:** plan, 2026-10-06 (user: "the code is still very hard to read; use common patterns so we can easily add
plugins; the endpoint head is tied to nemotron, make them one plugin"). Implemented step by step on branch
`plugins`, one commit per step (`R-NN: …` in the subject), each checked for parity before the next.

**Progress:** R-00 plan, R-01, R-02 done (2026-10-06).

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
  messages.py              # the neutral message format + every builder (reply prefix, shots, rounds, smart, bc)
  mouth/                   # the MOUTH process (python -m app.mouth)
    __main__.py            #   socket server, signal handling, one Session per connection
    service.py             #   Mouth: the loaded plugins, prompts, tool lists (shared by sessions)
    session.py             #   Session: wire in/out, the turn and backchannel workers
    turn.py                #   one reply turn: speak a stream, tool rounds, fillers, escalation
    speech.py              #   Speaker: per-turn audio bookkeeping (what was said, when it plays out)
  ears/                    # the EARS process (python -m app.ears; app.server stays as an alias for run.sh)
    __main__.py            #   aiohttp app, routes
    conversation.py        #   Conversation: the per-WebSocket state machine as a class with named states
  tools/                   # tool registry: one Tool per name (schema + handler); MOUTH-handled tools are Tools too
  plugins/
    base.py                # ABC bases (LLMPlugin, TTSPlugin, EarsPlugin) with defaults; Options dataclasses
    llm/  hosted.py        #   HostedLLM base: key check, debug-log guard, concurrency, error mapping
          schema.py        #   one tool-argument validator for every plugin
    ears/nemotron_tyger/   # ONE package for the joint model: nemotron-3.5-asr (encoder + ASR) + tyger 1.0 heads
      constants.py         #   frame / hop / window / sample rate, state names: one place
      asr.py               #   nemotron-3.5-asr: load, features on the GPU, the L15 encode the heads read,
                           #     transcription (everything specific to that checkpoint, model id from config)
      rnnt.py              #   nemotron-3.5-asr's greedy RNN-T decoder (its LSTM decoder + joint), CUDA-graphed
      heads.py             #   the tyger 1.0 inference model (fuse + backbone + heads), checkpoint loaded once
      engine.py            #   per-tick batching, RingKV / HeadBatch, the per-frame decision rule
      service.py           #   the GPU thread that gathers hops across conversations
      plugin.py            #   NemotronTyger: the EarsPlugin surface + its threshold / slider spec
scripts/reference/         # research reference code the validators compare against (LiveEars, EndpointEars, …)
```

## Steps

Each step is one commit and keeps behaviour identical unless it says otherwise. "GPU-free" = every
`tests/test_*.py`. "Stack" = `smoke_test.py` + `load_test.py --sessions 4` (3 runs) + `load_test_ears.py` at 1 and
4 conversations, compared against NOTES.md → "Box (2026-10-06)" on the config without `llm.smart`.

| step | what | parity check |
|---|---|---|
| R-01 | **Fix the review's correctness bugs** (the only behaviour change before R-08): ASR finals and partials in separate `try` blocks; a lost MOUTH connection ends the conversation (`error` to the page, socket closed) instead of a silent dead hop loop; `ears.nemotron` is read; an empty `say_clean` backchannel is not sent to the TTS. (Not a bug: the anthropic plugin ignoring `max_new` is deliberate, since Claude's thinking can't be turned off and `max_tokens` must cover it.) | GPU-free + new tests; stack |
| R-02 | **`app/messages.py`**: move the message builders out of `mouth.py` (no logic change); `tasks.py` imports them at the top (the cycle is gone). | GPU-free |
| R-03 | **`app/mouth/` package**: split `mouth.py` into service / session / turn / speech / `__main__`. Per-turn state becomes a `Speaker` object made per turn. `python -m app.mouth` unchanged. | GPU-free; stack |
| R-04 | **`app/ears/` package**: `Conversation` class replacing the `_conversation` closure; the 2nd-ear resampler becomes the stateful `StreamingAudioResampler` (a model-input change: measured with `validate_barge.py` and `load_test_ears.py`). | GPU-free (`test_ears_floor.py`); stack |
| R-05 | **Plugin-owned thresholds**: an EARS plugin declares its tunable thresholds (name, type, range, step, label); the server sends that spec in `ready`, and the page builds its sliders from it. The fake plugin drops its dummy thresholds. | GPU-free; the page by hand with `?debug=1` (the user) |
| R-06 | **`nemotron_tyger` package**: the encoder and the heads become one plugin package; constants in one module; one checkpoint load (weights check included); research-only code (`LiveEars`, `EndpointEars`, `EncoderStream`, `StateEndpointModel`) moves to `scripts/reference/`. | `validate_ears_batched.py`, `validate_rnnt.py`, `validate_barge.py`, `validate_long_turn.py`; contract test; stack |
| R-07 | **Plugin bases and options**: ABC bases with the shared defaults; each plugin declares an `Options` dataclass (typed, defaulted, unknown keys rejected at config load with the key named); `HostedLLM` base for `openai` / `anthropic`; one tool-argument validator. | GPU-free (config tests for unknown keys); contract test |
| R-08 | **Tool registry**: `app/tools/` with one `Tool` per name; Tavily, time, stubs and the MOUTH-handled tools register handlers, so `Session` stops switching on tool names. | GPU-free; smoke |
| R-09 | **Docs**: README "Adding a plugin" rewritten for the new pattern (a worked example), PLUGINS.md dated note, this file marked done. | — |

Order: R-01 first so the refactor commits are pure moves. R-02 → R-03 (MOUTH) and R-04 → R-06 (EARS) are
independent; R-07 / R-08 touch both sides and come after them.

## Rules

- No behaviour change inside a refactor step; if one is needed it gets its own commit and NOTES entry.
- The plugin contract (event shapes, the wire messages) doesn't change. Tests may move; they don't get weaker.
- Privacy rules unchanged: nothing new is logged; reference code and validators keep audio in memory.
