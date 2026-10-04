# tyger — plugin architecture (design)

**Status:** design, 2026-10-04. To be implemented by the GPU-box agent on branch `plugins`; see `HANDOFF.md`.

**Goal:** swap the LLM, TTS and turn-taking (EARS) models through config, without touching the orchestration.

**First set of plugins:**

| Kind | Local | Hosted |
|---|---|---|
| LLM | `qwen3-4b-2507`, `qwen3.5-4b` | `openai` (any OpenAI-compatible chat endpoint), `anthropic` (the Claude Messages API, or a compatible endpoint via `base_url`) |
| TTS | `voxcpm2` | — (an `openai` TTS was built and measured in step 8, then removed: see below) |
| EARS | `nemotron-tyger-1.0` (today's ASR + endpoint head) | — |
| Testing | `fake` for each kind: the fakes in `tests/fakes.py`, promoted | — |

**Then, built on top of the plugins:**
- **Smart escalation:** the local model handles most turns and hands only heavy ones to a remote "smart" model.
- **Background tasks:** long jobs run while the conversation continues, and the agent speaks up when a result is
  ready.

Both are described in "Smart escalation and background tasks" below.

**Non-goals:**
- Steps 1–9 change nothing about how a conversation behaves: chunking, fillers, backchannels, barge-in, the
  EARS↔MOUTH wire protocol. Background tasks (steps 12–14) add two wire messages, specified below.
- No hosted ASR.
- No dynamic loading of third-party packages: the registry is a dict in the repo.

## Principles

1. **Refactor first, then add.** The local plugins must reproduce today's behaviour and numbers before any hosted
   plugin is written.
2. **The hot path stays plain.** Token and audio streams stay generators that poll `stop_flag`: no new threads,
   queues or copies per item. Abort on barge-in stays per request.
3. **MOUTH owns the conversation; plugins own the model.**
   - MOUTH keeps history in one neutral format (below), plus prompts, chunking and tool execution.
   - Each plugin converts that history to what its model needs, and turns the model's output back into the
     neutral events.
4. **Lazy imports.** A plugin imports its heavy dependencies (`vllm`, `vllm_omni`, `anthropic`, `openai`) inside
   its constructor. A config that uses only hosted plugins must start MOUTH without loading vLLM or using the
   GPU.
5. **Privacy holds across plugins.**
   - Hosted plugins send transcripts, and for TTS the reply text, to a third party. That's allowed only because
     the config names such a plugin explicitly. The README must say so.
   - Never send voice clips (`voices/`) to a hosted service.
   - Never log request or response bodies. Leave SDK debug logging off: `ANTHROPIC_LOG` and `OPENAI_LOG` must
     not be `debug`.
6. **Secrets** are read from the environment variable the config names (`api_key_env`), loaded from `.env`.
   Never print them. `.env.example` lists the names with empty values: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GROQ_API_KEY`, `CEREBRAS_API_KEY`.

## Neutral formats (owned by MOUTH)

### Messages
OpenAI-style dicts, which is what MOUTH uses today:
- `{"role": "system" | "user" | "assistant", "content": str}`
- An assistant tool call: `{"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}`
- A tool result: `{"role": "tool", "tool_call_id", "content": str}`

Two additions:
- **Tool-call IDs come from the plugin and are unique for the session.** Today MOUTH writes `call_0`,
  `call_1`… on every turn, so IDs repeat across turns in history. Claude requires each `tool_result` to match
  its `tool_use` id, and the IDs must not collide. Local plugins generate `call_<8 hex>`.
- **Optional `"native"` field on an assistant tool-call message.** It's opaque, written and read by the same
  plugin. The Anthropic plugin stores the response's raw content blocks there, including thinking blocks, so
  the tool round can be continued exactly (see the Anthropic plugin). MOUTH copies it through and never inspects
  it.

### Tools
The OpenAI function schema in `app/tools.py` (`{"type": "function", "function": {name, description,
parameters}}`). Plugins convert it.

### LLM events (yielded by `stream`)
- `("text", delta)`: speakable text, never part of a tool call or a thinking block.
- `("tool_start", {"name": str | None})`: as soon as the model commits to a tool call. Local plugins don't know
  the name yet (`None`); hosted plugins usually do. MOUTH starts the filler on this event, as today.
- `("tool_calls", [{"id", "name", "arguments": dict}], native)`: once, at the end.
- `("refusal", category | None)`: the model declined (hosted only). MOUTH speaks a short neutral line, ends the
  turn, and keeps the user turn in history.

## Interfaces (`app/plugins/base.py`, `typing.Protocol`)

```python
class LLMPlugin(Protocol):
    remote: bool                     # True -> transcripts leave the box (logged once at startup, by name only)
    def stream(self, messages, tools=None, max_new=192, stop_flag=None) -> Iterator[LLMEvent]: ...
    def warmup(self) -> None: ...
    def shutdown(self) -> None: ...

class TTSPlugin(Protocol):
    out_sr: int                      # plugin's native rate; MOUTH resamples to the 24 kHz wire if different
    caps: TTSCaps                    # cloning: bool, max_batch: int, languages: list[str] | None
    remote: bool
    @classmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]: ...   # (names, default) WITHOUT loading the model
    def load_voice(self, name) -> Any: ...                            # opaque handle (VoxCPM2: speaker cache entry)
    def stream(self, text, voice, stop_flag=None) -> Iterator[np.ndarray]: ...   # float32 mono at out_sr
    def warmup(self, voice) -> None: ...
    def shutdown(self) -> None: ...

class EarsPlugin(Protocol):          # what server.py uses today from EarsService
    hop_samples: int; sr: int        # server.py stops importing HOP_FRAMES / ENC_SR / FRAME_S
    def open(self, **thresholds) -> EarsSession | None: ...
    def close(self, session) -> None: ...
    async def feed(self, session, user_hop, agent_hop, mode, more=False) -> list[dict]: ...  # FRAME / ENDPOINT / BARGE / BACKCHANNEL
    def full(self) -> bool: ...
    def active(self) -> int: ...
```

**The voice list moves to the TTS plugin.** EARS's `/voices` and the `ready` message call
`list_voices(cfg)` on the configured TTS plugin class:
- VoxCPM2 scans `voices/`, as `discover_voices` does today;
- OpenAI TTS returns the configured preset names.

EARS imports the plugin class but never constructs it, so the import must stay light.

## Registry and config

`app/plugins/__init__.py` maps kind and name to `"module:Class"` and resolves them with `importlib`:

```python
REGISTRY = {
  "llm":  {"qwen3-4b-2507": "app.plugins.llm.qwen3_2507:Qwen3Instruct2507",
           "qwen3.5-4b":    "app.plugins.llm.qwen35:Qwen35",
           "openai":        "app.plugins.llm.openai_compat:OpenAICompatLLM",
           "anthropic":     "app.plugins.llm.anthropic:AnthropicLLM",
           "fake":          "app.plugins.llm.fake:FakeLLM"},
  "tts":  {"voxcpm2": "app.plugins.tts.voxcpm2:VoxCPM2",
           "fake": "app.plugins.tts.fake:FakeTTS"},
  "ears": {"nemotron-tyger-1.0": "app.plugins.ears.nemotron:NemotronTyger", "fake": "app.plugins.ears.fake:EnergyEars"},
}
```

**LLM roles.** MOUTH uses an LLM in up to three roles:

| Role | Typical plugin | Does |
|---|---|---|
| `reply` | local | ordinary turns and the tool phase-B answer; decides when to escalate |
| `filler` | local; defaults to the `reply` instance | backchannels, the tool lead-in, still-looking lines: very low latency |
| `smart` | remote, optional | escalated turns (`ask_expert`) and background tasks only; see below |

- **Without a `smart` block,** escalation and background tasks are off and their tools are never offered.
- **Any role can use any plugin;** "typical" is only the default suggestion. In particular, `reply` (and `filler`) can
  be a cheap, fast remote model behind an OpenAI-compatible endpoint, e.g. Groq or Cerebras: no flag, just config.
  See "Remote `reply`" below.
- **Running `reply` remote and `filler` local** also works: the same code with the opposite split, so the two can be
  compared.
- **Each role block is its own plugin instance.** Two blocks may use the same plugin with different endpoints, e.g.
  `reply` on Groq and `smart` on OpenAI, both `plugin: openai`. Give a block an optional `name:` (default: the
  plugin name). Logs and prompt overrides use that name.

```yaml
llm:
  reply:
    plugin: qwen3.5-4b
    kv_cache_gib: 3.0
    max_model_len: 8192
    max_num_seqs: 8
    gpu_memory_utilization: 0.45
  filler: reply                    # same instance (default); or a full plugin block
tts:
  plugin: voxcpm2
  deploy_config: deploy/voxcpm2.yaml
  ref_max_s: 10.0
ears:
  plugin: nemotron-tyger-1.0
  # today's ears: thresholds etc. stay here unchanged
```

Local-first with a smart model (`config.smart.example.yaml`, the target setup):
```yaml
llm:
  reply: {plugin: qwen3.5-4b, kv_cache_gib: 3.0, max_num_seqs: 8}
  filler: reply
  smart: {plugin: anthropic, model: claude-opus-5-5, effort: low, max_tokens: 2048, api_key_env: ANTHROPIC_API_KEY}
mouth:
  escalation: {enabled: true, context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

Remote cheap `reply` + remote `smart` (`config.remote-reply.example.yaml`). The model IDs are placeholders: take
current ones from the provider's model list, and pick a model the provider documents as supporting tool calls:
```yaml
llm:
  reply: {plugin: openai, name: groq, base_url: https://api.groq.com/openai/v1, model: <groq model id>,
          api_key_env: GROQ_API_KEY, max_tokens: 512, max_concurrency: 8}
  filler: {plugin: qwen3.5-4b, kv_cache_gib: 1.0, max_num_seqs: 8}   # or: filler: reply (then no local LLM at all)
  smart: {plugin: anthropic, model: claude-opus-5-5, effort: low, max_tokens: 2048, api_key_env: ANTHROPIC_API_KEY}
```
Cerebras is the same block with `base_url: https://api.cerebras.ai/v1` and `api_key_env: CEREBRAS_API_KEY`. Check
both URLs against the providers' docs when wiring them up.

Hosted-reply example (`config.hosted.example.yaml`):
```yaml
llm:
  reply: {plugin: anthropic, model: claude-opus-5-5, effort: low, max_tokens: 1024, api_key_env: ANTHROPIC_API_KEY}
  filler: {plugin: qwen3.5-4b, kv_cache_gib: 1.0, max_num_seqs: 8}
tts:
  plugin: voxcpm2
```

**Config migration:**
- Move today's `llm:` keys under `llm.reply`, and `models:` into `ears:` (`nemotron`, `heads_repo`,
  `heads_file`).
- `load_config` must reject the old shape with a clear message that names the new keys. Don't silently
  reinterpret it.

**Package layout:**
```
app/plugins/
  base.py
  __init__.py
  llm/
    vllm_base.py
    qwen3_2507.py
    qwen35.py
    openai_compat.py
    anthropic.py
    fake.py
    _toolparse.py        # hermes_json + qwen_xml + <tool_call> splitting
  tts/
    voxcpm2.py
    openai_compat.py
    fake.py
  ears/
    nemotron.py          # wraps ears_core/batched.py
    fake.py
```
`app/llm.py`, `app/tts_voxcpm.py` and the direct engine construction in `mouth.py` / `server.py` go away. Their
code moves into plugins; it isn't copied.

## Prompts in one file: `prompts.yaml` (step 1b)

Every piece of text sent to an LLM lives in **one YAML file**, so prompts can be read and edited in one place
without touching code. Today they're scattered:
- **`app/chunker.py`:** `TTS_SYS`.
- **`app/mouth.py`:**
  - `TTS_TOOLS_SYS`, `LEADIN_SYS` + `LEADIN_SHOTS`, `STILL_SYS` + `STILL_SHOTS`, `BC_PROMPT` + `BC_SHOTS`;
  - f-strings built in code: the "don't repeat those" ack hint, the "user has only said a few words so far" and
    "user is currently saying" framings, the still-looking "You already said" line, the `INTERRUPTED` marker.
- **`app/tools.py`:** the tool and parameter descriptions. They're prompts too; they steer tool choice.

**Layout** (repo root, committed, synthetic examples only):
```yaml
version: 1
system:
  voice: |                 # was TTS_SYS
    You are a helpful voice assistant ...
  voice_tools: |           # was TTS_TOOLS_SYS (no longer built by string concatenation in code)
    ...
tool_leadin:
  system: |                # was LEADIN_SYS
    ...
  shots:                   # was LEADIN_SHOTS: list of [user, assistant] pairs
    - ["...", "..."]
still_looking:
  system: |
    ...
  shots: [...]
  user: 'The user asked: "$transcript". You already said: $said'
backchannel:
  system: |
    ...
  no_repeat: " You just said $acks -- don't repeat those."
  user_empty: "The user is speaking, but no words have been recognized yet."
  user_short: 'The user has only said a few words so far: "$partial"'
  user: 'The user is currently saying: "$partial"'
  shots: [...]
markers:
  interrupted: " —"
tools:                     # descriptions only; schemas (names, types, required) stay in app/tools.py
  web_search:
    description: "..."
    params: {query: "..."}
  get_weather: {...}
# later steps add: smart.system (SMART_SYS), escalation (criteria, shots), tasks (system, status_block, announce)
```

**Rules:**
- **Placeholders use `string.Template` (`$name`), not `str.format`.** Prompts can contain JSON or braces.
  Substitute with `substitute()`, so a missing variable fails loudly instead of sending a broken prompt.
- **One loader** (`app/prompts.py`) reads the file once at MOUTH start and validates it:
  - required keys exist;
  - every template's placeholders are exactly the ones the code passes;
  - shots are lists of `[user, assistant]` pairs.

  An invalid file stops startup with a message naming the key. Prompts are **not** reloaded live, so a session
  never sees half-changed prompts; restart MOUTH after editing.
- **Per-model overrides,** since prompts tuned on one model may not suit another. The 2507 few-shot turns exist
  because the 4B copied quoted examples.
  - Optional `prompts/<instance-name>.yaml` is deep-merged over `prompts.yaml` for requests that instance serves.
  - The instance name is the role block's `name:`, else its plugin name: e.g. `prompts/qwen3.5-4b.yaml` or
    `prompts/groq.yaml`. That keeps two `openai` instances apart.
  - Merging works key by key; a list such as `shots` is replaced whole.
  - Config: `mouth.prompts: prompts.yaml` (default), `mouth.prompt_overrides_dir: prompts/`.
- **The path is configurable** (`mouth.prompts`), so an experiment can point at a copy.
- **No user data** in the prompt files: they're committed. Few-shot examples stay synthetic.
- **The fakes route by prompt:** `FakeLLM._kind` matches system prompts by prefix. It must read them from the
  loader, not from module constants.

**This is a pure move, with a parity gate:**
- In the step-1b commit, a test asserts that the loaded `prompts.yaml` reproduces today's constants and f-string
  outputs **byte for byte**, for a few sample inputs.
- Then the constants are deleted from code.
- Separately, check the tool schemas rebuilt from `tools.py` plus `prompts.yaml` against today's `TOOLS`.

## Plugins

### `vllm_base.VLLMChatPlugin`: shared by the two Qwen plugins
Holds what's identical today in `app/llm.py`:
- the in-process `AsyncLLM` on MOUTH's `EngineLoop`;
- the pinned KV;
- the tokenizer lock;
- streaming deltas;
- `split_speakable` at `<tool_call>`;
- abort by request ID;
- shutdown.

Each Qwen plugin overrides only:
- `template_kwargs()`;
- `engine_kwargs()`;
- `parse_tool_calls(raw)`;
- `sampling()`;
- an optional `filter_text()`.

They're two separate plugins (user's call): the models differ in template, tool format and engine flags, so a
plugin per model keeps each one obvious.

- **`qwen3-4b-2507`:** Hermes JSON tool calls; no template switches.
- **`qwen3.5-4b`:**
  - `enable_thinking=False`;
  - `language_model_only=True`;
  - XML `<function=...>` tool calls;
  - a guard so `<think>` / `</think>` never reach the TTS.

  Both tool formats currently live in `tools.parse_calls`. Move them to `_toolparse.py`; each plugin uses its
  own format. Keep the existing tests passing, and split them per format.

### `voxcpm2`
Today's `VoxCPM2TTS`, unchanged in behaviour:
- the voice clip resampled once at load;
- the speaker cache primed at startup;
- warmup at every batch size up to `max_num_seqs`;
- abort by request ID;
- 48 → 24 kHz.

`caps.cloning = True`. `list_voices` scans `voices/` (with optional transcripts), as `discover_voices` does.

### `nemotron-tyger-1.0`
Wraps `EarsEngine` + `EarsService` (`app/ears_core/batched.py`) behind `EarsPlugin`. `ears_core/` itself doesn't
change. `server.py` then depends only on the interface; its state machine (speaking / halting / pending / reset)
stays in `server.py`.

### `openai` LLM (OpenAI-compatible chat completions)
- **Client:** the `openai` SDK's `AsyncOpenAI(base_url, api_key)` on MOUTH's `EngineLoop`, with
  `chat.completions.create(stream=True, ...)`. One client per plugin instance, so connections are reused.
- **Conversion:** messages are already in this shape. Serialise `arguments` dicts to JSON strings, and send tools
  as they are.
- **Streaming:**
  - `delta.content` → `text`.
  - The first `delta.tool_calls` fragment → `tool_start` (with the name if present). Accumulate arguments per
    `index`, then `json.loads` at the end.
  - Ignore `reasoning_content` and any other non-standard delta fields.
- **Validation:** check that the parsed arguments are a dict with the schema's `required` keys before MOUTH runs
  the tool. Invalid arguments become a tool error result; never run the tool on them.
- **Options:** `extra_body` passes server-specific options through. For example, a vLLM server needs
  `{"chat_template_kwargs": {"enable_thinking": false}}` for Qwen3.5.
- **Abort:** close the stream (`await stream.close()`) when `stop_flag` turns true, as `EngineLoop.iterate`
  already does by cancelling.
- **Failing fast:** `max_retries` 1, connect timeout ~2 s, read timeout ~10 s. A voice turn can't wait for long
  retries. On an error, MOUTH speaks a short apology line and ends the turn.
- **No reasoning text is ever spoken.** Some models on these endpoints put `<think>…</think>` in `content` instead
  of a separate field.
  - The plugin applies the same `<think>` guard as `qwen3.5-4b`, from `_toolparse.py` (`strip_think: true` by
    default), and also ignores `reasoning` / `reasoning_content` fields.
  - Provider switches that hide or limit reasoning go through `extra_body`, set per block in config. Don't
    hard-code them.
- **Rate limits.** A voice turn can make several requests: the reply, a filler, still-looking lines, plus
  backchannels if `filler` is also this plugin.
  - A `max_concurrency` semaphore per instance keeps under the provider's limits.
  - A 429 fails fast, with the apology line; don't wait out `retry-after` mid-turn.
  - Log 429 counts in the stats line.
- **Tool calls must come back as structured `tool_calls`,** not as `<tool_call>` text in `content`. If a model
  does emit them as text, the plugin falls back to the same `_toolparse` parsers. It logs that once, so a model
  without native tool support is noticed.

### Remote `reply`: what changes
- **The router moves off the box too.** `reply` decides tool calls and escalation, so the routing test set
  (step 11) must be run against it. That costs a little money, so it needs the user's OK. Pick the model that
  routes well, not just the fastest.
- **Every transcript now leaves the box,** not only escalated turns. The startup log must say which roles are
  remote, and the README's privacy note covers this case.
- **GPU memory:** if no role uses a local LLM plugin (`filler: reply` with a remote `reply`), MOUTH never loads vLLM,
  which frees ~10–12 GB. Record what the box can do with that (e.g. more conversations, or a larger TTS budget);
  don't assume it.
- **Latency:** measure the same numbers as for the local plugins, and compare against Qwen3.5-4B on the box:
  - time to first text;
  - first audio, plain and tool turns;
  - end of speech → first audio;
  - all at 1 and 4 sessions.

  Network round-trips hit backchannels and fillers hardest, so test `filler` both local and remote.

### `anthropic` LLM (Claude Messages API)
- **Client:** the official `anthropic` SDK's `AsyncAnthropic(api_key=..., base_url=...)`, using
  `client.beta.messages.stream(...)` because of `betas` (below). `base_url` lets it point at a Claude-compatible
  proxy. Don't use an OpenAI-compatible shim for Claude.
- **Model:** a config value. The default in the example is `claude-opus-5-5`. Measure time to first text and
  first audio against Qwen3.5-4B; the user picks the model for each role from those numbers.
- **Thinking:**
  - On Claude Opus 5.5, thinking is always on and can't be disabled: sending `{type: "disabled"}` or a budget
    returns a 400.
  - Effort is the control. Set `output_config.effort` from config (start at `low`; the default is `medium`).
  - Thinking blocks stream with empty text by default (`display: "omitted"`). The plugin ignores them for
    speech; they show up only as time before the first text.
- **`max_tokens` covers thinking plus text.** Size it per role from config (e.g. 1024 for `reply`, 256 for
  `filler`), not from MOUTH's `max_new` of 192 or 12. Keep replies short through the prompt.
- **Sampling:** don't send `temperature`, `top_p` or `top_k`; Opus 5.5 rejects them with a 400. Send them only
  if config sets them and the model accepts them.
- **Conversion:**
  - Top-level `system` comes from the system message.
  - Tools become `{name, description, input_schema}`, sent in a fixed order.
  - An assistant `tool_calls` message becomes `tool_use` blocks `{id, name, input}`.
  - Consecutive `role: "tool"` messages become **one** user message of `tool_result` blocks.
  - `tool_choice` is left at auto. Forced tool choice returns a 400 on this model.
- **Thinking blocks in history (preserved thinking):**
  - Inside one tool round, continue with the response's own content, thinking blocks included, **unchanged**.
    The plugin returns it as `native` with `tool_calls`; phase B sends it back as the assistant message.
  - Completed turns go into history as text plus `tool_use` / `tool_result` only, with no thinking blocks.
    Dropping all earlier thinking blocks is the allowed "remove a leading run" edit.
  - That keeps MOUTH's other history edits valid: dropping old turns, and cutting a barged reply to what was
    heard.
  - Test this against an account where the history check is enforced. Accounts created on or after 2026-08-31
    get a 400 on edited history.
- **Streaming:**
  - `content_block_start` with a `tool_use` block → `tool_start` (the name is known).
  - Text deltas → `text`; thinking deltas are ignored.
  - At the end: `get_final_message()`, then the `tool_use` blocks → `tool_calls` (the input is already a dict).
  - Set `eager_input_streaming: true` on each tool. The plugin then validates each input against its schema
    before MOUTH runs it: required keys and types. An invalid input becomes a `tool_result` with
    `is_error: true`.
  - Check `stop_reason` before running tools. On `max_tokens` with a `tool_use` block present, don't run it.
- **Refusals:**
  - Send `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`, so a policy decline is retried on
    a fallback model inside the same call.
  - If `stop_reason == "refusal"` still comes back, yield `("refusal", stop_details.category)`.
- **Prompt caching:**
  - The prefix (tools, then system) is stable, so set top-level `cache_control: {type: "ephemeral"}`.
  - Check that `usage.cache_read_input_tokens` is above zero on repeat turns. The voice prompt may be under the
    model's minimum cacheable size; if so, note it and move on.
- **Abort:** cancelling the async stream on `stop_flag` closes the HTTP connection. Tokens already generated are
  billed.
- **Errors:**
  - Typed SDK exceptions, most specific first: `RateLimitError`, then `APIStatusError` (5xx vs 4xx), then
    `APIConnectionError`.
  - `max_retries` 1 and short timeouts, as for `openai`.
  - On an error, MOUTH speaks a short apology line.

### `openai` TTS: removed (2026-10-04, user's call)
Built in step 8 (one streamed `/v1/audio/speech` request per text chunk, PCM, preset voices only) and measured live
with OpenAI `gpt-4o-mini-tts` next to the local Qwen3.5-4B. Turn first audio 680–837 ms p50 at 4 sessions (max
2.3 s), end of speech → first audio 1.8–2.2 s p50 (VoxCPM2: 1.0–1.3 s), and dead air between chunks up to 5 s,
since every chunk pays a request's time to first audio. Too slow for a voice reply, so the plugin was reverted;
NOTES.md has the numbers. The wire-rate wrapper (`AtWireRate`) stays for any future TTS that isn't 24 kHz.

### `fake` plugins
- **LLM and TTS:** today's `FakeLLM` / `FakeTTS` from `tests/fakes.py`, registered as plugins.
- **EARS:** an energy-threshold endpointer that emits `ENDPOINT` after N silent frames, with a fixed transcript
  from config. It lets the whole stack run on a laptop or in CI.
- **`config.fake.yaml`** runs `./run.sh` with no GPU and no network. Use it to check `server.py` / `mouth.py`
  wiring and the web page.

## Smart escalation and background tasks

Backchannels, fillers, and most conversational turns don't need a large model. The local model (`reply`, `filler`)
handles them on the box, and the remote `smart` model is called only for turns that need it. That gives lower
latency and cost on most turns, and only escalated content leaves the box.

**Routing rule:** escalation is a **tool the local model calls**, not a separate classifier.
- A classifier would add latency to every turn, the easy ones included.
- Answering locally first and escalating afterwards doesn't work in voice: the weak answer has already been
  spoken.

The tool path already exists, with fillers at the `<tool_call>` tag, still-looking lines, barge-in aborts, and tool
calls kept in history, so escalation reuses it.

### Escalation: the `ask_expert` tool (step 10)
All new prompt text from here on lives in `prompts.yaml`, under `smart`, `escalation` and `tasks`: the tool
descriptions, few-shot turns, `SMART_SYS`, the status block and the announcement prompt. None of it goes into
code.

**Tool schema** (offered only when `smart` is configured):
- `ask_expert(question: str)`.
- Its description gives the criteria for using it:
  - multi-step reasoning, maths, code;
  - detailed or precise facts beyond common knowledge;
  - comparisons, advice with consequences (health, money, legal), planning;
  - long explanations;
  - the user asking it to "think carefully", or being unhappy with an earlier answer.

  It also says: "when unsure whether you can answer well, use it". `question` is a self-contained restatement.

**Local model prompt:**
- The same criteria, plus few-shot turns showing an escalation and a local answer, in EN / RU / ES.
- Few-shot turns, not quoted examples: the 4B copies quoted examples verbatim (the 'mm-hm' and filler lessons).
- One rule for follow-ups: while the topic stays the same as an escalated turn, escalate the follow-up too.

**Filler that fits the tool:**
- Local plugins emit `tool_start` once the tool **name** has decoded, a few tokens after `<tool_call>` (~30–60
  ms), rather than at the bare tag. Both formats name the tool first: Hermes `{"name": "...`, and XML
  `<function=...>`.
- So the filler can match the tool: "let me think about that" for `ask_expert`, "let me check" for lookups.
- The `filler` role makes the lead-in, as today.

**Running it:**
1. MOUTH builds the smart model's request:
   - a `SMART_SYS` prompt with the same spoken persona and style rules as `TTS_TOOLS_SYS`: short (≤ ~4
     sentences unless asked for detail), no markdown;
   - the last `context_turns` turns of history;
   - the user's actual transcript;
   - the local model's `question` as a note.
2. The smart model gets the same tools (`web_search` etc.), up to `escalation.max_steps` tool rounds, with
   still-looking lines covering them.
3. Its answer streams **straight to the TTS** through the chunker, like a phase-B answer. The local model doesn't
   rephrase it, so no latency is added.
4. Other tools the local model called in the same turn run first, and their results go to the smart model as
   context.

**History:** user turn → assistant `tool_calls: [ask_expert]` → tool result `"answered by expert"` → assistant
message = the expert's answer.
- The call stays in history (the I-12 lesson), so the local model keeps escalating on follow-ups.
- The answer stays too, so it can handle easy follow-ups locally.

**Barge-in, refusals, errors:** as for any remote request.
- A barge-in aborts the smart stream.
- On a refusal or an error, MOUTH speaks a short neutral line, and the turn is remembered as interrupted or
  failed.

**Privacy:** only escalated turns and their `context_turns` of transcript leave the box. Log `escalated=true`,
timings and token counts; never the text.

### Escalation test set (step 11), no remote calls needed
`scripts/eval_escalation.py` checks the **routing decision**, which is the local model's alone. It runs only the
`reply` plugin, with a stub `smart` that returns a fixed answer, so it costs nothing.

- **Prompts:** ~80 synthetic ones in EN / RU / ES, single and multi-turn, each labelled with the expected route:
  - `local`: chit-chat, simple facts, opinions, short how-tos;
  - `tool`: weather, time, news, through the existing tools;
  - `expert`: as listed under the tool's criteria;
  - `background`: research, compare-and-recommend, plan a trip, draft something long.
- **Report:** a confusion matrix, plus the two error rates that matter:
  - **missed escalations** (hard → local): the main risk, quietly wrong answers. Target ≤ 10%;
  - **unneeded escalations** (easy → expert): cost and latency. Target ≤ 15%;
  - also how often `background` and `expert` are confused.
- **Models:** run it for `qwen3.5-4b` and `qwen3-4b-2507`. Tune the prompt and few-shot turns against it, keeping
  some held-out prompts.
- **Remote latency, with the user's OK and a few calls:**
  - filler first audio;
  - smart model time to first text;
  - end of speech → first answer audio for escalated turns.

### Background tasks (steps 12–14)
For work that takes minutes: research, comparing options, planning, drafting. The agent says it will look into
it, the conversation carries on, and it **takes the floor by itself** when the result is ready. Full duplex makes
this possible, because EARS already knows when the user is silent.

**Tools, offered with `ask_expert` when `tasks.enabled`:**
- `start_task(goal: str)` → `{task_id, status: "started"}`. The local model speaks a short acknowledgement ("I'll
  look into that and get back to you").
- `cancel_task(task_id)`, for when the user says "never mind".
- There's no status tool. Instead, a **task-status block** goes into the `reply` prompt on each turn:
  - for example, `Background tasks: #1 "cheapest flights to Almaty in May" — running 40 s; #2 "…" — done,
    announced`;
  - it sits as a system note just before the latest user message, not in the top-level system prompt, so the
    cached prefix stays the same;
  - the local model answers "how's it going?" from it.

**Task runner (MOUTH, one per session):**
1. Each task is an agentic loop on the `smart` plugin, in its own thread:
   - stream, then `tool_calls`, then execute, then append results, and repeat;
   - the task's own message list is append-only, so Claude's thinking blocks stay valid within it;
   - capped by `tasks.max_steps` tool rounds, `max_wall_s`, and the plugin's `max_tokens`.
2. The task's `SMART_SYS` asks for two outputs:
   - a **spoken summary** (2–4 sentences, opening with what it's about: "About the flights you asked for: …");
   - a **full result** (text, capped at ~4k characters).
3. Limits:
   - `max_concurrent` per session (default 1);
   - `max_per_session` (default 5);
   - a global cap across sessions, so a busy box can't fan out remote calls.
4. If the cap is hit, the tool returns an error and the local model says it can't start another task right now.
5. **Session end cancels every task.** Nothing persists: results live in memory for the session only.
6. **Failure or timeout:** the announcement says so in one sentence.

**Taking the floor: new wire messages**
- **MOUTH → EARS `F` `{"task": id}`: floor request.** Sent when a task finishes; MOUTH queues the announcement.
- **EARS → MOUTH `G` `{"task": id}`: floor granted.** EARS grants it only when all of these hold:
  - not speaking or halting, and no user turn pending or in flight;
  - no backchannel playing;
  - the user isn't mid-utterance, and has been silent for `announce_after_silence_ms`. The server counts
    consecutive `idle` `FRAME` events, so this works through the EARS interface.
- **Precedence:** a user `ENDPOINT` always wins over a pending grant; the request stays queued.
- **On grant:**
  - EARS enters speaking mode exactly as in `send_turn`, so barge-in and the second ear work as usual.
  - MOUTH speaks the summary through the normal turn path: `A` audio, then `D {"proactive": id}`.
  - The browser gets a `{"type": "task_done", "task": id}` JSON message for the UI.
- **If the user barges in during an announcement,** it stops like any reply. The task shows as "done, announced
  partially" in the status block, so the local model can repeat it when asked.

**History:**
- The announcement enters history as an assistant message, preceded by a tool-result-style note carrying a
  compact version of the full result (capped as in I-12).
- Easy follow-ups ("what was the cheapest one?") are answered locally from that note.
- Deeper ones ("compare the top two in detail") escalate. MOUTH then attaches the task's full result to the
  `ask_expert` context.

**Privacy and logs:**
- The goal and the context turns leave the box, as for escalation.
- Logs carry task IDs, timings, step and token counts, and status. Never goals, results or summaries.

**Tests:**
- **GPU-free:** a fake `smart` plugin with scripted tool rounds and delays, and the fake EARS. Cover:
  - start → continue chatting → done → `F` → wait for silence → `G` → announce;
  - a user turn arriving while a request is pending, which must win;
  - a barge-in mid-announcement;
  - cancel; caps; timeout; session close cancelling tasks.
- **On the box, with the user's OK:** one real task end to end, timed.

## Tests

- **Contract tests (`tests/test_plugin_contract.py`):** one suite for each kind, run against every plugin that
  can run there.
  - The fakes always run.
  - The real local plugins run only on the GPU box, behind an environment flag.
  - The hosted plugins run only with keys and an explicit flag, because they cost money: get the user's OK, use
    synthetic prompts only, and keep calls small.

  Checks:
  - events come in a valid order;
  - `stop_flag` aborts within one item;
  - `tool_calls` IDs are unique;
  - text output never contains tool-call or thinking markup;
  - TTS output is float32 at `out_sr`.
- **Conversion tests, GPU-free and network-free:**
  - Anthropic and OpenAI message/tool conversion are pure functions, so table-test them: tool rounds, several
    tool results in one turn, a barged turn, and a native block passed through.
  - Streaming parsers are tested on recorded synthetic event sequences (text, then a tool, then the end; a
    refusal; truncation at `max_tokens`; a split PCM byte).
- **Existing tests keep passing:**
  - `test_chunker.py`;
  - `test_llm_parse.py`, split per format;
  - `test_mouth_session.py`, driven through the fake plugins.

## Implementation order (each step is a commit on branch `plugins`)

| # | Step | Gate before the next step |
|---|---|---|
| 0 | Record the parity baseline on `voxcpm2` at the starting commit | numbers in NOTES.md |
| 1 | `base.py`, registry, the new config shape and loader (rejects the old shape) | unit tests |
| 1b | `prompts.yaml` + `app/prompts.py` loader; all prompts, f-string templates and tool descriptions move out of code | byte-identical test against the old constants and `TOOLS`; then delete the constants |
| 2 | LLM: `vllm_base` plus the two Qwen plugins; MOUTH uses `reply` / `filler`; plugin-issued tool-call IDs | parity: smoke test, `load_test.py` (3 runs × `--turns 6`), `--tools --fresh` |
| 3 | TTS: `voxcpm2` plugin; voice list through `list_voices`; resample wrapper | parity: smoke test, `load_test.py`, cloned-voice first audio |
| 4 | EARS: `nemotron-tyger-1.0` plugin; `server.py` on the interface | parity: `validate_ears_batched.py`, `load_test_ears.py` 1 and 4 |
| 5 | Fake plugins, `config.fake.yaml`, contract and conversion tests | the full stack runs with the fakes on CPU |
| 6 | `openai` LLM | contract tests; with a key, against a fast provider (Groq or Cerebras, whichever key the user gives) as `reply`: time to first text, first audio, tool turn, barge abort, 1 and 4 sessions, `filler` local vs remote |
| 7 | `anthropic` LLM | contract tests; with a key: same as 6, plus `cache_read_input_tokens`, history check (enforced account) |
| 8 | ~~`openai` TTS~~ (built, measured, removed: too slow) | — |
| 9 | README "Plugins" section: configs, how to add a plugin, the privacy note for hosted plugins | — |
| 10 | `smart` role + `ask_expert` tool; local plugins emit `tool_start` with the tool name; fillers that fit the tool | GPU-free tests with a fake `smart`; local-only turns at parity with step 9 |
| 11 | `scripts/eval_escalation.py` + prompt / few-shot tuning | missed escalations ≤ 10%, unneeded ≤ 15% on held-out prompts, for the chosen `reply` model; remote latency measured (with user OK) |
| 12 | Task runner: `start_task` / `cancel_task`, agentic loop on `smart`, caps, status block | GPU-free task tests |
| 13 | Floor request `F` / grant `G`; proactive turn; barge-in; `task_done` to the browser | GPU-free floor tests, then one live end-to-end task (with user OK) |
| 14 | Add `background` cases to the escalation test set; README section on escalation and tasks | routing report covers all four routes |

**Parity** means:
- unit tests pass;
- `smoke_test.py` passes;
- `validate_ears_batched.py` events match;
- `load_test.py` / `load_test_ears.py` medians land within the run-to-run noise the agent recorded (turn p50
  ±100 ms between identical runs, so compare 3-run medians);
- GPU peak memory is unchanged.

If a step can't reach parity, stop and report; don't paper over it.
