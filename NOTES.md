# tyger v1 — build notes

## Readability refactor (REFACTOR.md), 2026-10-06
The user asked for a readability refactor ("use common patterns so we can easily add plugins"; nemotron and the
tyger heads as one plugin). The plan and its steps are in REFACTOR.md: one commit per step, with parity checked on
a config without `llm.smart`.

**R-01, the review's correctness bugs (the only behaviour change before R-08):**
- **ASR thread:** finals and partials are transcribed and resolved separately
  (`EarsEngine._finish_finals` / `_finish_partials`). Before, one `try` covered both, so a partials failure re-ran
  `done("")` on finals already resolved. That could blank an endpoint transcript the hop had already handed back.
- **A lost MOUTH ends the conversation.** Every message to MOUTH goes through `to_mouth`. A broken connection (on a
  send, or in the reader) sends `{"type": "error", "error": "mouth_lost"}` to the page and closes the socket. Before,
  the hop loop died silently while the mic audio kept piling up in `buf`. The page now says the agent went away.
- **`ears.nemotron` is read** (it was ignored), and the startup log names the model.
- **An ack with nothing speakable** ("—", "*") is no ack: no TTS request, and not remembered as a recent ack. All
  text-to-TTS paths go through `Session._speech_audio`.
- Not changed: the anthropic plugin ignoring `max_new` is by design (thinking can't be turned off; `max_tokens`
  must cover it).

New tests: `test_ears_asr_jobs.py` (4 tests), `test_lost_mouth_ends_the_conversation` (it fails on the old
`server.py`), `test_unspeakable_backchannel_never_reaches_the_tts`. Stack on the no-smart config:
- smoke PASS: plain / weather / reminder / backchannel 235 / 230 / 286 / 128 ms;
- `load_test.py --sessions 4 --turns 6` turn first-audio p50 395 / 406 / 393 ms;
- `load_test_ears.py` end of speech → first audio p50 860 ms at 1 conv and 1071 ms at 4, no endpoint inside an
  utterance, one reply text.

**R-02, `app/messages.py`:** the message builders moved out of `mouth.py` unchanged. The two private helpers became
the module's public `with_shots` / `compact_result`. `tasks.py` imports them and `app.tools` at the top: before, it
imported `app.mouth` inside its functions, so that tests patching `mouth.execute_tool` reached tasks too. Both
modules now call `tools.execute_tool`, and tests stub it in one place (`tests/fakes.py` `stub_tools` /
`slow_tools`, replacing nine copied save / patch / restore blocks). `test_tasks_never_imports_mouth` reads
tasks.py's imports, inside functions too, and fails on the old file. Pure move: every GPU-free test passes with the
same counts.

**R-03, the `app/mouth/` package** (`app/mouth/__init__.py` lists the modules). `mouth.py` was split by job:
- `Server` (`__main__`) and `Mouth` (`service`, which gains `shutdown()`);
- `Session` (the conversation's state, the wire, the workers; `_dispatch` per wire message);
- `History` (turns trimmed whole);
- the turns: a `Turn` base with `AnnouncementTurn` and `SayTurn` (`turns.py`), and `ReplyTurn` (`reply.py`);
- `Speaker` (`speech.py`) and the thread helpers (`threads.py`).

Per-turn state used to sit on the Session (`_said`, `_audio_s`, `_play_end`, `_lang`, reset by hand in three
places). It is now a `Speaker` and the turn's own fields, created with each turn. The `box` dicts shared with
helper threads are now a `ToolRound` dataclass with `finished()`. The global session counter is now state on
`Server`. Tests read `s.history.messages`; `test_mouth_parts.py` tests History and Speaker directly.

One alignment: the apology / refusal line is spoken in the turn's voice. Before, it used the session's voice at
that moment, so a voice switch arriving mid-turn could change voices inside one reply, against the documented "one
voice per reply".

Stack on the no-smart config:
- smoke PASS: 210 / 238 / 286 / 158 ms;
- `load_test.py` turn first-audio p50 432 / 422 / 390 ms (R-01: 395 / 406 / 393; within the ±100 ms noise);
- `--tools --fresh` 381 ms p50, 0 errors;
- `load_test_ears.py` 873 / 1071 ms at 1 / 4 conv (2 reply texts at 4: batching, as before);
- no tracebacks in the stack log.

**R-04a, the `app/ears/` package** (`python -m app.ears`; `run.sh` starts it; `app/server.py` is gone). The parts:
- `EarsApp` (`web.py`) holds the plugin, config and voices, and its handlers are methods: it replaces four module
  globals.
- `Conversation` (`conversation.py`) replaces the 230-line `_conversation` closure and its 16-key `st` dict: named
  attributes, one method per plugin event and per browser command, and a dispatch table for MOUTH's messages.
- `AgentPlayback` (`playback.py`) holds what always changed together: `play_until`, `sent_s` and the 2nd-ear
  buffer. It also does the 24 → 16 kHz conversion, still the old per-chunk `np.interp` (R-04b changes that).
- The unused `bc_t0` was dropped.
- New `test_ears_parts.py` covers AgentPlayback's arithmetic, and the page / busy page / voices / slot release over
  a real aiohttp test server.

Stack on the no-smart config:
- smoke PASS: 200 / 207 / 275 / 179 ms;
- `load_test_ears.py` end of speech → first audio p50 879 ms at 1 conv, and 1066 / 1056 / 1078 ms at 4 (3 runs);
- one reply text throughout;
- `--full-speech` at 4 conv: the endpoint fired inside the utterance 9 / 16 times (the I-13 baseline: 8 / 12; O-01);
- no tracebacks.

**R-04b, the 2nd ear's resampler** (a model-input change, so measured on its own). MOUTH streams reply audio at
24 kHz in chunks, and EARS converts it to 16 kHz for the agent channel of the turn-taking heads. The old code ran
`np.interp` on each chunk separately, with `linspace(0, len-1, n)`: it dropped a fraction of a sample at every chunk
edge, stretched each chunk slightly, and had no anti-alias filter. `AgentPlayback` now uses one
`StreamingAudioResampler(24000 → 16000)` per reply, reset at the reply's start and at its end or barge.
`scripts/validate_second_ear.py` compares both paths on the EARS engine. Setup: a LibriSpeech clip as the agent
(16 → 24 kHz → chunks of 40 / 80 / 160 ms ± 50% → the 2nd ear), and another clip as the user cutting in 1.5 s in.

| 80 ms chunks, 36 trials | old (per-chunk interp) | new (stateful polyphase) |
|---|---|---|
| 2nd ear vs the original clip, SNR | **−0.9 dB** (−2.9 to +0.3 over chunk sizes) | **29.7 dB** (any chunk size) |
| agent alone: false barges / mean yield prob | 0 / 0.000 | 0 / 0.000 |
| user cuts in: barges / median latency | 29 / 36, 1.38 s | 28 / 36, 1.70 s |

Paired over the 28 trials where both barged: 24 fire in the same hop, 3 one hop later with the new ear, and 1 much
earlier (mean shift −0.18 hops). One trial barged only with the old ear. The medians differ by one hop because they
straddle a hop boundary. So barge detection is unchanged within this test's resolution, and the agent channel now
carries the actual reply audio instead of a drifting, aliased copy. This synthetic test has no echo of the agent in
the mic; a time-aligned 2nd ear is what would let the heads match echo against it.

Stack (no-smart config):
- smoke PASS;
- `load_test_ears.py` at 1 conv: 942, 892 and 870 ms. The first run's detection was 708 ms, and the repeats 653 /
  644 (R-04a: 641);
- at 4 conv: 1097 / 1093 ms, against 1056–1078 in R-04a;
- hop lag p50 / p90 unchanged; one ~590 ms max in the first run after a fresh start only;
- no tracebacks.

New test: `test_second_ear_does_not_depend_on_chunk_edges` (the same audio cut differently gives the same ear; it
fails on the old code).

**R-05, plugin-owned thresholds.**
- An EARS plugin declares `THRESHOLDS`: `Threshold(name, type, config_key, default, slider)` with an optional
  `Slider` (label, range, step, decimals, display scale and unit, the meter it marks, an "off" value behind a
  checkbox), in `app/plugins/base.py`.
- `Conversation` sends the plugin's values and slider specs in `ready`, and the page builds its tuning rows from
  them. EARS's server and the page no longer know any plugin's threshold names.
- nemotron-tyger keeps its thresholds exactly as they were (the user's ask): the same five sliders, ranges, steps,
  labels and config defaults, with fallback shown in seconds (frames × 0.08) and "off" = 100000.
  `test_nemotron_tyger_sliders_unchanged` guards that.
- The fake plugin drops its six unused dummy thresholds and declares the one it has (`endpoint_silence_frames`).
- The contract suite checks every plugin against its own spec, not against nemotron's names.

One display fix on the page: the orange threshold mark sat at the value's position inside the slider's range
(`(v − lo) / (hi − lo)`), but the meters show 0..1. So τ = 0.9146 was drawn at 89%, and barge / bc 0.5 at 37.5%. The
mark now sits at the value itself (91.5% / 50%).

Checked:
- the page logic in jsdom with nemotron's real spec (12 checks: decimals, frames ↔ seconds, off / on, preview-only
  while off, echoes);
- live, a WebSocket client read `ready` (the config values unchanged) and got the echoes for `set:` commands, with
  unknown names and bad values ignored as before;
- smoke PASS; `load_test_ears.py` 880 / 1089 ms at 1 / 4 conv.

I-15 corrected: the EARS engine alone aborts at exit (4 / 4 runs, on the code before R-01 too), not VoxCPM2 + EARS.

**R-06a, the `nemotron_tyger` plugin package** (`app/plugins/ears/nemotron_tyger/`; its `__init__` lists the
modules). nemotron-3.5-asr and the tyger 1.0 heads are now one plugin package, and `app/ears_core/` is gone:
- `asr.py`: `NemotronAsr`, everything specific to that checkpoint: GPU features, the layer-15 encode with its hook
  and all-zero cache, transcription. It is split out of `EarsEngine`, and the model id comes from `ears.nemotron`.
- `rnnt.py`: that model's decoder, moved unchanged.
- `constants.py`: every fixed number in one place. Before they sat in five modules: `STATE_NAMES` twice, the model
  id four times, and `FRAME_S` alone in a 3-line `labeling.py`.
- `head_batch.py` (RingKV, HeadBatch), `session.py`, `engine.py` and `service.py`: `batched.py` split by job.
- `heads.py`: for R-06a, the three research model classes, verbatim. R-06b replaces it.
- The research reference (LiveEars, EndpointEars, EncoderStream, the policy, the model classes) moved unchanged to
  `scripts/reference/ears_ref/`, still used by `validate_ears_batched.py`. The live path keeps only the state-head
  decision it uses, as `state_decision`, for the page's state name.
- `EarsSession.FALLBACK_SIL_FRAMES` is now `fallback_frames` (the threshold's name), so the plugin's attribute
  mapping is gone.

Parity, all on this box:
- `validate_ears_batched.py`: the same max diffs as the baseline to the last digit, events and transcripts match in
  all 5 sessions. Tick p50 at 4 sessions 67 / 58 / 52 ms over 3 runs, baseline 60;
- `validate_rnnt.py`: 40 / 40 identical, twice;
- `validate_barge.py`: 12 / 12, recall 0.94, first 3 words 0.83, 1.22 s;
- `validate_long_turn.py`: recall 1.00 / 0.97 / 0.94;
- the EARS contract passes;
- stack: smoke PASS, `load_test_ears.py` 874 / 1086 ms at 1 / 4 conv.

**R-06b, the heads from the TYGER-S2S-endpointer release** (the user's packaging of the heads: private repo
`nineninesix/TYGER-S2S-endpointer`, v0.1.0-rc1, pinned in `ears.heads_revision`).
- The release has `config.json` + `model.safetensors`: the Qwen3-0.6B decoder with the LoRA already merged (bf16),
  the added modules in fp32, the backbone's config embedded, and the frontend it was trained on pinned
  (nemotron-3.5-asr @ ea30d66, layer 15, 8 s window, 0.32 s hop, the energy norm).
- `TygerHeads` (`heads.py`) is its loader and inference model:
  - it builds the backbone from the embedded config: no peft, no Qwen3-0.6B download, rotary buffers in fp32 as
    `from_pretrained` builds them;
  - it loads all 335 tensors in their stored dtypes (strict);
  - it checks `model_type` and the frontend against `constants.py`, so a release needing another audio grid fails
    at load;
  - it keeps only `fuse` / `outputs`; the constant agent-text embedding is computed once.
- The checkpoint is now read once. Before, the research `.pt` was `torch.load`ed three times
  (`weights.verify_head_ckpt`, `YieldModel.load`, `EndpointModel.load`), with `strict=False`.
- EarsEngine loads the heads first, then the nemotron the release pins (`ears.nemotron` / `nemotron_revision`
  override it).
- Config: `ears.heads` + `ears.heads_revision` replace `heads_repo` / `heads_file`. The old keys are rejected with
  the new ones named. The research `.pt` stays only for the validators' reference
  (`scripts/reference/ears_ref` `SOURCE_REPO` / `SOURCE_FILE`, from the model card's provenance).
- Removed: `weights.ensure_public_cached` (unused) and `verify_head_ckpt`. Corrected: nemotron's licence is
  OpenMDW-1.1 (its card), not the NVIDIA Open Model Licence the README said.

Parity:
- new `scripts/validate_heads.py`: the research classes on the source checkpoint (LoRA merged as serving did)
  against the release, **weights identical 335 / 335, outputs max |diff| 0**;
- `validate_ears_batched.py`: the baseline's max diffs to the last digit, events and transcripts match, the same GPU
  memory (8447 MiB);
- `validate_barge.py` and `validate_long_turn.py` unchanged; the contract passes;
- stack: smoke PASS, `load_test_ears.py` 889 / 1073 ms at 1 / 4 conv, EARS start ~15 s and 5.7 GB as before.

**R-06c, the validators check the release** (the user's call). The research reference code and its checkpoint are
gone from the repo: `scripts/reference/` and `weights.research_checkpoint` (both in git history). R-06b's one-off
check (weights 335 / 335 identical, outputs exact) is the record that the release is the research model.
- **`rules.py`:** the per-frame turn rules (`begin_hop`, `switch_mode`, `on_frame`, `frame_energy`,
  `state_decision`) moved out of `EarsEngine`. They are plain functions on an `EarsSession` that say when the
  heads' context must restart, so the engine and the reference share them. New `tests/test_ears_rules.py` (6 tests:
  endpoint by head and by silence, minimum speech, backchannel debounce and refractory, barge-in and its seeded
  turn, energy) covers them directly, since the reference no longer checks them independently.
- **`scripts/ears_reference.py`:** `ReferenceEars` runs one conversation through the same release the plain way.
  That means HF processor features on the CPU, the full encoder with `hidden_states[15]`, `TygerHeads` stepped with
  HF's `DynamicCache`, and `generate()` on the whole turn. `validate_ears_batched.py` compares the engine with it.
  Result: events and transcripts match in all 5 sessions, max diffs 0.0022–0.0059 (against the research reference
  they were 0.0018–0.0077), tick p50 58 ms at 4 sessions.
- **`scripts/validate_heads.py`** now checks the release alone, on nemotron layer-15 features of LibriSpeech:
  - it loads (strict, dtypes kept, frontend fits);
  - its outputs are deterministic and well-formed;
  - probabilities frame by frame vs one pass differ by max 0.0071, alone vs batched by 0.0065 (tolerance 0.02).
  Random features would be useless here: they push the state logits to |46| and bf16 noise to 0.66, while in fp32
  the same comparisons agree to 2e-4.
- Stack: smoke PASS, `load_test_ears.py` 881 / 1092 ms at 1 / 4 conv.

**R-07a, plugin bases and typed options.**
- **Bases:** `app/plugins/base.py`'s Protocols are now ABC bases (`Plugin`, then `LLMPlugin`, `TTSPlugin` and
  `EarsPlugin`) that every registered plugin subclasses. They hold the shared defaults: `remote` False,
  no-op `warmup` / `shutdown`, `self.opts` / `name` / `eloop` from the constructor.
- **EARS:** `EarsPlugin` also holds the threshold plumbing both EARS plugins had copied (`defaults` from config,
  `open` merging overrides, `thresholds`, `set_threshold`). A plugin implements `open_session`, and a session keeps
  its thresholds as attributes; the fake's session used to keep them in a `th` dict.
- **Options:** each plugin declares its config block as an `Options` dataclass (`app/plugins/options.py`: typed,
  defaulted, frozen). `validate_config` builds it for every block, so a bad key fails at load, before any model
  loads, with the key named: `unknown option kv_cache_gb (did you mean kv_cache_gib?)`, `model is not set`,
  `allow_zero_shot must be a bool`. Before this, a typo was silently ignored and the default used. An EARS block's
  threshold keys are checked by its THRESHOLDS. Values are coerced as YAML leaves them: an int where a float is
  declared, and an all-digit revision as a string.
- **No behaviour change:** every default is the one the old `conf.get(key, default)` used.
- **Tests:**
  - `test_config.py`: unknown, missing and wrong-type keys, coercion, and `config.fake.yaml` loads; the first two
    fail on the old code.
  - `test_plugins.py`: every registered class is a concrete subclass of its kind's base and has its own `Options`.
- **Checks:**
  - GPU-free suite passes;
  - local contract test 6/6;
  - stack (no-smart config): smoke PASS, `load_test.py` turn p50 397 ms, `load_test_ears.py` 854 / 1074 ms.

**R-07b, `HostedLLM`** (`app/plugins/llm/hosted.py`), the base of `openai` and `anthropic`. The two plugins had
copied about 60 lines of set-up and request plumbing; it now lives once:
- the shared Options (model, base_url, api_key_env, remote, timeouts, retries, max_concurrency, queue_timeout_s);
- the key check and the `<SDK>_LOG=debug` refusal;
- the client and semaphore made on the EngineLoop (`make_client` is the subclass's);
- `guarded()`: a concurrency slot, `busy` after queue_timeout_s, SDK errors -> LLMError (`llm_error`, by the
  error names both SDKs share);
- shutdown.

A plugin keeps only its request shape and stream parser. The anthropic plugin still ignores `max_new` (its
`max_tokens` covers thinking). Behaviour unchanged; `test_hosted_openai.py` gained the timeout kind and a check
that a stopped request gives its slot back, and both pass on the old code too. One live gpt-5.4-mini request
through the ported plugin streamed text.

## Box (2026-10-06) — another new RTX 5090; the losing LLM plugins removed
NVIDIA RTX 5090 (32 607 MiB), driver 595.71.05 / CUDA 13.2; AMD EPYC 9B14, 384 threads, 503 GB RAM. Idle P8; the
clocks under load were not sampled this round, but latency matches the healthy boxes. Fresh-box install (README
step 1) went through unchanged. Downloads were fast (Gemma E4B 123 s, its NVFP4 backup 71 s, VoxCPM2 40 s). The
first `./run.sh` (models cached by the contract test) took 3.4 min, 83 s of it Gemma's first compile. A warm restart
takes 2.1 min: VoxCPM2 ~45 s, Gemma ~40 s, TTS warmup ~25 s, EARS ~13 s.

**The cleanup (user, 2026-10-06):** the responder stays `gemma-4-e4b` (FP8 at load), and `gemma-4-e4b-nvfp4` stays
registered as its backup. The other local LLM plugins were removed: the two earlier responders, the NVFP4 trials,
and the other Gemma 4 sizes. So were the tool-call formats only they used (Qwen XML, Nemotron `<TOOLCALL>`, and the
"other format" fallback). Their numbers stay in this file, and their code in git history.

**Contract test, both Gemma plugins (`TYGER_CONTRACT_LOCAL=1`):** this is the first time the suite ran on them (it
had still run the removed plugins). FP8 E4B passed as it stood. The NVFP4 backup failed the plain-question check:
"Tell me one short fun fact about otters." went to `web_search` 8 / 8, where FP8 E4B answered aloud 8 / 8. The call
itself was well-formed and parsed, so the plugin contract held; this is a routing difference. It fits the routing
eval's slightly lower no-smart score for NVFP4 (98.6 vs 99.6% right). The check now asks for a short joke, which has
nothing to look up (8 / 8 spoken on both models). Routing stays `eval_escalation.py`'s job. **If the backup is ever
switched in, expect more trivia lookups**, each paying a tool round (and failing while Tavily is off). Separately,
the test process aborts at exit after every suite has passed (I-15).

**Production config (with `llm.smart`), this box:**

| test | result |
|---|---|
| `smoke_test.py` | plain / weather / reminder / backchannel first-audio 279 / 275 / 316 / 160 ms; barge, EARS: PASS. The barge test's tool turn called `ask_expert` once (one synthetic prompt to OpenAI) |
| `load_test_ears.py`, 1 conv | end of speech → endpoint p50 / p90 641 / 753 ms; endpoint → first audio p50 263 ms; **end of speech → first audio p50 904 ms**; hop lag p50 / p90 39 / 64 ms |
| `load_test_ears.py`, 4 conv | 749 / 915 ms; 399 ms; **1109 ms**; hop lag 69 / 152 ms |

These are single runs. Gemma gave the same "count to ten" reply text in every turn (4 at 1 conv, 16 at 4 conv).
Detection at 1 conversation is 641 ms p50, against 803–883 on the last box and 684–686 on the box before. So the
last box's slow detection was the box, not the code (L-02).

## Box (2026-10-05) — another new RTX 5090, healthy
NVIDIA RTX 5090 (32 607 MiB), driver 595.71.05 / CUDA 13.2; AMD EPYC 7C13, 256 threads. **Not throttled**: under
load P1, memory 13 801 MHz. Fresh-box install (README step 1) went through unchanged. The first `./run.sh` gave up
waiting for MOUTH: slow downloads (VoxCPM2 + Gemma ~22 GB, Gemma's load counted 393 s of it) and VoxCPM2's first
compile ran past `run.sh`'s ~15 min wait, and its cleanup left Gemma's EngineCore holding 14.7 GB (I-14). The
second start: MOUTH ~5 min, EARS +5 min (nemotron + heads download); warm restart ~2.2 min.

### Gemma 4 memory: the encoders are skipped; per-layer embeddings; E2B against E4B
Same box, same scripts, both models with the config's options (FP8, `language_model_only`, thinking off, KV 3 GiB,
temperature 0.7; the model card recommends `temperature=1.0, top_p=0.95, top_k=64`, not used for either). Stack
runs use a copy of `config.yaml` **without `llm.smart`** (no OpenAI calls from `load_test.py`'s plain prompts;
tasks off with it); the routing eval measures the with-smart prompts offline. Tavily is off, so tool turns get the
tool's error result at once: tool-turn numbers don't compare with earlier rounds.

**The vision and audio encoders are not loaded.** With `language_model_only`, vLLM's
`MultiModalConfig.get_limit_per_prompt` is 0 for every modality, so `Gemma4ForConditionalGeneration` builds both
`vision_tower` and `audio_tower` as `StageMissingLayer` (`_mark_tower_model`). Checked inside the loaded model
(`apply_model`, every CUDA tensor by name): both towers are `StageMissingLayer`, no vision or audio tensor is on
the GPU. vLLM's "Model loading took" is 10.54 GiB for E4B (10.66 standalone). The tensors add up to 10.39 GiB:
10.20 GiB of weights, which the safetensors headers predict for the text model with FP8 linears (10.25), plus
0.19 GiB of buffers (rotary caches, the PLE input buffer for CUDA graphs). A tower would add 0.3–0.6 GiB.

| on the GPU (GiB) | E4B | E2B |
|---|---|---|
| per-layer embedding table (bf16; layers × 256 × 262 144) | **5.25** (42 layers, 2.82 B params) | **4.38** (35 layers, 2.35 B) |
| token embeddings (bf16) | 1.25 | 0.75 |
| decoder linears (FP8) | 3.70 | 1.75 |
| buffers | 0.19 | 0.19 |
| total tensors / vLLM "Model loading took" | 10.39 / 10.54 | 7.06 / 7.10 |
| checkpoint towers, not loaded (bf16): vision / audio | 0.32 / 0.58 | 0.31 / 0.57 |

The per-layer embedding table is half of E4B's GPU weights and 62% of E2B's. On-the-fly FP8 only quantizes Linear
layers, so it stays bf16. It is a lookup (a few rows per token), so it costs memory, not speed; offloading it to CPU
memory was ruled out by the user (too slow).

**E2B against E4B:**

| | E4B | E2B |
|---|---|---|
| single stream decode (512 tokens, alone) | 186 tok/s | 281 tok/s |
| GPU idle / peak at 4 sessions (`load_test.py`) | 29.8 / **30.3 GB** | 26.3 / **26.8 GB** |
| smoke plain / weather / reminder / backchannel first-audio | 256 / 247 / 293 / 170 ms, PASS | 183 / 188 / 217 / 147 ms, PASS |
| `load_test.py --sessions 4 --turns 6` turn first-audio p50, 3 runs | 455 / 425 / 458 (median 455) | 371 / 365 / 345 (median 365) |
| backchannel first-audio p50 | 448–492 ms | 323–378 ms |
| `--tools --fresh` turn first-audio p50, 3 runs | 412 / 401 / 422 (median 412) | 329 / 329 / 346 (median 329) |
| `load_test_ears.py` 1 conv: endpoint → first audio p50; end of speech → first audio p50 (2 runs) | 231, 237; 1113, 1034 ms | 179, 169; 1005, 978 ms |
| `load_test_ears.py` 4 conv: endpoint → first audio p50; end of speech → first audio p50 (3 runs) | 265–271; 1169 / 1183 / 1194 ms | 209–218; 1131 / 1145 / 1119 ms |
| **routing eval, with smart + tasks** (140 cases × 8): right; missed escalations | **98.7%; 0.6%** | **75.5%; 42.9%** (held-out 46.5%, fresh 61.3%) |
| with smart: unneeded escalations; tool → tool | 0%; 256 / 256 | 0%; 254 / 256 |
| with smart: speech in the wrong language / narration or markup | 6 / 0 of 347 | 75 / 95 of 599 |
| with smart: answer after a tool result in the wrong language | 2 of 412 | 64 of 414 |
| **routing eval, no smart** (70 cases × 8, production without a key): right; tool → tool | 99.6%; 246 / 248 | 99.8%; 247 / 248 |
| no smart: answer after a tool result in the wrong language | **0 / 246** | **78 / 247** |

E4B reproduces last round's eval (98.3% right, 0.4% missed) within sampling. E2B fails the routing gate (≤ 10%
missed) by a wide margin, and its failures aren't sampling noise:
- **It answers hard questions itself:** chest pain, mRNA vaccines, TCP handshake, palindrome code, job offers,
  RU / ES health and legal questions: 8 / 8 local. Sometimes it says the hand-off instead of calling the tool ("Ask
  expert explain how blockchain works in detail.", "I can ask an expert for the best financial advice on this.").
- **It opens the thinking channel despite `enable_thinking: false`:** `<|channel>thought\nThinking Process: …` on
  reasoning prompts (bat and ball, sum to 100, savings maths, plans); most of the 95 narration / markup flags.
  MOUTH would speak it.
- **Answers after a tool result follow the few-shot turns' language, not the user's:** English weather, reminder
  and message questions answered in Russian (78 / 247 without smart), although the language hint names English.
- Tool calls themselves are fine: lookups and actions are called as reliably as with E4B.

**Recommendation: keep E4B.** E2B frees ~3.5 GB (peak 30.3 → 26.8 GB of 32.6) and is ~50–90 ms faster to first
audio, but it regresses routing, the reply language and spoken markup, which the user named as what matters
("robust tool calling is crucial"). What the freed memory would have bought, not measured: VoxCPM2 or Gemma KV
headroom for a 5th–6th conversation, or room for L-02 / L-01 option 2 without crowding the 32 GB.

### NVFP4 responders: Qwen3-8B / 14B, Nemotron Nano 9B v2, Gemma 4 26B-A4B (user's request)
NVIDIA's ModelOpt NVFP4 checkpoints (W4A4, scales in the checkpoint: the plugins leave `quantization` unset). vLLM 0.30
serves all four on the 5090 (sm_120): FlashInfer-CUTLASS NVFP4 GEMM, and the FLASHINFER_CUTLASS NVFP4 MoE backend
for Gemma 26B (its first load JIT-compiles ~8 min of MoE kernels). New plugins: `qwen3-8b-nvfp4` / `qwen3-14b-nvfp4`
(hybrid Qwen3, `enable_thinking: false`, Hermes JSON, the `<think>` guard), `nemotron-nano-9b-v2-nvfp4` (NemotronH
Mamba-2 hybrid, `/no_think` in the system prompt, fp32 SSM cache as NVIDIA asks, a new `<TOOLCALL>[…]` parser
format; its template can't take a `None` content, so tool-call turns get `""`), `gemma-4-26b-a4b-nvfp4` (the E4B
plugin, another checkpoint). Same eval settings as above (temperature 0.7, 8 samples, prompts tuned on Gemma E4B).

**Memory budget.** On the running stack VoxCPM2 takes 9.2 GB, EARS 5.7 GB (+~0.4 under load) and the E4B engine
15.0 GB (10.5 GiB weights + 3 GiB KV + ~1.1 GiB graphs / activations). An LLM engine can have ~17 GB, i.e. ≤ ~12.5
GiB of weights with the same 3 GiB KV.

| | E4B FP8 (ref) | Qwen3-8B | Nemotron Nano 9B v2 | Qwen3-14B | Gemma 4 26B-A4B |
|---|---|---|---|---|---|
| weights (vLLM "Model loading took") | 10.54 GiB | 5.98 | 7.38 | 9.97 | **17.05** (vision tower skipped) |
| fits next to VoxCPM2 + EARS | yes | yes | yes | yes | **no** (~4.5 GiB over) |
| single stream decode | 186 tok/s | 217 | 180 | 141 | 180 |
| **with smart + tasks: right; missed** (gate ≤ 10%) | **98.7%; 0.6%** | 92.4%; 13.8% | 88.6%; 20.4% | 91.0%; 18.5% | **98.4%; 1.9%** |
| missed: held-out / fresh | 2.1 / 0.0% | 9.7 / 40.0% | 16.0 / 37.5% | 11.1 / 33.8% | 4.9 / 0.0% |
| tool → tool | 256 / 256 | 253 | 243 | 251 | 256 |
| speech: wrong language / narration | 6 / 0 of 347 | 8 / 3 of 419 | 4 / 27 of 466 | 2 / 0 of 445 | 0 / 0 of 354 |
| after a tool result: wrong language / markup | 2 / 5 of 412 | 15 / 5 of 432 | 12 / 5 of 388 | 1 / 0 of 435 | 0 / **333** of 424 |
| **no smart: right; tool → tool** | 99.6%; 246 / 248 | 100%; 248 | 99.1%; 243 | 100%; 248 | 98.4%; 239 |
| no smart: speech in the wrong language | 3 / 314 | **43** / 312 | **40** / 319 | 10 / 312 | 0 / 321 |
| no smart: after a tool result, wrong language | 0 / 246 | 11 / 248 | 19 / 243 | 0 / 248 | 0 / 239 |

None of the three that fit meets the routing gate, so none got the stack benchmark:
- **Qwen3-14B and 8B lean local**, like Qwen3.5-4B and 2507 before them (step 11): TCP handshake, regex / palindrome
  code, health questions, a rental contract, "recount carefully", follow-ups on an escalated topic. Fresh-split
  misses 34–40%. The 8B also answers plain English questions in Russian without smart (43 / 312): the Russian
  few-shot turns pull it. Both call tools reliably.
- **Nemotron Nano 9B v2** misses 20%, talks about its tools ("I'll use the ask_expert tool…") and answers English
  questions in Russian (40 / 312). Its model card doesn't list Russian.
- **Gemma 4 26B-A4B meets the gate** (98.4%, 1.9% missed, no wrong-language reply) but needs ~4.5 GiB more than this
  card has. Its 17 GiB: 10.6 NVFP4 packed weights (only the routed experts are NVFP4), 3.1 bf16 (every layer's
  attention and dense MLP: the checkpoint's ignore list), 1.4 token embeddings, 1.3 the FP4 weights' FP8 block scales. FP8 for the
  bf16 linears would save ~1.5 GiB, not the ~4.5 needed. After a tool result it nearly always opens an **empty** thought channel
  (`<|channel>thought\n<channel|>` then the answer): a filter would have to strip that before the TTS (E2B leaks
  the same channel with content). Without smart it answered 9 / 248 RU / ES reminder / message requests "not
  connected" without calling the tool.

### unsloth NVFP4 Gemma 4: E4B and 12B (user's request)
llm-compressor (compressed-tensors) checkpoints, mixed precision. **E4B** (`unsloth/gemma-4-E4B-it-NVFP4`, plugin
`gemma-4-e4b-nvfp4`): FP8-dynamic attention and 11 layers' MLPs, NVFP4 for the other MLPs, and the **per-layer
embedding table in int8** (`pack-quantized`), which vLLM loads (packed in int32 words): 2.63 GiB instead of 5.25.
**12B** (`unsloth/gemma-4-12b-it-NVFP4`, `gemma-4-12b-nvfp4`): the encoder-free "unified" Gemma 4 (vLLM
`Gemma4UnifiedForConditionalGeneration`: no towers, small vision / audio projections), 48 dense layers, FP8 attention,
NVFP4 MLPs, no per-layer embeddings. Same eval settings as above.

| | E4B FP8 (current) | **E4B NVFP4 (unsloth)** | 12B NVFP4 (unsloth) |
|---|---|---|---|
| weights on the GPU (vLLM "Model loading took") | 10.39 (10.54) GiB | **6.85 (7.10)**: PLE int8 2.63, token emb 1.25, FP8 1.49, NVFP4 1.14 | 9.06 (9.25): NVFP4 3.96, FP8 2.75, token emb 1.88 |
| single stream decode | 186 tok/s | **213** | 129 |
| with smart + tasks: right; missed (held-out / fresh) | 98.7%; 0.6% (2.1 / 0) | **98.7%; 0.2% (0.7 / 0)** | 89.6%; 14.8% (18.1 / 11.2) |
| with smart: speech wrong language / narration | 6 / 0 of 347 | 12 / 1 of 345 | 1 / **310** of 715 |
| with smart: after a tool result, wrong language / markup | 2 / 5 of 412 | 0 / 4 of 411 | 1 / **334** of 380 |
| no smart: right; tool → tool | 99.6%; 246 / 248 | 98.6%; 240 / 248 | 88.6%; **184** / 248 |
| no smart: speech wrong language | 3 / 314 | 5 / 320 | 0 / 381 |

- **12B fails:** it misses 15% of escalations, answers reminder / message requests "not connected" without calling
  the tool (25.8% of tool cases without smart), and opens the empty thought channel (`<|channel>thought\n<channel|>`)
  in plain replies too, once in a loop. It is also the slowest decoder here.
- **E4B NVFP4 routes like E4B FP8.** Slightly worse: "send a message to Alex" answered locally 7 / 8 without smart
  (FP8: 2 / 8; 8 / 248 tool cases vs 2), and Spanish opinion / naming questions answered in English a little more
  often (12 vs 6 with smart). No thought-channel tags at all.

Full stack, E4B NVFP4 against E4B FP8 (config without `llm.smart`, same scripts, same box, the FP8 runs from earlier
the same day):

| | E4B FP8 | E4B NVFP4 |
|---|---|---|
| GPU idle / **peak at 4 sessions** | 29.8 / **30.3 GB** | 26.2 / **26.7 GB** |
| smoke plain / weather / reminder / backchannel first-audio | 256 / 247 / 293 / 170 ms, PASS | 259 / 287 / 273 / 149 ms, PASS |
| `load_test.py` turn first-audio p50, 3 runs | 455 / 425 / 458 (median 455) | 443 / 421 / 440 (median 440) |
| `--tools --fresh` first-audio p50, 3 runs | 412 / 401 / 422 (median 412) | 414 / 459 / 429 (median 429) |
| `load_test_ears.py` end of speech → first audio p50, 1 conv (2 runs) | 1113, 1034 | 1119, 1012 |
| 4 conv (3 runs) | 1169 / 1183 / 1194 | 1204 / 1189 / 1154 |

Latency is the same within run-to-run noise: first audio waits on the 40-character first chunk and the TTS, and the
faster decode doesn't show. Under 4-way batching the forced "count to ten" reply varied in 2 of 3 runs (the test
warns). E4B NVFP4 frees **3.6 GB** at the same speed and routing, at the cost of slightly more skipped message calls
and English replies to Spanish. Not switched: the user decides (`plugin: gemma-4-e4b-nvfp4`, drop `quantization`).

Also on this box: end of speech → endpoint at 1 conversation is 803–883 ms p50 (last box: 684–686, two runs within
4 ms; here they differ by 80 ms) and 912–932 ms at 4. The test utterance renders a little longer here (3.04 s,
voiced 2.64 s; last box 2.88 / 2.62 s). Not looked into; L-02 measures detection next.

## Box (2026-10-04) — another new RTX 5090, healthy
NVIDIA RTX 5090 (32 607 MiB), driver 595.91.07 / CUDA 13.2; AMD EPYC 7742, 128 threads, 125 GB RAM. **Not
throttled**: under load P1, SM 2910 MHz, memory 13 801 MHz (max 3105 / 14 001). Fresh-box install (README step 1,
`uv` + Python 3.12) went through unchanged; first `./run.sh` with all downloads ~14 min, warm restart ~2.5 min.
Code at `b2b6523` (Qwen3.5-4B responder), cloned voice `en`:

| test | metric | this box | box of 2026-10-03 |
|---|---|---|---|
| `smoke_test.py` | plain / weather tool / reminder / backchannel first-audio | 243 / 351 / 429 / 194 ms, PASS | 263 / 264 / 301 ms with Qwen3-4B; plain 233–277 with Qwen3.5 |
| `load_test.py --sessions 4 --turns 6` | turn first-audio p50 (3 runs × 24) | 514 / 482 / 481 ms | 446–550 ms |
| | backchannel first-audio p50 | 530 / 421 / 495 ms | 451–490 ms |
| | GPU peak | 27 696 MiB | 27.5 GB |
| `load_test.py --tools --fresh` (4 sessions) | first-audio p50 / max, dead air max | 454 / 582 ms, 1343 ms | 456–460 ms (p50) |
| `load_test_ears.py` (old script) | 4 conv: hop lag p50 / p90, end of speech → first audio p50 | 107 / 268, 1105 ms | 90–100 / 192–256, 992–1109 ms |

The deterministic `load_test_ears.py` (I-13) baseline is in the next section; numbers from the old script don't
compare with it.

### `load_test_ears.py` made deterministic (I-13)
Its utterance was the LLM's reply to a "Say: …" prompt, and the agent's reply to that was free text, so utterance
and reply lengths changed run to run (NOTES 2026-10-03: hop lag p90 190–260 ms between identical runs). Now:
- **Fixed utterance:** "Could you please count from one to ten for me?" spoken by MOUTH's TTS through a new
  TTS-only wire message `S` (no LLM, not remembered; EARS never sends it). VoxCPM2 samples at temperature 0 with
  seed 42, so it is the same audio every run: 2.88 s, voiced 2.62 s, no inner pause > 100 ms.
- **Fixed reply:** counting to ten leaves the LLM nothing to vary. Qwen3.5-4B at temperature 0.7: 8/8 fresh
  sessions and 3/3 turns with history gave the same text; at 4 conversations 16/16. The test compares reply texts
  by hash (never printed) and warns if they differ. A seeded LLM wasn't needed.
- **The reply audio still varies with load:** VoxCPM2 isn't batch-invariant, so the same text renders 9.76 s alone
  and 8.8–10.4 s when batched with other conversations' requests. That is the real system; the test reports it.
- **Fixed hop phase:** EARS cuts hops at multiples of 320 ms of the stream, so where the speech ends inside a hop
  (0–320 ms of detection delay, L-02) depended on wall-clock turn timing. Each utterance now gets lead-in silence so
  its speech ends at 0, ¼, ½ or ¾ of a hop, cycling over turns and conversations. `--turns` defaults to 4, which
  covers the phases evenly.
- `--full-speech` is now a fixed three-sentence utterance (TTS pauses of up to 880 ms between sentences) ending
  in the same request: at 4 conversations × 3 turns the endpoint fired inside it 8/12 times (the O-01 guard rail).

Back-to-back runs, same stack (`load_test_ears.py`, 4 turns each):

| | end of speech → `endpoint` p50 / p90 | endpoint → first audio p50 | **end of speech → first audio** p50 / p90 | hop lag p50 / p90 |
|---|---|---|---|---|
| 1 conv, run 1 / 2 | 686 / 805, 684 / 803 ms | 283, 282 ms | **970 / 1088, 966 / 1086 ms** | 60 / 71, 60 / 62 ms |
| 4 conv, run 1 / 2 / 3 | 838 / 982, 885 / 964, 840 / 1016 ms | 417, 430, 430 ms | **1265 / 1415, 1250 / 1408, 1268 / 1453 ms** | 90 / 224, 89 / 223, 80 / 214 ms |

Before the hop-phase fix (fixed speech already), the 4-conversation gap p50 spread over 1208–1356 ms in 3 runs;
now 1250–1268. At 1 conversation, two runs agree within 4 ms. This is the yardstick for the plugin refactor's
parity checks: compare p50s over 3 runs at 4 conversations, and single runs at 1.

Detection is ~160 ms slower at 4 conversations than at 1 (p50 ~850 vs 685 ms), more than the hop lag difference
(~30 ms at p50) explains; not yet looked into (L-02 measures there next).

### Plugin refactor: parity baseline (PLUGINS.md step 0)
Branch `voxcpm2` at `a7f4cf8`, this box, config as committed (Qwen3.5-4B, VoxCPM2, cloned voice `en`). Every
plugin step on branch `plugins` is compared against these:

| test | metric | baseline |
|---|---|---|
| unit tests | `test_chunker`, `test_llm_parse`, `test_mouth_session` | all pass |
| `smoke_test.py` | plain / weather tool / reminder / backchannel first-audio | 243 / 351 / 429 / 194 ms, PASS |
| `load_test.py --sessions 4 --turns 6` | turn first-audio p50 (3 runs) | 514 / 482 / 481 ms (median 482) |
| | backchannel first-audio p50 | 530 / 421 / 495 ms |
| | GPU peak | 27 692–27 696 MiB |
| `load_test.py --sessions 4 --turns 6 --tools --fresh` | first-audio p50 / max, dead air p50 / max | 454 / 582, 1011 / 1343 ms |
| `load_test_ears.py --sessions 1` (2 runs) | end of speech → endpoint / → first audio p50 | 686, 684 / 970, 966 ms |
| `load_test_ears.py --sessions 4` (3 runs) | end of speech → endpoint / → first audio p50; hop lag p90 | 838–885 / 1250–1268 ms; 214–224 ms |
| `validate_ears_batched.py` | events / transcripts vs reference; max head diff | all match; 0.0077 |
| | tick ms p50 / p90, 1 and 4 sessions | 44 / 68, 71 / 166 |
| `validate_rnnt.py` | padded batch vs alone | 40/40 identical |
| `bench_tts_voice.py` | cloned voice (speaker cache) generate → first audio p50 | 89 ms (zero-shot 89, clip every request 115) |

**Step 1b (prompts.yaml):** byte-identical by test (constants, f-string outputs, tool JSON against `ddb1128`); smoke
PASS.

**Step 2 (LLM plugins `qwen3.5-4b` / `qwen3-4b-2507` on `vllm_base`, roles `reply` / `filler`, plugin-issued
tool-call IDs, neutral history):** rendered chat prompts for a tool-turn history are identical to the old ones for
both models (neither template renders the IDs).

| | baseline | step 2 |
|---|---|---|
| smoke plain / weather / reminder / backchannel | 243 / 351 / 429 / 194 ms | 240 / 305 / 419 / 198 ms, PASS |
| `load_test.py` turn first-audio p50, 3 runs | 514 / 482 / 481 (median 482) | 502 / 457 / 502 (median 502) |
| `--tools --fresh` first-audio p50, runs | 454 | 493 / 457 / 486; tool called 72/72, no parse failures |
| GPU peak | 27 696 MiB | 27 706 MiB |

`qwen3-4b-2507` (the one-line rollback, Hermes JSON calls): smoke PASS, `--tools --fresh` 24/24 tools called,
first-audio p50 360 ms, GPU peak 27 666 MiB.

**Step 3 (TTS plugin `voxcpm2`, voices through `list_voices` / `load_voice`, the wire-rate wrapper):** smoke PASS
(240 / 296 / 415 / 206 ms); `load_test.py` turn first-audio p50 543 / 503 / 488 (median 503 vs 482); GPU peak
27 728 MiB; cloned-voice generate → first audio p50 86 ms (baseline 89). The streaming resampler is vendored from
vLLM-Omni (`app/resample.py`, bit-identical output): importing it from there loads torch + vLLM (~12 s), which a
hosted-only MOUTH must not.

**Step 4 (EARS plugin `nemotron-tyger-1.0`; `server.py` only on the EarsPlugin interface):**
- `validate_ears_batched.py`: identical to the baseline (all events and transcripts match, same head diffs, tick
  p50 / p90 48 / 67 and 71 / 166 ms). `validate_barge.py` (recall 0.94, first 3 words 0.83, barge 1.22 s) and
  `validate_long_turn.py` (recall 1.00 / 0.97 / 0.94) unchanged.
- `load_test_ears.py`: 1 conversation, gap p50 970 / 969 ms (baseline 970 / 966); 4 conversations 1292 / 1260 /
  1230 ms (median 1260 vs 1265), detection p50 866 / 836 / 845 (baseline 838–885). Smoke PASS.
- In one of the three 4-conversation runs the reply text varied once (2 distinct texts in 16): counting to ten is
  forced, but not absolutely under 4-way batching at temperature 0.7. The test flags it; compare runs without
  the warning, or take medians over 3.

**Step 5 (fake plugins, `config.fake.yaml`, contract tests):** `CONFIG=config.fake.yaml ./run.sh` with
`CUDA_VISIBLE_DEVICES=""` starts in ~1 s; `load_test_ears.py` through it at 4 conversations: 8/8 turns, every
endpoint from the energy fake, hop lag ~1 ms, no vLLM in the log. `test_plugin_contract.py` passes for the fakes
and, with `TYGER_CONTRACT_LOCAL=1` (GPU free), for qwen3.5-4b and qwen3-4b-2507 (both call get_weather for two
cities, unique IDs, stop within one item), voxcpm2 and nemotron-tyger-1.0 (ENDPOINT on a LibriSpeech clip).

**Step 6 (`openai` LLM, live against OpenAI, user's OK, synthetic prompts only).** Model probe through the plugin
(5 plain + 3 weather prompts each; all called get_weather 3/3): first text p50 gpt-4.1-nano 517 ms, gpt-4o-mini
587 ms, gpt-4.1-mini 852 ms (min 402). Runs below use gpt-4.1-nano as `reply` (fastest; weakest replies: it did not
say reminders aren't connected in one smoke run).

| | local Qwen3.5-4B (step 2/4) | reply openai, filler local | reply + filler openai |
|---|---|---|---|
| MOUTH first text p50 (log) | ~80 idle / 254 at 4 sessions | 597 ms | — |
| smoke plain / weather / reminder / backchannel | 240 / 305 / 419 / 198 ms | 726 / 924 / 858 / 186 ms | 1096 / 1426 / 1679 / 697 ms |
| `load_test.py` 4 sessions turn first-audio p50 | 502 / 457 / 502 | 824 / 860 / 871 | 813 / 842 |
| backchannel first-audio p50 | 474–591 | 286–355 | 838–903 |
| `--tools --fresh` first-audio p50 / max, dead air max | 486 / 547, 1431 | 951 / 3454, 1893 | 1553 / 1902, 1716 |
| `load_test.py` 1 session turn p50 | — | 776 | — |
| `load_test_ears.py` end of speech → first audio p50, 1 / 4 conv | 970 / 1260 | 1504 / 1591–1661 | — |
| barge → D | 5 ms | 5–6 ms | — |
| GPU peak | 27.7 GB | 27.4 GB | **15.3 GB** (no local LLM: ~12 GB free) |

The network round trip (~0.5–0.6 s to first text) dominates; a remote `reply` costs ~0.35 s of first audio over
local Qwen, and a remote `filler` makes backchannels ~0.55 s slower and tool lead-ins slower still. With nano's
default sampling the "count to ten" reply varies (4–10 distinct texts), so those EARS runs are measurements, not
parity comparisons. `smoke_test.py`'s EARS part now speaks a fixed sentence through `S`: it used the previous
reply's TTS audio, which with a slow first audio let the rest of a multi-sentence "user" utterance barge in
before any reply audio arrived.

**Step 8 (`openai` TTS, live against OpenAI `gpt-4o-mini-tts`, voice alloy, LLM local Qwen3.5-4B; user's OK).**
Smoke PASS, barge → D 5 ms, no request errors, GPU peak 18.4 GB (no VoxCPM2). But:

| | VoxCPM2 (local) | openai TTS |
|---|---|---|
| smoke plain / weather / reminder / backchannel first-audio | 240 / 305 / 419 / 198 ms | 1868 / 1703 / 2265 / 803 ms |
| `load_test.py` 4 sessions turn first-audio p50 / max | 457–502 / ~900 ms | 837 / 2309, 680 / 1344 ms |
| longest dead air (between chunks) max | 0 ms | 1669, 1572 ms; tool turns 5058 ms |
| `load_test_ears.py` end of speech → first audio p50, 1 / 4 conv | 970 / 1260 ms | 2173 / 1830 ms |

Each text chunk is one request, and each pays the hosted time to first audio, so short chunks finish playing
before the next one starts. The user removed the plugin (reverted); VoxCPM2 is the only TTS.

**Step 10 (`smart` role, `ask_expert`, `tool_start` names the tool, fillers that fit the tool).** GPU-free: 12 new
session tests on a fake `smart` (answer straight to TTS, request starts under the lead-in, expert still-lines,
smart tool rounds and the `max_steps` cap, other tools' results as context, plain-text context turns, barge-in,
smart error / refusal) and `tool_start` naming tests; the local contract run checks both Qwen plugins name
`get_weather` at `tool_start`. Local-only parity (no `smart`: prompt and tools byte-identical):

| | step 2–4 | step 10 |
|---|---|---|
| smoke plain / weather / reminder / backchannel | 240 / 305 / 419 / 198 ms | 233 / 335 / 449 / 205 ms, PASS |
| `load_test.py` turn first-audio p50, 3 runs | 502 / 457 / 502 (median 502) | 537 / 476 / 464 (median 476) |
| backchannel first-audio p50 | 474–591 | 568 / 651 / 622, then 402 / 618 (same code; noise) |
| `--tools --fresh` first-audio p50, 3 runs; dead air p50 | 493 / 457 / 486 (median 486); 1011 | 569 / 533 / 517 (median 533); 1014–1090 |
| GPU peak | 27 706 MiB | 27 702–27 776 MiB |

Tool turns pay ~30–50 ms for the name: the lead-in now starts a few tokens after `<tool_call>` instead of at the
tag (no tool-call parse failures in 79 calls).

Live check with Qwen3.5-4B as `reply` and a **fake** `smart` (no remote calls; 10 synthetic prompts, one fresh
session each): 5/5 hard prompts (EN / RU / ES: savings maths, loan vs invest, a Kyoto plan, mortgage terms,
index vs active funds) escalated, 4/4 easy ones answered locally, the weather question used `get_weather`. The
lead-ins fit ("That's a tough choice, let me weigh the options.", "Давайте я сравню эти варианты…"). Escalated
turns' first audio (the lead-in) 412–461 ms alone; 4 at once 653–716 ms vs 580–808 for 4 weather turns at once
and 396–454 for plain ones. Barge-in during an escalated turn → D 6 ms. Routing on a real test set is step 11.

**Step 11 (routing eval; gate NOT met yet).** `scripts/eval_escalation.py` sends each case of
`scripts/escalation_set.yaml` exactly as MOUTH would with a `smart` configured (reply model only, no remote calls)
at temperature 0.7, N samples each. 86 synthetic cases EN / RU / ES: 28 local, 18 tool, 30 expert, 10 multi-turn
(follow-ups on an escalated topic, "check that again", thanks / topic change after an escalation); every third
case of each route is held out. Missed = expert → local (target ≤ 10%), unneeded = local → expert (≤ 15%).

| reply model, prompt | tune missed / unneeded | held-out missed / unneeded |
|---|---|---|
| qwen3.5-4b, step 10 prompt (×5 tune, ×8 held out) | 8.3% / 0% | **16.7%** / 1.2% |
| qwen3.5-4b, v2: follow-up reminder in the ask_expert tool result (×8) | 5.2% / 0.6% | **15.6%** / 0% |
| qwen3-4b-2507, v2 (×8) | 19.3% / 0% | 11.5% / 0% |

Tried and dropped on the tune split: a stronger follow-up rule in the system prompt plus RU / ES follow-up shots
(no better: 9.2%), and a "handing over costs little" nudge (worse: 7.8%). Differences of 1–3 points are a few
samples. Lookups are never confused with escalation (tool ↔ expert ~0%), and no call failed to parse.

The misses are systematic, not sampling noise. Qwen3.5-4B keeps RU / ES follow-ups on an escalated topic local (m04
7/8, m05 8/8) and some RU / ES health / legal questions (e30 5/8). Qwen3-4B-2507 handles follow-ups but answers
reasoning and code itself, every time (bat-and-ball, palindrome, sort without sort, TCP handshake, "recount
carefully": 8/8). Both lean local: unneeded escalations stay at ~0%, so there's room to push toward escalating.
The held-out split has now been looked at; further tuning needs fresh held-out cases.

**Gemma 4 E4B as `reply` (user's suggestion; plugin `gemma-4-e4b`, `google/gemma-4-E4B-it`, text only, thinking
off, same eval settings and prompt).** Routing: **0% missed, 0% unneeded** on all 86 cases × 8 (tune and held
out alike); the only misroute is "send a message to Alex" answered locally 3/8. Same with FP8 weights
(`quantization: fp8`, now passed through to vLLM).

| (GPU free, 3 GiB KV) | Qwen3.5-4B | Gemma E4B bf16 | Gemma E4B FP8 |
|---|---|---|---|
| weights / GPU used after load | — / 12.3 GiB | 14.2 / 18.5 GiB | 10.5 / 14.9 GiB |
| first text p50 (5–11 plain prompts, alone) | 76 ms | 80 ms | 72 ms |
| tool_start (weather / ask_expert) | 110–123 ms | 118–122 ms | 100 ms |

bf16 doesn't fit next to VoxCPM2 + EARS on the 5090 (~34 GB); FP8 would be ~30 GB of 32 (not tried in the stack
yet). Reply quality issues the routing eval doesn't catch: Gemma answered 2 of 5 Russian questions in Spanish /
English (Qwen: 0), and once spoke its tool use aloud ("I'll use the ask_expert tool for this because…") before
the call. Facts looked better (Volga vs Qwen's Danube for Europe's longest river).

**Responder switched to Gemma 4 E4B FP8 (user's call: "robust tool calling is crucial").** The eval grew first:
30 `fresh` cases (12 tool, incl. a lookup follow-up after a weather turn; never tuned on), and scoring of what
would be spoken: wrong language (app/lang.py guess), narration of tools / markup, and the answer after a stub tool
result (phase B). `--no-expert` measures the production prompt (no `smart` configured). What it found and fixed:

1. **Bare calls.** Gemma sometimes writes a call without its special tokens: `set_reminder{text:<|"|>…}` as
   plain text. Our parser missed it (counted "local") and MOUTH would have spoken the markup. Fix in the plugin:
   for the offered tools, `NAME{` / `call:NAME{` count as a call (`_toolparse.gemma_normalize`), and a trailing
   word that could become one is held back while streaming. Prompt wording that names the tools ("call
   set_reminder…") made bare calls far more frequent; the prompts don't name them.
2. **Answer language after a tool result.** Tool output is English (Tavily, the stub notes): 72 / 248 (29%)
   Russian / Spanish answers came out in English. A generic language rule in the system prompt changed nothing;
   a generic note on the result 72 → 62. Naming the language works: MOUTH appends "[Answer the user in Russian,
   even if this result is in another language.]" (`prompts.yaml` `language`, guessed from the transcript) to each
   tool result it sends for the answer, and the same hint to lead-in / still-looking requests (one lead-in had
   come out in English for a Spanish request). Plus lookup few-shot turns (below): 0 / 303 wrong.
3. **Few-shot turns for both modes** (`reply_shots`): local answers mixed with lookups (RU weather, ES time, EN
   reminder, RU message; English results, answers in the user's language). Lookups only made it web-search simple
   facts (17% of local cases); mixing in local answers fixed that. The old RU shot "how do you say thanks in
   Spanish" taught "спасибо → Spanish"; replaced. Without the shots, Gemma asked clarifying questions instead of
   calling (who is Maria? what time zone?) and declined forecasts (`get_weather` said "current weather" only:
   now "current weather or the forecast").
4. Prompt rules (both modes): never talk about tools / the expert, call reminder / message tools with the details
   given, reply in the language of the user's latest message.

Final, Gemma FP8, 8 samples per case (116 cases with a smart model, 70 without; tune / held-out / fresh alike):

| | no smart (production) | with smart |
|---|---|---|
| lookups and actions called (tool → tool) | 247 / 248 | 248 / 248 |
| local answered locally | 312 / 312 | 312 / 312 (1 sample escalated) |
| missed escalations | — | 1.4% (target ≤ 10%) |
| speech in the wrong language / narration | 2 / 1 of 313 | 8 / 0 of 317 |
| answer after a tool result in the wrong language | 0 / 247 | 2 / 248 |

Left: "Спасибо, ты очень помог" is sometimes answered "De nada" (Gemma quirk, in every variant); Spanish opinion
questions now and then answered in English (escalation mode).

Full stack (config.yaml: `gemma-4-e4b`, `quantization: fp8`, `gpu_memory_utilization: 0.5`, KV 3 GiB):

| | Qwen3.5-4B (step 10) | Gemma 4 E4B FP8 (final prompts) |
|---|---|---|
| smoke plain / weather / reminder / backchannel | 233 / 335 / 449 / 205 ms | 225 / 271 / 325 / 190 ms, PASS |
| `load_test.py` turn first-audio p50, 3 runs | 537 / 476 / 464 (median 476) | 460 / 438 / 426 (median 438) |
| backchannel first-audio p50 | 402–651 | 383–495 |
| `--tools --fresh` first-audio p50 (median); dead air p50 | 533; 1014–1090 | 439 / 443 / 432 (439); 785–809 |
| `load_test_ears.py` end of speech → first audio p50, 1 / 4 conv | 970 / 1260 | 940 / 1183, 1203 |
| GPU peak | 27.7 GB | 30.0 GB of 32.6 (idle 30.3 before the KV warm-up settles) |

Plain replies got shorter (load_test wall 25 s → 12 s for the same turns): the shots model short spoken answers.
No tool-call parse failures (86 + 94 calls in the stack logs), no errors. Live check (11 synthetic prompts, EN /
RU / ES): every action and lookup called, every lead-in and answer in the user's language, except the "De nada".
Rollback: `plugin: qwen3.5-4b`, drop `quantization`, `gpu_memory_utilization: 0.45`.

**`smart` = OpenAI gpt-5.4-mini (user's call; live probe with the user's OK, synthetic prompts only).** The exact
escalated-turn request (`smart_messages`, lookup tools) on 8 hard prompts EN / RU / ES, one needing the weather:

| | gpt-5.5 | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|
| first text p50 (max) | 981 ms (6.8 s) | 507 ms (3.7 s) | 510 ms (3.7 s) |
| total p50 | 2.7 s | 1.1 s | 1.2 s |
| right language / markdown / used the weather tool | 8/8, 0/8, yes | 8/8, 0/8, yes | 8/8, 0/8, yes |
| words p50 | 79 | 68 | 66 |

On Chat Completions, gpt-5.x takes function tools only with `reasoning_effort: none` (400 otherwise: "use
/v1/responses"), so all ran without thinking; thinking + tools would need a Responses API plugin. The plugin now
sends `max_completion_tokens` for gpt-5* / o-series (`token_param` overrides). gpt-5.4-mini: as fast as gpt-4.1,
newer, says numbers as words.

Live, Gemma `reply` + gpt-5.4-mini `smart` (5 synthetic hard prompts through MOUTH): lead-in first audio 363–374
ms; expert answer first text 0.86–1.14 s and first audio 1.03–1.33 s after the turn started, i.e. while the
lead-in (~2 s) is still playing: no dead air. Smart first event 530–710 ms after its request. Barge-in during the
answer → D 6 ms. The bike-in-Berlin question went to `get_weather` locally (answer 3.7 s, a still line covering
Tavily).

`tool_start` waits for the tool's name only when ask_expert is offered (`_toolparse.names_tool`), as asked.
Tool-turn first audio p50: no smart 395 / 367 / 380 (was 439 with the name wait); with smart on 469 / 462 (the
longer escalation prompt + the name). Plain turns with smart on 468 / 513 / 467 (438 without): the hand-off shots
add prefill. Smoke PASS both ways; GPU peak 30.4 GB.

**Step 12 (background tasks: `app/tasks.py`, start_task / cancel_task, status note).** GPU-free gate:
`tests/test_tasks.py` (13 tests on a fake `smart` with scripted tool rounds and delays):
- start → keep chatting → done → the next reply request carries the status note with the summary;
- "still working" while running;
- the task's own tool rounds;
- the caps: concurrent, per conversation, global across conversations (the slot comes back when a task ends);
- the wall-clock timeout (the smart stream is aborted);
- cancel by the user, and the conversation closing (both abort the stream and free the slot);
- `max_steps` then the limit note;
- a plain-text answer taken as the result;
- an LLM error failing the task;
- no goal, summary or result in any log line.

Live with tasks off (the default until step 13): smoke PASS, turn p50 477 / 470 ms (unchanged). Also fixed: a
connection reset from EARS ended `Session.run` with a traceback instead of a clean close.

**Step 13 (floor request `F` / grant `G`, the announcement, `task_done` to the browser).** GPU-free:
- `tests/test_ears_floor.py` (6): `server.py`'s state machine on the fake EARS plugin, a fake browser WebSocket
  and a fake MOUTH over TCP. Covers: the grant only after `announce_after_silence_ms`; a user turn winning (T
  before G); silence not counted while the agent speaks; a backchannel holding the grant; barge-in during the
  announcement; two requests taking turns. Two deliberate bugs were checked to make tests fail: the grant with no
  silence, and silence counted during the agent's speech. The second needed a feed that reports `idle` frames
  while the agent speaks: the fake and the real plugin report `SPEAK` then.
- `tests/test_tasks.py` (+5, 18 in all): F → G → summary spoken, D `proactive`, history note + summary,
  "told to the user" in the status note; barge → announced partially; a failed task → one sentence from the
  filler role in the user's language; G for a cancelled / unknown task → empty D; an escalation after a task gets
  its full result.

Live, real EARS + Gemma + VoxCPM2, **fake** `smart` (finishes in 4.6 s), synthetic speech: the request →
`start_task` → acknowledgement; the task finished while the acknowledgement was still playing; EARS granted the
floor after it played out + 1.28 s of silence; the announcement's first audio came 0.1 s after the grant. A
request worded like the English task shot ("the three best budget laptops for programming") was answered "I'm
still working on that" without a call: the 4B took it as a follow-up to the example task. A request on a topic
the shots don't cover worked. Step 14's eval will measure that kind of routing.

**Live tasks with OpenAI gpt-5.4-mini (user's OK; synthetic spoken requests, EN + RU):**

| | EN: e-bikes under $2000 by range | RU: reliable e-scooters under 30 000 ₽ |
|---|---|---|
| request → `start_task` → acknowledgement | yes ("I'm starting that research now…") | yes, in Russian |
| task: web searches, time | 4, 23.2 s | 2, 16.2 s (then 2, 15.5 s) |
| floor granted after silence | 15.7 s (already quiet) | 6.1 s (5.8 s) |
| announcement first audio after the grant | 0.1 s | 0.1 s |
| announcement | named the Lectric XPedition 2.0 + alternatives; ~15 s of audio | 1st run: no product named, "I'll tell you briefly", digits; 2nd run after the prompt fix: three models named + a recommendation, numbers as words; ~25 s |

The task prompt now asks for the actual findings (names, numbers, the recommendation), not what was done, with
numbers as words. Announcements run long (15–25 s); a shorter summary is a prompt change if wanted.

**Tasks on by default** (user: after the live check). Cost on ordinary turns, 4 sessions: plain first audio
p50 533 / 516 / 526 (468 with smart only, 438 without smart), tool turns 508 / 485 (465 / 380): the task prompt +
shots add prefill. Smoke PASS, GPU peak 30.4 GB. With smart on, `load_test.py`'s plain prompts escalate now and
then ("explain what a black hole is": 7–10 of 72 turns), i.e. the load test makes a few OpenAI calls; for pure
local parity runs use a config without `llm.smart`.

**Step 14 (background cases in the routing eval; README section).** `eval_escalation.py` has a fourth route,
`background` (`start_task`; `cancel_task` counts as a tool). Its mode follows config.yaml (tasks on), with
`--no-tasks` / `--no-expert`. Cases may carry `also` (planning: either expert or background is right) and a
`status` note, built as MOUTH builds it (`mouth.with_status`). New: 18 single-turn background requests (EN / RU / ES:
research and get back, compare many options, multi-day plans, long drafts) and 6 multi-turn ones: status questions
with a running or finished task, a cancel, an unrelated question meanwhile, a second research request. 140 cases
in all. Gemma 4 E4B FP8, 8 samples:

| run | right (incl. `also`) | missed (exp / bg → local) | unneeded | bg → expert (not allowed) | after-tool wrong language |
|---|---|---|---|---|---|
| first | 98.7% | 1.3% | 0% | 8 | 3 / 417 |
| + language named after the status note | 98.3% | 1.5% | 0% | 8 | 0 / 419 |
| + status / cancel shots, cancel_task description | 98.2% | 0.6% | 0% | 17 | 3 / 407 |
| + language from earlier turns for short utterances (final) | 98.3% | 0.4% | 0% | 17 | 0 / 407 |

Held-out 96.1%, fresh 100%. Found and fixed on the way:
- **The status note pulled replies into English.** A Russian "how's it going?" was answered in English (the note is
  English). The user's language is now named right after it.
- **A status question cancelled the task.** "Ну как там с тарифами?" called `cancel_task` 2 / 16 times, made up a
  `get_task_status` tool once, and escalated once. Fixed with task shots for a status question answered locally
  (EN, RU) and an explicit "drop it" → cancel_task (ES), plus a cancel_task description that rules out progress
  questions: now 16 / 16 local.
- **Short utterances lost the language.** "Actually, never mind, drop that one" was too short to guess, and the
  cancel was acknowledged in Russian. MOUTH now guesses from the user's last few turns too when the utterance can't
  tell (`app/lang.language_text`), for every language hint.

Left:
- "Research … and tell me later" in RU / ES sometimes goes to the expert, answered now, not as a task (b09 6/8,
  b10 8/8). Both reach the smart model, but the answer is quicker and shallower. The status / cancel shots made
  this more common (8 → 17 samples), a trade for the safe status answers.
- The "De nada" quirk; Spanish opinion questions now and then answered in English.
- The narration flags on b04 are false positives ("those tools" = project-management tools).

The eval stubs every tool, so it needs no Tavily or OpenAI key. Tavily is disabled for now (user, 2026-10-04): live
lookups and task searches return the error result until it is back.

## Box (2026-10-03) — new, healthy RTX 5090
NVIDIA RTX 5090 (32 607 MiB), driver 590.48.01 / CUDA 13.1; 384-core CPU, 755 GB RAM. Same venv versions as below
(torch 2.13.0+cu130, vllm 0.30.0, vllm-omni 0.30.0, transformers 5.14.1). **Not throttled**: under load it sits at
**P1, SM 2940 MHz, memory 13 801 MHz** (max 3090 / 14 001). The old box ran memory at 7001. Every number from
here on is from this box unless it says "throttled box".

Fresh-box install (`README` step 1) failed on a clean checkout: `vllm-omni==0.30.0` pins `accelerate==1.12.0`
but `requirements.txt` asked for 1.15.0, so uv found no solution. Pinned 1.12.0. The validation scripts also
need `datasets` (pulled in by vllm already; nothing to add). First-ever `./run.sh` with all downloads: ~12.5 min
(MOUTH ready after ~9.5 min, EARS after another ~3 min, mostly nemotron + head download).

### Baseline (code at `b64c772`, cloned voice `en` 9.3 s, no transcript)
| test | metric | healthy box | throttled box (V1.4) |
|---|---|---|---|
| `smoke_test.py` | plain / weather tool / reminder first-audio | **354 / 441 / 473 ms**, PASS | 503 / 743 / 770 ms |
| | backchannel first-audio | 363 ms | — |
| `load_test.py --sessions 4` | turn first-audio p50 / max | **539 / 732 ms** | 833 / 1126 ms |
| | backchannel p50 / max | 643 / 1011 ms | 829 / 1620 ms |
| | GPU peak | 28 356 MiB | 27.0 GB |
| `load_test.py --tools` (new, every turn a Tavily weather call) | 1 session: first-audio p50, dead air max | 441 ms, 348 ms | — |
| | 4 sessions: first-audio p50 / max, dead air max | 658 / 897 ms, **2224 ms** | — |
| | GPU peak | 29 268 MiB | — |
| `load_test_ears.py` (old script, whole TTS reply as speech) | 1 conv: hop lag p50/p90/max, endpoint→audio | 61 / 78 / 189, 266 ms | 59 / 114 / 202, 529 ms |
| | 4 conv | **97 / 238 / 455**, 399 ms | 122 / 353 / 716, 810 ms |
| `validate_ears_batched.py` | events match / max head diff | all True / 0.0067 | all True / ~0.007 |
| | tick ms 1 session, 4 sessions (p50 / p90) | 44 / 60, 80 / 138 | — |
| `validate_rnnt.py` | identical; graphed 8 s ×4, 20 s ×1 | 40/40 + 40/40; 53, 67 ms | 80/80; 74, 88 ms |

EARS stats under the 4-conversation load: encoder 12%, head 23%, ASR 14% busy (throttled box: ~40% / ~30%).
GPU peak with tool turns is 29.3 GB (28.6 GiB): the "28.5 GB" budget in README is about right, with ~3 GB free.

### Response gap (I-04): end of speech → endpoint → first reply audio
`load_test_ears.py` now times the end of each injected utterance (last voiced 20 ms frame, on the simulated mic
clock) and reports detection delay and the full gap next to the old endpoint→audio onset. It also injects only the
**first sentence** of the TTS speech (cut at the first ≥ 300 ms pause). The whole multi-sentence reply made the
endpoint fire at an inner pause in 5 of 8 turns at 4 conversations: the rest of the utterance then played over the
reply, and when that didn't barge, the conversation stalled with the words lost (3 of 4 conversations finished
only 1 of 2 turns). `--full-speech` restores the old input. Utterance ~1.9 s voiced; code at `b64c772`:

| conversations | end of speech → `endpoint` p50 / p90 | endpoint → first audio p50 / p90 | **end of speech → first audio** p50 / p90 / max | hop lag p50 / p90 |
|---|---|---|---|---|
| 1 (4 turns) | 779 / 784 ms | 229 / 242 ms | **1013 / 1019 / 1021 ms** | 56 / 73 ms |
| 4 (3 turns each) | 773 / 877 ms | 257 / 299 ms | **1022 / 1132 / 1192 ms** | 102 / 140 ms |

All endpoints by the head (none by the silence fallback, none inside the utterance). **Detection is ~77% of the
gap**; first audio arrives at the client ~1.0 s after the user stops (add the client's 40 ms lead-in + playback
buffering for what they hear). A rough split of the ~775 ms detection (estimated from the parts, not
instrumented): `ep_deb` 2 frames = 160 ms, hop quantisation ~160 ms average (L-02), hop lag ~60–100 ms, final ASR
~40–70 ms; the remaining ~350–400 ms is `p_end` climbing to 0.9146 after the speech stops.

**Targets for live tuning (O-01).** Drive *end of speech → endpoint* from ~775 ms p50 to **≤ 450 ms p50 (p90
≤ 550 ms)**, which puts end of speech → first audio at ~700 ms. With today's hop and debounce the floor is ~380 ms
(160 debounce + ~160 hop quantisation + ~60 lag), so the threshold part has ~70–150 ms to work with: in practice a
lower `ep_thr` (the offline 0.9146 was seen ~2× too high on a real mic). `ep_deb` 1 saves another 80 ms and L-02
(160 ms hops) ~80 ms. Guard rail: the endpoint must not fire inside an utterance more often. Track
`load_test_ears.py --full-speech` "fired inside the utterance" (5/8 at 4 conversations today, with the
multi-sentence test speech) and, live, how often the agent cuts in mid-sentence.

### MOUTH barge-in and backchannel fixes (I-01, I-03, I-07)
- **Barged turns stay in history (I-01).** On a barge MOUTH remembers the user's transcript and the reply up to the
  chunk that was playing, ending in " —" (the system prompt says a reply ending in a dash was cut off). EARS sends
  `B {"played_s"}`: reply audio sent minus what is still queued at the client (`play_until`), so generated-ahead
  audio the user never heard isn't remembered. Before, on the real stack: barge "tell me about octopuses", then
  "which animal were you telling me about?" got "I wasn't telling you about any animal"; now "Octopuses…".
  `smoke_test.py` checks this.
- **A barge doesn't wait for tools (I-03).** Barge → `D`: **5 ms** on a tool turn, against **2.0–2.5 s** before
  (live Tavily). Tool threads are daemons and finish on their own.
- **A reply turn cancels the backchannel in flight (I-07).** `T` bumps a per-session turn counter; a backchannel
  queued before it stops (LLM and TTS requests aborted) and sends `D {"bc", "aborted": true}`.
- `load_test.py --sessions 4` (6 turns, 24 per run), same protocol before and after:

  | | plain turns, first-audio p50 (3 runs) | backchannel 50 ms before each turn (`--bc-first`): first-audio p50 / max |
  |---|---|---|
  | before (`b64c772`) | 535 / 561 / 554 ms | 769 / 1312 ms (backchannel ran to the end) |
  | after | 596 / 563 / 495 ms; with L-04 too: 496 / 612 / 573 ms | 500 / 707 ms (24/24 aborted) |

  Plain turns are unchanged within run-to-run noise; a turn that starts while a backchannel is being made is
  ~200 ms faster.
- GPU-free tests: `tests/fakes.py` (fake LLM/TTS/Mouth) drives `Session.stream_turn` and `Session.run` over a
  socketpair; `tests/test_mouth_session.py` covers I-01/I-03/I-07 and fails on the old code (I-03 case: `D` came
  9982 ms after the barge with a 10 s tool).

### Barge-in keeps the start of the user's utterance (I-02)
While the agent speaks, user audio never entered `turn_u`, and both the barge and MOUTH's barged `D` reset it, so the
endpoint transcript lost the words that triggered the barge. Now EARS keeps the last `BARGE_PRE_S` (2 s) of user
audio in speaking mode; on `BARGE` the next reset seeds the new turn with it (and credits its voiced frames toward
the 350 ms minimum), and `finish_turn` doesn't reset again after a barge. `scripts/validate_barge.py` (GPU free;
12 public LibriSpeech barge-ins over agent speech; numbers only): first 3 words recalled **14% → 83%**, all words
**69% → 94%**; the barge fires a median 1.22 s after the user starts talking.

### Cloned-voice onset (I-06 / L-03): resample once, VoxCPM2's speaker cache
`scripts/bench_tts_voice.py` (TTS only, GPU free; clip `en` 9.3 s @ 44.1 kHz; p50 of 12 short sentences, time to the
first audio output of one VoxCPM2 request):

| how the clip is passed | prompt build | generate → first audio |
|---|---|---|
| 44.1 kHz list, every request (old) | 2.0 ms | **146 ms** |
| 16 kHz list (the VAE's rate), every request | 0.4 ms | 97 ms |
| 16 kHz + speaker cache (current) | 0.5 ms | **78 ms** |
| zero-shot, no clip | 0.5 ms | 81 ms |

The I-06 hypothesis (the tuple of ~480k Python floats) was wrong: building the prompt took 2 ms. The cost was the
stage worker **resampling the clip on the CPU for every request** (~50 ms) and **re-encoding it** (~20 ms). Now
`load_voice` resamples once, and reference-only voices (no transcript) use the worker's in-memory speaker cache:
the first request carries the clip + a `voice_name` (`tyger-<name>-<mtime>`), later ones only the encoded length.
MOUTH primes every clip in `voices/` at startup, so no conversation pays the first-use cost. Nothing derived from a
clip is written to disk (vLLM-Omni's file-based `custom_voice_dir` profiles would do the same from disk; not
needed). A clip passed as a tensor arrives garbled in the stage worker and kills it: keep lists.

Full stack after this (and everything above), same box:

| | baseline (`b64c772`) | now |
|---|---|---|
| smoke plain / weather tool / reminder first-audio | 354 / 441 / 473 ms | **263 / 264 / 301 ms** |
| `load_test.py --sessions 4`: turn first-audio p50 (3 runs × 24) | 535 / 561 / 554 ms | **425 / 410 / 400 ms** |
| backchannel first-audio p50 | 643 ms | 451–490 ms |
| GPU peak | 28.4 GB | 27.0 GB |
| `load_test_ears.py`: end of speech → first audio p50, 1 / 4 conv | 1013 / 1022 ms | 954 / 992–1109 ms |

The EARS numbers vary with the test's speech: it is the LLM's reply to a "Say: …" prompt, so the utterance and
the reply lengths change run to run (here 3–4× more hops in speaking mode than in the I-04 run: hop lag p90
223–256 ms, encoder ~33% busy). A fixed test utterance would make runs comparable.

### EARS encoder cost (L-01): features on the GPU; bf16 rejected
- **bf16 encoder: rejected.** Autocast to bf16 around the L15 encode (thread-local, so the ASR stayed fp32) gave
  no speedup (encoder busy 12.3 s vs 11.5 s fp32 over `validate_ears_batched.py`'s 4-session run), head-output
  diffs 4–20× larger (p_end up to 0.14 vs ~0.007), and one session's events changed.
- **Why: the encoder is launch-bound and the CPU features weren't.** To L15, 8 s windows ×1 / ×4 / ×8: GPU encoder
  14 / 15 / 21 ms, while the processor's CPU feature extraction took 4 / 14 / 28 ms. At 4 conversations a tick
  encodes 8 windows (user + agent channel each), so the CPU STFT cost more than the encoder.
- **Fix: `EarsEngine.features`** runs the processor's own extraction (pad, preemphasis, STFT, mel, mask) with the
  waveform on the GPU: 8 windows 19 → 1.9 ms, values within ~2e-4 of log-mels ~16, identical masks. Used for both
  the L15 encode and the ASR. (The extractor's `device="cuda"` leaves the waveform on the CPU and fails.)
  `validate_ears_batched.py`: all events match (endpoint transcripts included), head diffs ≤ 0.0077; 4-session
  tick p50 90 → 59 ms, encoder busy 11.5 → 8.4 s. Full stack, 4 conversations, 3 runs: encoder busy 21–23% (was
  32–35% with similar reply lengths), hop lag p90 192–216 ms (was 223–256), detection p50 550–778 ms.
- Not done: precomputing the agent channel's features while speaking (L-01 option 2), the streaming encoder
  (option 3, needs head work upstream). The head (~20–24% busy) is now as large as the encoder.

### Long user turns (I-05): transcribed in segments
A listening turn used to keep only its last 20 s for the endpoint transcript, silently. Now, past 15 s (`SEG_S`),
EARS cuts it at the quietest 80 ms frame of the last 4 s and transcribes that block on the ASR thread at once;
the endpoint transcribes the tail (≤ ~15 s) and joins. `scripts/validate_long_turn.py` (GPU free):

| turn | before: word recall (first 10 words) | after: recall (first 10), segments | endpoint hop before → after |
|---|---|---|---|
| 27 s | 0.85 (0.60) | **1.00 (1.00)**, 2 | 133 → 150 ms |
| 33 s | 0.67 (0.40) | **0.97 (1.00)**, 2 | 136 → 127 ms |
| 51 s | 0.42 (0.20) | **0.94 (1.00)**, 3 | 162 → 156 ms |

### Tool calls stay in history (I-12, found on the box)
History used to keep only the final reply text of a tool turn. The model then answered later live questions from
its own history, without calling a tool: 6 weather questions per session called the tool 2/6 times (1 session)
and 9/24 (4 sessions). History is now a list of turns trimmed whole, and a tool turn keeps its call plus a compact
result (`_compact_result`: no raw search hits, ≤ 600 chars): 6/6 and 24/24.

### Tool fillers (L-04): start at the `<tool_call>` tag, cover slow tools
- `LocalLLM.stream` yields `tool_start` the moment the tag appears. MOUTH then makes the filler (its own LLM + TTS
  requests, on a helper thread, relayed as reply audio) while a second helper thread finishes decoding the call and
  starts the tools the moment it parses. If the tools are still out `FILLER_GAP_S` (0.8 s) after the filler has
  played (MOUTH tracks the client's playback end like EARS does), one more "still looking" line that is told what
  was already said, up to `MAX_STILL` (2). All of it shares the turn's `stop_flag`.
- If the model spoke a full chunk before the tag ("Sure, let me look up…"), that is the lead-in: no filler.
- **Fillers are contextual via few-shot, not quoted examples.** The old `LEADIN_SYS` quoted 'Sure, let me check
  that for you.' and the 4B said exactly that on every tool turn. Now: no example inside the system prompt,
  few-shot turns in EN/RU/ES that name the topic (`LEADIN_SHOTS`, `STILL_SHOTS`). Sampled with a 3 s tool:
  "Checking today's temperature in Cairo." → "Cairo's temperature is still coming in."; "Сейчас смотрю погоду в
  Новосибирске." → "Данные из Новосибирска чуть дольше."
- `TYGER_TOOL_DELAY_S` (env, testing only) sleeps before every tool call; with it the call still hits Tavily, so
  "3 s stub" below is 3 s + Tavily.
- Measured with `load_test.py --tools --fresh` (every turn a weather tool turn, empty history; 30/30 turns
  called the tool) and the smoke weather turn:

  | | first audio p50 / max | longest dead air p50 / max |
  |---|---|---|
  | smoke weather turn | 441 → **285–315 ms** | — |
  | live Tavily, 1 session | 375 / 399 → **279 / 305 ms** | 257 / 2389 → 468 / **838 ms** |
  | live Tavily, 4 sessions | 640 / 882 → **408 / 740 ms** | 548 / 2584 → 342 / **1265 ms** |
  | 3 s stub, 1 session | 424 / 555 → **278 / 324 ms** | 3104 / 3489 → **1023 / 1232 ms** |
  | 3 s stub, 4 sessions | 583 / 752 → **384 / 658 ms** | 3312 / 5031 → **1067 / 1314 ms** |

  The worst silence is now bounded at ~1.0–1.3 s (the 0.8 s gap + making the next line). The median dead air
  with a fast tool can be a little longer (1 session: 257 → 468 ms): the filler now ends sooner, before Tavily
  returns. A barge during the filler: `D` 5 ms; the unit test asserts ≤ 1 audio chunk after the stop and the
  filler's TTS request aborted.

## Box (2026-10-02) — old, throttled
NVIDIA RTX 5090 (32 607 MiB, Blackwell sm_120), driver 595.91 / CUDA 13.2; 96-core CPU. `HF_HOME=/workspace/.hf_home`,
`HF_HUB_DISABLE_XET=1`, torch **2.13.0+cu130**, Python 3.12, one venv `.venv` (vllm 0.30.0, vllm-omni 0.30.0,
voxcpm 2.0.3, transformers 5.14.1).

**This box's GPU is throttled**: it never leaves P3 — memory clock 7001 of 14001 MHz, SM 2385 of 3412 MHz,
~200 W under load, no throttle reason reported (host-level lock). Qwen3-4B decodes at **92 tok/s** here vs
~150 on the previous box with the same vLLM. **Every latency number below from this box is pessimistic**;
re-measure on a healthy GPU before tuning anything against them.

## Responder: Qwen3.5-4B (2026-10-03, user's call)
`llm.model` is now `Qwen/Qwen3.5-4B` (was Qwen3-4B-Instruct-2507; rollback = one line in `config.yaml`). What changed:
- **Thinking off.** Qwen3.5's template opens `<think>` by default; `llm.enable_thinking: false` passes
  `enable_thinking=False` to the template (an empty think block), so replies start with speech.
- **Text only.** The checkpoint is multimodal; `llm.language_model_only: true` skips the vision tower: 8.0 GiB of
  weights loaded.
- **XML tool calls.** It emits `<tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>`
  instead of Hermes JSON; `tools.parse_calls` reads both. The `<tool_call>` tag the fillers start on is unchanged,
  and the template renders our history's tool calls in its own format (checked).
- **Hybrid attention** (24 linear + 8 full layers): the same pinned 3 GiB of KV holds 69k tokens instead of ~21k.
  vLLM runs its mamba cache in 'align' mode for prefix caching.
- Measured, same box, same code otherwise:

  | | Qwen3-4B-Instruct-2507 | Qwen3.5-4B |
  |---|---|---|
  | idle turn: first text / first audio | ~50 / 263–272 ms | ~78 / 233–277 ms |
  | `load_test.py --sessions 4` turn first-audio p50 (3 runs × 24) | 400–425 ms | 446–550 ms |
  | 4 sessions, MOUTH first text p50 (log) | 50 ms | 254 ms |
  | tool turns `--tools --fresh` / `--tools`: tool called | 24/24, 24/24 | 24/24, 24/24 |
  | tool-turn first-audio p50 (4 sessions) | 408 ms | 456–460 ms |
  | GPU peak (4 sessions) | 27.0 GB | 27.5 GB |

  Under concurrent load its prefill is slower (first text 50 → 254 ms p50), costing ~50–100 ms of turn
  first-audio at 4 sessions; idle it's level. Reply quality is the user's call (listen live).

## Architecture (V1.3) — two processes, one venv, both MOUTH models in-process on vLLM
- **MOUTH** (`app/mouth.py`): orchestrator + an `EngineLoop` (`app/engines.py`: one asyncio loop in a thread,
  bridged to MOUTH's synchronous generators; aborts the engine request on barge) hosting
  - **Qwen3-4B-Instruct-2507** on a vLLM **AsyncLLM** (`app/llm.py`): chat template with tools, Hermes
    `<tool_call>` text parsed in-process (text from the first `<tool_call>` on is never spoken);
  - **VoxCPM2** on a vLLM-Omni **AsyncOmni** (`app/tts_voxcpm.py`): one request per text chunk, audio per decode
    step, 48 kHz → 24 kHz via vLLM-Omni's stateful `StreamingAudioResampler`.
- **EARS** (`app/server.py`): nemotron L15 + the exp-10 chain + endpoint head (arm C). Serves the web app.
- Wired by the exp-12 TCP protocol (`app/wire.py`, frames `T/C/B/V → X/A/D`). `run.sh` starts MOUTH, waits
  for its port, then EARS.

**Why two engines, not one vLLM-Omni pipeline.** vLLM-Omni stages are a chain (stage 0's output feeds stage 1,
like Qwen-Omni thinker→talker). Chaining Qwen → VoxCPM2 inside the engine would remove MOUTH from between them:
no text chunking policy, tool calls + lead-in, contextual backchannels, or barge-in control.

**Why one venv now.** The old split existed for MOSS-TTS-Realtime (transformers 5.0.0) and vLLM's torch pin.
MOSS is gone; vllm 0.30 brings transformers 5.14.1, which satisfies nemotron's ≥ 5.13. EARS was A/B-tested in
`.venv` vs its old pinned venv (torch 2.9.1 / tf 5.17): identical per-hop latency (p50 275 / p90 404 ms).
`voxcpm` must be listed explicitly: the vLLM-Omni stage worker imports it but vllm-omni doesn't depend on it.

**Why VoxCPM2 (replacing MOSS-TTS-Realtime).** User's call. Measured in a one-process spike on this (throttled)
box, zero-shot voice: TTS **first-audio p50 113 ms**, **RTF 0.12–0.20** (MOSS: ~480 ms onset, RTF 0.56–0.7);
abort takes 2 ms and the next request runs at full speed. Apache-2.0, 30+ languages. Earlier note "vLLM-Omni
was slower than in-process" was MOSS-Nano behind vLLM-Omni's **HTTP** server; in-process VoxCPM2 doesn't pay that.

**VoxCPM2 streaming is audio-out only.** Text must be complete per request: the prompt builder puts all text
tokens in the prefill and the talker asserts the prompt length. vLLM-Omni's generic streaming-input path isn't
used by VoxCPM2. So onset is cut on the text side (chunking, below), not by feeding partial text.

## Text chunking (`app/chunker.py`, tests in `tests/test_chunker.py`)
- A punctuation mark is a boundary only once the next char is whitespace (after closing quotes/brackets): keeps
  `0.50$`, `3,000`, `1:30`, `example.com`, `Las-Palmas` intact and never cuts a streamed `0.` before `50` arrives.
  Abbreviations/initials (`Dr.`, `e.g.`, `U.S.`, `J. K.`) aren't sentence ends; `5 - 10` / `5–10` aren't clauses.
  Full-width CJK punctuation is always a boundary.
- **First chunk** of each spoken LLM stream may end at a clause (`, ; :` dash) or is cut at 12 words; later
  chunks are whole sentences. **No chunk below `MIN_CHARS` = 40** (user's call): short replies like
  "Привет, как у тебя сегодня дела? Всё хорошо." go out whole, and short sentences merge with the next.
- `say_clean` keeps `$` and `%`; clause dashes become commas (a pause cue).
- Spike (zero-shot, this box): whole-sentence chunking gave onset 292 / 304 / 574 ms (the 574 was a long first
  sentence); clause-first + 40-char window gave 363 / 391 / 422 ms — tighter spread, higher median, because the
  window now waits ~12 tokens. Faster LLM decode (healthy GPU) shrinks that wait directly.

## Engine requests: one per LLM call and one per TTS chunk (no turn-level request ID)
There is no single request that flows Qwen → VoxCPM2 inside vLLM. The two engines are independent; MOUTH reads
the LLM's text, chunks it, and submits each chunk to the TTS itself. Every engine call gets its own request ID
(`llm-<uuid>` in `app/llm.py`, `tts-<uuid>` in `app/tts_voxcpm.py`), and barge-in aborts whichever request is
live by that ID.

| MOUTH job | LLM requests (AsyncLLM) | TTS requests (AsyncOmni) |
|---|---|---|
| plain turn | 1 (reply, tools offered) | 1 per text chunk (clause-first, then sentences) |
| tool turn | 3: reply → tool call; lead-in; phase B answer | 1 for the lead-in + 1 per phase-B chunk |
| backchannel | 1 (`BC_PROMPT` + few-shot, ≤ `bc_max_new` 12 tokens) | 1 (the ack) |

These requests run concurrently: across sessions, and within a session a backchannel runs alongside a turn
(see "Concurrency" below). vLLM / vLLM-Omni batch whatever is in flight (`llm.max_num_seqs` 8,
`deploy/voxcpm2.yaml` `max_num_seqs` 4).

## How a backchannel works
1. **EARS decides when.** In listen mode, every 80 ms frame, the bc head gives `bc_prob`. A `BACKCHANNEL` event
   fires when `bc_prob ≥ ears.bc_thr` (0.5) for `bc_deb` (2) frames, once the user has ≥ 350 ms of voiced
   speech in this turn and EARS is past its ~0.5 s warm-up; then a refractory of 15 frames (1.2 s).
   `app/server.py` forwards it only if the agent isn't speaking, isn't halting after a barge, and no other
   backchannel is in flight.
2. **EARS attaches what the user is saying.** A rolling ASR partial is re-decoded from the last 8 s of user
   audio at most every 0.8 s, off the critical path; the event just reads the latest one. It can be empty early
   in a turn. EARS sends `C {"partial": ...}` to MOUTH.
3. **MOUTH asks the LLM for the ack** (`Session.stream_bc`, on the session's backchannel worker thread):
   `BC_PROMPT` + few-shot examples (EN/RU/ES reactions,
   `mm-hm` / `угу` hums) + the last 3 acks ("don't repeat those") + the partial. A partial under
   `mouth.bc_min_words` (4) or empty is framed as "only a few words so far", so the model answers with a
   neutral hum; otherwise it reacts to the content, in the user's language. There are no canned acks.
4. **MOUTH speaks it**: one VoxCPM2 request for the ack text, streamed back as `K` frames (backchannel audio,
   kept apart from reply audio `A`), then `D {"bc": true}`. Typical first audio ~260–500 ms after the `C`.
5. **EARS plays it through to the browser** while `bc_active` is set and no reply turn is running. Backchannel
   audio is not fed to the second ear and does not count toward the reply's `play_until`, and it never enters
   conversation history.

**Interaction with the user's turn.** If the user's endpoint fires while a backchannel is still being made,
the turn starts at once on its own worker (no waiting on the backchannel); EARS drops any further `K` audio once
the reply turn has started, and the client stops any still-playing ack when the endpoint arrives.

## Concurrency (V1.4) — MOUTH: many sessions, backchannels alongside turns
- **Structure** (`app/mouth.py`): `Mouth` holds the shared engines + voice loading; each EARS connection is a
  `Session` (history, voice, recent acks) on its own thread, with a turn worker and a backchannel worker.
  Barge (`B`) stops only the session's turn. Voices are loaded once per clip (`tts.load_voice` → `Voice`) and
  passed per TTS request, so sessions can use different voices at once. Tokenizer calls are locked (HF fast
  tokenizers aren't re-entrant across threads).
- **Engine sizing for 4 sessions**: Qwen `max_num_seqs` 8, KV 3 GiB (~21k tokens); VoxCPM2 `max_num_seqs` 4,
  KV 2 GiB.
- **VoxCPM2 `enable_unified_decode_graph` must be off.** With it on, vLLM-Omni's VoxCPM2 scheduler
  (`voxcpm2/scheduler.py`) defers *every* new admission while any request is decoding, so the decode batch stays
  on the CUDA-graph path. Under concurrency a new request (a backchannel, another session's sentence) waits until
  everything running finishes: 2 sessions saw a 42 s backchannel stall and turn p50 1.17 s. Off, single-session
  onset costs ~+60–100 ms and the stalls are gone.
- **VoxCPM2 compiles lazily per batch size**, and the first time 2/3/4 requests decode together the *whole engine*
  stalls ~25 s (seen live: the first backchannel next to a turn took 27.6 s, and the turn with it). TTS warmup
  therefore runs every batch size up to `max_num_seqs` concurrently before serving (+~18 s startup).
- **Load test** (`scripts/load_test.py`, MOUTH only: N sessions, 3 turns each, a backchannel fired mid-reply;
  this throttled box):

  | sessions | turn first-audio p50 / max | backchannel p50 / max | GPU peak |
  |---|---|---|---|
  | 1 (cold, first after start) | 510 / 584 ms | 461 / 621 ms | 27.0 GB |
  | 2 | 691 / 794 ms | 635 / 728 ms | 27.0 GB |
  | 4 | 833 / 1126 ms | 829 / 1620 ms | 27.0 GB |


## Concurrency (V1.4) — EARS: up to 4 conversations, batched on one GPU thread
- **Structure** (`app/ears_core/batched.py`): `EarsService` owns one GPU thread; each WebSocket conversation is
  an `EarsSession` on a slot (`server.max_sessions`, more visitors get the busy page). A conversation submits a
  320 ms hop and awaits its events; the GPU thread gathers the hops of all live conversations into one tick
  (`EarsEngine.process`): encoder windows batched by length, then 4 batched head steps. Validated against the
  single-session `EndpointEars` (kept as the reference): same events for 4 staggered sessions, head outputs within
  ~0.007 (bf16 noise).
- **Head**: Qwen3-0.6B + LoRA merged at load, over a static KV ring (`RingKV`, `ears.ring_frames` per slot, its
  own write pointer, validity mask and RoPE positions; reset each turn). One CUDA graph per attention width (129,
  257, … 2049 columns); a step uses the narrowest width covering every active row. Attending the full
  2049 mostly-masked columns was ~2/3 of a step (13.3 → 6.2 ms per step).
- **Encoder stops at layer 15** (the layer the heads read; a thread-local hook on that layer raises): 35 → 24 ms
  for 4 × 8 s windows, features bit-identical. All-zero windows (the agent channel while listening) are cached.
- **ASR** runs on its own thread + CUDA stream. Greedy RNN-T decoding is our own loop
  (`app/ears_core/rnnt.py`), captured as one CUDA graph of 16 steps over a padded batch: **same text as
  nemotron's `generate()`** (80/80 windows, 1–20 s, single and padded batches) and 3–5× faster (8 s window ×4:
  244 → 74 ms; 20 s final: 417 → 88 ms). HF's `generate()` spent ~1 ms of Python per step, holding the GIL the
  tick thread needed; that was the main reason 4 sessions fell behind. Partials of all due sessions decode as
  one batch; endpoint finals go first.
- **Gathering**: the server reads the mic and processes hops in separate tasks, so a conversation that is behind
  says so (`more`: its next hop is already buffered); the next gather waits up to `ears.expect_ms` (60) for those
  catch-up hops instead of `gather_ms` (15), so they share a tick.
- **Torch CPU threads capped** (`ears.cpu_threads` / `mouth.cpu_threads` = 4): the default 48 mostly spin-waited
  (~2400% CPU) and was slower.
- **Memory**: EARS ~6 GiB reserved under 4-session load (run.sh sets `expandable_segments` for EARS; without it the
  allocator cache fragmented on the varying window lengths and grew past 9 GiB).
- **Load test** (`scripts/load_test_ears.py`: N real-time WebSocket conversations, 2 turns each, speech = the
  MOUTH's own TTS; full system on this throttled box). Hop lag = a hop's `state` event back after the hop was sent:

  | conversations | hop lag p50 / p90 / max | endpoint → first reply audio p50 |
  |---|---|---|
  | 1 | 59 / 114 / 202 ms | 529 ms |
  | 2 | 87 / 193 / 397 ms | 636 ms |
  | 4 | 122 / 353 / 716 ms | 810 ms |

  Before this round (per-session gather, HF ASR, full-width head, full encoder), 4 conversations lagged
  1101 / 2960 / 3886 ms and fell further behind. At 4 the GPU thread is still ~70–80% busy (encoder ~40%,
  head ~30%) next to the MOUTH's decoding, so p90 sits just above one hop; a healthy GPU should clear it.
  `ears.stats_s` logs tick timing, busy %, and memory.
- **Shutdown**: MOUTH handles SIGTERM/SIGINT by shutting both engines down; a bare SIGTERM used to orphan the
  VoxCPM2 stage process (holding ~9 GB).

## Memory + memory pinning
Both engines pin their KV caches instead of filling a GPU fraction: VoxCPM2 2 GiB (`deploy/voxcpm2.yaml`, also
`max_num_seqs 4`, `max_model_len 2048`; stock deploy was 6 GiB / 8 seqs), Qwen 3 GiB (`llm.kv_cache_gib`,
`max_num_seqs 8`, `max_model_len 8192`). `gpu_memory_utilization` is then only vLLM's start-up free-memory
check. Total **~28.5 GB / 32 GB** warm and under 4-conversation load (EARS ~7 GB) (V1.3 single-session sizing was ~24 GB; MOSS +
vLLM server was ~30.5 GB).

## V1.3 — smoke test on the box (cloned voice `en`, 9.3 s clip, no transcript)
- `scripts/smoke_test.py` **PASS**: plain first-audio **503 ms** (first LLM text at 92 ms), Tavily weather tool
  turn **743 ms**, placeholder reminder turn **770 ms** (admits not connected), EARS speech → endpoint → reply.
- Plain-turn breakdown: ~200 ms waiting for the 40-char first chunk at 92 tok/s + VoxCPM2 encoding the 9.3 s
  reference clip on every request (zero-shot first-audio was 113 ms). Lever: VoxCPM2's precomputed
  `voice_profile` mode (vLLM-Omni loads custom voice profiles from a dir) to skip the per-request clip encode.
- Logs grep-checked: no transcript or reply text in either process's output.
- Startup (warm cache): MOUTH ~108 s (VoxCPM2 55 s, Qwen 32 s, TTS warmup 21 s), EARS ~15 s. First-ever start
  adds downloads, VoxCPM2's one-time compile (~2 min warmup) and Qwen's CUDA-graph capture.
- EARS per-hop compute was p50 275 / p90 404 ms per 320 ms hop on this box (same in the old pinned venv);
  superseded by the batched engine (single conversation now ~60 ms per tick, see Concurrency — EARS).

## What was copied from exp-12, and what was dropped
EARS (`app/ears_core/`) is exp-12 **arm C** copied and flattened (no `codebase/` imports): nemotron encoder,
StateChannelModel/YieldModel/EndpointModel, LiveEars, EndpointEars, plus a minimal `policy._decide` (the full
`fdb_eval`/VAD harness isn't needed live). **Dropped vs exp-12**: the 4-way endpointer, the SPEAK-lock
watchdog, affect/gender, A/B blinding + the sealed code→ckpt map, and all per-session transcript/frame logging
(privacy). MOUTH keeps exp-12's Hermes lead-in + phase-B flow. **Backchannels are always contextual LLM acks**
(user's call; canned acks removed), see "How a backchannel works" above. Hums are fine *when the context calls
for them* (short/empty partial). History: the first prompt led with 'mm-hm' as its example and the model said it
nearly every time; then acks came back in English for Russian speech until few-shot examples were added.
reminder/message tools return a "not connected" demo marker so the model says so (and never log their content).

## History (superseded)
- **V1.2** — three processes / three venvs: Qwen3-4B on a vLLM OpenAI server (`:8001`, TTFT 28 ms, ~150 tok/s),
  MOUTH with in-process **MOSS-TTS-Realtime** (transformers 5.0.0; vendored + patched to a static-cache /
  CUDA-graph fast path, RTF ~0.55–0.7, onset ~480 ms plain / ~577 ms tool turn), EARS in its own venv. GPU ~30.5
  / 32 GB. Removed in V1.3 (code is in git history).
- **V0** — MOSS-TTS-Nano bench: first-audio p50 112 ms, CER 1.17%, RTF 0.70–0.94; the user found its voice
  "weird" and moved to MOSS-TTS-Realtime.

## Still open
- Judge the VoxCPM2 cloned voice live (user), and add `voices/<name>.txt` transcripts if quality needs it.
- V2: retune `ep_thr`/`ep_deb` + barge/bc live with `?debug=1` (0.9146 was never retuned live). Freeze into `config.yaml`.
- V3: Docker build/run + container smoke.
- Re-benchmark everything on a healthy GPU.
