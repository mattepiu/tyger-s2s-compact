# Changelog

What changed in tyger, newest first, with the numbers behind each change.

**How to keep it:**
- **While developing,** every change goes under **Unreleased**, in the dated section for the day it was made.
  Group the lines under Added / Changed / Fixed / Removed, and give each one:
  - what changed, and why;
  - the number before → after, where there is one, naming the box and config it was measured on;
  - the commit.
- **At a release,** Unreleased becomes the version's heading, and a new empty Unreleased starts above it.
- **Comparing numbers:** the GPU box has changed several times. They compare only within one box and config.

## Headline numbers

Trend only: the rows come from different boxes and configs (noted). Every box is an RTX 5090; the first was
throttled to half memory clock.

| When | Code / config | Smoke: plain-turn first audio | 4 conversations: turn first audio p50 | End of speech → first audio p50, 1 / 4 conversations | GPU peak |
|---|---|---|---|---|---|
| 2026-10-02 | Qwen3-4B + VoxCPM2, 4 sessions (throttled box) | 503 ms | 833 ms | not measured yet | 27.0 GB |
| 2026-10-03 | `b64c772`, first healthy box | 354 ms | 539 ms | 1013 / 1022 ms | 28.4 GB (29.3 with tool turns) |
| 2026-10-03 | after the review fixes and the speaker cache, Qwen3-4B | 263–272 ms | 400–425 ms | — | 27.0 GB |
| 2026-10-04 | Qwen3.5-4B (`a7f4cf8`, the plugin-refactor baseline) | 243 ms | 482 ms | 970 / 1260 ms | 27.7 GB |
| 2026-10-04 | Gemma 4 E4B FP8 responder | 225 ms | 438 ms | 940 / ~1190 ms | 30.0 GB |
| 2026-10-06 | Gemma, production config (smart + tasks), new box | 279 ms | — | 904 / 1109 ms | 30.4 GB |
| 2026-10-06 | after the readability refactor, no `smart`, whole-sentence first chunk | 170–282 ms | 474 / 515 ms (2 runs) | ~880 / ~1090 ms | — |

"End of speech → first audio" is what the user hears as the response gap. About 650–850 ms of it is endpoint
detection, which live tuning of the endpoint thresholds is meant to cut.

## [Unreleased]

### 2026-10-06 — Readability refactor; Gemma-only responders; whole sentences to the TTS

#### Changed
- **The responder stays Gemma 4 E4B FP8, with `gemma-4-e4b-nvfp4` (unsloth) registered as the backup** (user's
  call). The backup routes like FP8 and uses 3.6 GB less GPU. But it sends trivia to `web_search` where FP8 answers
  aloud, so expect more lookups if it's switched in. `bcf2a59`
- **The first TTS chunk of a reply is a whole sentence** (`mouth.first_chunk: sentence`, the new default).
  - Why: each chunk is its own VoxCPM2 request, so the old clause-cut first chunk restarted the intonation
    mid-sentence. The user listened: "sounds much better".
  - Cost: +30 ms turn first audio at 1 conversation, ≈ +95 ms at 4.
  - `clause` stays available. `71e9522`, `df4eac5`
- **VoxCPM2's `max_model_len` is back to upstream's 4096.** It costs nothing; a request is ~200–300 tokens.
  `a4eef2e`
- **Readability refactor** (`REFACTOR.md`, steps R-01..R-11). Each step was parity-checked: smoke, load tests, and
  the EARS validators to the last digit. The new layout:
  - `app/`: `ears/`, `mouth/`, `plugins/`, `prompts/`, `runtime/`, `tools/`, plus `config.py` / `wire.py`;
  - `config/`: `default.yaml`, `fake.yaml`, `prompts.yaml`, `engines/voxcpm2.yaml`;
  - `scripts/`: `stack/`, `validate/`, `bench/`;
  - `tests/`: `ears/`, `mouth/`, `plugins/`;
  - `./stop.sh` next to `./run.sh`.

  `e19e609`
- **Config is checked at load.** Plugins subclass ABC bases with typed `Options`, so a config typo stops startup
  with the key named (`unknown option kv_cache_gb (did you mean kv_cache_gib?)`). Before, it was silently ignored.
  Renamed keys are rejected with the new name: `deploy_config` → `engine_config`. `29a8ce7`, `e19e609`
- **One base for the hosted LLMs** (`HostedLLM` for `openai` and `anthropic`). `54a5738`
- **One tool-argument check for both hosted plugins.** On `openai`, a call with a wrong type or an out-of-enum
  value is now rejected instead of run. `00dcb5e`
- **Tool registry** (`app/tools/`): one `Tool` per name, and MOUTH no longer switches on tool names. `82189bd`
- **EARS plugins declare their own thresholds,** and the page builds its tuning sliders from them. `c3c0638`
- **`nemotron_tyger` is one plugin package** (the ASR plus the turn-taking heads).
  - The heads load from the `TYGER-S2S-endpointer` release (v0.1.0-rc1): bit-identical to the research checkpoint
    (335 / 335 tensors), and read once instead of three times.
  - The validators check the release.

  `67a8ec7`, `657543e`, `63ec142`

#### Fixed
- **Partials failure:** a failure transcribing partials could blank an endpoint transcript that was already
  resolved. `d901503`
- **Lost MOUTH:** losing the connection to MOUTH now ends the conversation with an error to the page. Before, the
  session went silently dead while mic audio piled up. `d901503`
- **`ears.nemotron` is read** (it was ignored). `d901503`
- **Unspeakable acks:** a backchannel ack with nothing speakable ("—") no longer reaches the TTS. `d901503`
- **Voice:** the apology / refusal line is spoken in the turn's voice, not a voice switched mid-turn. `afc35d9`
- **Second ear:** the agent's own audio, fed to the turn-taking heads, goes through a stateful resampler instead of
  per-chunk `np.interp`. SNR against the original audio −0.9 dB → 29.7 dB; barge-in detection unchanged.
  `1b415eb`
- **Threshold mark:** the page's meter mark sits at the threshold's value (τ 0.9146 was drawn at 89%). `c3c0638`

#### Removed
- **LLM plugins that lost to Gemma 4 E4B:** `qwen3-4b-2507`, `qwen3.5-4b`, `qwen3-8b-nvfp4`, `qwen3-14b-nvfp4`,
  `nemotron-nano-9b-v2-nvfp4`, `gemma-4-e2b`, `gemma-4-26b-a4b-nvfp4`, `gemma-4-12b-nvfp4`, and the Qwen XML and
  Nemotron tool-call parsers. The code is in git history. `bcf2a59`
- **The research reference code and checkpoint path:** the validators now check the heads release. `63ec142`

#### Known issues
- The EARS engine aborts at process exit, after its work has finished.

### 2026-10-05 — Gemma 4 memory and responder trials (responder unchanged)

#### Measured
- **Gemma 4 E4B's vision and audio encoders are not loaded** (`language_model_only`). E4B loads 10.54 GiB, which
  is exactly the FP8 text model.
- **The bf16 per-layer embedding table is 5.25 GiB,** half the weights. Offloading it to CPU was ruled out as too
  slow (user).
- **Responder trials** on the routing test (`scripts/bench/escalation.py`; gate: ≤ 10% of hard questions answered
  locally):

  | Model | Weights | Fits | Missed escalations | Verdict |
  |---|---|---|---|---|
  | Gemma 4 E4B FP8 (current) | 10.5 GiB | yes | 0.6% | kept |
  | unsloth Gemma 4 E4B NVFP4 (int8 per-layer embeddings) | 7.1 GiB | yes | 0.2% | kept as backup (−3.6 GB GPU) |
  | Gemma 4 E2B FP8 | 7.1 GiB | yes | 42.9% | fails; leaks its thinking, wrong-language answers |
  | Qwen3-8B / 14B NVFP4 | 6.0 / 10.0 GiB | yes | 13.8% / 18.5% | fail: lean local |
  | Nemotron Nano 9B v2 NVFP4 | 7.4 GiB | yes | 20.4% | fails; narrates tools, no Russian |
  | unsloth Gemma 4 12B NVFP4 | 9.1 GiB | yes | 14.8% | fails; leaks the thought channel |
  | Gemma 4 26B-A4B NVFP4 | 17.1 GiB | **no** (~4.5 GiB over) | 1.9% | passes; candidate for a bigger card |

  `d80b951`, `564993e`, `f2ce33f`

#### Added
- **Bench helpers** for comparing stacks and LLM weights: `scripts/bench/stack.sh`, `summary.py`,
  `llm_weights.py`. `5c05eb2`

#### Known issues
- If `run.sh` gives up waiting on a slow first start, the LLM engine is left orphaned on the GPU, holding ~15 GB.
  `e5452ec`

### 2026-10-04 — Plugins, Gemma responder, smart escalation, background tasks

#### Added
- **Plugin architecture** (`PLUGINS.md` steps 0–9).
  - LLM, TTS and EARS are plugins chosen in config, with LLM roles `reply` / `filler` / `smart`.
  - Every prompt is in one file (`config/prompts.yaml`), with per-model overrides.
  - Fake plugins and `config/fake.yaml` run the whole stack without a GPU, and contract tests cover every plugin.
  - Local plugins matched the old numbers (parity) before any hosted plugin was written.

  `ddb1128` … `5a10efe`
- **Hosted LLM plugins:** `openai` (any OpenAI-compatible endpoint) and `anthropic` (tested offline only; no key).
  - A remote `reply` costs ~0.35 s of first audio against a local one.
  - A fully remote LLM frees ~12 GB of GPU (peak 15.3 GB).

  `a5c3549`, `af11348`, `ac2b4de`
- **Smart escalation:** the local responder calls `ask_expert` for heavy turns.
  - The lead-in filler is picked by the tool's name.
  - The expert's answer streams straight to the TTS.
  - `smart` = OpenAI gpt-5.4-mini: the expert's first audio comes at 1.0–1.3 s, while the lead-in is still
    playing, so there's no dead air. A barge-in stops it in 6 ms.

  `11dada2`, `80bd78b`
- **Routing test** (`scripts/bench/escalation.py`): 140 synthetic cases (EN / RU / ES; local, tool, expert and
  background routes; held-out and fresh splits). It also scores reply language and spoken markup. It stubs every
  tool and needs no keys. `27e9cab`, `c4b7710`
- **Background tasks:** `start_task` / `cancel_task` run research on the `smart` model. When a task finishes, the
  agent takes the floor (wire `F` / `G`) after the user has been silent, and announces the findings.
  - Checked live in EN and RU.
  - On by default.

  `8744f01`, `eb50c90`, `77ee1c0`
- **Deterministic EARS load test:** a fixed utterance spoken through the TTS-only wire message `S`, a forced reply,
  and fixed hop phases. Two runs at 1 conversation agree within 4 ms. `9b60649`

#### Changed
- **Responder Qwen3.5-4B → Gemma 4 E4B FP8** (user: "robust tool calling is crucial").
  - Routing: 0.4% missed escalations, against Qwen3.5's 15.6% on held-out cases.
  - 4-session turn first audio p50 476 → 438 ms.
  - GPU peak 27.7 → 30.0 GB.

  `4715880`, `2c661c3`
- **Gemma robustness:**
  - Tool calls written without their markers are parsed as calls, never spoken.
  - The user's language is named next to each tool result: Russian / Spanish answers that came out in English went
    from 29% to 0.
  - Few-shot turns cover both modes.

  `2c661c3`
- **Tool-call IDs** are unique per session (needed by hosted APIs). `fd9af75`

#### Removed
- **The hosted `openai` TTS plugin:** first audio 1.7–2.3 s and up to 1.7 s of dead air between chunks, against
  VoxCPM2's 240–420 ms and none. VoxCPM2 is the only TTS. `15383c5`

### 2026-10-03 — Code-review fixes on a healthy box; Qwen3.5-4B

#### Fixed
- **Interrupted turns:** a turn the user interrupts stays in the LLM's history, cut to what had played. `c4ed6a7`
- **Start of an interruption:** it's kept in the transcript. The first 3 words are recognised 83% of the time (was
  14%); all words 94% (was 69%). `a116c60`
- **Barge-in during a tool call:** it stops the turn in 5 ms (was 2–2.5 s, up to the tool's 15 s timeout).
  `c4ed6a7`
- **Long user turns:** transcribed in segments instead of keeping only the last 20 s. Word recall on a 51 s turn
  0.42 → 0.94. `d9ed3b8`
- **Stale backchannels:** a backchannel queued before a reply turn is cancelled. `c4ed6a7`
- **Tool calls in history:** tool calls and compact results stay in history, so later turns keep calling tools.
  Weather turns that called the tool: 9/24 → 24/24. `6a083e9`
- **Fresh-box install:** pinned `accelerate` to what vllm-omni requires. `a06a2e8`

#### Changed
- **Voice clip:** it's resampled once, and VoxCPM2's speaker cache holds it.
  - Per-request first audio 146 → 78 ms.
  - Plain turn 354 → 263 ms; 4-session turn p50 ~550 → ~410 ms.

  `826876f`
- **EARS features on the GPU:** EARS computes log-mel features there. 4-session tick p50 90 → 59 ms; a bf16
  encoder was tried and rejected. `8e4ed69`
- **Tool filler:** it starts at the `<tool_call>` tag, and still-looking lines cover slow tools. Tool-turn first
  audio 640 → 408 ms; worst dead air 2.6 → 1.3 s. `1b5f269`
- **Responder Qwen3-4B-Instruct-2507 → Qwen3.5-4B** (thinking off, text only). `8dee217`

#### Added
- **Response-gap metric:** the load test measures end of speech → endpoint → first audio. It showed a ~1.0 s gap,
  ~77% of it endpoint detection. `e920d02`
- **Docs:** the code review's issue list, and HANDOFF.md (the GPU-box agent's work queue). `b64c772`

### 2026-10-02 — VoxCPM2, one venv, 4 conversations per GPU

#### Changed
- **MOSS-TTS-Realtime → VoxCPM2.** VoxCPM2 runs in-process with the LLM in MOUTH on vLLM-Omni, and both processes
  share one venv. GPU memory went from ~30.5 GB to ~24 GB (single session). TTS first audio 113 ms (zero-shot),
  RTF 0.12–0.20. `e8496fb`
- **Chunker:**
  - the first chunk could end at a clause;
  - a 40-character minimum;
  - punctuation inside numbers and words is never a cut;
  - `$` and `%` are kept.

  `acc1846`, `2279d9a`
- **Backchannels are always contextual LLM acks.**
  - A neutral hum only when there's too little to react to.
  - Replies in the user's language, via few-shot examples.
  - No more constant "mm-hm".

  `4114492`, `660d214`, `2a80faa`

#### Added
- **MOUTH concurrency:** sessions run in parallel, and backchannels alongside turns. 4-session turn first audio
  p50 833 ms (throttled box). `ca84fef`
- **Batched EARS for 4 conversations:**
  - an own CUDA-graph RNN-T decoder (3–5× faster);
  - head graphs per attention width;
  - the encoder stops at layer 15.

  Hop lag p50 / p90 1101 / 2960 → 122 / 353 ms. `d2cb551`, `b2bf530`

#### Fixed
- **Barge-in over fast TTS:** the turn lasts until the browser has played the reply. Before, it ended when
  generation ended. `f8d4f73`
- **Stale voice:** a voice no longer carries over from one session to the next. `85615c5`
- **VoxCPM2 stalls under concurrency:**
  - The unified decode graph is off: it deferred every new request, causing stalls of up to 42 s.
  - Warmup now covers every batch size, ending ~25 s compile stalls mid-conversation.

  `ca84fef`

## v1 — imported baseline (`07d651e`, 2026-10-02)
- **Processes:** three, in three venvs.
- **TTS:** MOSS-TTS-Realtime with a static-cache / CUDA-graph patch (RTF 0.55–0.7, first audio ~480 ms).
- **LLM:** Qwen3-4B on a vLLM HTTP server.
- **Capacity:** one conversation per GPU, ~30.5 GB.
- **Earlier:** a MOSS-TTS-Nano benchmark (first audio 112 ms); the user found its voice "weird".
