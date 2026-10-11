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

Trend only: the rows come from different boxes and configs (noted). Every box was an RTX 5090 until 2026-10-11 (the
first throttled to half memory clock), and since then an RTX PRO 5000.

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

"End of speech → first audio" is what the user hears as the response gap. About 650–850 ms of it is endpoint
detection, which live tuning of the endpoint thresholds is meant to cut.

## [Unreleased]

### 2026-10-11 — A new card, the RTX PRO 5000 (48 GB); the silence fallback and the weather / web-search tools removed

Box: an **RTX PRO 5000 Blackwell** (48 935 MiB, sm_120 like the 5090; driver 595.84, CUDA 13.2; ECC off; 32 CPU
threads, 62 GB RAM, 100 GB disk). Under load: P1, SM ~2540 MHz (max 3090), memory 13 365 MHz, at most 234 W of its
300 W, with no throttle reason. The fresh-box install (README step 1) went through unchanged. The first `./run.sh`,
downloads included, took 11.5 min (Gemma's download and load 169 s, its first compile 75 s).

#### Measured
- **The current stack on this card, against the RTX 5090 of 2026-10-10** (the stack without `smart`, the code before
  the change below; `scripts/bench/stack.sh`, plus `scripts/stack/load.py --sessions 1 --turns 6` ×3, and with the
  stack stopped, `scripts/bench/llm_weights.py` and `scripts/bench/tts_voice.py`):

  | | RTX 5090 (2026-10-10) | RTX PRO 5000 |
  |---|---|---|
  | GPU peak at 4 sessions / the card | 30.2 / 32.6 GB | 29.9 / 48.9 GB (**~19 GB free**) |
  | Gemma E4B FP8: weights; single-stream decode | 10.39 GiB; 186 tok/s (2026-10-05) | 10.39 GiB; **154 tok/s** (83%) |
  | VoxCPM2 cloned-voice first audio per request | 78 ms (2026-10-03) | 79 ms |
  | smoke: plain / weather / reminder / backchannel first audio | 174 ms plain | 216 / 200 / 258 / 154 ms, PASS |
  | `load.py` turn first audio p50, 1 session (3 runs) | — | 262 / 255 / 262 ms |
  | `load.py` turn first audio p50, 4 sessions (3 runs) | 397 / 453 / 439 ms | **532 / 581 / 578 ms** |
  | `load.py --tools --fresh`, 4 sessions (3 runs) | — | 395 / 413 / 413 ms |
  | `load_ears.py` 1 conv: end of speech → endpoint; → first audio p50 | —; 793 ms | 600 ms; **837 ms** |
  | `load_ears.py` 4 conv: end of speech → endpoint; → first audio p50 | —; 992 ms | 707 ms; **1096 ms** |
  | EARS hop lag p50 / p90, 4 conversations | 47 / 124 ms | 68 / 181 ms |

  - The card doesn't make the stack miss its latency, but it is slower under load. At 1 conversation the response
    gap is +44 ms. At 4 it is +104 ms, and MOUTH's 4-session turn first audio is +~130 ms. Decode runs at 83% of
    the 5090's, as the bandwidth ratio predicts (1344 / 1792 GB/s = 75%), and EARS has less compute (hop lag p90
    124 → 181 ms).
  - VoxCPM2 alone is as fast as on the 5090, and the endpoint detection at 1 conversation (600 ms) is as fast as on
    the fastest 5090 box.
  - An LLM engine could have ~19 GB more than E4B's ~15 GB here: ≤ ~28 GiB of weights with the same 3 GiB of KV.
  - All 57 endpoints of the run came from the head and none from the silence fallback (the stack log's
    `endpoint by=`).
- **Bigger responders on the 48 GB card: three 4-bit mixtures of experts against Gemma E4B** (the user's list).
  - Routing: `scripts/bench/escalation.py --samples 8`, with `smart` + tasks (140 cases) and without (70 cases), on
    the prompts after the routing fix below (Qwen3.5 on the prompts before it).
  - Memory and speed: `scripts/bench/llm_weights.py`.
  - The stack: `scripts/bench/stack.sh` + `load.py --sessions 1` ×3, on the stack without `smart`.

  | | Gemma 4 E4B FP8 (current) | Qwen3.6-35B-A3B NVFP4 | Qwen3.5-35B-A3B GPTQ int4 | Gemma 4 26B-A4B NVFP4 |
  |---|---|---|---|---|
  | weights on the GPU; GPU peak at 4 sessions | 10.4 GiB; 29.9 GB | 19.5 GiB; 39.7 GB | 20.3 GiB; — | 16.8 GiB; 36.8 GB |
  | single-stream decode | 154 tok/s | **224 tok/s** | — | 151 tok/s |
  | with the expert: right; missed escalations | 99.1%; **0.2%** | 98.7%; 1.3% | 88.0%; **25.2%** (fails) | **99.3%**; 1.5% |
  | without the expert: right; tool requests answered locally | 99.8%; 0.4% | 100%; 0% | 100%; 0% | 98.4%; **3.6–7.3%** |
  | wrong-language replies, with the expert: spoken / after a tool | 8 / 346; 2 / 415 | 1 / 343; **9 / 442** | 1 / 479; 5 / 437 | **0 / 352; 0 / 437** |
  | wrong-language replies, without: spoken / after a tool | 3 / 313; 0 / 247 | 6 / 312; 4 / 248 | **40 / 312**; 0 / 248 | 0 / 321; 3 / 239 |
  | turn first audio p50, 1 session (3 runs) | 255–262 ms | 248–288 ms | — | 254–299 ms |
  | turn first audio p50, 4 sessions (3 runs): plain; tool turns | 527–571; 404–411 ms | 541–576; **606–636 ms** | — | 579–634; 474–499 ms |
  | end of speech → first audio p50, 1 / 4 conversations | 832 / 1093 ms | 846 / 1039 ms | — | 874 / 1028 ms |

  - **Qwen3.5-35B-A3B fails the routing gate.** It answers a quarter of the expert questions itself: code, "think
    carefully" maths, health and money questions, follow-ups. Every smaller Qwen did the same before. Without the
    expert, 40 of 312 replies came out in another language. Its checkpoint is deleted.
  - **Qwen3.6-35B-A3B routes well and decodes fastest, but its language is fragile.**
    - It needed Qwen's own sampling settings. With the temperature alone it switched language mid-word ("залей
      водой вtwice the volume") and answered Spanish in Russian. The plugin now sets the card's top_p 0.8, top_k 20
      and presence_penalty 1.5.
    - It also needed the no-expert few-shot turns reordered (`config/prompts/qwen3.6-35b-a3b-nvfp4.yaml`): it follows
      the language of the last non-English example. With the base order, Spanish was answered in Russian 24 times
      in 560.
    - Even then it copies example answers word for word in the example's language. A Spanish trip request got the
      Russian task example's reply ("Хорошо, займусь маршрутом…"), and once a Russian markdown news list.
    - Its tool turns are ~200 ms slower at 4 sessions, although it decodes 45% faster. A likely cause, not
      confirmed: vLLM sets its attention block size to 1056 tokens to match the DeltaNet state, and prefixes are
      cached only in whole blocks. So a short prompt, like the tool lead-in's, is read again in full every time.
  - **Gemma 4 26B-A4B is the strongest candidate.**
    - It routes like E4B (1.5% missed against 0.2%) and never replied in the wrong language with the expert, where
      E4B did 10 times.
    - Without the expert it often answers reminder and message requests "not connected" without calling the tool
      (3.6–7.3%, two runs; also seen on 2026-10-05).
    - It costs +50–70 ms of first audio at 4 sessions and ~7 GB more GPU than E4B.
    - Its "count to ten" reply in `load_ears.py` varies (3–6 texts a run), so its response-gap numbers compare less
      well than the others'.
    - Its thought channel: after a tool result it opens `<|channel>thought`, mostly empty and closed at once. Now and
      then (~0.5% of answers after a tool) it never closes the channel and writes the answer inside it, after a JSON
      "thought" object or a stray `<tool_response|>` token.
      - A filter that dropped everything in an unclosed channel left 1–4 answers in ~435 silent.
      - The plugin now keeps the answer after that object or token (the routing eval's empty answers 4 → 0 in the
        final runs, no thought spoken).
      - The vLLM plugins' end-of-stream check now treats a call that only appears in that final pass as a call, not
        speech. This also covers E4B's bare `set_reminder{…}` written at the very end.
  - **First loads compile kernels, and RAM limits how many at once.**
    - The first load of Qwen3.6 and of Gemma 26B JIT-compiles FlashInfer kernels: 8 min and 17 min at 4 jobs. Each
      compiler process takes 3–6 GB of RAM, and by default there is one per CPU thread (34 jobs here). That twice ran
      this box's 64.6 GB container out of memory and restarted it.
    - With `MAX_JOBS=4` the peak was 32 GiB. The builds are cached for later loads.
  - **The recommendation (for the user):**
    - Keep Gemma E4B as the default. It routes best, and nothing in these numbers beats it outright.
    - If a bigger responder is wanted, the candidate is Gemma 4 26B-A4B on this 48 GB card (it can't fit on the
      5090). It's worth a live listen with `smart` on, to judge its local answers, before any switch.
    - Not Qwen3.6, for its language copying and slower tool turns.
    - With E4B, the 5090 is the faster card for this stack: decode 186 against 154 tok/s, and 4-session turn first
      audio 397–453 against 527–571 ms.

  `4eb9ce2`
- **The responders again, English only** (the user's question: what if the agent spoke only English?).
  - Setup: `config/prompts_en.yaml`, with every Russian / Spanish few-shot turn translated and "always reply in
    English" (Added, below). The eval runs on the 71 English cases plus 48 new harder ones (`split: hard`), with
    `escalation.py --lang en --samples 8` and no per-model prompt override.

  | | Gemma 4 E4B FP8 | Qwen3.6-35B-A3B | Gemma 4 26B-A4B |
  |---|---|---|---|
  | with the expert, 119 cases: right; missed; unneeded escalations | 99.4%; 0.0%; 0.0% | 99.4%; 0.7%; 0.3% | 100%; 0%; 0% |
  | the 48 hard cases only: right; missed escalations | 98.7%; 0% | 99.0%; 1.7% | 100%; 0% |
  | without the expert: right; tool requests answered locally | 99.8%; 0.5% | 100%; 0% | 99.4%; 1.4% |
  | answers after a tool that say nothing (with / without the expert) | 0 / 0 | 0 / 0 | **25 / 8** (of ~320 / ~205) |

  - **Without other languages, routing no longer separates the three.** The hard cases didn't change that:
    - E4B's one steady miss is h35 ("That can't be right. Can you double check that?"), which goes to web search 5
      of 8 times, arguably a fair way to check a fact;
    - Qwen3.6 answers h37 ("Actually, I'm diabetic. Is that safe for me to eat?") itself 3 of 8 times.
  - **Qwen3.6's language problems are gone,** with no reordered examples needed. What's left against it is the
    ~200 ms on tool turns.
  - **Gemma 26B often answers a tool result with another tool call** (5–8% of answers after a tool). In the answer
    step MOUTH offers no tools and speaks only text, so the user hears the lead-in and then silence. A probe of the
    answer step found:
    - 15 calls in 320 answers: `ask_expert` 8 (after an inconclusive search), `get_weather` again 4, a made-up
      `get_exchange_rate` 2, `get_current_time` 1;
    - E4B and Qwen3.6 never do it.
    - The 26B would need MOUTH to allow a second tool round, or a fallback line, before a live listen.
  - **The recommendation stands:** keep E4B. The routing test can't show a bigger model doing better in English;
    what it would improve is the quality of local answers, which needs its own test. If English-only is the plan,
    Qwen3.6 becomes the more interesting candidate (once its tool-turn delay is understood), and Gemma 26B needs the
    second tool round first.

  `0ab7d95`
- **The default stack on the final code** (Gemma E4B, without `smart`): smoke passes (plain / weather / reminder /
  backchannel first audio 215 / 196 / 255 / 152 ms); `load.py --sessions 4 --turns 6 --tools --fresh` p50 377 ms, 0
  errors.
- **The live check of `openai-responses`** (the user's OK for paid calls; `gpt-5.4-mini`, web search on, synthetic
  questions; no code changed).
  - **The plugin works against the API:** streaming, OpenAI's web search (one search per live-fact question), function
    calls and their results sent back, Russian answered in Russian, the time from `get_current_time` without a
    search. Plugin-level probes sent MOUTH's own escalated-turn request (`smart_messages` + the smart tools).
  - **Reasoning effort:** web search accepts `none`; `minimal` is refused (HTTP 400); `low` (set) and `medium` work.
    Time to first text, warm, median of 4, `none` / `low`:

    | | weather EN | price EN | weather RU | savings maths |
    |---|---|---|---|---|
    | first text | **0.96** / 1.67 s | **1.07** / 1.67 s | 1.91 / 1.46 s | 1.84 / 1.58 s |
    | `get_current_time` called first (an extra round) | 0 / 4 | 0 / 4 | 3 / 2 of 4 | 2 / 1 of 4 |

    `none` is faster on live facts mostly because `low` checks the time first. But the maths got worse: of 3
    savings answers read, `none` gave one wrong total (111 000 instead of ~82 200); `low` gave 2 right and one
    **empty answer** (four `get_current_time` rounds in a row, then nothing). Once, `low` called `set_reminder`
    with the text "Dummy" for the same maths question. `low` stays until those are fixed.
  - **End to end on the stack** (default config, smart on, 3 runs of 6 transcript turns through MOUTH): E4B escalated
    every live-fact and maths turn and kept the time and thanks turns. The lead-in played at 255–439 ms; the smart
    model's first event came 0.48–1.31 s after its request (weather, maths), 3.5–4.2 s for the Formula One news.
  - **Found on the stack:**
    - an English question right after a Russian turn ("Who won the last Formula One race?") was answered in
      **Russian in 2 of 3 runs**. A plugin-level replay with a shorter context answered in English 4 / 4, and the
      real request isn't logged (privacy). Likely fix: name the reply language in the smart request, as MOUTH
      already does after tool results (`result_note`);
    - weather answers wrote "19°C, or 66°F" (4 of 6) though the smart prompt asks for spoken words only;
    - 7 of 12 escalated turns had `get_current_time` rounds first, one of them 3.
  - A probable cause of the repeated calls (untested): with `store: false` the plugin doesn't send the reasoning
    items back with the tool results. OpenAI's docs ask for them in stateless function calling with reasoning
    (`include: ["reasoning.encrypted_content"]`), or the model loses why it made the call.

#### Added
- **Responder trial plugins for a 48 GB card** (none is the default): `gemma-4-26b-a4b-nvfp4` (with the
  thought-channel filter), `qwen3.6-35b-a3b-nvfp4` and `qwen3.5-35b-a3b-gptq` (`app/plugins/llm/qwen_moe.py`: thinking
  off, text only, Qwen's sampling). Qwen XML tool calls are a third parsed format again (`qwen_xml`, vLLM's
  `qwen3_coder`), with tests, and `strip_think` takes another tag pair. `llm_weights.py` loads Qwen3.x text-only.
  Numbers under Measured. `4eb9ce2`
- **An English-only prompt file and harder English routing cases.** `config/prompts_en.yaml` is
  `config/prompts.yaml` with every Russian / Spanish few-shot turn translated (the same roles, tools and order) and
  "always reply in English"; use it with `mouth.prompts`. `scripts/bench/escalation_set.yaml` gains 48 harder
  English cases (`split: hard`):
  - disguised health, legal, money, maths and code questions;
  - hard-sounding questions that are local;
  - lookups with health or money words in them;
  - research for later against a question that can be answered now;
  - several turns (a follow-up after an escalation, a topic switch, unhappiness with an earlier answer, the stakes
    changing);
  - rambling turns with the question at the end.

  `escalation.py --lang en` evaluates one language. `0ab7d95`
- **`openai-responses`, an LLM plugin for OpenAI's Responses API with its built-in web search**
  (`app/plugins/llm/openai_responses.py`). It is now the `smart` role in `config/default.yaml` (`gpt-5.4-mini`,
  `web_search: true`, `reasoning_effort: low`), so the expert can look up the live facts the removed tools used to
  (Removed, below).
  - It sends `store: false`, offers the function tools MOUTH gives the expert next to `{"type": "web_search"}`, and
    maps the neutral history to Responses items (`function_call` /
    `function_call_output`).
  - The stream parser speaks the text deltas, names a function call as soon as it starts, and runs no call cut off by
    `max_output_tokens`. A refusal becomes a `refusal` event and a failed response an `LLMError`. Searches are only
    counted, never logged.
  - Tests: `tests/plugins/test_hosted_openai_responses.py` (no network, no key): history and tool conversion, the
    parser on synthetic event streams, and the plugin on a stub client (what it sends, stop closes the stream and
    frees the slot, HTTP 500 → `LLMError`).
  - Checked live later the same day (Measured, above): it works; `none` is accepted with web search; the open
    issues are listed there.

  `b28f889`
- **GPU-free tests for EARS's turn-taking state machine** (`tests/ears/test_conversation.py`, 6 tests). The
  Conversation runs on the `fake` EARS plugin, with a fake page and a fake MOUTH over TCP, and covers what the floor
  and 2nd-ear tests didn't:
  - **Halting** (after a barge, until MOUTH's D): no backchannel and no floor grant, then both once D arrives.
  - **An endpoint during a halt** is held and sent as the next turn the moment D arrives.
  - **The reset after a barge** comes exactly once, before the hop after the BARGE one, which is where the plugin
    seeds the new turn with the user's start (`test_rules.py`). There is none when MOUTH's barged D arrives, so the
    user's speech between the barge and D still makes a turn (I-02).
  - **A reply that plays out** resets the session before the next turn.
  - **MOUTH's D after the page has played everything** ends the turn at once, with no timeout left running.
  - **A D before the end, then a barge in the tail,** closes the turn on the spot. No B goes out, since MOUTH has
    nothing left to stop, and the user's next turn goes straight out instead of being held.

  Each test fails on a matching break of `app/ears/conversation.py`: a second reset on the barged D, no hold while
  halting, a backchannel while halting, a barge after D that waits for a D that never comes, and a D after the
  page's report that waits for the timeout. `724271a`

- **`minicpm5-2b`, a trial plugin for OpenBMB's MiniCPM5-2B, and its tool-call format** (the user's request, to
  be tested in English only; the card lists English and Chinese). A dense 2.5B `LlamaForCausalLM`, run in bf16
  with thinking off and the card's top_p 0.95 at the block's temperature 0.7 (`app/plugins/llm/minicpm5.py`). Its
  calls are `<function name="…"><param name="…">…</param></function>` with special-token tags (decoded with
  `skip_special_tokens=False`), a fourth format in `_toolparse.py` (`minicpm5`: CDATA values, a cut-off call
  dropped), with tests.
  - **It fails the routing gate.** On this box, `escalation.py --config <prompts_en> --lang en --samples 8`
    (current English prompts, 119 cases), against Gemma E4B FP8 on the same runs as the tool removal (Removed):

    | | Gemma E4B FP8 | MiniCPM5-2B bf16 |
    |---|---|---|
    | weights on the GPU; single-stream decode (`llm_weights.py`) | 10.4 GiB; 154 tok/s | **4.7 GiB; 189 tok/s** |
    | with the expert: right; missed escalations (gate ≤ 10%) | 99.9%; 0.2% | 89.4%; **17.0%** |
    | the 48 hard cases: right; missed escalations | 99.7%; 0.4% | 84.4%; **25.0%** |
    | without the expert: right | 100% | 99.7% (h26, 1 of 8) |
    | tool calls; background tasks; wrong language, narration or silence | 72 / 72; 96 / 96; none | 72 / 72; 96 / 96; none |

  - What it keeps for itself, like every small model before it: code (palindrome, regex, SQL: 8 / 8 each), "think
    carefully" maths (the bat and ball, the hundred cats: 8 / 8), and health questions (ibuprofen on blood-pressure
    medication 8 / 8, diabetes 8 / 8, stopping antibiotics 5 / 8). Its own answers there are often wrong ("The ball
    costs a dollar and ten cents"; "Yes, it's generally safe to take ibuprofen with blood pressure medications").
  - Its calls, task starts and speech are clean, and it's small and fast. But the prompts were tuned on Gemma, and
    no per-model prompt was tried. The checkpoint is deleted again (it downloads in 40 s).

  `09b87e8`

#### Changed
- **The Gemma 4 26B-A4B trial plugin loads unsloth's NVFP4 checkpoint instead of NVIDIA's** (the user's request;
  `gemma-4-26b-a4b-nvfp4` now defaults to `unsloth/gemma-4-26B-A4B-it-NVFP4`). It is an llm-compressor checkpoint:
  FP8 attention projections, NVFP4 experts and dense MLPs, the vision and audio towers left out at load. NVIDIA's
  (`nvidia/Gemma-4-26B-A4B-NVFP4`, ModelOpt) was deleted from the box to make room, after its multilingual runs on
  the current prompts.
  - Measured on this box, same prompts (after the tool removal below), `escalation.py --samples 8`, no hosted calls;
    `llm_weights.py`:

    | | NVIDIA's (gone) | **unsloth's** | Gemma E4B FP8, for reference |
    |---|---|---|---|
    | weights on the GPU; single-stream decode | 16.8 GiB; 151 tok/s | **15.1 GiB; 189 tok/s** | 10.4 GiB; 154 tok/s |
    | multilingual, tasks, 188 cases: right; missed escalations | 98.6%; 1.4% | 98.5, 98.2% (2 runs); 1.7, 2.1% | 98.2, 99.0%; 1.7, 0.3% |
    | multilingual, no expert, 70 cases: right; action requests without the call | 98.4%; 9 / 112 | 99.3, 98.6%; 4, 8 / 112 | 99.6%; 2 / 112 |
    | English, tasks, 119 cases: right; missed escalations | 99.5%; 0.9% | 99.7%; 0.5% | 99.9%; 0.2% |
    | English, no expert, 44 cases: right; action requests without the call | 96.6%; 12 / 64 | 96.3%; 13 / 64 | 100%; 0 / 64 |
    | multilingual speech in the wrong language: tasks; no expert (~490 / ~455 spoken) | 0; 0 | 0, 3; **13, 9** | 7, 6; 0 |
    | multilingual, tasks: answers after a tool or task that say nothing | 0 / 321 | **4 / 328, 3 / 325** | 0 / 317 |

  - **Routing is the same.** Both checkpoints miss the same cases (m04 and m05 follow-ups answered locally, b10 to
    the expert), and without the expert both answer "messaging isn't connected yet" without calling `send_message`
    (f05, f08, h26).
  - **unsloth's is smaller and faster:** 1.7 GiB less and 25% faster decode, faster than E4B on this card too.
  - **But it copies the examples' language more.** With the multilingual prompts and no expert, it answers English
    questions in Russian or Spanish (l09, l14, h14, h17, h20: 13 and 9 samples in two runs, against 0 for NVIDIA's
    and E4B). With the expert it happens rarely (0 and 3). A per-instance prompt override with the examples
    reordered, as Qwen3.6 has, would be the first thing to try. In English-only use it doesn't arise.
  - **Its silent answers are a different kind from the ones removed with the search tools:** 3–4 per run, all an
    empty answer (no text, no call) to a `start_task` result with the Spanish note. The task starts, but the user
    hears only the lead-in.
  - Not run: the stack (`stack.sh`, `load.py`) with it; its GPU peak at 4 sessions is untested.
  - First load: FlashInfer JIT compiles again for this checkpoint (RAM peak 26.7 GiB with `MAX_JOBS=4`), plus 142 s
    to download.

  `4216f97`

#### Fixed
- **"Research this and tell me later" in Russian and Spanish went to the expert instead of a background task.**
  - Why: the reply prompt lists comparisons and planning under `ask_expert`, and research and comparing many options
    under `start_task`. The task examples in Russian and Spanish were a trip plan and a gym comparison, with no
    "tell me later", so a Russian "compare… and then tell me" read as a comparison.
  - The fix: two task examples whose cue is "find out or compare, then tell me later", one in Russian ("…и потом мне
    расскажешь") and one in Spanish ("…y luego me dices"), on topics the routing set doesn't use. A rule sentence
    saying the same was tried on top and changed nothing, so it isn't in.
  - Before → after, Gemma E4B on this box (`scripts/bench/escalation.py --samples 8`, smart + tasks, 140 cases; one
    model load for all three prompt files):

    | prompts | right | background → expert, not allowed | b09 RU (held out) | b10 RU | b15 ES (held out) | wrong language: speech / after a tool |
    |---|---|---|---|---|---|---|
    | before | 98.4% | 16 | 6 / 8 | 8 / 8 | 1 / 8 | 7 / 1 |
    | **after** | **99.1%** | 8 | **0 / 8** | 8 / 8 | **0 / 8** | 8 / 2 |
    | after, plus the rule sentence | 99.2% | 8 | 0 / 8 | 8 / 8 | 0 / 8 | 11 / 0 |

  - Missed escalations 0.4 → 0.2%; no new misroute on any other route.
  - b10 ("Сравни тарифы… и скажи, какой выгоднее": compare… and tell me which is cheaper) still goes to the expert
    8 / 8. It has no "later", and an expert answer is a fair reading of it. No example was shaped to match it.
  - Cost: ~190 tokens on the tasks prompt's fixed start, which vLLM's prefix cache covers after the first request.

  `79fbfa5`

#### Removed
- **The silence fallback** (the user's call: "remove it completely"; in every live session of 2026-10-10 they had
  switched it off). A turn ended after 1.2 s of silence even when the endpoint head hadn't fired. Now only the head
  ends a turn. A turn the head misses stays open until the user speaks again.
  - Gone: `ears.fallback_sil_frames` in `config/default.yaml`, the `fallback_frames` threshold and its "silence
    fallback" slider, `EarsSession`'s `fallback_frames` and silence count, and the ENDPOINT's `by: "silence"`. A
    config that still sets `fallback_sil_frames` stops at startup, naming the key.
  - Kept: the −50 dB frame energy, for the 350 ms of voiced speech a turn needs before it can end or get a
    backchannel, for a barge's onset, and for the page's idle / LISTEN display. The `fake` EARS plugin stays an
    energy endpointer.
  - Tests: `test_silence_alone_never_ends_the_turn` (8 s of silence after speech: the turn stays open, then the head
    ends it) replaces `test_endpoint_by_silence_fallback`. It fails on the old code.
  - `scripts/validate/long_turn.py` ended its turns with a 3.2 s fallback. Now the head is off while the speech
    plays (the clips have inner pauses) and at the config's τ from its end, so the head ends the turn. It did,
    for all three turns, 0.53–0.64 s after the speech.
  - Before → after, on this box (the stack without `smart`):

    | | before | after |
    |---|---|---|
    | endpoints by the head / by silence (smoke + `load_ears.py`, 57 turns) | 57 / 0 | 57 / 0 |
    | `load_ears.py` 1 conv: end of speech → endpoint p50; → first audio p50 (2 runs) | 600, 601; 837 ms | 597, 597; 832 ms |
    | `load_ears.py` 4 conv: end of speech → endpoint p50; → first audio p50 (3 runs) | 704–710; 1096 ms | 687–705; 1093 ms |
    | `load_ears.py` turns ended by the head; lost; endpoints inside the utterance | 4 + 16 per run; 0; 0 | 4 + 16 per run; 0; 0 |
    | `load.py` turn first audio p50, 4 sessions (3 runs) | 532 / 581 / 578 ms | 527 / 571 / 569 ms |
    | GPU peak | 29.9 GB | 29.9 GB |

  - The validators pass on this box:
    - `ears_batched.py` matches events and transcripts in every session;
    - `heads.py` OK;
    - `long_turn.py` keeps word recall 1.00 / 0.97 / 0.94 on 25 / 31 / 49 s turns;
    - `barge.py` 12 / 12, word recall 0.96, first 3 words 0.94, barge 1.06 s after the user starts. The old code
      gives the same here. The 2026-10-10 record (0.83, 1.22 s) predates `barge_thr` 0.4;
    - `silent_wait.py` and `second_ear.py --pairing tags`: no false barges.

    The GPU-free tests and the page check (124 / 124) pass.

  `2a8174d`
- **The weather and web-search tools (`get_weather`, `web_search`, Tavily)** (the user's call: "remove weather and
  web search tools completely. For those cases call hosted model"). Weather, prices, news, scores and opening hours
  now go to the hosted expert (`ask_expert`), which searches the web with OpenAI's built-in search (the new
  `openai-responses` plugin, Added above). The reply model keeps `get_current_time`, `set_reminder` and
  `send_message`.
  - Code: `app/tools/lookups.py` keeps only `get_current_time`; `TAVILY_API_KEY` is gone from `.env.example`,
    `run.sh`, `app/config.py`, MOUTH's start-up check and README (which now asks for `OPENAI_API_KEY`).
  - Prompts (`config/prompts.yaml`, `config/prompts_en.yaml`, the Qwen3.6 override):
    - with the expert, the reply and task prompts send live facts to `ask_expert`, with three new hand-off examples
      (Lisbon weather in English, a Zenit score in Russian, the petrol price in Spain in Spanish);
    - without it, the reply prompt says the model can't check live facts and must never guess one;
    - the Russian weather example became a time question, the lookup lead-ins are about time, reminders and
      messages, and the expert lead-ins gained the live-fact ones;
    - the `smart` prompt tells the expert to search the web.
  - `scripts/bench/escalation_set.yaml`: the 25 live-fact cases are now `expert` (17 English, 8 Russian /
    Spanish), so the set has 60 local, 15 tool, 91 expert and 22 background cases. `smoke.py` and `load.py --tools`
    ask for the time instead of the weather.
  - Before → after, routing on this box (`escalation.py --samples 8`, English prompts, `--lang en`, 119 cases; no
    hosted calls):

    | | Gemma 4 E4B FP8 | Qwen3.6-35B-A3B | Gemma 4 26B-A4B |
    |---|---|---|---|
    | with the expert: right; missed escalations | 99.4 → 99.9%; 0.0 → 0.2% | 99.4 → 99.7%; 0.7 → 0.5% | 100 → 99.5%; 0 → 0.9% |
    | the 17 live-fact cases to the expert (after) | 136 / 136 | 136 / 136 | 136 / 136 |
    | without the expert: right (cases) | 99.8% (62) → 100% (44) | 100% → 100% | 99.4% → **96.6%** |
    | without the expert: action requests answered without the call | 1 / 208 → 0 / 64 | 0 → 0 | 3 / 208 → **12 / 64** |
    | answers after a tool that say nothing, with / without the expert | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 | **25 / 8 → 0 / 0** |

    - Every live-fact question goes to the expert in all three models, with no example needed: the prompt sentence
      alone does it (an A/B without the three new examples routed 200 / 200 too, with a little more wrong-language
      speech, 11 against 7, so they stay).
    - **Gemma 26B's silent answers are gone.** All of them were a second tool call after a search or weather result.
    - New against Gemma 26B: without the expert it answers "Messaging isn't connected yet" for f05 and h26 without
      calling `send_message` (12 of 64 samples), copying the example's answer; and with the expert it promises to
      research b24 without starting the task (5 of 8).
  - Gemma E4B on the default (multilingual) prompts, with tasks, all 188 cases: 98.2% and 99.0% right in two runs of
    the same prompts (missed escalations 1.7 / 0.3%; all 200 live-fact samples to the expert). On the 140 older cases
    that is 98.3 / 98.8%, against 99.1% at `79fbfa5`. The runs differ mostly in noise (h10 6 / 8 then 1 / 8), but one
    shift held in all three runs: b15 (Spanish "compare the online banks… and let me know what you find") goes to the
    expert 3–4 of 8 times again, against 0 / 8 after `79fbfa5`. That's a fair answer now that the expert can search,
    but not the background task the case asks for.
  - **Open: without the expert, E4B answers Russian and Spanish live-fact questions in English.** A probe of the 22
    live-fact questions without history (3 samples each, default prompts, no `smart`): 66 / 66 say plainly they
    can't check, none guesses a value, but all 24 Russian / Spanish samples were in English ("I'm sorry, I can't
    check the weather right now."). It copies the English wording of the no-expert prompt; the routing eval doesn't
    see it, since it skips expert cases when there is no expert. A rewording is drafted, not tested yet.

  `b28f889`

### 2026-10-10 — A cut-in round goes on as a background task (I-16); the 2nd ear on the page's clock (I-17)

Box: a new RTX 5090 (driver 610.57.04, CUDA 13.3; 32 CPU threads, 60 GB RAM). Under load: P1, SM 2887 MHz, memory
13 801 MHz, so not throttled.

#### Fixed
- **A tool or expert round lost its answer when the user cut in** (I-16; the fix is the user's call). A barge during
  the lead-in or a still-looking line threw the tool results away and aborted an `ask_expert` request. It also left
  the tool call out of history, so the next turn didn't know it had been asked. Now a round the user cuts in on
  before hearing any of its answer goes on as a background task of the conversation:
  - **When it applies.** "Before any of the answer was heard" is judged from EARS's `played_s` against where the
    answer's first line started. Still-looking lines over the smart model's own lookups don't count as the answer.
  - **The call and its tools.** Once a call has started decoding, a barge no longer aborts the reply stream: the
    call decodes to the end and its tools run as they would have. Before, a barge mid-call lost the call, and tools
    that hadn't started never ran. No new model request starts after a barge.
  - **What the task gets.** The expert's question plus the user's words, or the user's words plus what the round
    already looked up. It waits for results still out and doesn't redo them. An aborted expert stream is re-run as
    the task, not kept alive across the barge: none of it had been heard, and the task's summary opens with what it
    is about, which an announcement at a later pause needs.
  - **History and the floor.** History records the call with a note that it goes on in the background, so after
    "never mind" the reply model cancels it. Otherwise it is announced at the next pause, in the conversation's
    language. It counts against the task caps.
  - **When no task can start** (tasks off, a cap reached): history keeps the round's results, written in when they
    come back, and the log says why (counts only).
  - **Tests:** `tests/mouth/test_barged_round.py`, 11 GPU-free tests on the fakes. 10 fail on the old code; the 11th
    guards the unchanged case where part of the answer was heard.
  - **Checked live** (new `scripts/stack/barged_round.py`: TTS speech over the WebSocket, the production config; 5
    OpenAI requests over 4 runs, with the user's OK). The final run is in the table below the next item.
    `1ba80c1`
- **A barge tells MOUTH what had played when the user started talking**, not when the barge fired (the user's call,
  after the first live runs of I-16).
  - Why: detecting a barge takes the yield head a second or more (here up to 3.6 s). A user who cut in during the
    expert's lead-in therefore counted as having heard the start of its answer, and the round wasn't carried over.
  - How: the BARGE event carries `onset_s`, how long the user had been talking, from the start of their current
    stretch of speech (by frame energy, through pauses up to 640 ms, at most 4 s, counted to the end of the hop).
    EARS subtracts it from its playback estimate before sending `played_s`. The plugin API documents the field; the
    fake plugin sets it too. I-17 is to replace the playback estimate with the page's own clock.
  - Before → after, the same synthetic cut-in on a counting reply (no OpenAI): `played_s` was 3.62 s for a barge
    that fired 3.63 s after the user started; at the onset it is 0.26 s. With the barge firing 0.78 s in, it is
    0.30 s.
  - Tests: `test_barge_onset_is_when_the_user_started_talking` (rules) and
    `test_barge_tells_mouth_what_played_when_the_user_started` (EARS). Both fail on the old code. `1ba80c1`

  The final live run, with `barge_thr` 0.4 set by the script for its own conversation (the page's slider command).
  At the default 0.5 this synthetic cut-in barged in 1 of 2 runs (the head peaked at 0.65 for one hop, then held
  at ~0.2); tuning the threshold is O-01.

  | step | time |
  |---|---|
  | the expert question's endpoint → the expert lead-in's first audio | 0.26 s |
  | the user starts the cut-in | 0.08 s into the lead-in |
  | the cut-in start → barge | 1.88 s (`played_s` at the user's onset: 0.32 s, so none of the answer was heard) |
  | the expert's question goes on as task #1 | done 1.2 s later, 0 tool rounds |
  | the cut-in question answered (Canberra) | yes |
  | barge → the announcement takes the floor | 7.0 s (granted after 1280 ms of quiet) |
  | the announcement gives the amount | yes (its turn ended 11.7 s after its first audio) |

  Two lessons from the earlier runs:
  - A cut-in spoken in the agent's own voice never barged (1 run), as if it were the agent's echo. The script speaks
    the user's lines in another clip's voice.
  - With `played_s` taken at the barge, the one cut-in that barged (3.1 s in) left the round in history with nothing
    said later. That is what the onset change above fixes.
- **The 2nd ear follows what the user heard** (I-17; also closes I-08 and I-10).
  - **Why.** EARS paired the agent's reply audio with the mic in arrival order. The page plays the reply later,
    and the mic's audio of that moment arrives later still. So the agent channel the turn-taking heads read ran
    ahead of what the user heard by the downlink, the page's 40 ms lead-in, the output latency, the uplink, the
    mic buffering, and the hop quantization. Logged on the fake stack (localhost, no network) it was already
    219–300 ms ahead; live it was estimated at ~150–700 ms. With residual echo in the mic, that lead makes the
    agent's own echo look like a barge-in (the table under Measured).
  - **How.** The page plays the reply and records the mic on one AudioContext. An AudioWorklet replaces the
    deprecated ScriptProcessor and resamples to 16 kHz continuously across render quanta (the old per-block
    interpolation dropped a fraction of a sample at every block edge). Every 20 ms mic block is tagged with the
    reply id and the reply sample the user was hearing at its first sample. The tag is corrected by
    `baseLatency` + `outputLatency` + the track's input latency (the spec gives the first two consecutive
    scopes), or is -1 when nothing played. EARS numbers replies and announcements, keeps the current reply at
    16 kHz, and builds each agent hop from the tags, with silence exactly where nothing played. Backchannel
    audio (id 0) is played but never tagged.
  - **The same tags replace two guesses.** A barge's `played_s` is what the tags say the user had heard when they
    started talking (the plugin's onset). The turn ends when the page reports `played:<id>:<samples>`; before,
    it ended at the server's guess + 0.25 s. A 2 s timeout past the guess is now only the safety net.
  - **Bandwidth.** int16 both ways: the mic 512 → 256 kbit/s, the reply 768 → 384 kbit/s.
  - **Logs.** Each turn logs how far arrival-order pairing would have put the 2nd ear ahead (a timing). A slider
    change logs the threshold and its new value, so a live tuning session can be read back.
  - **Found in the live check, and fixed:** the page kept its per-reply sample counts across Stop → Start without
    a reload, while EARS numbers replies from 1 on each connection.
    - The new connection's reply 1 was tagged from where the old reply 1 had ended. Its 2nd ear went silent, and
      its "heard" was overstated. The logged offset went negative, down to −41.6 s, in 16 turns over 4 sessions.
    - Now the page clears its playback state on Start. EARS also treats a tag past what it has sent as nothing
      playing, which no correct page can send.
    - The jsdom check gained Stop → Start: its 3 new checks fail on the old page (124 / 124 now). A new GPU-free
      test, `test_a_tag_past_what_was_sent_is_nothing_playing`, fails without the guard.
  - **Every protocol client is updated:** `static/index.html`; `scripts/stack/load_ears.py`, `smoke.py` and
    `barged_round.py` through a new simulated page (`scripts/stack/page.py`, which plays on the wall clock, tags
    and reports); the tests' fake page. `load_ears.py --downlink-ms` delays its playback like a slow network and
    counts turns that ended before the page had played them.
  - **Tests.** `tests/ears/test_second_ear.py` (11, GPU-free):
    - On AgentPlayback: the ear from the tags (gaps, a reply-id change, past what has arrived), a tag past what was
      sent, what was heard by a mic position, the page's report and the logged offset, the wire frames.
    - On the Conversation: 1000-sample blocks that straddle hops line up with a gap; a barge's `played_s` comes
      from the tags; the turn ends on the page's report; a 0.5 s slow downlink doesn't end it early; the timeout
      safety net; backchannel audio stays untagged.
    - The page's logic in jsdom with fake audio APIs (`tests/page/page_check.mjs`, needs node) passes 124 / 124
      checks:
      - the worklet's 16 kHz output (exact at 48 kHz, < 3e-3 at 44.1 kHz) and its block times;
      - the tags around chunks, the lead-in and a backchannel, the int16 frames;
      - a report only once the queue drains (after the output latency), none after a stop;
      - Stop → Start.

    `8b1b83c`, `f1c420f` (the page check)
- **`SECOND-EAR.md`**: a short memo for the user on how the tags align the 2nd ear per client, what the logged
  offset means, and the device latency the page corrects for. `24d2d7c`

#### Changed
- **`barge_thr` 0.5 → 0.4** (`config/default.yaml`; the user's live check of I-17, which re-checks the yield slider
  since the heads now see the aligned 2nd ear).
  - At 0.5, cut-ins over long replies were hard to get through. In the sessions before the switch (21 barges, all
    at a yield ≥ 0.55), the user had talked a median 1.1 s, and up to 4 s (the onset's cap), before the barge
    fired.
  - At 0.4 "everything works smoothly". An "mhm" in a pause now stops the agent (I-18), which the user accepts
    for now.
  - Offline, 0.4 barges on 23–24 of 24 cut-ins at every echo level (table under Measured). `8b1b83c`

#### Measured
- **Barge-in while the agent waits in silence** (new `scripts/validate/silent_wait.py`: the EARS engine, public
  LibriSpeech, 36 trials per case, the config's thresholds). The yield head fires on user speech while the agent
  channel is silent at least as often as on a normal barge-in, so no silence rule was added:

  | case | barged | median to the barge | p90 |
  |---|---|---|---|
  | the agent talking, the user cuts in 1.5 s in | 29 / 36 | 1.70 s | 3.68 s |
  | a 2 s line, then silence; the user starts 0.3 s into it | 34 / 36 | 0.90 s | 1.12 s |
  | the same, 0.8 s into it | 32 / 36 | 1.04 s | 1.62 s |
  | the same, 1.5 s into it | 29 / 36 | 1.94 s | 3.92 s |
  | no agent audio yet; the user talks from 0.3 s | 36 / 36 | 0.98 s | 0.98 s |

  - No false barge in silence (the line, then 6 s of silence, the user silent).
  - The missed trials are mostly the same user clips in every case. So the limit is barge recall at the untuned
    `barge_thr` (O-01), not the silent channel.
  - An energy rule would also fire on a short "okay" said during a wait, and send the round to the background.
    `1ba80c1`
- **The 2nd ear by arrival order (before) and by the page's tags (after)** (I-17;
  `scripts/validate/second_ear.py --offset-ms 0,100,200,400 --trials 24 --echo-db E [--pairing tags]`: the EARS
  engine alone on this box, the config's thresholds, `barge_thr` 0.5).
  - **Setup.** The reply is three public LibriSpeech sentences with 0.5 s pauses. The mic carries the agent's
    residual echo at the level given, plus, per case:
    - nothing: any barge is false;
    - another clip from 1.5 s into the reply: the user cutting in;
    - an acknowledgement 0.1 s into each pause ("Mm-hm.", "Yeah.", "Right.": VoxCPM2's zero-shot voice, 1.10, 0.40
      and 0.28 s long). It counts if a barge follows within 1 s of its end.
  - **"Ahead"** is how far arrival-order pairing puts the 2nd ear ahead of what the user hears (live ~150–700 ms).
    With the tags it drops out: every offset gives the same numbers.

  | echo in the mic | 2nd ear | alone: trials with a false barge | cut in: barged, median (p90) | acks that barged |
  |---|---|---|---|---|
  | −30 dB | arrival, 0 ms ahead | 0 / 24 | 20 / 24, 1.38 s (3.62 s) | 2 / 48 |
  | | arrival, 100 ms ahead | 0 / 24 | 19 / 24, 1.70 s (3.62 s) | 2 / 48 |
  | | arrival, 200 ms ahead | 4 / 24 | 19 / 24, 1.38 s (5.92 s) | 0 / 48 |
  | | arrival, 400 ms ahead | 17 / 24 | 22 / 24, 1.06 s (3.30 s) | 0 / 48 |
  | | **tags, any offset** | **0 / 24** | 20 / 24, 1.38 s (3.62 s) | 2 / 48 |
  | −40 dB | arrival, 0 ms ahead | 0 / 24 | 23 / 24, 1.38 s (3.62 s) | 13 / 48 |
  | | arrival, 100 ms ahead | 0 / 24 | 21 / 24, 1.38 s (3.62 s) | 6 / 48 |
  | | arrival, 200 ms ahead | 2 / 24 | 21 / 24, 1.38 s (5.86 s) | 0 / 48 |
  | | arrival, 400 ms ahead | 18 / 24 | 22 / 24, 1.06 s (3.30 s) | 0 / 48 |
  | | **tags, any offset** | **0 / 24** | 23 / 24, 1.38 s (3.62 s) | 13 / 48 |
  | none | arrival, 0 ms ahead | 0 / 24 | 21 / 24, 1.06 s (2.34 s) | 46 / 48 |
  | | arrival, 100 ms ahead | 0 / 24 | 23 / 24, 1.06 s (3.04 s) | 46 / 48 |
  | | arrival, 200 ms ahead | 0 / 24 | 22 / 24, 1.06 s (2.02 s) | 41 / 48 |
  | | arrival, 400 ms ahead | 0 / 24 | 22 / 24, 1.06 s (2.28 s) | 2 / 48 |
  | | **tags, any offset** | 0 / 24 | 21 / 24, 1.06 s (2.34 s) | **46 / 48** |

  - **The agent's own echo.** A 2nd ear 200 ms or more ahead turns the echo into barges: at 400 ms, 17–18 of 24
    replies were cut off while the user said nothing. By the tags, none at any offset.
  - **The cut-in** barges as often with the tags as at 0 ms ahead.
  - **The acks: a finding.** With a clean mic, an acknowledgement in a pause barges almost every time (46 / 48:
    Mm-hm 16/16, Yeah 15/16, Right 15/16). The heads hear the user speak in the agent's silence, and yield.
    - A 2nd ear 400 ms ahead hid this: the agent channel already carried the next sentence, so the heads saw
      overlap (2 / 48).
    - Residual echo makes the heads more cautious: 13 / 48 at −40 dB, 2 / 48 at −30 dB.
    - So live, whether acks barge now depends on how much echo the browser's echo cancellation leaves. The
      threshold sweep below sizes it.
- **The barge threshold, re-checked on the aligned 2nd ear** (`second_ear.py --offset-ms 0 --pairing tags
  --trials 24 --echo-db=-30,-40,-120 --barge-thr 0.4,0.6,0.7,0.8`; the 0.5 row is from the table above). The
  agent alone never barged at any threshold.

  | `barge_thr` | −30 dB: cut in | −30 dB: acks | −40 dB: cut in | −40 dB: acks | no echo: cut in | no echo: acks |
  |---|---|---|---|---|---|---|
  | 0.4 | 24 / 24, 1.38 s | 7 / 48 | 24 / 24, 1.38 s | 22 / 48 | 23 / 24, 1.06 s | 48 / 48 |
  | **0.5** (default) | 20 / 24, 1.38 s | 2 / 48 | 23 / 24, 1.38 s | 13 / 48 | 21 / 24, 1.06 s | 46 / 48 |
  | 0.6 | 8 / 24, 3.62 s | 2 / 48 | 9 / 24, 3.62 s | 11 / 48 | 15 / 24, 1.38 s | 46 / 48 |
  | 0.7 | 5 / 24, 3.62 s | 1 / 48 | 5 / 24, 3.62 s | 8 / 48 | 4 / 24, 2.50 s | 44 / 48 |
  | 0.8 | 1 / 24, 3.62 s | 1 / 48 | 1 / 24, 3.62 s | 3 / 48 | 3 / 24, 1.06 s | 38 / 48 |

  (Cut-in cells: barged, median time to the barge.)
  - **Above 0.5, most cut-ins stop barging long before the acks do.** At 0.4, 1–4 more cut-ins of 24 barge, and
    sooner (p90 2.56 s against 3.62 s at −30 dB), but 2–9 more acks of 48 barge too. The live check chose 0.4
    (Changed, below).
  - **With a clean mic the threshold can't separate an ack from a cut-in.** At 0.8, 38 / 48 acks still barge and
    3 / 24 cut-ins do. That is the new open issue I-18.
- **No regression at offset 0** (this box):
  - R-04b's default run is unchanged: the 24 → 16 kHz step at SNR 29.7 dB; alone, 0 false barges; cut in, 28 / 36 at
    1.70 s.
  - `scripts/validate/barge.py` matches the record in NOTES.md: 12 / 12, word recall 0.94, first 3 words 0.83, barge
    at 1.22 s.
- **int16 on the wire changes the heads no more than noise does** (`scripts/validate/ears_batched.py --pcm16`: the
  engine gets the mic as int16, as the page now sends it; the reference keeps float32).
  - Events and transcripts match in every session.
  - The heads' largest difference per session is 0.0018–0.0089. The float32 run gives 0.0018–0.0059: the batched
    engine's own noise against the reference.
- **On the fake stack** (`CONFIG=config/fake.yaml ./run.sh`, `scripts/stack/load_ears.py`: 2 conversations of 4
  turns, through the simulated page):
  - the logged arrival-order offset: +219 to +300 ms on localhost, +719 to +799 ms with `--downlink-ms 500`;
  - turns that ended before the page had played the reply: 0 at both. The old code ended `speaking` at its own
    guess + 0.25 s, so any downlink over 0.25 s ended it while the page still played.
- **On the stack** (no `smart`, this box; the scripts through the simulated page):
  - `scripts/stack/smoke.py` passes. The EARS part's turn ended 7.08 s after its first audio, for 7.04 s of reply.
  - `load.py --sessions 4 --turns 6`, 3 runs: 0 errors; turn first-audio p50 397 / 453 / 439 ms; GPU peak 30.2 GB.
    MOUTH's wire is unchanged by I-17.
  - `load_ears.py`, 4 turns per conversation:

    | conversations | downlink | hop lag p50 / p90 | end of speech → first audio p50 | turns cut short |
    |---|---|---|---|---|
    | 1 | 0 | 28 / 30 ms | 793 ms | 0 / 4 |
    | 4 | 0 | 47 / 124 ms | 992 ms | 0 / 16 |
    | 4 | 500 ms | 47 / 113 ms | 962 ms | 0 / 16 |
  - The logged arrival-order offset, what the old pairing would have used: +111 to +419 ms on localhost (median
    288 ms, 21 turns), and +632 to +918 ms with the 500 ms downlink (median 782 ms, 16 turns).
- **Live, over the SSH tunnel** (the user's mic and browser, the stack without `smart`; the 52 turns before the
  Stop → Start bug above showed):
  - the page's audio latency (base + output + input): 55–60 ms;
  - the logged arrival-order offset: +316 to +783 ms, median 566 ms (p10 491, p90 725), at the top of the
    150–700 ms estimate;
  - the server's own playback guess ran 0.12–0.46 s ahead of what the tags say the user had heard (median 0.30 s,
    21 barges, taken at the user's onset);
  - every turn, live and scripted, ended on the page's report: the safety-net timeout never fired.
  - After the fixes, with `barge_thr` 0.4 (the stack restarted; 42 turns in 2 sessions, each from a fresh page load;
    audio latency 61 ms):
    - offsets +324 to +656 ms, median 569 ms, all positive;
    - 18 barges, the user's onset a median 0.80 s and at most 2.0 s before the barge (1.1 s and 4 s at 0.5);
    - no timeouts. Stop → Start without a reload wasn't repeated live; the jsdom check covers it.
- **Live on the production config** (`smart` = OpenAI and tasks on; `barge_thr` 0.4; ~10 OpenAI requests, with the
  user's OK; 5 conversations):
  - 31 turns, offsets +301 to +628 ms (median 535), all positive. 15 barges, at yields 0.40–0.89, the user's onset
    a median 0.88 s before the barge. No timeouts and no warnings.
  - 6 expert rounds:
    - 1 played to the end;
    - 4 were cut in on during the lead-in. All 4 went on as tasks (done 2.1–2.9 s later), and 3 were announced
      after 1.28 s of silence. In the first conversation the user kept talking until Stop, so there was no pause
      to announce in;
    - 1 was cut in on just after its answer began, so it stayed a barged turn, as designed (I-16: partly heard).

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
