# tyger v1 — build notes

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
