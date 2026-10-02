# tyger v1 — build notes

## Box (2026-10-02)
NVIDIA RTX 5090 (32 607 MiB, Blackwell sm_120), driver 595.91 / CUDA 13.2; 96-core CPU. `HF_HOME=/workspace/.hf_home`,
`HF_HUB_DISABLE_XET=1`, torch **2.13.0+cu130**, Python 3.12, one venv `.venv` (vllm 0.30.0, vllm-omni 0.30.0,
voxcpm 2.0.3, transformers 5.14.1).

**This box's GPU is throttled**: it never leaves P3 — memory clock 7001 of 14001 MHz, SM 2385 of 3412 MHz,
~200 W under load, no throttle reason reported (host-level lock). Qwen3-4B decodes at **92 tok/s** here vs
~150 on the previous box with the same vLLM. **Every latency number below from this box is pessimistic**;
re-measure on a healthy GPU before tuning anything against them.

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

MOUTH handles one job at a time: turns (`T`), backchannels (`C`) and voice switches (`V`) go through one queue
in `serve()`, so a backchannel never runs concurrently with a turn, even though the engines could batch
(`max_num_seqs` 4 / 2).

## How a backchannel works
1. **EARS decides when.** In listen mode, every 80 ms frame, the bc head gives `bc_prob`. A `BACKCHANNEL` event
   fires when `bc_prob ≥ ears.bc_thr` (0.5) for `bc_deb` (2) frames, once the user has ≥ 350 ms of voiced
   speech in this turn and EARS is past its ~0.5 s warm-up; then a refractory of 15 frames (1.2 s).
   `app/server.py` forwards it only if the agent isn't speaking, isn't halting after a barge, and no other
   backchannel is in flight.
2. **EARS attaches what the user is saying.** A rolling ASR partial is re-decoded from the last 8 s of user
   audio at most every 0.8 s, off the critical path; the event just reads the latest one. It can be empty early
   in a turn. EARS sends `C {"partial": ...}` to MOUTH.
3. **MOUTH asks the LLM for the ack** (`Mouth.stream_bc`): `BC_PROMPT` + few-shot examples (EN/RU/ES reactions,
   `mm-hm` / `угу` hums) + the last 3 acks ("don't repeat those") + the partial. A partial under
   `mouth.bc_min_words` (4) or empty is framed as "only a few words so far", so the model answers with a
   neutral hum; otherwise it reacts to the content, in the user's language. There are no canned acks.
4. **MOUTH speaks it**: one VoxCPM2 request for the ack text, streamed back as `A` frames, then `D {"bc": true}`.
   Typical first audio ~260–340 ms after the `C` (this box).
5. **EARS plays it through to the browser** while `bc_active` is set. Backchannel audio is not fed to the
   second ear and does not count toward the reply's `play_until`, and it never enters conversation history.

**Interaction with the user's turn.** If the user's endpoint fires while a backchannel is still being made,
EARS sends the turn at once, but MOUTH starts it only after the backchannel job finishes (one queue): the reply
can start up to ~0.3–0.5 s later. The client stops any still-playing ack when the endpoint arrives.

## Memory + memory pinning
Both engines pin their KV caches instead of filling a GPU fraction: VoxCPM2 1 GiB (`deploy/voxcpm2.yaml`, also
`max_num_seqs 2`, `max_model_len 2048`; stock deploy was 6 GiB / 8 seqs for multi-tenant serving), Qwen 2 GiB
(`llm.kv_cache_gib`, `max_num_seqs 4`, `max_model_len 8192`). `gpu_memory_utilization` is then only vLLM's
start-up free-memory check. Measured: VoxCPM2 7.4 GB loaded (+0.6 GB after warmup), Qwen ~11 GB, EARS ~4.8 GB;
**~24 GB / 32 GB** total warm (was ~30.5 GB with MOSS + vLLM server).

## V1.3 — smoke test on the box (cloned voice `en`, 9.3 s clip, no transcript)
- `scripts/smoke_test.py` **PASS**: plain first-audio **503 ms** (first LLM text at 92 ms), Tavily weather tool
  turn **743 ms**, placeholder reminder turn **770 ms** (admits not connected), EARS speech → endpoint → reply.
- Plain-turn breakdown: ~200 ms waiting for the 40-char first chunk at 92 tok/s + VoxCPM2 encoding the 9.3 s
  reference clip on every request (zero-shot first-audio was 113 ms). Lever: VoxCPM2's precomputed
  `voice_profile` mode (vLLM-Omni loads custom voice profiles from a dir) to skip the per-request clip encode.
- Logs grep-checked: no transcript or reply text in either process's output.
- Startup (warm cache): MOUTH ~108 s (VoxCPM2 55 s, Qwen 32 s, TTS warmup 21 s), EARS ~15 s. First-ever start
  adds downloads, VoxCPM2's one-time compile (~2 min warmup) and Qwen's CUDA-graph capture.
- EARS per-hop compute is p50 275 / p90 404 ms per 320 ms hop on this box (same in the old pinned venv) — p90
  exceeds real time here; re-check on a healthy GPU.

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
- Precomputed VoxCPM2 voice profile to drop the per-request reference encode.
- V2: retune `ep_thr`/`ep_deb` + barge/bc live with `?debug=1` (0.9146 was never retuned live). Freeze into `config.yaml`.
- V3: Docker build/run + container smoke.
- Re-benchmark everything on a healthy GPU.
