# tyger v1 — open issues

From a code review of branch `voxcpm2` at `0c1d8f7` (2026-10-03). The review read the code only; nothing was
run, since that needs the GPU box. Each issue says how sure it is:

- **confirmed in code**: the code path is unambiguous.
- **likely**: the code reads that way, but timing on the box decides how bad it is.
- **hypothesis**: measure before fixing.

IDs are stable; refer to them in commits (e.g. "Fix I-01").

| ID | Area | Severity | Status | One line |
|---|---|---|---|---|
| I-01 | MOUTH | high | **fixed** | Barged turns vanish from LLM history |
| I-02 | EARS | high | **fixed** | Start of a barge-in utterance is cut from the transcript |
| I-03 | MOUTH | medium | **fixed** | Barge during a tool call waits for the tool (up to 15 s) |
| I-04 | metrics | medium | **fixed** | "Onset" excludes endpoint detection; real gap is larger (it's ~1.0 s) |
| I-05 | EARS | medium | **fixed** | User turns > 20 s silently keep only the last 20 s |
| I-06 | MOUTH | medium | **not reproduced** (real cost elsewhere, fixed) | Voice clip copied as ~480k Python floats on every TTS request |
| I-07 | MOUTH | low | **fixed** | Backchannel keeps generating after the reply turn starts |
| I-08 | EARS | low | likely | 2nd-ear agent audio leads what the user actually hears |
| I-09 | EARS | low | confirmed in code | MOUTH dying leaves the conversation silently dead |
| I-10 | EARS | low | likely | `play_until` is a server-side guess at client playback |
| I-11 | security | low (demo) / high (public) | confirmed in code | No auth, no message-size or buffer limits on :7860 |
| I-12 | MOUTH | high | **fixed** | Tool calls never enter history, so later turns stop calling tools and invent answers |
| I-13 | metrics | medium | **fixed** | `scripts/stack/load_ears.py` runs don't compare: LLM-made speech and replies, random hop phase |
| I-14 | ops | low | confirmed on the box | `run.sh` giving up on a slow first start orphans the LLM engine on the GPU |
| I-15 | tests | low | confirmed on the box | Local contract test aborts at exit after the EARS engine ran in the process |
| L-01 | perf | lever | **partly done** | Encoder re-encodes the full 8 s window, twice, every hop, in fp32 |
| L-02 | latency | lever | — | Decisions quantised to 320 ms hops |
| L-03 | latency | lever | **done** | Per-request voice encode (precomputed voice profile) |
| L-04 | latency | lever | **done** | Tool filler (lead-in) starts only after the whole tool call decodes; one filler, then silence |
| O-01 | ops | — | — | Endpoint thresholds never tuned live (V2) |
| O-02 | ops | — | **blocked here** | Dockerfile never built |
| O-03 | ops | — | — | `main` is still at `First`; all work is on `setup/new-box` / `voxcpm2` |
| O-04 | ops | — | **done** | All latency numbers come from a throttled GPU |
| O-05 | ops | — | partly done | Little GPU-free test coverage; no CI |

---

## Correctness

### I-01 — Barged turns vanish from LLM history (high, confirmed in code)
**Where:**
- `app/mouth.py` `Session.stream_turn`: lines 214–215 for the plain path, 239–240 and 251 for the tool path.

**What happens:** `_remember(transcript, reply)` runs only when a turn completes. A turn stopped by barge-in
returns early, so neither the user's transcript nor the partial reply enters `self.history`.

Example: the user asks "what's the weather in Bishkek on Friday?", the agent starts answering, and the user
cuts in with "no, Saturday". The next LLM call sees only "no, Saturday". Barge-ins are routine in a
full-duplex system, so context loss is routine too.

**Fix:**
- On stop, still remember the user's transcript.
- Also remember the assistant text that was actually spoken, ideally truncated to roughly what had played and
  marked as interrupted (e.g. `"<partial reply> —"` or a system note "(interrupted by user)").
- MOUTH doesn't know playback position. Either remember the generated text (simple), or have EARS send the
  played duration with `B` so MOUTH can truncate to whole chunks that played.

**Verify:** a scripted session of turn, then barge, then follow-up. Assert that the follow-up's prompt
contains the first question. A unit test can drive `stream_turn` with a fake LLM/TTS and a stop flag.

**Status: fixed (2026-10-03).** Reproduced on the box: after a barged "octopuses" turn, "which animal were you
telling me about?" got "I wasn't telling you about any animal". Now the transcript + heard part of the reply (EARS
sends `played_s` with `B`) are remembered, marked with " —"; the follow-up answers "Octopuses". Tests:
`tests/mouth/test_session.py`, `scripts/stack/smoke.py` (MOUTH barge).

### I-02 — Start of a barge-in utterance is cut from the transcript (high, likely)
**Where:**
- `app/ears_core/batched.py:310-311`: `turn_u` grows only in `listen` mode.
- `app/server.py:213`: barge sets `reset`.
- `app/server.py:126`: `finish_turn` sets `reset` again.
- `app/ears_core/batched.py:306-307`: a reset runs `_clear()`, which empties `turn_u` and `win_u`.

**What happens:**
1. While the agent speaks, user audio is never appended to `turn_u`. That covers the user's speech onset and
   the hop where the yield head crosses `barge_thr` for `barge_deb` frames.
2. After `BARGE`, the next hop resets the session and starts a new `turn_u`.
3. When MOUTH's `D {"barged": true}` arrives, `finish_turn` sets `reset` again. The next hop clears `turn_u`
   once more, discarding everything recorded between the barge and that `D`.
4. Net: "No wait, I meant Saturday" can reach the LLM as "I meant Saturday" or less. The loss is roughly the
   barge hop plus the barge→`D` latency (the TTS abort plus a 5 ms poll, plus the I-03 tool wait if a tool
   was running).
5. Each reset also restarts `WARMUP_FRAMES` (6 frames, 480 ms) and the 350 ms voiced-speech minimum.

**Fix:**
- Keep a short rolling buffer of user audio in `speaking` mode too (e.g. the last 1–2 s).
- On `BARGE`, seed the new `turn_u` with it.
- Don't reset again in `finish_turn` when the turn was barged and EARS already reset on the barge. Reset only
  the head/agent state, not `turn_u`.

**Verify:** play a reply, inject user speech mid-reply (`scripts/stack/load_ears.py`-style), and compare the endpoint
transcript to the injected text. Check event parity with `scripts/validate/ears_batched.py` for the
non-barge path.

**Status: fixed (2026-10-03).** Confirmed: `scripts/validate/barge.py` (12 LibriSpeech barge-ins, engine-level, the
server's reset protocol) recalled only 14% of the utterances' first 3 words (69% of all words) with the old code.
Now the engine keeps the last 2 s of user audio while the agent speaks and seeds the barge's turn with it
(crediting its voiced frames), and `finish_turn` no longer resets again on a barged `D`: first 3 words 83%, all
words 94%. Barge fires a median 1.22 s after the user starts, so 2 s covers it. Dropping only the second reset gave
22% / 72%. `scripts/validate/ears_batched.py` parity unchanged.

### I-03 — Barge during a tool call waits for the tool (medium, confirmed in code)
**Where:**
- `app/mouth.py:237-238`: `for w in workers: w.join()`.
- `app/tools.py:52`: `urlopen(..., timeout=15)`.

**What happens:**
1. Tool threads are joined without checking `stop_flag`. A barge during a slow Tavily call blocks the turn
   worker for up to 15 s before it sends `D {"barged": true}`.
2. Until then EARS stays `halting`: the user's next endpoint is held in `pending` and not answered.

**Fix:**
- Join in a loop with a short timeout that checks `stop_flag()`, and leave the daemon threads to finish on
  their own.
- Alternatively, run tools as futures and `wait(..., timeout)` in a loop.

**Verify:** stub a tool that sleeps 10 s, barge during it, and assert `D` comes back in < 200 ms.

**Status: fixed (2026-10-03).** Barge → `D` on a live Tavily tool turn: 2.0–2.5 s before, 5 ms after. Unit test
with a 10 s tool: 9982 ms before, < 200 ms after.

### I-05 — User turns > 20 s keep only the last 20 s (medium, confirmed in code)
**Where:**
- `app/ears_core/batched.py:33`: `TURN_MAX_S = 20.0`.
- Line 311 truncates `turn_u`.
- Line 219 sizes the RNN-T graph to it.

**What happens:** the start of a long user turn is silently dropped from the endpoint transcript. No log or
event says so.

**Fix:**
- Option A: transcribe in segments. Finalise each ~15–20 s block on the ASR thread as the turn grows, and
  join the texts at the endpoint. This also cuts endpoint-time ASR latency for long turns.
- Option B: raise `TURN_MAX_S` and let `GraphedRNNT` fall back to eager decoding past `max_frames`. It
  already does this; it's just slower.
- Either way, log a count when truncation happens (counts only, never words).

**Verify:** a 30 s synthetic turn; the transcript covers all of it.

**Status: fixed (2026-10-03), option A.** Past 15 s (`SEG_S`) a listening turn is cut at the quietest 80 ms frame
of its last 4 s and that block is transcribed on the ASR thread right away; the endpoint transcribes only the tail
and joins. `scripts/validate/long_turn.py` (public LibriSpeech, 27 / 33 / 51 s turns): word recall 0.85 / 0.67 /
0.42 → 1.00 / 0.97 / 0.94, first 10 words 0.60 / 0.40 / 0.20 → 1.00; the endpoint hop (final ASR) stays 127–156 ms
at any length. The ENDPOINT event and the EARS log carry a segment count (no words).
`scripts/validate/ears_batched.py` now matches events by kind + frame and transcripts for turns ≤ 15 s (its single-session
reference still keeps only a turn's last 20 s).

### I-07 — Backchannel keeps generating after the reply turn starts (low, confirmed in code)
**Where:**
- `app/mouth.py:307`: `stop = self.closed.is_set`.
- `app/server.py:155-157`: EARS drops `K` once a turn starts.

**What happens:**
1. Once the endpoint fires, EARS discards backchannel audio, but MOUTH still finishes the ack's LLM call and
   TTS.
2. Under load that's a wasted VoxCPM2 batch slot (`max_num_seqs` 4) right when the reply's first chunk needs
   one.

**Fix:**
- Add a `bc_stop` event. Set it when `T` arrives, and on a new wire message if EARS wants to cancel
  explicitly.
- Use `stop = lambda: self.closed.is_set() or self.bc_stop.is_set()`.

**Verify:** in `scripts/stack/load.py`, fire the backchannel just before the turn. Check that the backchannel
request is aborted in the logs and that turn first-audio p50 doesn't regress.

**Status: fixed (2026-10-03).** Done with a per-session turn counter instead of a `bc_stop` event (a `C` sent
after `T`, as `scripts/stack/load.py` does mid-reply, still runs). `scripts/stack/load.py --bc-first` at 4 sessions: 12/12
backchannels aborted; turn first-audio p50 769 → 559 ms.

### I-08 — 2nd-ear agent audio leads what the user hears (low, likely)
**Where:** `app/server.py:166-170` and `244-246`.

**What happens:**
1. `agent_buf` is filled when audio arrives from MOUTH and drained one hop per user hop.
2. The browser starts playback after the network, a 40 ms lead-in and its own buffering. The user's
   reaction then comes back through mic capture plus the network.
3. So the agent channel the yield head sees is probably ahead of the real acoustic overlap by tens to
   hundreds of ms.
4. Separately, `ds24to16` uses `np.interp` per chunk: no anti-alias filter, and discontinuities at chunk
   edges.

**Fix:**
- Measure first: log `play_until` against hop time.
- Consider delaying the agent channel by an estimated client latency.
- Use a proper polyphase resampler, such as `scipy.signal.resample_poly` or a stateful one like MOUTH's.
- Check barge false-positive and false-negative rates before and after.

### I-09 — MOUTH dying leaves the conversation silently dead (low, confirmed in code)
**Where:**
- `app/server.py:185-186`: `pump_mouth` swallows the disconnect.
- `app/server.py:251-252`: `run_hops` swallows `ConnectionError`.

**What happens:** the WebSocket stays open, the client gets no message, and `speaking` may stay `True`. The
user just hears nothing.

**Fix:** on MOUTH disconnect, send `{"type": "error", ...}` to the client and close the WebSocket. Optionally
try to reconnect.

### I-10 — `play_until` is a server-side guess (low, likely)
**Where:** `app/server.py:168-169`, plus `PLAYBACK_MARGIN_S = 0.25`.

**What happens:**
1. If the client buffers longer than estimated (jitter, a backgrounded tab), EARS closes the turn while
   audio still plays.
2. Talking over that tail then becomes a new turn instead of a barge-in.
3. The client's `stopPlayback()` on `endpoint` prevents overlap, but cuts the reply.

**Fix:** have the client report its playback end (`nextPlay - ctx.currentTime`) with each audio ack, or send
a `played` message when its queue drains.

### I-11 — No auth, no limits on :7860 (low for the SSH-tunnel demo, high if public)
**Where:** `app/server.py:81` sets `max_msg_size=0` (unlimited); line 261 grows `buf` without a cap; there is
no auth on `/`, `/ws` or `/voices`.

**What happens:**
- Anyone who reaches the port can hold all 4 slots and spend Tavily credit.
- They can also grow memory without bound by sending audio faster than real time or in huge frames.
- `/voices` lists voice names.

**Fix before any public exposure:**
- Token or basic auth at the proxy or app.
- `max_msg_size` around 64 KB.
- Drop or close when `buf` exceeds a few seconds.
- A per-IP session cap.

### I-12 — Tool calls never enter history; later turns answer live questions from memory (high, found on the box)
**Where:** `app/mouth.py` `Session._remember`, called with only the final reply text.

**What happened:** the history kept `user` + final `assistant` text and dropped the tool call and its result. The
model then copied its own history: weather questions answered directly, without a tool, i.e. invented values.
Measured with `scripts/stack/load.py --tools` (6 weather questions per session, different cities): tool called on 2/6
turns at 1 session and 9/24 at 4 sessions. At 4 sessions turns 1–2 called it every time, turn 3 once, turns 4–6
never.

**Fix (2026-10-03):** history is a list of turns (trimmed whole, so a tool message never loses its call), and a
tool turn keeps the assistant `tool_calls` message plus a compact result (no raw search hits, ≤ 600 chars). Now
6/6 and 24/24. Tests: `tests/mouth/test_session.py`.

### I-14 — `run.sh` giving up on a slow first start orphans the LLM engine (low, confirmed on the box)
**Where:** `run.sh` `wait_tcp` (900 tries, ~15 min) and `cleanup` (kills the PIDs it started).

**What happened (box of 2026-10-05):** the first start downloaded VoxCPM2 + Gemma (~22 GB) slowly, and VoxCPM2's
first compile was still running when `wait_tcp` gave up ("MOUTH did not come up"). `cleanup` killed MOUTH's PID,
but Gemma's vLLM EngineCore (a child of MOUTH) survived with 14.7 GB of GPU memory. `scripts/stop.sh` finds stacks
by their `run.sh` process, which had exited, so it said "not running"; the engine had to be killed by its process
group by hand. The next start would have failed vLLM's free-memory check.

**Fix options:** in `cleanup`, signal MOUTH's whole process group (start it with `setsid`, or kill `-$$` when
`run.sh` leads its own group), the way `stop.sh` does; raise or drop the timeout while MOUTH is still alive (its
log shows progress), since `wait_tcp` already fails fast when MOUTH exits.

### I-15 — Local contract test aborts at exit after the EARS engine ran in the process (low, confirmed on the box)
**Where:** `TYGER_CONTRACT_LOCAL=1 tests/plugins/test_contract.py`, at interpreter exit.

**What happens (box of 2026-10-06):** every check passes (`ok` for all six suites), then the process prints
`terminate called without an active exception` and exits 134 (SIGABRT), so a script that reads the exit code sees
a failure. Isolated: `test_tts_local` alone, `test_ears_local` alone and `test_llm_local` alone each exit 0;
`test_tts_local` then `test_ears_local` in one process aborts. So it is VoxCPM2 (vLLM-Omni stage pool) and EARS
(nemotron + head CUDA graphs) torn down together, not the LLM plugins: a native thread still joinable when the
interpreter exits. The served stack runs them in separate processes and is not affected.

**Update (2026-10-06, R-05):** it is the EARS engine alone. `test_ears_local` by itself aborts at exit 4 / 4
times, on the code before R-01 too (the one clean run above was luck). The likely cause is EarsEngine's two daemon
threads (`ears-gpu`, `ears-asr`), with CUDA streams and graphs, still alive when the interpreter tears down.
VoxCPM2 is not involved.

**Fix options:** find the thread. VoxCPM2 is already shut down before EARS loads (the stage pool logs it), so
look at what that shutdown leaves running in-process (no ptrace in this container, so no gdb / py-spy; try
`threading.enumerate()` and `faulthandler` at exit). As a stopgap, end the test's `__main__` with `os._exit(code)`
after flushing, since the results are already printed.

---

## Measurement

### I-04 — "Onset" excludes endpoint detection (medium, confirmed in code)
**Where:**
- `scripts/stack/load_ears.py:85-86` starts the clock at the `endpoint` event.
- `server.py` `response_onset` starts at `send_turn`.

**What happens:**
1. The 529 / 636 / 810 ms figures for 1 / 2 / 4 conversations are endpoint to first audio.
2. The gap the user feels also includes endpoint detection:
   - the time for `p_end` to reach `ep_thr` (0.9146);
   - `ep_deb` (2 frames, 160 ms);
   - hop quantisation (up to 320 ms, see L-02);
   - hop lag;
   - the final ASR before `T` is sent.
3. The true end-of-speech to first-audio gap is probably several hundred ms larger than reported.

**Fix:**
- The load test knows exactly when its injected speech ends. Record `t_speech_end` and report end-of-speech
  to first reply audio alongside the existing metric.
- Also report endpoint detection delay (end of speech to the `endpoint` event) on its own. It's the number
  live tuning (O-01) should drive down.

**Status: fixed (2026-10-03).** `scripts/stack/load_ears.py` reports both. Healthy box: end of speech → endpoint p50 779 ms
(1 conv) / 773 ms (4 conv), end of speech → first audio p50 1013 / 1022 ms, against endpoint → audio 229 / 257 ms.
Table in NOTES.md → "Response gap (I-04)". The test also injects one sentence now (the whole reply hit inner pauses).

### I-13 — `scripts/stack/load_ears.py` runs don't compare (medium, found on the box)
**What happened:**
- The utterance was the LLM's reply to a "Say: …" prompt, and the agent's reply to it was free text, so both
  lengths changed run to run (hop lag p90 190–260 ms between identical runs).
- Where the speech ended inside a 320 ms hop depended on wall-clock turn timing, so detection moved by up to a
  hop between runs.
- It is the yardstick for the plugin refactor's parity checks (PLUGINS.md), so it has to be repeatable.

**Fix:** a fixed sentence spoken by MOUTH's TTS through a new TTS-only wire message `S`; a request whose reply the
LLM can't vary (count to ten), checked by comparing reply-text hashes; lead-in silence so each utterance ends at a
fixed hop phase (cycling ¼ hop steps over turns).

**Status: fixed (2026-10-04).** 4 conversations, 3 runs: end of speech → first audio p50 1250–1268 ms (was
1208–1356 with the fixed speech alone); 1 conversation, 2 runs within 4 ms. NOTES.md → "`scripts/stack/load_ears.py` made
deterministic (I-13)".

---

## Performance levers

### I-06 — Voice clip copied as Python floats per TTS request (medium, hypothesis)
**Where:**
- `app/tts_voxcpm.py:29`: `Voice.ref` is a `tuple` of Python floats.
- `:70`: `tuple(wav.tolist())`.
- `:74`: `list(voice.ref)` on every request, inside `_tok_lock`.

**What happens:**
1. A 10 s clip at 48 kHz is ~480k Python floats. Each TTS request (every text chunk, every backchannel)
   copies it to a list under a lock shared by all sessions.
2. The list then goes through `build_voxcpm2_prompt` and is serialised to the VoxCPM2 stage process.
3. This could be a real share of the ~400 ms reported as "chunk wait + ref encode" (NOTES.md, V1.3). It
   wasn't verified: `vllm_omni` isn't available off the box.

**Measure first:** time `self._prompt(text, voice)`, and the time from `generate()` to its first output, with
the clip as a tuple, as an `np.float32` array, and with no clip.

**Fix if it matters:**
- Store `ref` as a read-only `np.ndarray`. The `Voice` dataclass can drop `frozen` or use `eq=False`.
- Pass whatever type `build_voxcpm2_prompt` accepts without conversion.
- Resample the clip to the model's expected rate once, at load.

**Status (2026-10-03): the hypothesis didn't hold; the resample did.** Building the prompt from the tuple took 2 ms.
Per request, generate → first audio was 146 ms with the 44.1 kHz clip, 97 ms with the clip resampled once at load
to the VAE's 16 kHz (the stage worker resampled it on the CPU every time), 78 ms with L-03 on top (zero-shot: 81).
`scripts/bench/tts_voice.py`.

### L-01 — Encoder re-encodes the full 8 s window, twice, every hop, in fp32
**Where:**
- `app/ears_core/batched.py:308-314`: `win_u` and `win_a` are 8 s windows, both encoded every 320 ms.
- `:204`: `dtype=torch.float32`.

**What happens:**
1. Each hop re-encodes ~25× the new audio, for two channels.
2. The agent channel is cached only while all-zero (listening).
3. Per NOTES.md the encoder is ~40% of the EARS GPU thread at 4 conversations. It's the main thing keeping
   p90 hop lag above one hop.

**Options, cheapest first:**
1. **bf16 encoder.** Validate with `scripts/validate/ears_batched.py`: event parity and head-output diff. The
   heads were trained on fp32 features, so the threshold decisions must hold.
2. **In speaking mode, precompute the agent channel's features.** The TTS audio is known ahead of
   playback.
3. **Cache-aware streaming encode.** nemotron-3.5-asr-*streaming* supports it in principle. But the heads
   were trained on L15 of rolling 8 s windows, so the features differ, and this needs upstream
   validation or retraining (the head repo, not here).

**Status (2026-10-03).** Option 1 (bf16) rejected: no speedup (the encoder is launch-bound: 8 windows cost 21 ms
vs 14 ms for one), head diffs up to 0.14, an event changed. The real cost was the **CPU feature extraction**
(28 ms for 8 windows, more than the encoder); it now runs on the GPU (1.9 ms), same events and transcripts. 4
conversations: encoder busy ~33% → ~22%, hop lag p90 ~240 → ~205 ms. Option 2 not tried. **Option 3 stays a
recommendation for the head repo**: a cache-aware streaming encode would cut the encoder to the new 320 ms per
hop, but needs the heads validated or retrained on streaming L15 features.

### L-02 — Decisions quantised to 320 ms hops
**Where:** `HOP_FRAMES = 4` in `app/ears_core/live_ears.py`.

**What happens:** endpoint and barge decisions land on hop boundaries, adding ~160 ms on average to both.

**Lever:** a 160 ms hop (2 frames) doubles the tick rate. Only worth trying once L-01 frees GPU time.
Measure hop lag at 4 conversations.

### L-03 — Per-request voice encode
**What happens:** VoxCPM2 encodes the reference clip on every request. Zero-shot first audio was 113 ms;
cloned was ~500 ms on the plain turn (NOTES.md V1.3).

**Lever:** vLLM-Omni can load precomputed voice profiles from a directory. Profiles derived from the user's
clip are personal data: keep them on the box and gitignore them.

**Status: done (2026-10-03), in memory instead of files.** The VoxCPM2 worker has a process-wide speaker cache
keyed by `voice_name`: the first request stores the encoded clip, later ones send a `voice_profile` with only its
length. MOUTH primes every clip at startup. Per-request first audio 97 → 78 ms; with I-06's resample, 146 → 78 ms.
Smoke plain-turn first audio 354 → 263 ms. No voice-derived files on disk.

### L-04 — Contextual tool filler: start it earlier, cover slow tools
**Where:**
- `app/mouth.py:220-238`: the tool path and its lead-in.
- `app/llm.py:69-75`: `tool_calls` is yielded only after the generation ends.

**What happens today:**
1. A tool turn already speaks a contextual filler: a separate LLM request (`LEADIN_SYS`, its own `llm-<id>`)
   turned into one TTS request (`tts-<id>`), sent as reply audio `A` while the tools run in background
   threads.
2. **It starts late.** `LocalLLM.stream` reports `tool_calls` only after the first generation has finished,
   including the full call (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`) and end-of-sequence.
   Qwen typically emits `<tool_call>` as its first token, so the filler, and the tools, wait for the whole
   call to decode: a few dozen tokens at 92 tok/s on the throttled box. Then come the filler's own decode
   and the TTS onset.
3. **One filler only.** If Tavily takes 2–3 s, the user hears the filler and then silence until phase B
   speaks.
4. The filler sees only the transcript, not the tool name or arguments. That's usually fine: the transcript
   carries the topic.

**Proposed design:**
1. **Signal the tag immediately.** `split_speakable` already detects `<tool_call>` the moment it appears.
   Yield a `("tool_start", None)` event from `LocalLLM.stream` right then, and keep decoding the arguments.
2. **Start the filler on `tool_start`.**
   - A helper thread runs the filler's LLM request (`LEADIN_SYS` + transcript) and TTS request with the
     turn's `stop_flag`, putting the audio on a queue.
   - The turn generator drains that queue as `A` frames while the main request finishes the call.
   - vLLM batches the filler with the main request.
3. **Run the tools as soon as `tool_calls` is parsed,** not after the filler is generated.
4. **Second filler if the tools are slow.**
   - If they haven't returned ~1–1.5 s after the first filler's audio ends, make one more short request
     ("still looking, almost there"), passing the first filler so it doesn't repeat.
   - Cap it at 1–2 extra fillers.
   - Use EARS-side playback time or the filler's audio duration to decide when it has "ended".
5. **Barge-in:** everything shares the turn's `stop_flag`, so a barge aborts the filler's LLM and TTS
   requests by their IDs, as today. Pair this with I-03 (don't block on tool threads after a barge).
6. **Edge case:** text the model emits *before* `<tool_call>` is shorter than `MIN_CHARS`, so it's never
   spoken today (`_speak_stream` drops the tail when there are calls). With the filler starting earlier,
   that stays the right behaviour: don't speak both.

**Measure:**
- Tool-turn first-audio before and after (`scripts/stack/smoke.py` weather turn, and `scripts/stack/load.py` at 4 sessions).
- The longest silence during a tool turn with a stubbed tool that sleeps 3 s.

**Status: done (2026-10-03).** As proposed, plus: the filler prompts use few-shot turns instead of a quoted example
(the old prompt's 'Sure, let me check that for you.' came out verbatim every time). 4 sessions, live Tavily:
tool-turn first audio p50 640 → 408 ms, worst dead air 2584 → 1265 ms; with a 3 s stub tool, worst dead air
5031 → 1314 ms. Details in NOTES.md → "Tool fillers (L-04)".

---

## Ops

- **O-01 — Endpoint thresholds were never tuned live (V2).**
  - `ep_thr 0.9146`, `ep_deb 2`, barge and bc all ship at training defaults; offline thresholds ran ~2× high
    on a live mic.
  - It's the biggest risk to how the product feels.
  - Needs the user driving the mic with `?debug=1`. Do I-04 first so tuning has a real target.
- **O-02 — The Dockerfile was never built or run.**
  - Untested: `python:3.12-slim` + `uv` + cu130 wheels + vllm-omni + voxcpm.
  - **Status (2026-10-03): blocked on this box.** The GPU box is itself a Docker container (`/.dockerenv`) with no
    Docker CLI, daemon or socket, so nothing can be built here. Done instead: the README's non-Docker fresh-box
    install, run end to end on a clean box, failed on a pin (`accelerate` 1.15 vs vllm-omni's exact 1.12), now
    fixed in `requirements.txt`, which the Dockerfile installs too. Needs a host with Docker + the NVIDIA runtime.
- **O-03 — Branches.** `main` = `07d651e First`. Everything since is on `origin/setup/new-box` (=
  local `voxcpm2`). The user pushes and merges.
- **O-04 — Throttled GPU.** The current 5090 is stuck at P3 (half memory clock); Qwen3-4B decodes at 92
  tok/s against ~150 healthy. Re-run every benchmark on a healthy box before tuning against it.
  - **Status: done (2026-10-03).** New box, healthy (P1, memory 13 801 MHz); everything re-baselined, NOTES.md
    → "Box (2026-10-03)".
- **O-05 — Tests.**
  - GPU-free unit tests cover only the chunker and tool-call parsing.
  - Nothing covers the EARS server state machine (speaking / halting / pending / reset) or the MOUTH
    session flow. Several issues above (I-01, I-02, I-03, I-07) live there.
  - A fake-engine harness would let them run without a GPU.
  - There is no CI.
  - **Status (2026-10-03):** `tests/fakes.py` + `tests/mouth/test_session.py` cover the MOUTH session (barge
    history, tool barge, fillers, backchannel cancel, tool history), no GPU. The EARS server state machine is
    still only covered by GPU scripts (`scripts/validate/barge.py`, `scripts/stack/smoke.py`). No CI.
