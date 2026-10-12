"""Conversation: one browser WebSocket, its EARS plugin session, and its own TCP connection to MOUTH. It owns the
turn-taking state machine; the plugin owns the model.

States (the plugin decides from the audio, this class decides what happens):
  * listening: user audio goes to the plugin in "listen" mode. ENDPOINT sends the transcript to MOUTH (T) and the
    agent starts speaking; BACKCHANNEL asks MOUTH for a short ack (C) while the user keeps the floor.
  * speaking: from the turn's start until the page reports it has PLAYED the whole reply (`played:<id>:<samples>`;
    not when MOUTH is done generating), or, without a report, PLAYBACK_TIMEOUT_S past the server's own guess. The
    plugin hears the user and the agent's own audio (the 2nd ear, built from the page's playback tags: see
    AgentPlayback) in "speaking" mode; BARGE stops playback in the page and the reply in MOUTH (B, with how much of
    it the user had heard when they started talking: the tags at the plugin's onset, as detecting a barge takes a
    second or more).
  * halting: after a barge, until MOUTH confirms the stop (its D). An ENDPOINT meanwhile is held (`pending`) and
    sent as soon as the halt ends.
Taking the floor (DEVELOPING.md -> "Background tasks and the floor"): when a background task finishes, MOUTH asks
for the floor (F {"task": id}). EARS grants it (G) only when the agent isn't speaking or halting, no user turn is
held, no backchannel is playing, and the user has been quiet for mouth.tasks.announce_after_silence_ms (consecutive
`idle` frames). A user endpoint always wins: it starts a reply turn first and the request waits. On a grant the
agent speaks exactly as for a reply.

PRIVACY: no audio or transcript is written to disk or logs; only timings and event types. The page gets the user's
own transcript back (as it grows on `partial`, then whole on `endpoint`), to show it.
"""
from __future__ import annotations
import asyncio, json, logging, math, time
from collections import deque

import numpy as np
from aiohttp import WSMsgType

from app.ears.playback import AgentPlayback, BACKCHANNEL_ID, audio_frame, mic_block
from app.wire import async_send_msg, async_recv_msg

log = logging.getLogger("ears")
PLAYBACK_TIMEOUT_S = 2.0                        # no `played` report this long past the server's guess: end the turn
FRAME_MS = 80                                   # one FRAME event per 80 ms of user audio


class Conversation:
    def __init__(self, ws, ears, session, cfg, voices=(), default_voice=None, labels=None):
        self.ws = ws; self.ears = ears; self.session = session; self.cfg = cfg
        self.voices = list(voices); self.default_voice = default_voice
        self.labels = labels or {}                       # voice -> its picker entry, e.g. "Ori (ori)"
        self.hop_n = ears.hop_samples
        tasks = (cfg.get("mouth") or {}).get("tasks", {})
        self.floor_frames = math.ceil(float(tasks.get("announce_after_silence_ms", 1200)) / FRAME_MS)
        self.mic = np.zeros(0, np.float32)               # user audio not yet cut into hops
        self.ear = np.zeros(0, np.float32)               # the 2nd ear for the same samples (from the page's tags)
        self.mic_pos = 0; self.hop_pos = 0               # mic samples received / cut into hops so far
        self.hop_ready = asyncio.Event()
        self.playback = AgentPlayback(ears.sr)
        self.reader = self.writer = None; self.have_mouth = False
        # the state machine (module docstring)
        self.speaking = False; self.halting = False
        self.pending = None                    # an endpoint transcript held while halting
        self.reset_due = False                 # reset the plugin session before the next hop (a new turn)
        self.bc_active = False                 # a backchannel is being made / played
        self.mouth_done = False                # MOUTH finished generating; the reply may still be playing
        self.done_task = None                  # waits for the playback to end, then finishes the turn
        self.floor_q = deque()                 # task ids waiting for the floor
        self.idle_frames = 0                   # both sides quiet for this many frames
        self.mouth_gone = False                # MOUTH's connection broke: the conversation is ending
        self.turn_t0 = None; self.await_onset = False; self.barge_t0 = None   # timings for the logs

    # ---- the conversation's lifetime ----
    async def run(self):
        await self._connect_mouth()
        await self.ws.send_json(self._ready_message())
        log.info("conversation start (active=%d)", self.ears.active())
        pump = asyncio.create_task(self._pump_mouth()) if self.have_mouth else None
        hops = asyncio.create_task(self._hop_loop())
        try:
            async for msg in self.ws:
                await self._on_browser(msg)
        finally:
            hops.cancel()
            if pump:
                pump.cancel()
            if self.done_task is not None:
                self.done_task.cancel()
            if self.have_mouth:
                try:
                    self.writer.close()
                except Exception:
                    pass
            log.info("conversation end (active=%d)", self.ears.active() - 1)

    async def _connect_mouth(self):
        mp = self.cfg["server"]
        try:
            self.reader, self.writer = await asyncio.open_connection(mp["mouth_host"], int(mp["mouth_port"]))
            self.have_mouth = True
        except OSError as e:
            log.warning("no MOUTH: %s", e)

    def _ready_message(self):
        """The page's start-up info: audio format, MOUTH there or not, voices (with their characters' labels), and
        the plugin's tunable thresholds (their sliders and this conversation's values: the page builds its tuning
        panel from them)."""
        return {"type": "ready", "hop_ms": int(self.hop_n / self.ears.sr * 1e3), "sr": self.ears.sr,
                "mouth": self.have_mouth, "voices": self.voices, "voice": self.default_voice, "labels": self.labels,
                "thresholds": self.ears.thresholds(self.session),
                "sliders": [t.slider_spec() for t in self.ears.THRESHOLDS if t.slider is not None]}

    # ---- the browser ----
    async def _on_browser(self, msg):
        """Mic blocks (binary, tagged with what was playing) are only buffered here, with the 2nd ear built for them
        from the tags; _hop_loop cuts both into hops. Text messages are commands."""
        if msg.type == WSMsgType.BINARY:
            block = mic_block(msg.data)
            if block is None:
                return
            rid, at24, pcm = block
            self.mic = np.concatenate([self.mic, pcm])
            self.ear = np.concatenate([self.ear, self.playback.agent_block(rid, at24, len(pcm), self.mic_pos)])
            self.mic_pos += len(pcm)
            if len(self.mic) >= self.hop_n:
                self.hop_ready.set()
        elif msg.type == WSMsgType.TEXT:
            await self._on_command(msg.data)

    async def _on_command(self, data):
        if data == "reset":
            self.session.reset(); self.hop_pos += len(self.mic)
            self.mic = np.zeros(0, np.float32); self.ear = np.zeros(0, np.float32)
        elif data.startswith("played:"):              # the page played a reply to its end: played:<id>:<samples>
            try:
                _, rid, samples = data.split(":")
                self.playback.page_played(int(rid), int(samples))
            except ValueError:
                return
            await self._finish_if_played()
        elif data.startswith("voice:") and self.have_mouth:
            name = data.split(":", 1)[1]
            if name in self.voices:
                await self.to_mouth(b"V", {"voice": name})
                await self.ws.send_json({"type": "voice", "voice": name})
        elif data.startswith("set:"):                  # a threshold, for this conversation only (the sliders)
            try:
                _, attr, val = data.split(":", 2)
                cur = self.ears.set_threshold(self.session, attr, val)
                log.info("threshold %s = %s (this conversation's slider)", attr, cur)
                await self.ws.send_json({"type": "threshold", "attr": attr, "value": cur})
            except (ValueError, KeyError):
                pass

    # ---- hops: mic audio through the plugin, one hop at a time ----
    async def _hop_loop(self):
        """Decoupled from the mic reader, so the socket keeps draining while a hop is on the GPU, and `more` (another
        hop already buffered: this conversation is behind) is visible to the plugin's batcher."""
        try:
            while True:
                while len(self.mic) < self.hop_n:
                    self.hop_ready.clear(); await self.hop_ready.wait()
                user = np.ascontiguousarray(self.mic[:self.hop_n]); self.mic = self.mic[self.hop_n:]
                ear = np.ascontiguousarray(self.ear[:self.hop_n]); self.ear = self.ear[self.hop_n:]
                self.hop_pos += self.hop_n; hop_end = self.hop_pos
                if self.reset_due:
                    self.session.reset(); self.reset_due = False
                if self.speaking:
                    agent, mode = ear, "speaking"
                else:
                    agent, mode = np.zeros(self.hop_n, np.float32), "listen"
                events = await self.ears.feed(self.session, user, agent, mode, more=len(self.mic) >= self.hop_n)
                await self._on_events(events, hop_end)
        except (ConnectionError, asyncio.CancelledError):   # the page went away (MOUTH errors end in to_mouth)
            pass
        except Exception as e:
            log.exception("hop loop error: %s", type(e).__name__)
            await self.ws.close()

    async def _on_events(self, events, hop_end):
        """One hop's plugin events, in order (`hop_end`: the mic position the hop ends at); then the floor, then one
        state update to the page."""
        peak = {"yield": 0.0, "bc": 0.0, "p_end": 0.0}; last = None
        for e in events:
            kind = e["kind"]
            if kind == "FRAME":
                self._on_frame(e, peak); last = e
            elif kind == "ENDPOINT" and not self.speaking:
                await self._on_endpoint(e)
            elif kind == "BACKCHANNEL" and not self.speaking and not self.halting and not self.bc_active:
                await self._on_backchannel(e)
            elif kind == "BARGE" and self.speaking:
                await self._on_barge(e, hop_end)
            elif kind == "PARTIAL" and not self.speaking:   # the user's words so far, for the page
                await self.ws.send_json({"type": "partial", "text": e["text"]})
        if self.have_mouth:
            await self._maybe_grant_floor()             # after this hop's events: an endpoint wins
        if last is not None:
            await self.ws.send_json({"type": "state", "state": last["state"], "yield": peak["yield"],
                                     "bc": peak["bc"], "hazard": last.get("hazard", 0.0),
                                     "p_end": round(peak["p_end"], 3)})

    def _on_frame(self, e, peak):
        quiet = e.get("state") == "idle" and not self.speaking and not self.halting   # both silent
        self.idle_frames = self.idle_frames + 1 if quiet else 0
        peak["yield"] = max(peak["yield"], e.get("yield_prob", 0.0) or 0.0)
        peak["bc"] = max(peak["bc"], e.get("bc_prob", 0.0) or 0.0)
        peak["p_end"] = max(peak["p_end"], e.get("p_end", 0.0) or 0.0)

    async def _on_endpoint(self, e):
        log.info("endpoint by=%s turn_s=%.2f segments=%d", e.get("by"), e.get("turn_s", 0.0), e.get("segments", 0))
        await self.ws.send_json({"type": "endpoint", "turn_s": round(e["turn_s"], 1), "by": e.get("by"),
                                 "transcript": e["transcript"]})                # the user's words, for the page
        if self.have_mouth and e["transcript"].strip():
            if self.halting:
                self.pending = e["transcript"]
            else:
                await self._send_turn(e["transcript"])

    async def _on_backchannel(self, e):
        log.info("backchannel bc_prob=%.2f partial_chars=%d", e.get("bc_prob", 0.0),
                 len(e.get("partial") or ""))           # length only: never log the words
        await self.ws.send_json({"type": "bc_cue", "bc": e.get("bc_prob")})
        if self.have_mouth:
            self.bc_active = True
            await self.to_mouth(b"C", {"partial": e.get("partial", "") or ""})

    async def _on_barge(self, e, hop_end):
        self.speaking = False; self.halting = True; self.reset_due = True
        onset = float(e.get("onset_s") or 0.0)
        heard = self.playback.heard_s(hop_end - int(onset * self.ears.sr))   # by the tags, when the user started
        self.barge_t0 = time.perf_counter()
        log.info("barge yield=%.2f onset %.2f s: heard %.2f s by then (the server's guess at the barge: %.2f s)",
                 e.get("yield_prob", 0.0), onset, heard, self.playback.estimate_s())
        await self.ws.send_json({"type": "barge", "yield": e["yield_prob"]})   # the page stops playback
        if self.mouth_done:
            # MOUTH already finished generating (its D was held for playback): no D will come back for this turn,
            # so close it here instead of waiting
            if self.done_task is not None:
                self.done_task.cancel()
            self.halting = False
            await self._finish_turn(True)
        elif self.have_mouth:            # + how much of the reply had played, so MOUTH remembers what was heard
            await self.to_mouth(b"B", {"played_s": round(heard, 2)})

    # ---- the agent's turn ----
    def _start_speaking(self):
        self.speaking = True; self.turn_t0 = time.perf_counter(); self.await_onset = True
        self.mouth_done = False; self.playback.start()

    async def _send_turn(self, transcript):
        self._start_speaking()
        await self.to_mouth(b"T", {"transcript": transcript})

    async def _maybe_grant_floor(self):
        """A finished task's announcement gets the floor only when nobody else has it (module docstring)."""
        if not self.floor_q or self.speaking or self.halting or self.pending is not None or self.bc_active \
                or self.idle_frames < self.floor_frames:
            return
        tid = self.floor_q.popleft()
        self._start_speaking()
        log.info("floor granted to task #%s after %d ms of silence", tid, self.idle_frames * FRAME_MS)
        await self.to_mouth(b"G", {"task": tid})
        await self.ws.send_json({"type": "task_done", "task": tid})

    async def _finish_turn(self, barged):
        """The reply is over (played out, or barged): back to listening; send a held turn if any. A barge already
        reset EARS (seeding the new turn with the start of the user's speech, I-02): don't wipe that turn again.
        Logs how far arrival-order pairing would have put the 2nd ear ahead of what the user heard (timing only)."""
        self.speaking = False
        if not barged:
            self.reset_due = True
        self.mouth_done = False; self.done_task = None
        if self.barge_t0 is not None and barged:
            log.info("barge_halted tail %.0f ms", (time.perf_counter() - self.barge_t0) * 1e3)
        self.barge_t0 = None
        off = self.playback.offset_ms()
        log.info("turn_done barged=%s; 2nd ear by arrival order %s", bool(barged),
                 "untagged" if off is None else f"{off:+.0f} ms ahead of the page's playback")
        await self.ws.send_json({"type": "turn_done", "barged": barged})
        if self.halting:
            self.halting = False
            if self.pending is not None:
                transcript, self.pending = self.pending, None
                await self._send_turn(transcript)

    async def _finish_if_played(self):
        """MOUTH is done and the page reports it has played everything: the turn is over."""
        if self.speaking and self.mouth_done and self.playback.page_done():
            if self.done_task is not None:
                self.done_task.cancel()
            await self._finish_turn(False)

    async def _finish_after_playback(self):
        """The safety net when the page never reports: end the turn PLAYBACK_TIMEOUT_S past the server's own guess
        at when it will have played everything."""
        try:
            while (wait := self.playback.play_until + PLAYBACK_TIMEOUT_S - time.perf_counter()) > 0:
                await asyncio.sleep(wait)
            log.warning("no playback report from the page: turn ended by the timeout")
            await self._finish_turn(False)
        except asyncio.CancelledError:
            pass

    # ---- MOUTH ----
    async def to_mouth(self, kind, payload):
        """Every message to MOUTH goes through here: a broken connection ends the conversation (_mouth_lost)."""
        try:
            await async_send_msg(self.writer, kind, payload)
        except ConnectionError as e:
            await self._mouth_lost(type(e).__name__)

    async def _mouth_lost(self, what):
        """MOUTH's connection broke (it restarted or died): this conversation can't answer any more. Tell the page
        and close the socket, which ends the conversation (no dead hop loop, no audio piling up in `mic`)."""
        if self.mouth_gone:
            return
        self.mouth_gone = True
        log.warning("MOUTH connection lost (%s): closing the conversation", what)
        try:
            await self.ws.send_json({"type": "error", "error": "mouth_lost"})
            await self.ws.close()
        except ConnectionError:
            pass

    async def _pump_mouth(self):
        handlers = {b"A": self._on_reply_audio, b"K": self._on_bc_audio, b"X": self._on_text, b"L": self._on_links,
                    b"D": self._on_done, b"F": self._on_floor_request}
        try:
            while True:
                kind, payload = await async_recv_msg(self.reader)
                handler = handlers.get(kind)
                if handler is not None:
                    await handler(payload)
        except (asyncio.IncompleteReadError, ConnectionError) as e:
            await self._mouth_lost(type(e).__name__)

    async def _on_reply_audio(self, payload):
        if not self.speaking:
            return
        if self.await_onset and self.turn_t0 is not None:
            self.await_onset = False
            log.info("response_onset %.0f ms", (time.perf_counter() - self.turn_t0) * 1e3)
        a24 = np.frombuffer(payload, np.float32)
        self.playback.add(a24, self.hop_pos)            # before the page can tag any of it
        await self.ws.send_bytes(audio_frame(self.playback.rid, a24))

    async def _on_bc_audio(self, payload):
        if self.bc_active and not self.speaking:       # only while the user holds the floor (a reply supersedes it)
            await self.ws.send_bytes(audio_frame(BACKCHANNEL_ID, np.frombuffer(payload, np.float32)))

    async def _on_text(self, payload):
        d = json.loads(payload)
        await self.ws.send_json({"type": "backchannel" if d.get("bc") else "reply", "text": d["text"]})

    async def _on_links(self, payload):
        await self.ws.send_json({"type": "links", "links": json.loads(payload)["links"]})

    async def _on_done(self, payload):
        d = json.loads(payload)
        if d.get("bc"):
            self.bc_active = False
            await self.ws.send_json({"type": "backchannel_done", "text": d.get("text", ""),
                                     "aborted": bool(d.get("aborted"))})          # dropped unvoiced: the turn ended first
        elif d.get("barged") or not self.speaking:      # halted: nothing left to play out
            await self._finish_turn(bool(d.get("barged")))
        else:                                           # generated; hold until the page has played it
            self.mouth_done = True
            if self.playback.page_done():
                await self._finish_turn(False)
            else:
                self.done_task = asyncio.create_task(self._finish_after_playback())

    async def _on_floor_request(self, payload):
        tid = json.loads(payload).get("task")
        if tid not in self.floor_q:
            self.floor_q.append(tid)
        log.info("floor requested by task #%s (%d waiting)", tid, len(self.floor_q))

