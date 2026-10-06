"""Conversation: one browser WebSocket, its EARS plugin session, and its own TCP connection to MOUTH. It owns the
turn-taking state machine; the plugin owns the model.

States (the plugin decides from the audio, this class decides what happens):
  * listening: user audio goes to the plugin in "listen" mode. ENDPOINT sends the transcript to MOUTH (T) and the
    agent starts speaking; BACKCHANNEL asks MOUTH for a short ack (C) while the user keeps the floor.
  * speaking: from the turn's start until the browser has finished PLAYING the reply (not when MOUTH is done
    generating: see AgentPlayback). The plugin hears the user and the agent's own audio (the 2nd ear) in "speaking"
    mode; BARGE stops playback in the browser and the reply in MOUTH (B, with how much was heard).
  * halting: after a barge, until MOUTH confirms the stop (its D). An ENDPOINT meanwhile is held (`pending`) and
    sent as soon as the halt ends.
Taking the floor (PLUGINS.md step 13): when a background task finishes, MOUTH asks for the floor (F {"task": id}).
EARS grants it (G) only when the agent isn't speaking or halting, no user turn is held, no backchannel is playing,
and the user has been quiet for mouth.tasks.announce_after_silence_ms (consecutive `idle` frames). A user endpoint
always wins: it starts a reply turn first and the request waits. On a grant the agent speaks exactly as for a reply.

PRIVACY: no audio or transcript is written to disk or logs; only timings and event types.
"""
from __future__ import annotations
import asyncio, json, logging, math, time
from collections import deque

import numpy as np
from aiohttp import WSMsgType

from app.ears.playback import AgentPlayback
from app.wire import async_send_msg, async_recv_msg

log = logging.getLogger("ears")
PLAYBACK_MARGIN_S = 0.25                        # slack for network + client scheduling on top of play_until
FRAME_MS = 80                                   # one FRAME event per 80 ms of user audio


class Conversation:
    def __init__(self, ws, ears, session, cfg, voices=(), default_voice=None):
        self.ws = ws; self.ears = ears; self.session = session; self.cfg = cfg
        self.voices = list(voices); self.default_voice = default_voice
        self.hop_n = ears.hop_samples
        tasks = (cfg.get("mouth") or {}).get("tasks", {})
        self.floor_frames = math.ceil(float(tasks.get("announce_after_silence_ms", 1200)) / FRAME_MS)
        self.mic = np.zeros(0, np.float32)               # user audio not yet cut into hops
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
        th = self.ears.thresholds(self.session)
        return {"type": "ready", "hop_ms": int(self.hop_n / self.ears.sr * 1e3), "sr": self.ears.sr,
                "mouth": self.have_mouth, "voices": self.voices, "voice": self.default_voice,
                "thresholds": {"barge_thr": th["barge_thr"], "bc_thr": th["bc_thr"],
                               "ep_thr": th["ep_thr"], "ep_deb": th["ep_deb"],
                               "endpoint": "ep", "fallback_frames": th["fallback_frames"]}}

    # ---- the browser ----
    async def _on_browser(self, msg):
        """Mic audio (binary) is only buffered here; _hop_loop cuts it into hops. Text messages are commands."""
        if msg.type == WSMsgType.BINARY:
            self.mic = np.concatenate([self.mic, np.frombuffer(msg.data, dtype=np.float32)])
            if len(self.mic) >= self.hop_n:
                self.hop_ready.set()
        elif msg.type == WSMsgType.TEXT:
            await self._on_command(msg.data)

    async def _on_command(self, data):
        if data == "reset":
            self.session.reset(); self.mic = np.zeros(0, np.float32)
        elif data.startswith("voice:") and self.have_mouth:
            name = data.split(":", 1)[1]
            if name in self.voices:
                await self.to_mouth(b"V", {"voice": name})
                await self.ws.send_json({"type": "voice", "voice": name})
        elif data.startswith("set:"):                  # a threshold, for this conversation only (the sliders)
            try:
                _, attr, val = data.split(":", 2)
                cur = self.ears.set_threshold(self.session, attr, val)
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
                if self.reset_due:
                    self.session.reset(); self.reset_due = False
                if self.speaking:
                    agent, mode = self.playback.take_hop(self.hop_n), "speaking"
                else:
                    agent, mode = np.zeros(self.hop_n, np.float32), "listen"
                events = await self.ears.feed(self.session, user, agent, mode, more=len(self.mic) >= self.hop_n)
                await self._on_events(events)
        except (ConnectionError, asyncio.CancelledError):   # the page went away (MOUTH errors end in to_mouth)
            pass
        except Exception as e:
            log.exception("hop loop error: %s", type(e).__name__)
            await self.ws.close()

    async def _on_events(self, events):
        """One hop's plugin events, in order; then the floor, then one state update to the page."""
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
                await self._on_barge(e)
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
        await self.ws.send_json({"type": "endpoint", "turn_s": round(e["turn_s"], 1), "by": e.get("by")})
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

    async def _on_barge(self, e):
        self.speaking = False; self.halting = True; self.reset_due = True
        self.playback.clear_ear(); self.barge_t0 = time.perf_counter()
        log.info("barge yield=%.2f", e.get("yield_prob", 0.0))
        await self.ws.send_json({"type": "barge", "yield": e["yield_prob"]})   # the page stops playback
        if self.mouth_done:
            # MOUTH already finished generating (its D was held for playback): no D will come back for this turn,
            # so close it here instead of waiting
            if self.done_task is not None:
                self.done_task.cancel()
            self.halting = False
            await self._finish_turn(True)
        elif self.have_mouth:            # + how much of the reply had played, so MOUTH remembers what was heard
            await self.to_mouth(b"B", {"played_s": round(self.playback.played_s(), 2)})

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
        reset EARS (seeding the new turn with the start of the user's speech, I-02): don't wipe that turn again."""
        self.speaking = False; self.playback.clear_ear()
        if not barged:
            self.reset_due = True
        self.mouth_done = False; self.done_task = None
        if self.barge_t0 is not None and barged:
            log.info("barge_halted tail %.0f ms", (time.perf_counter() - self.barge_t0) * 1e3)
        self.barge_t0 = None
        log.info("turn_done barged=%s", bool(barged))
        await self.ws.send_json({"type": "turn_done", "barged": barged})
        if self.halting:
            self.halting = False
            if self.pending is not None:
                transcript, self.pending = self.pending, None
                await self._send_turn(transcript)

    async def _finish_after_playback(self):
        try:
            while (wait := self.playback.play_until + PLAYBACK_MARGIN_S - time.perf_counter()) > 0:
                await asyncio.sleep(wait)
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
        handlers = {b"A": self._on_reply_audio, b"K": self._on_bc_audio, b"X": self._on_text, b"D": self._on_done,
                    b"F": self._on_floor_request}
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
        await self.ws.send_bytes(payload)
        if self.speaking:
            self.playback.add(np.frombuffer(payload, np.float32))

    async def _on_bc_audio(self, payload):
        if self.bc_active and not self.speaking:       # only while the user holds the floor (a reply supersedes it)
            await self.ws.send_bytes(payload)

    async def _on_text(self, payload):
        d = json.loads(payload)
        await self.ws.send_json({"type": "backchannel" if d.get("bc") else "reply", "text": d["text"]})

    async def _on_done(self, payload):
        d = json.loads(payload)
        if d.get("bc"):
            self.bc_active = False
            await self.ws.send_json({"type": "backchannel_done", "text": d.get("text", "")})
        elif d.get("barged") or not self.speaking:      # halted: nothing left to play out
            await self._finish_turn(bool(d.get("barged")))
        else:                                           # generated; hold until it has played
            self.mouth_done = True
            self.done_task = asyncio.create_task(self._finish_after_playback())

    async def _on_floor_request(self, payload):
        tid = json.loads(payload).get("task")
        if tid not in self.floor_q:
            self.floor_q.append(tid)
        log.info("floor requested by task #%s (%d waiting)", tid, len(self.floor_q))

