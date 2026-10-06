"""Session: one conversation = one EARS connection. It holds the conversation's state (history, voice, recent
backchannel acks, background tasks), reads the wire, and runs two workers: turns (ReplyTurn, AnnouncementTurn,
SayTurn) and backchannels, so a backchannel is generated while a turn is (their own engine requests). Several
Sessions run at once on the shared engines. Barge-in (B) stops the current turn only.
"""
from __future__ import annotations
import json, logging, queue, threading, time
from collections import deque

from app.messages import bc_messages
from app.chunker import say_clean
from app.mouth.history import History
from app.mouth.reply import ReplyTurn
from app.mouth.speech import speech_audio
from app.mouth.threads import text_of
from app.mouth.turns import AnnouncementTurn, SayTurn
from app.plugins.base import LLMError
from app.tasks import TaskRunner
from app.wire import send_msg, recv_msg

log = logging.getLogger("mouth")
RECENT_ACKS = 3


class Session:
    def __init__(self, m, conn):
        self.m = m; self.conn = conn
        self.send_lock = threading.Lock()
        self.history = History(m.history_turns, m.prompts["reply"])
        self.recent_acks = deque(maxlen=RECENT_ACKS)   # in memory only; so backchannels don't repeat
        self.voice_name, self.voice = m.default_voice()  # every session starts on the configured default
        self.turn_stop = threading.Event()
        self.closed = threading.Event()
        self.turn_seq = 0                  # T frames seen; a backchannel queued before the latest T is stale
        self.barge_played_s = None         # reply audio the client had played at the barge (from EARS's B)
        self.turn_q: queue.Queue = queue.Queue()
        self.bc_q: queue.Queue = queue.Queue()
        self.tasks = TaskRunner(m, context=self._task_context, on_done=self._task_done) if m.tasks_on else None

    def send(self, kind, payload=b""):
        if self.closed.is_set():
            return
        try:
            with self.send_lock:
                send_msg(self.conn, kind, payload)
        except OSError:                                # EARS went away mid-stream -> stop this session
            self.closed.set(); self.turn_stop.set()

    def set_voice(self, name):
        if name == self.voice_name:
            return
        v = self.m.voice(name)
        if v is None:
            log.warning("voice '%s' not found (keeping %s)", name, self.voice_name); return
        self.voice_name, self.voice = name, v
        log.info("voice -> %s", name)

    # ---- what the workers run: generators of MOUTH events ----
    def stream_turn(self, transcript, stop_flag):
        """The reply to the user's words (may call tools, may escalate)."""
        return ReplyTurn(self, stop_flag, transcript).run()

    def stream_announce(self, task_id, stop_flag):
        """A finished background task speaks up (EARS gave it the floor: G)."""
        return AnnouncementTurn(self, stop_flag, task_id).run()

    def stream_say(self, text, stop_flag):
        """Speak a given text, TTS only (S; test tools)."""
        return SayTurn(self, stop_flag, text).run()

    def stream_bc(self, partial, stop_flag):
        """A backchannel: always a contextual LLM ack off the ASR partial. The LLM picks it: a reaction to the
        content, or a neutral hum when the partial is too short (< bc_min_words, flagged to it explicitly) or gives
        nothing to react to. An ack with nothing speakable in it ("—", "*") counts as no ack: not spoken, not
        remembered."""
        m = self.m
        messages = bc_messages(m.prompts["filler"], list(self.recent_acks), partial, m.bc_min_words)
        text = text_of(m.llms["filler"].stream(messages, max_new=m.bc_max_new, stop_flag=stop_flag)).strip()
        if not say_clean(text):
            text = ""
        if text:
            self.recent_acks.append(text)
            yield ("text", text)
            yield from (("audio", w) for w in speech_audio(m.tts, text, self.voice, stop_flag))
        yield ("done", text)

    # ---- background tasks ----
    def _task_context(self):
        return self.history.recent(self.m.context_turns)

    def _task_done(self, task):
        """A task finished or failed (from its thread): ask EARS for the floor. EARS answers G when the user is
        silent, and the announcement runs as a turn. Until then the status note says it's done."""
        log.info("task #%d %s: asking for the floor", task.id, task.status)
        self.send(b"F", {"task": task.id})

    # ---- workers ----
    def _turn_worker(self):
        while (turn := self.turn_q.get()) is not None:
            self.turn_stop.clear(); self.barge_played_s = None
            stop = lambda: self.turn_stop.is_set() or self.closed.is_set()   # noqa: E731
            t0 = time.perf_counter(); t_txt = t_aud = None
            if "say" in turn:
                gen = self.stream_say(turn["say"], stop)
            elif "announce" in turn:
                gen = self.stream_announce(turn["announce"], stop)
            else:
                gen = self.stream_turn(turn["transcript"], stop)
            extra = {"proactive": turn["announce"]} if "announce" in turn else {}
            try:
                for kind, val in gen:
                    if kind == "text":
                        t_txt = t_txt or time.perf_counter() - t0
                        self.send(b"X", {"text": val})
                    elif kind == "audio":
                        if t_aud is None:
                            t_aud = time.perf_counter() - t0
                            log.info("onset: first-text %.0fms first-audio %.0fms", (t_txt or 0) * 1e3, t_aud * 1e3)
                        self.send(b"A", val.tobytes())
                    elif kind == "done":
                        self.send(b"D", {"text": val, "barged": self.turn_stop.is_set(), **extra})
            except Exception as e:
                if not self.closed.is_set():
                    if isinstance(e, LLMError):           # a hosted request failed past the turn's own handling
                        log.warning("turn: LLM request failed (%s)", e.kind)
                    else:
                        log.exception("turn error: %s", type(e).__name__)
                    self.send(b"D", {"text": "", "error": type(e).__name__, **extra})

    def _bc_worker(self):
        while (item := self.bc_q.get()) is not None:
            seq, bc = item                                  # a reply turn (T) after this C makes it moot: EARS
            stop = lambda seq=seq: self.closed.is_set() or self.turn_seq != seq  # noqa: E731  drops its K audio
            t0 = time.perf_counter(); t_aud = None; partial = bc.get("partial", "")
            try:
                for kind, val in self.stream_bc(partial, stop):
                    if kind == "text":
                        self.send(b"X", {"text": val, "bc": True})
                    elif kind == "audio":                   # K = backchannel audio, kept apart from replies
                        t_aud = t_aud or time.perf_counter() - t0
                        self.send(b"K", val.tobytes())
                    elif kind == "done":
                        aborted = stop()
                        self.send(b"D", {"text": val, "bc": True, "aborted": aborted})
                        # counts + timing only, never the words
                        log.info("backchannel: partial_words=%d ack_chars=%d first-audio %.0fms%s",
                                 len(partial.split()), len(val), (t_aud or 0) * 1e3,
                                 " (aborted: reply turn started)" if aborted else "")
            except Exception as e:
                if not self.closed.is_set():
                    if isinstance(e, LLMError):              # hosted LLM failed: no ack this time, no traceback
                        log.warning("backchannel: LLM request failed (%s)", e.kind)
                    else:
                        log.exception("backchannel error: %s", type(e).__name__)
                    self.send(b"D", {"text": "", "bc": True, "error": type(e).__name__})

    def run(self):
        """Read the wire until EARS disconnects; then stop whatever is live and end the background tasks."""
        workers = [threading.Thread(target=self._turn_worker, daemon=True),
                   threading.Thread(target=self._bc_worker, daemon=True)]
        for w in workers:
            w.start()
        try:
            while True:
                try:
                    kind, payload = recv_msg(self.conn)
                except OSError:                             # reset by the peer: the same as a clean close
                    break
                if kind is None:
                    break
                self._dispatch(kind, payload)
        finally:                                            # disconnect: stop + abort whatever is live
            self.closed.set(); self.turn_stop.set()
            if self.tasks is not None:                      # background tasks end with the conversation
                self.tasks.close()
            self.turn_q.put(None); self.bc_q.put(None)
            for w in workers:
                w.join(timeout=5)

    def _dispatch(self, kind, payload):
        """One message from EARS (app/wire.py has the kinds)."""
        if kind == b"T":
            self.turn_seq += 1                              # stops a backchannel still being made
            self.turn_q.put(json.loads(payload))
        elif kind == b"G":                                  # EARS gave a finished task the floor
            self.turn_q.put({"announce": json.loads(payload).get("task")})
        elif kind == b"S":                                  # test tools only: speak a fixed text
            self.turn_q.put({"say": json.loads(payload).get("text", "")})
        elif kind == b"C":
            self.bc_q.put((self.turn_seq, json.loads(payload)))
        elif kind == b"V":
            self.set_voice(json.loads(payload).get("voice", ""))
        elif kind == b"B":                                  # EARS may say how much reply audio had played
            self.barge_played_s = (json.loads(payload) if payload else {}).get("played_s")
            self.turn_stop.set()
