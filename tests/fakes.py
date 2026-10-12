"""Driving MOUTH's Session and EARS's Conversation without a GPU (tests only). FakeLLM / FakeTTS are the `fake`
plugins (app/plugins/*/fake.py); FakeMouth carries the attributes Session reads from Mouth. For the LLM plugins'
call formats: stream_speakable and scripted_stream feed a model's text in small deltas. For EARS: FakePage is the
browser's side of the WebSocket, behaving like static/index.html, and MouthServer a TCP server standing in for
MOUTH. Test text and audio are synthetic.
"""
import os, sys, asyncio, json, threading, time
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from aiohttp import WSMsgType  # noqa: E402
from app import mouth as M  # noqa: E402
from app.ears.playback import AUDIO_HEADER, BACKCHANNEL_ID, LEAD_IN_S, MIC_HEADER, to_pcm16  # noqa: E402
from app.wire import async_send_msg, async_recv_msg  # noqa: E402
from app.tools import registry as T  # noqa: E402
from app.prompts import Characters, default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.plugins.llm.fake import FakeLLM  # noqa: E402
from app.plugins.tts.fake import FakeTTS  # noqa: E402
from app.mouth.tasks import TaskSlots  # noqa: E402

PROMPTS = default_prompts()     # the repo's config/prompts.yaml; FakeLLM routes requests by their system prompt


@contextmanager
def stub_tools(fn):
    """Every tool run for real (MOUTH's and background tasks') goes to fn(name, args) instead (execute_tool, patched
    where it lives: app/tools/registry.py)."""
    orig = T.execute_tool
    T.execute_tool = fn
    try:
        yield
    finally:
        T.execute_tool = orig


def slow_tools(seconds, result=None):
    """Tools that block for `seconds` (polling nothing, like a real HTTP call), then answer `result`."""
    return stub_tools(lambda name, args: (time.sleep(seconds), result or {"answer": "sunny"})[1])


class FakeMouth:
    """`smart`: a FakeLLM for the smart role, which turns escalation on (ask_expert offered), as an llm.smart block
    does in Mouth; `tasks`: a mouth.tasks dict, which with `smart` turns background tasks on. `slots`: share one
    TaskSlots between FakeMouths to test the global cap. `voices`: the config's voices block (character,
    characters); the voices that exist are VOICES."""
    VOICES = ("fake", "alt", "new")

    def __init__(self, llm=None, tts=None, smart=None, context_turns=4, max_steps=3, tasks=None, slots=None,
                 first_chunk="sentence", voices=None):
        self.llm = llm or FakeLLM(); self.tts = tts or FakeTTS()
        self.max_gen = 192; self.history_turns = 8; self.bc_min_words = 4; self.bc_max_new = 12
        self.temperature = 0.7; self.sr = self.tts.out_sr; self.first_chunk = first_chunk
        self.prompts = {"reply": PROMPTS, "filler": PROMPTS}
        self.characters = Characters({"voices": voices or {}}, PROMPTS)
        self.llms = {"reply": self.llm, "filler": self.llm}
        self.expert = smart is not None
        self.context_turns, self.max_steps = context_turns, max_steps
        self.tasks_cfg = tasks or {}
        self.tasks_on = self.expert and bool(tasks)
        self.task_slots = slots or TaskSlots(self.tasks_cfg.get("max_global", 4))
        self.tools = build_tools(PROMPTS, expert=self.expert, tasks=self.tasks_on)
        self.smart_tools = build_tools(PROMPTS) if self.expert else None
        self.task_tools = build_tools(PROMPTS, finish=True) if self.tasks_on else None
        if smart is not None:
            self.prompts["smart"] = PROMPTS; self.llms["smart"] = smart

    def default_voice(self):
        return "fake", None

    def voice(self, name):
        """A handle for a voice that exists (the fake TTS ignores it), else None, like Mouth.voice."""
        return name if name in self.VOICES else None


class FakeConn:
    """Collects what Session.send writes: (kind, payload) with JSON payloads decoded."""
    def __init__(self):
        self.out = []; self.lock = threading.Lock(); self.cond = threading.Condition(self.lock)

    def sendall(self, data):
        import json, struct
        kind, ln = struct.unpack(">cI", data[:5]); payload = data[5:5 + ln]
        if kind in (b"X", b"D"):
            payload = json.loads(payload)
        with self.cond:
            self.out.append((kind, payload, time.perf_counter())); self.cond.notify_all()

    def wait_for(self, pred, timeout=5.0):
        end = time.perf_counter() + timeout
        with self.cond:
            while True:
                hit = [o for o in self.out if pred(o)]
                if hit:
                    return hit[0]
                rem = end - time.perf_counter()
                if rem <= 0:
                    return None
                self.cond.wait(rem)


def make_session(llm=None, tts=None, **kw):
    m = FakeMouth(llm, tts, **kw)
    return M.Session(m, FakeConn()), m


def run_turn(sess, transcript, stop_at=None):
    """Drain stream_turn; stop_at(kind, n_audio) -> True sets the stop flag. Returns the event list."""
    stop = threading.Event(); evs = []; n_audio = 0
    for kind, val in sess.stream_turn(transcript, stop.is_set):
        evs.append((kind, val, time.perf_counter()))
        n_audio += kind == "audio"
        if stop_at and stop_at(kind, n_audio):
            stop.set()
    return evs, stop


# ---- LLM plugins: a call format's text, streamed --------------------------------------------------------------
def stream_speakable(raw, fmt, step=2):
    """What a plugin would speak of `raw` in call format `fmt`, fed in `step`-character deltas: (spoken, saw a
    call). Checks that nothing spoken is ever taken back."""
    from app.plugins.llm.calls import split_speakable
    said, buf, saw = "", "", False
    for i in range(0, len(raw), step):
        buf += raw[i:i + step]
        if saw:
            continue
        speak, saw = split_speakable(buf, fmt)
        assert speak.startswith(said), (speak, said)
        if len(speak) > len(said):
            said = speak
    if not saw:
        said = buf
    return said, saw


def scripted_stream(plugin_cls, raw, tools=None, step=3):
    """A local plugin's (VLLMChatPlugin) stream without an engine: the model's raw text arrives in `step`-character
    deltas. Returns the events."""
    p = plugin_cls.__new__(plugin_cls)
    p._tok_lock = threading.Lock(); p.max_tokens = None
    p.tok = type("Tok", (), {"apply_chat_template": lambda self, *a, **k: "prompt"})()
    p.sampling = lambda max_new: None
    p.engine = type("Engine", (), {"abort": lambda self, rid: None})()
    p.eloop = type("Loop", (), {"iterate": lambda self, gen, abort=None, stop_flag=None:
                                iter([raw[i:i + step] for i in range(0, len(raw), step)])})()
    return list(p.stream([{"role": "user", "content": "x"}], tools=tools))


# ---- EARS: the page and MOUTH around a Conversation -----------------------------------------------------------
class Msg:
    def __init__(self, kind, data):
        self.type = kind; self.data = data


class FakePage:
    """The browser's side of EARS's WebSocket, as static/index.html behaves (I-17): mic blocks go in tagged with the
    reply sample this page is playing when the block is pushed (or an explicit tag); reply audio "plays" in real time
    from its arrival, back to back (LEAD_IN_S when nothing is queued); when a reply's queue drains it reports
    `played:<id>:<samples>` (unless report=False); a barge or endpoint stops playback. `downlink_s` delays playback as
    a slow network would. Received JSON is in `sent`, audio as ("audio", id, samples)."""
    def __init__(self, report=True, downlink_s=0.0):
        self.q = asyncio.Queue(); self.sent = []; self.closed = False; self.report = report
        self.downlink_s = downlink_s
        self.sched = []                 # (id, t0, t1, first sample): the chunks queued, on the loop's clock
        self.next_play = 0.0; self.queued = {}; self.timers = []

    def tag_now(self):
        """(id, sample at 24 kHz) of the reply playing now; (last id, -1) when nothing is."""
        now = asyncio.get_running_loop().time()
        for rid, t0, t1, first in self.sched:
            if t0 <= now < t1:
                return rid, first + int((now - t0) * 24000)
        return (self.sched[-1][0] if self.sched else 0), -1

    def push(self, f32, tag=None):
        rid, at24 = tag if tag is not None else self.tag_now()
        self.q.put_nowait(Msg(WSMsgType.BINARY, MIC_HEADER.pack(rid, at24) + to_pcm16(np.asarray(f32)).tobytes()))

    def command(self, text):
        self.q.put_nowait(Msg(WSMsgType.TEXT, text))

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.q.get()
        if m is None:
            raise StopAsyncIteration
        return m

    async def send_json(self, d):
        self.sent.append(d)
        if d.get("type") in ("barge", "endpoint"):        # the page stops playback
            self.sched = []; self.next_play = 0.0
            for h in self.timers:
                h.cancel()
            self.timers = []

    async def send_bytes(self, b):
        rid = AUDIO_HEADER.unpack_from(b)[0]; n = (len(b) - AUDIO_HEADER.size) // 2
        self.sent.append(("audio", rid, n))
        if rid == BACKCHANNEL_ID:
            return                                         # played, never tagged
        loop = asyncio.get_running_loop(); now = loop.time()
        if self.next_play < now + self.downlink_s:
            self.next_play = now + self.downlink_s + LEAD_IN_S
        t0 = self.next_play; t1 = t0 + n / 24000; self.next_play = t1
        first = self.queued.get(rid, 0); self.queued[rid] = first + n
        self.sched.append((rid, t0, t1, first))
        if self.report:
            self.timers.append(loop.call_later(t1 - now, self._ended, rid, t1))

    def _ended(self, rid, t1):
        if self.next_play <= t1 + 1e-6:                    # the last chunk queued: the reply has played out
            self.command(f"played:{rid}:{self.queued[rid]}")

    async def close(self):                                 # like aiohttp: closing ends the message loop
        self.closed = True; self.q.put_nowait(None)

    def types(self):
        return [x["type"] for x in self.sent if isinstance(x, dict) and x["type"] not in ("state",)]


class MouthServer:
    """A TCP server standing in for MOUTH: records what EARS sends, and lets the test send F / A / D."""
    def __init__(self):
        self.got = []; self.writer = None; self.connected = asyncio.Event()

    async def start(self):
        self.srv = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.srv.sockets[0].getsockname()[1]

    async def handle(self, reader, writer):
        self.writer = writer; self.connected.set()
        try:
            while True:
                kind, payload = await async_recv_msg(reader)
                if kind is None:
                    break
                self.got.append((kind, json.loads(payload) if payload else {}))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass

    async def send(self, kind, obj):
        await async_send_msg(self.writer, kind, obj)

    def kinds(self):
        return [k for k, _ in self.got]

    async def drop(self):
        """MOUTH goes away (restart / crash): close its end of the connection."""
        self.writer.close(); await self.writer.wait_closed()
