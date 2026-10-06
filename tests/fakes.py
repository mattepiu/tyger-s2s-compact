"""Driving MOUTH's Session without a GPU (tests only). FakeLLM / FakeTTS are the `fake` plugins (app/plugins/*/
fake.py); FakeMouth carries the attributes Session reads from Mouth. Test text is synthetic.
"""
import os, sys, threading, time
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mouth as M  # noqa: E402
from app import tools as T  # noqa: E402
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.plugins.llm.fake import FakeLLM, wait_or_stop  # noqa: E402,F401
from app.plugins.tts.fake import FakeTTS  # noqa: E402
from app.tasks import TaskSlots  # noqa: E402

PROMPTS = default_prompts()     # the repo's prompts.yaml; FakeLLM routes requests by their system prompt


@contextmanager
def stub_tools(fn):
    """Every tool call (MOUTH's and background tasks') goes to fn(name, args) instead of the real executors."""
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
    TaskSlots between FakeMouths to test the global cap."""
    def __init__(self, llm=None, tts=None, smart=None, context_turns=4, max_steps=3, tasks=None, slots=None):
        self.llm = llm or FakeLLM(); self.tts = tts or FakeTTS()
        self.max_gen = 192; self.history_turns = 8; self.bc_min_words = 4; self.bc_max_new = 12
        self.temperature = 0.7; self.sr = self.tts.out_sr
        self.prompts = {"reply": PROMPTS, "filler": PROMPTS}
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
        return None


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
