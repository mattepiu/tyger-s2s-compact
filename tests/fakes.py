"""Driving MOUTH's Session without a GPU (tests only). FakeLLM / FakeTTS are the `fake` plugins (app/plugins/*/
fake.py); FakeMouth carries the attributes Session reads from Mouth. Test text is synthetic.
"""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mouth as M  # noqa: E402
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.plugins.llm.fake import FakeLLM, wait_or_stop  # noqa: E402,F401
from app.plugins.tts.fake import FakeTTS  # noqa: E402

PROMPTS = default_prompts()     # the repo's prompts.yaml; FakeLLM routes requests by their system prompt


class FakeMouth:
    def __init__(self, llm=None, tts=None):
        self.llm = llm or FakeLLM(); self.tts = tts or FakeTTS()
        self.max_gen = 192; self.history_turns = 8; self.bc_min_words = 4; self.bc_max_new = 12
        self.temperature = 0.7; self.sr = self.tts.out_sr
        self.prompts = {"reply": PROMPTS, "filler": PROMPTS}; self.tools = build_tools(PROMPTS)
        self.llms = {"reply": self.llm, "filler": self.llm}

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


def make_session(llm=None, tts=None):
    m = FakeMouth(llm, tts)
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
