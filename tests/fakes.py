"""Fake engines for driving MOUTH's Session without a GPU (tests only).

FakeLLM / FakeTTS mimic LocalLLM.stream / VoxCPM2TTS.stream: generators that poll stop_flag between items, as
EngineLoop.iterate does. FakeMouth carries the attributes Session reads from Mouth. Test text is synthetic.
"""
import os, sys, threading, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mouth as M  # noqa: E402
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402

PROMPTS = default_prompts()     # the repo's prompts.yaml; FakeLLM routes requests by their system prompt


def wait_or_stop(seconds, stop_flag, poll=0.002):
    """Sleep `seconds`, returning early (True) if stop_flag() turns true, like the engine poll loop."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if stop_flag and stop_flag():
            return True
        time.sleep(poll)
    return bool(stop_flag and stop_flag())


class FakeLLM:
    """Routes by prompt: lead-in / backchannel / phase B (a tool message present) / main reply.
    `reply` is the main reply's raw text (may contain a Hermes <tool_call>); each call is recorded."""
    def __init__(self, reply="Sure, here is a reasonably long first sentence for you. And a second one.",
                 phase_b="It is sunny and twenty degrees there today, a lovely day.",
                 leadin="One sec, let me look that up.", ack="oh nice", step_s=0.002, chars=4):
        self.reply, self.phase_b, self.leadin, self.ack = reply, phase_b, leadin, ack
        self.step_s, self.chars = step_s, chars
        self.calls = []                       # (kind, messages) per request
        self.live = 0; self.aborted = 0; self.lock = threading.Lock()

    def _kind(self, messages):
        sys_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if sys_txt.startswith(PROMPTS.text("tool_leadin", "system")[:40]):
            return "leadin"
        if sys_txt.startswith(PROMPTS.text("backchannel", "system")[:40]):
            return "bc"
        if sys_txt.startswith(PROMPTS.text("still_looking", "system")[:40]):
            return "still"
        if messages[-1]["role"] == "tool":
            return "phase_b"
        return "reply"

    def text_for(self, kind):
        return {"leadin": self.leadin, "bc": self.ack, "phase_b": self.phase_b, "reply": self.reply,
                "still": "Still looking, almost there."}[kind]

    def stream(self, messages, tools=None, max_new=192, temperature=None, stop_flag=None):
        from app.plugins.llm._toolparse import split_speakable, parse_calls, new_call_id
        kind = self._kind(messages)
        self.calls.append((kind, [dict(m) for m in messages]))
        raw_all = self.text_for(kind)
        with self.lock:
            self.live += 1
        raw, said, saw_tool, started = "", 0, False, False
        try:
            for i in range(0, len(raw_all), self.chars):
                if wait_or_stop(self.step_s, stop_flag):
                    with self.lock:
                        self.aborted += 1
                    return
                raw += raw_all[i:i + self.chars]
                if saw_tool:
                    continue
                speak, saw_tool = split_speakable(raw)
                if len(speak) > said:
                    yield ("text", speak[said:]); said = len(speak)
                if saw_tool and not started and getattr(self, "emit_tool_start", True):
                    started = True
                    yield ("tool_start", {"name": None})
            if stop_flag and stop_flag():
                return
            if not saw_tool and len(raw) > said:
                yield ("text", raw[said:])
            if saw_tool:
                calls = parse_calls(raw, "hermes_json")
                yield ("tool_calls", [{"id": new_call_id(), **c} for c in calls], None)
        finally:
            with self.lock:
                self.live -= 1


class FakeTTS:
    """Yields `n` chunks of `chunk_s` audio per request, `step_s` apart; records each request's text."""
    def __init__(self, out_sr=24000, n=5, chunk_s=0.08, step_s=0.002, onset_s=0.0):
        self.out_sr, self.n, self.chunk_s, self.step_s, self.onset_s = out_sr, n, chunk_s, step_s, onset_s
        self.requests = []; self.aborted = 0; self.live = 0; self.lock = threading.Lock()

    def stream(self, text, voice=None, stop_flag=None):
        self.requests.append(text)
        with self.lock:
            self.live += 1
        try:
            if wait_or_stop(self.onset_s, stop_flag):
                with self.lock:
                    self.aborted += 1
                return
            for _ in range(self.n):
                if wait_or_stop(self.step_s, stop_flag):
                    with self.lock:
                        self.aborted += 1
                    return
                yield np.full(int(self.chunk_s * self.out_sr), 0.1, np.float32)
        finally:
            with self.lock:
                self.live -= 1


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
