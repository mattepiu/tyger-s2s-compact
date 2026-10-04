"""MOUTH Session flow with fake engines (no GPU): barge-in history (I-01), barge during tools (I-03), backchannel
cancelled by a turn (I-07). Run: .venv/bin/python tests/test_mouth_session.py
"""
import os, sys, json, socket, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import FakeLLM, FakeTTS, FakeMouth, make_session, run_turn  # noqa: E402
from app import mouth as M  # noqa: E402
from app import tools as T  # noqa: E402
from app.wire import send_msg, recv_msg  # noqa: E402

TOOL_REPLY = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'


class SlowTool:
    """Patch execute_tool with one that sleeps (polling nothing: a real blocking HTTP call)."""
    def __init__(self, seconds):
        self.seconds = seconds; self.orig = M.execute_tool

    def __enter__(self):
        M.execute_tool = lambda name, args: (time.sleep(self.seconds), {"answer": "sunny"})[1]
        return self

    def __exit__(self, *a):
        M.execute_tool = self.orig


def live_session(llm=None, tts=None):
    """A Session on one end of a socketpair, running Session.run on a thread; returns (sess, peer, mouth)."""
    m = FakeMouth(llm, tts)
    a, b = socket.socketpair()
    s = M.Session(m, a)
    threading.Thread(target=s.run, daemon=True).start()
    return s, b, m


def read_until(peer, pred, timeout=5.0):
    """Read frames until pred(kind, payload) is true; returns all frames read (kind, payload, t)."""
    peer.settimeout(timeout); got = []
    while True:
        kind, p = recv_msg(peer)
        if kind is None:
            return got
        if kind in (b"X", b"D"):
            p = json.loads(p)
        got.append((kind, p, time.perf_counter()))
        if pred(kind, p):
            return got


# ---- I-01: barged turns stay in history ------------------------------------------------------------
def test_plain_turn_remembered():
    s, _ = make_session()
    run_turn(s, "first question")
    assert [h["role"] for h in s.history] == ["user", "assistant"]
    assert s.history[0]["content"] == "first question"


def test_barged_plain_turn_remembered():
    llm = FakeLLM(reply="Here is the first part of a long answer, which goes on. Then the second sentence comes. "
                        "And a third sentence that nobody will ever hear because they cut in.")
    s, m = make_session(llm, FakeTTS(n=4))
    evs, _ = run_turn(s, "what's the weather on Friday?", stop_at=lambda k, n: n == 2)   # barge mid chunk 1
    assert evs[-1][0] == "done"
    roles = [h["role"] for h in s.history]
    assert roles == ["user", "assistant"], roles
    assert s.history[0]["content"] == "what's the weather on Friday?"
    a = s.history[1]["content"]
    assert a.startswith("Here is the first part") and a.endswith(M.INTERRUPTED), a
    assert "third sentence" not in a, a                     # never generated -> never remembered
    # the follow-up turn sees the first question
    run_turn(s, "no, Saturday")
    prompt = llm.calls[-1][1]
    assert any(x["content"] == "what's the weather on Friday?" for x in prompt if x["role"] == "user")


def test_barge_truncates_to_played_audio():
    """With EARS's played_s, only chunks that had started playing are remembered."""
    llm = FakeLLM(reply="The first sentence of the answer is right here, friend. The second sentence is also "
                        "fairly long and complete. The third sentence closes the reply for good.")
    s, m = make_session(llm, FakeTTS(n=5, chunk_s=0.1))      # 0.5 s of audio per chunk
    s.barge_played_s = 0.2                                   # heard 0.2 s: inside chunk 1
    run_turn(s, "tell me", stop_at=lambda k, n: n == 11)      # MOUTH had generated into chunk 3
    a = s.history[1]["content"]
    assert "first sentence" in a and "second sentence" not in a, a
    assert a.endswith(M.INTERRUPTED)


def test_barge_before_any_audio_keeps_user():
    llm = FakeLLM(reply="A reply that will be cut before any audio at all is played, sadly for everyone.")
    s, m = make_session(llm, FakeTTS(n=3, onset_s=0.5))
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()
    list(s.stream_turn("hello?", stop.is_set))
    assert [h["role"] for h in s.history] == ["user", "assistant"]
    assert s.history[1]["content"] == M.INTERRUPTED.strip()


# ---- I-03: barge during a tool call doesn't wait for the tool ------------------------------------------
def test_barge_during_slow_tool_returns_fast():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with SlowTool(10.0):
        stop = threading.Event(); t_stop = None; t_done = None
        for kind, val in s.stream_turn("weather in Paris?", stop.is_set):
            if kind == "audio" and t_stop is None:          # filler is playing -> barge
                stop.set(); t_stop = time.perf_counter()
            if kind == "done":
                t_done = time.perf_counter()
    assert t_stop is not None and t_done is not None
    assert t_done - t_stop < 0.2, f"done {1e3 * (t_done - t_stop):.0f} ms after barge"
    assert s.history and s.history[0]["content"] == "weather in Paris?"


def test_tool_turn_completes():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with SlowTool(0.05):
        evs, _ = run_turn(s, "weather in Paris?")
    kinds = [k for k, _, _ in evs]
    assert kinds[-1] == "done" and kinds.count("audio") >= 6, kinds      # filler + answer
    assert [c[0] for c in llm.calls][:1] == ["reply"] and "phase_b" in [c[0] for c in llm.calls]
    assert s.history[-1]["content"].startswith("It is sunny")


# ---- I-12: tool calls stay in history, trimmed by whole turns ----------------------------------------------
def test_tool_exchange_kept_in_history():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=2))
    orig = M.execute_tool
    M.execute_tool = lambda n, a: {"location": "Paris", "answer": "sunny, 20 C",
                                   "sources": [{"content": "x" * 2000}]}
    try:
        run_turn(s, "weather in Paris?")
    finally:
        M.execute_tool = orig
    assert [h["role"] for h in s.history] == ["user", "assistant", "tool", "assistant"]
    assert s.history[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    tool = s.history[2]["content"]
    assert "sunny, 20 C" in tool and "xxxx" not in tool, tool       # raw search hits dropped
    run_turn(s, "and in Rome?")                                     # the next prompt carries the exchange
    nxt = next(c[1] for c in reversed(llm.calls) if c[0] == "reply")
    assert any(x["role"] == "tool" for x in nxt)


def test_history_trimmed_by_whole_turns():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=1))
    with SlowTool(0.0):
        for i in range(m.history_turns + 3):
            run_turn(s, f"weather question {i}")
    assert len(s.turns) == m.history_turns
    assert s.history[0]["role"] == "user" and s.history[0]["content"] == "weather question 3"


# ---- L-04: filler starts at the <tool_call> tag, covers slow tools, stops on barge ---------------------------
def dead_air(evs, sr=24000):
    """Longest gap in a client playback queue fed by the turn's audio events (as load_test.py measures it)."""
    end = None; gap = 0.0
    for kind, val, t in evs:
        if kind == "audio":
            if end is not None:
                gap = max(gap, t - end)
            end = max(end or t, t) + len(val) / sr
    return gap


def test_filler_starts_before_call_finishes_decoding():
    llm = FakeLLM(reply=TOOL_REPLY, step_s=0.02, chars=2)    # the call takes ~0.8 s to decode
    s, m = make_session(llm, FakeTTS(n=3))
    t_tool = []
    orig = M.execute_tool
    M.execute_tool = lambda n, a: (t_tool.append(time.perf_counter()), {"answer": "sunny"})[1]
    try:
        t0 = time.perf_counter()
        evs, _ = run_turn(s, "weather in Paris?")
    finally:
        M.execute_tool = orig
    first_audio = next(t for k, _, t in evs if k == "audio") - t0
    assert len(t_tool) == 1 and 0.6 < t_tool[0] - t0 < 1.0, "tool should start as soon as the call is parsed"
    assert first_audio < t_tool[0] - t0 - 0.2, \
        f"filler first audio {1e3 * first_audio:.0f} ms, call parsed at {1e3 * (t_tool[0] - t0):.0f} ms"
    assert evs[-1][0] == "done" and "phase_b" in [c[0] for c in llm.calls]


def test_still_filler_covers_slow_tool():
    llm = FakeLLM(reply=TOOL_REPLY)
    tts = FakeTTS(n=5, chunk_s=0.08, step_s=0.01)           # each line: 0.4 s of audio
    s, m = make_session(llm, tts)
    with SlowTool(3.0):
        evs, _ = run_turn(s, "weather in Paris?")
    kinds = [c[0] for c in llm.calls]
    assert kinds.count("still") == M.MAX_STILL, kinds
    first_still = next(c[1] for c in llm.calls if c[0] == "still")
    assert llm.leadin in first_still[-1]["content"], "the still line must know what was already said"
    gap = dead_air(evs)
    assert gap < M.FILLER_GAP_S + 0.3, f"dead air {1e3 * gap:.0f} ms"


def test_no_still_filler_for_fast_tool():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with SlowTool(0.05):
        run_turn(s, "weather in Paris?")
    assert "still" not in [c[0] for c in llm.calls]


def test_barge_during_filler_halts_within_a_chunk():
    llm = FakeLLM(reply=TOOL_REPLY)
    tts = FakeTTS(n=20, step_s=0.02)
    s, m = make_session(llm, tts)
    with SlowTool(3.0):
        stop = threading.Event(); t_stop = None; after = 0
        t_done = None
        for kind, val in s.stream_turn("weather in Paris?", stop.is_set):
            if kind == "audio":
                if t_stop is None:
                    stop.set(); t_stop = time.perf_counter()
                else:
                    after += 1
            if kind == "done":
                t_done = time.perf_counter()
    assert after <= 1, f"{after} audio chunks after the barge"
    assert t_done - t_stop < 0.05
    time.sleep(0.05)
    assert tts.live == 0 and tts.aborted >= 1, "filler TTS still running after the barge"


def test_model_words_before_tag_replace_filler():
    llm = FakeLLM(reply="Sure, let me look up the current weather in Paris for you right now. " + TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with SlowTool(0.05):
        run_turn(s, "weather in Paris?")
    assert "leadin" not in [c[0] for c in llm.calls], "spoke both the model's own lead-in and a filler"


# ---- I-07: a reply turn cancels the backchannel in flight ------------------------------------------------
def test_turn_cancels_backchannel():
    llm = FakeLLM(ack="oh wow, that is great news really", step_s=0.03, chars=2)   # slow ack: ~0.5 s
    tts = FakeTTS(n=3)
    s, peer, m = live_session(llm, tts)
    send_msg(peer, b"C", {"partial": "so I finally got the job"})
    time.sleep(0.05)
    t0 = time.perf_counter()
    send_msg(peer, b"T", {"transcript": "and that's my news"})
    got = read_until(peer, lambda k, p: k == b"D" and isinstance(p, dict) and p.get("bc"))
    t_bc_done = got[-1][2] - t0
    assert t_bc_done < 0.15, f"backchannel finished {1e3 * t_bc_done:.0f} ms after the turn started"
    assert got[-1][1].get("aborted") is True
    assert not any(k == b"K" for k, _, _ in got), "backchannel audio after the turn started"
    peer.close()


def test_backchannel_alongside_turn_still_runs():
    """A C that arrives after T (EARS never sends one, but load_test does) is not cancelled by that T."""
    s, peer, m = live_session(FakeLLM(), FakeTTS(n=3))
    send_msg(peer, b"T", {"transcript": "tell me a story"})
    send_msg(peer, b"C", {"partial": "and then the dog ran off"})
    got = read_until(peer, lambda k, p: k == b"D" and isinstance(p, dict) and p.get("bc"))
    assert any(k == b"K" for k, _, _ in got) and not got[-1][1].get("aborted")
    peer.close()


# ---- S: TTS only, for test tools (load_test_ears.py's fixed utterance) --------------------------------------
def test_say_is_tts_only():
    s, peer, m = live_session(FakeLLM(), FakeTTS(n=3))
    send_msg(peer, b"S", {"text": "Could you count to ten for me?"})
    got = read_until(peer, lambda k, p: k == b"D")
    assert sum(k == b"A" for k, _, _ in got) == 3
    assert got[-1][1]["text"] == "Could you count to ten for me?" and not got[-1][1]["barged"]
    assert m.llm.calls == [] and m.tts.requests == ["Could you count to ten for me?"]
    assert s.history == []
    peer.close()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
