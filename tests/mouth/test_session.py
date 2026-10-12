"""MOUTH Session flow with fake engines (no GPU): barge-in history (I-01), barge during tools (I-03), backchannel
cancelled by a turn (I-07). Run: .venv/bin/python tests/mouth/test_session.py
"""
import os, sys, json, socket, threading, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from fakes import FakeLLM, FakeTTS, FakeMouth, PROMPTS, make_session, run_turn, stub_tools, slow_tools  # noqa: E402
from app import mouth as M  # noqa: E402
from app.mouth.messages import reply_prefix  # noqa: E402
from app.mouth.turns import MAX_STILL, FILLER_GAP_S  # noqa: E402
from app.wire import send_msg, recv_msg  # noqa: E402

TOOL_REPLY = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'



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
    assert [h["role"] for h in s.history.messages] == ["user", "assistant"]
    assert s.history.messages[0]["content"] == "first question"


def test_first_chunk_rule():
    """mouth.first_chunk: `clause` sends the first sentence's start as its own TTS request (audio sooner), `sentence`
    waits for the whole sentence (pipecat's rule), so VoxCPM2's intonation doesn't restart mid-sentence."""
    reply = "It is 3.5 km away, about ten minutes on foot, and the road is fine today. Take the coast path then."
    for rule, want in (("clause", ["It is 3.5 km away, about ten minutes on foot,",      # < MIN_CHARS: merged
                                   "and the road is fine today. Take the coast path then."]),
                       ("sentence", ["It is 3.5 km away, about ten minutes on foot, and the road is fine today.",
                                     "Take the coast path then."])):
        s, m = make_session(FakeLLM(reply=reply), first_chunk=rule)
        evs, _ = run_turn(s, "how far is the beach")
        assert evs[-1][0] == "done" and m.tts.requests == want, (rule, m.tts.requests)


def test_barged_plain_turn_remembered():
    llm = FakeLLM(reply="Here is the first part of a long answer, which goes on. Then the second sentence comes. "
                        "And a third sentence that nobody will ever hear because they cut in.")
    s, m = make_session(llm, FakeTTS(n=4))
    evs, _ = run_turn(s, "what's the weather on Friday?", stop_at=lambda k, n: n == 2)   # barge mid chunk 1
    assert evs[-1][0] == "done"
    roles = [h["role"] for h in s.history.messages]
    assert roles == ["user", "assistant"], roles
    assert s.history.messages[0]["content"] == "what's the weather on Friday?"
    a = s.history.messages[1]["content"]
    assert a.startswith("Here is the first part") and a.endswith(PROMPTS.text("markers", "interrupted")), a
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
    a = s.history.messages[1]["content"]
    assert "first sentence" in a and "second sentence" not in a, a
    assert a.endswith(PROMPTS.text("markers", "interrupted"))


def test_barge_before_any_audio_keeps_user():
    llm = FakeLLM(reply="A reply that will be cut before any audio at all is played, sadly for everyone.")
    s, m = make_session(llm, FakeTTS(n=3, onset_s=0.5))
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()
    list(s.stream_turn("hello?", stop.is_set))
    assert [h["role"] for h in s.history.messages] == ["user", "assistant"]
    assert s.history.messages[1]["content"] == PROMPTS.text("markers", "interrupted").strip()


# ---- I-03: barge during a tool call doesn't wait for the tool ------------------------------------------
def test_barge_during_slow_tool_returns_fast():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with slow_tools(10.0):
        stop = threading.Event(); t_stop = None; t_done = None
        for kind, val in s.stream_turn("weather in Paris?", stop.is_set):
            if kind == "audio" and t_stop is None:          # filler is playing -> barge
                stop.set(); t_stop = time.perf_counter()
            if kind == "done":
                t_done = time.perf_counter()
    assert t_stop is not None and t_done is not None
    assert t_done - t_stop < 0.2, f"done {1e3 * (t_done - t_stop):.0f} ms after barge"
    assert s.history.messages and s.history.messages[0]["content"] == "weather in Paris?"


def test_tool_turn_completes():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with slow_tools(0.05):
        evs, _ = run_turn(s, "weather in Paris?")
    kinds = [k for k, _, _ in evs]
    assert kinds[-1] == "done" and kinds.count("audio") >= 6, kinds      # filler + answer
    assert [c[0] for c in llm.calls][:1] == ["reply"] and "phase_b" in [c[0] for c in llm.calls]
    assert s.history.messages[-1]["content"].startswith("It is sunny")


# ---- I-12: tool calls stay in history, trimmed by whole turns ----------------------------------------------
def test_tool_exchange_kept_in_history():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=2))
    with stub_tools(lambda n, a: {"location": "Paris", "answer": "sunny, 20 C",
                                   "sources": [{"content": "x" * 2000}]}):
        run_turn(s, "weather in Paris?")
    assert [h["role"] for h in s.history.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.history.messages[1]["tool_calls"][0]["name"] == "get_weather"   # neutral format: {"id", "name", "arguments"}
    tool = s.history.messages[2]["content"]
    assert "sunny, 20 C" in tool and "xxxx" not in tool, tool       # raw search hits dropped
    run_turn(s, "and in Rome?")                                     # the next prompt carries the exchange
    nxt = next(c[1] for c in reversed(llm.calls) if c[0] == "reply")
    assert any(x["role"] == "tool" for x in nxt)


def test_tool_call_ids_unique_and_matched():
    """Plugin-issued IDs: each tool result answers its own call, and IDs never repeat across turns."""
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=2))
    with slow_tools(0):
        for q in ("weather in Paris?", "and in Rome?", "and in Oslo?"):
            run_turn(s, q)
    calls = [c for h in s.history.messages if h.get("tool_calls") for c in h["tool_calls"]]
    results = [h["tool_call_id"] for h in s.history.messages if h["role"] == "tool"]
    assert len(calls) == 3 and len({c["id"] for c in calls}) == 3
    assert results == [c["id"] for c in calls]


def test_history_trimmed_by_whole_turns():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=1))
    with slow_tools(0.0):
        for i in range(m.history_turns + 3):
            run_turn(s, f"weather question {i}")
    assert len(s.history.turns) == m.history_turns
    assert s.history.messages[0]["role"] == "user" and s.history.messages[0]["content"] == "weather question 3"


# ---- L-04: filler starts at the <tool_call> tag, covers slow tools, stops on barge ---------------------------
def dead_air(evs, sr=24000):
    """Longest gap in a client playback queue fed by the turn's audio events (as scripts/stack/load.py measures it)."""
    end = None; gap = 0.0
    for kind, val, t in evs:
        if kind == "audio":
            if end is not None:
                gap = max(gap, t - end)
            end = max(end or t, t) + len(val) / sr
    return gap


LONG_CALL = ('<tool_call>\n{"name": "web_search", "arguments": {"query": "the opening hours of the old town '
             'museum in Paris this coming weekend"}}\n</tool_call>')


def test_filler_starts_before_call_finishes_decoding():
    llm = FakeLLM(reply=LONG_CALL, step_s=0.01, chars=2)     # named at ~0.15 s, decoded at ~0.75 s
    s, m = make_session(llm, FakeTTS(n=3))
    t_tool = []
    with stub_tools(lambda n, a: (t_tool.append(time.perf_counter()), {"answer": "sunny"})[1]):
        t0 = time.perf_counter()
        evs, _ = run_turn(s, "weather in Paris?")
    first_audio = next(t for k, _, t in evs if k == "audio") - t0
    assert len(t_tool) == 1 and 0.6 < t_tool[0] - t0 < 1.1, "tool should start as soon as the call is parsed"
    assert first_audio < t_tool[0] - t0 - 0.2, \
        f"filler first audio {1e3 * first_audio:.0f} ms, call parsed at {1e3 * (t_tool[0] - t0):.0f} ms"
    assert evs[-1][0] == "done" and "phase_b" in [c[0] for c in llm.calls]


def test_still_filler_covers_slow_tool():
    llm = FakeLLM(reply=TOOL_REPLY)
    tts = FakeTTS(n=5, chunk_s=0.08, step_s=0.01)           # each line: 0.4 s of audio
    s, m = make_session(llm, tts)
    with slow_tools(3.0):
        evs, _ = run_turn(s, "weather in Paris?")
    kinds = [c[0] for c in llm.calls]
    assert kinds.count("still") == MAX_STILL, kinds
    first_still = next(c[1] for c in llm.calls if c[0] == "still")
    assert llm.leadin in first_still[-1]["content"], "the still line must know what was already said"
    gap = dead_air(evs)
    assert gap < FILLER_GAP_S + 0.3, f"dead air {1e3 * gap:.0f} ms"


def test_no_still_filler_for_fast_tool():
    llm = FakeLLM(reply=TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=3))
    with slow_tools(0.05):
        run_turn(s, "weather in Paris?")
    assert "still" not in [c[0] for c in llm.calls]


def test_barge_during_filler_halts_within_a_chunk():
    llm = FakeLLM(reply=TOOL_REPLY)
    tts = FakeTTS(n=20, step_s=0.02)
    s, m = make_session(llm, tts)
    with slow_tools(3.0):
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
    with slow_tools(0.05):
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
    """A C that arrives after T (EARS never sends one, but scripts/stack/load.py does) is not cancelled by that T."""
    s, peer, m = live_session(FakeLLM(), FakeTTS(n=3))
    send_msg(peer, b"T", {"transcript": "tell me a story"})
    send_msg(peer, b"C", {"partial": "and then the dog ran off"})
    got = read_until(peer, lambda k, p: k == b"D" and isinstance(p, dict) and p.get("bc"))
    assert any(k == b"K" for k, _, _ in got) and not got[-1][1].get("aborted")
    peer.close()


def test_unspeakable_backchannel_never_reaches_the_tts():
    """An ack with nothing speakable in it is no ack: no TTS request, not remembered as a recent ack."""
    for ack in ("—", "*", " , "):
        s, m = make_session(FakeLLM(ack=ack), FakeTTS(n=2))
        evs = list(s.stream_bc("so I finally got the job", lambda: False))
        assert evs == [("done", "")] and m.tts.requests == [] and not s.recent_acks, (ack, evs)
    s, m = make_session(FakeLLM(ack="oh nice"), FakeTTS(n=2))
    evs = list(s.stream_bc("so I finally got the job", lambda: False))
    assert m.tts.requests == ["oh nice"] and list(s.recent_acks) == ["oh nice"] and evs[-1] == ("done", "oh nice")


# ---- hosted LLM failures: a spoken apology / neutral line, never silence; bad tool arguments never run --------
def spoken(evs):
    return [v for k, v, _ in evs if k == "text"]


def test_llm_error_speaks_apology_and_remembers():
    for fail_after in (0, 3):                                      # before any text, and mid-reply
        llm = FakeLLM(fail={"reply": "rate_limit"}, fail_after=fail_after)
        s, m = make_session(llm, FakeTTS(n=2))
        evs, _ = run_turn(s, "what's up?")
        line = PROMPTS.text("spoken", "error")
        assert spoken(evs)[-1] == line and evs[-1][:2] == ("done", line)
        assert any(k == "audio" for k, _, _ in evs) and m.tts.requests[-1] == line
        assert [h["role"] for h in s.history.messages] == ["user", "assistant"] and s.history.messages[1]["content"] == line


def test_llm_error_while_tool_call_decodes():
    llm = FakeLLM(reply=TOOL_REPLY, fail={"reply": "timeout"}, fail_after=4)   # dies after the tag, mid-call
    s, m = make_session(llm, FakeTTS(n=2))
    evs, _ = run_turn(s, "weather in Paris?")
    assert evs[-1][:2] == ("done", PROMPTS.text("spoken", "error"))


def test_refusal_speaks_neutral_line():
    llm = FakeLLM(refuse=("reply",))
    s, m = make_session(llm, FakeTTS(n=2))
    evs, _ = run_turn(s, "something it won't do")
    assert evs[-1][:2] == ("done", PROMPTS.text("spoken", "refusal"))
    assert s.history.messages[0]["content"] == "something it won't do"


def test_flagged_tool_call_never_runs():
    class Flagging(FakeLLM):
        def stream(self, messages, **kw):
            for ev in super().stream(messages, **kw):
                if ev[0] == "tool_calls":
                    ev = ("tool_calls", [{**c, "error": "invalid arguments for 'get_weather': missing location"}
                                         for c in ev[1]], ev[2])
                yield ev
    ran = []
    s, m = make_session(Flagging(reply=TOOL_REPLY), FakeTTS(n=2))
    with stub_tools(lambda n, a: ran.append(n) or {"answer": "sunny"}):
        run_turn(s, "weather?")
    assert ran == [] and "invalid arguments" in s.history.messages[2]["content"]


# ---- escalation (step 10): ask_expert, answered by the smart model straight to the TTS ----------------------
ASK = ('<tool_call>\n{"name": "ask_expert", "arguments": {"question": "How much would 200 dollars a month in an '
       'index fund grow to over 20 years?"}}\n</tool_call>')


def kinds(llm):
    return [c[0] for c in llm.calls]


def test_no_expert_without_smart():
    """Without a smart model the reply prompt and tools are exactly the local-only ones (parity)."""
    s, m = make_session(FakeLLM())
    run_turn(s, "hello")
    msgs = s.m.llm.calls[0][1]
    ori = m.characters.default
    assert msgs[0]["content"] == PROMPTS.system(ori, "system", "voice_tools") and msgs[-1]["content"] == "hello"
    assert msgs[:-1] == reply_prefix(PROMPTS, ori, False)           # system + the lookup few-shot turns
    assert not any(c["name"] == "ask_expert" for x in msgs if x.get("tool_calls") for c in x["tool_calls"])
    assert "ask_expert" not in [t["function"]["name"] for t in m.tools]


def persona_of(msgs):
    """The character whose text opens a request's system prompt."""
    return next(c for c in PROMPTS.characters() if msgs[0]["content"].startswith(PROMPTS.d["persona"]["characters"][c]["text"]))


def test_voice_switch_changes_the_character_next_turn():
    """The page picks another voice (V): the next turn's reply and expert requests open with that voice's character
    (voices.characters), and the conversation's history stays."""
    llm, smart = FakeLLM(), FakeLLM()
    s, m = make_session(llm, FakeTTS(n=2), smart=smart, voices={"character": "ori", "characters": {"alt": "mike"}})
    run_turn(s, "hello there")
    assert persona_of(llm.calls[-1][1]) == "ori"
    s._dispatch(b"V", json.dumps({"voice": "alt"}).encode())
    assert s.voice_name == "alt" and s.character == "mike"
    llm.text["reply"] = ASK                                        # this turn escalates: the expert speaks as Mike too
    run_turn(s, "and how much would that be?")
    reply = next(msgs for kind, msgs in llm.calls if kind == "reply" and msgs[-1]["content"].startswith("and how"))
    assert persona_of(reply) == "mike" and persona_of(smart.calls[-1][1]) == "mike"
    assert any(x["role"] == "user" and x["content"] == "hello there" for x in reply), "history kept"


def test_unlisted_voice_gets_the_default_character():
    llm = FakeLLM()
    s, m = make_session(llm, voices={"character": "ori", "characters": {"alt": "mike"}})
    s.set_voice("alt"); run_turn(s, "hi")
    assert persona_of(llm.calls[-1][1]) == "mike"
    s.set_voice("new"); run_turn(s, "hi again")                    # a voice with no entry: the default character
    assert s.character == "ori" and persona_of(llm.calls[-1][1]) == "ori"


def test_expert_prompt_has_shots_and_tool():
    s, m = make_session(FakeLLM(), smart=FakeLLM())
    run_turn(s, "hello")
    msgs = s.m.llm.calls[0][1]
    assert msgs[0]["content"] == PROMPTS.system(m.characters.default, "escalation", "system")
    calls = [c for x in msgs if x.get("tool_calls") for c in x["tool_calls"]]
    asks = [c for c in calls if c["name"] == "ask_expert"]
    assert asks and len(asks) < len(calls) and msgs[-1]["content"] == "hello"   # lookup shots come first
    assert len({c["id"] for c in calls}) == len(calls)
    assert [t["function"]["name"] for t in m.tools][-1] == "ask_expert"
    assert "ask_expert" not in [t["function"]["name"] for t in m.smart_tools]
    assert s.m.llms["smart"].calls == [], "a plain turn must not reach the smart model"


def test_expert_turn_answered_by_smart():
    llm, smart = FakeLLM(reply=ASK), FakeLLM()
    s, m = make_session(llm, FakeTTS(n=2), smart=smart)
    evs, _ = run_turn(s, "what do I get from 200 a month over 20 years?")
    assert "expert_leadin" in kinds(llm) and "leadin" not in kinds(llm), kinds(llm)
    assert "phase_b" not in kinds(llm), "the reply model must not answer an escalated turn"
    assert kinds(smart) == ["smart"]
    req = smart.calls[0][1]
    assert req[0]["content"] == PROMPTS.system(m.characters.default, "smart", "system")
    assert req[-1]["content"].startswith("what do I get from 200") and "index fund" in req[-1]["content"]
    assert evs[-1][1].endswith(smart.smart) and evs[-1][1].startswith(llm.expert_leadin)
    assert [h["role"] for h in s.history.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.history.messages[1]["tool_calls"][0]["name"] == "ask_expert" and s.history.messages[3]["content"] == smart.smart
    first_id = s.history.messages[1]["tool_calls"][0]["id"]
    run_turn(s, "and if I doubled it?")                 # the follow-up sees the escalation in history
    follow = [c[1] for c in llm.calls if c[0] == "reply"][-1]
    assert any(x.get("tool_calls") and x["tool_calls"][0]["id"] == first_id for x in follow)


def test_smart_request_starts_under_the_leadin():
    """The smart request goes out as soon as the call parses, not after the lead-in has been spoken."""
    t = {}

    class Timed(FakeLLM):
        def stream(self, messages, **kw):
            t.setdefault("smart", time.perf_counter())
            yield from super().stream(messages, **kw)
    llm = FakeLLM(reply=ASK, chars=8)
    s, m = make_session(llm, FakeTTS(n=10, step_s=0.03), smart=Timed())       # lead-in audio takes 0.3 s
    evs, _ = run_turn(s, "what do I get from 200 a month over 20 years?")
    leadin_audio = [tt for k, v, tt in evs if k == "audio"]
    assert t["smart"] < leadin_audio[5], "smart request waited for the lead-in"


def test_expert_still_covers_a_slow_smart_model():
    """A smart model slow to its first words (thinking): expert still-looking lines, which know the lead-in."""
    llm = FakeLLM(reply=ASK, chars=16)
    smart = FakeLLM(step_s=3.0, chars=80)                        # first text after 3 s
    tts = FakeTTS(n=3, chunk_s=0.08, step_s=0.005)               # each line: 0.24 s of audio
    s, m = make_session(llm, tts, smart=smart)
    evs, _ = run_turn(s, "plan a three day trip to Rome")
    assert kinds(llm).count("expert_still") == MAX_STILL and "still" not in kinds(llm), kinds(llm)
    first = next(c[1] for c in llm.calls if c[0] == "expert_still")
    assert llm.expert_leadin in first[-1]["content"]
    assert evs[-1][1].endswith(smart.smart)


def test_smart_tool_round():
    llm = FakeLLM(reply=ASK)
    smart = FakeLLM(smart="Let me check that. " + TOOL_REPLY)
    s, m = make_session(llm, FakeTTS(n=2), smart=smart)
    ran = []
    with stub_tools(lambda n, a: ran.append(n) or {"answer": "sunny"}):
        evs, _ = run_turn(s, "should I bike to work in Paris tomorrow?")
    assert ran == ["get_weather"] and kinds(smart) == ["smart", "smart_b"]
    second = smart.calls[1][1]
    assert second[-2]["tool_calls"][0]["name"] == "get_weather" and "sunny" in second[-1]["content"]
    assert evs[-1][1].endswith(smart.smart_b) and s.history.messages[-1]["content"].endswith(smart.smart_b)
    assert "phase_b" not in kinds(llm)


def test_smart_tool_rounds_capped():
    llm = FakeLLM(reply=ASK)
    smart = FakeLLM(smart=TOOL_REPLY, smart_b=TOOL_REPLY)         # never stops calling tools
    s, m = make_session(llm, FakeTTS(n=2), smart=smart, max_steps=1)
    with slow_tools(0):
        evs, _ = run_turn(s, "keep checking")
    limit = PROMPTS.text("smart", "limit")
    assert kinds(smart) == ["smart", "smart_b", "smart_b"], kinds(smart)
    assert limit not in json.dumps(smart.calls[1][1], ensure_ascii=False)
    assert limit in json.dumps(smart.calls[2][1][-1], ensure_ascii=False)
    assert evs[-1][:2] == ("done", PROMPTS.text("spoken", "error"))


def test_other_tool_results_go_to_smart():
    both = ('<tool_call>\n{"name": "get_weather", "arguments": {"location": "Rome"}}\n</tool_call>\n' + ASK)
    llm, smart = FakeLLM(reply=both), FakeLLM()
    s, m = make_session(llm, FakeTTS(n=2), smart=smart)
    with stub_tools(lambda n, a: {"answer": "rainy, 14 C"}):
        run_turn(s, "is Rome nice for a walk tomorrow, and what should I see?")
    last = smart.calls[0][1][-1]["content"]
    assert "get_weather" in last and "rainy, 14 C" in last
    assert [c["name"] for c in s.history.messages[1]["tool_calls"]] == ["get_weather", "ask_expert"]


def test_smart_context_is_recent_plain_turns():
    llm, smart = FakeLLM(), FakeLLM()
    s, m = make_session(llm, FakeTTS(n=1), smart=smart, context_turns=2)
    with slow_tools(0):
        for q in ("first question", "second question"):
            run_turn(s, q)
        llm.text["reply"] = TOOL_REPLY; run_turn(s, "weather in Paris?")
        llm.text["reply"] = ASK; run_turn(s, "now the hard one")
    req = smart.calls[0][1]
    assert [x["role"] for x in req] == ["system", "user", "assistant", "user", "assistant", "user"], req
    assert req[1]["content"] == "second question" and req[3]["content"] == "weather in Paris?"
    assert all("tool_calls" not in x for x in req)


def test_barge_during_expert_answer():
    llm = FakeLLM(reply=ASK)
    smart = FakeLLM(smart="The first part of a careful answer is right here, friend. Then a second sentence "
                          "follows it. And a third one that is never heard.", step_s=0.01)
    s, m = make_session(llm, FakeTTS(n=4), smart=smart)
    evs, _ = run_turn(s, "explain compound interest", stop_at=lambda k, n: n == 6)   # 4 lead-in + 2 answer
    time.sleep(0.05)
    assert evs[-1][0] == "done" and smart.aborted >= 1 and smart.live == 0
    assert [h["role"] for h in s.history.messages] == ["user", "assistant", "tool", "assistant"]
    assert s.history.messages[-1]["content"].endswith(PROMPTS.text("markers", "interrupted"))
    assert "third one" not in s.history.messages[-1]["content"]


def test_smart_failure_and_refusal_are_spoken():
    for smart, key in ((FakeLLM(fail={"smart": "timeout"}), "error"), (FakeLLM(refuse=("smart",)), "refusal")):
        s, m = make_session(FakeLLM(reply=ASK), FakeTTS(n=2), smart=smart)
        evs, _ = run_turn(s, "a hard question")
        assert evs[-1][:2] == ("done", PROMPTS.text("spoken", key)), (key, evs[-1])
        assert s.history.messages[0]["content"] == "a hard question"


# ---- S: TTS only, for test tools (scripts/stack/load_ears.py's fixed utterance) --------------------------------------
def test_say_is_tts_only():
    s, peer, m = live_session(FakeLLM(), FakeTTS(n=3))
    send_msg(peer, b"S", {"text": "Could you count to ten for me?"})
    got = read_until(peer, lambda k, p: k == b"D")
    assert sum(k == b"A" for k, _, _ in got) == 3
    assert got[-1][1]["text"] == "Could you count to ten for me?" and not got[-1][1]["barged"]
    assert m.llm.calls == [] and m.tts.requests == ["Could you count to ten for me?"]
    assert s.history.messages == []
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
