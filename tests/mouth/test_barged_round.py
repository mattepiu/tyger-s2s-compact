"""I-16, GPU-free: a tool or expert round the user cuts in on before its answer was heard goes on as a background task
of the conversation (announced at the next pause, cancellable), or, where no task can start, leaves its results in
history. Fake plugins, a slow fake tool / smart model, the wire over a socketpair. Test text is synthetic.
Run: .venv/bin/python tests/mouth/test_barged_round.py
"""
import os, sys, json, logging, socket, threading, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from fakes import FakeLLM, FakeTTS, FakeMouth, PROMPTS, make_session, run_turn, stub_tools  # noqa: E402
from app import mouth as M  # noqa: E402
from app.wire import send_msg, recv_msg  # noqa: E402

WEATHER = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
ASK = ('<tool_call>\n{"name": "ask_expert", "arguments": {"question": "How much would 200 dollars a month in an '
       'index fund grow to over 20 years?"}}\n</tool_call>')
CANCEL = '<tool_call>\n{"name": "cancel_task", "arguments": {"task_id": 1}}\n</tool_call>'
TASKS = {"enabled": True, "max_concurrent": 1, "max_per_session": 5, "max_wall_s": 30, "max_steps": 4}
SUNNY = {"location": "Paris", "answer": "sunny, 20 C"}
MARK = PROMPTS.text("markers", "interrupted")
NOTE = PROMPTS.text("tasks", "continued")


def slow_tool(seconds, ran):
    def tool(name, args):
        ran.append(name); time.sleep(seconds)
        return SUNNY
    return tool


def live(reply, smart=None, tasks=TASKS, tts=None):
    """A Session over a socketpair, run on its own thread: (session, peer, mouth, reply llm)."""
    llm = FakeLLM(reply=reply)
    m = FakeMouth(llm, tts or FakeTTS(n=4), smart=smart, tasks=tasks)
    a, b = socket.socketpair()
    s = M.Session(m, a)
    threading.Thread(target=s.run, daemon=True).start()
    b.settimeout(5)
    return s, b, m, llm


def frames(peer, until):
    """Frames (kind, payload) until until(kind, payload); JSON payloads decoded."""
    got = []
    while True:
        k, p = recv_msg(peer)
        if k is None:
            return got
        p = json.loads(p) if k in (b"X", b"D", b"F") else p
        got.append((k, p))
        if until(k, p):
            return got


def quiet_for(peer, seconds):
    """The frames that arrive within `seconds` (nothing announced: no F among them)."""
    peer.settimeout(seconds); got = []
    try:
        while True:
            k, p = recv_msg(peer)
            if k is None:
                break
            got.append((k, p))
    except socket.timeout:
        pass
    peer.settimeout(5)
    return got


def barge_in_leadin(peer, transcript, played_s=0.1, before=None):
    """Send the turn, wait for the lead-in's first audio (and `before()`), cut in; returns the turn's D."""
    send_msg(peer, b"T", {"transcript": transcript})
    frames(peer, lambda k, p: k == b"A")
    if before is not None:
        assert wait(before)
    send_msg(peer, b"B", {"played_s": played_s})
    return frames(peer, lambda k, p: k == b"D")[-1][1]


def wait(pred, timeout=5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def kinds(llm):
    return [c[0] for c in llm.calls]


def barged_turn(s, i=0):
    """The roles of history turn i and the turn itself."""
    turn = s.history.turns[i]
    return [x["role"] for x in turn], turn


# ---- the round goes on as a task: announced at the next pause ------------------------------------------------
def test_lookup_round_goes_on_as_a_task():
    """A slow lookup, cut in on during the lead-in: the round goes on as task #1, which waits for the lookup the turn
    started (not redone) and gets its result; F, then the announcement in the same session. History has the call."""
    ran = []; smart = FakeLLM()
    s, peer, m, llm = live(WEATHER, smart=smart)
    with stub_tools(slow_tool(0.6, ran)):
        d = barge_in_leadin(peer, "what's the weather in Paris?")
        assert d["barged"]
        assert frames(peer, lambda k, p: k == b"F")[-1][1] == {"task": 1}
        send_msg(peer, b"G", {"task": 1})
        got = frames(peer, lambda k, p: k == b"D")
    ann = got[-1][1]
    assert ann["proactive"] == 1 and not ann["barged"] and ann["text"].startswith("About your task")
    assert any(k == b"A" for k, _ in got)
    assert ran == ["get_weather"] and kinds(smart) == ["task"], (ran, kinds(smart))
    req = smart.calls[0][1][-1]["content"]
    assert req.startswith('The user said: "what\'s the weather in Paris?"') and "sunny, 20 C" in req
    assert req.endswith(PROMPTS.fill("tasks", "looked_up", results='get_weather: ' + json.dumps(SUNNY)))
    roles, turn = barged_turn(s)
    assert roles == ["user", "assistant", "tool", "assistant"], roles
    assert turn[1]["tool_calls"][0]["name"] == "get_weather"
    assert json.loads(turn[2]["content"]) == {"task_id": 1, "status": NOTE}
    assert turn[3]["content"] == llm.leadin + MARK                 # the lead-in, cut
    assert s.tasks.get(1).announced == "yes"
    peer.close()


def test_expert_round_goes_on_as_a_task():
    """Cut in on the expert lead-in while the smart model is still thinking: its request is aborted and the question
    re-runs as task #1 (the question restated by the reply model, and the user's words), announced at the pause."""
    smart = FakeLLM(step_s=0.3, chars=40)                       # slow to its first words
    s, peer, m, llm = live(ASK, smart=smart)
    d = barge_in_leadin(peer, "what do I get from 200 a month over 20 years?", before=lambda: smart.calls)
    assert d["barged"]
    assert frames(peer, lambda k, p: k == b"F")[-1][1] == {"task": 1}
    assert kinds(smart) == ["smart", "task"] and smart.aborted >= 1, kinds(smart)
    req = smart.calls[1][1][-1]["content"]
    assert "Task: How much would 200 dollars a month in an index fund grow" in req
    assert '"what do I get from 200 a month over 20 years?"' in req and "Already looked up" not in req
    roles, turn = barged_turn(s)
    assert roles == ["user", "assistant", "tool", "assistant"] and turn[1]["tool_calls"][0]["name"] == "ask_expert"
    assert json.loads(turn[2]["content"]) == {"task_id": 1, "status": NOTE}
    send_msg(peer, b"G", {"task": 1})
    assert frames(peer, lambda k, p: k == b"D")[-1][1]["text"].startswith("About your task")
    peer.close()


def test_never_mind_cancels_it():
    """"Never mind" after the cut-in: the reply model sees the task running in its status note and cancels it; the
    task never asks the smart model, and nothing is announced."""
    ran = []; smart = FakeLLM()
    s, peer, m, llm = live(WEATHER, smart=smart)
    with stub_tools(slow_tool(1.0, ran)):
        barge_in_leadin(peer, "what's the weather in Paris?")
        llm.text["reply"] = CANCEL
        send_msg(peer, b"T", {"transcript": "never mind"})
        frames(peer, lambda k, p: k == b"D")
        late = quiet_for(peer, 1.5)                            # the lookup comes back meanwhile
    status = [c[1] for c in llm.calls if c[0] == "reply"][-1][-1]["content"]
    assert status.startswith("[Background tasks") and "still working on it" in status and status.endswith("never mind")
    assert json.loads(s.history.messages[-2]["content"]) == {"task_id": 1, "status": "cancelled"}
    assert s.tasks.get(1).status == "cancelled" and not any(k == b"F" for k, _ in late)
    assert kinds(smart) == [] and m.task_slots.used == 0
    peer.close()


# ---- no task: the results stay in history, nothing is announced ----------------------------------------------
def test_tasks_off_results_go_to_history():
    """Tasks off (no smart model): history keeps the call, its result written in when the slow lookup comes back;
    nothing is announced."""
    ran = []
    s, peer, m, llm = live(WEATHER, smart=None, tasks=None)
    assert s.tasks is None
    with stub_tools(slow_tool(0.5, ran)):
        assert barge_in_leadin(peer, "what's the weather in Paris?")["barged"]
        roles, turn = barged_turn(s)
        assert roles == ["user", "assistant", "tool", "assistant"] and turn[1]["tool_calls"][0]["name"] == "get_weather"
        assert json.loads(turn[2]["content"]) == {"status": PROMPTS.text("tasks", "result_pending")}
        late = quiet_for(peer, 1.0)
    assert json.loads(turn[2]["content"]) == SUNNY and ran == ["get_weather"]
    assert not any(k == b"F" for k, _ in late)
    llm.text["reply"] = "OK."                                   # the next request carries the result
    send_msg(peer, b"T", {"transcript": "so?"})
    frames(peer, lambda k, p: k == b"D")
    assert any(x["role"] == "tool" and "sunny, 20 C" in x["content"] for x in llm.calls[-1][1])
    peer.close()


def test_cap_reached_results_go_to_history():
    """Tasks on, but this conversation's one running task fills max_concurrent: no task, the result goes to
    history (logged with the cap, counts only), and only the first task is ever announced."""
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())
    h = H(); logging.getLogger().addHandler(h); old = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        smart = FakeLLM(step_s=0.05, chars=2)                    # task #1 takes ~5 s
        s, m = make_session(FakeLLM(reply=WEATHER), FakeTTS(n=2), smart=smart, tasks=TASKS)
        s.tasks.start("Compare three budget laptops", "research laptops")
        with stub_tools(lambda n, a: SUNNY):
            run_turn(s, "what's the weather in Paris?", stop_at=lambda k, n: n == 1)
            turn = s.history.turns[-1]
            assert wait(lambda: json.loads(turn[2]["content"]) == SUNNY), turn[2]["content"]
    finally:
        logging.getLogger().removeHandler(h); logging.getLogger().setLevel(old)
    assert len(s.tasks.tasks) == 1
    assert "barged tool round: no task (cap_concurrent), results kept in history" in records
    assert not any("Paris" in r or "sunny" in r for r in records)
    s.tasks.close()


def test_expert_without_tasks_noted_in_history():
    """A smart model but tasks off: the expert's question can't go on, and its history result says the expert never
    answered (not the usual "the expert answered" note)."""
    s, m = make_session(FakeLLM(reply=ASK), FakeTTS(n=2), smart=FakeLLM(step_s=0.3, chars=40))
    assert s.tasks is None
    run_turn(s, "what do I get from 200 a month over 20 years?", stop_at=lambda k, n: n == 1)
    turn = s.history.turns[-1]
    assert turn[1]["tool_calls"][0]["name"] == "ask_expert"
    assert json.loads(turn[2]["content"]) == {"error": PROMPTS.text("escalation", "interrupted")}


# ---- where the line between "go on" and "the user heard it" falls ---------------------------------------------
def test_barge_while_the_call_still_decodes():
    """Cut in while the call's arguments are still decoding: the call decodes to the end (the barge doesn't abort the
    reply stream once a call started), so history and the task get the whole call."""
    long_call = ('<tool_call>\n{"name": "web_search", "arguments": {"query": "the opening hours of the old town '
                 'museum in Paris this coming weekend"}}\n</tool_call>')
    llm = FakeLLM(reply=long_call, step_s=0.01, chars=2)        # named at ~0.15 s, decoded at ~0.75 s
    s, m = make_session(llm, FakeTTS(n=2), smart=FakeLLM(), tasks=TASKS)
    with stub_tools(lambda n, a: {"answer": "nine to five"}):
        t0 = time.perf_counter()
        evs, _ = run_turn(s, "when is the museum open?", stop_at=lambda k, n: n == 1)
        t_done = evs[-1][2] - t0
    turn = s.history.turns[-1]
    assert turn[1]["tool_calls"][0]["arguments"]["query"].endswith("this coming weekend")
    assert json.loads(turn[2]["content"]) == {"task_id": 1, "status": NOTE}
    assert t_done < 1.2, f"turn ended {1e3 * t_done:.0f} ms after it started"
    assert wait(lambda: s.tasks.get(1).status == "done")
    assert "nine to five" in s.m.llms["smart"].calls[0][1][-1]["content"]


def test_answer_made_but_not_heard_goes_on():
    """The answer was generated (MOUTH runs ahead of playback) but the user cut in while the lead-in still played
    (played_s): none of the answer was heard, so the round goes on; its results, already back, are reused."""
    ran = []
    s, m = make_session(FakeLLM(reply=WEATHER), FakeTTS(n=4, chunk_s=0.08), smart=FakeLLM(), tasks=TASKS)
    s.barge_played_s = 0.1                                      # inside the lead-in (0.32 s)
    with stub_tools(lambda n, a: ran.append(n) or SUNNY):
        run_turn(s, "what's the weather in Paris?", stop_at=lambda k, n: n == 6)   # 2 chunks into the answer
        assert wait(lambda: s.tasks.get(1) is not None and s.tasks.get(1).status == "done")
    assert ran == ["get_weather"]
    assert json.loads(s.history.turns[-1][2]["content"]) == {"task_id": 1, "status": NOTE}
    assert "sunny, 20 C" in s.m.llms["smart"].calls[0][1][-1]["content"]


def test_still_line_over_the_experts_lookup_is_not_the_answer():
    """The smart model looks something up before saying anything; the user hears the still-looking line over that
    lookup and cuts in. None of the answer was heard, so the question goes on as a task."""
    llm, smart = FakeLLM(reply=ASK), FakeLLM(smart=WEATHER)    # the expert's first move: a lookup, no words
    s, m = make_session(llm, FakeTTS(n=4, chunk_s=0.08), smart=smart, tasks=TASKS)
    s.barge_played_s = 0.42                                     # the lead-in (0.32 s), then 0.1 s of the still line
    with stub_tools(lambda n, a: (time.sleep(3.0), SUNNY)[1]):
        run_turn(s, "should I bike to work in Paris tomorrow?", stop_at=lambda k, n: n == 5)
        assert kinds(llm) == ["reply", "expert_leadin", "still"], kinds(llm)
        assert s.tasks.get(1) is not None
        assert json.loads(s.history.turns[-1][2]["content"]) == {"task_id": 1, "status": NOTE}
    s.tasks.close()


def test_part_of_the_answer_heard_stays_as_it_was():
    """The user heard the start of the answer: no task (nothing is said later); history keeps the compact result and
    what was heard, as before I-16."""
    s, m = make_session(FakeLLM(reply=WEATHER), FakeTTS(n=4, chunk_s=0.08), smart=FakeLLM(), tasks=TASKS)
    s.barge_played_s = 0.4                                      # 0.08 s into the answer
    with stub_tools(lambda n, a: SUNNY):
        run_turn(s, "what's the weather in Paris?", stop_at=lambda k, n: n == 6)
    turn = s.history.turns[-1]
    assert s.tasks.tasks == [] and json.loads(turn[2]["content"]) == SUNNY
    heard = turn[3]["content"]                                  # the lead-in, then the start of the answer, cut
    assert heard.startswith(m.llm.leadin + " It is sunny") and heard.endswith(MARK), heard


def test_task_tools_round_needs_no_task():
    """A round of start_task alone, cut in on before it ran: it still runs (a barge stops what is said, not what was
    asked), and its result is written into history when it is back; no second task."""
    start = '<tool_call>\n{"name": "start_task", "arguments": {"goal": "Compare three budget laptops"}}\n</tool_call>'
    s, m = make_session(FakeLLM(reply=start), FakeTTS(n=2), smart=FakeLLM(step_s=0.05, chars=2), tasks=TASKS)
    run_turn(s, "research budget laptops", stop_at=lambda k, n: n == 1)
    result = s.history.turns[-1][2]
    assert wait(lambda: json.loads(result["content"]) == {"task_id": 1, "status": "started"}), result["content"]
    assert len(s.tasks.tasks) == 1
    s.tasks.close()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
