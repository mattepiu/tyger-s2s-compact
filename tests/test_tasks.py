"""Background tasks (PLUGINS.md step 12), GPU-free: the reply model starts / cancels tasks, each runs as an agentic
loop on a fake `smart` model, the status note reaches the next reply request, caps / timeout / cancel / session
close, and nothing about a task's content is logged. Run: .venv/bin/python tests/test_tasks.py
"""
import os, sys, json, logging, socket, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import FakeLLM, FakeTTS, FakeMouth, PROMPTS, make_session, run_turn, stub_tools  # noqa: E402
from app import mouth as M  # noqa: E402
from app.tasks import TaskSlots  # noqa: E402
from app.wire import send_msg, recv_msg  # noqa: E402

GOAL = "Compare three budget laptops for programming"
START = '<tool_call>\n{"name": "start_task", "arguments": {"goal": "%s"}}\n</tool_call>' % GOAL
CANCEL = '<tool_call>\n{"name": "cancel_task", "arguments": {"task_id": 1}}\n</tool_call>'
SEARCH = '<tool_call>\n{"name": "web_search", "arguments": {"query": "budget laptops"}}\n</tool_call>'
TASKS = {"enabled": True, "max_concurrent": 1, "max_per_session": 5, "max_wall_s": 30, "max_steps": 4}


def session(reply=START, smart=None, tasks=None, **kw):
    llm = FakeLLM(reply=reply, phase_b="Sure, I'll look into it and get back to you.")
    s, m = make_session(llm, FakeTTS(n=1), smart=smart or FakeLLM(), tasks={**TASKS, **(tasks or {})}, **kw)
    done = []
    s.tasks.on_done = done.append
    return s, m, llm, done


def wait(pred, timeout=5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def last_user(llm, kind="reply"):
    return [c[1] for c in llm.calls if c[0] == kind][-1][-1]["content"]


def test_tools_offered_only_with_tasks():
    s, m = make_session(FakeLLM(), smart=FakeLLM())
    assert s.tasks is None and "start_task" not in [t["function"]["name"] for t in m.tools]
    s, m, _, _ = session()
    names = [t["function"]["name"] for t in m.tools]
    assert names[-3:] == ["ask_expert", "start_task", "cancel_task"]
    assert [t["function"]["name"] for t in m.task_tools][-1] == "finish_task"


def test_start_chat_done_status():
    s, m, llm, done = session()
    run_turn(s, "can you research budget laptops for programming and get back to me?")
    assert [h["role"] for h in s.history] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(s.history[2]["content"]) == {"task_id": 1, "status": "started"}
    assert wait(lambda: done) and done[0].status == "done" and done[0].summary.startswith("About your task")
    llm.text["reply"] = "Not yet, I'm still on it."
    run_turn(s, "so how's it going?")
    user = last_user(llm)
    assert user.startswith("[Background tasks") and "not told to the user yet" in user and done[0].summary in user
    assert user.endswith("\n\nso how's it going?")
    assert s.history[-2]["content"] == "so how's it going?"           # history keeps the words only


def test_status_while_running_and_request():
    smart = FakeLLM(step_s=0.05, chars=2)                              # the finish call takes ~2 s to stream
    s, m, llm, done = session(smart=smart)
    run_turn(s, "Подбери мне ноутбук для программирования")
    llm.text["reply"] = "Still working on it."
    run_turn(s, "ну что там?")
    assert "still working on it" in last_user(llm) and not done
    req = smart.calls[0][1]
    assert smart.calls[0][0] == "task" and req[0]["content"] == PROMPTS.text("tasks", "system")
    assert GOAL in req[-1]["content"] and "Russian" in req[-1]["content"]        # the language to write in
    assert '"Подбери мне ноутбук для программирования"' in req[-1]["content"]       # the user's own words
    s.tasks.close()


def test_task_tool_rounds():
    smart = FakeLLM(task=SEARCH)
    s, m, llm, done = session(smart=smart)
    ran = []
    with stub_tools(lambda n, a: ran.append(n) or {"answer": "three good laptops"}):
        run_turn(s, "research budget laptops")
        assert wait(lambda: done)
    assert ran == ["web_search"] and [c[0] for c in smart.calls] == ["task", "task_b"]
    assert done[0].status == "done" and done[0].steps == 1 and done[0].result
    second = smart.calls[1][1]
    assert second[-2]["tool_calls"][0]["name"] == "web_search" and "three good laptops" in second[-1]["content"]


def test_caps_concurrent_and_per_session():
    smart = FakeLLM(step_s=0.05, chars=2)
    s, m, llm, done = session(smart=smart)
    run_turn(s, "research laptops")
    run_turn(s, "and research phones too")
    assert json.loads(s.history[-2]["content"]) == {"error": PROMPTS.text("tasks", "cap_concurrent")}
    s.tasks.close()
    s, m, llm, done = session(tasks={"max_per_session": 2, "max_concurrent": 2})
    for i in range(3):
        run_turn(s, f"research thing {i}")
        wait(lambda: len(done) == min(i + 1, 2))
    assert json.loads(s.history[-2]["content"]) == {"error": PROMPTS.text("tasks", "cap_session")}


def test_global_cap_across_sessions():
    slots = TaskSlots(1)
    slow = FakeLLM(step_s=0.05, chars=2)
    a, _, _, done_a = session(smart=slow, slots=slots)
    b, _, _, done_b = session(smart=FakeLLM(), slots=slots)
    run_turn(a, "research laptops")
    run_turn(b, "research phones")
    assert json.loads(b.history[-2]["content"]) == {"error": PROMPTS.text("tasks", "cap_global")}
    assert wait(lambda: done_a) and wait(lambda: slots.used == 0)          # the slot comes back when it ends
    run_turn(b, "research phones again")
    assert wait(lambda: done_b) and done_b[0].status == "done"


def test_timeout_fails_and_frees_slot():
    smart = FakeLLM(step_s=0.05, chars=2)
    s, m, llm, done = session(smart=smart, tasks={"max_wall_s": 0.3})
    run_turn(s, "research laptops")
    assert wait(lambda: done) and done[0].status == "failed" and done[0].why == "timeout"
    assert wait(lambda: smart.live == 0) and smart.aborted >= 1 and m.task_slots.used == 0
    llm.text["reply"] = "It didn't work out."
    run_turn(s, "any news?")
    assert "could not be finished" in last_user(llm)


def test_cancel_by_the_user():
    smart = FakeLLM(step_s=0.05, chars=2)
    s, m, llm, done = session(smart=smart)
    run_turn(s, "research laptops")
    llm.text["reply"] = CANCEL
    run_turn(s, "never mind, drop it")
    assert json.loads(s.history[-2]["content"]) == {"task_id": 1, "status": "cancelled"}
    assert wait(lambda: smart.live == 0) and smart.aborted >= 1 and not done
    assert s.tasks.get(1).status == "cancelled" and m.task_slots.used == 0
    run_turn(s, "drop task one again")
    assert json.loads(s.history[-2]["content"]) == {"error": PROMPTS.text("tasks", "unknown_task")}


def test_session_close_cancels_tasks():
    smart = FakeLLM(step_s=0.05, chars=2)
    m = FakeMouth(FakeLLM(reply=START), FakeTTS(n=1), smart=smart, tasks=TASKS)
    a, b = socket.socketpair()
    s = M.Session(m, a)
    t = threading.Thread(target=s.run, daemon=True); t.start()
    send_msg(b, b"T", {"transcript": "research laptops"})
    assert wait(lambda: s.tasks.tasks and smart.live == 1)
    b.close(); t.join(timeout=5)
    assert s.tasks.get(1).status == "cancelled"
    assert wait(lambda: smart.live == 0) and smart.aborted >= 1 and m.task_slots.used == 0


def test_max_steps_then_limit_note():
    smart = FakeLLM(task=SEARCH, task_b=SEARCH)                       # never finishes
    s, m, llm, done = session(smart=smart, tasks={"max_steps": 2})
    with stub_tools(lambda n, a: {"answer": "x"}):
        run_turn(s, "research laptops")
        assert wait(lambda: done)
    assert done[0].status == "failed" and done[0].why == "max_steps" and len(smart.calls) == 4
    limit = PROMPTS.text("tasks", "limit")
    assert limit not in json.dumps(smart.calls[1][1]) and limit in json.dumps(smart.calls[3][1], ensure_ascii=False)


def test_plain_text_answer_finishes():
    s, m, llm, done = session(smart=FakeLLM(task="Here is what I found: the second laptop is the best value."))
    run_turn(s, "research laptops")
    assert wait(lambda: done) and done[0].status == "done" and done[0].summary.startswith("Here is what I found")


def test_llm_error_fails_task():
    s, m, llm, done = session(smart=FakeLLM(fail={"task": "rate_limit"}))
    run_turn(s, "research laptops")
    assert wait(lambda: done) and done[0].status == "failed" and done[0].why == "rate_limit"


def test_logs_carry_no_task_content():
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())
    h = H(); logging.getLogger().addHandler(h); old = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        s, m, llm, done = session(smart=FakeLLM(task=SEARCH))
        with stub_tools(lambda n, a: {"answer": "x"}):
            run_turn(s, "research budget laptops please")
            assert wait(lambda: done)
    finally:
        logging.getLogger().removeHandler(h); logging.getLogger().setLevel(old)
    text = "\n".join(records)
    assert "task #1 started" in text and "task #1 done" in text
    for secret in (GOAL, "budget laptops please", done[0].summary, done[0].result):
        assert secret not in text, secret


# ---- step 13: F (floor request) -> G (grant) -> the announcement as a turn ------------------------------------
def live(reply=START, smart=None, tasks=None, tts=None):
    llm = FakeLLM(reply=reply, phase_b="Sure, I'll look into it.")
    m = FakeMouth(llm, tts or FakeTTS(n=2), smart=smart or FakeLLM(), tasks={**TASKS, **(tasks or {})})
    a, b = socket.socketpair()
    s = M.Session(m, a)
    threading.Thread(target=s.run, daemon=True).start()
    b.settimeout(5)
    return s, b, m, llm


def frames(peer, until):
    """Read (kind, payload) frames until until(kind, payload); JSON payloads decoded."""
    got = []
    while True:
        k, p = recv_msg(peer)
        if k is None:
            return got
        p = json.loads(p) if k in (b"X", b"D", b"F") else p
        got.append((k, p))
        if until(k, p):
            return got


def start_and_wait_floor(s, peer):
    send_msg(peer, b"T", {"transcript": "research budget laptops and get back to me"})
    seen = {}                                 # the acknowledgement's D and the floor request F, in either order

    def both(k, p):
        if k in (b"D", b"F"):
            seen[k] = p
        return b"D" in seen and b"F" in seen
    frames(peer, both)
    return seen[b"F"]


def test_announce_after_grant():
    s, peer, m, llm = live()
    assert start_and_wait_floor(s, peer) == {"task": 1}
    send_msg(peer, b"G", {"task": 1})
    got = frames(peer, lambda k, p: k == b"D")
    d = got[-1][1]
    assert d["proactive"] == 1 and not d["barged"] and d["text"].startswith("About your task")
    assert any(k == b"A" for k, _ in got) and any(k == b"X" for k, _ in got)
    note, said = s.history[-2], s.history[-1]
    assert note["role"] == "user" and note["content"].startswith("[Background task #1") and "full result" in note["content"]
    assert said == {"role": "assistant", "content": d["text"]} and s.tasks.get(1).announced == "yes"
    assert "told to the user" in s.tasks.status_block()
    peer.close()


def test_barge_during_announcement_marks_partial():
    s, peer, m, llm = live(tts=FakeTTS(n=20, step_s=0.02))
    start_and_wait_floor(s, peer)
    send_msg(peer, b"G", {"task": 1})
    frames(peer, lambda k, p: k == b"A")
    send_msg(peer, b"B", {"played_s": 0.1})
    d = frames(peer, lambda k, p: k == b"D")[-1][1]
    assert d["barged"] and d["proactive"] == 1
    assert s.tasks.get(1).announced == "partial" and "heard only part" in s.tasks.status_block()
    assert s.history[-1]["content"].endswith(PROMPTS.text("markers", "interrupted"))
    peer.close()


def test_failed_task_announced_in_one_sentence():
    s, peer, m, llm = live(smart=FakeLLM(step_s=0.05, chars=2), tasks={"max_wall_s": 0.3})
    start_and_wait_floor(s, peer)
    send_msg(peer, b"G", {"task": 1})
    d = frames(peer, lambda k, p: k == b"D")[-1][1]
    assert d["text"] == llm.task_failed and d["proactive"] == 1
    req = [c[1] for c in llm.calls if c[0] == "task_failed"][-1]
    assert GOAL in req[-1]["content"] and req[-1]["content"].endswith("Say it in English.")
    assert s.history[-2]["content"] == PROMPTS.fill("tasks", "failed_note", id=1, goal=GOAL)
    peer.close()


def test_grant_for_a_task_that_no_longer_needs_it():
    s, peer, m, llm = live(smart=FakeLLM(step_s=0.05, chars=2))
    send_msg(peer, b"T", {"transcript": "research budget laptops"})
    frames(peer, lambda k, p: k == b"D")
    s.tasks.cancel(1)
    send_msg(peer, b"G", {"task": 1})                                 # e.g. granted just as it was cancelled
    d = frames(peer, lambda k, p: k == b"D")[-1][1]
    assert d == {"text": "", "barged": False, "proactive": 1}
    send_msg(peer, b"G", {"task": 9})
    assert frames(peer, lambda k, p: k == b"D")[-1][1]["text"] == ""
    peer.close()


def test_escalation_gets_the_full_result():
    smart = FakeLLM()
    s, peer, m, llm = live(smart=smart)
    start_and_wait_floor(s, peer)
    send_msg(peer, b"G", {"task": 1})
    frames(peer, lambda k, p: k == b"D")
    llm.text["reply"] = ('<tool_call>\n{"name": "ask_expert", "arguments": {"question": "Compare the top two laptops '
                         'in detail."}}\n</tool_call>')
    send_msg(peer, b"T", {"transcript": "compare the top two in detail"})
    frames(peer, lambda k, p: k == b"D")
    req = [c[1] for c in smart.calls if c[0] == "smart"][-1]
    assert "The full result of the task, in plain text." in req[-1]["content"]
    assert any("[Background task #1" in x["content"] for x in req if x["role"] == "user")   # the note, as context
    peer.close()


def tasks_imports(source):
    """Every module app/tasks.py imports, at the top or inside a function."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    return out


def test_tasks_never_imports_mouth():
    """tasks.py gets the message builders from app.messages and tools from app.tools: no import cycle with MOUTH
    (it used to import app.mouth inside its functions)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "app", "tasks.py")) as f:
        assert "app.mouth" not in tasks_imports(f.read())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
