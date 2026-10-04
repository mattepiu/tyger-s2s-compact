"""Background tasks (PLUGINS.md step 12), GPU-free: the reply model starts / cancels tasks, each runs as an agentic
loop on a fake `smart` model, the status note reaches the next reply request, caps / timeout / cancel / session
close, and nothing about a task's content is logged. Run: .venv/bin/python tests/test_tasks.py
"""
import os, sys, json, logging, socket, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import FakeLLM, FakeTTS, FakeMouth, PROMPTS, make_session, run_turn  # noqa: E402
from app import mouth as M  # noqa: E402
from app.tasks import TaskSlots  # noqa: E402
from app.wire import send_msg  # noqa: E402

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
    orig = M.execute_tool
    M.execute_tool = lambda n, a: ran.append(n) or {"answer": "three good laptops"}
    try:
        run_turn(s, "research budget laptops")
        assert wait(lambda: done)
    finally:
        M.execute_tool = orig
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
    orig = M.execute_tool
    M.execute_tool = lambda n, a: {"answer": "x"}
    try:
        run_turn(s, "research laptops")
        assert wait(lambda: done)
    finally:
        M.execute_tool = orig
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
        orig = M.execute_tool
        M.execute_tool = lambda n, a: {"answer": "x"}
        try:
            run_turn(s, "research budget laptops please")
            assert wait(lambda: done)
        finally:
            M.execute_tool = orig
    finally:
        logging.getLogger().removeHandler(h); logging.getLogger().setLevel(old)
    text = "\n".join(records)
    assert "task #1 started" in text and "task #1 done" in text
    for secret in (GOAL, "budget laptops please", done[0].summary, done[0].result):
        assert secret not in text, secret


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
