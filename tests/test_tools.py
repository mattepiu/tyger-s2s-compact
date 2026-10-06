"""The tool registry (app/tools/, no GPU, no network): one Tool per name in the order the models see them, and which
context answers which call: a flagged call is never run, MOUTH's own tools answer only where they are on (otherwise
they are unknown tools, as before R-08), everything else runs for real. Run: .venv/bin/python tests/test_tools.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fakes import PROMPTS, stub_tools  # noqa: E402
from app import tools  # noqa: E402
from app.tools import registry  # noqa: E402
from app.tools.registry import Tool, ToolContext, run_call, run_round  # noqa: E402


class Tasks:
    """A conversation's task runner, recording what it was asked."""
    def __init__(self):
        self.asked = []

    def start(self, goal, transcript, lang):
        self.asked.append(("start", goal, transcript, lang)); return {"task_id": 1, "status": "started"}

    def cancel(self, task_id):
        self.asked.append(("cancel", task_id)); return {"task_id": task_id, "status": "cancelled"}


def call(name, **arguments):
    return {"id": "call_1", "name": name, "arguments": arguments}


def test_registry_order_and_groups():
    assert tools.names() == ["get_weather", "web_search", "get_current_time", "set_reminder", "send_message",
                             "ask_expert", "start_task", "cancel_task", "finish_task"]
    assert tools.names("base") == tools.names()[:5] and tools.names("tasks") == ["start_task", "cancel_task"]
    offered = [t["function"]["name"] for t in tools.build_tools(PROMPTS, expert=True)]
    assert offered == tools.names("base", "expert")


def test_flagged_call_never_runs():
    with stub_tools(lambda n, a: (_ for _ in ()).throw(AssertionError("ran"))):
        assert run_call({**call("get_weather", unit="kelvin"), "error": "bad unit"}, ToolContext()) == {"error": "bad unit"}


def test_lookups_run_for_real():
    ran = []
    with stub_tools(lambda n, a: ran.append((n, a)) or {"answer": "sunny"}):
        assert run_round([call("get_weather", location="Oslo"), call("web_search", query="q")], ToolContext()) == \
            [{"answer": "sunny"}] * 2
    assert sorted(ran) == [("get_weather", {"location": "Oslo"}), ("web_search", {"query": "q"})]


def test_expert_only_where_escalation_is_on():
    on = run_call(call("ask_expert", question="q"), ToolContext(prompts=PROMPTS, expert=True))
    assert on == {"result": PROMPTS.text("escalation", "tool_result")}
    assert run_call(call("ask_expert", question="q"), ToolContext(prompts=PROMPTS)) == \
        {"error": "unknown tool 'ask_expert'"}


def test_tasks_only_where_tasks_are_on():
    t = Tasks(); ctx = ToolContext(prompts=PROMPTS, transcript="find me a laptop", lang="en", tasks=t)
    assert run_call(call("start_task", goal="laptops"), ctx)["status"] == "started"
    assert run_call(call("cancel_task", task_id="1"), ctx)["status"] == "cancelled"
    assert run_call(call("cancel_task", task_id="one"), ctx) == {"error": PROMPTS.text("tasks", "unknown_task")}
    assert t.asked == [("start", "laptops", "find me a laptop", "en"), ("cancel", 1)]
    off = ToolContext(prompts=PROMPTS)
    assert run_call(call("start_task", goal="x"), off) == {"error": "unknown tool 'start_task'"}
    assert run_call(call("finish_task", summary="s", result="r"), ctx) == {"error": "unknown tool 'finish_task'"}
    assert run_call(call("fly"), ctx) == {"error": "unknown tool 'fly'"}


def test_execute_tool_errors():
    assert tools.execute_tool("get_current_time", ["UTC"]) == {"error": "bad arguments for 'get_current_time'"}
    assert tools.execute_tool("get_current_time", {"tz": "UTC"})["error"].startswith("bad arguments")
    assert tools.execute_tool("set_reminder", {"text": "t", "time": "5"})["demo"] is True


def test_a_new_tool_is_one_registration():
    """Adding a tool: register it (and give it a description in prompts.yaml); MOUTH's code doesn't change."""
    registry.tool("roll_die", {"type": "object", "properties": {"sides": {"type": "integer"}}, "required": []})(
        lambda sides=6: {"rolled": sides})
    try:
        assert run_call(call("roll_die", sides=20), ToolContext()) == {"rolled": 20}
        try:
            registry.register(Tool("roll_die", {})); raise AssertionError("registered a name twice")
        except ValueError:
            pass
    finally:
        registry.TOOLS.pop("roll_die")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
