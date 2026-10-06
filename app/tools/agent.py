"""The tools MOUTH answers itself, with what the ToolContext gives it.

  * ask_expert (escalation, PLUGINS.md step 10): offered to the `reply` model only when a `smart` model is
    configured. The smart model answers the turn; the reply model's call gets a short note as its result.
  * start_task / cancel_task (background tasks, PLUGINS.md steps 12-14): offered to the `reply` model with
    ask_expert when mouth.tasks.enabled; they go to the conversation's task runner (app/tasks.py).
  * finish_task: offered only to the smart model running a task. The task loop (app/tasks.py) reads it from the
    calls: it ends the task, and it is never run as a tool.
Where a context doesn't answer one (escalation or tasks off), it is an unknown tool.
"""
from app.tools.registry import Tool, agent_tool, register

EXPERT, START, CANCEL, FINISH = "ask_expert", "start_task", "cancel_task", "finish_task"


@agent_tool(EXPERT, {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            group="expert")
def ask_expert(args, ctx):
    if ctx.expert:
        return {"result": ctx.prompts.text("escalation", "tool_result")}


@agent_tool(START, {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]},
            group="tasks")
def start_task(args, ctx):
    if ctx.tasks is not None:
        return ctx.tasks.start(args.get("goal"), ctx.transcript, ctx.lang)


@agent_tool(CANCEL, {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]},
            group="tasks")
def cancel_task(args, ctx):
    if ctx.tasks is not None:
        try:
            return ctx.tasks.cancel(int(args.get("task_id")))
        except (TypeError, ValueError):
            return {"error": ctx.prompts.text("tasks", "unknown_task")}


register(Tool(FINISH, {"type": "object", "properties": {"summary": {"type": "string"}, "result": {"type": "string"}},
                      "required": ["summary", "result"]}, group="finish"))   # no handler: the task loop reads it
