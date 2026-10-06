"""The tool registry: one Tool per name, registered where it is implemented (lookups.py, stubs.py, agent.py), in the
order the models see them.

A tool is either
  * run for real (`fn(**arguments)` -> a small JSON-able dict; lookups and actions): execute_tool runs it, and a
    failure is an {"error": ...} result, so the model can tell the user instead of the turn crashing; or
  * answered by MOUTH itself (`handle(arguments, ctx)`; ask_expert, the task tools): the handler reads what it needs
    from the ToolContext and returns the result, or None when this context doesn't answer it (escalation or tasks
    off), and then it is an unknown tool, as for a name nobody registered.
Schemas hold names, types and required keys; the descriptions are prompts and live in prompts.yaml (`tools:`), so
build_tools joins the two. Adding a tool: write its function in a module here and register it with @tool.
"""
from __future__ import annotations
import os, threading, time
from dataclasses import dataclass
from typing import Any, Callable

GROUPS = ("base", "expert", "tasks", "finish")   # which tool lists offer a tool (build_tools)


@dataclass(frozen=True)
class Tool:
    name: str
    parameters: dict                       # JSON schema: types, required (descriptions: prompts.yaml)
    group: str = "base"
    fn: Callable | None = None             # run for real: fn(**arguments) -> result
    handle: Callable | None = None         # answered by MOUTH: handle(arguments, ctx) -> result | None


@dataclass(frozen=True)
class ToolContext:
    """What MOUTH's own tools may read: the calling turn's prompts (for the notes they answer with), the user's
    words and language, the conversation's task runner (None: tasks off) and whether the smart model answers
    ask_expert."""
    prompts: Any = None
    transcript: str = ""
    lang: Any = None
    tasks: Any = None
    expert: bool = False


TOOLS: dict[str, Tool] = {}


def register(t: Tool) -> Tool:
    if t.name in TOOLS:
        raise ValueError(f"tool {t.name} registered twice")
    if t.group not in GROUPS:
        raise ValueError(f"tool {t.name}: group {t.group!r} is not one of {GROUPS}")
    TOOLS[t.name] = t
    return t


def tool(name, parameters, group="base"):
    """Register the decorated function as a tool run for real: fn(**arguments) -> result."""
    def deco(fn):
        register(Tool(name, parameters, group, fn=fn))
        return fn
    return deco


def agent_tool(name, parameters, group):
    """Register the decorated function as a tool MOUTH answers: handle(arguments, ctx) -> result | None."""
    def deco(fn):
        register(Tool(name, parameters, group, handle=fn))
        return fn
    return deco


def names(*groups) -> list[str]:
    """The registered tool names in these groups (all groups if none), in order."""
    return [n for n, t in TOOLS.items() if not groups or t.group in groups]


def build_tools(prompts, expert=False, tasks=False, finish=False):
    """The OpenAI function schemas with their descriptions from `prompts` (app.prompts.Prompts): the lookup /
    action tools, then ask_expert if `expert`, start_task / cancel_task if `tasks`, finish_task if `finish` (the
    smart model running a task). Key order is the original's (name, description, parameters; a property's type,
    then description), since chat templates render the JSON as given."""
    groups = {"base"} | {g for g, on in (("expert", expert), ("tasks", tasks), ("finish", finish)) if on}
    text = prompts.tool_text(); out = []
    for t in TOOLS.values():
        if t.group not in groups:
            continue
        d = text[t.name]; pdesc = d.get("params") or {}
        props = {}
        for p, spec in t.parameters["properties"].items():
            props[p] = {"type": spec["type"], "description": pdesc[p],
                        **{k: v for k, v in spec.items() if k != "type"}} if p in pdesc else dict(spec)
        out.append({"type": "function", "function": {"name": t.name, "description": d["description"],
                    "parameters": {**t.parameters, "properties": props}}})
    return out


_DELAY_S = float(os.environ.get("TYGER_TOOL_DELAY_S") or 0)   # testing only: a slow tool (L-04 / I-03 checks)


def execute_tool(name, args):
    """Run a tool for real (tests stub this function: tests/fakes.py stub_tools)."""
    if _DELAY_S:
        time.sleep(_DELAY_S)
    t = TOOLS.get(name)
    if t is None or t.fn is None:
        return {"error": f"unknown tool '{name}'"}
    if not isinstance(args, dict):
        return {"error": f"bad arguments for '{name}'"}
    try:
        return t.fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for '{name}': {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {type(e).__name__}"}


def run_call(call, ctx: ToolContext):
    """One call's result. A call the plugin flagged (bad arguments) is never run; a tool MOUTH answers gets its
    handler; anything else runs for real."""
    if call.get("error"):
        return {"error": call["error"]}
    t = TOOLS.get(call["name"])
    if t is not None and t.handle is not None:
        out = t.handle(call.get("arguments") or {}, ctx)
        if out is not None:
            return out
    return execute_tool(call["name"], call.get("arguments", {}))


def run_round(calls, ctx: ToolContext):
    """One round of calls, run at once (each on its own thread); results in call order."""
    results = [None] * len(calls)

    def run(i, c):
        results[i] = run_call(c, ctx)
    workers = [threading.Thread(target=run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return results
