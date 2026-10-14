"""MOUTH's tools: a registry of one Tool per name (registry.py). The modules below register theirs on import, in
the order the models see them: the lookups, the action stubs, then the tools MOUTH answers itself.

  build_tools(prompts, expert=, tasks=, finish=)   the OpenAI function schemas a model is offered
  run_round(calls, ctx) / run_call(call, ctx)      the results of a model's calls (ToolContext: what MOUTH's own
                                                   tools may read)
  execute_tool(name, args)                         run a lookup / action for real
  runs_for_real(name)                              a lookup / action, not one of MOUTH's own tools
"""
from app.tools.registry import (TOOLS, Tool, ToolContext, build_tools, execute_tool, names, run_call, run_round,
                                runs_for_real)
from app.tools import lookups, stubs, agent  # noqa: E402  (registers their tools, in this order)
from app.tools.agent import EXPERT, START, CANCEL, FINISH  # noqa: E402

__all__ = ["TOOLS", "Tool", "ToolContext", "build_tools", "execute_tool", "names", "run_call", "run_round",
           "runs_for_real", "lookups", "stubs", "agent", "EXPERT", "START", "CANCEL", "FINISH"]
