"""What every tool-call format shares. A model that writes its tool calls into its text declares a CallFormat: the
tag a call opens with, the pattern of the tool's name right after it, parse(), and an optional repair, normalize().
Each format lives with its model: Gemma 4's in llm/local/gemma4/calls.py, Hermes JSON (the fake LLM's) in fake.py.

The helpers here take the format object, so nothing in this module names a model:
  * split_speakable: the speakable text before a call's tag (a trailing partial tag is held back until it resolves);
  * tool_start: the tool named early, as soon as its name has decoded;
  * new_call_id: tool-call ids;
  * check_arguments / flag_invalid: the one check of a call's arguments against the offered tool's schema (the
    hosted plugins run it on every call; their APIs return structured calls, so they need no format).
"""
from __future__ import annotations
import re, uuid
from abc import ABC, abstractmethod


class CallFormat(ABC):
    """How a model writes its tool calls into its text."""
    tag: str                     # a call opens with it
    name: re.Pattern             # the tool's name right after the tag: group 1

    @abstractmethod
    def parse(self, raw: str) -> list[dict]:
        """[{"name", "arguments"}] for each call in `raw`."""

    def normalize(self, raw: str, names, final: bool = False) -> str:
        """`raw` with the model's malformed calls rewritten into the tagged form (`names`: the offered tools).
        Mid-stream it may hold back a trailing word that could still start a call; final=True (the stream ended)
        holds nothing back. No repair by default."""
        return raw


def _hold_back(text: str, tag: str) -> int:
    """Length of a trailing partial prefix of `tag` (to hold back until it resolves)."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-k:]):
            return k
    return 0


def split_speakable(raw: str, fmt: CallFormat):
    """(speakable_prefix, saw_tool_call) for the raw model text so far. A trailing partial tag ("<tool_c") is held
    back until it resolves; text from the first call tag on is never speakable."""
    i = raw.find(fmt.tag)
    if i >= 0:
        return raw[:i], True
    k = _hold_back(raw, fmt.tag)
    return (raw[:-k] if k else raw), False


NAME_WAIT = 64          # characters after the tag; a call that hasn't named its tool by then starts unnamed


def tool_start(raw: str, fmt: CallFormat):
    """The ("tool_start") payload for raw text that has a call tag: {"name": ...} once the first call's name has
    decoded (every format names the tool first), {"name": None} if NAME_WAIT characters past the tag still don't
    name one, else None (wait for more text)."""
    i = raw.find(fmt.tag)
    if i < 0:
        return None
    body = raw[i + len(fmt.tag):]
    m = fmt.name.match(body)
    if m:
        return {"name": m.group(1)}
    return {"name": None} if len(body) >= NAME_WAIT else None


def names_tool(tools) -> bool:
    """Whether tool_start should wait for the tool's name: only when MOUTH picks the lead-in by it, i.e. when
    ask_expert is offered. Otherwise tool_start comes at the bare tag, a few tokens sooner."""
    return any(t["function"]["name"] == "ask_expert" for t in tools or [])


def new_call_id() -> str:
    """A tool-call ID unique for the session (the neutral format; Claude needs unique, matching IDs)."""
    return f"call_{uuid.uuid4().hex[:8]}"


_JSON_TYPES = {"string": str, "object": dict, "array": list, "boolean": bool, "integer": int, "number": (int, float)}


def check_arguments(name, args, tools):
    """None if `args` fits tool `name`'s schema in `tools` (an object, its required keys present, declared types and
    enums respected), else the reason."""
    if not isinstance(args, dict):
        return "arguments are not an object"
    for t in tools or []:
        f = t["function"]
        if f["name"] != name:
            continue
        params = f["parameters"]
        missing = [k for k in params.get("required", []) if k not in args]
        if missing:
            return f"missing {', '.join(missing)}"
        for k, v in args.items():
            spec = params.get("properties", {}).get(k, {})
            typ = _JSON_TYPES.get(spec.get("type"))
            if typ and (not isinstance(v, typ) or (spec.get("type") in ("integer", "number") and isinstance(v, bool))):
                return f"{k} is not a {spec['type']}"
            if "enum" in spec and v not in spec["enum"]:
                return f"{k} is not one of {spec['enum']}"
        return None
    return "unknown tool"


def flag_invalid(call: dict, tools) -> dict:
    """The neutral call, with "error" set if its arguments don't fit (MOUTH returns that as the tool's result and
    never runs it); arguments that aren't an object become {}."""
    bad = check_arguments(call["name"], call["arguments"], tools)
    if bad:
        call = {**call, "error": f"invalid arguments for '{call['name']}': {bad}",
                "arguments": call["arguments"] if isinstance(call["arguments"], dict) else {}}
    return call
