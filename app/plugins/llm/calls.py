"""Text-level helpers for models that write tool calls into their text: splitting the speakable part off at a call's
opening tag, and parsing the two call formats in use.

  * Gemma 4 (gemma-4-e4b): <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|> (special tokens: decode them).
    Gemma sometimes drops the markers; gemma_normalize puts them back for the offered tools.
  * Hermes JSON (the fake LLM): <tool_call>{"name": ..., "arguments": {...}}</tool_call>

Each format's calls open with their own tag (TAGS), passed to the splitting / naming helpers. check_arguments /
flag_invalid: the one check of a call's arguments against the offered tool's schema (the hosted plugins run it on
every call).
"""
from __future__ import annotations
import json, logging, re, uuid

log = logging.getLogger("llm")
TAG = "<tool_call>"
GEMMA_TAG, GEMMA_Q = "<|tool_call>", '<|"|>'
TAGS = {"hermes_json": TAG, "gemma4": GEMMA_TAG}      # the tag a format's calls open with
GCALL = re.compile(r"<\|tool_call>call:([\w.-]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>|$)", re.S)
TCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def _hold_back(text: str, tags) -> int:
    """Length of a trailing partial prefix of any of `tags` (to hold back until it resolves)."""
    for k in range(min(max(len(t) for t in tags) - 1, len(text)), 0, -1):
        if any(t.startswith(text[-k:]) for t in tags):
            return k
    return 0


def split_speakable(raw: str, tag: str = TAG):
    """(speakable_prefix, saw_tool_call) for the raw model text so far. A trailing partial "<tool_c" is held
    back until it resolves; text from the first call `tag` on is never speakable."""
    i = raw.find(tag)
    if i >= 0:
        return raw[:i], True
    k = _hold_back(raw, (tag,))
    return (raw[:-k] if k else raw), False


def _hermes(body):
    try:
        c = json.loads(body)
    except Exception:
        return None
    return c if isinstance(c, dict) else None


def _gemma_value(v: str):
    v = v.strip()
    if v.startswith(GEMMA_Q) and v.endswith(GEMMA_Q) and len(v) >= 2 * len(GEMMA_Q):
        return v[len(GEMMA_Q):-len(GEMMA_Q)]
    try:
        return json.loads(v)                       # numbers, true / false, null
    except Exception:
        return v


def _gemma_args(body: str) -> dict:
    """key:value pairs, comma-separated; string values are wrapped in <|"|> and may contain commas or colons.
    Flat values only (every tool here takes strings); a nested {...} value stays a string."""
    out = {}; pos = 0
    while pos < len(body):
        colon = body.find(":", pos)
        if colon < 0:
            break
        key = body[pos:colon].strip().strip(",").strip()
        rest = body[colon + 1:]
        if rest.startswith(GEMMA_Q):
            end = rest.find(GEMMA_Q, len(GEMMA_Q))
            end = len(rest) if end < 0 else end + len(GEMMA_Q)
        else:
            end = rest.find(",")
            end = len(rest) if end < 0 else end
        if key:
            out[key] = _gemma_value(rest[:end])
        pos = colon + 1 + end + 1
    return out


def gemma_normalize(raw: str, names, final: bool = False) -> str:
    """Gemma 4 sometimes writes a call without its markers: a bare `NAME{...}` or `call:NAME{...}` as plain text
    (seen for set_reminder / send_message). For the offered tool `names`, rewrite those into the tagged form so
    they split, name and parse like any call, and are never spoken. Mid-stream (not `final`), a trailing word that
    could still become `NAME{` or `call:` is held back until it resolves."""
    if not names:
        return raw
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    out = re.sub(r"(?<![\w|>:])(?:call:)?(" + alt + r")\{", lambda m: GEMMA_TAG + "call:" + m.group(1) + "{", raw)
    out = out.replace(GEMMA_TAG + "call:" + GEMMA_TAG + "call:", GEMMA_TAG + "call:")      # already tagged
    out = re.sub(r"(" + re.escape(GEMMA_TAG) + r"call:)(?:" + re.escape(GEMMA_TAG) + r"call:)+", r"\1", out)
    if final or GEMMA_TAG in out:
        return out
    m = re.search(r"(?:^|(?<=[\s.,!?;]))[\w:]+$", out)              # the trailing word, if it starts a call
    if m and any((t + "{").startswith(m.group(0)) or ("call:" + t + "{").startswith(m.group(0)) for t in names):
        return out[:m.start()]
    return out


def parse_calls(raw: str, fmt: str):
    """[{"name", "arguments"}] for each call block in raw, read in `fmt` (a TAGS key); a Hermes block that doesn't
    parse (or never closes: cut off) is dropped (logged by kind only, never its text)."""
    if fmt == "gemma4":
        return [{"name": m.group(1), "arguments": _gemma_args(m.group(2))} for m in GCALL.finditer(raw)]
    if fmt != "hermes_json":
        raise ValueError(f"unknown tool-call format {fmt!r}")
    out = []
    for m in TCALL.finditer(raw):
        c = _hermes(m.group(1).strip())
        if c is None or not c.get("name"):
            log.warning("unparseable tool call dropped")
            continue
        out.append({"name": c["name"], "arguments": c.get("arguments") or {}})
    return out


# the first call's name, complete: Hermes {"name": "x" (closing quote seen) or Gemma call:x{ (after its tag)
_NAME = (re.compile(r'\s*\{\s*"name"\s*:\s*"([^"\\]+)"'), re.compile(r"\s*call:([\w.-]+)\{"))
NAME_WAIT = 64          # characters after the tag; a call that hasn't named its tool by then starts unnamed


def tool_start(raw: str, tag: str = TAG):
    """The ("tool_start") payload for raw text that has a call `tag`: {"name": ...} once the first call's name
    has decoded (every format names the tool first), {"name": None} if NAME_WAIT characters past the tag still
    don't name one, else None (wait for more text)."""
    i = raw.find(tag)
    if i < 0:
        return None
    body = raw[i + len(tag):]
    for rx in _NAME:
        m = rx.match(body)
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
