"""Text-level helpers for models that write tool calls (and possibly thinking) into their text: splitting the
speakable part off at <tool_call>, parsing the two formats the Qwen chat templates emit, and a <think> guard.

  * Hermes JSON (Qwen3-4B-Instruct-2507): <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  * Qwen XML (Qwen3.5): <tool_call><function=NAME><parameter=P>value</parameter>...</function></tool_call>

Each plugin parses its own format first; a block in the other format still parses (logged once), so a model that
switches format mid-deployment doesn't silently lose its tool calls.
"""
from __future__ import annotations
import json, logging, re, uuid

log = logging.getLogger("llm")
TAG = "<tool_call>"
TCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TFUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
TPARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_warned = set()


def _hold_back(text: str, tags) -> int:
    """Length of a trailing partial prefix of any of `tags` (to hold back until it resolves)."""
    for k in range(min(max(len(t) for t in tags) - 1, len(text)), 0, -1):
        if any(t.startswith(text[-k:]) for t in tags):
            return k
    return 0


def split_speakable(raw: str):
    """(speakable_prefix, saw_tool_call) for the raw model text so far. A trailing partial "<tool_c" is held
    back until it resolves; text from the first <tool_call> on is never speakable."""
    i = raw.find(TAG)
    if i >= 0:
        return raw[:i], True
    k = _hold_back(raw, (TAG,))
    return (raw[:-k] if k else raw), False


def strip_think(raw: str, final: bool = False) -> str:
    """raw without thinking: complete <think>...</think> blocks and stray tags removed, everything after an
    unclosed <think> dropped, a trailing partial tag held back unless `final` (the text is complete). Stable as
    raw grows: what it returned for a prefix is a prefix of what it returns later, so it can sit in front of a
    streaming speaker."""
    out = []; pos = 0
    while True:
        i = raw.find(THINK_OPEN, pos); j = raw.find(THINK_CLOSE, pos)
        if j >= 0 and (i < 0 or j < i):              # a stray close tag (thinking opened in the prompt)
            out.append(raw[pos:j]); pos = j + len(THINK_CLOSE); continue
        if i < 0:
            break
        out.append(raw[pos:i])
        j = raw.find(THINK_CLOSE, i + len(THINK_OPEN))
        if j < 0:
            return "".join(out)                      # still thinking: nothing after the tag is speech
        pos = j + len(THINK_CLOSE)
    tail = raw[pos:]
    k = 0 if final else _hold_back(tail, (THINK_OPEN, THINK_CLOSE))
    return "".join(out) + (tail[:-k] if k else tail)


def _param(v):
    """An XML-format parameter value: JSON if it parses as a number/bool/list/object, else the raw string."""
    try:
        x = json.loads(v)
        return x if not isinstance(x, str) else v
    except Exception:
        return v


def _xml(body):
    f = TFUNC.search(body)
    if not f:
        return None
    return {"name": f.group(1), "arguments": {p: _param(v) for p, v in TPARAM.findall(f.group(2))}}


def _hermes(body):
    try:
        c = json.loads(body)
    except Exception:
        return None
    return c if isinstance(c, dict) else None


FORMATS = {"hermes_json": _hermes, "qwen_xml": _xml}


def parse_calls(raw: str, fmt: str):
    """[{"name", "arguments"}] for each <tool_call> block in raw, read as `fmt` first, then as the other format;
    a block neither reads is dropped (logged by kind only, never its text)."""
    first = FORMATS[fmt]; other = next(f for k, f in FORMATS.items() if k != fmt)
    out = []
    for m in TCALL.finditer(raw):
        body = m.group(1).strip()
        c = first(body)
        if c is None:
            c = other(body)
            if c is not None and fmt not in _warned:
                _warned.add(fmt); log.warning("tool call not in the expected %s format; parsed as the other", fmt)
        if c is None or not c.get("name"):
            log.warning("unparseable tool call dropped")
            continue
        out.append({"name": c["name"], "arguments": c.get("arguments") or {}})
    return out


# the first call's name, complete: Hermes {"name": "x" (closing quote seen) or XML <function=x>
_NAME = (re.compile(r'\s*\{\s*"name"\s*:\s*"([^"\\]+)"'), re.compile(r"\s*<function=([^>\s]+)>"))
NAME_WAIT = 64          # characters after the tag; a call that hasn't named its tool by then starts unnamed


def tool_start(raw: str):
    """The ("tool_start") payload for raw text that has a <tool_call>: {"name": ...} once the first call's name
    has decoded (both formats name the tool first), {"name": None} if NAME_WAIT characters past the tag still
    don't name one, else None (wait for more text)."""
    i = raw.find(TAG)
    if i < 0:
        return None
    body = raw[i + len(TAG):]
    for rx in _NAME:
        m = rx.match(body)
        if m:
            return {"name": m.group(1)}
    return {"name": None} if len(body) >= NAME_WAIT else None


def new_call_id() -> str:
    """A tool-call ID unique for the session (the neutral format; Claude needs unique, matching IDs)."""
    return f"call_{uuid.uuid4().hex[:8]}"
