"""Text-level helpers for models that write tool calls (and possibly thinking) into their text: splitting the
speakable part off at <tool_call>, parsing the two formats the Qwen chat templates emit, and a <think> guard.

  * Hermes JSON (Qwen3-4B-Instruct-2507): <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  * Qwen XML (Qwen3.5): <tool_call><function=NAME><parameter=P>value</parameter>...</function></tool_call>
  * Gemma 4: <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|> (special tokens: decode them)
  * Nemotron (Nano 9B v2): <TOOLCALL>[{"name": ..., "arguments": {...}}, ...]</TOOLCALL>

Each Qwen plugin parses its own format first; a block in the other Qwen format still parses (logged once), so a
model that switches format mid-deployment doesn't silently lose its tool calls. Gemma's calls open with their own
tag (TAGS), passed to the splitting / naming helpers.
"""
from __future__ import annotations
import json, logging, re, uuid

log = logging.getLogger("llm")
TAG = "<tool_call>"
GEMMA_TAG, GEMMA_Q = "<|tool_call>", '<|"|>'
NEMO_TAG = "<TOOLCALL>"
TAGS = {"hermes_json": TAG, "qwen_xml": TAG, "gemma4": GEMMA_TAG,     # the tag a format's calls open with
        "nemotron_json": NEMO_TAG}
NCALL = re.compile(r"<TOOLCALL>(.*?)(?:</TOOLCALL>|$)", re.S)
GCALL = re.compile(r"<\|tool_call>call:([\w.-]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>|$)", re.S)
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


def split_speakable(raw: str, tag: str = TAG):
    """(speakable_prefix, saw_tool_call) for the raw model text so far. A trailing partial "<tool_c" is held
    back until it resolves; text from the first call `tag` on is never speakable."""
    i = raw.find(tag)
    if i >= 0:
        return raw[:i], True
    k = _hold_back(raw, (tag,))
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


def _nemotron(body: str):
    """A <TOOLCALL> body: a JSON list of {"name", "arguments"} (arguments may be a JSON string). A list cut off
    before its closing bracket still parses if closing it is enough."""
    body = body.strip()
    for b in (body, body + "]", body + "}]"):
        try:
            calls = json.loads(b)
            break
        except Exception:
            calls = None
    if isinstance(calls, dict):
        calls = [calls]
    out = []
    for c in calls if isinstance(calls, list) else []:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        a = c.get("arguments") or {}
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        out.append({"name": c["name"], "arguments": a if isinstance(a, dict) else {}})
    if not out:
        log.warning("unparseable tool call dropped")
    return out


def parse_calls(raw: str, fmt: str):
    """[{"name", "arguments"}] for each call block in raw. The Qwen formats are read as `fmt` first, then as the
    other; a block neither reads is dropped (logged by kind only, never its text)."""
    if fmt == "gemma4":
        return [{"name": m.group(1), "arguments": _gemma_args(m.group(2))} for m in GCALL.finditer(raw)]
    if fmt == "nemotron_json":
        return [c for m in NCALL.finditer(raw) for c in _nemotron(m.group(1))]
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


# the first call's name, complete: Hermes {"name": "x" (closing quote seen), XML <function=x>, Gemma call:x{, or
# Nemotron [{"name": "x"
_NAME = (re.compile(r'\s*\{\s*"name"\s*:\s*"([^"\\]+)"'), re.compile(r"\s*<function=([^>\s]+)>"),
         re.compile(r"\s*call:([\w.-]+)\{"), re.compile(r'\s*\[\s*\{\s*"name"\s*:\s*"([^"\\]+)"'))
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
