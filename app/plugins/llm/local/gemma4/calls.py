"""Gemma 4's tool-call format: <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>, written in special tokens
(the plugin decodes them). Gemma sometimes drops the markers and writes a bare `NAME{...}`; normalize() puts them back
for the offered tools, so such a call splits, names and parses like any other and is never spoken.
"""
from __future__ import annotations
import json, re

from app.plugins.llm.calls import CallFormat

TAG, Q = "<|tool_call>", '<|"|>'
CALL = re.compile(r"<\|tool_call>call:([\w.-]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>|$)", re.S)


def _value(v: str):
    v = v.strip()
    if v.startswith(Q) and v.endswith(Q) and len(v) >= 2 * len(Q):
        return v[len(Q):-len(Q)]
    try:
        return json.loads(v)                       # numbers, true / false, null
    except Exception:
        return v


def _arguments(body: str) -> dict:
    """key:value pairs, comma-separated; string values are wrapped in <|"|> and may contain commas or colons.
    Flat values only (every tool here takes strings); a nested {...} value stays a string."""
    out = {}; pos = 0
    while pos < len(body):
        colon = body.find(":", pos)
        if colon < 0:
            break
        key = body[pos:colon].strip().strip(",").strip()
        rest = body[colon + 1:]
        if rest.startswith(Q):
            end = rest.find(Q, len(Q))
            end = len(rest) if end < 0 else end + len(Q)
        else:
            end = rest.find(",")
            end = len(rest) if end < 0 else end
        if key:
            out[key] = _value(rest[:end])
        pos = colon + 1 + end + 1
    return out


class GemmaCalls(CallFormat):
    tag = TAG
    name = re.compile(r"\s*call:([\w.-]+)\{")

    def parse(self, raw):
        return [{"name": m.group(1), "arguments": _arguments(m.group(2))} for m in CALL.finditer(raw)]

    def normalize(self, raw, names, final=False):
        """A bare `NAME{...}` or `call:NAME{...}` as plain text (seen for set_reminder / send_message) becomes the
        tagged form, for the offered tool `names`. Mid-stream (not `final`), a trailing word that could still become
        `NAME{` or `call:` is held back until it resolves."""
        if not names:
            return raw
        alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        out = re.sub(r"(?<![\w|>:])(?:call:)?(" + alt + r")\{", lambda m: TAG + "call:" + m.group(1) + "{", raw)
        out = out.replace(TAG + "call:" + TAG + "call:", TAG + "call:")      # already tagged
        out = re.sub(r"(" + re.escape(TAG) + r"call:)(?:" + re.escape(TAG) + r"call:)+", r"\1", out)
        if final or TAG in out:
            return out
        m = re.search(r"(?:^|(?<=[\s.,!?;]))[\w:]+$", out)              # the trailing word, if it starts a call
        if m and any((t + "{").startswith(m.group(0)) or ("call:" + t + "{").startswith(m.group(0)) for t in names):
            return out[:m.start()]
        return out
