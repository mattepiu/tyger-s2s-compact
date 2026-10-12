"""fake: a scripted LLM, no model (tests, CI, and running the stack without a GPU: config/fake.yaml).

Routes each request by its system prompt (read from config/prompts.yaml, like MOUTH does): tool lead-in /
still-looking / backchannel / the expert lead-in and still-looking lines / the smart model's answer (`smart`, or
`smart_b` once a tool result is in) / a background task on the smart model (`task`, `task_b`; by default it calls
finish_task) / phase B (a tool result present) / the main reply, and streams the scripted text for that kind a
few characters at a time, polling stop_flag between items like the real engine loop. `reply` and `smart` may contain
tool calls in Hermes JSON, its own call format (HermesCalls, below), which go through the same splitting, naming
(tool_start) and parsing as a local plugin's calls. For tests of hosted behaviour, `fail` maps a request kind to an
LLMError kind raised for it (after `fail_after` items), and `refuse` lists kinds answered with a refusal. Every
request is recorded in `calls` (kind, messages); nothing of a request is logged.
"""
from __future__ import annotations
import json, logging, re, threading, time
from dataclasses import dataclass, field

from app.plugins.base import LLMError, LLMPlugin
from app.plugins.options import PluginOptions
from app.plugins.llm.calls import CallFormat, split_speakable, new_call_id, tool_start, names_tool

log = logging.getLogger("llm")


class HermesCalls(CallFormat):
    """Hermes JSON: <tool_call>{"name": ..., "arguments": {...}}</tool_call>."""
    tag = "<tool_call>"
    name = re.compile(r'\s*\{\s*"name"\s*:\s*"([^"\\]+)"')       # the closing quote seen
    CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

    def parse(self, raw):
        """A block that doesn't parse (or never closes: cut off) is dropped, logged by kind only, never its text."""
        out = []
        for m in self.CALL.finditer(raw):
            try:
                c = json.loads(m.group(1).strip())
            except Exception:
                c = None
            if not isinstance(c, dict) or not c.get("name"):
                log.warning("unparseable tool call dropped")
                continue
            out.append({"name": c["name"], "arguments": c.get("arguments") or {}})
        return out


HERMES = HermesCalls()
KINDS = ("reply", "phase_b", "leadin", "still", "ack", "expert_leadin", "expert_still", "smart", "smart_b", "task",
         "task_b", "task_failed")
FINISH_CALL = ('<tool_call>\n{"name": "finish_task", "arguments": {"summary": "About your task: it is done, and it '
               'went well.", "result": "The full result of the task, in plain text."}}\n</tool_call>')


def wait_or_stop(seconds, stop_flag, poll=0.002):
    """Sleep `seconds`, returning early (True) if stop_flag() turns true, like the engine poll loop."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if stop_flag and stop_flag():
            return True
        time.sleep(poll)
    return bool(stop_flag and stop_flag())


class FakeLLM(LLMPlugin):
    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        # the scripted text per request kind (KINDS)
        reply: str = "Sure, here is a reasonably long first sentence for you. And a second one."
        phase_b: str = "It is sunny and twenty degrees there today, a lovely day."
        leadin: str = "One sec, let me look that up."
        still: str = "Still looking, almost there."
        ack: str = "oh nice"
        expert_leadin: str = "Let me think that through."
        expert_still: str = "Still working it out."
        smart: str = "Here is the careful answer, worked out step by step. It holds up."
        smart_b: str = "Having checked, the careful answer is forty two."
        task: str = FINISH_CALL
        task_b: str = FINISH_CALL
        task_failed: str = "Sorry, I couldn't finish that one."
        step_s: float = 0.002                 # between items
        chars: int = 4                        # per item
        fail: dict = field(default_factory=dict)   # kind -> the LLMError kind raised for it
        fail_after: int = 0                   # ...after this many items
        refuse: list = field(default_factory=list)  # kinds answered with a refusal

    def __init__(self, conf: dict | None = None, ctx=None, **kw):
        """conf: the config block; keyword arguments override it (tests)."""
        super().__init__({**(conf or {}), **kw}, ctx)
        o = self.opts
        self.text = {k: getattr(o, k) for k in KINDS}
        self.step_s, self.chars = o.step_s, o.chars
        self.fail, self.fail_after, self.refuse = dict(o.fail), o.fail_after, set(o.refuse)
        if ctx is not None:
            from app.prompts import load_prompts
            self.prompts = load_prompts(ctx.cfg)
        else:
            from app.prompts import default_prompts
            self.prompts = default_prompts()
        self.calls = []                       # (kind, messages) per request
        self.live = 0; self.aborted = 0; self.lock = threading.Lock()

    def __getattr__(self, kind):                  # llm.reply, llm.leadin, ...: the scripted text of a kind
        if kind in KINDS and "text" in self.__dict__:
            return self.text[kind]
        raise AttributeError(kind)

    def _kind(self, messages):
        sys_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        tool_result = messages[-1]["role"] == "tool"
        for kind, section in (("leadin", "tool_leadin"), ("ack", "backchannel"), ("still", "still_looking"),
                              ("expert_leadin", "expert_leadin"), ("expert_still", "expert_still"),
                              ("smart", "smart"), ("task", "tasks"), ("task_failed", "task_failed")):
            if sys_txt.startswith(self.prompts.text(section, "system")):     # + appended hints (backchannel)
                return kind + "_b" if kind in ("smart", "task") and tool_result else kind
        return "phase_b" if tool_result else "reply"

    def text_for(self, kind):
        return self.text[kind]

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        kind = self._kind(messages)
        self.calls.append((kind, [dict(m) for m in messages]))
        raw_all = self.text_for(kind)
        with self.lock:
            self.live += 1
        raw, said, saw_tool, started = "", 0, False, None
        try:
            if kind in self.refuse:
                yield ("refusal", None); return
            for i in range(0, len(raw_all), self.chars):
                if kind in self.fail and i // self.chars >= self.fail_after:
                    raise LLMError(self.fail[kind])
                if wait_or_stop(self.step_s, stop_flag):
                    with self.lock:
                        self.aborted += 1
                    return
                raw += raw_all[i:i + self.chars]
                if started:
                    continue
                if not saw_tool:
                    speak, saw_tool = split_speakable(raw, HERMES)
                    if len(speak) > said:
                        yield ("text", speak[said:]); said = len(speak)
                if saw_tool and getattr(self, "emit_tool_start", True):
                    started = tool_start(raw, HERMES) if names_tool(tools) else {"name": None}
                    if started:
                        yield ("tool_start", started)
            if stop_flag and stop_flag():
                return
            if not saw_tool and len(raw) > said:
                yield ("text", raw[said:])
            if saw_tool:
                calls = HERMES.parse(raw)
                if not started and getattr(self, "emit_tool_start", True):
                    yield ("tool_start", {"name": calls[0]["name"] if calls else None})
                yield ("tool_calls", [{"id": new_call_id(), **c} for c in calls], None)
        finally:
            with self.lock:
                self.live -= 1
