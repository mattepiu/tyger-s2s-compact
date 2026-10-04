"""fake: a scripted LLM, no model (tests, CI, and running the stack without a GPU: config.fake.yaml).

Routes each request by its system prompt (read from prompts.yaml, like MOUTH does): tool lead-in / still-looking /
backchannel / the expert lead-in and still-looking lines / the smart model's answer (`smart`, or `smart_b` once a
tool result is in) / a background task on the smart model (`task`, `task_b`; by default it calls finish_task) /
phase B (a tool result present) / the main reply, and streams the scripted text for that kind a
few characters at a time, polling stop_flag between items like the real engine loop. `reply` and `smart` may contain
Hermes <tool_call>s, which go through the same splitting, naming (tool_start) and parsing as the local Qwen plugins. For tests of hosted
behaviour, `fail` maps a request kind to an LLMError kind raised for it (after `fail_after` items), and `refuse`
lists kinds answered with a refusal. Every request is recorded in `calls` (kind, messages); nothing is logged.
"""
from __future__ import annotations
import threading, time

from app.plugins.base import LLMError
from app.plugins.llm._toolparse import split_speakable, parse_calls, new_call_id, tool_start, names_tool

KINDS = ("reply", "phase_b", "leadin", "still", "ack", "expert_leadin", "expert_still", "smart", "smart_b", "task",
         "task_b", "task_failed")
FINISH_CALL = ('<tool_call>\n{"name": "finish_task", "arguments": {"summary": "About your task: it is done, and it '
               'went well.", "result": "The full result of the task, in plain text."}}\n</tool_call>')
DEFAULTS = {"reply": "Sure, here is a reasonably long first sentence for you. And a second one.",
            "phase_b": "It is sunny and twenty degrees there today, a lovely day.",
            "leadin": "One sec, let me look that up.", "still": "Still looking, almost there.", "ack": "oh nice",
            "expert_leadin": "Let me think that through.", "expert_still": "Still working it out.",
            "smart": "Here is the careful answer, worked out step by step. It holds up.",
            "smart_b": "Having checked, the careful answer is forty two.",
            "task": FINISH_CALL, "task_b": FINISH_CALL, "task_failed": "Sorry, I couldn't finish that one.",
            "step_s": 0.002, "chars": 4, "fail": {}, "fail_after": 0, "refuse": ()}


def wait_or_stop(seconds, stop_flag, poll=0.002):
    """Sleep `seconds`, returning early (True) if stop_flag() turns true, like the engine poll loop."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if stop_flag and stop_flag():
            return True
        time.sleep(poll)
    return bool(stop_flag and stop_flag())


class FakeLLM:
    remote = False

    def __init__(self, conf: dict | None = None, ctx=None, **kw):
        o = {**DEFAULTS, **(conf or {}), **kw}
        self.text = {k: o[k] for k in KINDS}
        self.step_s, self.chars = float(o["step_s"]), int(o["chars"])
        self.fail, self.fail_after, self.refuse = dict(o["fail"]), int(o["fail_after"]), set(o["refuse"])
        if ctx is not None:
            from app.prompts import load_prompts
            self.prompts = load_prompts(ctx.cfg, [ctx.name])[ctx.name]
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
                    speak, saw_tool = split_speakable(raw)
                    if len(speak) > said:
                        yield ("text", speak[said:]); said = len(speak)
                if saw_tool and getattr(self, "emit_tool_start", True):
                    started = tool_start(raw) if names_tool(tools) else {"name": None}
                    if started:
                        yield ("tool_start", started)
            if stop_flag and stop_flag():
                return
            if not saw_tool and len(raw) > said:
                yield ("text", raw[said:])
            if saw_tool:
                calls = parse_calls(raw, "hermes_json")
                if not started and getattr(self, "emit_tool_start", True):
                    yield ("tool_start", {"name": calls[0]["name"] if calls else None})
                yield ("tool_calls", [{"id": new_call_id(), **c} for c in calls], None)
        finally:
            with self.lock:
                self.live -= 1

    def warmup(self):
        pass

    def shutdown(self):
        pass
