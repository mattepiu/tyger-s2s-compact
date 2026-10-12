"""anthropic: the Claude Messages API (or a Claude-compatible endpoint via `base_url`), official `anthropic` SDK.

HOSTED: transcripts (and, in a tool turn, tool results) go to Anthropic. Allowed only because the config names this
plugin; the startup log says which roles are remote. Bodies are never logged; ANTHROPIC_LOG must not be debug.

  * Client: AsyncAnthropic on MOUTH's EngineLoop, one per instance; client.beta.messages.stream(...) with the
    server-side fallback beta: `fallbacks: "default"` re-runs a policy decline on Anthropic's recommended fallback
    model inside the same call (betas server-side-fallback-2026-07-01).
  * Thinking can't be turned off on Claude Opus 5.5: `effort` (output_config) is the control, from config (start
    at low; the model's default is medium). Thinking blocks stream with empty text by default and are never
    spoken: they only show up as time before the first text. `max_tokens` covers thinking + text, so it comes from
    config per role, not from MOUTH's max_new. No temperature / top_p / top_k (rejected); tool_choice stays auto.
  * Tools: {name, description, input_schema, eager_input_streaming: true}, in the fixed order of app/tools/.
    With eager streaming the server doesn't validate inputs, so every input is checked against its schema here
    (calls.check_arguments, as in openai-responses); a bad one comes back with "error" set and MOUTH never
    runs it.
    stop_reason max_tokens: no call is run.
  * Thinking blocks in history (preserved thinking): inside one tool round, phase B sends back the response's own
    content unchanged (the `native` of the tool_calls event). Completed turns are converted from the neutral
    fields only (text + tool_use / tool_result), so no thinking block is ever replayed after MOUTH edits history
    (trimming old turns, cutting a barged reply): dropping every earlier block is the allowed "remove a leading
    run" edit.
  * After a mid-stream fallback only blocks after the last `fallback` block count: tool calls are taken from
    there, and the native content keeps just the text before it (thinking / tool_use there must not be echoed).
  * Refusal that survives the fallback: ("refusal", stop_details.category); MOUTH speaks a neutral line.
  * Prompt caching: top-level cache_control (tools + system are a stable prefix).
  * Key, timeouts, concurrency, errors -> LLMError, shutdown: HostedLLM (hosted/base.py), as for openai-responses.
  * Abort: leaving the stream (stop_flag) closes the HTTP connection; tokens generated so far are billed.
"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass

from app.plugins.llm.calls import flag_invalid
from app.plugins.llm.hosted.base import HostedLLM

log = logging.getLogger("llm")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_claude(messages):
    """Neutral messages -> (system, Messages-API messages). Consecutive tool results become one user message of
    tool_result blocks. An assistant tool-call message carries its `native` content only while it is the open tool
    round (nothing but tool results after it); completed turns are rebuilt from text + tool_use."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    out = []
    for i, m in enumerate(rest):
        role = m["role"]
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if _is_error(m["content"]):
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("tool_calls"):
            open_round = all(x["role"] == "tool" for x in rest[i + 1:])
            if open_round and m.get("native"):
                out.append({"role": "assistant", "content": m["native"]})
            else:
                content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments", {})}
                            for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": m.get("content") or "..."})
    return system, out


def _is_error(content):
    try:
        x = json.loads(content)
    except (TypeError, ValueError):
        return False
    return isinstance(x, dict) and "error" in x


def to_claude_tools(tools):
    """OpenAI function schemas -> Claude tools, same order, with eager input streaming."""
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"]["parameters"], "eager_input_streaming": True} for t in tools or []]


def _d(x):
    """An SDK object or a dict -> dict (tests feed dicts)."""
    return x if isinstance(x, dict) else x.to_dict()


def _get(x, k):
    return x.get(k) if isinstance(x, dict) else getattr(x, k, None)


class ClaudeStreamParser:
    """Raw stream events + the final message -> neutral LLM events. Pure, so it is tested on synthetic events."""
    def __init__(self, tools=None):
        self.tools = tools; self.started = False

    def feed(self, event):
        t = _get(event, "type")
        if t == "content_block_start":
            block = _get(event, "content_block")
            if _get(block, "type") == "tool_use" and not self.started:
                self.started = True
                return [("tool_start", {"name": _get(block, "name")})]
        elif t == "content_block_delta":
            delta = _get(event, "delta")
            if _get(delta, "type") == "text_delta" and not self.started and _get(delta, "text"):
                return [("text", _get(delta, "text"))]
        return []

    def finish(self, final):
        if _get(final, "stop_reason") == "refusal":
            details = _get(final, "stop_details")
            return [("refusal", _get(details, "category") if details else None)]
        blocks = [_d(b) for b in _get(final, "content") or []]
        last_fb = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
        if last_fb >= 0:
            log.info("reply continued on a fallback model")
        after = blocks[last_fb + 1:]
        uses = [b for b in after if b.get("type") == "tool_use"]
        if not uses:
            return [("tool_calls", [], None)] if self.started else []
        if _get(final, "stop_reason") == "max_tokens":
            log.warning("tool input cut off at max_tokens; not run")
            return [("tool_calls", [], None)]
        native = [b for b in blocks[:last_fb + 1] if b.get("type") == "text"] + after
        calls = [flag_invalid({"id": b["id"], "name": b["name"], "arguments": b.get("input")}, self.tools)
                 for b in uses]
        return [("tool_calls", calls, native)]


class AnthropicLLM(HostedLLM):
    SDK = "anthropic"

    @dataclass(frozen=True, kw_only=True)
    class Options(HostedLLM.Options):
        model: str = "claude-opus-5-5"
        api_key_env: str = "ANTHROPIC_API_KEY"
        read_timeout_s: float = 15.0
        max_tokens: int = 1024                 # thinking + text: per role, never MOUTH's max_new
        effort: str = "low"
        fallbacks: str | None = "default"      # null turns server-side fallback off
        temperature: float | None = None       # sent only if set (Claude Opus 5.5 rejects them)
        top_p: float | None = None
        top_k: int | None = None

    def __init__(self, conf: dict, ctx):
        super().__init__(conf, ctx)
        o = self.opts
        self.max_tokens = o.max_tokens
        self.effort = o.effort
        self.fallbacks = o.fallbacks
        self.sampling = {k: getattr(o, k) for k in ("temperature", "top_p", "top_k") if getattr(o, k) is not None}
        self.cache_reads = 0

    def make_client(self, key):
        import anthropic
        o = self.opts
        return anthropic.AsyncAnthropic(api_key=key, base_url=o.base_url or None, max_retries=o.max_retries,
                                        timeout=anthropic.Timeout(o.read_timeout_s, connect=o.connect_timeout_s))

    def request(self, messages, tools=None):
        """The stream() keyword arguments for these neutral messages (also used by tests)."""
        system, msgs = to_claude(messages)
        kw = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs,
                  output_config={"effort": self.effort}, cache_control={"type": "ephemeral"}, **self.sampling)
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = to_claude_tools(tools)
        if self.fallbacks:
            kw["fallbacks"] = self.fallbacks; kw["betas"] = [FALLBACK_BETA]
        return kw

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        kw = self.request(messages, tools)
        box = {}

        async def events():
            async with self.client.beta.messages.stream(**kw) as s:   # leaving it closes the connection
                async for ev in s:
                    if ev.type in ("content_block_start", "content_block_delta"):
                        yield ev
                box["final"] = await s.get_final_message()

        p = ClaudeStreamParser(tools)
        for ev in self.eloop.iterate(lambda: self.guarded(events), stop_flag=stop_flag):
            yield from p.feed(ev)
        if (stop_flag and stop_flag()) or "final" not in box:
            return
        usage = box["final"].usage
        if getattr(usage, "cache_read_input_tokens", 0):
            self.cache_reads += 1
        log.info("llm %s: in %d (cache read %d, write %d) out %d tokens", self.name, usage.input_tokens or 0,
                 usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0, usage.output_tokens or 0)
        yield from p.finish(box["final"])
