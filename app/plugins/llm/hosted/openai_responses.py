"""openai-responses: OpenAI's Responses API with OpenAI's own web search, for the `smart` role. Weather, news,
prices and other live facts go to the expert (ask_expert) or a background task, and this model looks them up itself:
the local tools have no web search (2026-10-11, the user's call).

HOSTED: transcripts (and, in a tool turn, tool results) go to OpenAI, and the model's searches run on OpenAI's side.
That is allowed only because the config names this plugin explicitly; the startup log says which roles are remote.
Request and response bodies are never logged; keep OPENAI_LOG unset (the SDK's debug logging prints bodies). Each web
search is billed on top of the tokens.

  * Client: AsyncOpenAI on MOUTH's EngineLoop, one per instance; responses.create(stream=True, store=False).
  * Tools: the offered function tools (chat-completions schemas, flattened to the Responses form), plus
    {"type": "web_search"} when `web_search` is on. The searches happen inside the response; they are only counted
    in the log.
  * Streaming: response.output_text.delta -> text; a function_call output item -> tool_start (with its name, as
    the API sends it whole); at response.completed the function calls -> tool_calls (arguments parsed and checked
    against the offered schema: calls.flag_invalid; IDs are this plugin's own). A refusal -> ("refusal",
    "refusal"). A response cut off by max_output_tokens never runs a call cut off with it. response.failed / an
    error event -> LLMError.
  * History: neutral messages -> Responses input items: {role, content} messages; an assistant tool call ->
    function_call items {call_id, name, arguments}; a tool result -> function_call_output {call_id, output}.
  * Reasoning: `reasoning_effort` -> reasoning.effort (None: the model's default). Thinking is never read.
  * Key, timeouts, concurrency, errors -> LLMError, shutdown: HostedLLM (hosted/base.py), as for openai and anthropic.
  * Abort: when stop_flag turns true the stream is closed (tokens already generated are still billed).
"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass

from app.plugins.base import LLMError
from app.plugins.llm.hosted.base import HostedLLM
from app.plugins.llm.calls import flag_invalid, new_call_id

log = logging.getLogger("llm")


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def to_responses_input(messages):
    """Neutral history -> Responses input items. An assistant turn with tool calls becomes its text (if any) and one
    function_call item per call; a tool result a function_call_output; `native` dropped (this plugin writes none)."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"type": "function_call", "call_id": c["id"], "name": c["name"],
                     "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)} for c in m["tool_calls"]]
        elif role == "tool":
            out.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def to_responses_tools(tools, web_search=False):
    """Chat-completions function schemas -> the Responses form ({type, name, description, parameters}), plus OpenAI's
    web search tool."""
    out = [{"type": "function", "name": t["function"]["name"], "description": t["function"].get("description", ""),
            "parameters": t["function"]["parameters"]} for t in tools or []]
    return out + ([{"type": "web_search"}] if web_search else [])


class ResponsesStreamParser:
    """Responses stream events -> neutral LLM events. Pure (no I/O), so it is tested on synthetic event sequences.
    feed(event) and finish() return lists of events."""
    def __init__(self, tools=None):
        self.tools = tools
        self.started = False; self.calls = []; self.refused = False; self.cut_off = False; self.searches = 0
        self.done = False

    def feed(self, ev):
        t = _get(ev, "type"); out = []
        if t == "response.output_text.delta":
            d = _get(ev, "delta")
            if d:
                out.append(("text", d))
        elif t == "response.output_item.added":
            item = _get(ev, "item")
            if _get(item, "type") == "function_call" and not self.started:
                self.started = True; out.append(("tool_start", {"name": _get(item, "name")}))
        elif t == "response.output_item.done":
            item = _get(ev, "item"); kind = _get(item, "type")
            if kind == "function_call":
                self.calls.append((_get(item, "name") or "", _get(item, "arguments") or ""))
            elif kind == "web_search_call":
                self.searches += 1
        elif t in ("response.refusal.delta", "response.refusal.done"):
            self.refused = True
        elif t == "response.incomplete":
            reason = _get(_get(_get(ev, "response"), "incomplete_details"), "reason")
            self.cut_off = reason == "max_output_tokens"; self.done = True
        elif t == "response.completed":
            self.done = True
        elif t in ("response.failed", "error"):
            raise LLMError("server")
        return out

    def finish(self):
        if self.searches:
            log.info("smart model web searches: %d", self.searches)   # a count only, never the queries
        if self.refused and not self.calls:
            return [("refusal", "refusal")]
        if not self.calls:
            return []
        out = [] if self.started else [("tool_start", {"name": self.calls[0][0]})]
        if self.cut_off:                              # cut off mid-call: never run a truncated call
            log.warning("tool call cut off at max_output_tokens; not run")
            return out + [("tool_calls", [], None)]
        calls = []
        for name, raw in self.calls:
            try:
                args = json.loads(raw) if raw.strip() else {}
            except ValueError:
                args = None
            calls.append(flag_invalid({"id": new_call_id(), "name": name, "arguments": args}, self.tools))
        return out + [("tool_calls", calls, None)]


class OpenAIResponsesLLM(HostedLLM):
    SDK = "openai"

    @dataclass(frozen=True, kw_only=True)
    class Options(HostedLLM.Options):
        api_key_env: str = "OPENAI_API_KEY"
        read_timeout_s: float = 30.0           # a web search can take several seconds before the first text
        max_tokens: int | None = None          # if set, every request gets it instead of MOUTH's max_new
        web_search: bool = True                # offer OpenAI's web search tool
        reasoning_effort: str | None = None    # none / minimal / low / ...; None: the model's default
        temperature: float | None = None       # sent only if set

    def __init__(self, conf: dict, ctx):
        super().__init__(conf, ctx)
        o = self.opts
        self.max_tokens = o.max_tokens

    def make_client(self, key):
        import httpx
        from openai import AsyncOpenAI
        o = self.opts
        return AsyncOpenAI(api_key=key, base_url=o.base_url, max_retries=o.max_retries,
                           timeout=httpx.Timeout(o.read_timeout_s, connect=o.connect_timeout_s))

    def request(self, messages, tools=None, max_new=192) -> dict:
        """The responses.create keyword arguments for one request (no I/O: tested as is)."""
        o = self.opts
        kw = dict(model=self.model, input=to_responses_input(messages), stream=True, store=False,
                  max_output_tokens=int(self.max_tokens or max_new))
        rtools = to_responses_tools(tools, o.web_search)
        if rtools:
            kw["tools"] = rtools
        if o.reasoning_effort:
            kw["reasoning"] = {"effort": o.reasoning_effort}
        if o.temperature is not None:
            kw["temperature"] = float(o.temperature)
        return kw

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        kw = self.request(messages, tools, max_new)

        async def events():
            s = await self.client.responses.create(**kw)
            try:
                async for e in s:
                    yield e
            finally:
                await s.close()                          # also on cancel (barge-in): closes the HTTP stream

        p = ResponsesStreamParser(tools)
        for e in self.eloop.iterate(lambda: self.guarded(events), stop_flag=stop_flag):
            yield from p.feed(e)
        if stop_flag and stop_flag():
            return
        yield from p.finish()
