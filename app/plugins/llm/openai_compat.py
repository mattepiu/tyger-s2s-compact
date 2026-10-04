"""openai: any OpenAI-compatible chat-completions endpoint (OpenAI, Groq, Cerebras, a vLLM server, ...).

HOSTED: transcripts (and, in a tool turn, tool results) go to the endpoint in `base_url`. That is allowed only
because the config names this plugin explicitly; the startup log says which roles are remote. Request and response
bodies are never logged; keep OPENAI_LOG unset (the SDK's debug logging prints bodies).

  * Client: AsyncOpenAI(base_url, api_key from the env var `api_key_env`) on MOUTH's EngineLoop, one per instance
    so connections are reused; chat.completions.create(stream=True).
  * Streaming: delta.content -> text (through a <think> guard: some models put reasoning in content); the first
    delta.tool_calls fragment -> tool_start (with the name when the server sends it); arguments accumulate per
    index and are parsed at the end. reasoning / reasoning_content fields are never read. Tool calls written as
    <tool_call> text in content are parsed too (logged once: the model lacks native tool calls); their tool_start
    waits for the written name, as in the local plugins.
  * Tool-call IDs are this plugin's own (call_<8 hex>), unique for the session whatever the server sends. A call
    whose arguments don't parse or lack a required key is returned with "error" set: MOUTH never runs it.
  * Reasoning models (gpt-5*, o-series): max_completion_tokens instead of max_tokens (guessed from the model name,
    `token_param` overrides); `reasoning_effort` / `verbosity` go in `extra_body`. Their thinking is never read.
  * Fails fast (a voice turn can't wait): max_retries 1, short timeouts, a 429 is not waited out. Every failure is
    an LLMError, and MOUTH speaks a short apology. `max_concurrency` bounds requests in flight per instance.
  * Abort: when stop_flag turns true the stream is closed (tokens already generated are still billed).
"""
from __future__ import annotations
import asyncio, json, logging, os

from app.plugins.base import LLMError
from app.plugins.llm._toolparse import split_speakable, strip_think, parse_calls, new_call_id, tool_start

log = logging.getLogger("llm")


def to_openai_messages(messages):
    """Neutral history -> chat-completions messages: tool calls in the nested shape with JSON-string arguments;
    `native` dropped (this plugin writes none)."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content"),
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"],
                                                     "arguments": json.dumps(c.get("arguments", {}),
                                                                             ensure_ascii=False)}}
                                       for c in m["tool_calls"]]})
        elif m.get("role") == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content")})
    return out


def check_arguments(name, args, tools):
    """None if `args` suits tool `name`'s schema (a dict with its required keys), else the reason."""
    if not isinstance(args, dict):
        return "arguments are not an object"
    for t in tools or []:
        f = t["function"]
        if f["name"] == name:
            missing = [k for k in f["parameters"].get("required", []) if k not in args]
            return f"missing {', '.join(missing)}" if missing else None
    return "unknown tool"


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


class ChatStreamParser:
    """Chat-completions stream chunks -> neutral LLM events. Pure (no I/O), so it is tested on recorded synthetic
    chunk sequences. feed(chunk) and finish() return lists of events."""
    def __init__(self, tools=None, think_guard=True):
        self.tools = tools; self.think_guard = think_guard
        self.raw = ""; self.said = 0; self.text_tool = False; self.started = False
        self.calls = {}                       # index -> {"name", "args"}
        self.finish_reason = None; self.refusal = None

    def _visible(self, final=False):
        return strip_think(self.raw, final) if self.think_guard else self.raw

    def feed(self, chunk):
        out = []
        for choice in _get(chunk, "choices") or []:
            delta = _get(choice, "delta") or {}
            if _get(choice, "finish_reason"):
                self.finish_reason = _get(choice, "finish_reason")
            if _get(delta, "refusal"):
                self.refusal = (self.refusal or "") + _get(delta, "refusal")
            for tc in _get(delta, "tool_calls") or []:
                fn = _get(tc, "function") or {}
                acc = self.calls.setdefault(_get(tc, "index") or 0, {"name": None, "args": ""})
                acc["name"] = acc["name"] or _get(fn, "name")
                acc["args"] += _get(fn, "arguments") or ""
                if not self.started:
                    self.started = True; out.append(("tool_start", {"name": acc["name"]}))
            content = _get(delta, "content")
            if content:
                self.raw += content                  # kept whole: a <tool_call> written as text is parsed at the end
                if not self.started and not self.text_tool:
                    speak, self.text_tool = split_speakable(self._visible())
                    if len(speak) > self.said:
                        out.append(("text", speak[self.said:])); self.said = len(speak)
                if self.text_tool and not self.started:      # start once the written call names its tool
                    start = tool_start(self._visible())
                    if start:
                        self.started = True; out.append(("tool_start", start))
        return out

    def finish(self):
        out = []
        if self.refusal and not self.calls and not self.said:
            return [("refusal", None)]
        if not self.started and not self.text_tool:
            visible = self._visible(final=True)
            if len(visible) > self.said:
                out.append(("text", visible[self.said:]))
            return out
        if not self.started:                          # a written call that ended before naming its tool
            self.started = True; out.append(("tool_start", {"name": None}))
        if self.finish_reason == "length":            # cut off mid-call: never run a truncated call
            log.warning("tool call cut off at max_tokens; not run")
            return out + [("tool_calls", [], None)]
        calls = []
        if self.calls:
            for i in sorted(self.calls):
                acc = self.calls[i]
                try:
                    args = json.loads(acc["args"]) if acc["args"].strip() else {}
                except ValueError:
                    args = None
                calls.append({"name": acc["name"] or "", "arguments": args})
        else:                                        # <tool_call> written as text: no native tool calls
            if not getattr(ChatStreamParser, "_warned", False):
                ChatStreamParser._warned = True
                log.warning("model wrote tool calls as text, not structured tool_calls; parsing the text")
            calls = parse_calls(self._visible(final=True), "hermes_json")
        events = []
        for c in calls:
            c = {"id": new_call_id(), "name": c["name"], "arguments": c["arguments"]}
            bad = check_arguments(c["name"], c["arguments"], self.tools)
            if bad:
                c["error"] = f"invalid arguments for '{c['name']}': {bad}"
                c["arguments"] = c["arguments"] if isinstance(c["arguments"], dict) else {}
            events.append(c)
        return out + [("tool_calls", events, None)]


class OpenAICompatLLM:
    def __init__(self, conf: dict, ctx):
        import httpx
        from openai import AsyncOpenAI
        self.conf = conf; self.name = ctx.name; self.eloop = ctx.eloop
        self.model = conf["model"]
        self.remote = bool(conf.get("remote", True))
        key_env = conf.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"llm {self.name}: {key_env} is not set (put it in .env)")
        if (os.environ.get("OPENAI_LOG") or "").lower() == "debug":
            raise RuntimeError("OPENAI_LOG=debug would log request bodies (transcripts): unset it")
        self.max_tokens = conf.get("max_tokens")
        # OpenAI's reasoning models (gpt-5*, o-series) take max_completion_tokens (thinking + text) and reject
        # max_tokens; other endpoints (Groq, Cerebras, vLLM) take max_tokens. `token_param` overrides the guess.
        self.token_param = conf.get("token_param") or (
            "max_completion_tokens" if str(self.model).startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens")
        self.temperature = conf.get("temperature")             # sent only if set
        self.extra_body = conf.get("extra_body") or None
        self.think_guard = bool(conf.get("strip_think", True))
        timeout = httpx.Timeout(float(conf.get("read_timeout_s", 10.0)), connect=float(conf.get("connect_timeout_s", 2.0)))

        async def make():
            return AsyncOpenAI(api_key=key, base_url=conf.get("base_url"), timeout=timeout,
                               max_retries=int(conf.get("max_retries", 1))), \
                asyncio.Semaphore(int(conf.get("max_concurrency", 8)))
        self.client, self.sem = self.eloop.run(make())
        self.wait_s = float(conf.get("queue_timeout_s", 2.0))
        self.rate_limited = 0

    def _error(self, e):
        import openai
        if isinstance(e, openai.RateLimitError):
            self.rate_limited += 1
            log.warning("llm %s: rate limited (429), %d so far", self.name, self.rate_limited)
            return LLMError("rate_limit")
        if isinstance(e, openai.APITimeoutError):
            return LLMError("timeout")
        if isinstance(e, openai.APIStatusError):
            log.warning("llm %s: HTTP %d", self.name, e.status_code)
            return LLMError("server" if e.status_code >= 500 else "request")
        if isinstance(e, openai.APIConnectionError):
            return LLMError("connection")
        return None

    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        kw = dict(model=self.model, messages=to_openai_messages(messages), stream=True,
                  **{self.token_param: int(self.max_tokens or max_new)})
        if tools:
            kw["tools"] = tools
        if self.temperature is not None:
            kw["temperature"] = float(self.temperature)
        if self.extra_body:
            kw["extra_body"] = self.extra_body

        async def chunks():
            try:
                await asyncio.wait_for(self.sem.acquire(), self.wait_s)
            except asyncio.TimeoutError:
                raise LLMError("busy") from None
            try:
                s = await self.client.chat.completions.create(**kw)
                try:
                    async for c in s:
                        yield c
                finally:
                    await s.close()                      # also on cancel (barge-in): closes the HTTP stream
            except Exception as e:
                err = self._error(e)
                if err is None:
                    raise
                raise err from None
            finally:
                self.sem.release()

        p = ChatStreamParser(tools, self.think_guard)
        for c in self.eloop.iterate(chunks, stop_flag=stop_flag):
            yield from p.feed(c)
        if stop_flag and stop_flag():
            return
        yield from p.finish()

    def warmup(self):
        pass                      # a request would cost money; the first turn pays the connection set-up

    def shutdown(self):
        try:
            self.eloop.run(self.client.close(), timeout=5)
        except Exception:
            pass
