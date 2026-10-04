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
  * Tools: {name, description, input_schema, eager_input_streaming: true}, in the fixed order of app/tools.py.
    With eager streaming the server doesn't validate inputs, so every input is checked against its schema here;
    a bad one comes back with "error" set and MOUTH never runs it. stop_reason max_tokens: no call is run.
  * Thinking blocks in history (preserved thinking): inside one tool round, phase B sends back the response's own
    content unchanged (the `native` of the tool_calls event). Completed turns are converted from the neutral
    fields only (text + tool_use / tool_result), so no thinking block is ever replayed after MOUTH edits history
    (trimming old turns, cutting a barged reply): dropping every earlier block is the allowed "remove a leading
    run" edit.
  * After a mid-stream fallback only blocks after the last `fallback` block count: tool calls are taken from
    there, and the native content keeps just the text before it (thinking / tool_use there must not be echoed).
  * Refusal that survives the fallback: ("refusal", stop_details.category); MOUTH speaks a neutral line.
  * Prompt caching: top-level cache_control (tools + system are a stable prefix).
  * Errors: typed SDK exceptions, most specific first, -> LLMError; max_retries 1, short timeouts.
  * Abort: leaving the stream (stop_flag) closes the HTTP connection; tokens generated so far are billed.
"""
from __future__ import annotations
import asyncio, json, logging, os

from app.plugins.base import LLMError

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


_JSON_TYPES = {"string": str, "object": dict, "array": list, "boolean": bool, "integer": int, "number": (int, float)}


def check_input(name, args, tools):
    """None if `args` fits tool `name`'s schema (an object, required keys present, declared types and enums
    respected), else the reason. Eager streaming means the server didn't check it."""
    if not isinstance(args, dict):
        return "input is not an object"
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
        calls = []
        for b in uses:
            c = {"id": b["id"], "name": b["name"], "arguments": b.get("input")}
            bad = check_input(c["name"], c["arguments"], self.tools)
            if bad:
                c["error"] = f"invalid arguments for '{c['name']}': {bad}"
                c["arguments"] = c["arguments"] if isinstance(c["arguments"], dict) else {}
            calls.append(c)
        return [("tool_calls", calls, native)]


class AnthropicLLM:
    def __init__(self, conf: dict, ctx):
        import anthropic
        self.conf = conf; self.name = ctx.name; self.eloop = ctx.eloop
        self.model = conf.get("model") or "claude-opus-5-5"
        self.remote = bool(conf.get("remote", True))
        key_env = conf.get("api_key_env", "ANTHROPIC_API_KEY")
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"llm {self.name}: {key_env} is not set (put it in .env)")
        if (os.environ.get("ANTHROPIC_LOG") or "").lower() == "debug":
            raise RuntimeError("ANTHROPIC_LOG=debug would log request bodies (transcripts): unset it")
        self.max_tokens = int(conf.get("max_tokens", 1024))
        self.effort = conf.get("effort", "low")
        self.fallbacks = conf.get("fallbacks", "default")       # null in config turns server-side fallback off
        self.sampling = {k: conf[k] for k in ("temperature", "top_p", "top_k") if k in conf}   # only if set
        timeout = anthropic.Timeout(float(conf.get("read_timeout_s", 15.0)), connect=float(conf.get("connect_timeout_s", 2.0)))

        async def make():
            return anthropic.AsyncAnthropic(api_key=key, base_url=conf.get("base_url") or None, timeout=timeout,
                                            max_retries=int(conf.get("max_retries", 1))), \
                asyncio.Semaphore(int(conf.get("max_concurrency", 8)))
        self.client, self.sem = self.eloop.run(make())
        self.wait_s = float(conf.get("queue_timeout_s", 2.0))
        self.rate_limited = 0; self.cache_reads = 0

    def _error(self, e):
        import anthropic
        if isinstance(e, anthropic.RateLimitError):
            self.rate_limited += 1
            log.warning("llm %s: rate limited (429), %d so far", self.name, self.rate_limited)
            return LLMError("rate_limit")
        if isinstance(e, anthropic.APIStatusError):
            log.warning("llm %s: HTTP %d", self.name, e.status_code)
            return LLMError("server" if e.status_code >= 500 else "request")
        if isinstance(e, anthropic.APITimeoutError):
            return LLMError("timeout")
        if isinstance(e, anthropic.APIConnectionError):
            return LLMError("connection")
        return None

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
            try:
                await asyncio.wait_for(self.sem.acquire(), self.wait_s)
            except asyncio.TimeoutError:
                raise LLMError("busy") from None
            try:
                async with self.client.beta.messages.stream(**kw) as s:   # leaving it closes the connection
                    async for ev in s:
                        if ev.type in ("content_block_start", "content_block_delta"):
                            yield ev
                    box["final"] = await s.get_final_message()
            except LLMError:
                raise
            except Exception as e:
                err = self._error(e)
                if err is None:
                    raise
                raise err from None
            finally:
                self.sem.release()

        p = ClaudeStreamParser(tools)
        for ev in self.eloop.iterate(events, stop_flag=stop_flag):
            yield from p.feed(ev)
        if (stop_flag and stop_flag()) or "final" not in box:
            return
        usage = box["final"].usage
        if getattr(usage, "cache_read_input_tokens", 0):
            self.cache_reads += 1
        log.info("llm %s: in %d (cache read %d, write %d) out %d tokens", self.name, usage.input_tokens or 0,
                 usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0, usage.output_tokens or 0)
        yield from p.finish(box["final"])

    def warmup(self):
        pass                      # a request would cost money; the first turn pays the connection set-up

    def shutdown(self):
        try:
            self.eloop.run(self.client.close(), timeout=5)
        except Exception:
            pass
