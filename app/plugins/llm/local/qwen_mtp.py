"""qwen3.5-2b-mtp: unsloth/Qwen3.5-2B-MTP-GGUF via local llama-server or vLLM GGUF engine.

Connects to a local OpenAI-compatible endpoint with MTP (Multi-Token Prediction) speculative
decoding enabled for 1.5x-2x faster token generation rates (80-120+ tokens/s).
Parses Hermes tool call syntax:
  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
Zero data leaves the box (remote: false).
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator

from app.plugins.base import LLMError, LLMPlugin, LLMEvent
from app.plugins.options import PluginOptions
from app.plugins.llm.fake import HERMES
from app.plugins.llm.calls import (
    CallFormat,
    split_speakable,
    tool_start,
    names_tool,
    new_call_id,
    flag_invalid,
)

log = logging.getLogger("llm")


def to_chat_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral Tyger messages to OpenAI Chat Completions messages."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                calls.append({
                    "id": c.get("id") or new_call_id(),
                    "type": "function",
                    "function": {"name": c["name"], "arguments": args_str},
                })
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": calls,
            })
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": str(m.get("content", "")),
            })
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


class Qwen35MTPLocal(LLMPlugin):
    calls: CallFormat = HERMES

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        model: str = "unsloth/Qwen3.5-2B-MTP-GGUF"
        base_url: str = "http://127.0.0.1:8080/v1"
        api_key_env: str | None = None
        api_key: str | None = None
        remote: bool = False
        temperature: float = 0.7
        max_tokens: int | None = None
        read_timeout_s: float = 10.0
        connect_timeout_s: float = 2.0
        max_retries: int = 1
        max_concurrency: int = 4
        queue_timeout_s: float = 2.0

    def __init__(self, conf: dict | None = None, ctx=None):
        super().__init__(conf, ctx)
        if self.eloop is None:
            from app.runtime.event_loop import EngineLoop
            self.eloop = EngineLoop()
        o = self.opts
        self.model = o.model
        self.remote = o.remote
        self.temperature = o.temperature
        self.max_tokens = o.max_tokens

        key = "none"
        if o.api_key:
            key = o.api_key
        elif o.api_key_env:
            key = os.environ.get(o.api_key_env) or "none"

        import httpx
        from openai import AsyncOpenAI

        async def make():
            client = AsyncOpenAI(
                api_key=key,
                base_url=o.base_url,
                max_retries=o.max_retries,
                timeout=httpx.Timeout(o.read_timeout_s, connect=o.connect_timeout_s),
            )
            sem = asyncio.Semaphore(o.max_concurrency)
            return client, sem

        self.client, self.sem = self.eloop.run(make())

    async def guarded(self, responses):
        """Run responses() in a concurrency slot with SDK error conversion."""
        try:
            await asyncio.wait_for(self.sem.acquire(), self.opts.queue_timeout_s)
        except asyncio.TimeoutError:
            raise LLMError("busy") from None
        try:
            async for item in responses():
                yield item
        except LLMError:
            raise
        except Exception as e:
            err = self.llm_error(e)
            if err is None:
                raise
            raise err from None
        finally:
            self.sem.release()

    def llm_error(self, e: Exception) -> LLMError | None:
        """Map OpenAI exceptions to Tyger LLMError kinds."""
        try:
            import openai
            if isinstance(e, openai.RateLimitError):
                log.warning("llm %s: rate limited (429)", self.name)
                return LLMError("rate_limit")
            if isinstance(e, openai.APIStatusError):
                log.warning("llm %s: HTTP %d", self.name, e.status_code)
                return LLMError("server" if e.status_code >= 500 else "request")
            if isinstance(e, openai.APITimeoutError):
                return LLMError("timeout")
            if isinstance(e, openai.APIConnectionError):
                return LLMError("connection")
        except Exception:
            pass
        return None

    def stream(self, messages, tools=None, max_new=192, stop_flag=None) -> Iterator[LLMEvent]:
        chat_msgs = to_chat_messages(messages)
        max_output = int(self.max_tokens or max_new)
        kw = {
            "model": self.model,
            "messages": chat_msgs,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": max_output,
        }
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"

        async def events():
            stream_obj = await self.client.chat.completions.create(**kw)
            try:
                async for chunk in stream_obj:
                    yield chunk
            finally:
                await stream_obj.close()

        fmt = self.calls
        raw, said, saw_tool, started = "", 0, False, None
        names = [t["function"]["name"] for t in tools or []]
        wait_name = names_tool(tools)
        visible_now = lambda final=False: fmt.normalize(raw, names, final)  # noqa: E731

        structured_calls: dict[int, dict] = {}
        saw_structured = False

        for chunk in self.eloop.iterate(lambda: self.guarded(events), stop_flag=stop_flag):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "refusal", None) or getattr(choice, "finish_reason", None) == "refusal":
                yield ("refusal", "refusal")
                return

            # Check for OpenAI-structured tool calls
            if getattr(delta, "tool_calls", None):
                saw_tool = True
                saw_structured = True
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in structured_calls:
                        structured_calls[idx] = {"name": "", "arguments": ""}
                    if tc.function:
                        if tc.function.name:
                            structured_calls[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            structured_calls[idx]["arguments"] += tc.function.arguments
                if not started:
                    first_name = structured_calls[0]["name"] if (0 in structured_calls and structured_calls[0]["name"]) else None
                    if not wait_name or first_name:
                        started = {"name": first_name}
                        yield ("tool_start", started)
                continue

            # Standard content delta (may contain Hermes JSON tool calls)
            d = getattr(delta, "content", None) or ""
            if not d:
                continue
            raw += d
            if started:
                continue
            if not saw_tool:
                speak, saw_tool = split_speakable(visible_now(), fmt)
                if len(speak) > said:
                    yield ("text", speak[said:])
                    said = len(speak)
            if saw_tool:
                started = tool_start(visible_now(), fmt) if wait_name else {"name": None}
                if started:
                    yield ("tool_start", started)

        if stop_flag and stop_flag():
            return

        if saw_structured:
            if not started:
                first_name = structured_calls[0]["name"] if (0 in structured_calls and structured_calls[0]["name"]) else None
                yield ("tool_start", {"name": first_name})
            parsed_calls = []
            for idx in sorted(structured_calls.keys()):
                sc = structured_calls[idx]
                name = sc["name"]
                try:
                    args = json.loads(sc["arguments"]) if sc["arguments"].strip() else {}
                except Exception:
                    args = {}
                parsed_calls.append(flag_invalid({"id": new_call_id(), "name": name, "arguments": args}, tools))
            yield ("tool_calls", parsed_calls, None)
            return

        visible = visible_now(final=True)
        if not saw_tool:
            speak, saw_tool = split_speakable(visible, fmt)
            if len(speak) > said:
                yield ("text", speak[said:])
        if saw_tool:
            calls = fmt.parse(visible)
            if not started:
                yield ("tool_start", {"name": calls[0]["name"] if calls else None})
            parsed_calls = [flag_invalid({"id": new_call_id(), **c}, tools) for c in calls]
            yield ("tool_calls", parsed_calls, None)

    def warmup(self) -> None:
        pass

    def shutdown(self) -> None:
        if hasattr(self, "client") and self.client is not None and self.eloop is not None:
            try:
                self.eloop.run(self.client.close(), timeout=5)
            except Exception:
                pass
