"""HostedLLM: what the hosted LLM plugins (openai-responses, anthropic) share. Each subclass names its SDK and writes
its request and stream parsing; this base does the rest the same way for both:

  * the API key from the env var `api_key_env` (.env); missing -> the plugin refuses to start, naming the var;
  * the SDK's debug logging (`<SDK>_LOG=debug`) prints request bodies, i.e. transcripts: the plugin refuses to start;
  * one client per instance, made on MOUTH's EngineLoop, with short timeouts and `max_retries` (fails fast: a voice
    turn can't wait);
  * at most `max_concurrency` requests in flight; one that waits longer than `queue_timeout_s` for a slot is
    LLMError("busy");
  * the SDK's typed errors -> LLMError (rate_limit, timeout, connection, server, request), and MOUTH speaks a
    short apology. 429s are counted and logged; bodies never are;
  * no warmup (a request would cost money; the first turn pays the connection set-up); shutdown closes the client.
"""
from __future__ import annotations
import asyncio, importlib, logging, os
from abc import abstractmethod
from dataclasses import dataclass

from app.plugins.base import LLMError, LLMPlugin
from app.plugins.options import PluginOptions

log = logging.getLogger("llm")


class HostedLLM(LLMPlugin):
    SDK: str                                   # the SDK's module name: openai, anthropic

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        model: str
        base_url: str | None = None            # None: the SDK's default endpoint
        api_key_env: str                       # the env var (.env) that holds the key
        remote: bool = True                    # False only for an endpoint on this box
        read_timeout_s: float = 10.0
        connect_timeout_s: float = 2.0
        max_retries: int = 1
        max_concurrency: int = 8               # requests in flight per instance
        queue_timeout_s: float = 2.0           # waiting longer for a slot is LLMError("busy")

    def __init__(self, conf: dict, ctx):
        super().__init__(conf, ctx)
        o = self.opts
        self.model = o.model
        self.remote = o.remote
        key = os.environ.get(o.api_key_env)
        if not key:
            raise RuntimeError(f"llm {self.name}: {o.api_key_env} is not set (put it in .env)")
        log_env = f"{self.SDK.upper()}_LOG"
        if (os.environ.get(log_env) or "").lower() == "debug":
            raise RuntimeError(f"{log_env}=debug would log request bodies (transcripts): unset it")

        async def make():                      # on the loop that will use them
            return self.make_client(key), asyncio.Semaphore(o.max_concurrency)
        self.client, self.sem = self.eloop.run(make())
        self.rate_limited = 0

    @abstractmethod
    def make_client(self, key):
        """The SDK's async client for this instance's options (called on the EngineLoop)."""

    async def guarded(self, responses):
        """Run `responses()` (an async generator of stream items) in a concurrency slot, its SDK errors as
        LLMError. Leaving it early (stop_flag, barge-in) closes the generator, and with it the HTTP stream."""
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

    def llm_error(self, e) -> LLMError | None:
        """An SDK exception as LLMError (both SDKs name their errors alike), or None if it isn't the SDK's."""
        sdk = importlib.import_module(self.SDK)
        if isinstance(e, sdk.RateLimitError):          # an APIStatusError: first
            self.rate_limited += 1
            log.warning("llm %s: rate limited (429), %d so far", self.name, self.rate_limited)
            return LLMError("rate_limit")
        if isinstance(e, sdk.APIStatusError):
            log.warning("llm %s: HTTP %d", self.name, e.status_code)
            return LLMError("server" if e.status_code >= 500 else "request")
        if isinstance(e, sdk.APITimeoutError):         # an APIConnectionError: first
            return LLMError("timeout")
        if isinstance(e, sdk.APIConnectionError):
            return LLMError("connection")
        return None

    def shutdown(self):
        try:
            self.eloop.run(self.client.close(), timeout=5)
        except Exception:
            pass
