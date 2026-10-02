"""One asyncio event loop, in a background thread, hosting MOUTH's in-process engines (vLLM AsyncLLM for the
responder, vLLM-Omni AsyncOmni for VoxCPM2). Both run "offline" in this process -- no HTTP hop between MOUTH
and the models. MOUTH itself stays synchronous (generators + threads), so EngineLoop bridges:

  run(coro)                    -> result of a coroutine on the loop (engine construction, warmup, ...)
  iterate(agen_fn, abort, stop_flag)
                               -> a plain generator over an async generator running on the loop. When
                                  stop_flag() turns true (barge-in) or the consumer stops early, the async
                                  side is cancelled and abort() is awaited so the engine frees the request.
"""
from __future__ import annotations
import asyncio, queue, threading

_END = object()


class _Raised:
    def __init__(self, exc):
        self.exc = exc


class EngineLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="engine-loop", daemon=True)
        self._thread.start()

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def iterate(self, agen_fn, abort=None, stop_flag=None, poll_s=0.005):
        """Yield the items of agen_fn() (an async generator, created on the loop) as they arrive."""
        q: queue.Queue = queue.Queue()

        async def pump():
            try:
                async for item in agen_fn():
                    q.put(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:                       # surface engine errors in the consumer thread
                q.put(_Raised(e))
            finally:
                q.put(_END)

        fut = asyncio.run_coroutine_threadsafe(pump(), self.loop)
        try:
            while not (stop_flag and stop_flag()):
                try:
                    item = q.get(timeout=poll_s)
                except queue.Empty:
                    continue
                if item is _END:
                    return
                if isinstance(item, _Raised):
                    raise item.exc
                yield item
        finally:
            if not fut.done():
                fut.cancel()
                if abort is not None:
                    asyncio.run_coroutine_threadsafe(abort(), self.loop)
