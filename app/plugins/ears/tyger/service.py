"""EarsService: the GPU thread. Conversations call `await feed(...)` once per hop; the thread gathers the hops of all
active conversations (up to `gather_s`) and runs them as one batched tick of the engine. A conversation that is
behind (said `more`: its next hop is already buffered) resubmits right after its tick resolves, so the next gather
waits up to `expect_s` for it: catch-up hops share one tick instead of each costing a tick of their own.
"""
from __future__ import annotations
import logging, queue, threading, time

import numpy as np, torch

from .engine import EarsEngine, Job
from .session import EarsSession

log = logging.getLogger("ears")


class EarsService:
    """The GPU thread, the conversations' slots, and the tick statistics log (module docstring)."""
    def __init__(self, engine: EarsEngine, gather_s=0.015, expect_s=0.06, stats_every_s=60.0):
        self.engine = engine; self.gather_s = gather_s; self.expect_s = expect_s
        self.stats_every_s = stats_every_s; self.expect = set()
        self.free = list(range(engine.slots)); self.sessions = set(); self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self._ticks = []; self._t_stats = time.time()
        threading.Thread(target=self._loop, name="ears-gpu", daemon=True).start()

    def full(self):
        with self.lock:
            return not self.free

    def active(self):
        with self.lock:
            return len(self.sessions)

    def open(self, **thresholds):
        with self.lock:
            if not self.free:
                return None
            s = EarsSession(self.free.pop(0), **thresholds); self.sessions.add(s)
            return s

    def close(self, sess):
        sess.closed = True
        self.q.put(("close", sess))                           # slot freed on the GPU thread, after its last hop

    async def feed(self, sess, user_hop, agent_hop, mode, more=False):
        import asyncio
        loop = asyncio.get_running_loop(); fut = loop.create_future()
        self.q.put(("feed", sess, user_hop, agent_hop, mode, fut, loop, more))
        return await fut

    @staticmethod
    def _resolve(item, events=None, exc=None):
        fut, loop = item[5], item[6]

        def _set():
            if not fut.done():
                fut.set_exception(exc) if exc is not None else fut.set_result(events)
        loop.call_soon_threadsafe(_set)

    def _loop(self):
        while True:
            items = [self.q.get()]
            t0 = time.perf_counter()
            while True:
                with self.lock:
                    n_live = len(self.sessions)
                have = {it[1] for it in items if it[0] == "feed"}
                if len(have) >= n_live:
                    break
                waiting = any(s not in have and not s.closed for s in self.expect)
                rem = t0 + (self.expect_s if waiting else self.gather_s) - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    items.append(self.q.get(timeout=rem))
                except queue.Empty:
                    break
            self.expect = {it[1] for it in items if it[0] == "feed" and it[7] and not it[1].closed}
            feeds = [it for it in items if it[0] == "feed" and not it[1].closed]
            for it in items:
                if it[0] == "feed" and it[1].closed:
                    self._resolve(it, [])
            if feeds:
                self._tick(feeds)
            for it in items:
                if it[0] == "close":
                    with self.lock:
                        self.sessions.discard(it[1])
                        self.engine.head.reset_slot(it[1].slot); self.free.append(it[1].slot)

    def _tick(self, feeds):
        t0 = time.perf_counter()
        try:
            res = self.engine.process([(it[1], it[2], it[3], it[4]) for it in feeds])
        except Exception as e:
            log.exception("EARS tick error: %s", type(e).__name__)
            for it in feeds:
                self._resolve(it, exc=e)
            return
        self._ticks.append((time.perf_counter() - t0, len(feeds)))
        for it in feeds:
            events, finals = res[it[1]]
            if not finals:
                self._resolve(it, events); continue
            left = [len(finals)]

            def done(text, ev, it=it, events=events, left=left):
                segs = ev.pop("seg_texts", None) or []                 # a long turn's earlier blocks (I-05)
                ev["transcript"] = " ".join(t for t in [*segs, text] if t)
                left[0] -= 1
                if left[0] == 0:
                    self._resolve(it, events)
            for ev, wav in finals:
                self.engine.asr_q.put(Job("final", it[1], it[1].turn_gen, wav,
                                           done=lambda t, ev=ev, done=done: done(t, ev)))
        if time.time() - self._t_stats >= self.stats_every_s and self._ticks:
            ms = np.array([t for t, _ in self._ticks]) * 1e3; bs = np.array([b for _, b in self._ticks])
            e = self.engine; span = time.time() - self._t_stats
            log.info("EARS ticks: n=%d sessions/tick avg %.2f | tick p50 %.0f p90 %.0f max %.0f ms (hop 320 ms) | "
                     "busy: encoder %.0f%% head %.0f%% asr %.0f%% | GPU mem %.1f GiB (reserved %.1f)", len(ms),
                     bs.mean(), np.percentile(ms, 50), np.percentile(ms, 90), ms.max(), 100 * e.t_enc / span,
                     100 * e.t_head / span, 100 * e.t_asr / span, torch.cuda.memory_allocated() / 2**30,
                     torch.cuda.memory_reserved() / 2**30)
            self._ticks = []; self._t_stats = time.time(); e.t_enc = e.t_head = e.t_asr = 0.0
