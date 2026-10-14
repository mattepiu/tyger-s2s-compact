"""MOUTH's concurrency helpers. MOUTH is synchronous: plugins yield plain generators, and work that must run while a
turn speaks (finishing a tool call, a filler line, the smart model's answer) runs on a daemon thread and is read
back through a queue. A stop (barge-in) ends the reading side; the threads see the same stop flag and wind down.
"""
from __future__ import annotations
import logging, queue, threading, time

log = logging.getLogger("mouth")
END = object()                      # end-of-stream marker on the queues


def text_of(events):
    """The text of a finished LLM event stream."""
    return "".join(e[1] for e in events if e[0] == "text")


def spawn(gen_fn, what="filler"):
    """Run the generator gen_fn() on a daemon thread; returns a queue of its items, then END. An exception ends
    the stream early and is logged by type only (`what` names the stream)."""
    q: queue.Queue = queue.Queue()

    def run():
        try:
            for x in gen_fn():
                q.put(x)
        except Exception as e:
            log.warning("%s error: %s", what, type(e).__name__)
        finally:
            q.put(END)
    threading.Thread(target=run, daemon=True).start()
    return q


class Prefetch:
    """An LLM stream running on a helper thread from now on, so it is under way while this one speaks; iterate it
    here, in order. `ready` is set at its first event (or when it ends), `t_first` is when that event came. An
    exception in the stream is raised to the reader; a stop ends the iteration."""
    def __init__(self, gen_fn, stop_flag):
        self.q: queue.Queue = queue.Queue(); self.ready = threading.Event(); self.stop_flag = stop_flag
        self.t_first = None
        threading.Thread(target=self._run, args=(gen_fn,), daemon=True).start()

    def _run(self, gen_fn):
        try:
            for x in gen_fn():
                if self.t_first is None:
                    self.t_first = time.perf_counter()
                self.q.put(x); self.ready.set()
        except Exception as e:
            self.q.put(e)
        finally:
            self.q.put(END); self.ready.set()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                x = self.q.get(timeout=0.01)
            except queue.Empty:
                if self.stop_flag():
                    raise StopIteration
                continue
            if x is END:
                self.q.put(END)                        # stays ended
                raise StopIteration
            if isinstance(x, Exception):
                raise x
            return x
