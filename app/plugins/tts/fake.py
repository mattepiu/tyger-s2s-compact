"""fake: a scripted TTS, no model (tests, CI, and running the stack without a GPU: config/fake.yaml).

Each request yields `n` chunks of `chunk_s` audio, `step_s` apart (after `onset_s`), polling stop_flag between
chunks like the real engine loop. With `per_char_s` the length follows the text instead (n = text length x
per_char_s / chunk_s), and with `tone_hz` the audio is a quiet sine tone you can hear in the browser instead of a
constant level. Every request's text is recorded in `requests` for tests; nothing is logged.
"""
from __future__ import annotations
import math, threading
from dataclasses import dataclass

import numpy as np

from app.plugins.base import TTSCaps, TTSPlugin
from app.plugins.llm.fake import wait_or_stop
from app.plugins.options import PluginOptions


class FakeTTS(TTSPlugin):
    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        out_sr: int = 24000
        n: int = 5                         # chunks per request (unless per_char_s is set)
        chunk_s: float = 0.08
        step_s: float = 0.002              # between chunks
        onset_s: float = 0.0               # before the first
        per_char_s: float = 0.0            # > 0: audio length follows the text
        tone_hz: float = 0.0               # > 0: a sine tone instead of a constant level
        level: float = 0.1

    def __init__(self, conf: dict | None = None, ctx=None, **kw):
        """conf: the config block; keyword arguments override it (tests)."""
        super().__init__({**(conf or {}), **kw}, ctx)
        o = self.opts
        self.out_sr = o.out_sr; self.n = o.n; self.chunk_s = o.chunk_s
        self.step_s = o.step_s; self.onset_s = o.onset_s; self.per_char_s = o.per_char_s
        self.tone_hz = o.tone_hz; self.level = o.level
        self.caps = TTSCaps(cloning=False, max_batch=4)
        self.requests = []; self.aborted = 0; self.live = 0; self.lock = threading.Lock()

    @classmethod
    def list_voices(cls, cfg=None):
        return ["fake"], "fake"

    def load_voice(self, name):
        return None

    def _chunk(self, i):
        m = int(self.chunk_s * self.out_sr)
        if not self.tone_hz:
            return np.full(m, self.level, np.float32)
        t = (np.arange(m) + i * m) / self.out_sr
        return (self.level * np.sin(2 * math.pi * self.tone_hz * t)).astype(np.float32)

    def stream(self, text, voice=None, stop_flag=None):
        self.requests.append(text)
        n = max(1, math.ceil(len(text) * self.per_char_s / self.chunk_s)) if self.per_char_s else self.n
        with self.lock:
            self.live += 1
        try:
            if wait_or_stop(self.onset_s, stop_flag):
                with self.lock:
                    self.aborted += 1
                return
            for i in range(n):
                if wait_or_stop(self.step_s, stop_flag):
                    with self.lock:
                        self.aborted += 1
                    return
                yield self._chunk(i)
        finally:
            with self.lock:
                self.live -= 1
