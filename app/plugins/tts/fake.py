"""fake: a scripted TTS, no model (tests, CI, and running the stack without a GPU: config.fake.yaml).

Each request yields `n` chunks of `chunk_s` audio, `step_s` apart (after `onset_s`), polling stop_flag between
chunks like the real engine loop. With `per_char_s` the length follows the text instead (n = text length x
per_char_s / chunk_s), and with `tone_hz` the audio is a quiet sine tone you can hear in the browser instead of a
constant level. Every request's text is recorded in `requests` for tests; nothing is logged.
"""
from __future__ import annotations
import math, threading

import numpy as np

from app.plugins.base import TTSCaps
from app.plugins.llm.fake import wait_or_stop

DEFAULTS = {"out_sr": 24000, "n": 5, "chunk_s": 0.08, "step_s": 0.002, "onset_s": 0.0, "per_char_s": 0.0,
            "tone_hz": 0.0, "level": 0.1}


class FakeTTS:
    remote = False

    def __init__(self, conf: dict | None = None, ctx=None, **kw):
        o = {**DEFAULTS, **{k: v for k, v in (conf or {}).items() if k in DEFAULTS}, **kw}
        self.out_sr = int(o["out_sr"]); self.n = int(o["n"]); self.chunk_s = float(o["chunk_s"])
        self.step_s = float(o["step_s"]); self.onset_s = float(o["onset_s"]); self.per_char_s = float(o["per_char_s"])
        self.tone_hz = float(o["tone_hz"]); self.level = float(o["level"])
        self.caps = TTSCaps(cloning=False, max_batch=4)
        self.requests = []; self.aborted = 0; self.live = 0; self.lock = threading.Lock()

    @classmethod
    def list_voices(cls, cfg=None):
        return ["fake"], "fake"

    def load_voice(self, name):
        return None

    def warmup(self, voice=None):
        pass

    def shutdown(self):
        pass

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
