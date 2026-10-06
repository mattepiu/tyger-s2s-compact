"""The agent's reply audio as the browser plays it. EARS forwards MOUTH's audio as it arrives, and the TTS runs
several times faster than real time, so most of a long reply is still queued in the browser when MOUTH says it is
done. AgentPlayback estimates when the browser will have played everything (play_until), how much it has played
(for a barge-in: MOUTH keeps what was heard in history), and keeps the same audio at the EARS rate for the turn-taking
model's 2nd ear, handed out one hop at a time. The 2nd ear's audio goes through one stateful polyphase resampler per
reply (app/runtime/audio.py), so it doesn't depend on where MOUTH's chunks happen to end (scripts/validate/second_ear.py).
"""
from __future__ import annotations
import time

import numpy as np

from app.plugins.tts import WIRE_SR
from app.runtime.audio import StreamingAudioResampler

LEAD_IN_S = 0.04                 # the page starts a chunk this long after it arrives when nothing is queued


class AgentPlayback:
    def __init__(self, ears_sr):
        self.ears_sr = ears_sr
        self.play_until = 0.0          # perf_counter time at which the browser will have played all of it
        self.sent_s = 0.0              # reply audio sent to the browser this turn, in s
        self.ear = np.zeros(0, np.float32)
        self.to_ears_rate = StreamingAudioResampler(WIRE_SR, ears_sr)

    def start(self):
        """A new reply starts: nothing sent or queued yet."""
        self.play_until = 0.0; self.sent_s = 0.0
        self.to_ears_rate.reset()

    def add(self, a24):
        """One chunk of reply audio (float32 at WIRE_SR) went to the browser, which queues chunks back to back."""
        d = len(a24) / WIRE_SR
        self.play_until = max(self.play_until, time.perf_counter() + LEAD_IN_S) + d
        self.sent_s += d
        self.ear = np.concatenate([self.ear, self.to_ears_rate.process(a24)])

    def played_s(self):
        """Reply audio the browser has played so far, in s."""
        return max(0.0, self.sent_s - max(0.0, self.play_until - time.perf_counter()))

    def take_hop(self, n):
        """The next `n` samples of the 2nd ear (zeros past what has arrived)."""
        hop = np.zeros(n, np.float32)
        k = min(n, len(self.ear))
        hop[:k] = self.ear[:k]; self.ear = self.ear[k:]
        return hop

    def clear_ear(self):
        """The reply stopped (played out or barged): drop the 2nd ear's audio and the resampler's history."""
        self.ear = np.zeros(0, np.float32)
        self.to_ears_rate.reset()
