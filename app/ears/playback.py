"""The agent's reply audio as the browser plays it. EARS forwards MOUTH's audio as it arrives, and the TTS runs
several times faster than real time, so most of a long reply is still queued in the browser when MOUTH says it is
done. AgentPlayback estimates when the browser will have played everything (play_until), how much it has played
(for a barge-in: MOUTH keeps what was heard in history), and keeps the same audio at the EARS rate for the turn-taking
model's 2nd ear, handed out one hop at a time.
"""
from __future__ import annotations
import time

import numpy as np

from app.plugins.tts import WIRE_SR

LEAD_IN_S = 0.04                 # the page starts a chunk this long after it arrives when nothing is queued


def to_ears_rate(a24, ears_sr):
    """Reply audio (WIRE_SR) -> the EARS rate, for the 2nd ear."""
    n = int(len(a24) * ears_sr / WIRE_SR)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(a24) - 1, n), np.arange(len(a24)), a24).astype(np.float32)


class AgentPlayback:
    def __init__(self, ears_sr):
        self.ears_sr = ears_sr
        self.play_until = 0.0          # perf_counter time at which the browser will have played all of it
        self.sent_s = 0.0              # reply audio sent to the browser this turn, in s
        self.ear = np.zeros(0, np.float32)

    def start(self):
        """A new reply starts: nothing sent or queued yet."""
        self.play_until = 0.0; self.sent_s = 0.0

    def add(self, a24):
        """One chunk of reply audio (float32 at WIRE_SR) went to the browser, which queues chunks back to back."""
        d = len(a24) / WIRE_SR
        self.play_until = max(self.play_until, time.perf_counter() + LEAD_IN_S) + d
        self.sent_s += d
        self.ear = np.concatenate([self.ear, to_ears_rate(a24, self.ears_sr)])

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
        self.ear = np.zeros(0, np.float32)
