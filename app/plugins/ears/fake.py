"""fake: an energy-threshold endpointer, no model (tests, CI, and running the stack without a GPU: config/fake.yaml).

Per 80 ms frame of user audio: voiced if its RMS level is above `vad_db`. A turn that has at least `min_speech_ms`
of voiced audio ends after `endpoint_silence_frames` silent frames: ENDPOINT with the fixed `transcript` from config
(there is no ASR). While the agent speaks, `barge_frames` voiced frames in a row are a BARGE. Never backchannels.
Events have the real plugin's shape (FRAME / ENDPOINT / BARGE), so EARS's Conversation state machine runs unchanged.
Its one tunable threshold is the endpoint silence.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from app.plugins.base import EarsPlugin, Slider, Threshold
from app.plugins.options import PluginOptions

SR = 16000
FRAME_N = 1280                         # 80 ms
HOP_N = 4 * FRAME_N                    # 320 ms, as nemotron-tyger-1.0

THRESHOLDS = (
    Threshold("endpoint_silence_frames", int, default=6,
              slider=Slider("endpoint silence", 0.16, 2.0, 0.08, scale=FRAME_N / SR, unit="s")),
)


class FakeEarsSession:
    def __init__(self, slot, endpoint_silence_frames):
        self.slot = slot; self.endpoint_silence_frames = endpoint_silence_frames
        self._clear()

    def _clear(self):
        self.voiced_ms = 0.0; self.sil_run = 0; self.in_turn = False; self.voice_run = 0; self.turn_s = 0.0

    def reset(self):
        self._clear()


class EnergyEars(EarsPlugin):
    hop_samples = HOP_N
    sr = SR
    THRESHOLDS = THRESHOLDS

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        transcript: str = "hello there"    # every ENDPOINT's (there is no ASR)
        vad_db: float = -45.0
        min_speech_ms: float = 350.0
        barge_frames: int = 4

    def __init__(self, conf: dict | None = None, ctx=None):
        super().__init__(conf, ctx)
        self.slots = int(ctx.cfg["server"].get("max_sessions", 4)) if ctx else 4
        self.free = list(range(self.slots)); self.sessions = set()

    def open_session(self, thresholds):
        if not self.free:
            return None
        s = FakeEarsSession(self.free.pop(0), **thresholds); self.sessions.add(s)
        return s

    def close(self, session):
        if session in self.sessions:
            self.sessions.discard(session); self.free.append(session.slot)

    def full(self):
        return not self.free

    def active(self):
        return len(self.sessions)

    async def feed(self, session, user_hop, agent_hop, mode, more=False):
        s = session; o = self.opts; events = []
        u = np.asarray(user_hop, np.float32)
        for j in range(len(u) // FRAME_N):
            f = u[j * FRAME_N:(j + 1) * FRAME_N]
            voiced = 20 * np.log10(np.sqrt(float((f ** 2).mean())) + 1e-9) > o.vad_db
            if mode == "speaking":
                s.voice_run = s.voice_run + 1 if voiced else 0
                if s.voice_run >= o.barge_frames:
                    events.append(dict(kind="BARGE", yield_prob=1.0)); s.voice_run = 0
                events.append(dict(kind="FRAME", state="SPEAK", yield_prob=float(voiced), hazard=0.0, p_end=0.0))
                continue
            if voiced:
                s.voiced_ms += 80.0; s.sil_run = 0; s.in_turn = True
            else:
                s.sil_run += 1
            if s.in_turn:
                s.turn_s += 0.08
            if s.in_turn and s.voiced_ms >= o.min_speech_ms and s.sil_run >= s.endpoint_silence_frames:
                events.append(dict(kind="ENDPOINT", transcript=o.transcript, turn_s=round(s.turn_s, 2),
                                   segments=0, by="energy", p_end=1.0))
                s._clear()
                events.append(dict(kind="FRAME", state="idle", yield_prob=0.0, hazard=0.0, bc_prob=0.0, p_end=1.0))
                continue
            events.append(dict(kind="FRAME", state="LISTEN" if s.in_turn or voiced else "idle", yield_prob=0.0,
                               hazard=0.0, bc_prob=0.0,
                               p_end=round(min(1.0, s.sil_run / max(1, s.endpoint_silence_frames)), 3)))
        return events
