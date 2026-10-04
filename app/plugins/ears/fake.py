"""fake: an energy-threshold endpointer, no model (tests, CI, and running the stack without a GPU: config.fake.yaml).

Per 80 ms frame of user audio: voiced if its RMS level is above `vad_db`. A turn that has at least `min_speech_ms`
of voiced audio ends after `endpoint_silence_frames` silent frames: ENDPOINT with the fixed `transcript` from config
(there is no ASR). While the agent speaks, `barge_frames` voiced frames in a row are a BARGE. Never backchannels.
Events have the real plugin's shape (FRAME / ENDPOINT / BARGE), so server.py's state machine runs unchanged.
"""
from __future__ import annotations
import numpy as np

SR = 16000
FRAME_N = 1280                         # 80 ms
HOP_N = 4 * FRAME_N                    # 320 ms, as nemotron-tyger-1.0

THRESHOLDS = {"ep_thr": float, "ep_deb": int, "barge_thr": float, "barge_deb": int, "bc_thr": float, "bc_deb": int,
              "fallback_frames": int}


class FakeEarsSession:
    def __init__(self, slot, **th):
        self.slot = slot; self.th = th
        self._clear()

    def _clear(self):
        self.voiced_ms = 0.0; self.sil_run = 0; self.in_turn = False; self.voice_run = 0; self.turn_s = 0.0

    def reset(self):
        self._clear()


class EnergyEars:
    hop_samples = HOP_N
    sr = SR

    def __init__(self, conf: dict, ctx):
        self.transcript = str(conf.get("transcript", "hello there"))
        self.vad_db = float(conf.get("vad_db", -45.0))
        self.min_speech_ms = float(conf.get("min_speech_ms", 350.0))
        self.silence_frames = int(conf.get("endpoint_silence_frames", 6))
        self.barge_frames = int(conf.get("barge_frames", 4))
        self.slots = int(ctx.cfg["server"].get("max_sessions", 4)) if ctx else 4
        self.defaults = {"ep_thr": float(conf.get("ep_thr", 0.5)), "ep_deb": int(conf.get("ep_deb", 2)),
                         "barge_thr": float(conf.get("barge_thr", 0.5)), "barge_deb": int(conf.get("barge_deb", 2)),
                         "bc_thr": float(conf.get("bc_thr", 0.5)), "bc_deb": int(conf.get("bc_deb", 2)),
                         "fallback_frames": int(conf.get("fallback_sil_frames", self.silence_frames))}
        self.free = list(range(self.slots)); self.sessions = set()

    def open(self, **thresholds):
        if not self.free:
            return None
        s = FakeEarsSession(self.free.pop(0), **{**self.defaults, **thresholds}); self.sessions.add(s)
        return s

    def close(self, session):
        if session in self.sessions:
            self.sessions.discard(session); self.free.append(session.slot)

    def full(self):
        return not self.free

    def active(self):
        return len(self.sessions)

    def thresholds(self, session):
        return dict(session.th)

    def set_threshold(self, session, name, value):
        session.th[name] = THRESHOLDS[name](value)
        return session.th[name]

    async def feed(self, session, user_hop, agent_hop, mode, more=False):
        s = session; events = []
        u = np.asarray(user_hop, np.float32)
        for j in range(len(u) // FRAME_N):
            f = u[j * FRAME_N:(j + 1) * FRAME_N]
            voiced = 20 * np.log10(np.sqrt(float((f ** 2).mean())) + 1e-9) > self.vad_db
            if mode == "speaking":
                s.voice_run = s.voice_run + 1 if voiced else 0
                if s.voice_run >= self.barge_frames:
                    events.append(dict(kind="BARGE", yield_prob=1.0)); s.voice_run = 0
                events.append(dict(kind="FRAME", state="SPEAK", yield_prob=float(voiced), hazard=0.0, p_end=0.0))
                continue
            if voiced:
                s.voiced_ms += 80.0; s.sil_run = 0; s.in_turn = True
            else:
                s.sil_run += 1
            if s.in_turn:
                s.turn_s += 0.08
            if s.in_turn and s.voiced_ms >= self.min_speech_ms and s.sil_run >= s.th["fallback_frames"]:
                events.append(dict(kind="ENDPOINT", transcript=self.transcript, turn_s=round(s.turn_s, 2),
                                   segments=0, by="energy", p_end=1.0))
                s._clear()
                events.append(dict(kind="FRAME", state="idle", yield_prob=0.0, hazard=0.0, bc_prob=0.0, p_end=1.0))
                continue
            events.append(dict(kind="FRAME", state="LISTEN" if s.in_turn or voiced else "idle", yield_prob=0.0, hazard=0.0, bc_prob=0.0,
                               p_end=round(min(1.0, s.sil_run / max(1, s.th["fallback_frames"])), 3)))
        return events
