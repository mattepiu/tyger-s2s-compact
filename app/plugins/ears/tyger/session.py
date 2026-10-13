"""EarsSession: one conversation's turn-taking state (its audio windows, the turn being heard, the decision runs) and
its thresholds. The turn rules and the engine change it on the GPU thread; reset() may come from any thread.
"""
from __future__ import annotations

import numpy as np

from .constants import LISTEN, SR


class EarsSession:
    """One conversation's state for the turn rules (rules.py) and its thresholds (the module docstring)."""
    def __init__(self, slot, ep_thr=0.9146, ep_deb=2, barge_thr=0.5, barge_deb=2, bc_thr=0.5, bc_deb=2,
                 theta=0.2, bc_refractory=15, partial_refresh_s=0.8):
        self.slot = slot
        self.ep_thr = ep_thr; self.ep_deb = ep_deb; self.barge_thr = barge_thr; self.barge_deb = barge_deb
        self.bc_thr = bc_thr; self.bc_deb = bc_deb
        self.theta = theta; self.bc_refractory = bc_refractory; self.partial_refresh_s = partial_refresh_s
        self.closed = False; self.turn_gen = 0; self.partial_inflight = False
        self.trace = None                      # set to a list to record per-frame head outputs (tests only)
        self._clear()
        self.pending_reset = True              # the slot's GRU state is cleared on the first hop

    def _clear(self):
        self.win_u = np.zeros(0, np.float32); self.win_a = np.zeros(0, np.float32)
        self.turn_u = np.zeros(0, np.float32)  # this turn's user audio not yet sent for transcription
        self.seg_s = 0.0                       # seconds of this turn already cut into segments
        self.segs = []                         # their texts, filled in by the ASR thread (None until done)
        self.pre_u = np.zeros(0, np.float32)   # last BARGE_PRE_S of user audio in speaking mode
        self.seed_u = None                     # set on BARGE: the next reset starts turn_u with it
        self.talk_frames = 0                   # speaking mode: frames since the user's current stretch of speech began
        self.quiet_frames = 0                  #   (0: none), and its trailing unvoiced frames
        self.prev_state = LISTEN; self.yield_run = 0
        self.voiced_ms = 0.0; self.in_turn = False; self.warm = 0
        self._mode = "listen"; self.bc_run = 0; self.bc_cooldown = 0; self.ep_run = 0
        self.partial_asr = ""; self.partial_at_s = 0.0
        self.words = ""; self.words_sent = ""  # the turn's words so far (blocks, then the partial), for the page

    def reset(self):
        """Thread-safe: applied by the GPU thread before this session's next hop."""
        self.pending_reset = True


    def turn_len_s(self):
        return self.seg_s + len(self.turn_u) / SR
