"""Minimal 4-way turn-state decision (copied from codebase/exp-4/exp4/fdb_policy.py, verbatim _decide +
_softmax). Only `_decide` is used at live time, and for arm C it is LOGGED ONLY (the endpoint head fires
LISTEN->SPEAK). The full FDB eval harness (fdb_eval, VAD, Event) is not needed live, so it is not vendored."""
from __future__ import annotations
import numpy as np
from .model import LISTEN, HOLD, SPEAK, BACKCHANNEL


def _softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()


def _decide(state_logit_row, hazard, prev, theta, bc_thresh, onset="hazard", speak_thresh=0.5):
    p = _softmax(state_logit_row)
    # BACKCHANNEL fires on its own class, NEVER gated by theta
    if prev in (LISTEN, HOLD) and p[BACKCHANNEL] >= bc_thresh and p[BACKCHANNEL] == p.max():
        return BACKCHANNEL
    if prev in (LISTEN, HOLD):
        haz_fire = hazard >= theta
        state_fire = (p[SPEAK] >= speak_thresh) and (p[SPEAK] == p.max())
        fire = {"hazard": haz_fire, "state": state_fire, "both": haz_fire and state_fire}[onset]
        if fire:
            return SPEAK
        return HOLD if p[HOLD] > p[LISTEN] else LISTEN
    if prev == SPEAK:
        return SPEAK if p[SPEAK] >= max(p[LISTEN], p[HOLD]) else LISTEN
    if prev == BACKCHANNEL:
        return BACKCHANNEL if p[BACKCHANNEL] == p.max() else LISTEN
    return LISTEN
