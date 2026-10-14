"""The turn rules: what one conversation's audio and the heads' per-frame outputs mean for the turn. Plain bookkeeping
on an EarsSession (no GPU, no threads), shared by the engine and by the plain single-session reference that checks it
(scripts/validate/reference.py). Where the heads' context must start over, the functions say so and the caller resets
it (the engine its slot's GRU state, the reference its own).

  * listening: ENDPOINT when p_end >= ep_thr for ep_deb frames once the user has spoken MIN_SPEECH_MS (only the head
    ends a turn: there is no silence rule); BACKCHANNEL when the bc head fires for bc_deb frames (then
    bc_refractory frames of quiet); PARTIAL, once a hop, when the user's words so far (from the ASR thread) changed;
  * speaking: BARGE when the yield head fires for barge_deb frames, with `onset_s`: how long the user had been
    talking by then (their current stretch of speech, by frame energy), since the head takes a second or more to be
    sure, and EARS tells MOUTH what had played when they started;
  * every frame: a FRAME event with the state the page shows and the heads' probabilities.
"""
from __future__ import annotations

import numpy as np

from .constants import (BACKCHANNEL, BARGE_PRE_S, FRAME_S, HOLD, LISTEN, MIN_SPEECH_MS, ONSET_GAP_FRAMES,
                        ONSET_MAX_S, SPEAK, SR, STATE_NAMES, VAD_DB, WARMUP_FRAMES, WINDOW_S)


def frame_energy(wav):
    """Per-80 ms-frame energy in dB."""
    hop = int(round(FRAME_S * SR)); n = len(wav) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    seg = wav[:n * hop].reshape(n, hop).astype(np.float32)
    return (20 * np.log10(np.sqrt((seg ** 2).mean(1) + 1e-12) + 1e-9)).astype(np.float32)


def state_decision(state_logits, hazard, prev, theta, bc_thresh=0.5):
    """The 4-way state head's decision (exp-4's policy, onset by hazard). Live it only names the FRAME state shown on
    the page: the endpoint, yield and backchannel heads make the real decisions (arm C)."""
    p = np.exp(state_logits - state_logits.max()); p = p / p.sum()
    if prev in (LISTEN, HOLD) and p[BACKCHANNEL] >= bc_thresh and p[BACKCHANNEL] == p.max():
        return BACKCHANNEL
    if prev in (LISTEN, HOLD):
        if hazard >= theta:
            return SPEAK
        return HOLD if p[HOLD] > p[LISTEN] else LISTEN
    if prev == SPEAK:
        return SPEAK if p[SPEAK] >= max(p[LISTEN], p[HOLD]) else LISTEN
    if prev == BACKCHANNEL:
        return BACKCHANNEL if p[BACKCHANNEL] == p.max() else LISTEN
    return LISTEN


def begin_hop(sess, user_hop, agent_hop, mode) -> bool:
    """Before a hop's frames: apply a pending reset (a new turn; after a barge-in it starts with the user's speech so
    far, I-02), slide the 8 s windows, and add the hop to the turn's audio (listening) or to the barge-in seed buffer
    (speaking). True if the heads' context must start over."""
    reset = sess.pending_reset
    if reset:
        seed = sess.seed_u
        sess._clear(); sess.turn_gen += 1; sess.pending_reset = False
        if seed is not None:            # a barge-in: the user started talking over the agent; keep that start
            sess.turn_u = seed
            sess.voiced_ms = float((frame_energy(seed) > VAD_DB).sum()) * FRAME_S * 1e3
            sess.in_turn = sess.voiced_ms > 0
    sess.seed_u = None                  # only the reset right after the BARGE hop uses it
    sess.win_u = np.concatenate([sess.win_u, user_hop])[-int(WINDOW_S * SR):]
    sess.win_a = np.concatenate([sess.win_a, agent_hop])[-int(WINDOW_S * SR):]
    if mode == "listen":
        sess.turn_u = np.concatenate([sess.turn_u, user_hop])
    else:
        sess.pre_u = np.concatenate([sess.pre_u, user_hop])[-int(BARGE_PRE_S * SR):]
    return reset


def switch_mode(sess, mode) -> bool:
    """A change between listening and speaking: fresh decision runs, the previous state seeded from the new mode.
    True if it changed (the heads' context must start over)."""
    if mode == sess._mode:
        return False
    sess.warm = 0; sess.yield_run = 0; sess.bc_run = 0; sess.bc_cooldown = 0; sess.ep_run = 0
    sess.talk_frames = sess.quiet_frames = 0
    sess.prev_state = SPEAK if mode == "speaking" else LISTEN
    sess._mode = mode
    return True


def words_event(sess, mode):
    """The user's words so far in this turn, for the page: a PARTIAL event when they changed since the last one (only
    while listening), else None."""
    words = sess.words                  # set on the ASR thread
    if mode != "listen" or words == sess.words_sent:
        return None
    sess.words_sent = words
    return dict(kind="PARTIAL", text=words)


def track_talk(sess, voiced):
    """Speaking mode, every frame: how long the user has been talking. A stretch of speech starts at a voiced frame
    and lasts through pauses of up to ONSET_GAP_FRAMES."""
    if sess.talk_frames:
        sess.talk_frames += 1
        sess.quiet_frames = 0 if voiced else sess.quiet_frames + 1
        if sess.quiet_frames > ONSET_GAP_FRAMES:
            sess.talk_frames = sess.quiet_frames = 0
    elif voiced:
        sess.talk_frames, sess.quiet_frames = 1, 0


def on_frame(sess, mode, outputs, act_db, left=0):
    """One frame's decision from the heads' outputs (state logits x4, hazard, yield, bc, p_end) and the frame's
    energy; `left`: the hop's frames after this one (a barge's onset counts to the hop's end, the newest audio EARS
    has when it reads the event). Returns (events, final, reset): `final` is (the ENDPOINT event, the turn's audio)
    when the turn ended (the caller fills in its transcript), and `reset` says the heads' context must start over."""
    events = []
    slog = outputs[:4]; haz, yp, bcp, pep = (float(v) for v in outputs[4:8])
    if sess.trace is not None:
        sess.trace.append((mode, haz, yp, bcp, pep))
    st = state_decision(slog, haz, sess.prev_state, sess.theta)
    voiced = act_db > VAD_DB
    if mode == "speaking":                                 # BARGE path (yield head)
        sess.warm += 1
        track_talk(sess, voiced)
        if sess.warm > WARMUP_FRAMES:
            sess.yield_run = sess.yield_run + 1 if yp >= sess.barge_thr else 0
            if sess.yield_run >= sess.barge_deb:
                onset_s = min((sess.talk_frames + left) * FRAME_S, ONSET_MAX_S) if sess.talk_frames else 0.0
                events.append(dict(kind="BARGE", yield_prob=round(yp, 3), onset_s=round(onset_s, 2)))
                sess.yield_run = 0; sess.seed_u = sess.pre_u.copy()
        events.append(dict(kind="FRAME", state="SPEAK", yield_prob=round(yp, 3), hazard=round(haz, 3),
                           p_end=round(pep, 3)))
        sess.prev_state = SPEAK
        return events, None, False
    sess.warm += 1                                         # LISTEN mode
    if voiced:
        sess.voiced_ms += FRAME_S * 1e3; sess.in_turn = True
    if sess.bc_cooldown > 0:                               # BACKCHANNEL (floor-preserving)
        sess.bc_cooldown -= 1; sess.bc_run = 0
    elif sess.in_turn and sess.voiced_ms >= MIN_SPEECH_MS and sess.warm > WARMUP_FRAMES:
        sess.bc_run = sess.bc_run + 1 if bcp >= sess.bc_thr else 0
        if sess.bc_run >= sess.bc_deb:
            events.append(dict(kind="BACKCHANNEL", partial=sess.partial_asr, bc_prob=round(bcp, 3),
                               turn_s=round(sess.turn_len_s(), 2)))
            sess.bc_run = 0; sess.bc_cooldown = sess.bc_refractory
    if sess.warm > WARMUP_FRAMES:                          # ENDPOINT: p_end >= tau for deb frames
        sess.ep_run = sess.ep_run + 1 if pep >= sess.ep_thr else 0
    else:
        sess.ep_run = 0
    spoke = sess.in_turn and sess.voiced_ms >= MIN_SPEECH_MS
    if spoke and sess.ep_run >= sess.ep_deb:
        ev = dict(kind="ENDPOINT", transcript=None, turn_s=sess.turn_len_s(), segments=len(sess.segs),
                  seg_texts=sess.segs, by="head", p_end=round(pep, 3))
        final = (ev, sess.turn_u.copy())
        events.append(ev)
        sess.turn_u = np.zeros(0, np.float32); sess.voiced_ms = 0.0; sess.in_turn = False
        sess.seg_s = 0.0; sess.segs = []
        sess.prev_state = LISTEN; sess.warm = 0
        sess.bc_run = 0; sess.bc_cooldown = 0; sess.ep_run = 0
        sess.partial_asr = ""; sess.partial_at_s = 0.0; sess.turn_gen += 1
        sess.words = sess.words_sent = ""
        events.append(dict(kind="FRAME", state="idle", yield_prob=round(yp, 3), hazard=round(haz, 3),
                           bc_prob=round(bcp, 3), p_end=round(pep, 3)))
        return events, final, True
    disp = STATE_NAMES[st] if sess.in_turn else ("idle" if not voiced else "LISTEN")
    events.append(dict(kind="FRAME", state=disp, yield_prob=round(yp, 3), hazard=round(haz, 3),
                       bc_prob=round(bcp, 3), p_end=round(pep, 3)))
    sess.prev_state = LISTEN                               # free-run: the ep head's operating regime
    return events, None, False
