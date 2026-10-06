"""EarsEngine: one batched tick for every conversation with a hop ready, and the per-frame decisions.

Per tick: each conversation's (user, agent) 8 s windows go through nemotron-3.5-asr to layer 15 together (NemotronAsr,
equal-length windows batched), then the tyger heads step frame by frame for all of them at once (HeadBatch), and each
frame's head outputs become events by the turn rules in _frame (a line-by-line port of the research reference,
scripts/reference/ears_ref/endpoint_ears.py, which scripts/validate_ears_batched.py checks it against):
  * listening: ENDPOINT when p_end >= ep_thr for ep_deb frames (or fallback_frames of silence) after at least
    MIN_SPEECH_MS of speech; BACKCHANNEL when the bc head fires (then a refractory period); FRAME every frame;
  * speaking: BARGE when the yield head fires for barge_deb frames.
Transcription runs on the ASR thread (Job queue): an ENDPOINT's hop resolves when its transcript is ready; long turns
are transcribed in segments as they grow (I-05); the rolling partial for backchannels lands asynchronously.
Privacy: audio and transcripts stay in memory; logs carry counts and timings only.
"""
from __future__ import annotations
import logging, queue, threading, time

import numpy as np, torch

from .asr import NemotronAsr
from .constants import (BACKCHANNEL, BARGE_PRE_S, FRAME_S, HOLD, HOP_FRAMES, LISTEN, MIN_SPEECH_MS, SEG_S,
                        SEG_SEARCH_S, SPEAK, SR, STATE_NAMES, TURN_MAX_S, VAD_DB, WARMUP_FRAMES, WINDOW_S)
from .head_batch import HeadBatch
from .heads import TygerHeads

log = logging.getLogger("ears")


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


class Job:
    """A transcription for the ASR thread. kind: "final" (an endpoint's transcript), "segment" (a finished block of
    a long turn), "partial" (the rolling partial for backchannels); gen: the session's turn it belongs to."""
    __slots__ = ("kind", "sess", "gen", "wav", "done")

    def __init__(self, kind, sess, gen, wav, done=None):
        self.kind = kind; self.sess = sess; self.gen = gen; self.wav = wav; self.done = done


class EarsEngine:
    def __init__(self, heads_dir, slots=4, ring_len=2048, device="cuda", cuda_graph=True, nemotron=None,
                 nemotron_revision=None):
        """heads_dir: a TYGER-S2S-endpointer release (weights.heads_release). The encoder is the one the release
        pins (its config's frontend), unless `nemotron` names another checkpoint."""
        self.device = device; self.slots = slots
        self.heads = TygerHeads.from_pretrained(heads_dir, device=device)
        fe = self.heads.frontend
        model_id, revision = (nemotron, nemotron_revision) if nemotron else (fe["repo_id"], fe["revision"])
        self.asr = NemotronAsr(model_id, revision, device=device, max_batch=slots, cuda_graph=cuda_graph)
        self.head = HeadBatch(self.heads, slots, ring_len, device, cuda_graph=cuda_graph)
        log.info("EARS engine: %d slots, ring %d frames, heads %s, encoder %s@%s, head CUDA graph=%s", slots,
                 ring_len, self.heads.config.get("version"), model_id, (revision or "latest")[:8],
                 self.head.graph is not None)
        self.asr_q: queue.Queue = queue.Queue()
        self.t_enc = self.t_head = self.t_asr = 0.0          # busy seconds, reset by the stats log
        threading.Thread(target=self._asr_loop, name="ears-asr", daemon=True).start()

    # ---- the ASR thread ----
    def _asr_loop(self):
        while True:
            jobs = [self.asr_q.get()]
            while True:
                try:
                    jobs.append(self.asr_q.get_nowait())
                except queue.Empty:
                    break
            t0 = time.perf_counter()
            # enqueue order: a turn's segments are done before its final; finals first, an endpoint waits on them
            self._finish_finals([j for j in jobs if j.kind in ("final", "segment")])
            self._finish_partials([j for j in jobs if j.kind == "partial"])
            self.t_asr += time.perf_counter() - t0

    def _transcribe_or_none(self, jobs, what):
        """Their texts, or None if transcription failed (logged by kind only)."""
        try:
            return self.asr.transcribe([j.wav for j in jobs])
        except Exception as e:
            log.exception("ASR error (%s): %s", what, type(e).__name__)
            return None

    def _finish_finals(self, jobs):
        """Endpoint transcripts and long-turn segments: each job's done() runs exactly once, with "" if
        transcription failed, so the waiting hop always resolves."""
        if not jobs:
            return
        texts = self._transcribe_or_none(jobs, "finals") or [""] * len(jobs)
        for j, t in zip(jobs, texts):
            j.done(t)

    def _finish_partials(self, jobs):
        """Backchannel partials: best effort. A failure keeps the previous partial; either way the session may ask
        for the next one. Partials from a turn that already ended are dropped."""
        if not jobs:
            return
        texts = self._transcribe_or_none(jobs, "partials") or [None] * len(jobs)
        for j, t in zip(jobs, texts):
            if t is not None and j.sess.turn_gen == j.gen:
                j.sess.partial_asr = t
            j.sess.partial_inflight = False

    # ---- one tick ----
    def process(self, items):
        """items: [(sess, user_hop, agent_hop, mode)] (one hop per session). Returns {sess: (events, finals)}
        where finals are (event, user_audio) pairs still needing an endpoint transcript."""
        res = {}
        for sess, uh, ah, mode in items:
            if sess.pending_reset:
                seed = sess.seed_u
                sess._clear(); sess.turn_gen += 1; self.head.reset_slot(sess.slot); sess.pending_reset = False
                if seed is not None:        # a barge-in: the user started talking over the agent; keep that start
                    sess.turn_u = seed
                    sess.voiced_ms = float((frame_energy(seed) > VAD_DB).sum()) * FRAME_S * 1e3
                    sess.in_turn = sess.voiced_ms > 0
            sess.seed_u = None              # only the reset right after the BARGE hop uses it
            sess.win_u = np.concatenate([sess.win_u, uh])[-int(WINDOW_S * SR):]
            sess.win_a = np.concatenate([sess.win_a, ah])[-int(WINDOW_S * SR):]
            if mode == "listen":
                sess.turn_u = np.concatenate([sess.turn_u, uh])
                if len(sess.turn_u) > SEG_S * SR:
                    self._cut_segment(sess)
                sess.turn_u = sess.turn_u[-int(TURN_MAX_S * SR):]
            else:
                sess.pre_u = np.concatenate([sess.pre_u, uh])[-int(BARGE_PRE_S * SR):]
            res[sess] = ([], [])
        t0 = time.perf_counter()
        hs = self.asr.encode([w for sess, *_ in items for w in (sess.win_u, sess.win_a)])
        torch.cuda.current_stream().synchronize(); t1 = time.perf_counter()
        plan = []                                              # (sess, mode, hu, ha, act, n)
        for k, (sess, uh, ah, mode) in enumerate(items):
            hu, ha = hs[2 * k], hs[2 * k + 1]
            act = frame_energy(sess.win_u)
            if mode == "listen":
                self._maybe_partial(sess)
            if mode != sess._mode:                             # mode switch: fresh context + seed
                self.head.reset_slot(sess.slot); sess.warm = 0; sess.yield_run = 0
                sess.bc_run = 0; sess.bc_cooldown = 0; sess.ep_run = 0
                sess.prev_state = SPEAK if mode == "speaking" else LISTEN
                sess._mode = mode
            n = min(HOP_FRAMES, hu.shape[0], ha.shape[0], len(act))
            plan.append((sess, mode, hu, ha, act, n))
        for j in range(HOP_FRAMES):                            # frame j of the hop, all sessions at once
            live = [p for p in plan if j >= HOP_FRAMES - p[5]]
            if not live:
                continue
            i = j - HOP_FRAMES                                 # newest-n frames, as feed() did: -n..-1
            out = self.head.step([(p[0].slot, p[2][i], p[3][i], float(p[4][i]), p[0].prev_state) for p in live])
            for sess, mode, hu, ha, act, n in live:
                self._frame(sess, mode, out[sess.slot], float(act[i]), res[sess])
        self.t_enc += t1 - t0; self.t_head += time.perf_counter() - t1
        return res

    def _cut_segment(self, sess):
        """Send the turn's audio so far, up to a pause, for transcription now (on the ASR thread); keep the rest."""
        hop = int(FRAME_S * SR)
        lo = int((SEG_S - SEG_SEARCH_S) * SR) // hop
        e = frame_energy(sess.turn_u[lo * hop:int(SEG_S * SR)])
        cut = (lo + int(np.argmin(e)) + 1) * hop if e.size else int(SEG_S * SR)
        seg, sess.turn_u = sess.turn_u[:cut], sess.turn_u[cut:]
        segs, i = sess.segs, len(sess.segs)
        segs.append(None); sess.seg_s += cut / SR

        def done(text, segs=segs, i=i):
            segs[i] = text
        self.asr_q.put(Job("segment", sess, sess.turn_gen, seg, done=done))

    def _maybe_partial(self, sess):
        """Ask for a fresh partial (for backchannels) at most every partial_refresh_s, decoded on the ASR thread."""
        cur_s = sess.turn_len_s()
        if sess.partial_inflight or cur_s - sess.partial_at_s < sess.partial_refresh_s:
            return
        if sess.win_u.size < int(0.3 * SR):
            return
        sess.partial_at_s = cur_s; sess.partial_inflight = True
        self.asr_q.put(Job("partial", sess, sess.turn_gen, sess.win_u.copy()))

    def _frame(self, sess, mode, vals, act_db, res):
        """One frame's decision (the research reference's per-frame body, verbatim in behaviour)."""
        events, finals = res
        slog = vals[:4]; haz, yp, bcp, pep = (float(v) for v in vals[4:8])
        if sess.trace is not None:
            sess.trace.append((mode, haz, yp, bcp, pep))
        st = state_decision(slog, haz, sess.prev_state, sess.theta)
        voiced = act_db > VAD_DB
        if mode == "speaking":                                 # BARGE path (yield head)
            sess.warm += 1
            if sess.warm > WARMUP_FRAMES:
                sess.yield_run = sess.yield_run + 1 if yp >= sess.barge_thr else 0
                if sess.yield_run >= sess.barge_deb:
                    events.append(dict(kind="BARGE", yield_prob=round(yp, 3)))
                    sess.yield_run = 0; sess.seed_u = sess.pre_u.copy()
            events.append(dict(kind="FRAME", state="SPEAK", yield_prob=round(yp, 3), hazard=round(haz, 3),
                               p_end=round(pep, 3)))
            sess.prev_state = SPEAK
            return
        sess.warm += 1                                         # LISTEN mode
        if voiced:
            sess.voiced_ms += FRAME_S * 1e3; sess.sil_run = 0; sess.in_turn = True
        else:
            sess.sil_run += 1
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
        learned_end = sess.ep_run >= sess.ep_deb
        fallback_end = sess.sil_run >= sess.fallback_frames
        if spoke and (learned_end or fallback_end):
            ev = dict(kind="ENDPOINT", transcript=None, turn_s=sess.turn_len_s(), segments=len(sess.segs),
                      seg_texts=sess.segs,
                      by="head" if learned_end else "silence", p_end=round(pep, 3))
            events.append(ev); finals.append((ev, sess.turn_u.copy()))
            sess.turn_u = np.zeros(0, np.float32); sess.voiced_ms = 0.0; sess.in_turn = False
            sess.seg_s = 0.0; sess.segs = []
            self.head.reset_slot(sess.slot); sess.prev_state = LISTEN; sess.warm = 0
            sess.bc_run = 0; sess.bc_cooldown = 0; sess.ep_run = 0
            sess.partial_asr = ""; sess.partial_at_s = 0.0; sess.turn_gen += 1
            events.append(dict(kind="FRAME", state="idle", yield_prob=round(yp, 3), hazard=round(haz, 3),
                               bc_prob=round(bcp, 3), p_end=round(pep, 3)))
            return
        disp = STATE_NAMES[st] if sess.in_turn else ("idle" if not voiced else "LISTEN")
        events.append(dict(kind="FRAME", state=disp, yield_prob=round(yp, 3), hazard=round(haz, 3),
                           bc_prob=round(bcp, 3), p_end=round(pep, 3)))
        sess.prev_state = LISTEN                               # free-run: the ep head's operating regime
