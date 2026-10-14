"""EarsEngine: one batched tick for every conversation with a hop ready.

Per tick: each conversation's (user, agent) 8 s windows go through nemotron-3.5-asr to layer 15 together (NemotronAsr,
equal-length windows batched), then the turn-taker steps frame by frame for all of them at once (GruBatch: a row of
GRU state per conversation), and each frame's outputs become events by the turn rules (rules.py). Transcription runs
on the ASR thread (Job queue): an ENDPOINT's hop resolves when its transcript is ready; long turns are transcribed in
segments as they grow (I-05); the rolling partial of the turn (for backchannels, and the user's words so far for the
page) lands asynchronously.
scripts/validate/ears_batched.py checks the engine against a plain single-session run of the same bundle
(scripts/validate/reference.py: CPU features, the full encoder, the GRU one conversation at a time, HF generate).
Privacy: audio and transcripts stay in memory; logs carry counts and timings only.
"""
from __future__ import annotations
import logging, queue, threading, time

import numpy as np, torch

from . import rules
from .asr import NemotronAsr
from .bundle import load
from .constants import FRAME_S, HOP_FRAMES, SEG_S, SEG_SEARCH_S, SR, TURN_MAX_S
from .gru_batch import GruBatch

log = logging.getLogger("ears")


class Job:
    """A transcription for the ASR thread. kind: "final" (an endpoint's transcript), "segment" (a finished block of
    a long turn), "partial" (the rolling partial of the turn); gen: the session's turn it belongs to; segs (a
    partial): how many of the turn's blocks were cut before its audio."""
    __slots__ = ("kind", "sess", "gen", "wav", "done", "segs")

    def __init__(self, kind, sess, gen, wav, done=None, segs=0):
        self.kind = kind; self.sess = sess; self.gen = gen; self.wav = wav; self.done = done; self.segs = segs


class EarsEngine:
    def __init__(self, bundle_dir, slots=4, device="cuda", cuda_graph=True):
        """bundle_dir: a tyger bundle (bundle.fetch). cuda_graph: the RNN-T decoder's (the GRU runs eager)."""
        nemotron, proc, turn_taker, config = load(bundle_dir)
        self.asr = NemotronAsr(nemotron, proc, device=device, max_batch=slots, cuda_graph=cuda_graph)
        self.head = GruBatch(turn_taker, slots, device)     # step(entries) -> (slots, 8), reset_slot(slot)
        self.slots = slots; self.device = device
        log.info("EARS engine: %d slots, tyger bundle %s (nemotron %s, GRU %dx%d), RNN-T CUDA graph=%s", slots,
                 config["weights_sha256"][:8], config["frontend"]["revision"][:8], turn_taker.core.num_layers,
                 turn_taker.core.hidden_size, self.asr.rnnt.graph is not None)
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
        """Partials: best effort. A failure keeps the previous partial; either way the session may ask for the next
        one. Partials from a turn that already ended are dropped. The user's words so far are the blocks cut before
        the partial's audio (their jobs were queued first, so they're done), then the partial."""
        if not jobs:
            return
        texts = self._transcribe_or_none(jobs, "partials") or [None] * len(jobs)
        for j, t in zip(jobs, texts):
            if t is not None and j.sess.turn_gen == j.gen:
                j.sess.partial_asr = t
                j.sess.words = " ".join(x for x in [*j.sess.segs[:j.segs], t] if x)
            j.sess.partial_inflight = False

    # ---- one tick ----
    def process(self, items):
        """items: [(sess, user_hop, agent_hop, mode)] (one hop per session). Returns {sess: (events, finals)}
        where finals are (event, user_audio) pairs still needing an endpoint transcript."""
        res = {}
        for sess, uh, ah, mode in items:
            if rules.begin_hop(sess, uh, ah, mode):
                self.head.reset_slot(sess.slot)
            if mode == "listen":                               # long turns: transcribe a block now (I-05)
                if len(sess.turn_u) > SEG_S * SR:
                    self._cut_segment(sess)
                sess.turn_u = sess.turn_u[-int(TURN_MAX_S * SR):]
            words = rules.words_event(sess, mode)              # the partial that landed since the last hop
            res[sess] = ([words] if words else [], [])
        t0 = time.perf_counter()
        hs = self.asr.encode([w for sess, *_ in items for w in (sess.win_u, sess.win_a)])
        torch.cuda.current_stream().synchronize(); t1 = time.perf_counter()
        plan = []                                              # (sess, mode, hu, ha, act, n)
        for k, (sess, uh, ah, mode) in enumerate(items):
            hu, ha = hs[2 * k], hs[2 * k + 1]
            act = rules.frame_energy(sess.win_u)
            if mode == "listen":
                self._maybe_partial(sess)
            if rules.switch_mode(sess, mode):
                self.head.reset_slot(sess.slot)
            plan.append((sess, mode, hu, ha, act, min(HOP_FRAMES, hu.shape[0], ha.shape[0], len(act))))
        for j in range(HOP_FRAMES):                            # frame j of the hop, all sessions at once
            live = [p for p in plan if j >= HOP_FRAMES - p[5]]
            if not live:
                continue
            i = j - HOP_FRAMES                                 # the hop's newest n frames: -n..-1
            out = self.head.step([(p[0].slot, p[2][i], p[3][i], float(p[4][i]), p[0].prev_state) for p in live])
            for sess, mode, hu, ha, act, n in live:
                events, final, reset = rules.on_frame(sess, mode, out[sess.slot], float(act[i]), left=-i - 1)
                res[sess][0].extend(events)
                if final is not None:
                    res[sess][1].append(final)
                if reset:
                    self.head.reset_slot(sess.slot)
        self.t_enc += t1 - t0; self.t_head += time.perf_counter() - t1
        return res

    def _cut_segment(self, sess):
        """Send the turn's audio so far, up to a pause, for transcription now (on the ASR thread); keep the rest."""
        hop = int(FRAME_S * SR)
        lo = int((SEG_S - SEG_SEARCH_S) * SR) // hop
        e = rules.frame_energy(sess.turn_u[lo * hop:int(SEG_S * SR)])
        cut = (lo + int(np.argmin(e)) + 1) * hop if e.size else int(SEG_S * SR)
        seg, sess.turn_u = sess.turn_u[:cut], sess.turn_u[cut:]
        segs, i = sess.segs, len(sess.segs)
        segs.append(None); sess.seg_s += cut / SR

        def done(text, segs=segs, i=i):
            segs[i] = text
        self.asr_q.put(Job("segment", sess, sess.turn_gen, seg, done=done))

    def _maybe_partial(self, sess):
        """Ask for a fresh partial of the turn (its audio since the last block: for backchannels, and the user's words
        so far for the page) at most every partial_refresh_s once the user has made a sound, decoded on the ASR
        thread."""
        cur_s = sess.turn_len_s()
        if not sess.in_turn or sess.partial_inflight or cur_s - sess.partial_at_s < sess.partial_refresh_s:
            return
        if sess.turn_u.size < int(0.3 * SR):
            return
        sess.partial_at_s = cur_s; sess.partial_inflight = True
        self.asr_q.put(Job("partial", sess, sess.turn_gen, sess.turn_u.copy(), segs=len(sess.segs)))
