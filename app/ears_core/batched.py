"""Multi-session EARS: shared models, per-session state, one batched GPU step per 320 ms hop for all sessions
that have a hop ready.

Same decisions as EndpointEars (arm C, endpoint_ears.py -- kept as the single-session reference; the per-frame
logic in EarsEngine._frame is a line-by-line port of EndpointEars.feed). What changes is how the GPU work runs:
  * encoder: the (user, agent) windows of every session in a tick go through nemotron together, grouped by
    window length (after the first 8 s every window is 8 s, so it is one batch), stopping at layer 15;
  * heads: the head backbone (Qwen3-0.6B + LoRA, merged at load) steps all sessions at once through RingKV, a
    static KV cache with one slot per session. Each slot has its own write pointer, RoPE positions and
    validity mask, so it attends to exactly its own context (its last `ring_len` frames; the old per-session
    DynamicCache grew without bound). Static shapes -> the step is one CUDA graph replay (one graph per
    attention width; the narrowest that covers every active row) instead of ~46 ms of eager kernels per frame;
  * ASR (the endpoint transcript, and the rolling partial used for backchannels) runs on a side thread with
    its own CUDA stream, so it never stalls the hop clock; greedy RNN-T decoding is a CUDA graph (rnnt.py). A hop that ends with an ENDPOINT resolves when its
    transcript is ready; partials land asynchronously (at most one refresh behind).
Dropped vs EndpointEars: the L15 "affect" vectors on ENDPOINT/BACKCHANNEL events (nothing consumed them).
Privacy: audio and transcripts stay in memory; logs carry counts and timings only.
"""
from __future__ import annotations
import time, queue, threading, logging
import numpy as np, torch
from .encoder_stream import ENC_SR
from .labeling import FRAME_S
from .model import LISTEN, SPEAK
from .live_ears import HOP_FRAMES, WINDOW_S, STATE_NAMES
from .endpoint_head import EndpointModel
from .rnnt import GraphedRNNT
from . import policy as fp

log = logging.getLogger("ears")
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
VAD_DB = -50.0; MIN_SPEECH_MS = 350.0; WARMUP_FRAMES = 6      # as LiveEars
TURN_MAX_S = 20.0                       # longest audio one transcription gets (sizes the RNN-T graph); a safety cap
SEG_S = 15.0                            # a turn longer than this is transcribed in blocks as it grows (I-05):
SEG_SEARCH_S = 4.0                      #   cut at the quietest 80 ms frame of the block's last 4 s
BARGE_PRE_S = 2.0                     # user audio kept while the agent speaks: seeds the turn a barge-in starts (I-02)
L15 = 15                                                        # encoder layer the heads read
HOP_N = int(HOP_FRAMES * FRAME_S * ENC_SR)                      # samples per hop (320 ms @ 16 kHz = 5120)


def frame_energy(wav):
    """Per-80 ms-frame energy in dB (LiveEars._frame_energy, vectorised)."""
    hop = int(round(FRAME_S * ENC_SR)); n = len(wav) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    seg = wav[:n * hop].reshape(n, hop).astype(np.float32)
    return (20 * np.log10(np.sqrt((seg ** 2).mean(1) + 1e-12) + 1e-9)).astype(np.float32)


class RingKV:
    """Static KV cache for the head backbone: `slots` rows x (1 scratch + `ring_len`) columns per layer. Each
    row is written at its own column (`widx`); idle rows write the scratch column 0. Passed to the HF backbone
    as `past_key_values` (it only needs `update`); attention reads only the first `cols` columns (the CUDA graph
    for the current width), since a turn's context sits at the front of the ring until it wraps."""
    def __init__(self, n_layers, slots, ring_len, n_kv, head_dim, dtype, device):
        cols = ring_len + 1
        self.k = [torch.zeros(slots, n_kv, cols, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros_like(x) for x in self.k]
        self.i0 = torch.arange(slots, device=device)[:, None]
        self.i1 = torch.arange(n_kv, device=device)[None, :]
        self.widx = torch.zeros(slots, dtype=torch.long, device=device)
        self.cols = cols

    def update(self, key, value, layer_idx, *args, **kwargs):          # key/value: (slots, n_kv, 1, head_dim)
        w = self.widx[:, None]
        self.k[layer_idx].index_put_((self.i0, self.i1, w), key[:, :, 0])
        self.v[layer_idx].index_put_((self.i0, self.i1, w), value[:, :, 0])
        return self.k[layer_idx][:, :, :self.cols], self.v[layer_idx][:, :, :self.cols]

    def get_seq_length(self, layer_idx=0):
        return 0


class HeadBatch:
    """One head step for up to `slots` sessions at once. step(entries) -> (slots, 8) numpy:
    [state logits x4, hazard, yield, bc, p_end] (probabilities sigmoid'ed, state logits raw)."""
    def __init__(self, ym, slots, ring_len, device, cuda_graph=True):
        self.ym = ym; self.S = slots; self.L = ring_len; self.device = device
        cfg = ym.base.backbone.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.kv = RingKV(cfg.num_hidden_layers, slots, ring_len, cfg.num_key_value_heads, hd, ym.base.dtype, device)
        S, cols = slots, ring_len + 1
        f32 = dict(dtype=torch.float32, device=device)
        self.hu = torch.zeros(S, 1, 1024, **f32); self.ha = torch.zeros(S, 1, 1024, **f32)
        self.act = torch.zeros(S, 1, **f32)
        self.prev = torch.zeros(S, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(S, 1, dtype=torch.long, device=device)
        self.part = torch.zeros(S, dtype=torch.bool, device=device)
        self.valid = torch.zeros(S, cols, dtype=torch.bool, device=device)
        self.mask = torch.zeros(S, 1, 1, cols, dtype=torch.bool, device=device)
        self.ar = torch.arange(S, device=device)
        self.count = [0] * S          # frames written since the slot's last reset -> ring pointer, width
        self.npos = [0] * S           # RoPE position since the slot's last reset (= DynamicCache length before)
        # attention widths (scratch + ring columns): one CUDA graph each; a step uses the narrowest one covering
        # every active row's context (attending 2049 mostly-masked columns cost ~2/3 of a step)
        self.widths = []
        w = 128
        while w < ring_len:
            self.widths.append(w + 1); w *= 2
        self.widths.append(ring_len + 1)
        self.graphs = {}; self.outs = {}; self.graph = None
        if cuda_graph:
            try:
                self._capture()
            except Exception as e:     # fall back to eager (still batched) rather than fail start-up
                log.warning("head CUDA graph capture failed (%s); running eager", type(e).__name__)
                self.graph = None

    def _compute(self, cols):
        ym, b = self.ym, self.ym.base
        self.kv.cols = cols
        self.valid.index_put_((self.ar, self.kv.widx), self.part)       # new column valid for active rows
        self.mask[:, 0, 0, :].copy_(self.valid)
        self.mask[:, 0, 0, 0] = ~self.part                              # idle rows attend the scratch column
        x = ym.fuse2(self.hu, self.ha, self.act, self.prev)
        o = b.backbone(inputs_embeds=x, attention_mask={"full_attention": self.mask[..., :cols]},
                       position_ids=self.pos,
                       past_key_values=self.kv, use_cache=True)
        z = o.last_hidden_state.float()
        return torch.cat([b.state_head(z), torch.sigmoid(b.hazard_head(z)), torch.sigmoid(ym.yield_head(z)),
                          torch.sigmoid(ym.bc_head(z)), torch.sigmoid(ym.ep_head(z))], -1)[:, 0]

    @torch.no_grad()
    def _capture(self):
        # all slots idle -> warmup/capture only touch the scratch column
        pool = None
        for cols in reversed(self.widths):                              # widest first; all share one pool
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._compute(cols)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self.outs[cols] = self._compute(cols)
            pool = g.pool(); self.graphs[cols] = g
        self.graph = self.graphs[self.widths[-1]]

    def reset_slot(self, s):
        self.valid[s] = False; self.npos[s] = 0; self.count[s] = 0

    @torch.no_grad()
    def step(self, entries):
        """entries: [(slot, hu (1024,) cuda f32, ha (1024,) cuda f32, act_dB float, prev_state int)]."""
        S, L = self.S, self.L
        part = torch.zeros(S, dtype=torch.bool); widx = torch.zeros(S, dtype=torch.long)
        pos = torch.zeros(S, 1, dtype=torch.long); prev = torch.zeros(S, 1, dtype=torch.long)
        act = torch.zeros(S, 1)
        for s, hu, ha, a, p in entries:
            part[s] = True; widx[s] = 1 + self.count[s] % L; pos[s, 0] = self.npos[s]; prev[s, 0] = p
            act[s, 0] = float(np.clip((a + 50) / 25, -3, 3))
            self.hu[s, 0].copy_(hu); self.ha[s, 0].copy_(ha)
            self.count[s] += 1; self.npos[s] += 1
        self.part.copy_(part); self.kv.widx.copy_(widx); self.pos.copy_(pos); self.prev.copy_(prev)
        self.act.copy_(act)
        need = 1 + max(min(self.count[e[0]], L) for e in entries)       # columns in use, scratch included
        cols = next(w for w in self.widths if w >= need)
        if self.graphs:
            self.graphs[cols].replay(); out = self.outs[cols]
        else:
            out = self._compute(cols)
        return out.cpu().numpy()


class EarsSession:
    """One conversation's EARS state (what LiveEars/EndpointEars kept on self), plus its thresholds."""
    def __init__(self, slot, ep_thr=0.9146, ep_deb=2, barge_thr=0.5, barge_deb=2, bc_thr=0.5, bc_deb=2,
                 fallback_frames=15, theta=0.2, bc_refractory=15, partial_refresh_s=0.8):
        self.slot = slot
        self.ep_thr = ep_thr; self.ep_deb = ep_deb; self.barge_thr = barge_thr; self.barge_deb = barge_deb
        self.bc_thr = bc_thr; self.bc_deb = bc_deb; self.FALLBACK_SIL_FRAMES = fallback_frames
        self.theta = theta; self.bc_refractory = bc_refractory; self.partial_refresh_s = partial_refresh_s
        self.closed = False; self.turn_gen = 0; self.partial_inflight = False
        self.trace = None                      # set to a list to record per-frame head outputs (tests only)
        self._clear()
        self.pending_reset = True              # the slot's ring row is cleared on the first hop

    def _clear(self):
        self.win_u = np.zeros(0, np.float32); self.win_a = np.zeros(0, np.float32)
        self.turn_u = np.zeros(0, np.float32)  # this turn's user audio not yet sent for transcription
        self.seg_s = 0.0                       # seconds of this turn already cut into segments
        self.segs = []                         # their texts, filled in by the ASR thread (None until done)
        self.pre_u = np.zeros(0, np.float32)   # last BARGE_PRE_S of user audio in speaking mode
        self.seed_u = None                     # set on BARGE: the next reset starts turn_u with it
        self.prev_state = LISTEN; self.yield_run = 0
        self.voiced_ms = 0.0; self.sil_run = 0; self.in_turn = False; self.warm = 0
        self._mode = "listen"; self.bc_run = 0; self.bc_cooldown = 0; self.ep_run = 0
        self.partial_asr = ""; self.partial_at_s = 0.0

    def reset(self):
        """Thread-safe: applied by the GPU thread before this session's next hop."""
        self.pending_reset = True


    def turn_len_s(self):
        return self.seg_s + len(self.turn_u) / ENC_SR


class _StopAtL15(Exception):
    pass


class _Job:
    __slots__ = ("kind", "sess", "gen", "wav", "done")
    # kind: "final" (endpoint transcript), "segment" (a finished block of a long turn), "partial" (backchannels)

    def __init__(self, kind, sess, gen, wav, done=None):
        self.kind = kind; self.sess = sess; self.gen = gen; self.wav = wav; self.done = done


class EarsEngine:
    def __init__(self, heads_ckpt, slots=4, ring_len=2048, device="cuda", merge_lora=True, cuda_graph=True,
                 model_id=MODEL_ID):
        from transformers import AutoModel, AutoProcessor
        self.device = device; self.slots = slots
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.fe = self.proc.feature_extractor
        self.fe.mel_filters = self.fe.mel_filters.to(device)   # features are computed on the GPU (self.features)
        self.asr = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        self.enc = self.asr.encoder
        self.ym = EndpointModel(device=device); self.ym.load(heads_ckpt); self.ym.eval()
        if merge_lora:                         # LoRA folded into the frozen weights: fewer kernels per step
            self.ym.base.backbone = self.ym.base.backbone.merge_and_unload()
        self.head = HeadBatch(self.ym, slots, ring_len, device, cuda_graph=cuda_graph)
        log.info("EARS engine: %d slots, ring %d frames, LoRA merged=%s, head CUDA graph=%s",
                 slots, ring_len, merge_lora, self.head.graph is not None)
        self._zero_enc = {}                    # window length -> L15 of an all-zero window
        # The heads read layer 15 of 24: EARS encodes stop there (a hook on that layer raises), ~1/3 less encoder
        # work per tick. Thread-local, so the ASR thread's full-depth encodes of the same module are untouched.
        self._tls = threading.local()
        self.enc.layers[L15 - 1].register_forward_hook(self._l15_hook)
        self.asr_stream = torch.cuda.Stream()
        self._gen_lock = threading.Lock()      # the decoder's static graph buffers: one transcription at a time
        self.rnnt = GraphedRNNT(self.asr, max_batch=slots, max_frames=int(TURN_MAX_S / FRAME_S) + 8)
        if cuda_graph:
            self.rnnt.capture()
        self.asr_q: queue.Queue = queue.Queue()
        self.t_enc = self.t_head = self.t_asr = 0.0          # busy seconds, reset by the stats log
        threading.Thread(target=self._asr_loop, name="ears-asr", daemon=True).start()

    def _l15_hook(self, mod, args, out):
        if getattr(self._tls, "stop", False):
            self._tls.h = out[0] if isinstance(out, tuple) else out
            raise _StopAtL15

    # ---- GPU helpers ----
    @torch.no_grad()
    def features(self, wavs):
        """list of 16 kHz clips -> (log-mel features, attention mask) on the GPU: the processor's feature extraction
        (right-pad, preemphasis, STFT, mel, mask) with the waveform on the GPU. Its CPU STFT was ~20 ms per tick at
        4 conversations (8 windows), more than the encoder; this is ~2 ms. Same values within float noise (~2e-4 on
        log-mels of ~16). The extractor's own device="cuda" leaves the waveform on the CPU and fails."""
        fe, dev = self.fe, self.device
        n = max(len(w) for w in wavs)
        x = np.zeros((len(wavs), n), np.float32)
        for i, w in enumerate(wavs):
            x[i, :len(w)] = w
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        lens = torch.tensor([len(w) for w in wavs], device=dev)
        inside = torch.arange(n, device=dev)[None] < lens[:, None]
        x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], 1).masked_fill(~inside, 0.0)
        f = fe._torch_extract_fbank_features(x, dev, center=True)
        flen = torch.floor_divide(lens + fe.n_fft // 2 * 2 - fe.n_fft, fe.hop_length)
        am = torch.arange(f.shape[1], device=dev)[None] < flen[:, None]
        return f * am.unsqueeze(-1), am

    @torch.no_grad()
    def encode(self, wavs):
        """list of 16 kHz windows -> list of (T, 1024) L15 tensors on the GPU; equal lengths are batched.
        All-zero windows (the agent channel while listening) are encoded once per length and cached: same input,
        same features, half the feature-extraction CPU and encoder work while everyone listens."""
        out = [None] * len(wavs); groups = {}
        for i, w in enumerate(wavs):
            if len(w) in self._zero_enc and not w.any():
                out[i] = self._zero_enc[len(w)]
            else:
                groups.setdefault(len(w), []).append(i)
        for n, idx in groups.items():
            feats, am = self.features([wavs[i] for i in idx])
            self._tls.stop = True
            try:
                self.enc(input_features=feats, attention_mask=am, num_lookahead_tokens=0)
            except _StopAtL15:
                pass
            finally:
                self._tls.stop = False
            hs = self._tls.h.float()
            for j, i in enumerate(idx):
                out[i] = hs[j]
                if n not in self._zero_enc and not wavs[i].any():
                    self._zero_enc[n] = hs[j].clone()
        return out

    @torch.no_grad()
    def transcribe(self, wavs):
        """list of 16 kHz clips -> list of text, on the ASR stream, as one padded batch. Greedy RNN-T as a CUDA
        graph (rnnt.py): same text as nemotron's generate(), ~4x faster and without its per-step Python."""
        if not wavs:
            return []
        with self._gen_lock, torch.cuda.stream(self.asr_stream):
            feats, am = self.features(wavs)
            seqs = self.rnnt(feats, am)
            return [self.proc.decode(q, skip_special_tokens=True).strip() for q in seqs]

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
            return self.transcribe([j.wav for j in jobs])
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
            sess.win_u = np.concatenate([sess.win_u, uh])[-int(WINDOW_S * ENC_SR):]
            sess.win_a = np.concatenate([sess.win_a, ah])[-int(WINDOW_S * ENC_SR):]
            if mode == "listen":
                sess.turn_u = np.concatenate([sess.turn_u, uh])
                if len(sess.turn_u) > SEG_S * ENC_SR:
                    self._cut_segment(sess)
                sess.turn_u = sess.turn_u[-int(TURN_MAX_S * ENC_SR):]
            else:
                sess.pre_u = np.concatenate([sess.pre_u, uh])[-int(BARGE_PRE_S * ENC_SR):]
            res[sess] = ([], [])
        t0 = time.perf_counter()
        hs = self.encode([w for sess, *_ in items for w in (sess.win_u, sess.win_a)])
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
        hop = int(FRAME_S * ENC_SR)
        lo = int((SEG_S - SEG_SEARCH_S) * ENC_SR) // hop
        e = frame_energy(sess.turn_u[lo * hop:int(SEG_S * ENC_SR)])
        cut = (lo + int(np.argmin(e)) + 1) * hop if e.size else int(SEG_S * ENC_SR)
        seg, sess.turn_u = sess.turn_u[:cut], sess.turn_u[cut:]
        segs, i = sess.segs, len(sess.segs)
        segs.append(None); sess.seg_s += cut / ENC_SR

        def done(text, segs=segs, i=i):
            segs[i] = text
        self.asr_q.put(_Job("segment", sess, sess.turn_gen, seg, done=done))

    def _maybe_partial(self, sess):
        """LiveEars._refresh_partial, asynchronous: same throttle, decoded on the ASR thread."""
        cur_s = sess.turn_len_s()
        if sess.partial_inflight or cur_s - sess.partial_at_s < sess.partial_refresh_s:
            return
        if sess.win_u.size < int(0.3 * ENC_SR):
            return
        sess.partial_at_s = cur_s; sess.partial_inflight = True
        self.asr_q.put(_Job("partial", sess, sess.turn_gen, sess.win_u.copy()))

    def _frame(self, sess, mode, vals, act_db, res):
        """One frame's decision: EndpointEars.feed's per-frame body, verbatim in behaviour."""
        events, finals = res
        slog = vals[:4]; haz, yp, bcp, pep = (float(v) for v in vals[4:8])
        if sess.trace is not None:
            sess.trace.append((mode, haz, yp, bcp, pep))
        st = fp._decide(slog, haz, sess.prev_state, sess.theta, 0.5)
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
        fallback_end = sess.sil_run >= sess.FALLBACK_SIL_FRAMES
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


class EarsService:
    """Owns the GPU thread. Sessions call `await feed(...)` once per hop; the thread gathers the hops of all
    active sessions (up to `gather_s`) and runs them as one batched tick. A session that is behind (said
    `more`: its next hop is already buffered) resubmits right after its tick resolves, so the next gather
    waits up to `expect_s` for it: catch-up hops share one tick instead of each costing a tick of their own."""
    def __init__(self, engine: EarsEngine, gather_s=0.015, expect_s=0.06, stats_every_s=60.0):
        self.engine = engine; self.gather_s = gather_s; self.expect_s = expect_s
        self.stats_every_s = stats_every_s; self.expect = set()
        self.free = list(range(engine.slots)); self.sessions = set(); self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self._ticks = []; self._t_stats = time.time()
        threading.Thread(target=self._loop, name="ears-gpu", daemon=True).start()

    def full(self):
        with self.lock:
            return not self.free

    def active(self):
        with self.lock:
            return len(self.sessions)

    def open(self, **thresholds):
        with self.lock:
            if not self.free:
                return None
            s = EarsSession(self.free.pop(0), **thresholds); self.sessions.add(s)
            return s

    def close(self, sess):
        sess.closed = True
        self.q.put(("close", sess))                           # slot freed on the GPU thread, after its last hop

    async def feed(self, sess, user_hop, agent_hop, mode, more=False):
        import asyncio
        loop = asyncio.get_running_loop(); fut = loop.create_future()
        self.q.put(("feed", sess, user_hop, agent_hop, mode, fut, loop, more))
        return await fut

    @staticmethod
    def _resolve(item, events=None, exc=None):
        fut, loop = item[5], item[6]

        def _set():
            if not fut.done():
                fut.set_exception(exc) if exc is not None else fut.set_result(events)
        loop.call_soon_threadsafe(_set)

    def _loop(self):
        while True:
            items = [self.q.get()]
            t0 = time.perf_counter()
            while True:
                with self.lock:
                    n_live = len(self.sessions)
                have = {it[1] for it in items if it[0] == "feed"}
                if len(have) >= n_live:
                    break
                waiting = any(s not in have and not s.closed for s in self.expect)
                rem = t0 + (self.expect_s if waiting else self.gather_s) - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    items.append(self.q.get(timeout=rem))
                except queue.Empty:
                    break
            self.expect = {it[1] for it in items if it[0] == "feed" and it[7] and not it[1].closed}
            feeds = [it for it in items if it[0] == "feed" and not it[1].closed]
            for it in items:
                if it[0] == "feed" and it[1].closed:
                    self._resolve(it, [])
            if feeds:
                self._tick(feeds)
            for it in items:
                if it[0] == "close":
                    with self.lock:
                        self.sessions.discard(it[1])
                        self.engine.head.reset_slot(it[1].slot); self.free.append(it[1].slot)

    def _tick(self, feeds):
        t0 = time.perf_counter()
        try:
            res = self.engine.process([(it[1], it[2], it[3], it[4]) for it in feeds])
        except Exception as e:
            log.exception("EARS tick error: %s", type(e).__name__)
            for it in feeds:
                self._resolve(it, exc=e)
            return
        self._ticks.append((time.perf_counter() - t0, len(feeds)))
        for it in feeds:
            events, finals = res[it[1]]
            if not finals:
                self._resolve(it, events); continue
            left = [len(finals)]

            def done(text, ev, it=it, events=events, left=left):
                segs = ev.pop("seg_texts", None) or []                 # a long turn's earlier blocks (I-05)
                ev["transcript"] = " ".join(t for t in [*segs, text] if t)
                left[0] -= 1
                if left[0] == 0:
                    self._resolve(it, events)
            for ev, wav in finals:
                self.engine.asr_q.put(_Job("final", it[1], it[1].turn_gen, wav,
                                           done=lambda t, ev=ev, done=done: done(t, ev)))
        if time.time() - self._t_stats >= self.stats_every_s and self._ticks:
            ms = np.array([t for t, _ in self._ticks]) * 1e3; bs = np.array([b for _, b in self._ticks])
            e = self.engine; span = time.time() - self._t_stats
            log.info("EARS ticks: n=%d sessions/tick avg %.2f | tick p50 %.0f p90 %.0f max %.0f ms (hop 320 ms) | "
                     "busy: encoder %.0f%% head %.0f%% asr %.0f%% | GPU mem %.1f GiB (reserved %.1f)", len(ms),
                     bs.mean(), np.percentile(ms, 50), np.percentile(ms, 90), ms.max(), 100 * e.t_enc / span,
                     100 * e.t_head / span, 100 * e.t_asr / span, torch.cuda.memory_allocated() / 2**30,
                     torch.cuda.memory_reserved() / 2**30)
            self._ticks = []; self._t_stats = time.time(); e.t_enc = e.t_head = e.t_asr = 0.0
