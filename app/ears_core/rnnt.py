"""Lean greedy RNN-T decoding for nemotron-3.5-asr: the same steps as HF `generate()` (blank start token, joint
argmax per step, blank or `max_symbols_per_step` in a row advances the encoder frame, LSTM decoder updated on
non-blank tokens), as a plain batched tensor loop. HF's generate spends ~1 ms of Python per step (one step per
frame + one per token: ~170 steps for 8 s), holding the GIL the EARS tick thread needs; this loop does the same
math with no per-step host sync (it checks for the end every `check_every` steps)."""
import torch


@torch.no_grad()
def greedy_rnnt(asr, input_features, attention_mask=None, check_every=16):
    """-> (B, steps) token ids, laid out as generate().sequences (start token first, pad after the end)."""
    cfg = asr.config
    blank, pad, max_sym = cfg.blank_token_id, cfg.pad_token_id, asr.max_symbols_per_step
    eo = asr.get_audio_features(input_features=input_features, attention_mask=attention_mask,
                                output_attention_mask=True,
                                prompt_ids=torch.full((input_features.shape[0],), cfg.default_prompt_id,
                                                      dtype=torch.long, device=input_features.device))
    enc = eo.pooler_output                                       # (B, T, H)
    B, T, _ = enc.shape
    dev = enc.device
    lens = (eo.attention_mask.sum(-1) if eo.attention_mask is not None
            else torch.full((B,), T, dtype=torch.long, device=dev))
    dec, joint = asr.decoder, asr.joint
    ar = torch.arange(B, device=dev)

    def run_decoder(tok, hc):
        out, (h, c) = dec.lstm(dec.embedding(tok[:, None]), hc)
        return dec.decoder_projector(out)[:, 0], h, c

    tok = torch.full((B,), blank, dtype=torch.long, device=dev)
    d, h, c = run_decoder(tok, None)                             # decoder state after the start token
    idx = torch.zeros(B, dtype=torch.long, device=dev)
    sym = torch.zeros(B, dtype=torch.long, device=dev)
    done = idx >= lens
    seq = [tok]
    for step in range(max_sym * T - 1):
        logits = joint(decoder_hidden_states=d, encoder_hidden_states=enc[ar, idx.clamp(max=T - 1)])
        tok = logits.argmax(-1)
        seq.append(torch.where(done, torch.full_like(tok, pad), tok))
        is_blank = tok == blank
        sym = torch.where(is_blank, torch.zeros_like(sym), sym + 1)
        force = sym >= max_sym
        sym = torch.where(is_blank | force, torch.zeros_like(sym), sym)
        idx = idx + (is_blank | force).long()
        upd = (~is_blank)[:, None]                               # the decoder only moves on a real token
        d2, h2, c2 = run_decoder(tok, (h, c))
        d = torch.where(upd, d2, d); h = torch.where(upd[None], h2, h); c = torch.where(upd[None], c2, c)
        done = done | (idx >= lens)
        if (step + 1) % check_every == 0 and bool(done.all()):
            break
    return torch.stack(seq, 1)


class GraphedRNNT:
    """greedy_rnnt with the decode loop captured as one CUDA graph of `k` unrolled steps over static buffers of
    `max_batch` rows x `max_frames` encoder frames (padding is free: a row is done once its frame pointer
    passes its length, and pad rows have length 0). One replay = k steps for one launch; the host checks for
    the end once per replay. Longer inputs fall back to the eager loop. Same tokens as greedy_rnnt."""

    def __init__(self, asr, max_batch=4, max_frames=256, k=16):
        cfg = asr.config
        self.asr, self.B, self.T, self.k = asr, max_batch, max_frames, k
        self.blank, self.pad, self.max_sym = cfg.blank_token_id, cfg.pad_token_id, asr.max_symbols_per_step
        dev = next(asr.parameters()).device; H = asr.joint.head.in_features
        dec = asr.decoder
        nl, hd = dec.lstm.num_layers, dec.lstm.hidden_size
        self.enc = torch.zeros(self.B, self.T, H, device=dev)
        self.lens = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.d = torch.zeros(self.B, H, device=dev)
        self.h = torch.zeros(nl, self.B, hd, device=dev); self.c = torch.zeros_like(self.h)
        self.idx = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.sym = torch.zeros_like(self.idx); self.done = torch.zeros(self.B, dtype=torch.bool, device=dev)
        self.out = torch.zeros(self.B, k, dtype=torch.long, device=dev)
        self.ar = torch.arange(self.B, device=dev)
        with torch.no_grad():                                    # decoder state after the start token
            out, (h0, c0) = dec.lstm(dec.embedding(torch.full((1, 1), self.blank, device=dev)))
            self.d0, self.h0, self.c0 = dec.decoder_projector(out)[:, 0], h0, c0
        self.graph = None

    def _steps(self):
        dec, joint = self.asr.decoder, self.asr.joint
        d, h, c, idx, sym, done = self.d, self.h, self.c, self.idx, self.sym, self.done
        toks = []
        for _ in range(self.k):
            tok = joint(decoder_hidden_states=d, encoder_hidden_states=self.enc[self.ar, idx.clamp(max=self.T - 1)]
                        ).argmax(-1)
            toks.append(torch.where(done, torch.full_like(tok, self.pad), tok))
            is_blank = tok == self.blank
            sym = torch.where(is_blank, torch.zeros_like(sym), sym + 1)
            force = sym >= self.max_sym
            sym = torch.where(is_blank | force, torch.zeros_like(sym), sym)
            idx = idx + (is_blank | force).long()
            upd = (~is_blank)[:, None]
            o, (h2, c2) = dec.lstm(dec.embedding(tok[:, None]), (h, c))
            d = torch.where(upd, dec.decoder_projector(o)[:, 0], d)
            h = torch.where(upd[None], h2, h); c = torch.where(upd[None], c2, c)
            done = done | (idx >= self.lens)
        self.d.copy_(d); self.h.copy_(h); self.c.copy_(c)
        self.idx.copy_(idx); self.sym.copy_(sym); self.done.copy_(done)
        self.out.copy_(torch.stack(toks, 1))

    def _reset(self):
        self.d.copy_(self.d0.expand(self.B, -1)); self.h.copy_(self.h0.expand(-1, self.B, -1))
        self.c.copy_(self.c0.expand(-1, self.B, -1))
        self.idx.zero_(); self.sym.zero_(); self.done.copy_(self.lens == 0)

    @torch.no_grad()
    def capture(self):
        self.lens.zero_(); self._reset()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._steps()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._steps()

    @torch.no_grad()
    def __call__(self, input_features, attention_mask=None):
        """-> list of 1-D token tensors (start token first; may end in pad), one per input row."""
        cfg = self.asr.config; n = input_features.shape[0]
        eo = self.asr.get_audio_features(input_features=input_features, attention_mask=attention_mask,
                                         output_attention_mask=True,
                                         prompt_ids=torch.full((n,), cfg.default_prompt_id, dtype=torch.long,
                                                               device=input_features.device))
        enc = eo.pooler_output; T = enc.shape[1]
        lens = (eo.attention_mask.sum(-1) if eo.attention_mask is not None
                else torch.full((n,), T, dtype=torch.long, device=enc.device))
        if self.graph is None or T > self.T:
            return list(greedy_rnnt(self.asr, input_features, attention_mask))
        res = []
        for a in range(0, n, self.B):
            m = min(self.B, n - a)
            self.enc.zero_(); self.enc[:m, :T].copy_(enc[a:a + m]); self.lens.zero_(); self.lens[:m].copy_(lens[a:a + m])
            self._reset()
            outs = [torch.full((self.B, 1), self.blank, dtype=torch.long, device=enc.device)]
            for _ in range(-(-self.max_sym * T // self.k)):
                self.graph.replay(); outs.append(self.out.clone())
                if bool(self.done.all()):
                    break
            seq = torch.cat(outs, 1)
            res.extend(seq[i] for i in range(m))
        return res
