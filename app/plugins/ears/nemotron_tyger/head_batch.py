"""The heads' batched step for every live conversation at once: RingKV, a static KV cache with one row (slot) per
conversation, and HeadBatch, one head step for all of them as a CUDA-graph replay. Each slot has its own write
pointer, RoPE positions and validity mask, so it attends to exactly its own context (its last `ring_len` frames).
Static shapes make the step one graph replay (one graph per attention width; the narrowest that covers every active
row) instead of ~46 ms of eager kernels per frame.
"""
from __future__ import annotations
import logging

import numpy as np, torch

log = logging.getLogger("ears")


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
