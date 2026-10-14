"""GruBatch: the turn-taker's step for every live conversation at once. Each conversation (slot) has its own row of
GRU state; a step gathers the live rows, runs one frame of all of them as one batch, and writes their new state back.
A reset zeroes the row, which is where the GRU starts a fresh context. Small enough to run eager: a 3-layer GRU step
is a few kernels.
"""
from __future__ import annotations

import numpy as np, torch


class GruBatch:
    """step(entries) -> (slots, 8) numpy: [state logits x4, hazard, yield, bc, p_end], for the engine (EarsEngine)."""
    def __init__(self, model, slots, device):
        self.model = model.to(device); self.S = slots; self.device = device
        self.h = torch.zeros(model.core.num_layers, slots, model.core.hidden_size, device=device)

    def reset_slot(self, s):
        self.h[:, s] = 0

    @torch.no_grad()
    def step(self, entries):
        """entries: [(slot, hu (1024,) f32, ha (1024,) f32, act_dB float, prev_state int)]."""
        dev = self.device
        rows = torch.tensor([e[0] for e in entries], device=dev)
        hu = torch.stack([e[1] for e in entries])[:, None]
        ha = torch.stack([e[2] for e in entries])[:, None]
        db = torch.tensor([[e[3]] for e in entries], dtype=torch.float32, device=dev)
        prev = torch.tensor([[e[4]] for e in entries], dtype=torch.long, device=dev)
        out, h = self.model(hu, ha, ((db + 50) / 25).clamp(-3, 3), prev, self.h[:, rows].contiguous())
        self.h[:, rows] = h
        res = np.zeros((self.S, out.shape[-1]), np.float32)
        res[[e[0] for e in entries]] = out[:, 0].cpu().numpy()
        return res
