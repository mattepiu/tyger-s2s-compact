"""TurnTaker: the bundle's turn-taking controller (`turn_taker.*`), a causal 3-layer GRU over nemotron's layer 15.

Per frame it reads the layer-15 features of the user's audio and of the agent's own audio (the 2nd ear), the user's
normalised frame energy and the previous state (LISTEN while listening, SPEAK while speaking: never its own decision).
Their projections are summed into one input per frame (`fuse`), the GRU carries the conversation (its state: 9 KiB per
conversation, no KV cache), and five heads read its output (`outputs`): the 4-way state logits, and the hazard, yield
(barge), backchannel and endpoint probabilities, the layout the turn rules read. Everything is fp32.

Module names and the order of the sums are the research runtime's (tyger_endpointer.controller), so its tensors load
as they are and the outputs are the same.
"""
from __future__ import annotations

import torch, torch.nn as nn

from .constants import ENC_HIDDEN, N_STATE


def _mlp_head(d):
    return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))


class TurnTaker(nn.Module):
    def __init__(self, width=768, layers=3):
        """The modules (random weights: bundle.load fills them in)."""
        super().__init__()
        d = width
        self.core = nn.GRU(d, d, layers, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        self.proj_user = nn.Linear(ENC_HIDDEN, d)          # the user's layer-15 features
        self.proj_agent = nn.Linear(ENC_HIDDEN, d)         # the agent's (the 2nd ear)
        self.proj_energy = nn.Linear(1, d)                 # the user's frame energy
        self.state_emb = nn.Embedding(N_STATE + 1, d)      # the previous state (+ a begin id, unused live)
        self.in_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, N_STATE)
        self.hazard_head = nn.Linear(d, 1)
        self.yield_head = _mlp_head(d)                     # the user is barging in: yield now (speaking)
        self.bc_head = _mlp_head(d)                        # the user gave a cue: backchannel now (listening)
        self.ep_head = _mlp_head(d)                        # the user's turn ended: take the floor (listening)

    def fuse(self, h_user, h_agent, act, prev_state):
        """(B, T, 1024) user and agent features, (B, T) normalised energy, (B, T) previous state -> (B, T, d)."""
        return self.in_norm(self.proj_user(h_user) + self.proj_agent(h_agent) + self.proj_energy(act.unsqueeze(-1))
                            + self.state_emb(prev_state))

    def outputs(self, z):
        """The GRU's output (B, T, d) -> (B, T, 8): state logits x4, then the hazard, yield, backchannel and endpoint
        probabilities."""
        z = self.out_norm(z)
        return torch.cat([self.state_head(z), torch.sigmoid(self.hazard_head(z)), torch.sigmoid(self.yield_head(z)),
                          torch.sigmoid(self.bc_head(z)), torch.sigmoid(self.ep_head(z))], -1)

    def forward(self, h_user, h_agent, act, prev_state, state=None):
        """T frames of B conversations from their GRU state (layers, B, d; None: a fresh context) -> (outputs, the
        new state)."""
        z, state = self.core(self.fuse(h_user, h_agent, act, prev_state), state)
        return self.outputs(z), state
