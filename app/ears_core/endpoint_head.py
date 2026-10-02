"""exp-12 LISTEN->SPEAK ENDPOINT head -- exp-7's bc recipe, one more time.

`EndpointModel` extends exp-7's `YieldModel` (codebase/exp-7/exp7/yield_head.py, UNTOUCHED) with a
single zero-init `ep_head` (Linear-SiLU-Linear on the backbone output, last layer zero-init exactly
like `yield_head` / `bc_head`). Initialised from the exp-10 chain `bc_s1` (which has no `ep_head`), so
at load the model *is* `bc_s1` and p_end = sigmoid(0) = 0.5 everywhere (additive identity). The
checkpoint gains an `ep_head` entry; everything else is byte-identical to a YieldModel ckpt.

forward returns 5 heads: (state, hazard, yield, bc, ep). The multitask loss (53_train_endpoint) keeps
every prior term supervised so state/yield/bc cannot drift (selection floors on all three, as exp-7 62).
"""
from __future__ import annotations
import torch, torch.nn as nn
from .yield_head import YieldModel


class EndpointModel(YieldModel):
    def __init__(self, backbone_id="Qwen/Qwen3-0.6B", enc_hidden=1024, device="cuda",
                 lora=True, lora_r=16):
        super().__init__(backbone_id=backbone_id, enc_hidden=enc_hidden, device=device,
                         lora=lora, lora_r=lora_r)
        d = self.base.d_model
        # ep_head: the LISTEN-mode "the user's turn just ended -> take the floor now" head. Distinct from
        # the 4-way state head (teacher-forced to keep prev, so weak at ENTERING SPEAK) exactly as
        # yield/bc are. Zero-init last layer => p_end == 0.5 at load (additive identity); loading bc_s1
        # preserves the 0.930/0.969/0.883 chain and ep is learned on top.
        self.ep_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.ep_head[-1].weight); nn.init.zeros_(self.ep_head[-1].bias)
        self.ep_head.to(device).float()

    def forward(self, h_user, h_agent, act, prev_state, attn_mask=None):
        x = self.fuse2(h_user, h_agent, act, prev_state)
        z = self.base.backbone_hidden(x, attn_mask)                    # (B,T,d) float
        return (self.base.state_head(z), self.base.hazard_head(z).squeeze(-1),
                self.yield_head(z).squeeze(-1), self.bc_head(z).squeeze(-1),
                self.ep_head(z).squeeze(-1))                           # state,haz,yield,bc,ep

    def trainable_parameters(self):
        return super().trainable_parameters() + list(self.ep_head.parameters())

    def save(self, path):
        keep = ("proj_h.", "state_emb.", "in_norm.", "state_head.", "hazard_head.")
        base_sd = {k: v for k, v in self.base.state_dict().items()
                   if k.startswith(keep) or "lora" in k.lower()}
        torch.save({"base": base_sd, "proj_agent": self.proj_agent.state_dict(),
                    "proj_act": self.proj_act.state_dict(), "yield_head": self.yield_head.state_dict(),
                    "bc_head": self.bc_head.state_dict(), "ep_head": self.ep_head.state_dict()}, path)

    def load(self, path):
        super().load(path)                                             # base + proj + yield + bc
        ck = torch.load(path, map_location=self.device)
        if "ep_head" in ck:
            self.ep_head.load_state_dict(ck["ep_head"])                # else: stays zero-init (from bc_s1)
