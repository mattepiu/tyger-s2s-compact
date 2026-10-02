"""exp-7 barge yield head: a TWO-STREAM, additive extension of the exp-4 StateChannelModel.

Why (step-1 findings, scripts/55): the frozen single-stream head is sticky in the SPEAK->LISTEN yield
direction; the ceiling probe showed (a) the ASSISTANT channel adds real signal (both 0.774 > user 0.662)
and (b) user activity/energy out-reads the L15 user feats. So we ADD:
  - proj_agent: the assistant-channel L15 (the second ear, reason #2),
  - proj_act:   user activity (energy dB),
  - yield_head: an EARLY-WARNING "the user is barging, yield now" head. Distinct from the 4-way state
    head by design: at a barge ONSET the ground-truth state is still SPEAK (the agent hasn't stopped),
    so the state head can't fire early -- the yield head predicts the imminent SPEAK->LISTEN from the
    overlap, which is exactly the signal the state head lacks.

Additive & non-destructive: proj_agent/proj_act and the yield head's last layer are ZERO-init, so at
load the fused input == the original single-stream model (the 0.903 behavior is the starting point). The
4-way state + hazard heads stay supervised during training (multitask) so states can't be forgotten --
the direct guard against exp-6-style degradation (and it lives on the 0.6B, never the frozen 4B).
"""
from __future__ import annotations
import torch, torch.nn as nn
from .model import StateChannelModel, TEXT_PAD


class YieldModel(nn.Module):
    def __init__(self, backbone_id="Qwen/Qwen3-0.6B", enc_hidden=1024, device="cuda",
                 lora=True, lora_r=16):
        super().__init__()
        self.base = StateChannelModel(backbone_id=backbone_id, enc_hidden=enc_hidden,
                                      device=device, lora=lora, lora_r=lora_r)
        d = self.base.d_model
        self.device = device
        self.proj_agent = nn.Linear(enc_hidden, d)
        self.proj_act = nn.Linear(1, d)
        self.yield_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        # bc_head: the EARLY-WARNING "the user gave a cue -> backchannel now" head (LISTEN mode). Distinct
        # from the 4-way state head for the same reason yield_head is: entering BACKCHANNEL from the user
        # cue is the hard free-run transition the frozen head can't do (p[BC] max 0.12; exp-4's 0.84 was
        # teacher-forced). Zero-init last layer => at load bc == 0.5 everywhere (additive identity), so
        # loading a yield ckpt preserves the 0.903/0.944 model exactly and bc is learned on top.
        self.bc_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        for lin in (self.proj_agent, self.proj_act, self.yield_head[-1], self.bc_head[-1]):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)   # additive identity at start
        for m in (self.proj_agent, self.proj_act, self.yield_head, self.bc_head):
            m.to(device).float()

    def load_base(self, ckpt):
        sd = torch.load(ckpt, map_location=self.device)
        self.base.load_state_dict(sd, strict=False)

    def fuse2(self, h_user, h_agent, act, prev_state):
        b = self.base
        x = b.proj_h(h_user.float()) + b.state_emb(prev_state)
        with torch.no_grad():
            te = b.embed_tokens(torch.full_like(prev_state, TEXT_PAD)).float()
        x = x + te + self.proj_agent(h_agent.float()) + self.proj_act(act.float().unsqueeze(-1))
        return b.in_norm(x).to(b.dtype)

    def forward(self, h_user, h_agent, act, prev_state, attn_mask=None):
        x = self.fuse2(h_user, h_agent, act, prev_state)
        z = self.base.backbone_hidden(x, attn_mask)                # (B,T,d) float
        return (self.base.state_head(z), self.base.hazard_head(z).squeeze(-1),
                self.yield_head(z).squeeze(-1),                    # state (B,T,4), hazard (B,T), yield (B,T)
                self.bc_head(z).squeeze(-1))                       # bc (B,T)

    def trainable_parameters(self):
        ps = list(self.proj_agent.parameters()) + list(self.proj_act.parameters()) + \
             list(self.yield_head.parameters()) + list(self.bc_head.parameters()) + \
             list(self.base.state_head.parameters()) + list(self.base.hazard_head.parameters())
        ps += [p for p in self.base.backbone.parameters() if p.requires_grad]   # LoRA
        return ps

    def save(self, path):
        keep = ("proj_h.", "state_emb.", "in_norm.", "state_head.", "hazard_head.")
        base_sd = {k: v for k, v in self.base.state_dict().items()
                   if k.startswith(keep) or "lora" in k.lower()}
        torch.save({"base": base_sd, "proj_agent": self.proj_agent.state_dict(),
                    "proj_act": self.proj_act.state_dict(), "yield_head": self.yield_head.state_dict(),
                    "bc_head": self.bc_head.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.base.load_state_dict(ck["base"], strict=False)
        self.proj_agent.load_state_dict(ck["proj_agent"]); self.proj_act.load_state_dict(ck["proj_act"])
        self.yield_head.load_state_dict(ck["yield_head"])
        if "bc_head" in ck:                                        # older (yield-only) ckpts -> bc stays zero-init
            self.bc_head.load_state_dict(ck["bc_head"])
