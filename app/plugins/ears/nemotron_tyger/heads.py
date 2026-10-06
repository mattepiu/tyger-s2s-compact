"""The tyger 1.0 heads (exp-12 arm C) as the research code defined them: StateChannelModel (exp-4: the frozen
backbone over fused per-frame inputs, the 4-way state head and the hazard head), YieldModel (exp-7: the agent channel,
the user's energy, the yield and backchannel heads) and EndpointModel (exp-12: the endpoint head). EARS uses only
their inference path (fuse2, the merged backbone, the heads). Training-only methods are kept as they were.
"""
from __future__ import annotations
import torch, torch.nn as nn

from .constants import ENC_HIDDEN, HEAD_BACKBONE, N_STATE, TEXT_PAD


class StateChannelModel(nn.Module):
    def __init__(self, backbone_id=HEAD_BACKBONE, enc_hidden=ENC_HIDDEN, device="cuda",
                 dtype=torch.bfloat16, lora=False, lora_r=16):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.device = device; self.dtype = dtype; self.backbone_id = backbone_id
        lm = AutoModelForCausalLM.from_pretrained(backbone_id, dtype=dtype)
        self.backbone = lm.model                      # base decoder (no LM head)
        self.embed_tokens = self.backbone.embed_tokens  # frozen text embeddings (Etext)
        d = self.backbone.config.hidden_size
        self.d_model = d
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0, bias="none",
                task_type="FEATURE_EXTRACTION")
            self.backbone = get_peft_model(self.backbone, cfg)
        # trainable bits (fp32 for stable head training)
        self.proj_h = nn.Linear(enc_hidden, d)
        self.state_emb = nn.Embedding(N_STATE + 1, d)   # +START
        self.in_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, N_STATE)
        self.hazard_head = nn.Linear(d, 1)
        self.to(device)
        # cast trainable modules to fp32, keep backbone in dtype
        for m in (self.proj_h, self.state_emb, self.in_norm, self.state_head, self.hazard_head):
            m.float()

    def fuse(self, h_user, prev_state, text_tok):
        """h_user (B,T,enc_hidden) float; prev_state (B,T) long; text_tok (B,T) long -> (B,T,d)."""
        x = self.proj_h(h_user.float())
        x = x + self.state_emb(prev_state)
        with torch.no_grad():
            te = self.embed_tokens(text_tok).float()
        x = x + te
        return self.in_norm(x).to(self.dtype)

    def backbone_hidden(self, x, attn_mask=None):
        out = self.backbone(inputs_embeds=x, attention_mask=attn_mask,
                            use_cache=False, output_hidden_states=False)
        return out.last_hidden_state.float()            # (B,T,d)

    def forward(self, h_user, prev_state, text_tok, attn_mask=None):
        x = self.fuse(h_user, prev_state, text_tok)
        z = self.backbone_hidden(x, attn_mask)
        return self.state_head(z), self.hazard_head(z).squeeze(-1)   # (B,T,4),(B,T)

    def trainable_parameters(self):
        ps = list(self.proj_h.parameters()) + list(self.state_emb.parameters()) + \
             list(self.in_norm.parameters()) + list(self.state_head.parameters()) + \
             list(self.hazard_head.parameters())
        if self.lora:
            ps += [p for p in self.backbone.parameters() if p.requires_grad]
        return ps

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())
    def n_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class YieldModel(nn.Module):
    def __init__(self, backbone_id=HEAD_BACKBONE, enc_hidden=ENC_HIDDEN, device="cuda",
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


class EndpointModel(YieldModel):
    def __init__(self, backbone_id=HEAD_BACKBONE, enc_hidden=ENC_HIDDEN, device="cuda",
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
