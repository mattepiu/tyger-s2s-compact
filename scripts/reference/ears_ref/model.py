"""exp-4 state-channel model: frozen encoder h_t + agent state/text history -> frozen Qwen
backbone -> z_s -> trainable k=0 state head (LISTEN/HOLD/SPEAK/BACKCHANNEL) + scalar hazard head.

Reconciled with exp-2/3 (arch doc §1/§3): the backbone predicts ONLY the low-entropy, text-native
channels -- it does NOT generate audio codec tokens (exp-2 closed that). Here it is a FROZEN
sequence backbone over per-frame FUSED embeddings; the only trainable params are a small input
projection, a 5-way state embedding, and the two heads (H4: 'small trainable heads', no full FT).
Backbone weights frozen (grad flows to the input projection); LoRA fallback = A1-LoRA.

Frame grid = 12.5 Hz. Per frame s the backbone input is additive (arch §3, 'h_t projected, additive'):
    x_s = Wh(h_t^user_s) + E_state(prev_state_s) + Etext(agent_text_tok_s)
Etext is the backbone's own (frozen) token embedding -- keeps the k=1 inner-monologue in the
frozen-LLM's native space. Causal backbone -> last_hidden_state z_s -> heads.
"""
from __future__ import annotations
import torch, torch.nn as nn

# state vocabulary (k=0)
LISTEN, HOLD, SPEAK, BACKCHANNEL = 0, 1, 2, 3
STATE_NAMES = ["LISTEN", "HOLD", "SPEAK", "BACKCHANNEL"]
N_STATE = 4
START = 4                      # prev-state BOS for the first frame
TEXT_PAD = 0                   # frame with no active agent word (uses token id 0 region)

class StateChannelModel(nn.Module):
    def __init__(self, backbone_id="Qwen/Qwen3-0.6B", enc_hidden=1024, device="cuda",
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


class StateEndpointModel(StateChannelModel):
    """exp-9 Stage 2a (§13): the state-channel model + an explicit ENDPOINT head (+ optional user-energy input).

    endpoint_head: the transition-detector recipe of exp-7's yield/bc heads -- Linear(d,d)-SiLU-Linear(d,1), LAST layer
    ZERO-init (p_end = 0.5 everywhere at start, the state/hazard heads keep training as auxiliary losses).
    proj_act (energy=True, arm A): the user frame energy (exp-7 norm_act) through a ZERO-init Linear(1, d), added to the
    fused input exactly as exp7.yield_head.YieldModel.fuse2 adds it (additive identity at start). energy=False = arm B.
    Backbone frozen (no LoRA), as the G1' recipe."""
    SAVE_KEYS = ("proj_h.", "state_emb.", "in_norm.", "state_head.", "hazard_head.", "endpoint_head.", "proj_act.")

    def __init__(self, energy=True, **kw):
        super().__init__(**kw)
        d = self.d_model
        self.energy = bool(energy)
        self.endpoint_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.endpoint_head[-1].weight); nn.init.zeros_(self.endpoint_head[-1].bias)
        mods = [self.endpoint_head]
        if self.energy:
            self.proj_act = nn.Linear(1, d)
            nn.init.zeros_(self.proj_act.weight); nn.init.zeros_(self.proj_act.bias)
            mods.append(self.proj_act)
        for m in mods:
            m.to(self.device).float()

    def fuse(self, h_user, prev_state, text_tok, act=None):
        x = self.proj_h(h_user.float()) + self.state_emb(prev_state)
        with torch.no_grad():
            te = self.embed_tokens(text_tok).float()
        x = x + te
        if self.energy:
            x = x + self.proj_act(act.float().unsqueeze(-1))
        return self.in_norm(x).to(self.dtype)

    def forward(self, h_user, prev_state, text_tok, act=None, attn_mask=None):
        z = self.backbone_hidden(self.fuse(h_user, prev_state, text_tok, act), attn_mask)
        return self.state_head(z), self.hazard_head(z).squeeze(-1), self.endpoint_head(z).squeeze(-1)

    def trainable_parameters(self):
        ps = super().trainable_parameters() + list(self.endpoint_head.parameters())
        return ps + (list(self.proj_act.parameters()) if self.energy else [])

    def head_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(self.SAVE_KEYS)}
