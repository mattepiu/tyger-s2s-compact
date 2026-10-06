"""TygerHeads: the loader and inference model for the TYGER-S2S-endpointer release (the tyger 1.0 heads, exp-12 arm
C; private repo `ears.heads`, weights only: its model card says the loader lives here).

Per frame the heads read nemotron-3.5-asr's layer-15 features of the user's audio and of the agent's own audio (the
2nd ear), the user's normalised frame energy and the previous turn state. These are fused additively into one input
per frame (`fuse`), run through the Qwen3-0.6B decoder (its LoRA already merged in the release), and the last hidden
state feeds five heads (`outputs`): the 4-way state logits, and the hazard, yield (barge), backchannel and endpoint
probabilities.

The release is self-describing (config.json): the backbone's config is embedded (nothing else is downloaded), the
dtype policy is bf16 backbone / fp32 added modules ("keep the stored dtypes"), and `frontend` pins the encoder the
heads were trained on (repo, revision, layer, window, hop, energy norm). The loader checks that frontend against
constants.py, so a release that needs a different audio grid fails at load instead of misbehaving.
scripts/validate/heads.py checks the release as loaded here (on real speech features); when it was introduced (R-06b)
it was bit-identical to the research model it was exported from (NOTES.md).
"""
from __future__ import annotations
import json
from pathlib import Path

import torch, torch.nn as nn

from .constants import ENC_HIDDEN, FRAME_S, HOP_N, L15, N_STATE, SR, TEXT_PAD, WINDOW_S

MODEL_TYPE = "tyger_endpointer"
ADDED_MODULES = ("proj_h", "proj_agent", "proj_act", "state_emb", "in_norm", "state_head", "hazard_head",
                 "yield_head", "bc_head", "ep_head")
_NOT_QWEN3_CONFIG = ("architectures", "torch_dtype", "transformers_version")


def _mlp_head(d):
    return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))


def check_frontend(frontend: dict):
    """The release's pinned frontend must be the audio grid this plugin runs (constants.py); ValueError if not."""
    want = {"layer": L15, "sample_rate": SR, "frame_period_s": FRAME_S, "window_s": WINDOW_S,
            "hop_s": HOP_N / SR, "num_lookahead_tokens": 0, "energy_norm": "clip((dB + 50) / 25, -3, 3)"}
    bad = [f"{k}: release {frontend.get(k)!r}, plugin {v!r}" for k, v in want.items() if frontend.get(k) != v]
    if bad:
        raise ValueError("the heads release needs another audio frontend than this plugin runs: " + "; ".join(bad))


class TygerHeads(nn.Module):
    def __init__(self, config: dict):
        """The model's modules from the release's config.json (random weights: use from_pretrained)."""
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model
        if config.get("model_type") != MODEL_TYPE:
            raise ValueError(f"not a {MODEL_TYPE} release (model_type {config.get('model_type')!r})")
        check_frontend(config["frontend"])
        self.config = config
        self.dtype = getattr(torch, config["dtype_policy"]["backbone"])
        bb = Qwen3Config(**{k: v for k, v in config["backbone_config"].items() if k not in _NOT_QWEN3_CONFIG})
        # as from_pretrained builds it: parameters in the backbone dtype, rotary buffers in fp32
        self.backbone = Qwen3Model._from_config(bb, dtype=self.dtype, attn_implementation=config["attn_implementation"])
        d, enc = config["hidden_size"], config["encoder_hidden_size"]
        assert enc == ENC_HIDDEN and config["num_states"] == N_STATE and config["text_pad_token_id"] == TEXT_PAD
        self.proj_h = nn.Linear(enc, d)                    # the user's layer-15 features
        self.proj_agent = nn.Linear(enc, d)                # the agent's (the 2nd ear)
        self.proj_act = nn.Linear(1, d)                    # the user's frame energy
        self.state_emb = nn.Embedding(config["num_state_embeddings"], d)   # the previous state (+ START)
        self.in_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, config["num_states"])
        self.hazard_head = nn.Linear(d, 1)
        self.yield_head = _mlp_head(d)                     # the user is barging in: yield now (speaking)
        self.bc_head = _mlp_head(d)                        # the user gave a cue: backchannel now (listening)
        self.ep_head = _mlp_head(d)                        # the user's turn ended: take the floor (listening)

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        """Load a release from a local directory (config.json + model.safetensors): every tensor, in its stored
        dtype. Missing or unexpected tensors are an error."""
        from safetensors.torch import load_file
        path = Path(path)
        model = cls(json.loads((path / "config.json").read_text()))
        added = getattr(torch, model.config["dtype_policy"]["added_modules"])
        for name in ADDED_MODULES:
            getattr(model, name).to(added)
        model.to(device)
        state = load_file(str(path / "model.safetensors"), device=str(device))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"heads release {path}: {len(missing)} tensors missing, {len(unexpected)} unexpected")
        wrong = [k for k, v in model.state_dict().items() if k in state and v.dtype != state[k].dtype]
        if wrong:
            raise ValueError(f"heads release {path}: stored dtypes changed on load ({', '.join(wrong[:3])})")
        model.eval()
        # the agent-text input is always TEXT_PAD live (no agent words are fed): its embedding, once
        model.register_buffer("text_pad", model.backbone.embed_tokens.weight[TEXT_PAD].float().clone(),
                              persistent=False)
        return model

    @property
    def frontend(self) -> dict:
        """The encoder the heads were trained on: repo_id, revision, dtype, layer, ..."""
        return self.config["frontend"]

    def fuse(self, h_user, h_agent, act, prev_state):
        """One input per frame: (B, T, 1024) user and agent features, (B, T) normalised energy, (B, T) previous
        state -> (B, T, d) in the backbone's dtype. The same sums, in the same order, as the research fuse2."""
        x = self.proj_h(h_user.float()) + self.state_emb(prev_state)
        x = x + self.text_pad + self.proj_agent(h_agent.float()) + self.proj_act(act.float().unsqueeze(-1))
        return self.in_norm(x).to(self.dtype)

    def outputs(self, z):
        """The backbone's last hidden state (B, T, d) -> (B, T, 8): state logits x4, then the hazard, yield,
        backchannel and endpoint probabilities."""
        z = z.float()
        return torch.cat([self.state_head(z), torch.sigmoid(self.hazard_head(z)), torch.sigmoid(self.yield_head(z)),
                          torch.sigmoid(self.bc_head(z)), torch.sigmoid(self.ep_head(z))], -1)
