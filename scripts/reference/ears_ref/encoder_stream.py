"""Per-frame h_t stream from the frozen nemotron-3.5 streaming-ASR encoder (the user 'ears').

Unlike exp-3/affect.py (which mean-POOLS a whole utterance to one affect vector), exp-4 needs the
PER-FRAME hidden sequence at ~12.5 Hz (80 ms) -- the user-stream features the turn-state/hazard
heads read (speech activity + pre-boundary prosody). Frozen (exp-0 gate); layer-weighted mix over
an affect/turn-bearing band (init L9-16, exp-0). Lookahead R (num_lookahead_tokens) sweeps the
causality/latency frontier (H3); the zero-lookahead [*,0] mode matches streaming deployment.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
ENC_SR = 16000
ENC_HIDDEN = 1024
FRAME_HZ = 12.5          # measured (script 10): 80ms/frame asymptotically

class EncoderStream(nn.Module):
    """Frozen encoder -> (T, 1024) per-frame features. Layer weights are a tiny trainable mix
    over a chosen band (default a single layer = exp-0's L15)."""
    def __init__(self, model_id=MODEL, layers=(15,), device="cuda", lookahead=0,
                 learn_layer_mix=False, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        self.device = device; self.lookahead = lookahead; self.layers = tuple(layers)
        self.proc = AutoProcessor.from_pretrained(model_id)
        full = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.encoder = full.encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        del full
        self.hidden = ENC_HIDDEN
        if learn_layer_mix and len(self.layers) > 1:
            self.layer_logits = nn.Parameter(torch.zeros(len(self.layers)))
        else:
            self.register_parameter("layer_logits", None)

    @torch.no_grad()
    def _feats(self, wavs16, lookahead=None):
        R = self.lookahead if lookahead is None else lookahead
        inp = self.proc(wavs16, sampling_rate=ENC_SR, return_tensors="pt", padding=True)
        feats = inp["input_features"].to(self.device)
        amask = inp["attention_mask"].to(self.device)
        eo = self.encoder(input_features=feats, attention_mask=amask,
                          num_lookahead_tokens=R, output_hidden_states=True)
        hs = eo.hidden_states
        fmask = getattr(eo, "attention_mask", None)
        return hs, fmask

    @torch.no_grad()
    def _stream_one(self, wav16: np.ndarray, lookahead=None) -> torch.Tensor:
        hs, _ = self._feats([np.asarray(wav16, dtype=np.float32)], lookahead)
        sel = torch.stack([hs[i][0] for i in self.layers], 0)   # (L,T,D)
        if self.layer_logits is not None:
            w = torch.softmax(self.layer_logits, 0).view(-1, 1, 1)
            return (sel * w).sum(0)
        return sel.mean(0)

    @torch.no_grad()
    def stream(self, wav16: np.ndarray, lookahead=None, chunk_s: float = 120.0) -> torch.Tensor:
        """mono float32 @16k -> (T, 1024) frozen per-frame features (on device).
        Long audio is processed in chunks to bound VRAM (output_hidden_states stores all 25
        layers); the cache-aware encoder has limited right-context so chunking ~matches streaming.
        Features are moved to CPU to keep long-conversation memory flat."""
        wav16 = np.asarray(wav16, dtype=np.float32)
        n = len(wav16); step = int(chunk_s * ENC_SR)
        if n <= step:
            return self._stream_one(wav16, lookahead)
        outs = []
        for s in range(0, n, step):
            h = self._stream_one(wav16[s:s + step], lookahead)
            outs.append(h.cpu())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def stream_batch(self, wavs16: list, lookahead=None):
        """list of mono@16k -> (feats list of (T_i,D), lengths)."""
        hs, fmask = self._feats([np.asarray(w, dtype=np.float32) for w in wavs16], lookahead)
        sel = torch.stack([hs[i] for i in self.layers], 0)      # (L,B,T,D)
        if self.layer_logits is not None:
            w = torch.softmax(self.layer_logits, 0).view(-1, 1, 1, 1)
            mix = (sel * w).sum(0)
        else:
            mix = sel.mean(0)                                    # (B,T,D)
        outs, lens = [], []
        for b in range(mix.shape[0]):
            if fmask is not None and fmask.shape[1] == mix.shape[1]:
                T = int(fmask[b].sum())
            else:
                T = mix.shape[1]
            outs.append(mix[b, :T]); lens.append(T)
        return outs, lens
