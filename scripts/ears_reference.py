"""ReferenceEars: one conversation through the TYGER-S2S-endpointer release, computed the plain way, for
scripts/validate_ears_batched.py to check the live engine against. Same release, same turn rules (rules.py); every
step the engine speeds up is done here without the speed-up:
  * features: the HF processor on the CPU (the engine computes them on the GPU);
  * encoder: the full nemotron forward pass with output_hidden_states, layer 15 taken from it (the engine stops at
    layer 15 with a hook, batches windows and caches all-zero ones);
  * heads: TygerHeads stepped one frame at a time with HF's DynamicCache (the engine: a KV ring slot per conversation,
    batched across conversations, CUDA graphs);
  * transcripts: nemotron's HF generate() on the whole turn (the engine: its graphed greedy RNN-T, on the ASR thread,
    long turns in segments). No backchannel partials (they don't change events).
Audio is held in memory only.
"""
from __future__ import annotations

import numpy as np, torch

from app.plugins.ears.nemotron_tyger import rules
from app.plugins.ears.nemotron_tyger.constants import HOP_FRAMES, L15, SR
from app.plugins.ears.nemotron_tyger.heads import TygerHeads


class ReferenceEars:
    def __init__(self, heads_dir, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.heads = TygerHeads.from_pretrained(heads_dir, device=device)
        fe = self.heads.frontend
        self.proc = AutoProcessor.from_pretrained(fe["repo_id"], revision=fe["revision"])
        self.asr = AutoModel.from_pretrained(fe["repo_id"], revision=fe["revision"], dtype=torch.float32).to(device).eval()
        self.cache = None                      # the heads' KV cache for the conversation's current context

    @torch.no_grad()
    def encode_l15(self, wav):
        inp = self.proc(wav, sampling_rate=SR, return_tensors="pt")
        am = inp.get("attention_mask")
        eo = self.asr.encoder(input_features=inp["input_features"].to(self.device),
                              attention_mask=am.to(self.device) if am is not None else None,
                              num_lookahead_tokens=0, output_hidden_states=True)
        return eo.hidden_states[L15][0].float()

    @torch.no_grad()
    def head_step(self, hu, ha, act_db, prev_state):
        """One frame through the heads, the cache carried over: -> the 8 outputs (numpy)."""
        h = self.heads
        act = torch.tensor([[float(np.clip((act_db + 50) / 25, -3, 3))]], dtype=torch.float32, device=self.device)
        prev = torch.tensor([[prev_state]], dtype=torch.long, device=self.device)
        out = h.backbone(inputs_embeds=h.fuse(hu[None, None], ha[None, None], act, prev), past_key_values=self.cache,
                         use_cache=True)
        self.cache = out.past_key_values
        return h.outputs(out.last_hidden_state)[0, 0].cpu().numpy()

    @torch.no_grad()
    def transcribe(self, wav):
        out = self.asr.generate(**self.proc(wav, sampling_rate=SR, return_tensors="pt").to(self.device))
        return self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    def feed(self, sess, user_hop, agent_hop, mode):
        """One hop for one session (an EarsSession: the rules' state) -> its events, transcripts filled in."""
        if rules.begin_hop(sess, user_hop, agent_hop, mode):
            self.cache = None
        hu, ha = self.encode_l15(sess.win_u), self.encode_l15(sess.win_a)
        act = rules.frame_energy(sess.win_u)
        if rules.switch_mode(sess, mode):
            self.cache = None
        events = []
        for i in range(-min(HOP_FRAMES, len(hu), len(ha), len(act)), 0):
            out = self.head_step(hu[i], ha[i], act[i], sess.prev_state)
            evs, final, reset = rules.on_frame(sess, mode, out, float(act[i]))
            if final is not None:
                ev, wav = final
                ev.pop("seg_texts", None)
                ev["transcript"] = self.transcribe(wav)
            events += evs
            if reset:
                self.cache = None
        return events
