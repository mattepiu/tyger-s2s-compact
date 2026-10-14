"""ReferenceEars: one conversation through the tyger bundle, computed the plain way, for scripts/validate/ears_batched.py
to check the live engine against. Same bundle, same turn rules (rules.py); every step the engine speeds up is done here
without the speed-up:
  * features: the HF processor on the CPU (the engine computes them on the GPU);
  * encoder: the full nemotron forward pass with output_hidden_states, layer 15 taken from it (the engine stops at
    layer 15 with a hook, batches windows and caches all-zero ones);
  * the turn-taker: its GRU stepped one frame at a time for this one conversation, as the research runtime's
    StreamSession steps it (the engine: GruBatch, a state row per conversation, all of them stepped together);
  * transcripts: nemotron's HF generate() on the whole turn (the engine: its graphed greedy RNN-T, on the ASR thread,
    long turns in segments). No backchannel partials (they don't change events).
Audio is held in memory only.
"""
from __future__ import annotations

import torch

from app.plugins.ears.tyger import rules
from app.plugins.ears.tyger.bundle import load
from app.plugins.ears.tyger.constants import HOP_FRAMES, L15, SR


class ReferenceEars:
    def __init__(self, bundle_dir, device="cuda"):
        nemotron, self.proc, turn_taker, _ = load(bundle_dir)
        self.device = device
        self.asr = nemotron.to(device); self.turn_taker = turn_taker.to(device)
        self.cache = None                      # the GRU's state for the conversation's current context

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
        """One frame through the turn-taker, its state carried over: -> the 8 outputs (numpy)."""
        act = ((torch.tensor([[float(act_db)]], device=self.device) + 50) / 25).clamp(-3, 3)
        prev = torch.tensor([[prev_state]], dtype=torch.long, device=self.device)
        out, self.cache = self.turn_taker(hu[None, None], ha[None, None], act, prev, self.cache)
        return out[0, 0].cpu().numpy()

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
            evs, final, reset = rules.on_frame(sess, mode, out, float(act[i]), left=-i - 1)
            if final is not None:
                ev, wav = final
                ev.pop("seg_texts", None)
                ev["transcript"] = self.transcribe(wav)
            events += evs
            if reset:
                self.cache = None
        return events

