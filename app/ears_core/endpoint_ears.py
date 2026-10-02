"""exp-12 arm C live EARS: exp-7's LiveEars with the LISTEN->SPEAK endpoint decided by the dedicated
endpoint head (p_end >= tau for `deb` frames), NOT the 4-way state flip. Subclasses exp-7 LiveEars
(UNTOUCHED); only the model (EndpointModel with ep_head) and the listen-mode `learned_end` change.

Differences from LiveEars (spec §4):
  * self.ym is an EndpointModel (has ep_head); _head_step also returns p_end.
  * listen mode: `learned_end = p_end >= ep_thr for ep_deb consecutive frames`, after the same `spoke`
    gate (in-turn, voiced >= 350 ms) and warmup; it REPLACES `prev in {LISTEN,HOLD} and st == SPEAK`.
  * prev_state is held at LISTEN throughout listen mode (the ep head's free-run operating regime, matching
    the offline selection), so the 4-way `st` is computed for LOGGING only.
  * every FRAME event carries `p_end`; the silence fallback and barge/bc heads are unchanged.
"""
from __future__ import annotations
import time
import numpy as np, torch
from .live_ears import LiveEars, HOP_FRAMES, WINDOW_S, STATE_NAMES
from .encoder_stream import ENC_SR
from .labeling import FRAME_S
from .model import LISTEN, HOLD, SPEAK
from .endpoint_head import EndpointModel


class EndpointEars(LiveEars):
    def __init__(self, ep_ckpt, ep_thr=0.9146, ep_deb=2, **kw):
        super().__init__(ep_ckpt, **kw)                      # loads ASR + a YieldModel from ep_ckpt (ep_head ignored)
        self.ep_thr = ep_thr; self.ep_deb = ep_deb; self.ep_run = 0
        self.ym = EndpointModel(device=self.device); self.ym.load(ep_ckpt); self.ym.base.backbone.eval()  # swap in ep_head
        print(f"    [arm C] endpoint head: ep_thr={ep_thr} ep_deb={ep_deb}", flush=True)

    def reset(self):
        super().reset(); self.ep_run = 0

    @torch.no_grad()
    def _head_step(self, hu_f, ha_f, act_f):
        """One frame through the heads (KV-cached). Returns (state_id, hazard, yield_prob, bc_prob, p_end)."""
        from . import policy as fp
        m = self.ym
        hu = torch.from_numpy(hu_f[None, None]).to(self.device)
        ha = torch.from_numpy(ha_f[None, None]).to(self.device)
        act = torch.tensor([[np.clip((act_f + 50) / 25, -3, 3)]], dtype=torch.float32, device=self.device)
        prev = torch.tensor([[self.prev_state]], dtype=torch.long, device=self.device)
        x = m.fuse2(hu, ha, act, prev)
        out = m.base.backbone(inputs_embeds=x, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        z = out.last_hidden_state.float()
        slog = m.base.state_head(z)[0, 0].cpu().numpy()
        haz = torch.sigmoid(m.base.hazard_head(z)[0, 0, 0]).item()
        yp = torch.sigmoid(m.yield_head(z)[0, 0, 0]).item()
        bcp = torch.sigmoid(m.bc_head(z)[0, 0, 0]).item()
        pep = torch.sigmoid(m.ep_head(z)[0, 0, 0]).item()
        st = fp._decide(slog, haz, self.prev_state, self.theta, 0.5)     # 4-way state: LOGGED ONLY for arm C
        return st, haz, yp, bcp, pep

    def feed(self, user_hop, agent_hop, mode="listen"):
        """As LiveEars.feed, but listen-mode endpoint = p_end>=ep_thr for ep_deb frames (see module doc)."""
        events = []
        self.win_u = np.concatenate([self.win_u, user_hop])[-int(WINDOW_S * ENC_SR):]
        self.win_a = np.concatenate([self.win_a, agent_hop])[-int(WINDOW_S * ENC_SR):]
        if mode == "listen":
            self.turn_u = np.concatenate([self.turn_u, user_hop])[-int(20 * ENC_SR):]
        hu = self._encode_l15(self.win_u).cpu().numpy()
        ha = self._encode_l15(self.win_a).cpu().numpy()
        act = self._frame_energy(self.win_u)
        if mode == "listen":
            self._refresh_partial()
        if mode != self._mode:
            self.past = None; self.warm = 0; self.yield_run = 0
            self.bc_run = 0; self.bc_cooldown = 0; self.ep_run = 0
            self.prev_state = SPEAK if mode == "speaking" else LISTEN
            self._mode = mode
        n = min(HOP_FRAMES, len(hu), len(ha), len(act))
        for i in range(-n, 0):
            t0 = time.perf_counter()
            st, haz, yp, bcp, pep = self._head_step(hu[i], ha[i], act[i])
            self.t_frame_ms.append((time.perf_counter() - t0) * 1e3)
            voiced = act[i] > self.VAD_DB
            if mode == "speaking":                              # BARGE path unchanged (yield head)
                self.warm += 1
                if self.warm > self.WARMUP_FRAMES:
                    self.yield_run = self.yield_run + 1 if yp >= self.barge_thr else 0
                    if self.yield_run >= self.barge_deb:
                        events.append(dict(kind="BARGE", yield_prob=round(yp, 3)))
                        self.yield_run = 0
                events.append(dict(kind="FRAME", state="SPEAK", yield_prob=round(yp, 3), hazard=round(haz, 3),
                                   p_end=round(pep, 3)))
                self.prev_state = SPEAK
                continue
            # LISTEN mode
            self.warm += 1
            if voiced:
                self.voiced_ms += FRAME_S * 1e3; self.sil_run = 0; self.in_turn = True
            else:
                self.sil_run += 1
            # BACKCHANNEL (floor-preserving) -- unchanged
            if self.bc_cooldown > 0:
                self.bc_cooldown -= 1; self.bc_run = 0
            elif self.in_turn and self.voiced_ms >= self.MIN_SPEECH_MS and self.warm > self.WARMUP_FRAMES:
                self.bc_run = self.bc_run + 1 if bcp >= self.bc_thr else 0
                if self.bc_run >= self.bc_deb:
                    mL = min(len(hu), len(act)); vmask = act[:mL] > self.VAD_DB
                    aff = hu[:mL][vmask].mean(0) if vmask.any() else hu[:mL].mean(0)
                    events.append(dict(kind="BACKCHANNEL", partial=self.partial_asr, affect=aff,
                                       bc_prob=round(bcp, 3), turn_s=round(len(self.turn_u) / ENC_SR, 2)))
                    self.bc_run = 0; self.bc_cooldown = self.bc_refractory
            # ENDPOINT: the dedicated head (p_end>=tau for deb frames) after the spoke gate; + silence fallback
            if self.warm > self.WARMUP_FRAMES:
                self.ep_run = self.ep_run + 1 if pep >= self.ep_thr else 0
            else:
                self.ep_run = 0
            spoke = self.in_turn and self.voiced_ms >= self.MIN_SPEECH_MS
            learned_end = self.ep_run >= self.ep_deb
            fallback_end = self.sil_run >= self.FALLBACK_SIL_FRAMES
            if spoke and (learned_end or fallback_end):
                txt, aff = self._asr_affect(self.turn_u)
                events.append(dict(kind="ENDPOINT", transcript=txt, affect=aff, turn_s=len(self.turn_u) / ENC_SR,
                                   by="head" if learned_end else "silence", p_end=round(pep, 3)))
                self.turn_u = np.zeros(0, np.float32); self.voiced_ms = 0.0; self.in_turn = False
                self.past = None; self.prev_state = LISTEN; self.warm = 0
                self.bc_run = 0; self.bc_cooldown = 0; self.ep_run = 0
                self.partial_asr = ""; self.partial_at_s = 0.0
                events.append(dict(kind="FRAME", state="idle", yield_prob=round(yp, 3), hazard=round(haz, 3),
                                   bc_prob=round(bcp, 3), p_end=round(pep, 3)))
                continue
            disp = STATE_NAMES[st] if self.in_turn else ("idle" if not voiced else "LISTEN")
            events.append(dict(kind="FRAME", state=disp, yield_prob=round(yp, 3), hazard=round(haz, 3),
                               bc_prob=round(bcp, 3), p_end=round(pep, 3)))
            self.prev_state = LISTEN                            # free-run: the ep head's operating regime (NOT st)
        return events
