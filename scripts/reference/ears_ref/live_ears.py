"""exp-7 live-mic session, increment 1: the real-time EARS engine (git-main venv).

Streams audio through the nemotron encoder + the state/yield heads on the 80 ms frame clock, INCREMENTALLY
(rolling encode window + KV-cached head steps), emitting:
  - per-frame decision (LISTEN/HOLD/SPEAK/BACKCHANNEL) + yield probability,
  - ENDPOINT (LISTEN/HOLD -> SPEAK): the user turn ended -> RNNT ASR + L15 affect on the buffered turn,
  - BARGE (during SPEAK: yield head >= thr for `deb` frames): the user grabbed the floor.
This is the piece the offline pipeline faked (it precomputed ASR/affect/barge). All of EARS is one
git-main process; the MOUTH (4B + MOSS) is a separate venv_moss process wired over IPC (increment 2).

Audio source is abstracted (file/loopback now, mic/WebRTC later): feed() takes a hop of (user, agent)
16 kHz mono. `agent` is the assistant's own audio = the yield head's second ear (loopback in the live loop;
the other conversation channel when testing on yap).
"""
from __future__ import annotations
import time
import numpy as np, torch
from .encoder_stream import ENC_SR
from .model import LISTEN, HOLD, SPEAK, BACKCHANNEL, START, TEXT_PAD
from . import policy as fp
from .labeling import FRAME_S

WINDOW_S = 8.0            # rolling encode-context window
HOP_FRAMES = 4           # frames emitted per feed hop (320 ms)
STATE_NAMES = ["LISTEN", "HOLD", "SPEAK", "BACKCHANNEL"]


class LiveEars:
    def __init__(self, yield_ckpt, model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
                 device="cuda", theta=0.2, barge_thr=0.9, barge_deb=2, echo=True,
                 bc_thr=0.85, bc_deb=2, bc_refractory=15, partial_refresh_s=0.8):
        from transformers import AutoModel, AutoProcessor
        from .yield_head import YieldModel
        self.device = device; self.theta = theta; self.barge_thr = barge_thr; self.barge_deb = barge_deb
        self.bc_thr = bc_thr; self.bc_deb = bc_deb; self.bc_refractory = bc_refractory
        self.partial_refresh_s = partial_refresh_s
        self.echo = echo   # echo mode: no MOUTH -> after an endpoint, return to LISTEN (keep listening)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.asr = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        self.enc = self.asr.encoder
        self.ym = YieldModel(device=device); self.ym.load(yield_ckpt); self.ym.base.backbone.eval()
        self.reset()

    def reset(self):
        self.win_u = np.zeros(0, np.float32); self.win_a = np.zeros(0, np.float32)
        self.past = None; self.prev_state = LISTEN  # a mic session starts with the assistant listening
        self.turn_u = np.zeros(0, np.float32)        # user audio accumulated for ASR since last endpoint
        self.yield_run = 0
        self.voiced_ms = 0.0; self.sil_run = 0; self.in_turn = False   # endpointer state (echo mode)
        self.warm = 0                                # frames seen this turn (warmup guards a cold-start SPEAK lock)
        self._mode = "listen"                        # current feed mode (switches reset context + seed)
        self.t_frame_ms = []                         # per-frame head latency (real-time budget audit)
        self.bc_run = 0; self.bc_cooldown = 0        # backchannel debounce + refractory (LISTEN mode)
        self.partial_asr = ""; self.partial_at_s = 0.0   # rolling partial hypothesis (updated off the BC path)

    # endpoint tuning (echo mode): the LEARNED head (LISTEN/HOLD->SPEAK) fires the endpoint, guarded by
    # "user actually spoke" (min voiced) to reject the startup transient + empty ASR; a silence FALLBACK
    # catches the rare case the learned head doesn't flip.
    VAD_DB = -50.0; MIN_SPEECH_MS = 350.0; FALLBACK_SIL_FRAMES = 15  # 1.2 s hard fallback
    WARMUP_FRAMES = 6                                                # hold LISTEN ~0.5 s so a cold start can't lock SPEAK

    @torch.no_grad()
    def _encode_l15(self, wav):
        inp = self.proc(wav, sampling_rate=ENC_SR, return_tensors="pt")
        feats = inp["input_features"].to(self.device); am = inp.get("attention_mask")
        eo = self.enc(input_features=feats, attention_mask=(am.to(self.device) if am is not None else None),
                      num_lookahead_tokens=0, output_hidden_states=True)
        return eo.hidden_states[15][0].float()       # (T,1024)

    @torch.no_grad()
    def _frame_energy(self, wav_frames):
        hop = int(round(FRAME_S * ENC_SR)); n = len(wav_frames) // hop
        out = []
        for i in range(n):
            seg = wav_frames[i*hop:(i+1)*hop]
            out.append(20*np.log10(np.sqrt(np.mean(seg**2)+1e-12)+1e-9) if seg.size else -90.0)
        return np.array(out, np.float32)

    @torch.no_grad()
    def _head_step(self, hu_f, ha_f, act_f):
        """One frame through the heads (KV-cached). Returns (state_id, hazard, yield_prob, bc_prob)."""
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
        st = fp._decide(slog, haz, self.prev_state, self.theta, 0.5)
        return st, haz, yp, bcp

    @torch.no_grad()
    def _refresh_partial(self):
        """Update the rolling partial hypothesis from the 8s rolling window (bounded cost), THROTTLED so it
        never runs on the BC critical path -- a BC event just reads self.partial_asr, no fresh decode."""
        cur_s = len(self.turn_u) / ENC_SR
        if cur_s - self.partial_at_s < self.partial_refresh_s:
            return
        wav = self.win_u
        if wav.size < int(0.3 * ENC_SR):
            return
        out = self.asr.generate(**self.proc(wav, sampling_rate=ENC_SR, return_tensors="pt").to(self.device))
        self.partial_asr = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        self.partial_at_s = cur_s

    @torch.no_grad()
    def _asr_affect(self, wav):
        aff = self._encode_l15(wav).mean(0).cpu().numpy()               # L15 affect vector (mean-pool)
        out = self.asr.generate(**self.proc(wav, sampling_rate=ENC_SR, return_tensors="pt").to(self.device))
        txt = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        return txt, aff

    def feed(self, user_hop, agent_hop, mode="listen"):
        """Ingest one hop of (user, agent) 16 kHz mono. `mode`:
          'listen'   -- assistant idle: the learned turn-state head endpoints the user (LISTEN->SPEAK),
                        min-voiced-guarded + a silence fallback -> ENDPOINT (transcript + affect).
          'speaking' -- assistant is talking: seed prev=SPEAK, run the two-stream YIELD head over
                        (user mic, assistant audio as the 2nd ear) -> BARGE when the user takes the floor."""
        events = []
        self.win_u = np.concatenate([self.win_u, user_hop])[-int(WINDOW_S * ENC_SR):]
        self.win_a = np.concatenate([self.win_a, agent_hop])[-int(WINDOW_S * ENC_SR):]
        if mode == "listen":
            self.turn_u = np.concatenate([self.turn_u, user_hop])[-int(20 * ENC_SR):]
        hu = self._encode_l15(self.win_u).cpu().numpy()
        ha = self._encode_l15(self.win_a).cpu().numpy()
        act = self._frame_energy(self.win_u)
        if mode == "listen":
            self._refresh_partial()                                     # keep the rolling hypothesis fresh (throttled)
        if mode != self._mode:                                          # mode switch: fresh context + seed
            self.past = None; self.warm = 0; self.yield_run = 0
            self.bc_run = 0; self.bc_cooldown = 0
            self.prev_state = SPEAK if mode == "speaking" else LISTEN
            self._mode = mode
        n = min(HOP_FRAMES, len(hu), len(ha), len(act))                 # emit the newest n frames
        for i in range(-n, 0):
            t0 = time.perf_counter()
            st, haz, yp, bcp = self._head_step(hu[i], ha[i], act[i])
            self.t_frame_ms.append((time.perf_counter() - t0) * 1e3)
            prev = self.prev_state
            voiced = act[i] > self.VAD_DB
            if mode == "speaking":
                # BARGE: the two-stream yield head fires when the user grabs the floor over the assistant.
                self.warm += 1
                if self.warm > self.WARMUP_FRAMES:
                    self.yield_run = self.yield_run + 1 if yp >= self.barge_thr else 0
                    if self.yield_run >= self.barge_deb:
                        events.append(dict(kind="BARGE", yield_prob=round(yp, 3)))
                        self.yield_run = 0
                events.append(dict(kind="FRAME", state="SPEAK", yield_prob=round(yp, 3), hazard=round(haz, 3)))
                self.prev_state = SPEAK                                 # hold the floor until barge/done
                continue
            # LISTEN mode: learned endpointer (min-voiced-guarded) + silence fallback
            self.warm += 1
            if self.warm <= self.WARMUP_FRAMES:                         # cold-start: don't let a transient lock SPEAK
                st = LISTEN; self.prev_state = LISTEN
            if voiced:
                self.voiced_ms += FRAME_S * 1e3; self.sil_run = 0; self.in_turn = True
            else:
                self.sil_run += 1
            # BACKCHANNEL (floor-preserving): the bc head fires a short ack cue while the user keeps the
            # floor. Debounced + refractory, only once the user is actually in a turn. Does NOT flip state,
            # does NOT reset turn_u -> the real endpoint still fires later. prev_state stays LISTEN.
            if self.bc_cooldown > 0:
                self.bc_cooldown -= 1; self.bc_run = 0
            elif self.in_turn and self.voiced_ms >= self.MIN_SPEECH_MS and self.warm > self.WARMUP_FRAMES:
                self.bc_run = self.bc_run + 1 if bcp >= self.bc_thr else 0
                if self.bc_run >= self.bc_deb:
                    # affect for the ack = mean L15 over the VOICED frames in the window (the user's actual
                    # speech), NOT the raw recent window -- early in a turn that window is mostly onset
                    # silence, whose L15 mean is a degenerate/OOD vector that collapses the ack to "Hmm".
                    mL = min(len(hu), len(act))                        # encoder vs energy can differ by 1 frame
                    vmask = act[:mL] > self.VAD_DB
                    aff = hu[:mL][vmask].mean(0) if vmask.any() else hu[:mL].mean(0)
                    events.append(dict(kind="BACKCHANNEL", partial=self.partial_asr, affect=aff,
                                       bc_prob=round(bcp, 3), turn_s=round(len(self.turn_u) / ENC_SR, 2)))
                    self.bc_run = 0; self.bc_cooldown = self.bc_refractory
            spoke = self.in_turn and self.voiced_ms >= self.MIN_SPEECH_MS
            learned_end = prev in (LISTEN, HOLD) and st == SPEAK
            fallback_end = self.sil_run >= self.FALLBACK_SIL_FRAMES
            if spoke and (learned_end or fallback_end):
                txt, aff = self._asr_affect(self.turn_u)
                events.append(dict(kind="ENDPOINT", transcript=txt, affect=aff, turn_s=len(self.turn_u)/ENC_SR,
                                   by="head" if learned_end else "silence"))
                self.turn_u = np.zeros(0, np.float32); self.voiced_ms = 0.0; self.in_turn = False
                self.past = None; self.prev_state = LISTEN; self.warm = 0
                self.bc_run = 0; self.bc_cooldown = 0; self.partial_asr = ""; self.partial_at_s = 0.0
                events.append(dict(kind="FRAME", state="idle", yield_prob=round(yp, 3), hazard=round(haz, 3),
                                   bc_prob=round(bcp, 3)))
                continue
            disp = STATE_NAMES[st] if self.in_turn else ("idle" if not voiced else "LISTEN")
            events.append(dict(kind="FRAME", state=disp, yield_prob=round(yp, 3), hazard=round(haz, 3),
                               bc_prob=round(bcp, 3)))
            self.prev_state = st
        return events

    def latency_summary(self):
        v = np.array(self.t_frame_ms) if self.t_frame_ms else np.zeros(1)
        return dict(head_ms_p50=round(float(np.percentile(v, 50)), 1),
                    head_ms_p90=round(float(np.percentile(v, 90)), 1),
                    head_ms_max=round(float(v.max()), 1), n_frames=len(self.t_frame_ms))
