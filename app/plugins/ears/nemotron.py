"""nemotron-tyger-1.0: the nemotron streaming ASR encoder (nvidia/nemotron-3.5-asr-streaming-0.6b) + the tyger 1.0
turn-taking heads (exp-12 arm C: LoRA + state / hazard / yield / backchannel / endpoint heads; private weights,
`heads_repo` / `heads_file`). Wraps EarsEngine + EarsService (app/ears_core/batched.py, unchanged): one GPU thread,
one batched step per 320 ms hop for every conversation with a hop ready.

Per hop, feed() returns FRAME / ENDPOINT / BARGE / BACKCHANNEL events. A session is an EarsSession: reset() starts
a new turn (applied before its next hop); its thresholds are per conversation (the page's live-tuning sliders).
"""
from __future__ import annotations
import logging

from app.ears_core.labeling import FRAME_S                  # one decision frame: the fallback slider's unit
from app.plugins.base import Slider, Threshold, find_threshold

log = logging.getLogger("ears")

# per-conversation thresholds; defaults from the `ears:` config block
THRESHOLDS = (
    Threshold("ep_thr", float, slider=Slider("endpoint p_end τ", 0.30, 0.99, 0.005, decimals=3, meter="p_end")),
    Threshold("ep_deb", int, slider=Slider("endpoint deb", 1, 5, 1, decimals=0)),
    Threshold("barge_thr", float, slider=Slider("barge (yield)", 0.20, 1.00, 0.01, meter="yield")),
    Threshold("barge_deb", int),
    Threshold("bc_thr", float, slider=Slider("backchannel", 0.20, 1.00, 0.01, meter="bc")),
    Threshold("bc_deb", int),
    Threshold("fallback_frames", int, config_key="fallback_sil_frames", default=15,
              slider=Slider("silence fallback", 0.4, 2.5, 0.1, decimals=1, scale=FRAME_S, unit="s", off=100000)),
)
_ATTR = {"fallback_frames": "FALLBACK_SIL_FRAMES"}           # EarsSession's attribute name, where it differs


class NemotronTyger:
    THRESHOLDS = THRESHOLDS

    def __init__(self, conf: dict, ctx):
        import torch
        from app import weights
        from app.ears_core.batched import EarsEngine, EarsService, HOP_N, MODEL_ID
        from app.ears_core.encoder_stream import ENC_SR
        torch.set_num_threads(int(conf.get("cpu_threads", 4)))   # default 48 threads mostly spin-wait
        self.sr = ENC_SR
        self.hop_samples = HOP_N                                 # 320 ms @ 16 kHz = 5120
        self.defaults = {t.name: t.from_config(conf) for t in THRESHOLDS}
        heads = weights.ensure_heads(conf["heads_repo"], conf["heads_file"])
        self.slots = int(ctx.cfg["server"].get("max_sessions", 4))
        model_id = conf.get("nemotron") or MODEL_ID
        log.info("loading EARS (nemotron-tyger-1.0: %s) for %d sessions, ep_thr=%s ep_deb=%s", model_id, self.slots,
                 self.defaults["ep_thr"], self.defaults["ep_deb"])
        engine = EarsEngine(heads, slots=self.slots, ring_len=int(conf.get("ring_frames", 2048)), model_id=model_id)
        self.service = EarsService(engine, gather_s=float(conf.get("gather_ms", 15)) / 1e3,
                                   expect_s=float(conf.get("expect_ms", 60)) / 1e3,
                                   stats_every_s=float(conf.get("stats_s", 60)))

    def open(self, **thresholds):
        """A new conversation's session with the configured thresholds (overridable), or None when full."""
        return self.service.open(**{**self.defaults, **thresholds})

    def close(self, session):
        self.service.close(session)

    async def feed(self, session, user_hop, agent_hop, mode, more=False):
        return await self.service.feed(session, user_hop, agent_hop, mode, more=more)

    def full(self):
        return self.service.full()

    def active(self):
        return self.service.active()

    def thresholds(self, session) -> dict:
        return {t.name: getattr(session, _ATTR.get(t.name, t.name)) for t in THRESHOLDS}

    def set_threshold(self, session, name, value):
        """Set one of THRESHOLDS for this conversation; returns the value now in effect. KeyError for an unknown
        name, ValueError for a bad value."""
        v = find_threshold(THRESHOLDS, name).parse(value)
        setattr(session, _ATTR.get(name, name), v)
        return v
