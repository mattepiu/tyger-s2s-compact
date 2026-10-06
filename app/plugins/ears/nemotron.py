"""nemotron-tyger-1.0: the nemotron streaming ASR encoder (nvidia/nemotron-3.5-asr-streaming-0.6b) + the tyger 1.0
turn-taking heads (exp-12 arm C: LoRA + state / hazard / yield / backchannel / endpoint heads; private weights,
`heads_repo` / `heads_file`). Wraps EarsEngine + EarsService (app/ears_core/batched.py, unchanged): one GPU thread,
one batched step per 320 ms hop for every conversation with a hop ready.

Per hop, feed() returns FRAME / ENDPOINT / BARGE / BACKCHANNEL events. A session is an EarsSession: reset() starts
a new turn (applied before its next hop); its thresholds are per conversation (the ?debug=1 sliders).
"""
from __future__ import annotations
import logging

log = logging.getLogger("ears")

# per-conversation thresholds: name -> (type, config key)
THRESHOLDS = {"ep_thr": (float, "ep_thr"), "ep_deb": (int, "ep_deb"), "barge_thr": (float, "barge_thr"),
              "barge_deb": (int, "barge_deb"), "bc_thr": (float, "bc_thr"), "bc_deb": (int, "bc_deb"),
              "fallback_frames": (int, "fallback_sil_frames")}
_ATTR = {"fallback_frames": "FALLBACK_SIL_FRAMES"}           # EarsSession's attribute name, where it differs


class NemotronTyger:
    def __init__(self, conf: dict, ctx):
        import torch
        from app import weights
        from app.ears_core.batched import EarsEngine, EarsService, HOP_N, MODEL_ID
        from app.ears_core.encoder_stream import ENC_SR
        torch.set_num_threads(int(conf.get("cpu_threads", 4)))   # default 48 threads mostly spin-wait
        self.sr = ENC_SR
        self.hop_samples = HOP_N                                 # 320 ms @ 16 kHz = 5120
        conf = {"fallback_sil_frames": 15, **conf}
        self.defaults = {k: typ(conf[key]) for k, (typ, key) in THRESHOLDS.items()}
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
        return {k: getattr(session, _ATTR.get(k, k)) for k in THRESHOLDS}

    def set_threshold(self, session, name, value):
        """Set one of THRESHOLDS for this conversation; returns the value now in effect. KeyError for an unknown
        name, ValueError for a bad value."""
        typ = THRESHOLDS[name][0]
        setattr(session, _ATTR.get(name, name), typ(value))
        return getattr(session, _ATTR.get(name, name))
