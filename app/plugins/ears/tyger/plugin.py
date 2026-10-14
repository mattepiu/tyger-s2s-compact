"""tyger, the EarsPlugin surface: the tyger bundle (`ears.model` at `ears.revision`; private), nemotron-3.5-asr and the
turn-taker GRU in one file. Builds EarsEngine + EarsService: one GPU thread, one batched step per 320 ms hop for every
conversation with a hop ready. The class stays light to import (EARS's page and tests read THRESHOLDS); the models
load in the constructor. Only the head ends a turn: the bundle's silence fallback isn't used.

Per hop, feed() returns FRAME / ENDPOINT / BARGE / BACKCHANNEL / PARTIAL events. A session is an EarsSession: reset()
starts a new turn (applied before its next hop); its thresholds are per conversation (the page's live-tuning sliders).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from app.plugins.base import EarsPlugin, Slider, Threshold
from app.plugins.options import PluginOptions
from .constants import HOP_N, SR

log = logging.getLogger("ears")

# per-conversation thresholds; defaults from the `ears:` config block
THRESHOLDS = (
    Threshold("ep_thr", float, slider=Slider("endpoint p_end τ", 0.30, 0.99, 0.005, decimals=3, meter="p_end")),
    Threshold("ep_deb", int, slider=Slider("endpoint deb", 1, 5, 1, decimals=0)),
    Threshold("barge_thr", float, slider=Slider("barge (yield)", 0.20, 1.00, 0.01, meter="yield")),
    Threshold("barge_deb", int),
    Threshold("bc_thr", float, slider=Slider("backchannel", 0.20, 1.00, 0.01, meter="bc")),
    Threshold("bc_deb", int),
)


class Tyger(EarsPlugin):
    THRESHOLDS = THRESHOLDS
    sr = SR
    hop_samples = HOP_N                    # 320 ms @ 16 kHz = 5120

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        model: str                         # the bundle (HF repo id or a local directory)
        revision: str | None = None        # pin it (None: latest)
        gather_ms: float = 15              # wait this long to batch other conversations' hops into one GPU step
        expect_ms: float = 60              # ...or this long when one that is behind is about to resubmit
        stats_s: float = 60                # log timing + busy % this often
        cpu_threads: int = 4               # torch CPU threads (the default 48 mostly spin-wait)

    def __init__(self, conf: dict, ctx):
        import torch
        from .bundle import fetch
        from .engine import EarsEngine
        from .service import EarsService
        super().__init__(conf, ctx)
        o = self.opts
        torch.set_num_threads(o.cpu_threads)
        self.slots = int(ctx.cfg["server"].get("max_sessions", 4))
        log.info("loading EARS (tyger: %s) for %d sessions, ep_thr=%s ep_deb=%s", o.model, self.slots,
                 self.defaults["ep_thr"], self.defaults["ep_deb"])
        engine = EarsEngine(fetch(o.model, o.revision), slots=self.slots)
        self.service = EarsService(engine, gather_s=o.gather_ms / 1e3, expect_s=o.expect_ms / 1e3,
                                   stats_every_s=o.stats_s)

    def open_session(self, thresholds):
        return self.service.open(**thresholds)

    def close(self, session):
        self.service.close(session)

    async def feed(self, session, user_hop, agent_hop, mode, more=False):
        return await self.service.feed(session, user_hop, agent_hop, mode, more=more)

    def full(self):
        return self.service.full()

    def active(self):
        return self.service.active()
