"""tyger-s2s, the EarsPlugin surface: the TYGER S2S bundle (`ears.model` at `ears.revision`; private), nemotron-3.5-asr
and the turn-taker GRU in one file. It runs on nemotron-tyger-1.0's machinery (nemotron_tyger/: the audio grid, the
batched tick, the turn rules, the ASR and GPU threads), with the same thresholds and sliders; only the model behind the
decisions differs. As there, only the head ends a turn: the bundle's silence fallback isn't used.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from app.plugins.base import EarsPlugin
from app.plugins.ears.nemotron_tyger.constants import HOP_N, SR
from app.plugins.ears.nemotron_tyger.plugin import THRESHOLDS
from app.plugins.options import PluginOptions

log = logging.getLogger("ears")


class TygerS2S(EarsPlugin):
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
        from app.plugins.ears.nemotron_tyger.service import EarsService
        from .bundle import fetch
        from .engine import S2SEngine
        super().__init__(conf, ctx)
        o = self.opts
        torch.set_num_threads(o.cpu_threads)
        self.slots = int(ctx.cfg["server"].get("max_sessions", 4))
        log.info("loading EARS (tyger-s2s: %s) for %d sessions, ep_thr=%s ep_deb=%s", o.model, self.slots,
                 self.defaults["ep_thr"], self.defaults["ep_deb"])
        engine = S2SEngine(fetch(o.model, o.revision), slots=self.slots)
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
