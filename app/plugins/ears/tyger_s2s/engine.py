"""S2SEngine: nemotron_tyger's EarsEngine (the batched tick, the turn rules, the ASR thread) on the bundle: its
nemotron for the layer-15 encodes and the transcripts, and its turn-taker stepped per conversation (GruBatch)."""
from __future__ import annotations
import logging

from app.plugins.ears.nemotron_tyger.asr import NemotronAsr
from app.plugins.ears.nemotron_tyger.engine import EarsEngine
from .bundle import load
from .gru_batch import GruBatch

log = logging.getLogger("ears")


class S2SEngine(EarsEngine):
    def __init__(self, bundle_dir, slots=4, device="cuda", cuda_graph=True):
        """bundle_dir: a TYGER S2S bundle (bundle.fetch). cuda_graph: the RNN-T decoder's (the GRU runs eager)."""
        nemotron, proc, turn_taker, config = load(bundle_dir)
        asr = NemotronAsr(nemotron, proc, device=device, max_batch=slots, cuda_graph=cuda_graph)
        head = GruBatch(turn_taker, slots, device)
        log.info("EARS engine: %d slots, TYGER S2S bundle %s (nemotron %s, GRU %dx%d)", slots,
                 config["weights_sha256"][:8], config["frontend"]["revision"][:8],
                 turn_taker.core.num_layers, turn_taker.core.hidden_size)
        self._start(asr, head, slots, device)
