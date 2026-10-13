"""tyger-s2s: ONE plugin for the TYGER S2S bundle, nemotron-3.5-asr (encoder + transcription) and the turn-taker GRU
that reads its layer 15, in one model.safetensors (private repo `ears.model`). Built on nemotron_tyger's engine,
service, session, turn rules and ASR; what is its own:

Modules:
  plugin      TygerS2S: the EarsPlugin surface (nemotron-tyger-1.0's THRESHOLDS; light to import)
  bundle      fetch the bundle (HF_TOKEN), check it fits this plugin, load nemotron + the turn-taker from it
  controller  TurnTaker: the GRU and its heads (the model class)
  gru_batch   GruBatch: the GRU's step for every conversation at once, a state row each
  engine      S2SEngine: EarsEngine on the bundle's parts
"""
from .plugin import TygerS2S

__all__ = ["TygerS2S"]
