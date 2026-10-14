"""tyger: ONE plugin for the tyger bundle (private repo `ears.model`, nineninesix/tyger): nemotron-3.5-asr (encoder +
transcription) and the turn-taker GRU that reads its layer 15, in one model.safetensors.

Modules:
  plugin      Tyger: the EarsPlugin surface and its tunable THRESHOLDS (light to import)
  constants   the audio grid, the turn-taker's layer and state vocabulary, the turn rules' numbers
  service     EarsService: the GPU thread that gathers every conversation's hop into one tick
  engine      EarsEngine: one tick (encode, the turn-taker, per-frame decisions), the ASR thread's jobs
  session     EarsSession: one conversation's turn-taking state and thresholds
  rules       the per-frame turn rules: endpoint, barge-in, backchannel
  asr         NemotronAsr: nemotron-3.5-asr's features, its layer-15 encode, transcription
  rnnt        nemotron-3.5-asr's greedy RNN-T decoder (its LSTM decoder + joint) as a CUDA graph
  bundle      fetch the bundle (HF_TOKEN), check it fits this plugin, load nemotron + the turn-taker from it
  controller  TurnTaker: the GRU and its heads (the model class)
  gru_batch   GruBatch: the GRU's step for every conversation at once, a state row each
"""
from .plugin import Tyger

__all__ = ["Tyger"]
