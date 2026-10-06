"""nemotron-tyger-1.0: ONE plugin for a jointly trained pair, nemotron-3.5-asr (encoder + transcription) and the tyger
1.0 turn-taking heads, which read that encoder's layer 15 and nothing else.

Modules:
  plugin      NemotronTyger: the EarsPlugin surface and its tunable THRESHOLDS (light to import)
  constants   the audio grid, the heads' layer / backbone / state vocabulary, the turn rules' numbers
  service     EarsService: the GPU thread that gathers every conversation's hop into one tick
  engine      EarsEngine: one tick (encode, heads, per-frame decisions), the ASR thread's jobs
  session     EarsSession: one conversation's turn-taking state and thresholds
  asr         NemotronAsr: nemotron-3.5-asr's features, its layer-15 encode, transcription
  rnnt        nemotron-3.5-asr's greedy RNN-T decoder (its LSTM decoder + joint) as a CUDA graph
  heads       the tyger 1.0 heads (the model classes)
  head_batch  RingKV + HeadBatch: the heads' batched, CUDA-graphed step for every conversation
"""
from .plugin import NemotronTyger

__all__ = ["NemotronTyger"]
