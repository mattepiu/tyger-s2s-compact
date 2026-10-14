"""What both processes' code runs on, independent of any model or conversation:

  event_loop   EngineLoop: one asyncio loop on a background thread for MOUTH's in-process engines and API clients
  audio        StreamingAudioResampler: chunk-invariant resampling of streamed audio (the 24 kHz wire, the 2nd ear)
"""
