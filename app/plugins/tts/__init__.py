"""TTS plugins (PLUGINS.md); see app/plugins/__init__.py for the registry. Audio leaves MOUTH at WIRE_SR (the
EARS<->browser wire and EARS's 2nd ear); a plugin with another out_sr is wrapped in AtWireRate."""
from __future__ import annotations
import numpy as np

WIRE_SR = 24000


class AtWireRate:
    """A TTS plugin resampled to WIRE_SR, per request with a stateful resampler (clean chunk edges). Everything
    else passes through to the plugin."""
    def __init__(self, tts):
        self.inner = tts; self.out_sr = WIRE_SR

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def stream(self, text, voice, stop_flag=None):
        from app.runtime.audio import StreamingAudioResampler
        rs = StreamingAudioResampler(self.inner.out_sr, WIRE_SR)
        for a in self.inner.stream(text, voice, stop_flag=stop_flag):
            out = rs.process(a)
            if out.size:
                yield out
        if not (stop_flag and stop_flag()):
            tail = rs.process(np.zeros(0, np.float32), final=True)
            if tail.size:
                yield tail
