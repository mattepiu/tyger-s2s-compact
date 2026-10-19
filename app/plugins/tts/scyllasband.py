"""scyllasband: managed multi-voice TTS via Scylla's Band (https://github.com/lowkeytea/scyllasband).

Rectified-flow continuous acoustic latent model + Vocos neural vocoder at native 24 kHz mono.
LiteRT CPU backend (>35x real time, 0 MB GPU VRAM) or ONNX GPU backend (<250 MB VRAM).
Supports continuous delivery controls: energy, tension, valence, assertiveness (0 to 4, 2 neutral).
Maps Tyger personas (mira -> ariadne, aria -> scylla, kai -> ink, rex -> rex).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from app.plugins.base import TTSCaps, TTSPlugin
from app.plugins.options import PluginOptions

log = logging.getLogger("tts")

# Map Tyger persona names to Scylla's Band native voice IDs
VOICE_MAP: dict[str, str] = {
    "mira": "ariadne",
    "aria": "scylla",
    "kai": "ink",
    "rex": "rex",
}

# Reverse mapping to allow referencing native voice IDs directly
KNOWN_VOICE_IDS: set[str] = set(VOICE_MAP.values())

try:
    from scyllasband import SynthesisRequest
except ImportError:
    @dataclass
    class SynthesisRequest:
        text: str
        voice_id: str
        language: str = "en_us"
        delivery: dict[str, float] | None = None
        speed: float = 1.0

# Persona delivery profiles (calibrated in REDUCTION.md)
PERSONA_DELIVERY: dict[str, dict[str, float]] = {
    "mira": {"energy": 1.8, "valence": 2.2, "tension": 1.5, "assertiveness": 2.0},
    "aria": {"energy": 2.2, "valence": 2.2, "tension": 2.0, "assertiveness": 2.2},
    "kai": {"energy": 1.8, "tension": 1.4, "valence": 2.0, "assertiveness": 2.0},
    "rex": {"energy": 2.3, "assertiveness": 2.4, "tension": 2.2, "valence": 2.0},
    "ariadne": {"energy": 1.8, "valence": 2.2, "tension": 1.5, "assertiveness": 2.0},
    "scylla": {"energy": 2.2, "valence": 2.2, "tension": 2.0, "assertiveness": 2.2},
    "ink": {"energy": 1.8, "tension": 1.4, "valence": 2.0, "assertiveness": 2.0},
}


class ScyllasBandTTS(TTSPlugin):
    out_sr = 24000
    caps = TTSCaps(cloning=False, max_batch=1)

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        bundle: str = "scyllasband/models/litert"  # or "scyllasband/models/onnx"
        backend: str = "litert"                   # "litert", "onnx", "coreml"
        language: str = "en_us"
        energy: float = 2.0                       # 0 to 4 (2 is neutral)
        tension: float = 2.0
        valence: float = 2.0
        assertiveness: float = 2.0
        speed: float = 1.0
        out_sr: int = 24000

    @classmethod
    def list_voices(cls, cfg=None) -> tuple[list[str], str | None]:
        default = "mira"
        if isinstance(cfg, dict):
            default = cfg.get("voices", {}).get("default", "mira")
        if default not in VOICE_MAP and default not in KNOWN_VOICE_IDS:
            default = "mira"
        return sorted(list(VOICE_MAP.keys())), default

    def __init__(self, conf: dict | None = None, ctx=None):
        super().__init__(conf, ctx)
        self.out_sr = self.opts.out_sr
        self.runtime = None
        try:
            from scyllasband import ScyllasBandRuntime
            self.runtime = ScyllasBandRuntime.from_bundle(self.opts.bundle, backend=self.opts.backend)
            log.info("ScyllasBandRuntime initialized with bundle=%s backend=%s", self.opts.bundle, self.opts.backend)
        except ImportError:
            log.warning("scyllasband package not installed; runtime deferred")

    def load_voice(self, name: str) -> Any:
        if not isinstance(name, str):
            return None
        if name in VOICE_MAP:
            return VOICE_MAP[name]
        if name in KNOWN_VOICE_IDS:
            return name
        return None

    def _delivery_for(self, voice_name: str) -> dict[str, float]:
        """Delivery parameters: start with persona baseline, allow Options to override if set away from neutral 2.0."""
        base = PERSONA_DELIVERY.get(voice_name, {}).copy()
        for k in ("energy", "tension", "valence", "assertiveness"):
            opt_val = getattr(self.opts, k, 2.0)
            if opt_val != 2.0 or k not in base:
                base[k] = opt_val
        return base

    def stream(self, text: str, voice=None, stop_flag=None) -> Iterator[np.ndarray]:
        if stop_flag and stop_flag():
            return
        if not text or not text.strip():
            return
        if self.runtime is None:
            raise RuntimeError("scyllasband package is not installed or runtime not loaded")

        voice_key = voice if isinstance(voice, str) else "mira"
        voice_id = VOICE_MAP.get(voice_key, voice_key)
        delivery = self._delivery_for(voice_key)

        req = SynthesisRequest(
            text=text,
            voice_id=voice_id,
            language=self.opts.language,
            delivery=delivery,
            speed=self.opts.speed,
        )

        res = self.runtime.synthesize(req)
        if stop_flag and stop_flag():
            return
        if getattr(res, "audio", None) is not None:
            audio = np.asarray(res.audio, dtype=np.float32).squeeze()
            if audio.size > 0:
                yield audio

    def warmup(self, voice=None):
        if self.runtime is not None:
            if hasattr(self.runtime, "warmup"):
                try:
                    self.runtime.warmup()
                except Exception as e:
                    log.warning("ScyllasBand runtime warmup error: %s", e)
            try:
                for _ in self.stream("Warmup.", voice or "mira"):
                    pass
            except Exception as e:
                log.warning("ScyllasBand stream warmup error: %s", e)

    def shutdown(self):
        if self.runtime is not None:
            del self.runtime
            self.runtime = None
