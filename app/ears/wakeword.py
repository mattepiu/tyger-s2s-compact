"""openWakeWord integration for Tyger-S2S (optional, activated by command arg or config).

Provides lightweight, zero-GPU standby wake-word detection on CPU (using openWakeWord ONNX/TFLite runtime)
with an automatic inactivity watchdog timer (default 40s) that drops back to STANDBY.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("wakeword")


@dataclass
class WakeWordConfig:
    enabled: bool = False
    model: str = "hey_jarvis"
    threshold: float = 0.6
    timeout_s: float = 40.0
    inference_framework: str = "onnx"

    def create_session(self, detector: Any | None = None) -> WakeWordSession:
        return WakeWordSession(self, detector=detector)


class WakeWordDetector:
    """Wraps openWakeWord's Model runtime."""
    def __init__(self, model_name: str = "hey_jarvis", inference_framework: str = "onnx"):
        self.model_name = model_name
        self.framework = inference_framework
        self.model = None
        self._load()

    def _load(self):
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "openwakeword is required for wake word detection. "
                "Install it with: pip install openwakeword"
            )
        try:
            self.model = Model(wakeword_models=[self.model_name], inference_framework=self.framework)
            log.info("openWakeWord model loaded: '%s' (%s)", self.model_name, self.framework)
        except Exception as e:
            raise RuntimeError(f"failed to load openWakeWord model '{self.model_name}': {e}") from e

    def predict(self, pcm_float32: np.ndarray) -> dict[str, float]:
        """Feed float32 16kHz audio samples. Returns {model_name: score}."""
        if self.model is None or len(pcm_float32) == 0:
            return {}
        # openWakeWord expects 16-bit signed PCM audio samples
        pcm_int16 = (pcm_float32 * 32767.0).clip(-32768, 32767).astype(np.int16)
        return self.model.predict(pcm_int16)

    def reset(self):
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()


class MockWakeWordDetector:
    """Mock detector for unit tests and environments without openWakeWord binaries."""
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self._next_score = 0.0

    def trigger(self, score: float = 0.9):
        self._next_score = score

    def predict(self, pcm_float32: np.ndarray) -> dict[str, float]:
        score = self._next_score
        self._next_score = 0.0
        return {self.model_name: score}

    def reset(self):
        self._next_score = 0.0


class WakeWordSession:
    """Manages STANDBY <-> ACTIVE transitions and silence watchdog per conversation."""
    def __init__(self, cfg: WakeWordConfig, detector: Any | None = None):
        self.cfg = cfg
        if detector is not None:
            self.detector = detector
        elif cfg.model == "mock":
            self.detector = MockWakeWordDetector()
        else:
            self.detector = WakeWordDetector(cfg.model, cfg.inference_framework)
        self.state = "STANDBY"
        self.last_speech_time = time.time()

    def feed(self, pcm: np.ndarray) -> tuple[bool, str | None, float]:
        """Feed audio frame in STANDBY. If wake word detected, transitions to ACTIVE."""
        if self.state != "STANDBY":
            return False, None, 0.0
        scores = self.detector.predict(pcm)
        for name, score in scores.items():
            if score >= self.cfg.threshold:
                self.state = "ACTIVE"
                self.last_speech_time = time.time()
                self.detector.reset()
                return True, name, float(score)
        return False, None, 0.0

    def notify_speech(self):
        """Called whenever user speaks or agent is active."""
        self.last_speech_time = time.time()

    def check_timeout(self) -> bool:
        """Called periodically. If silence exceeds timeout_s, returns to STANDBY."""
        if self.state == "ACTIVE":
            if (time.time() - self.last_speech_time) >= self.cfg.timeout_s:
                self.state = "STANDBY"
                self.detector.reset()
                return True
        return False

    def reset(self):
        self.detector.reset()
