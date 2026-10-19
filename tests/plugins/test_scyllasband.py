"""Tests for Scylla's Band TTS plugin (app/plugins/tts/scyllasband.py).
Run: python tests/plugins/test_scyllasband.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]

from app.plugins.tts.scyllasband import ScyllasBandTTS, VOICE_MAP, PERSONA_DELIVERY
from app.plugins.tts import WIRE_SR


class MockAudioResult:
    def __init__(self, audio):
        self.audio = audio


class TestScyllasBandTTS(unittest.TestCase):
    def test_voice_discovery_and_mapping(self):
        voices, default = ScyllasBandTTS.list_voices({})
        self.assertEqual(voices, ["aria", "kai", "mira", "rex"])
        self.assertEqual(default, "mira")

        # Custom default from cfg
        voices, default = ScyllasBandTTS.list_voices({"voices": {"default": "rex"}})
        self.assertEqual(default, "rex")

        tts = ScyllasBandTTS()
        self.assertEqual(tts.load_voice("mira"), "ariadne")
        self.assertEqual(tts.load_voice("aria"), "scylla")
        self.assertEqual(tts.load_voice("kai"), "ink")
        self.assertEqual(tts.load_voice("rex"), "rex")
        self.assertEqual(tts.load_voice("ariadne"), "ariadne")
        self.assertIsNone(tts.load_voice("no-such-voice"))

    def test_delivery_tuning(self):
        tts = ScyllasBandTTS()
        mira_delivery = tts._delivery_for("mira")
        self.assertEqual(mira_delivery["energy"], 1.8)
        self.assertEqual(mira_delivery["valence"], 2.2)
        self.assertEqual(mira_delivery["tension"], 1.5)
        self.assertEqual(mira_delivery["assertiveness"], 2.0)

        rex_delivery = tts._delivery_for("rex")
        self.assertEqual(rex_delivery["energy"], 2.3)
        self.assertEqual(rex_delivery["assertiveness"], 2.4)

        # Options override away from 2.0
        custom_tts = ScyllasBandTTS({"energy": 3.5})
        custom_delivery = custom_tts._delivery_for("mira")
        self.assertEqual(custom_delivery["energy"], 3.5)

    def test_wire_rate_and_caps(self):
        tts = ScyllasBandTTS()
        self.assertEqual(tts.out_sr, WIRE_SR)
        self.assertFalse(tts.caps.cloning)
        self.assertEqual(tts.caps.max_batch, 1)

    def test_synthesis_stream(self):
        tts = ScyllasBandTTS()
        mock_runtime = MagicMock()
        dummy_audio = np.ones(2400, dtype=np.float32) * 0.1
        mock_runtime.synthesize.return_value = MockAudioResult(dummy_audio)
        tts.runtime = mock_runtime

        # Synthesis
        chunks = list(tts.stream("Hello world, this is a test.", "mira"))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(isinstance(chunks[0], np.ndarray))
        self.assertEqual(chunks[0].dtype, np.float32)
        self.assertEqual(len(chunks[0]), 2400)

        # Stop flag aborts immediately
        chunks = list(tts.stream("Hello again.", "mira", stop_flag=lambda: True))
        self.assertEqual(len(chunks), 0)

        # Empty text yields nothing
        chunks = list(tts.stream("", "mira"))
        self.assertEqual(len(chunks), 0)

    def test_warmup_and_shutdown(self):
        tts = ScyllasBandTTS()
        mock_runtime = MagicMock()
        mock_runtime.synthesize.return_value = MockAudioResult(np.zeros(240, dtype=np.float32))
        tts.runtime = mock_runtime
        tts.warmup("mira")
        mock_runtime.warmup.assert_called_once()
        mock_runtime.synthesize.assert_called()

        tts.shutdown()
        self.assertIsNone(tts.runtime)


if __name__ == "__main__":
    unittest.main()
