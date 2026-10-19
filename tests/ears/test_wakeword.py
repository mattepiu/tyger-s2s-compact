"""Unit tests for the optional openWakeWord integration and state machine.
Run: python tests/ears/test_wakeword.py
"""
import asyncio
import copy
import os
import sys
import time
import unittest

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]

from app.ears.wakeword import WakeWordConfig, MockWakeWordDetector, WakeWordSession
from app.ears.conversation import Conversation
from app.plugins.ears.fake import EnergyEars
from app.plugins.base import PluginContext
from fakes import FakePage


class TestWakeWord(unittest.IsolatedAsyncioTestCase):
    def test_wakeword_session_state_machine(self):
        cfg = WakeWordConfig(enabled=True, model="mock", threshold=0.6, timeout_s=0.2)
        detector = MockWakeWordDetector()
        session = WakeWordSession(cfg, detector=detector)

        self.assertEqual(session.state, "STANDBY")

        dummy_pcm = np.zeros(1600, dtype=np.float32)

        # 1. No trigger: stays in STANDBY
        detected, name, score = session.feed(dummy_pcm)
        self.assertFalse(detected)
        self.assertEqual(session.state, "STANDBY")

        # 2. Trigger wake word
        detector.trigger(score=0.85)
        detected, name, score = session.feed(dummy_pcm)
        self.assertTrue(detected)
        self.assertEqual(name, "mock")
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(session.state, "ACTIVE")

        # 3. In ACTIVE, feed returns False (not processed as standby)
        detected, _, _ = session.feed(dummy_pcm)
        self.assertFalse(detected)

        # 4. Check timeout: not expired yet
        self.assertFalse(session.check_timeout())
        self.assertEqual(session.state, "ACTIVE")

        # 5. User speech refreshes timer
        session.notify_speech()
        self.assertFalse(session.check_timeout())

        # 6. Wait for silence timeout (0.2s)
        time.sleep(0.25)
        self.assertTrue(session.check_timeout())
        self.assertEqual(session.state, "STANDBY")

    async def test_conversation_with_wakeword(self):
        from app.config import load_config
        ROOT = os.path.dirname(TESTS)
        cfg_root = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
        cfg_root["server"]["mouth_port"] = 1
        ears = EnergyEars(cfg_root["ears"], PluginContext(None, cfg_root, "fake"))
        ears_session = ears.open()

        page = FakePage()
        ww_cfg = WakeWordConfig(enabled=True, model="mock", threshold=0.6, timeout_s=0.2)
        detector = MockWakeWordDetector()
        ww_session = WakeWordSession(ww_cfg, detector=detector)

        conv = Conversation(page, ears, ears_session, cfg_root, wakeword=ww_session)
        conv._connect_mouth = lambda: asyncio.sleep(0)
        self.assertEqual(conv.state, "STANDBY")

        # Ready message contains wakeword status
        ready = conv._ready_message()
        self.assertIn("wakeword", ready)
        self.assertTrue(ready["wakeword"]["enabled"])
        self.assertEqual(ready["wakeword"]["state"], "STANDBY")

        # Start background conversation runner
        runner = asyncio.create_task(conv.run())
        await asyncio.sleep(0.05)

        dummy_chunk = np.zeros(1600, dtype=np.float32)

        # Simulate mic block from browser while in STANDBY (no trigger)
        page.push(dummy_chunk)
        await asyncio.sleep(0.05)

        # self.mic must NOT have buffered the block (stays in standby)
        self.assertEqual(len(conv.mic), 0)
        self.assertEqual(conv.state, "STANDBY")

        # Trigger wake word
        detector.trigger(0.95)
        page.push(dummy_chunk)
        await asyncio.sleep(0.05)

        # Must transition to ACTIVE and buffer the block
        self.assertEqual(conv.state, "ACTIVE")
        self.assertEqual(len(conv.mic), 1600)

        # Verify browser received wakeword active event
        types = [m.get("type") for m in page.sent if isinstance(m, dict)]
        self.assertIn("wakeword", types)
        active_ev = next(m for m in page.sent if m.get("type") == "wakeword" and m.get("state") == "active")
        self.assertEqual(active_ev["model"], "mock")

        # Wait for watchdog silence timeout (0.2s)
        await asyncio.sleep(0.35)
        self.assertEqual(conv.state, "STANDBY")
        standby_ev = next(m for m in page.sent if m.get("type") == "wakeword" and m.get("state") == "standby")
        self.assertEqual(standby_ev["reason"], "timeout")

        # Test manual wake and sleep commands
        conv.session.reset()
        page.command("wake")
        await asyncio.sleep(0.05)
        self.assertEqual(conv.state, "ACTIVE")

        page.command("sleep")
        await asyncio.sleep(0.05)
        self.assertEqual(conv.state, "STANDBY")

        page.q.put_nowait(None)
        runner.cancel()
        ears.close(ears_session)


if __name__ == "__main__":
    unittest.main()
