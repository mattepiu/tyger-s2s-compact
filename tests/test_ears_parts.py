"""EARS's small parts on their own, GPU-free: AgentPlayback (the browser's playback estimate and the 2nd ear's hops)
and EarsApp (the page, the busy page, the voice list) over a real aiohttp test server with the fake EARS plugin.
Run: .venv/bin/python tests/test_ears_parts.py"""
import os, sys, asyncio, copy, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.playback import AgentPlayback  # noqa: E402
from app.ears.web import EarsApp  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_playback_estimate():
    pb = AgentPlayback(16000)
    pb.start()
    pb.add(np.zeros(24000, np.float32))                         # 1 s of reply audio, queued after a 40 ms lead-in
    assert pb.sent_s == 1.0 and pb.played_s() < 0.05
    assert abs(pb.play_until - (time.perf_counter() + 1.04)) < 0.02
    pb.add(np.zeros(12000, np.float32))                         # queued back to back
    assert pb.sent_s == 1.5 and abs(pb.play_until - (time.perf_counter() + 1.54)) < 0.02
    pb.start()
    assert pb.sent_s == 0.0 and pb.played_s() == 0.0


def test_second_ear_hops():
    pb = AgentPlayback(16000)
    pb.add(np.ones(2400, np.float32))                           # 0.1 s at 24 kHz -> 1600 samples at 16 kHz
    hop = pb.take_hop(5120)
    assert hop.shape == (5120,) and hop[:1600].min() == 1.0 and not hop[1600:].any()
    assert not pb.take_hop(5120).any()                          # nothing left: zeros
    pb.add(np.ones(2400, np.float32)); pb.clear_ear()
    assert not pb.take_hop(16).any()


def app(max_sessions=1):
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config.fake.yaml")))
    cfg["server"]["max_sessions"] = max_sessions
    cfg["server"]["mouth_port"] = 1                             # nothing there: EARS-only conversations
    return EarsApp(cfg, EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake")))


def test_page_busy_page_and_voices():
    async def go():
        ears = app(max_sessions=1)
        async with TestClient(TestServer(ears.application())) as client:
            assert (await client.get("/")).status == 200
            v = await (await client.get("/voices")).json()
            assert set(v) == {"voices", "voice"}
            ws1 = await client.ws_connect("/ws")
            ready = await ws1.receive_json()
            assert ready["type"] == "ready" and ready["mouth"] is False and ready["hop_ms"] == 320
            assert (await client.get("/")).status == 503                    # the one slot is taken
            ws2 = await client.ws_connect("/ws")
            assert (await ws2.receive_json()) == {"type": "busy"}
            await ws1.close()
            for _ in range(50):                                             # the slot frees when it ends
                if not ears.ears.full():
                    break
                await asyncio.sleep(0.02)
            assert (await client.get("/")).status == 200
    asyncio.run(go())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
