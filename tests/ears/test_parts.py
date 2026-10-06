"""EARS's small parts on their own, GPU-free: AgentPlayback (the browser's playback estimate and the 2nd ear's hops)
and EarsApp (the page, the busy page, the voice list) over a real aiohttp test server with the fake EARS plugin.
Run: .venv/bin/python tests/ears/test_parts.py"""
import os, sys, asyncio, copy, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.playback import AgentPlayback  # noqa: E402
from app.ears.web import EarsApp  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402

ROOT = os.path.dirname(TESTS)


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
    pb.add(np.ones(2400, np.float32))                           # 0.1 s at 24 kHz -> ~1600 samples at 16 kHz
    hop = pb.take_hop(5120)
    n = int(np.count_nonzero(hop))
    assert hop.shape == (5120,) and 1580 <= n <= 1600, n        # a few held back for the filter's future input
    assert np.allclose(hop[40:n - 40], 1.0, atol=1e-3)          # the level is kept inside the clip
    assert not pb.take_hop(5120).any()                          # nothing left: zeros
    pb.add(np.ones(2400, np.float32)); pb.clear_ear()
    assert not pb.take_hop(16).any()


def ear_audio(chunks):
    pb = AgentPlayback(16000); pb.start()
    for c in chunks:
        pb.add(c)
    return pb.take_hop(len(pb.ear))


def test_second_ear_does_not_depend_on_chunk_edges():
    """The same reply audio, cut into chunks differently, gives the same 2nd-ear audio (R-04b: the old per-chunk
    interpolation dropped a fractional sample at every chunk edge, so the ear drifted)."""
    rng = np.random.default_rng(0)
    a24 = (0.3 * np.sin(2 * np.pi * 300 * np.arange(24000 * 2) / 24000) + 0.01 * rng.standard_normal(48000)) \
        .astype(np.float32)
    whole = ear_audio([a24])
    cuts = np.sort(rng.choice(np.arange(1, len(a24)), 60, replace=False))
    pieces = ear_audio(np.split(a24, cuts))
    assert len(whole) == len(pieces) and np.allclose(whole, pieces, atol=1e-5)
    pb = AgentPlayback(16000); pb.add(a24[:24000]); pb.clear_ear(); pb.start()   # a new reply starts clean
    pb.add(a24)
    assert np.allclose(pb.take_hop(len(pb.ear)), whole, atol=1e-5)


def app(max_sessions=1):
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
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
            assert [x["name"] for x in ready["sliders"]] == ["endpoint_silence_frames"]   # the plugin's own spec
            assert ready["thresholds"] == {"endpoint_silence_frames": 6}
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
