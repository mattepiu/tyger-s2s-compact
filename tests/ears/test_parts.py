"""EARS's small parts on their own, GPU-free: AgentPlayback (the reply at the EARS rate, the server's playback
guess). The web app's HTTP side is in test_web.py.
Run: .venv/bin/python tests/ears/test_parts.py"""
import os, sys, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from app.ears.playback import AgentPlayback  # noqa: E402


def test_playback_estimate():
    """The server's own guess at the page's playback: only the safety net and the logs now (I-17)."""
    pb = AgentPlayback(16000)
    pb.start()
    pb.add(np.zeros(24000, np.float32), 0)                      # 1 s of reply audio, queued after a 40 ms lead-in
    assert pb.sent24 == 24000 and pb.estimate_s() < 0.05
    assert abs(pb.play_until - (time.perf_counter() + 1.04)) < 0.02
    pb.add(np.zeros(12000, np.float32), 0)                      # queued back to back
    assert pb.sent24 == 36000 and abs(pb.play_until - (time.perf_counter() + 1.54)) < 0.02
    pb.start()
    assert pb.sent24 == 0 and pb.estimate_s() == 0.0 and pb.n16 == 0


def test_reply_at_the_ears_rate():
    pb = AgentPlayback(16000); pb.start()
    pb.add(np.ones(2400, np.float32), 0)                        # 0.1 s at 24 kHz -> ~1600 samples at 16 kHz
    ear = pb.audio[:pb.n16]
    assert 1580 <= len(ear) <= 1600, len(ear)                   # a few held back for the filter's future input
    assert np.allclose(ear[40:-40], 1.0, atol=1e-3)             # the level is kept inside the clip
    pb.start()
    assert pb.n16 == 0                                          # a new reply starts empty


def ear_audio(chunks):
    pb = AgentPlayback(16000); pb.start()
    for c in chunks:
        pb.add(c, 0)
    return pb.audio[:pb.n16].copy()


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
    pb = AgentPlayback(16000); pb.start(); pb.add(a24[:24000], 0); pb.start()   # a new reply starts clean
    pb.add(a24, 0)
    assert np.allclose(pb.audio[:pb.n16], whole, atol=1e-5)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
