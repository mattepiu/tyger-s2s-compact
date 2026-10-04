"""Plugin plumbing (no GPU): plugin classes import without their heavy dependencies, and a TTS that isn't at the
24 kHz wire rate is resampled. Run: .venv/bin/python tests/test_plugins.py"""
import os, subprocess, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fakes import FakeTTS  # noqa: E402
from app.plugins import REGISTRY  # noqa: E402
from app.plugins.tts import AtWireRate, WIRE_SR  # noqa: E402
from app.config import ROOT  # noqa: E402


def test_plugin_classes_import_light():
    """EARS imports the TTS class for list_voices, and a hosted-only config must not load vLLM: importing any
    registered class must not pull in vllm / vllm_omni (they load in constructors)."""
    code = ("import sys; from app.plugins import REGISTRY, plugin_class\n"
            "[plugin_class(k, n) for k in REGISTRY for n in REGISTRY[k]]\n"
            "bad = [m for m in ('vllm', 'vllm_omni') if m in sys.modules]\n"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"importing plugin classes loaded {out.stdout.strip()}"


def test_off_rate_tts_resampled_to_wire():
    tts = AtWireRate(FakeTTS(out_sr=22050, n=10, chunk_s=0.1))
    out = np.concatenate(list(tts.stream("hello", None)))
    assert tts.out_sr == WIRE_SR and out.dtype == np.float32
    assert abs(len(out) - WIRE_SR * 1.0) <= 2, len(out)                 # 1.0 s at 22.05 kHz -> 1.0 s at 24 kHz
    assert tts.list_voices() == (["fake"], "fake")                      # everything else passes through


def test_off_rate_tts_stops():
    tts = AtWireRate(FakeTTS(out_sr=22050, n=50, chunk_s=0.1, step_s=0.005))
    n = 0
    stop = lambda: n >= 3   # noqa: E731
    for _ in tts.stream("hello", None, stop_flag=stop):
        n += 1
    assert n <= 4 and tts.inner.aborted == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
