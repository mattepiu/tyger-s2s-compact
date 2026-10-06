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



def test_threshold_spec():
    from app.plugins.base import Slider, Threshold, find_threshold
    t = Threshold("fallback_frames", int, config_key="fallback_sil_frames", default=15)
    assert t.from_config({}) == 15 and t.from_config({"fallback_sil_frames": "9"}) == 9
    try:
        Threshold("ep_thr", float).from_config({})
        raise AssertionError("a threshold without a default needs its config key")
    except KeyError:
        pass
    assert t.parse("3") == 3 and Threshold("x", float).parse("0.25") == 0.25
    for bad in ("x", "1.5"):
        try:
            t.parse(bad); raise AssertionError(f"int threshold took {bad!r}")
        except ValueError:
            pass
    spec = Threshold("ep_thr", float, slider=Slider("endpoint", 0.3, 0.99, 0.005, decimals=3, meter="p_end"))
    assert spec.slider_spec() == {"name": "ep_thr", "type": "float", "label": "endpoint", "lo": 0.3, "hi": 0.99,
                                  "step": 0.005, "decimals": 3, "scale": 1.0, "unit": "", "meter": "p_end", "off": None}
    assert find_threshold([t, spec], "ep_thr") is spec
    try:
        find_threshold([t], "nope"); raise AssertionError("found an unknown threshold")
    except KeyError:
        pass


def test_nemotron_tyger_sliders_unchanged():
    """R-05 moved the sliders into the plugin; the user asked to keep nemotron-tyger's as they were on the page."""
    from app.plugins.ears.nemotron_tyger.plugin import THRESHOLDS
    got = {t.name: t.slider_spec() for t in THRESHOLDS if t.slider}
    want = {"ep_thr": ("endpoint p_end τ", 0.30, 0.99, 0.005, 3, 1.0, "", "p_end", None),
            "ep_deb": ("endpoint deb", 1, 5, 1, 0, 1.0, "", None, None),
            "barge_thr": ("barge (yield)", 0.20, 1.00, 0.01, 2, 1.0, "", "yield", None),
            "bc_thr": ("backchannel", 0.20, 1.00, 0.01, 2, 1.0, "", "bc", None),
            "fallback_frames": ("silence fallback", 0.4, 2.5, 0.1, 1, 0.08, "s", None, 100000)}
    keys = ("label", "lo", "hi", "step", "decimals", "scale", "unit", "meter", "off")
    assert {n: tuple(d[k] for k in keys) for n, d in got.items()} == want
    assert [t.name for t in THRESHOLDS] == ["ep_thr", "ep_deb", "barge_thr", "barge_deb", "bc_thr", "bc_deb",
                                            "fallback_frames"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
