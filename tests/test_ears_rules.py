"""nemotron-tyger-1.0's turn rules (app/plugins/ears/nemotron_tyger/rules.py) on synthetic head outputs, GPU-free:
endpoint by the head and by silence, the minimum speech, backchannel debounce and refractory, barge-in and the turn it
seeds, mode switches. The engine and the validators' reference share these rules, so they are tested here.
Run: .venv/bin/python tests/test_ears_rules.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from app.plugins.ears.nemotron_tyger import rules  # noqa: E402
from app.plugins.ears.nemotron_tyger.constants import LISTEN, SPEAK, SR, WARMUP_FRAMES  # noqa: E402
from app.plugins.ears.nemotron_tyger.session import EarsSession  # noqa: E402

VOICED, SILENT = -20.0, -80.0                       # frame energy (dB) above / below VAD_DB


def outputs(p_end=0.0, yield_=0.0, bc=0.0, hazard=0.0):
    return np.array([0.0, 0.0, 0.0, 0.0, hazard, yield_, bc, p_end], np.float32)


def session(**th):
    s = EarsSession(0, **{"ep_thr": 0.9, "ep_deb": 2, "barge_thr": 0.5, "barge_deb": 2, "bc_thr": 0.5, "bc_deb": 2,
                          "fallback_frames": 15, **th})
    rules.begin_hop(s, np.zeros(5120, np.float32), np.zeros(5120, np.float32), "listen")
    return s


def frames(s, n, mode="listen", act=VOICED, **out):
    evs, final, resets = [], None, 0
    for _ in range(n):
        e, f, r = rules.on_frame(s, mode, outputs(**out), act)
        evs += e; final = f or final; resets += r
    return [e["kind"] for e in evs if e["kind"] != "FRAME"], final, resets


def test_endpoint_by_the_head_after_enough_speech():
    s = session()
    kinds, final, _ = frames(s, WARMUP_FRAMES + 1, p_end=0.95)       # warm-up, then the 1st of ep_deb frames
    assert kinds == [] and final is None and s.ep_run == 1
    kinds, final, resets = frames(s, 1, p_end=0.95)                  # the 2nd: the endpoint
    assert kinds == ["ENDPOINT"] and resets == 1
    ev, wav = final
    assert ev["by"] == "head" and ev["transcript"] is None and len(wav) == 5120   # the turn's audio, for ASR
    assert s.prev_state == LISTEN and s.warm == 0 and not s.in_turn and len(s.turn_u) == 0


def test_no_endpoint_without_enough_speech():
    s = session()
    frames(s, 3, act=VOICED)                                          # 240 ms < MIN_SPEECH_MS
    kinds, final, _ = frames(s, 20, act=SILENT, p_end=0.99)
    assert kinds == [] and final is None


def test_endpoint_by_silence_fallback():
    s = session(ep_thr=2.0)                                           # the head can't fire
    frames(s, 10, act=VOICED)
    kinds, _, _ = frames(s, 14, act=SILENT)
    assert kinds == []
    kinds, final, _ = frames(s, 1, act=SILENT)                        # the 15th silent frame
    assert kinds == ["ENDPOINT"] and final[0]["by"] == "silence"


def test_backchannel_debounce_and_refractory():
    s = session(ep_thr=2.0, bc_refractory=5)
    frames(s, WARMUP_FRAMES + 1)
    s.partial_asr = "so I went"
    kinds, _, _ = frames(s, 1, bc=0.9)
    assert kinds == []                                                # bc_deb is 2
    kinds, _, _ = frames(s, 1, bc=0.9)
    assert kinds == ["BACKCHANNEL"]
    kinds, _, _ = frames(s, 6, bc=0.9)                                # 5 refractory frames, then the run restarts
    assert kinds == []
    kinds, _, _ = frames(s, 1, bc=0.9)
    assert kinds == ["BACKCHANNEL"]


def test_barge_seeds_the_next_turn_with_the_users_speech():
    s = session()
    speech = np.full(5120, 0.1, np.float32)
    assert rules.switch_mode(s, "speaking") and s.prev_state == SPEAK
    assert not rules.switch_mode(s, "speaking")
    rules.begin_hop(s, speech, np.zeros(5120, np.float32), "speaking")   # user audio while the agent speaks
    kinds, _, _ = frames(s, WARMUP_FRAMES, mode="speaking", yield_=0.9)
    assert kinds == []                                                    # warm-up
    kinds, _, _ = frames(s, 2, mode="speaking", yield_=0.9)
    assert kinds == ["BARGE"] and s.seed_u is not None and len(s.seed_u) == 5120
    s.reset()                                                             # the server resets on a barge
    assert rules.begin_hop(s, np.zeros(5120, np.float32), np.zeros(5120, np.float32), "listen")
    assert s.in_turn and s.voiced_ms > 0 and len(s.turn_u) == 2 * 5120    # the seed, then this hop
    assert s.turn_len_s() == 2 * 5120 / SR


def test_frame_energy():
    e = rules.frame_energy(np.concatenate([np.zeros(1280, np.float32), np.full(1280, 0.1, np.float32)]))
    assert len(e) == 2 and e[0] < -100 and abs(e[1] - (-20.0)) < 0.01


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
