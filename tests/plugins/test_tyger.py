"""tyger without its weights (no GPU, no download): the bundle check, and a small random TurnTaker stepped as EARS
steps it. Run: .venv/bin/python tests/plugins/test_tyger.py"""
import os, sys

import numpy as np, torch

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]
from app.plugins.ears.tyger.bundle import check_config  # noqa: E402
from app.plugins.ears.tyger.controller import TurnTaker  # noqa: E402
from app.plugins.ears.tyger.gru_batch import GruBatch  # noqa: E402

# what the published bundle's config.json says (nineninesix/tyger @ e35bfa54), the fields check_config reads
CONFIG = {
    "model_type": "tyger_nemotron_gru", "format_version": 1, "dtype": "float32",
    "state_names": ["LISTEN", "HOLD", "SPEAK", "BACKCHANNEL"],
    "output_order": ["state_logits", "hazard_logits", "yield_logits", "bc_logits", "endpoint_logits"],
    "controller": {"architecture": {"kind": "gru", "width": 768, "layers": 3}},
    "frontend": {"layer": 15, "hidden_size": 1024, "num_lookahead_tokens": 0, "sample_rate": 16000,
                 "frame_period_s": 0.08, "live_window_s": 8.0, "live_hop_s": 0.32,
                 "energy_norm": "clip((dB + 50) / 25, -3, 3)"},
    "streaming": {"live_previous_state": {"listen": 0, "speaking": 2}},
    "default_policy": {"ep_thr": 0.91, "ep_deb": 2, "barge_thr": 0.5, "barge_deb": 2, "bc_thr": 0.4, "bc_deb": 2,
                       "bc_refractory_frames": 15, "fallback_sil_frames": 15, "vad_db": -50.0,
                       "min_speech_ms": 350.0, "warmup_frames": 6},
}


def _with(path, value):
    """CONFIG with one nested field changed."""
    import copy
    c = copy.deepcopy(CONFIG); d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return c


def test_bundle_check():
    """The published bundle's config passes; one that needs another frontend or other turn rules, or another
    format, fails before anything loads, naming the field."""
    check_config(CONFIG)
    for path, value, named in ((("frontend", "layer"), 9, "layer"), (("frontend", "live_hop_s"), 0.16, "live_hop_s"),
                               (("streaming", "live_previous_state"), {"listen": 0, "speaking": 3}, "previous state"),
                               (("default_policy", "vad_db"), -40.0, "vad_db"),
                               (("controller", "architecture"), {"kind": "tcn"}, "controller")):
        try:
            check_config(_with(path, value)); raise AssertionError(f"accepted {path} = {value}")
        except ValueError as e:
            assert named in str(e), e
    try:
        check_config(_with(("model_type",), "tyger_endpointer")); raise AssertionError("accepted another model type")
    except ValueError:
        pass


def test_plugin_surface():
    """tyger keeps the page's sliders (THRESHOLDS), and its block names the bundle: a key of the removed tyger 1.0
    heads there is an error at config load."""
    from app.plugins import plugin_class
    from app.plugins.ears.tyger.plugin import THRESHOLDS
    cls = plugin_class("ears", "tyger")
    assert cls.THRESHOLDS is THRESHOLDS and cls.hop_samples == 5120 and cls.sr == 16000
    block = {"plugin": "tyger", "model": "nineninesix/tyger", "revision": "e35bfa54", "ep_thr": 0.915,
             "ep_deb": 2, "barge_thr": 0.4, "barge_deb": 2, "bc_thr": 0.4, "bc_deb": 2}
    assert cls.options(block).model == "nineninesix/tyger"
    for bad in ({"heads": "nineninesix/TYGER-S2S-endpointer"}, {"model": None}):
        try:
            cls.options({**block, **bad}); raise AssertionError(f"accepted {bad}")
        except (ValueError, KeyError, TypeError):
            pass


def _model(seed=0):
    torch.manual_seed(seed)
    return TurnTaker(width=32, layers=3).eval()


def _inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(T, 1024, generator=g), torch.randn(T, 1024, generator=g),
            (torch.rand(T, generator=g) * 70 - 90).tolist())


@torch.no_grad()
def test_turn_taker_frame_by_frame_is_the_whole_window():
    """Stepping one frame at a time with the carried state gives the outputs of the whole window at once (what EARS
    does vs what the model was evaluated on); probabilities are in [0, 1]."""
    m = _model(); hu, ha, db = _inputs(24, 1)
    act = ((torch.tensor(db) + 50) / 25).clamp(-3, 3)
    prev = torch.zeros(1, 24, dtype=torch.long)
    whole, _ = m(hu[None], ha[None], act[None], prev)
    state, steps = None, []
    for i in range(24):
        o, state = m(hu[None, i:i + 1], ha[None, i:i + 1], act[None, i:i + 1], prev[:, i:i + 1], state)
        steps.append(o)
    assert torch.allclose(whole, torch.cat(steps, 1), atol=1e-5)
    assert whole.shape == (1, 24, 8) and float(whole[..., 4:].min()) >= 0 and float(whole[..., 4:].max()) <= 1


@torch.no_grad()
def test_gru_batch_rows_are_conversations():
    """Each slot steps as its own conversation: interleaved slots give each one's single-conversation outputs; a
    reset starts its context over; a slot that didn't step keeps its state and reports nothing."""
    m = _model(); gb = GruBatch(m, slots=4, device="cpu")
    convs = {0: _inputs(12, 2), 2: _inputs(12, 3)}
    alone = {}
    for s, (hu, ha, db) in convs.items():                    # each conversation by itself, from a fresh state
        state, outs = None, []
        for i in range(12):
            act = ((torch.tensor([[db[i]]]) + 50) / 25).clamp(-3, 3)
            o, state = m(hu[None, i:i + 1], ha[None, i:i + 1], act, torch.zeros(1, 1, dtype=torch.long), state)
            outs.append(o[0, 0].numpy())
        alone[s] = outs
    pos = {0: 0, 2: 0}
    for t in range(24):                                        # slot 2 starts later and skips a few hops
        live = [s for s in (0, 2) if pos[s] < 12 and not (s == 2 and (t < 3 or t % 4 == 0))]
        if not live:                                           # as the engine: no step without a conversation
            continue
        out = gb.step([(s, convs[s][0][pos[s]], convs[s][1][pos[s]], convs[s][2][pos[s]], 0) for s in live])
        for s in live:
            assert np.allclose(out[s], alone[s][pos[s]], atol=1e-5), (t, s)
            pos[s] += 1
        assert not out[1].any() and not out[3].any()
        if t == 3:
            kept = gb.h[:, 2].clone()
        if t == 4:
            assert torch.equal(gb.h[:, 2], kept)                 # slot 2 was idle at t=4
    assert pos == {0: 12, 2: 12}
    gb.reset_slot(0)                                           # a new turn: slot 0 starts over
    hu, ha, db = convs[0]
    assert np.allclose(gb.step([(0, hu[0], ha[0], db[0], 0)])[0], alone[0][0], atol=1e-5)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
