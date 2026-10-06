"""EARS's ASR thread job handling (app/ears_core/batched.py), GPU-free: transcription is stubbed, so this checks only
how finals (an endpoint waits on them) and partials (backchannels) are resolved, especially when transcription
fails. Run: .venv/bin/python tests/test_ears_asr_jobs.py"""
import os, sys, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from app.ears_core.batched import EarsEngine, _Job  # noqa: E402

logging.disable(logging.CRITICAL)                  # the failure paths log tracebacks on purpose


class Sess:
    def __init__(self):
        self.turn_gen = 1; self.partial_asr = "old"; self.partial_inflight = True


def engine(transcribe):
    """An EarsEngine with no models: only the ASR job handling runs."""
    e = EarsEngine.__new__(EarsEngine)
    e.transcribe = transcribe
    return e


def final(sess, got):
    return _Job("final", sess, sess.turn_gen, np.zeros(16, np.float32), done=got.append)


def partial(sess, gen=None):
    return _Job("partial", sess, sess.turn_gen if gen is None else gen, np.zeros(16, np.float32))


def test_finals_and_partials_resolved():
    s = Sess(); got = []
    e = engine(lambda wavs: [f"t{i}" for i in range(len(wavs))])
    e._finish_finals([final(s, got)]); e._finish_partials([partial(s)])
    assert got == ["t0"] and s.partial_asr == "t0" and not s.partial_inflight


def test_partials_failing_after_finals_leave_the_finals_alone():
    """The bug: one try around both meant a partials failure re-ran done("") on finals already resolved."""
    s = Sess(); got = []
    calls = []

    def transcribe(wavs):
        calls.append(len(wavs))
        if len(calls) == 2:
            raise RuntimeError("partials failed")
        return ["the transcript"] * len(wavs)
    e = engine(transcribe)
    e._finish_finals([final(s, got)]); e._finish_partials([partial(s)])
    assert got == ["the transcript"], got                       # done() exactly once, with the real text
    assert s.partial_asr == "old" and not s.partial_inflight     # the failed partial keeps the previous one


def test_failed_finals_resolve_empty_once():
    s = Sess(); got = []

    def transcribe(wavs):
        raise RuntimeError("finals failed")
    engine(transcribe)._finish_finals([final(s, got), final(s, got)])
    assert got == ["", ""]


def test_stale_partial_dropped():
    s = Sess()
    engine(lambda wavs: ["new"])._finish_partials([partial(s, gen=0)])   # from a turn that already ended
    assert s.partial_asr == "old" and not s.partial_inflight


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
