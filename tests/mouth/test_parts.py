"""MOUTH's small parts on their own (no GPU, no threads): History (turns trimmed whole, barged turns marked) and
Speaker (what was said when, what the user heard at a barge, playback timing). Run:
.venv/bin/python tests/mouth/test_parts.py"""
import os, sys, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from fakes import FakeTTS, PROMPTS  # noqa: E402
from app.mouth.history import History  # noqa: E402
from app.mouth.speech import Speaker, speech_audio  # noqa: E402

INTERRUPTED = PROMPTS.text("markers", "interrupted")


def test_history_trims_whole_turns_and_skips_empty_users():
    h = History(2, PROMPTS)
    call = [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "name": "x", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    h.add("one", "a"); h.add("two", "b", call); h.add("three", "c"); h.add("   ", "ignored")
    assert [t[0]["content"] for t in h.turns] == ["two", "three"]
    assert [m["role"] for m in h.messages] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert h.recent(1) == h.turns[-1:] and h.recent(0) == []
    h.add("four", "")
    assert h.messages[-1]["content"] == PROMPTS.text("markers", "empty_reply")


def test_history_barged_turn():
    h = History(8, PROMPTS)
    h.add_barged("tell me a story", ["Once upon a time, ", "there was a fox."])
    assert h.messages[-1]["content"] == "Once upon a time, there was a fox." + INTERRUPTED
    h.add_barged("again", [])
    assert h.messages[-1]["content"] == INTERRUPTED.strip()


def test_speaker_tracks_lines_and_what_was_heard():
    sp = Speaker(FakeTTS(n=2, chunk_s=0.5), sr=24000)
    for line in ("First line.", "Second line."):
        assert [k for k, _ in sp.say(line, None, lambda: False)] == ["audio", "audio"]
    assert sp.said == [("First line.", 0.0), ("Second line.", 1.0)] and sp.audio_s == 2.0
    assert sp.heard(0.5) == ["First line."] and sp.heard(1.2) == ["First line.", "Second line."]
    assert sp.heard(None) == ["First line.", "Second line."]


def test_speaker_playback_timing():
    sp = Speaker(FakeTTS(), sr=24000)
    assert sp.played_out_for(0.0)
    sp.add_audio(np.zeros(2400, np.float32))                  # 0.1 s queued from now
    assert not sp.played_out_for(0.0)
    time.sleep(0.12)
    assert sp.played_out_for(0.0) and not sp.played_out_for(0.5)


def test_nothing_speakable_never_reaches_the_tts():
    tts = FakeTTS(n=1)
    assert list(speech_audio(tts, " — ", None, lambda: False)) == [] and tts.requests == []
    assert len(list(speech_audio(tts, "Hello.", None, lambda: False))) == 1 and tts.requests == ["Hello."]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
