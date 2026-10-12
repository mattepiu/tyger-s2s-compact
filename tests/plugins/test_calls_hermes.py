"""Hermes JSON, the fake LLM's call format (HermesCalls in app/plugins/llm/fake.py). No GPU.
Run: .venv/bin/python tests/plugins/test_calls_hermes.py"""
import os, sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from fakes import stream_speakable  # noqa: E402
from app.plugins.llm.calls import tool_start  # noqa: E402
from app.plugins.llm.fake import HERMES  # noqa: E402

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'


def test_hermes_never_spoken_and_parsed():
    for step in (1, 2, 5):
        assert stream_speakable(CALL, HERMES, step) == ("", True)
    assert HERMES.parse(CALL) == [{"name": "get_weather", "arguments": {"location": "Paris"}}]


def test_hermes_text_before_call_is_spoken():
    raw = 'Let me check.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "btc price"}}\n</tool_call>'
    said, saw = stream_speakable(raw, HERMES, 3)
    assert said.strip() == "Let me check." and saw
    assert HERMES.parse(raw) == [{"name": "web_search", "arguments": {"query": "btc price"}}]


def test_hermes_unparseable_dropped():
    assert HERMES.parse("<tool_call>{not json</tool_call>" + CALL) == \
        [{"name": "get_weather", "arguments": {"location": "Paris"}}]


def test_hermes_tool_start_waits_for_the_whole_name():
    for i in range(1, len(CALL) + 1):
        got = tool_start(CALL[:i], HERMES)
        if got:
            break
    assert got == {"name": "get_weather"} and CALL[:i].endswith('"get_weather"') and i < len(CALL) - 20
    assert tool_start('Okay. <tool_call>\n{"name": "get_wea', HERMES) is None


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
