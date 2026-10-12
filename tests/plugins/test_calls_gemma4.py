"""Gemma 4's call format (app/plugins/llm/local/gemma4/calls.py): special-token tags, <|"|>-quoted strings, the
markers Gemma sometimes drops, and gemma-4-e4b's stream on it. No GPU.
Run: .venv/bin/python tests/plugins/test_calls_gemma4.py"""
import os, sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from fakes import scripted_stream  # noqa: E402
from app.plugins.llm.calls import split_speakable, tool_start  # noqa: E402
from app.plugins.llm.fake import HERMES  # noqa: E402
from app.plugins.llm.local.gemma4.calls import GemmaCalls  # noqa: E402
from app.plugins.llm.local.gemma4.plugin import Gemma4E4B  # noqa: E402

GEMMA = GemmaCalls()
RAW = 'Sure.<|tool_call>call:get_weather{location:<|"|>San Francisco, CA<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>'
NAMES = ["get_weather", "set_reminder", "send_message", "web_search", "ask_expert"]


def test_gemma_parsed_named_never_spoken():
    assert GEMMA.parse(RAW) == \
        [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}]
    for step in (1, 3):
        said, saw = "", False
        for i in range(step, len(RAW) + step, step):
            speak, saw = split_speakable(RAW[:i], GEMMA)
            assert speak.startswith(said) and "<|" not in speak; said = speak
        assert said == "Sure." and saw
    got, i = None, 0
    while not got:
        i += 1; got = tool_start(RAW[:i], GEMMA)
    assert got == {"name": "get_weather"} and RAW[:i].endswith("call:get_weather{") and i < len(RAW) - 30
    assert tool_start(RAW, HERMES) is None                           # Hermes's tag is not Gemma's


def test_gemma_values_and_multiple():
    raw = ('<|tool_call>call:ask_expert{question:<|"|>Rent or buy: which, and why?<|"|>}<tool_call|>'
           '<|tool_call>call:x{n:3,ok:true,s:<|"|>a, b<|"|>}<tool_call|>')
    assert GEMMA.parse(raw) == [{"name": "ask_expert", "arguments": {"question": "Rent or buy: which, and why?"}},
                                {"name": "x", "arguments": {"n": 3, "ok": True, "s": "a, b"}}]


def test_gemma_bare_call_is_a_call_never_spoken():
    """Gemma sometimes drops the markers: `set_reminder{...}` as plain text. Parsed as a call, nothing spoken."""
    for raw in ('set_reminder{text:<|"|>call my mom<|"|>,time:<|"|>6 pm<|"|>}<tool_call|>',
                'call:set_reminder{text:<|"|>call my mom<|"|>,time:<|"|>6 pm<|"|>}'):
        said = ""
        for i in range(1, len(raw) + 1):
            speak, saw = split_speakable(GEMMA.normalize(raw[:i], NAMES), GEMMA)
            assert speak.startswith(said) and "{" not in speak; said = speak
        assert said == "" and saw
        assert GEMMA.parse(GEMMA.normalize(raw, NAMES, final=True)) == \
            [{"name": "set_reminder", "arguments": {"text": "call my mom", "time": "6 pm"}}]
    assert tool_start(GEMMA.normalize("send_message{", NAMES), GEMMA) == {"name": "send_message"}


def test_gemma_normalize_leaves_speech_alone():
    assert GEMMA.normalize("I'll get the weather for you.", NAMES, final=True) == "I'll get the weather for you."
    assert GEMMA.normalize("Let me get", NAMES) == "Let me " and GEMMA.normalize("Let me get", NAMES, True) == "Let me get"
    assert GEMMA.normalize(RAW, NAMES) == RAW                           # already tagged: unchanged
    assert GEMMA.normalize("set_reminder{x}", []) == "set_reminder{x}"  # no tools offered: plain text


def test_gemma_plugin_declares_its_format():
    assert isinstance(Gemma4E4B.calls, GemmaCalls)


def test_gemma_plugin_stream_bare_call_at_the_end():
    """gemma-4-e4b's stream: speech, then a bare `set_reminder{...}` as the very last text, in any deltas (one
    character at a time, or the whole call in the final one): a call, never spoken."""
    from app.prompts import default_prompts
    from app.tools import build_tools
    raw = 'Okay. set_reminder{text:<|"|>call my mom<|"|>,time:<|"|>6 pm<|"|>}'
    for step in (1, 3, len(raw)):
        got = scripted_stream(Gemma4E4B, raw, tools=build_tools(default_prompts()), step=step)
        assert "".join(e[1] for e in got if e[0] == "text").strip() == "Okay.", (step, got)
        assert [e[0] for e in got if e[0] != "text"] == ["tool_start", "tool_calls"], (step, got)
        assert got[-1][1] and got[-1][1][0]["name"] == "set_reminder" and \
            got[-1][1][0]["arguments"] == {"text": "call my mom", "time": "6 pm"}, (step, got)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
