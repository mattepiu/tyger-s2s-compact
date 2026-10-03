"""Tool-call text splitting for the in-process responder (no GPU). Run: .venv/bin/python tests/test_llm_parse.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.llm import split_speakable  # noqa: E402
from app.tools import parse_calls  # noqa: E402


def stream_speakable(raw, step=2):
    """What LocalLLM.stream would speak, feeding raw text in small deltas."""
    said, buf, saw = "", "", False
    for i in range(0, len(raw), step):
        buf += raw[i:i + step]
        if saw:
            continue
        speak, saw = split_speakable(buf)
        if len(speak) > len(said):
            said = speak
    if not saw:
        said = buf
    return said, saw


def test_plain_text_passes_through():
    assert stream_speakable("Sure, it's 5 < 7 and that's fine.") == ("Sure, it's 5 < 7 and that's fine.", False)


def test_tool_call_never_spoken():
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
    for step in (1, 2, 5):
        assert stream_speakable(raw, step) == ("", True)
    assert parse_calls(raw) == [{"name": "get_weather", "arguments": {"location": "Paris"}}]


def test_text_before_tool_call_is_spoken():
    raw = 'Let me check.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "btc price"}}\n</tool_call>'
    said, saw = stream_speakable(raw, 3)
    assert said.strip() == "Let me check." and saw


def test_xml_tool_call_qwen35():
    raw = ('<tool_call>\n<function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n'
           '<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>')
    assert parse_calls(raw) == [{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}]
    for step in (1, 3, 7):
        assert stream_speakable(raw, step) == ("", True)


def test_xml_tool_call_values_and_multiple():
    raw = ('<tool_call>\n<function=web_search>\n<parameter=query>\nbtc price\n</parameter>\n</function>\n</tool_call>\n'
           '<tool_call>\n<function=x>\n<parameter=n>\n3\n</parameter>\n<parameter=s>\n"quoted"\n</parameter>\n'
           '</function>\n</tool_call>')
    assert parse_calls(raw) == [{"name": "web_search", "arguments": {"query": "btc price"}},
                                {"name": "x", "arguments": {"n": 3, "s": '"quoted"'}}]


def test_partial_tag_held_back():
    assert split_speakable("Okay <tool_c") == ("Okay ", False)
    assert split_speakable("Okay <b") == ("Okay <b", False)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
