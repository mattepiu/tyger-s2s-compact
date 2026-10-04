"""Text-level LLM helpers (no GPU): speakable splitting at <tool_call>, the two tool-call formats, the <think>
guard, and the neutral -> chat-template history conversion. Run: .venv/bin/python tests/test_llm_parse.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.plugins.llm._toolparse import split_speakable, parse_calls, strip_think, new_call_id, tool_start, \
    NAME_WAIT  # noqa: E402
from app.plugins.llm.vllm_base import to_template_messages  # noqa: E402


def stream_speakable(raw, step=2, filt=lambda r, final=False: r):
    """What VLLMChatPlugin.stream would speak, feeding raw text in small deltas."""
    said, buf, saw = "", "", False
    for i in range(0, len(raw), step):
        buf += raw[i:i + step]
        if saw:
            continue
        speak, saw = split_speakable(filt(buf))
        assert speak.startswith(said), (speak, said)               # never takes back what was spoken
        if len(speak) > len(said):
            said = speak
    if not saw:
        said = filt(buf, True)
    return said, saw


# ---- splitting (both formats) ----------------------------------------------------------------------------
def test_plain_text_passes_through():
    assert stream_speakable("Sure, it's 5 < 7 and that's fine.") == ("Sure, it's 5 < 7 and that's fine.", False)


def test_partial_tag_held_back():
    assert split_speakable("Okay <tool_c") == ("Okay ", False)
    assert split_speakable("Okay <b") == ("Okay <b", False)


# ---- Hermes JSON (qwen3-4b-2507) ---------------------------------------------------------------------------
HERMES = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'


def test_hermes_never_spoken_and_parsed():
    for step in (1, 2, 5):
        assert stream_speakable(HERMES, step) == ("", True)
    assert parse_calls(HERMES, "hermes_json") == [{"name": "get_weather", "arguments": {"location": "Paris"}}]


def test_hermes_text_before_call_is_spoken():
    raw = 'Let me check.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "btc price"}}\n</tool_call>'
    said, saw = stream_speakable(raw, 3)
    assert said.strip() == "Let me check." and saw
    assert parse_calls(raw, "hermes_json") == [{"name": "web_search", "arguments": {"query": "btc price"}}]


def test_hermes_unparseable_dropped():
    assert parse_calls("<tool_call>{not json</tool_call>" + HERMES, "hermes_json") == \
        [{"name": "get_weather", "arguments": {"location": "Paris"}}]


# ---- Qwen XML (qwen3.5-4b) ---------------------------------------------------------------------------------
XML = ('<tool_call>\n<function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n'
       '<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>')


def test_xml_never_spoken_and_parsed():
    assert parse_calls(XML, "qwen_xml") == \
        [{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}]
    for step in (1, 3, 7):
        assert stream_speakable(XML, step) == ("", True)


def test_xml_values_and_multiple():
    raw = ('<tool_call>\n<function=web_search>\n<parameter=query>\nbtc price\n</parameter>\n</function>\n</tool_call>\n'
           '<tool_call>\n<function=x>\n<parameter=n>\n3\n</parameter>\n<parameter=s>\n"quoted"\n</parameter>\n'
           '</function>\n</tool_call>')
    assert parse_calls(raw, "qwen_xml") == [{"name": "web_search", "arguments": {"query": "btc price"}},
                                            {"name": "x", "arguments": {"n": 3, "s": '"quoted"'}}]


def test_other_format_still_parses():
    assert parse_calls(HERMES, "qwen_xml") == parse_calls(HERMES, "hermes_json")
    assert parse_calls(XML, "hermes_json") == parse_calls(XML, "qwen_xml")


# ---- <think> guard (qwen3.5-4b) ----------------------------------------------------------------------------
def test_think_never_spoken():
    raw = "<think>\nthe user wants a joke\n</think>\n\nWhy did the chicken cross the road? To get across."
    for step in (1, 2, 5):
        said, saw = stream_speakable(raw, step, strip_think)
        assert said.strip() == "Why did the chicken cross the road? To get across." and not saw


def test_think_unclosed_and_stray():
    assert strip_think("Hi. <think>planning more") == "Hi. "
    assert strip_think("</think>\n\nHello there.") == "\n\nHello there."      # thinking opened in the prompt
    assert strip_think("Hello <thi") == "Hello " and strip_think("Hello <thi", final=True) == "Hello <thi"
    assert strip_think("5 < 7, fine.") == "5 < 7, fine."


def test_think_then_tool_call():
    raw = "<think>need weather</think>" + XML
    assert stream_speakable(raw, 3, strip_think) == ("", True)
    assert parse_calls(strip_think(raw, True), "qwen_xml")[0]["name"] == "get_weather"


# ---- tool_start names the tool (the filler is picked by it) -------------------------------------------------
def first_start(raw, step=1):
    """The first tool_start payload while raw streams in, and how many characters had arrived by then."""
    for i in range(step, len(raw) + step, step):
        got = tool_start(raw[:i])
        if got:
            return got, min(i, len(raw))
    return None, len(raw)


def test_tool_start_waits_for_the_whole_name():
    for raw, fmt in ((HERMES, "hermes"), (XML, "xml")):
        got, at = first_start(raw)
        assert got == {"name": "get_weather"}, (fmt, got)
        assert raw[:at].endswith('"get_weather"' if fmt == "hermes" else "<function=get_weather>"), raw[:at]
        assert at < len(raw) - 20, "named well before the call finished decoding"
    assert tool_start('Okay. <tool_call>\n{"name": "get_wea') is None
    assert tool_start("no call here") is None


def test_tool_start_gives_up_on_a_garbled_call():
    raw = "<tool_call>" + "x" * NAME_WAIT
    assert tool_start(raw[:-1]) is None and tool_start(raw) == {"name": None}


# ---- IDs and history conversion ----------------------------------------------------------------------------
def test_call_ids_unique():
    ids = {new_call_id() for _ in range(1000)}
    assert len(ids) == 1000 and all(i.startswith("call_") and len(i) == 13 for i in ids)


def test_template_messages_are_the_old_nested_shape():
    neutral = [{"role": "user", "content": "weather?"},
               {"role": "assistant", "content": None, "native": {"x": 1},
                "tool_calls": [{"id": "call_ab12cd34", "name": "get_weather", "arguments": {"location": "Oslo"}}]},
               {"role": "tool", "tool_call_id": "call_ab12cd34", "content": "{}"},
               {"role": "assistant", "content": "Sunny."}]
    assert to_template_messages(neutral) == [
        neutral[0],
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_ab12cd34", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"location": "Oslo"}}}]},
        neutral[2], neutral[3]]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
