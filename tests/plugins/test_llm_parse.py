"""Text-level LLM helpers (no GPU): speakable splitting at a call tag, the tool-call formats (Gemma 4, Hermes
JSON, Qwen XML), the <think> guard, and the neutral -> chat-template history conversion.
Run: .venv/bin/python tests/plugins/test_llm_parse.py"""
import os, sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.plugins.llm._toolparse import split_speakable, parse_calls, strip_think, new_call_id, tool_start, \
    NAME_WAIT, TAG, GEMMA_TAG, gemma_normalize, names_tool, check_arguments, flag_invalid  # noqa: E402
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


# ---- Hermes JSON (the fake LLM; the openai plugin's fallback for calls returned as text) -------------------
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


# ---- Gemma 4 (gemma-4-e4b): special-token tags, <|"|>-quoted strings --------------------------------------
GEMMA = 'Sure.<|tool_call>call:get_weather{location:<|"|>San Francisco, CA<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>'


def test_gemma_parsed_named_never_spoken():
    assert parse_calls(GEMMA, "gemma4") == \
        [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}]
    for step in (1, 3):
        said, saw = "", False
        for i in range(step, len(GEMMA) + step, step):
            speak, saw = split_speakable(GEMMA[:i], GEMMA_TAG)
            assert speak.startswith(said) and "<|" not in speak; said = speak
        assert said == "Sure." and saw
    got, i = None, 0
    while not got:
        i += 1; got = tool_start(GEMMA[:i], GEMMA_TAG)
    assert got == {"name": "get_weather"} and i < len(GEMMA) - 30
    assert tool_start(GEMMA, TAG) is None                            # Hermes's tag is not Gemma's


def test_gemma_values_and_multiple():
    raw = ('<|tool_call>call:ask_expert{question:<|"|>Rent or buy: which, and why?<|"|>}<tool_call|>'
           '<|tool_call>call:x{n:3,ok:true,s:<|"|>a, b<|"|>}<tool_call|>')
    assert parse_calls(raw, "gemma4") == [{"name": "ask_expert", "arguments": {"question": "Rent or buy: which, and why?"}},
                                          {"name": "x", "arguments": {"n": 3, "ok": True, "s": "a, b"}}]


NAMES = ["get_weather", "set_reminder", "send_message", "web_search", "ask_expert"]


def test_gemma_bare_call_is_a_call_never_spoken():
    """Gemma sometimes drops the markers: `set_reminder{...}` as plain text. Parsed as a call, nothing spoken."""
    for raw in ('set_reminder{text:<|"|>call my mom<|"|>,time:<|"|>6 pm<|"|>}<tool_call|>',
                'call:set_reminder{text:<|"|>call my mom<|"|>,time:<|"|>6 pm<|"|>}'):
        said = ""
        for i in range(1, len(raw) + 1):
            speak, saw = split_speakable(gemma_normalize(raw[:i], NAMES), GEMMA_TAG)
            assert speak.startswith(said) and "{" not in speak; said = speak
        assert said == "" and saw
        assert parse_calls(gemma_normalize(raw, NAMES, final=True), "gemma4") == \
            [{"name": "set_reminder", "arguments": {"text": "call my mom", "time": "6 pm"}}]
    assert tool_start(gemma_normalize("send_message{", NAMES), GEMMA_TAG) == {"name": "send_message"}


def test_gemma_normalize_leaves_speech_alone():
    assert gemma_normalize("I'll get the weather for you.", NAMES, final=True) == "I'll get the weather for you."
    assert gemma_normalize("Let me get", NAMES) == "Let me " and gemma_normalize("Let me get", NAMES, True) == "Let me get"
    assert gemma_normalize(GEMMA, NAMES) == GEMMA                       # already tagged: unchanged
    assert gemma_normalize("set_reminder{x}", []) == "set_reminder{x}"  # no tools offered: plain text


# ---- <think> guard (the openai plugin's strip_think: true) -------------------------------------------------
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
    raw = "<think>need weather</think>" + HERMES
    assert stream_speakable(raw, 3, strip_think) == ("", True)
    assert parse_calls(strip_think(raw, True), "hermes_json")[0]["name"] == "get_weather"


QWEN = ("Let me check.<tool_call>\n<function=get_weather>\n<parameter=location>\nParis, France\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")


def test_qwen_xml_parsed_named_never_spoken():
    assert stream_speakable(QWEN, 3, strip_think) == ("Let me check.", True)
    assert parse_calls(QWEN, "qwen_xml") == [{"name": "get_weather",
                                              "arguments": {"location": "Paris, France", "days": 3}}]
    got, at = first_start(QWEN)
    assert got == {"name": "get_weather"} and QWEN[:at].endswith("<function=get_weather>")
    two = QWEN + "<tool_call>\n<function=web_search>\n<parameter=query>\n1, 2: \"x\"\n</parameter>\n</function>\n" \
                 "</tool_call>"
    assert [c["name"] for c in parse_calls(two, "qwen_xml")] == ["get_weather", "web_search"]
    assert parse_calls(two, "qwen_xml")[1]["arguments"] == {"query": '1, 2: "x"'}


def test_qwen_xml_after_an_empty_think_block():
    raw = "<think>\n\n</think>\n\n" + QWEN
    assert stream_speakable(raw, 2, strip_think)[0].strip() == "Let me check."
    assert parse_calls(strip_think(raw, True), "qwen_xml")[0]["name"] == "get_weather"


def test_qwen_xml_unparseable_dropped():
    assert parse_calls("<tool_call>\n{\"name\": \"x\"}\n</tool_call>", "qwen_xml") == []


def test_gemma_thought_channel_never_spoken():
    from app.plugins.llm._toolparse import GEMMA_THOUGHT
    filt = lambda r, final=False: strip_think(r, final, GEMMA_THOUGHT)          # noqa: E731
    split = lambda r: split_speakable(r, GEMMA_TAG)                              # noqa: E731
    for raw, want in (("<|channel>thought\n<channel|>It's 14 degrees in Oslo.", "It's 14 degrees in Oslo."),
                      ("<|channel>thought\nThe user wants the weather.<channel|>Sunny today.", "Sunny today."),
                      ("<|channel>thought\nStill thinking about it", "")):
        said = ""
        for i in range(1, len(raw) + 1):                       # streamed one character at a time
            speak, _ = split(filt(raw[:i]))
            assert speak.startswith(said), (speak, said)       # never takes back what was spoken
            said = speak
        assert filt(raw, True) == want and "channel" not in said, (raw, said)
    raw = "<|channel>thought\n<channel|>" + GEMMA
    assert parse_calls(filt(raw, True), "gemma4")[0]["name"] == "get_weather"


def scripted_stream(plugin_cls, raw, tools=None, step=3):
    """VLLMChatPlugin.stream on a plugin without an engine: the model's raw text arrives in `step`-character deltas."""
    import threading
    p = plugin_cls.__new__(plugin_cls)
    p._tok_lock = threading.Lock(); p.max_tokens = None
    p.tok = type("Tok", (), {"apply_chat_template": lambda self, *a, **k: "prompt"})()
    p.sampling = lambda max_new: None
    p.engine = type("Engine", (), {"abort": lambda self, rid: None})()
    p.eloop = type("Loop", (), {"iterate": lambda self, gen, abort=None, stop_flag=None:
                                iter([raw[i:i + step] for i in range(0, len(raw), step)])})()
    return list(p.stream([{"role": "user", "content": "x"}], tools=tools))


def test_gemma_26b_answer_after_an_unclosed_json_thought():
    """Seen on the box: the thought as a JSON object, then the answer, the channel never closed. The answer is spoken
    (at the end of the stream), the thought never; a call after such a thought is a call, not speech."""
    from app.plugins.llm.gemma4 import Gemma4_26B_A4B
    raw = '<|channel>thought\n{\n  "thought": "No clear answer, {sorry}."\n}Извините, мне не удалось это найти.'
    assert scripted_stream(Gemma4_26B_A4B, raw) == [("text", "Извините, мне не удалось это найти.")]
    raw = '<|channel>thought\n<channel|>It is sunny.'                     # the usual empty channel: streamed
    got = scripted_stream(Gemma4_26B_A4B, raw)
    assert "".join(t for k, t in got if k == "text") == "It is sunny." and len(got) > 1
    raw = '<|channel>thought\n{"thought": "check"}' + GEMMA
    got = scripted_stream(Gemma4_26B_A4B, raw)
    assert got[0] == ("text", "Sure.") and [k for k, *_ in got[1:]] == ["tool_start", "tool_calls"], got
    assert got[2][1][0]["name"] == "get_weather" and "tool_call" not in got[0][1]
    raw = "<|channel>thought\n<tool_response|>Курс сейчас около девяноста двух рублей."     # a stray closing token
    assert scripted_stream(Gemma4_26B_A4B, raw) == [("text", "Курс сейчас около девяноста двух рублей.")]
    assert scripted_stream(Gemma4_26B_A4B, "<|channel>thought\nStill thinking about it") == []   # thinking only


def test_unknown_format_rejected():
    try:
        parse_calls(HERMES, "xml")
    except ValueError:
        return
    raise AssertionError("an unknown format name must not parse silently")


# ---- tool_start names the tool (the filler is picked by it) -------------------------------------------------
def first_start(raw, tag=TAG, step=1):
    """The first tool_start payload while raw streams in, and how many characters had arrived by then."""
    for i in range(step, len(raw) + step, step):
        got = tool_start(raw[:i], tag)
        if got:
            return got, min(i, len(raw))
    return None, len(raw)


def test_tool_start_waits_for_the_whole_name():
    for raw, tag, named in ((HERMES, TAG, '"get_weather"'), (GEMMA, GEMMA_TAG, "call:get_weather{")):
        got, at = first_start(raw, tag)
        assert got == {"name": "get_weather"}, (tag, got)
        assert raw[:at].endswith(named), raw[:at]
        assert at < len(raw) - 20, "named well before the call finished decoding"
    assert tool_start('Okay. <tool_call>\n{"name": "get_wea') is None
    assert tool_start("no call here") is None


def test_name_wait_only_with_ask_expert():
    from app.prompts import default_prompts
    from app.tools import build_tools
    p = default_prompts()
    assert not names_tool(build_tools(p)) and names_tool(build_tools(p, expert=True)) and not names_tool(None)


def test_tool_start_gives_up_on_a_garbled_call():
    raw = "<tool_call>" + "x" * NAME_WAIT
    assert tool_start(raw[:-1]) is None and tool_start(raw) == {"name": None}


# ---- IDs and history conversion ----------------------------------------------------------------------------
def test_call_ids_unique():
    ids = {new_call_id() for _ in range(1000)}
    assert len(ids) == 1000 and all(i.startswith("call_") and len(i) == 13 for i in ids)


def test_check_arguments():
    """The one schema check of a call's arguments (both hosted plugins)."""
    from app.prompts import default_prompts
    from app.tools import build_tools
    tools = build_tools(default_prompts())
    assert check_arguments("get_weather", {"location": "Oslo"}, tools) is None
    assert "missing location" in check_arguments("get_weather", {"unit": "celsius"}, tools)
    assert "not one of" in check_arguments("get_weather", {"location": "Oslo", "unit": "kelvin"}, tools)
    assert "not a string" in check_arguments("web_search", {"query": 5}, tools)
    assert check_arguments("web_search", "btc", tools) == "arguments are not an object"
    assert check_arguments("fly", {}, tools) == "unknown tool"
    ok = {"id": "call_1", "name": "web_search", "arguments": {"query": "btc"}}
    assert flag_invalid(ok, tools) == ok
    bad = flag_invalid({"id": "call_2", "name": "web_search", "arguments": ["btc"]}, tools)
    assert bad["arguments"] == {} and bad["error"].startswith("invalid arguments for 'web_search'")


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
