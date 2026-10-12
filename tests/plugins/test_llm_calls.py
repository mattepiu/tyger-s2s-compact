"""What every tool-call format shares (app/plugins/llm/calls.py), on a toy format no model writes, so nothing here
depends on a real model's tag: speech split from a call at its tag, a partial tag held back, the tool named early,
call ids and the argument check. Also the local plugin base: a local plugin must declare its call format, and its
stream runs on any format; the neutral -> chat-template history conversion. No GPU.
Each real format has its own file: test_calls_gemma4.py, test_calls_hermes.py.
Run: .venv/bin/python tests/plugins/test_llm_calls.py"""
import inspect, os, re, sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from fakes import scripted_stream, stream_speakable  # noqa: E402
from app.plugins.llm.calls import CallFormat, NAME_WAIT, check_arguments, flag_invalid, names_tool, new_call_id, \
    split_speakable, tool_start  # noqa: E402
from app.plugins.llm.local.vllm_base import VLLMChatPlugin, to_template_messages  # noqa: E402


class ToyCalls(CallFormat):
    """<<call>>NAME(key=value, ...)<<end>>"""
    tag = "<<call>>"
    name = re.compile(r"\s*(\w+)\(")
    CALL = re.compile(r"<<call>>(\w+)\((.*?)\)<<end>>", re.S)

    def parse(self, raw):
        return [{"name": m.group(1), "arguments": dict(kv.split("=", 1) for kv in m.group(2).split(", ") if kv)}
                for m in self.CALL.finditer(raw)]


TOY = ToyCalls()
CALL = "<<call>>get_current_time(timezone=Europe/Oslo)<<end>>"


# ---- splitting ------------------------------------------------------------------------------------------------
def test_plain_text_passes_through():
    assert stream_speakable("Sure, it's 5 < 7 and that's fine.", TOY) == ("Sure, it's 5 < 7 and that's fine.", False)


def test_partial_tag_held_back():
    assert split_speakable("Okay <<ca", TOY) == ("Okay ", False)
    assert split_speakable("Okay <b", TOY) == ("Okay <b", False)


def test_call_never_spoken_text_before_it_is():
    for step in (1, 2, 5):
        assert stream_speakable(CALL, TOY, step) == ("", True)
    said, saw = stream_speakable("Let me check. " + CALL, TOY, 3)
    assert said.strip() == "Let me check." and saw


# ---- tool_start names the tool (the filler is picked by it) -------------------------------------------------
def first_start(raw, fmt, step=1):
    """The first tool_start payload while raw streams in, and how many characters had arrived by then."""
    for i in range(step, len(raw) + step, step):
        got = tool_start(raw[:i], fmt)
        if got:
            return got, min(i, len(raw))
    return None, len(raw)


def test_tool_start_waits_for_the_whole_name():
    got, at = first_start(CALL, TOY)
    assert got == {"name": "get_current_time"} and CALL[:at].endswith("get_current_time(")
    assert at < len(CALL) - 20, "named well before the call finished decoding"
    assert tool_start("Okay. <<call>>get_curr", TOY) is None
    assert tool_start("no call here", TOY) is None


def test_tool_start_gives_up_on_a_garbled_call():
    raw = TOY.tag + "x" * NAME_WAIT
    assert tool_start(raw[:-1], TOY) is None and tool_start(raw, TOY) == {"name": None}


def test_name_wait_only_with_ask_expert():
    from app.prompts import default_prompts
    from app.tools import build_tools
    p = default_prompts()
    assert not names_tool(build_tools(p)) and names_tool(build_tools(p, expert=True)) and not names_tool(None)


# ---- the local plugin base ------------------------------------------------------------------------------------
class ToyLocal(VLLMChatPlugin):
    calls = TOY


def test_a_local_plugin_declares_its_call_format():
    """VLLMChatPlugin has no default format: a local plugin without one can't be built (it fails before any engine
    loads), and one that declares it can."""
    bare = type("NoFormat", (VLLMChatPlugin,), {})
    assert inspect.isabstract(bare) and not inspect.isabstract(ToyLocal)
    try:
        bare({"plugin": "x"}, None)
    except TypeError as e:
        assert "calls" in str(e)
    else:
        raise AssertionError("built a local plugin with no call format")


def test_local_stream_on_any_format():
    """The local plugin base runs on whatever format the plugin declares: speech before the call, the tool named,
    the call parsed with a fresh id, nothing of it spoken."""
    from app.prompts import default_prompts
    from app.tools import build_tools
    for step in (1, 3, 100):
        got = scripted_stream(ToyLocal, "Okay. " + CALL, tools=build_tools(default_prompts(), expert=True), step=step)
        assert "".join(e[1] for e in got if e[0] == "text") == "Okay. ", (step, got)
        assert [e for e in got if e[0] == "tool_start"] == [("tool_start", {"name": "get_current_time"})], got
        (calls,) = [e[1] for e in got if e[0] == "tool_calls"]
        assert [(c["name"], c["arguments"]) for c in calls] == [("get_current_time", {"timezone": "Europe/Oslo"})]
        assert calls[0]["id"].startswith("call_")


# ---- IDs, the argument check, history conversion ------------------------------------------------------------
def test_call_ids_unique():
    ids = {new_call_id() for _ in range(1000)}
    assert len(ids) == 1000 and all(i.startswith("call_") and len(i) == 13 for i in ids)


def test_check_arguments():
    """The one schema check of a call's arguments (both hosted plugins)."""
    from app.prompts import default_prompts
    from app.tools import build_tools
    tools = build_tools(default_prompts())
    assert check_arguments("set_reminder", {"text": "call mom", "time": "18:00"}, tools) is None
    assert "missing time" in check_arguments("set_reminder", {"text": "call mom"}, tools)
    assert "not a string" in check_arguments("get_current_time", {"timezone": 5}, tools)
    assert check_arguments("get_current_time", "Oslo", tools) == "arguments are not an object"
    assert check_arguments("fly", {}, tools) == "unknown tool"
    unit = [{"type": "function", "function": {"name": "convert", "parameters": {"type": "object", "properties": {
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["unit"]}}}]
    assert "not one of" in check_arguments("convert", {"unit": "kelvin"}, unit)
    ok = {"id": "call_1", "name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}
    assert flag_invalid(ok, tools) == ok
    bad = flag_invalid({"id": "call_2", "name": "get_current_time", "arguments": ["Tokyo"]}, tools)
    assert bad["arguments"] == {} and bad["error"].startswith("invalid arguments for 'get_current_time'")


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
