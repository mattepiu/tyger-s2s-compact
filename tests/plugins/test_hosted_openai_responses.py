"""The `openai-responses` LLM plugin without a network (no GPU, no key, no cost): history and tool conversion, the
request it sends (OpenAI's web search tool, store off, reasoning), the stream parser on synthetic event sequences,
and the plugin driven by a stub client (stop closes the stream, errors become LLMError).
Run: .venv/bin/python tests/plugins/test_hosted_openai_responses.py"""
import asyncio, json, os, sys, time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.plugins.base import LLMError, PluginContext  # noqa: E402
from app.plugins.llm.hosted.openai_responses import (OpenAIResponsesLLM, ResponsesStreamParser,  # noqa: E402
                                              to_responses_input, to_responses_tools)
from app.runtime.event_loop import EngineLoop  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_current_time", "description": "Get the current time in a timezone or city.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA tz"}},
                   "required": ["timezone"]}}}]


def text(d):
    return {"type": "response.output_text.delta", "delta": d}


def call_added(name):
    return {"type": "response.output_item.added", "item": {"type": "function_call", "name": name, "arguments": ""}}


def call_done(name, args):
    return {"type": "response.output_item.done", "item": {"type": "function_call", "name": name, "arguments": args}}


SEARCH = {"type": "response.output_item.done", "item": {"type": "web_search_call", "status": "completed"}}
DONE = {"type": "response.completed", "response": {}}


def parse(events, tools=TOOLS):
    p = ResponsesStreamParser(tools); out = []
    for e in events:
        out += p.feed(e)
    return out + p.finish()


# ---- conversion -------------------------------------------------------------------------------------------
def test_history_to_responses_input():
    neutral = [{"role": "system", "content": "sys"}, {"role": "user", "content": "time in Oslo and Rome?"},
               {"role": "assistant", "content": None, "native": {"x": 1},
                "tool_calls": [{"id": "call_a", "name": "get_current_time", "arguments": {"timezone": "Europe/Oslo"}},
                               {"id": "call_b", "name": "get_current_time", "arguments": {"timezone": "Europe/Rome"}}]},
               {"role": "tool", "tool_call_id": "call_a", "content": "{}"},
               {"role": "tool", "tool_call_id": "call_b", "content": "{}"},
               {"role": "assistant", "content": "Ten past nine in both. —"}]
    out = to_responses_input(neutral)
    assert out[0] == {"role": "system", "content": "sys"} and out[1]["role"] == "user"
    assert out[2] == {"type": "function_call", "call_id": "call_a", "name": "get_current_time",
                      "arguments": '{"timezone": "Europe/Oslo"}'} and out[3]["call_id"] == "call_b"
    assert out[4] == {"type": "function_call_output", "call_id": "call_a", "output": "{}"}
    assert out[6] == {"role": "assistant", "content": "Ten past nine in both. —"} and "native" not in json.dumps(out)


def test_tools_flattened_with_web_search():
    out = to_responses_tools(TOOLS, web_search=True)
    assert out[0] == {"type": "function", "name": "get_current_time", "description": TOOLS[0]["function"]["description"],
                      "parameters": TOOLS[0]["function"]["parameters"]}
    assert out[1] == {"type": "web_search"} and to_responses_tools(None, web_search=False) == []


# ---- the stream parser --------------------------------------------------------------------------------------
def test_text_with_searches_in_between():
    evs = parse([SEARCH, text("It's 14 degrees "), text("and cloudy in Oslo."), DONE])
    assert evs == [("text", "It's 14 degrees "), ("text", "and cloudy in Oslo.")]


def test_function_call_named_at_once_and_parsed():
    evs = parse([text("One moment. "), call_added("get_current_time"),
                 call_done("get_current_time", '{"timezone": "Asia/Tokyo"}'), DONE])
    assert evs[:2] == [("text", "One moment. "), ("tool_start", {"name": "get_current_time"})]
    kind, calls, native = evs[2]
    assert kind == "tool_calls" and native is None and len(calls) == 1
    assert calls[0]["name"] == "get_current_time" and calls[0]["arguments"] == {"timezone": "Asia/Tokyo"}
    assert calls[0]["id"].startswith("call_") and "error" not in calls[0]


def test_bad_arguments_flagged_and_cut_off_call_not_run():
    evs = parse([call_added("get_current_time"), call_done("get_current_time", '{"tz": 1}'), DONE])
    assert "error" in evs[1][1][0]                       # doesn't fit the schema: returned, never run
    cut = {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
    evs = parse([call_added("get_current_time"), call_done("get_current_time", '{"timezone": "Eu'), cut])
    assert evs == [("tool_start", {"name": "get_current_time"}), ("tool_calls", [], None)]


def test_refusal_and_failure():
    assert parse([{"type": "response.refusal.delta", "delta": "I can't"}, DONE]) == [("refusal", "refusal")]
    try:
        parse([text("Hi"), {"type": "response.failed", "response": {}}])
        raise AssertionError("a failed response must raise")
    except LLMError:
        pass


# ---- the plugin on a stub client ----------------------------------------------------------------------------
class StubStream:
    def __init__(self, events, delay=0.0):
        self.events, self.delay, self.closed = list(events), delay, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        return self.events.pop(0)

    async def close(self):
        self.closed = True


def make_plugin(**conf):
    os.environ["TYGER_TEST_KEY"] = "test-not-a-key"
    return OpenAIResponsesLLM({"model": "stub", "base_url": "http://127.0.0.1:9", "api_key_env": "TYGER_TEST_KEY",
                               **conf}, PluginContext(EngineLoop(), {}, "stub"))


def stub_create(p, make_stream, record=None):
    async def create(**kw):
        if record is not None:
            record.append(kw)
        return make_stream()
    p.client.responses.create = create


def test_plugin_sends_web_search_store_off_reasoning():
    p = make_plugin(max_tokens=600, reasoning_effort="low"); sent = []
    stub_create(p, lambda: StubStream([text("Sunny."), DONE]), sent)
    evs = list(p.stream([{"role": "user", "content": "weather in Oslo?"}], tools=TOOLS, max_new=12))
    assert evs == [("text", "Sunny.")]
    kw = sent[0]
    assert kw["stream"] and kw["store"] is False and kw["max_output_tokens"] == 600
    assert kw["tools"][-1] == {"type": "web_search"} and kw["tools"][0]["name"] == "get_current_time"
    assert kw["reasoning"] == {"effort": "low"} and "temperature" not in kw
    p2 = make_plugin(web_search=False); sent2 = []
    stub_create(p2, lambda: StubStream([text("ok"), DONE]), sent2)
    list(p2.stream([{"role": "user", "content": "hi"}], max_new=12))
    assert "tools" not in sent2[0] and "reasoning" not in sent2[0] and sent2[0]["max_output_tokens"] == 12


def test_plugin_stop_closes_stream():
    p = make_plugin(max_concurrency=1, queue_timeout_s=0.5); streams = []

    def mk():
        streams.append(StubStream([text(f"word{i} ") for i in range(200)], delay=0.01)); return streams[-1]
    stub_create(p, mk)
    got = []
    for ev in p.stream([{"role": "user", "content": "hi"}], stop_flag=lambda: len(got) >= 3):
        got.append(ev)
    time.sleep(0.1)                                  # the close runs on the engine loop
    assert len(got) <= 4 and streams[0].closed and len(streams[0].events) > 150
    again = []
    for ev in p.stream([{"role": "user", "content": "again"}], stop_flag=lambda: bool(again)):
        again.append(ev)
    assert again and len(streams) == 2               # the stopped request gave its one slot back


def test_plugin_errors_become_llmerror():
    import httpx, openai
    p = make_plugin()

    async def create(**kw):
        raise openai.APIStatusError("boom", response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
                                    body=None)
    p.client.responses.create = create
    try:
        list(p.stream([{"role": "user", "content": "hi"}]))
        raise AssertionError("an HTTP 500 must raise LLMError")
    except LLMError as e:
        assert str(e) == "server"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
