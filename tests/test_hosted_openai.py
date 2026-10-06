"""The `openai` LLM plugin without a network (no GPU, no key, no cost): message conversion, the stream parser on
synthetic chunk sequences, and the plugin driven by a stub client (stop closes the stream, errors become
LLMError, the concurrency cap). Run: .venv/bin/python tests/test_hosted_openai.py"""
import asyncio, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.plugins.base import LLMError, PluginContext  # noqa: E402
from app.plugins.llm.openai_compat import (ChatStreamParser, OpenAICompatLLM, check_arguments,  # noqa: E402
                                           to_openai_messages)
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.engines import EngineLoop  # noqa: E402

TOOLS = build_tools(default_prompts())


def chunk(content=None, tool=None, finish=None, refusal=None, **extra):
    """A chat-completions stream chunk as a plain dict (the parser reads dicts and SDK objects alike)."""
    delta = {"content": content, "refusal": refusal, **extra}
    if tool:
        delta["tool_calls"] = [tool]
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def tc(index=0, id=None, name=None, args=None):
    return {"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": args}}


def parse(chunks, tools=TOOLS):
    p = ChatStreamParser(tools); evs = []
    for c in chunks:
        evs += p.feed(c)
    return evs + p.finish()


# ---- conversion -------------------------------------------------------------------------------------------
def test_messages_to_openai():
    neutral = [{"role": "system", "content": "sys"}, {"role": "user", "content": "weather in Oslo and Rome?"},
               {"role": "assistant", "content": None, "native": {"x": 1},
                "tool_calls": [{"id": "call_a", "name": "get_weather", "arguments": {"location": "Oslo"}},
                               {"id": "call_b", "name": "get_weather", "arguments": {"location": "Rome"}}]},
               {"role": "tool", "tool_call_id": "call_a", "content": "{}"},
               {"role": "tool", "tool_call_id": "call_b", "content": "{}"},
               {"role": "assistant", "content": "Both sunny. —"}]
    out = to_openai_messages(neutral)
    assert out[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Oslo"}'}},
        {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Rome"}'}}]}
    assert out[3:5] == neutral[3:5] and out[5] == neutral[5] and "native" not in json.dumps(out)


# ---- the stream parser --------------------------------------------------------------------------------------
def test_text_only():
    evs = parse([chunk("Otters "), chunk("hold hands while sleeping."), chunk(finish="stop")])
    assert evs == [("text", "Otters "), ("text", "hold hands while sleeping.")]


def test_text_then_tool_call_split_across_chunks():
    evs = parse([chunk("Let me check. "), chunk(tool=tc(0, "srv_1", "get_weather", "")),
                 chunk(tool=tc(0, None, None, '{"loca')), chunk(tool=tc(0, None, None, 'tion": "Oslo"}')),
                 chunk(finish="tool_calls")])
    assert evs[0] == ("text", "Let me check. ") and evs[1] == ("tool_start", {"name": "get_weather"})
    kind, calls, native = evs[-1]
    assert kind == "tool_calls" and native is None and len(evs) == 3
    assert calls[0]["name"] == "get_weather" and calls[0]["arguments"] == {"location": "Oslo"}
    assert calls[0]["id"].startswith("call_") and calls[0]["id"] != "srv_1" and "error" not in calls[0]


def test_two_tool_calls_by_index_unique_ids():
    evs = parse([chunk(tool=tc(0, "x", "get_weather", '{"location": "Oslo"}')),
                 chunk(tool=tc(1, "x", "get_weather", '{"location": "Rome"}')), chunk(finish="tool_calls")])
    calls = evs[-1][1]
    assert [c["arguments"]["location"] for c in calls] == ["Oslo", "Rome"] and calls[0]["id"] != calls[1]["id"]
    assert [e[0] for e in evs].count("tool_start") == 1


def test_invalid_arguments_flagged_not_dropped():
    evs = parse([chunk(tool=tc(0, "x", "get_weather", '{"unit": "celsius"}')), chunk(finish="tool_calls")])
    c = evs[-1][1][0]
    assert "missing location" in c["error"] and c["arguments"] == {"unit": "celsius"}
    evs = parse([chunk(tool=tc(0, "x", "web_search", '{"query": "bt')), chunk(finish="tool_calls")])
    assert "not an object" in evs[-1][1][0]["error"]
    assert check_arguments("fly", {}, TOOLS) == "unknown tool"


def test_truncated_tool_call_not_run():
    evs = parse([chunk(tool=tc(0, "x", "get_weather", '{"location": "Os')), chunk(finish="length")])
    assert evs[-1] == ("tool_calls", [], None)


def test_text_written_tool_call_parsed():
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Lima"}}\n</tool_call>'
    evs = parse([chunk(raw[:15]), chunk(raw[15:]), chunk(finish="stop")])
    assert evs[0] == ("tool_start", {"name": "get_weather"}) and evs[-1][1][0]["arguments"] == {"location": "Lima"}
    assert not any(e[0] == "text" for e in evs)


def test_think_in_content_never_spoken_reasoning_ignored():
    evs = parse([chunk("<think>the user ", reasoning_content="hidden"), chunk("wants a fact</think>\n\n"),
                 chunk("Otters are cute."), chunk(finish="stop")])
    text = "".join(e[1] for e in evs if e[0] == "text")
    assert text.strip() == "Otters are cute." and "hidden" not in text


def test_refusal():
    assert parse([chunk(refusal="I can't help with that."), chunk(finish="stop")]) == [("refusal", None)]


# ---- the plugin, on a stub client ---------------------------------------------------------------------------
class StubStream:
    def __init__(self, chunks, delay=0.0):
        self.chunks, self.delay, self.closed = list(chunks), delay, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        return self.chunks.pop(0)

    async def close(self):
        self.closed = True


def make_plugin(**conf):
    os.environ["TYGER_TEST_KEY"] = "test-not-a-key"
    p = OpenAICompatLLM({"model": "stub", "base_url": "http://127.0.0.1:9", "api_key_env": "TYGER_TEST_KEY",
                         **conf}, PluginContext(EngineLoop(), {}, "stub"))
    return p


def stub_create(p, make_stream, record=None):
    async def create(**kw):
        if record is not None:
            record.append(kw)
        return make_stream()
    p.client.chat.completions.create = create


def test_plugin_streams_and_sends_options():
    p = make_plugin(temperature=0.3, max_tokens=256, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    sent = []
    stub_create(p, lambda: StubStream([chunk("Hi "), chunk("there."), chunk(finish="stop")]), sent)
    evs = list(p.stream([{"role": "user", "content": "hi"}], tools=TOOLS, max_new=12))
    assert evs == [("text", "Hi "), ("text", "there.")]
    kw = sent[0]
    assert kw["stream"] and kw["max_tokens"] == 256 and kw["temperature"] == 0.3 and kw["tools"] == TOOLS
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    p2 = make_plugin(); sent2 = []
    stub_create(p2, lambda: StubStream([chunk("ok")]), sent2)
    list(p2.stream([{"role": "user", "content": "hi"}], max_new=12))
    assert "temperature" not in sent2[0] and sent2[0]["max_tokens"] == 12 and "tools" not in sent2[0]
    p3 = make_plugin(model="gpt-5.4-mini", max_tokens=900, extra_body={"reasoning_effort": "low"})
    sent3 = []
    stub_create(p3, lambda: StubStream([chunk("ok")]), sent3)
    list(p3.stream([{"role": "user", "content": "hi"}], max_new=12))
    assert sent3[0]["max_completion_tokens"] == 900 and "max_tokens" not in sent3[0]
    assert sent3[0]["extra_body"] == {"reasoning_effort": "low"}


def test_plugin_stop_closes_stream():
    p = make_plugin(max_concurrency=1, queue_timeout_s=0.5); streams = []

    def mk():
        streams.append(StubStream([chunk(f"word{i} ") for i in range(200)], delay=0.01)); return streams[-1]
    stub_create(p, mk)
    n = 0
    for _ in p.stream([{"role": "user", "content": "talk"}], stop_flag=lambda: n >= 2):
        n += 1
    time.sleep(0.1)
    assert n <= 3 and streams[0].closed and len(streams[0].chunks) > 150
    got = []
    for ev in p.stream([{"role": "user", "content": "again"}], stop_flag=lambda: bool(got)):
        got.append(ev)
    assert got and len(streams) == 2               # the stopped request gave its one slot back


def test_plugin_errors_become_llmerror():
    import httpx, openai
    req = httpx.Request("POST", "http://127.0.0.1:9/chat/completions")
    for exc, kind in ((openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None),
                       "rate_limit"),
                      (openai.InternalServerError("boom", response=httpx.Response(503, request=req), body=None),
                       "server"),
                      (openai.APIConnectionError(request=req), "connection"),
                      (openai.APITimeoutError(request=req), "timeout")):
        p = make_plugin()

        async def create(exc=exc, **kw):
            raise exc
        p.client.chat.completions.create = create
        try:
            list(p.stream([{"role": "user", "content": "hi"}]))
        except LLMError as e:
            assert e.kind == kind, (e.kind, kind)
        else:
            raise AssertionError("no LLMError")


def test_plugin_concurrency_cap():
    p = make_plugin(max_concurrency=1, queue_timeout_s=0.2)
    stub_create(p, lambda: StubStream([chunk("slow ")] * 50, delay=0.02))
    out = {}

    def first():
        out["first"] = len(list(p.stream([{"role": "user", "content": "a"}])))
    t = threading.Thread(target=first); t.start(); time.sleep(0.1)
    try:
        list(p.stream([{"role": "user", "content": "b"}]))
    except LLMError as e:
        out["second"] = e.kind
    t.join()
    assert out == {"first": 50, "second": "busy"}, out


def test_missing_key_is_a_clear_error():
    os.environ.pop("TYGER_NO_SUCH_KEY", None)
    try:
        OpenAICompatLLM({"model": "m", "api_key_env": "TYGER_NO_SUCH_KEY"}, PluginContext(EngineLoop(), {}, "x"))
    except RuntimeError as e:
        assert "TYGER_NO_SUCH_KEY" in str(e)
    else:
        raise AssertionError("constructed without a key")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
