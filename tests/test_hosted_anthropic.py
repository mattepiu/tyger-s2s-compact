"""The `anthropic` LLM plugin without a network (no GPU, no key, no cost): message / tool conversion, input checks,
the stream parser on synthetic events (tool round, mid-stream fallback, refusal, truncation), the request it would
send, and the plugin on a stub client. Run: .venv/bin/python tests/test_hosted_anthropic.py"""
import asyncio, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.plugins.base import LLMError, PluginContext  # noqa: E402
from app.plugins.llm.anthropic import (AnthropicLLM, ClaudeStreamParser, FALLBACK_BETA,  # noqa: E402
                                       to_claude, to_claude_tools)
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.runtime.event_loop import EngineLoop  # noqa: E402

TOOLS = build_tools(default_prompts())
THINK = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


# ---- conversion -------------------------------------------------------------------------------------------
def history(native=None, open_round=True):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "weather in Oslo and Rome?"},
            {"role": "assistant", "content": "Checking.", "native": native,
             "tool_calls": [{"id": "toolu_1", "name": "get_weather", "arguments": {"location": "Oslo"}},
                            {"id": "toolu_2", "name": "get_weather", "arguments": {"location": "Rome"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": '{"answer": "sunny"}'},
            {"role": "tool", "tool_call_id": "toolu_2", "content": '{"error": "tavily http 500"}'}]
    if not open_round:
        msgs += [{"role": "assistant", "content": "Sunny in Oslo. —"}, {"role": "user", "content": "and Lima?"}]
    return msgs


def test_open_round_sends_native_unchanged():
    native = [THINK, {"type": "text", "text": "Checking."},
              {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "Oslo"}},
              {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"location": "Rome"}}]
    system, msgs = to_claude(history(native))
    assert system == "SYS" and msgs[0] == {"role": "user", "content": "weather in Oslo and Rome?"}
    assert msgs[1] == {"role": "assistant", "content": native}
    assert msgs[2]["role"] == "user" and [b["tool_use_id"] for b in msgs[2]["content"]] == ["toolu_1", "toolu_2"]
    assert "is_error" not in msgs[2]["content"][0] and msgs[2]["content"][1]["is_error"] is True
    assert len(msgs) == 3                                         # both results in ONE user message


def test_completed_round_has_no_thinking():
    system, msgs = to_claude(history([THINK], open_round=False))
    assert msgs[1]["content"][0] == {"type": "text", "text": "Checking."}
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert "sig-abc" not in json.dumps(msgs) and msgs[-1] == {"role": "user", "content": "and Lima?"}


def test_tools_converted_in_order_with_eager_streaming():
    t = to_claude_tools(TOOLS)
    assert [x["name"] for x in t] == [x["function"]["name"] for x in TOOLS]
    assert all(x["eager_input_streaming"] is True and x["input_schema"]["type"] == "object" for x in t)


# ---- the stream parser --------------------------------------------------------------------------------------
def start(block):
    return {"type": "content_block_start", "content_block": block}


def text(t):
    return {"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}}


def run(events, final):
    p = ClaudeStreamParser(TOOLS); out = []
    for e in events:
        out += p.feed(e)
    return out + p.finish(final)


def test_text_only_thinking_never_spoken():
    evs = run([start(THINK), {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
               start({"type": "text", "text": ""}), text("Otters "), text("hold hands.")],
              {"stop_reason": "end_turn", "content": [THINK, {"type": "text", "text": "Otters hold hands."}]})
    assert evs == [("text", "Otters "), ("text", "hold hands.")]


def test_tool_round_native_kept_whole():
    content = [THINK, {"type": "text", "text": "Let me look."},
               {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"location": "Lima"}}]
    evs = run([start(THINK), start({"type": "text"}), text("Let me look."),
               start({"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {}}), text("ignored")],
              {"stop_reason": "tool_use", "content": content})
    assert evs[:2] == [("text", "Let me look."), ("tool_start", {"name": "get_weather"})]
    kind, calls, native = evs[-1]
    assert kind == "tool_calls" and calls == [{"id": "toolu_9", "name": "get_weather", "arguments": {"location": "Lima"}}]
    assert native == content and len(evs) == 3


def test_mid_stream_fallback_only_after_boundary_counts():
    content = [THINK, {"type": "text", "text": "Sure, "},
               {"type": "tool_use", "id": "toolu_a", "name": "get_weather", "input": {"location": "Os"}},
               {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"}},
               {"type": "thinking", "thinking": "", "signature": "sig-fb"},
               {"type": "tool_use", "id": "toolu_b", "name": "get_weather", "input": {"location": "Oslo"}}]
    evs = run([], {"stop_reason": "tool_use", "content": content})
    _, calls, native = evs[-1]
    assert [c["id"] for c in calls] == ["toolu_b"]
    assert [b["type"] for b in native] == ["text", "thinking", "tool_use"] and native[1]["signature"] == "sig-fb"


def test_refusal_and_truncation_and_bad_input():
    assert run([text("I")], {"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []})[-1] \
        == ("refusal", "cyber")
    trunc = {"stop_reason": "max_tokens",
             "content": [{"type": "tool_use", "id": "t", "name": "get_weather", "input": {"location": "Os"}}]}
    assert run([start({"type": "tool_use", "name": "get_weather"})], trunc)[-1] == ("tool_calls", [], None)
    bad = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "t", "name": "get_weather", "input": {}}]}
    c = run([], bad)[-1][1][0]
    assert "missing location" in c["error"]


# ---- the request ----------------------------------------------------------------------------------------------
def make_plugin(**conf):
    os.environ["TYGER_TEST_KEY"] = "test-not-a-key"
    return AnthropicLLM({"model": "claude-opus-5-5", "api_key_env": "TYGER_TEST_KEY", "max_tokens": 1024,
                         **conf}, PluginContext(EngineLoop(), {}, "claude"))


def test_request_shape():
    kw = make_plugin().request(history(), TOOLS)
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 1024 and kw["system"] == "SYS"
    assert kw["output_config"] == {"effort": "low"} and kw["cache_control"] == {"type": "ephemeral"}
    assert kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]
    for k in ("temperature", "top_p", "top_k", "tool_choice", "thinking"):
        assert k not in kw, k
    assert kw["tools"][0]["eager_input_streaming"] is True
    kw2 = make_plugin(fallbacks=None, effort="medium").request([{"role": "user", "content": "hi"}])
    assert "fallbacks" not in kw2 and "betas" not in kw2 and kw2["output_config"]["effort"] == "medium"
    assert "system" not in kw2 and "tools" not in kw2


# ---- the plugin on a stub client ------------------------------------------------------------------------------
class Obj(dict):
    __getattr__ = dict.get


class StubStream:
    def __init__(self, events, final, delay=0.0):
        self.events, self.final, self.delay, self.closed = list(events), final, delay, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        return Obj(self.events.pop(0))

    async def get_final_message(self):
        return self.final


def final(content, stop="end_turn"):
    return Obj(stop_reason=stop, content=content, stop_details=None,
               usage=Obj(input_tokens=1200, output_tokens=20, cache_read_input_tokens=1000,
                         cache_creation_input_tokens=0))


def test_plugin_streams_text():
    p = make_plugin(); sent = []

    def stream(**kw):
        sent.append(kw)
        return StubStream([start({"type": "text"}), text("Hi "), text("there.")],
                          final([{"type": "text", "text": "Hi there."}]))
    p.client.beta.messages.stream = stream
    evs = list(p.stream([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], tools=TOOLS))
    assert evs == [("text", "Hi "), ("text", "there.")] and p.cache_reads == 1 and sent[0]["system"] == "S"


def test_plugin_stop_closes_stream():
    p = make_plugin(); streams = []

    def stream(**kw):
        streams.append(StubStream([text(f"w{i} ") for i in range(200)], final([]), delay=0.01)); return streams[-1]
    p.client.beta.messages.stream = stream
    n = 0
    for _ in p.stream([{"role": "user", "content": "talk"}], stop_flag=lambda: n >= 2):
        n += 1
    time.sleep(0.1)
    assert n <= 3 and streams[0].closed and len(streams[0].events) > 150


def test_plugin_errors_become_llmerror():
    import anthropic, httpx2
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = ((anthropic.RateLimitError("slow", response=httpx2.Response(429, request=req), body=None), "rate_limit"),
             (anthropic.InternalServerError("boom", response=httpx2.Response(500, request=req), body=None), "server"),
             (anthropic.APITimeoutError(request=req), "timeout"),
             (anthropic.APIConnectionError(request=req), "connection"))
    for exc, kind in cases:
        p = make_plugin()

        class Failing:
            async def __aenter__(self, exc=exc):
                raise exc

            async def __aexit__(self, *a):
                pass
        p.client.beta.messages.stream = lambda **kw: Failing()
        try:
            list(p.stream([{"role": "user", "content": "hi"}]))
        except LLMError as e:
            assert e.kind == kind, (e.kind, kind)
        else:
            raise AssertionError("no LLMError")


def test_mouth_tool_turn_through_the_plugin():
    """A whole MOUTH tool turn on the plugin (stub client): the lead-in, the tool, phase B with the round's own
    thinking blocks sent back unchanged, then a next turn whose history carries no thinking at all."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fakes import FakeTTS, PROMPTS, make_session, run_turn, stub_tools
    p = make_plugin(); sent = []
    round_content = [THINK, {"type": "tool_use", "id": "toolu_w", "name": "get_weather", "input": {"location": "Oslo"}}]

    def stream(**kw):
        sent.append(kw); last = kw["messages"][-1]["content"]
        if isinstance(last, list) and last and last[0].get("type") == "tool_result":
            return StubStream([start({"type": "text"}), text("It's sunny in Oslo today, around twenty degrees.")],
                              final([{"type": "text", "text": "It's sunny in Oslo today, around twenty degrees."}]))
        if kw.get("system", "").startswith(PROMPTS.text("tool_leadin", "system")[:40]):
            return StubStream([text("Checking Oslo.")], final([{"type": "text", "text": "Checking Oslo."}]))
        if "and tomorrow" in str(last):
            return StubStream([text("Probably sunny again tomorrow, honestly.")],
                              final([{"type": "text", "text": "Probably sunny again tomorrow, honestly."}]))
        return StubStream([start(THINK), start({"type": "tool_use", "id": "toolu_w", "name": "get_weather"})],
                          final(round_content, stop="tool_use"))
    p.client.beta.messages.stream = stream
    s, m = make_session(p, FakeTTS(n=2))
    with stub_tools(lambda n, a: {"location": a["location"], "answer": "sunny, 20 C"}):
        evs, _ = run_turn(s, "weather in Oslo?")
        run_turn(s, "and tomorrow?")
    phase_b = next(k for k in sent if isinstance(k["messages"][-1]["content"], list))
    assert phase_b["messages"][-2] == {"role": "assistant", "content": round_content}      # native, unchanged
    assert phase_b["messages"][-1]["content"][0]["tool_use_id"] == "toolu_w"
    nxt = sent[-1]
    assert "sig-abc" not in json.dumps(nxt["messages"]) and "toolu_w" in json.dumps(nxt["messages"])
    assert evs[-1][0] == "done" and "sunny" in evs[-1][1]


def test_missing_key_and_debug_log_refused():
    os.environ.pop("TYGER_NO_SUCH_KEY", None)
    try:
        AnthropicLLM({"api_key_env": "TYGER_NO_SUCH_KEY"}, PluginContext(EngineLoop(), {}, "x"))
    except RuntimeError as e:
        assert "TYGER_NO_SUCH_KEY" in str(e)
    else:
        raise AssertionError("constructed without a key")
    os.environ["ANTHROPIC_LOG"] = "debug"
    try:
        make_plugin()
    except RuntimeError as e:
        assert "ANTHROPIC_LOG" in str(e)
    else:
        raise AssertionError("started with ANTHROPIC_LOG=debug")
    finally:
        os.environ.pop("ANTHROPIC_LOG")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
