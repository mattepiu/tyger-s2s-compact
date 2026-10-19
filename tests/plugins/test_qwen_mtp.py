"""Tests for Qwen35MTPLocal LLM plugin (app/plugins/llm/local/qwen_mtp.py).
Run: python tests/plugins/test_qwen_mtp.py
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]

from app.plugins.base import LLMError
from app.plugins.llm.local.qwen_mtp import Qwen35MTPLocal, to_chat_messages
MARKUP = ("<tool_call>", "</tool_call>", "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>")


def check_llm_events(evs):
    kinds = [e[0] for e in evs]
    assert set(kinds) <= {"text", "tool_start", "tool_calls", "refusal"}, kinds
    assert kinds.count("tool_start") <= 1 and kinds.count("tool_calls") <= 1 and kinds.count("refusal") <= 1, kinds
    if "tool_start" in kinds:
        assert "text" not in kinds[kinds.index("tool_start"):], "text after tool_start"
        assert isinstance(evs[kinds.index("tool_start")][1], dict) and "name" in evs[kinds.index("tool_start")][1]
    if "tool_calls" in kinds:
        assert kinds[-1] == "tool_calls" and "tool_start" in kinds, kinds
        ev = evs[-1]
        assert len(ev) == 3, "tool_calls carries (calls, native)"
        named = evs[kinds.index("tool_start")][1]["name"]
        assert named is None or (ev[1] and named == ev[1][0]["name"]), (named, [c["name"] for c in ev[1]])
        ids = [c["id"] for c in ev[1]]
        assert len(ids) == len(set(ids)) and all(isinstance(c["arguments"], dict) and c["name"] for c in ev[1])
    if "refusal" in kinds:
        assert kinds[-1] == "refusal"
    text = "".join(e[1] for e in evs if e[0] == "text")
    assert not any(m in text for m in MARKUP), "markup in spoken text"
    return text


class MockChoiceDelta:
    def __init__(self, content=None, tool_calls=None, refusal=None):
        self.content = content
        self.tool_calls = tool_calls
        self.refusal = refusal


class MockChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class MockChunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [MockChoice(delta, finish_reason=finish_reason)]


class AsyncChunkStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        self._iter = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


class TestQwen35MTP(unittest.TestCase):
    def test_to_chat_messages(self):
        neutral = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the time?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [{"id": "c1", "name": "get_time", "arguments": {"tz": "UTC"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "12:00 PM"},
        ]
        chat = to_chat_messages(neutral)
        self.assertEqual(len(chat), 4)
        self.assertEqual(chat[0]["role"], "system")
        self.assertEqual(chat[1]["role"], "user")
        self.assertEqual(chat[2]["role"], "assistant")
        self.assertEqual(chat[2]["tool_calls"][0]["function"]["name"], "get_time")
        self.assertEqual(chat[2]["tool_calls"][0]["function"]["arguments"], '{"tz": "UTC"}')
        self.assertEqual(chat[3]["role"], "tool")
        self.assertEqual(chat[3]["tool_call_id"], "c1")

    def test_plain_streaming(self):
        llm = Qwen35MTPLocal()
        chunks = [
            MockChunk(MockChoiceDelta(content="Hello ")),
            MockChunk(MockChoiceDelta(content="there! ")),
            MockChunk(MockChoiceDelta(content="How are you?")),
        ]
        mock_create = AsyncMock(return_value=AsyncChunkStream(chunks))
        llm.client.chat.completions.create = mock_create

        msgs = [{"role": "user", "content": "Hi"}]
        evs = list(llm.stream(msgs))
        text = check_llm_events(evs)
        self.assertEqual(text, "Hello there! How are you?")

    def test_hermes_tool_calling(self):
        llm = Qwen35MTPLocal()
        hermes_stream = [
            MockChunk(MockChoiceDelta(content="Let me look ")),
            MockChunk(MockChoiceDelta(content="that up for you.\n<tool_call>")),
            MockChunk(MockChoiceDelta(content='\n{"name": "get_current_time", ')),
            MockChunk(MockChoiceDelta(content='"arguments": {"timezone": "Europe/Paris"}}\n</tool_call>')),
        ]
        mock_create = AsyncMock(return_value=AsyncChunkStream(hermes_stream))
        llm.client.chat.completions.create = mock_create

        tools = [{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
            },
        }]
        msgs = [{"role": "user", "content": "Time in Paris?"}]
        evs = list(llm.stream(msgs, tools=tools))
        text = check_llm_events(evs)
        self.assertEqual(text.strip(), "Let me look that up for you.")

        # Verify tool_start and tool_calls (without ask_expert, tool_start is emitted at bare tag with name=None)
        kinds = [e[0] for e in evs]
        self.assertIn("tool_start", kinds)
        self.assertIn("tool_calls", kinds)
        start_payload = evs[kinds.index("tool_start")][1]
        self.assertIsNone(start_payload["name"])
        calls = evs[kinds.index("tool_calls")][1]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_current_time")
        self.assertEqual(calls[0]["arguments"], {"timezone": "Europe/Paris"})

        # With ask_expert offered, tool_start must wait for and carry the tool name
        tools_with_expert = tools + [{"type": "function", "function": {"name": "ask_expert", "parameters": {}}}]
        mock_create = AsyncMock(return_value=AsyncChunkStream(hermes_stream))
        llm.client.chat.completions.create = mock_create
        evs_expert = list(llm.stream(msgs, tools=tools_with_expert))
        check_llm_events(evs_expert)
        kinds_exp = [e[0] for e in evs_expert]
        start_exp = evs_expert[kinds_exp.index("tool_start")][1]
        self.assertEqual(start_exp["name"], "get_current_time")

    def test_stop_flag_early_abort(self):
        llm = Qwen35MTPLocal()
        chunks = [
            MockChunk(MockChoiceDelta(content=f"Token {i} ")) for i in range(10)
        ]
        mock_create = AsyncMock(return_value=AsyncChunkStream(chunks))
        llm.client.chat.completions.create = mock_create

        n = 0
        def stop():
            return n >= 2

        out = []
        for ev in llm.stream([{"role": "user", "content": "Count"}], stop_flag=stop):
            out.append(ev)
            n += 1

        self.assertLessEqual(len(out), 3)

    def test_refusal_event(self):
        llm = Qwen35MTPLocal()
        chunks = [
            MockChunk(MockChoiceDelta(refusal="I cannot answer this"), finish_reason="refusal")
        ]
        mock_create = AsyncMock(return_value=AsyncChunkStream(chunks))
        llm.client.chat.completions.create = mock_create

        evs = list(llm.stream([{"role": "user", "content": "Bad prompt"}]))
        self.assertEqual(evs, [("refusal", "refusal")])

    def test_llm_error_mapping(self):
        llm = Qwen35MTPLocal()
        import openai
        import httpx

        req = httpx.Request("POST", "http://test")
        resp_429 = httpx.Response(429, request=req)
        resp_500 = httpx.Response(500, request=req)

        e_rate = openai.RateLimitError(message="rate limit", response=resp_429, body=None)
        self.assertEqual(llm.llm_error(e_rate).kind, "rate_limit")

        e_server = openai.InternalServerError(message="internal error", response=resp_500, body=None)
        self.assertEqual(llm.llm_error(e_server).kind, "server")

        e_timeout = openai.APITimeoutError(request=req)
        self.assertEqual(llm.llm_error(e_timeout).kind, "timeout")

        e_conn = openai.APIConnectionError(request=req)
        self.assertEqual(llm.llm_error(e_conn).kind, "connection")


if __name__ == "__main__":
    unittest.main()
