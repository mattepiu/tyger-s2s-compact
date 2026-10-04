"""qwen3-4b-2507: Qwen/Qwen3-4B-Instruct-2507 in-process on vLLM (the responder until 2026-10-03).

An instruct model with no thinking mode, so no template switches; Hermes JSON tool calls:
<tool_call>{"name": ..., "arguments": {...}}</tool_call>.
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin


class Qwen3Instruct2507(VLLMChatPlugin):
    default_model = "Qwen/Qwen3-4B-Instruct-2507"
    tool_format = "hermes_json"
