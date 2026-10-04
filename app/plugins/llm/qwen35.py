"""qwen3.5-4b: Qwen/Qwen3.5-4B in-process on vLLM (the responder since 2026-10-03).

  * Thinking off: its chat template opens <think> by default, far too slow for a voice reply; enable_thinking=False
    renders an empty think block, so replies start with speech. A guard still keeps any <think>...</think> the
    model writes away from the TTS.
  * Text only: the checkpoint is multimodal; language_model_only skips the vision tower (8.0 GiB of weights).
  * Tool calls in XML: <tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>.
  * Hybrid attention (24 linear + 8 full layers): the pinned 3 GiB of KV holds ~69k tokens.
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import strip_think


class Qwen35(VLLMChatPlugin):
    default_model = "Qwen/Qwen3.5-4B"
    tool_format = "qwen_xml"

    def engine_kwargs(self):
        return {"language_model_only": True}

    def template_kwargs(self):
        return {"enable_thinking": False}

    def filter_text(self, raw, final=False):
        return strip_think(raw, final)
