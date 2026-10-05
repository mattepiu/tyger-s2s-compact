"""qwen3-8b-nvfp4 / qwen3-14b-nvfp4: NVIDIA's NVFP4 (ModelOpt) checkpoints of Qwen3-8B / Qwen3-14B in-process on vLLM
(tried as `reply` models against Gemma 4 E4B, 2026-10-05).

  * The original Qwen3 hybrid models: the template thinks by default; enable_thinking=False renders an empty think
    block, and the <think> guard keeps any thinking the model writes anyway away from the TTS.
  * Hermes JSON tool calls: <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
  * NVFP4 weights (W4A4, ModelOpt), read from the checkpoint's hf_quant_config.json: leave `quantization` unset.
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import strip_think


class Qwen3_8B(VLLMChatPlugin):
    default_model = "nvidia/Qwen3-8B-NVFP4"
    tool_format = "hermes_json"

    def template_kwargs(self):
        return {"enable_thinking": False}

    def filter_text(self, raw, final=False):
        return strip_think(raw, final)


class Qwen3_14B(Qwen3_8B):
    default_model = "nvidia/Qwen3-14B-NVFP4"
