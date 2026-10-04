"""gemma-4-e4b: google/gemma-4-E4B-it in-process on vLLM (tried as a `reply` model, PLUGINS.md step 11).

  * Thinking off (enable_thinking=False, the template's default): replies start with speech.
  * Text only: the checkpoint is multimodal; language_model_only skips the vision and audio towers. The text model
    is still ~15 GB in bf16 (per-layer embeddings), about twice Qwen3.5-4B's weights.
  * Tool calls: <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>, written with special tokens, so the output is
    decoded with skip_special_tokens=False (the call tags would vanish otherwise).
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin


class Gemma4E4B(VLLMChatPlugin):
    default_model = "google/gemma-4-E4B-it"
    tool_format = "gemma4"

    def engine_kwargs(self):
        return {"language_model_only": True}

    def template_kwargs(self):
        return {"enable_thinking": False}

    def sampling(self, max_new):
        sp = super().sampling(max_new)
        sp.skip_special_tokens = False
        return sp
