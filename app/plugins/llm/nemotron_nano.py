"""nemotron-nano-9b-v2-nvfp4: nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 in-process on vLLM (tried as a `reply` model
against Gemma 4 E4B, 2026-10-05).

  * Hybrid Mamba-2 / attention (NemotronH). NVIDIA asks for an fp32 SSM cache for accuracy (mamba_ssm_cache_dtype).
  * Thinking is switched by "/no_think" in the system prompt (the template strips it and opens an empty think
    block); the <think> guard keeps any thinking away from the TTS.
  * Tool calls: <TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>.
  * The model card lists English, German, Spanish, French, Italian and Japanese: not Russian.
  * NVFP4 weights (ModelOpt; first and last layers in bf16), read from the checkpoint: leave `quantization` unset.
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import strip_think

NO_THINK = "/no_think"


class NemotronNano9B(VLLMChatPlugin):
    default_model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    tool_format = "nemotron_json"

    def engine_kwargs(self):
        return {"mamba_ssm_cache_dtype": "float32"}

    def prepare_messages(self, messages):
        # the template runs `'</think>' in content` on every message: a tool-call turn's None content breaks it
        messages = [{**m, "content": m.get("content") or ""} for m in messages]
        if messages and messages[0].get("role") == "system":
            return [{**messages[0], "content": f"{messages[0].get('content') or ''}\n{NO_THINK}"}] + messages[1:]
        return [{"role": "system", "content": NO_THINK}] + messages

    def filter_text(self, raw, final=False):
        return strip_think(raw, final)
