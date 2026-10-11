"""The Qwen3.5 / 3.6 35B-A3B mixtures of experts in 4 bits, in-process on vLLM: responder trials for a 48 GB card
(HANDOFF 2026-10-11, item 2). 35B parameters, ~3B active per token; 40 layers, 30 Gated DeltaNet and 10 gated
attention with 2 KV heads, so the KV cache is small.

  * Thinking off: the template thinks by default; enable_thinking=False renders an empty think block, so replies
    start with speech. strip_think still keeps any <think>...</think> the model writes away from the TTS.
  * Text only: the checkpoints carry a vision encoder; language_model_only skips it.
  * Tool calls in Qwen XML (vLLM's qwen3_coder): <tool_call><function=NAME><parameter=P>value</parameter>...
  * The quantization comes from the checkpoint: leave `quantization` unset.
  * Sampling as Qwen's card recommends for non-thinking mode (top_p 0.8, top_k 20, presence_penalty 1.5; the
    temperature stays the block's 0.7, which the card also gives). With only the temperature, Qwen3.6 switched
    language mid-word ("залей водой вtwice the volume") and answered Spanish in Russian (routing eval, 2026-10-11).
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import strip_think


class Qwen36MoE_NVFP4(VLLMChatPlugin):
    """qwen3.6-35b-a3b-nvfp4: NVIDIA's ModelOpt checkpoint, mixed precision (its hf_quant_config.json): the routed
    and shared experts and lm_head in NVFP4 weights with 16-bit activations (W4A16), every attention and DeltaNet
    projection in FP8, an FP8 KV cache; the MTP head unquantized."""
    default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    tool_format = "qwen_xml"

    def engine_kwargs(self):
        return {"language_model_only": True}

    def template_kwargs(self):
        return {"enable_thinking": False}

    def sampling(self, max_new):
        sp = super().sampling(max_new)
        sp.top_p = 0.8; sp.top_k = 20; sp.presence_penalty = 1.5
        return sp

    def filter_text(self, raw, final=False):
        return strip_think(raw, final)


class Qwen35MoE_GPTQ(Qwen36MoE_NVFP4):
    """qwen3.5-35b-a3b-gptq: Qwen's own GPTQ int4 checkpoint."""
    default_model = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
