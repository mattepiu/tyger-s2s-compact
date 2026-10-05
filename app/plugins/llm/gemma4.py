"""gemma-4-e4b: google/gemma-4-E4B-it in-process on vLLM (tried as a `reply` model, PLUGINS.md step 11).

  * Thinking off (enable_thinking=False, the template's default): replies start with speech.
  * Text only: the checkpoint is multimodal; language_model_only skips the vision and audio towers. The text model
    is still ~15 GB in bf16 (per-layer embeddings), about twice Qwen3.5-4B's weights.
  * Tool calls: <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>, written with special tokens, so the output is
    decoded with skip_special_tokens=False (the call tags would vanish otherwise). It sometimes drops the markers
    and writes a bare `set_reminder{...}`: for the offered tools that counts as a call too (never spoken).
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import gemma_normalize


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

    def normalize_calls(self, text, names, final=False):
        return gemma_normalize(text, names, final)


class Gemma4E2B(Gemma4E4B):
    """gemma-4-e2b: the smaller sibling (2.3B effective, 5.1B with per-layer embeddings), tried against E4B to save
    GPU memory. Same template, tool format and options; its own instance name for logs and prompt overrides."""
    default_model = "google/gemma-4-E2B-it"


class Gemma4_26B_A4B(Gemma4E4B):
    """gemma-4-26b-a4b-nvfp4: NVIDIA's NVFP4 (ModelOpt) checkpoint of the 26B mixture-of-experts Gemma 4 (~4B active),
    tried against E4B. ~17 GiB of weights: on a 32 GB card it doesn't fit next to VoxCPM2 + EARS. The quantization
    comes from the checkpoint: leave `quantization` unset."""
    default_model = "nvidia/Gemma-4-26B-A4B-NVFP4"


class Gemma4E4B_NVFP4(Gemma4E4B):
    """gemma-4-e4b-nvfp4: unsloth's llm-compressor checkpoint of E4B: FP8 attention (and 11 layers' MLPs), NVFP4 for
    the other MLPs, and the per-layer embedding table in int8. The quantization comes from the checkpoint: leave
    `quantization` unset."""
    default_model = "unsloth/gemma-4-E4B-it-NVFP4"


class Gemma4_12B_NVFP4(Gemma4E4B):
    """gemma-4-12b-nvfp4: unsloth's llm-compressor checkpoint of the dense 12B Gemma 4 (the encoder-free "unified"
    variant: no vision / audio towers): FP8 attention, NVFP4 MLPs. Leave `quantization` unset."""
    default_model = "unsloth/gemma-4-12b-it-NVFP4"
