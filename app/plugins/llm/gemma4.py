"""gemma-4-e4b: google/gemma-4-E4B-it in-process on vLLM, the responder since PLUGINS.md step 11 (FP8 at load).

  * Thinking off (enable_thinking=False, the template's default): replies start with speech.
  * Text only: the checkpoint is multimodal; language_model_only skips the vision and audio towers. The text model
    is still ~15 GB in bf16 (per-layer embeddings), so config/default.yaml quantizes its linears to FP8 at load.
  * Tool calls: <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>, written with special tokens, so the output is
    decoded with skip_special_tokens=False (the call tags would vanish otherwise). It sometimes drops the markers
    and writes a bare `set_reminder{...}`: for the offered tools that counts as a call too (never spoken).
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
import json, re

from app.plugins.llm._toolparse import gemma_normalize, strip_think, GEMMA_THOUGHT


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


class Gemma4E4B_NVFP4(Gemma4E4B):
    """gemma-4-e4b-nvfp4, the responder's backup: unsloth's llm-compressor checkpoint of E4B: FP8 attention (and 11
    layers' MLPs), NVFP4 for the other MLPs, and the per-layer embedding table in int8 (3.6 GB less GPU, same routing
    and latency, NOTES.md). The quantization comes from the checkpoint: leave `quantization` unset."""
    default_model = "unsloth/gemma-4-E4B-it-NVFP4"


class Gemma4_26B_A4B(Gemma4E4B):
    """gemma-4-26b-a4b-nvfp4: NVIDIA's NVFP4 (ModelOpt) checkpoint of the 26B mixture-of-experts Gemma 4 (~4B active),
    a responder trial for a 48 GB card (~17 GiB of weights: it didn't fit a 32 GB one next to VoxCPM2 + EARS). The
    quantization comes from the checkpoint: leave `quantization` unset.

    Despite enable_thinking=False it opens Gemma's thought channel after a tool result, mostly empty
    (`<|channel>thought\n<channel|>`, then the answer): the channel is stripped before anything is spoken or
    parsed. Now and then (~0.5% of answers after a tool, routing eval 2026-10-11) it never closes the channel and
    writes the answer inside it, after its thought as a JSON object or after a stray closing token such as
    `<tool_response|>`: when the whole text ends up held back as thinking, the answer after that object or token is
    kept, at the end of the stream."""
    default_model = "nvidia/Gemma-4-26B-A4B-NVFP4"

    def filter_text(self, raw, final=False):
        out = strip_think(raw, final, GEMMA_THOUGHT)
        if final and not out.strip():
            out = answer_after_open_thought(raw)
        return out


def answer_after_open_thought(raw: str) -> str:
    """The text after an unclosed `<|channel>thought` that opens with one JSON object or a stray closing token
    (`<name|>`), or "" (thinking only)."""
    i = raw.rfind(GEMMA_THOUGHT[0])
    if i < 0 or GEMMA_THOUGHT[1] in raw[i:]:
        return ""
    body = raw[i + len(GEMMA_THOUGHT[0]):]
    body = body[len("thought"):] if body.startswith("thought") else body
    body = body.lstrip()
    stray = re.match(r"<\w+\|>", body)
    if stray:
        return body[stray.end():].lstrip()
    if not body.startswith("{"):
        return ""
    try:
        _, end = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return ""
    return body[end:].lstrip()
