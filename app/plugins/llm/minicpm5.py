"""minicpm5-2b: openbmb/MiniCPM5-2B in-process on vLLM, a small-responder trial (the user's request, 2026-10-11;
English only: its card lists English and Chinese). A dense 2.5B LlamaForCausalLM (42 layers, 2 KV heads: ~43 KB of
bf16 KV per token), 4.7 GiB in bf16; no custom code.

  * Thinking off: the template thinks unless enable_thinking=False, which renders an empty think block, so replies
    start with speech. strip_think keeps any <think>...</think> the model still writes away from the TTS.
  * Tool calls: <function name="NAME"><param name="P">value</param>...</function> (_toolparse's `minicpm5`).
    <function and <param are special tokens, so the output is decoded with skip_special_tokens=False.
  * Sampling: the block's temperature with the card's top_p 0.95 (the card's temperature 1.0 is for its thinking
    mode; the other responders run at 0.7).
"""
from app.plugins.llm.vllm_base import VLLMChatPlugin
from app.plugins.llm._toolparse import strip_think


class MiniCPM5_2B(VLLMChatPlugin):
    default_model = "openbmb/MiniCPM5-2B"
    tool_format = "minicpm5"

    def template_kwargs(self):
        return {"enable_thinking": False}

    def sampling(self, max_new):
        sp = super().sampling(max_new)
        sp.top_p = 0.95; sp.skip_special_tokens = False
        return sp

    def filter_text(self, raw, final=False):
        return strip_think(raw, final)
