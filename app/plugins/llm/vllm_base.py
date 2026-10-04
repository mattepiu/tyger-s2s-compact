"""In-process chat LLM on a vLLM AsyncLLM engine (no server, no HTTP), on MOUTH's EngineLoop. Shared by the local
Qwen plugins, which override only the hooks: default_model, engine_kwargs, template_kwargs, tool_format,
sampling, filter_text.

stream(messages, tools=...) yields ("text", delta) for speakable text as it decodes; ("tool_start", {"name": ...})
once a <tool_call> tag has appeared: with ask_expert offered, only after the tool's name has decoded (a few tokens;
the arguments are still decoding) so MOUTH can pick a lead-in that fits the tool, otherwise at the bare tag
({"name": None}); then at most one ("tool_calls", [{"id", "name",
"arguments"}], None) at the end. Text from the first <tool_call> on is
never yielded as speech (a trailing partial "<tool_c" is held back until it resolves). The KV budget is pinned
(kv_cache_gib), so GPU memory doesn't depend on what else is loaded.
"""
from __future__ import annotations
import asyncio, threading, uuid, logging

from app.plugins.llm._toolparse import split_speakable, parse_calls, new_call_id, tool_start, TAGS, names_tool

log = logging.getLogger("llm")


def to_template_messages(messages):
    """Neutral history -> what the Qwen chat templates read: an assistant tool call in the OpenAI nested shape
    ({"id", "type": "function", "function": {"name", "arguments"}}); `native` dropped (local models have none)."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = {"role": "assistant", "content": m.get("content"),
                 "tool_calls": [{"id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                                for c in m["tool_calls"]]}
        out.append(m)
    return out


class VLLMChatPlugin:
    remote = False
    default_model: str = ""
    tool_format: str = "hermes_json"            # _toolparse.FORMATS key the model writes

    def __init__(self, conf: dict, ctx):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        self.conf = conf; self.name = ctx.name; self.eloop = ctx.eloop
        self.model = conf.get("model") or self.default_model
        self.temperature = float(conf.get("temperature", 0.7))
        self.max_tokens = conf.get("max_tokens")   # if set, every request gets it instead of MOUTH's max_new
        # (backchannel / filler limits included): meant for a block that serves only `smart`, as in openai_compat
        args = AsyncEngineArgs(model=self.model, gpu_memory_utilization=float(conf.get("gpu_memory_utilization", 0.45)),
                               kv_cache_memory_bytes=int(float(conf.get("kv_cache_gib", 3.0)) * 2**30),
                               max_model_len=int(conf.get("max_model_len", 8192)),
                               max_num_seqs=int(conf.get("max_num_seqs", 8)),
                               **({"quantization": conf["quantization"]} if conf.get("quantization") else {}),
                               **self.engine_kwargs())

        async def make():                                     # construct on the loop that will drive it
            eng = AsyncLLM.from_engine_args(args)
            tok = eng.get_tokenizer()
            return eng, (await tok if asyncio.iscoroutine(tok) else tok)
        self.engine, self.tok = self.eloop.run(make())
        self._tok_lock = threading.Lock()

    # ---- hooks ----
    def engine_kwargs(self) -> dict:
        return {}

    def template_kwargs(self) -> dict:
        return {}

    def sampling(self, max_new):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        return SamplingParams(max_tokens=int(max_new), output_kind=RequestOutputKind.DELTA,
                              temperature=self.temperature)

    def normalize_calls(self, text: str, names, final: bool = False) -> str:
        """Text with the model's malformed calls rewritten into its tagged format (`names`: the offered tools)."""
        return text

    def filter_text(self, raw: str, final: bool = False) -> str:
        """The model's raw text -> the part that may be spoken or parsed (e.g. without thinking). Mid-stream it
        may hold back a trailing partial tag; final=True (the stream ended) holds nothing back."""
        return raw

    # ---- LLMPlugin ----
    def stream(self, messages, tools=None, max_new=192, stop_flag=None):
        with self._tok_lock:      # sessions call this from several threads; HF fast tokenizers aren't re-entrant
            prompt = self.tok.apply_chat_template(to_template_messages(messages), tools=tools or None,
                                                  tokenize=False, add_generation_prompt=True,
                                                  **self.template_kwargs())
        sp = self.sampling(self.max_tokens or max_new)
        rid = f"llm-{uuid.uuid4().hex[:12]}"

        async def deltas():
            async for out in self.engine.generate(prompt, sp, request_id=rid):
                if out.outputs and out.outputs[0].text:
                    yield out.outputs[0].text

        raw, said, saw_tool, started = "", 0, False, None
        names = [t["function"]["name"] for t in tools or []]
        wait_name = names_tool(tools)
        visible_now = lambda final=False: self.normalize_calls(self.filter_text(raw, final), names, final)  # noqa: E731
        for d in self.eloop.iterate(deltas, abort=lambda: self.engine.abort(rid), stop_flag=stop_flag):
            raw += d
            if started:
                continue
            if not saw_tool:
                speak, saw_tool = split_speakable(visible_now(), TAGS[self.tool_format])
                if len(speak) > said:
                    yield ("text", speak[said:]); said = len(speak)
            if saw_tool:                  # the call's arguments are still decoding; the caller can start a filler
                started = tool_start(visible_now(), TAGS[self.tool_format]) if wait_name else {"name": None}
                if started:
                    yield ("tool_start", started)
        if stop_flag and stop_flag():
            return
        visible = visible_now(final=True)
        if not saw_tool and len(visible) > said:              # flush a held-back "<..." that wasn't a tag
            yield ("text", visible[said:])
        if saw_tool:
            calls = parse_calls(visible, self.tool_format)
            if not started:               # ended before the name decoded (a cut-off or garbled call)
                yield ("tool_start", {"name": calls[0]["name"] if calls else None})
            yield ("tool_calls", [{"id": new_call_id(), **c} for c in calls], None)

    def warmup(self) -> None:
        pass                      # MOUTH's own start-up request (prompts.yaml `warmup`) pays the first prefill

    def shutdown(self) -> None:
        self.engine.shutdown()
