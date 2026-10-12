"""In-process chat LLM on a vLLM AsyncLLM engine (no server, no HTTP), on MOUTH's EngineLoop. The base of the local
plugins (gemma-4-e4b, local/gemma4/), which override only the hooks: default_model, tool_format, engine_kwargs,
template_kwargs, sampling, normalize_calls.

stream(messages, tools=...) yields ("text", delta) for speakable text as it decodes; ("tool_start", {"name": ...})
once the format's call tag (TAGS) has appeared: with ask_expert offered, only after the tool's name has decoded (a
few tokens; the arguments are still decoding) so MOUTH can pick a lead-in that fits the tool, otherwise at the bare
tag ({"name": None}); then at most one ("tool_calls", [{"id", "name", "arguments"}], None) at the end. Text from the
first call tag on is never yielded as speech (a trailing partial tag is held back until it resolves). The KV budget
is pinned (kv_cache_gib), so GPU memory doesn't depend on what else is loaded.
"""
from __future__ import annotations
import asyncio, threading, uuid, logging
from dataclasses import dataclass

from app.plugins.base import LLMPlugin
from app.plugins.options import PluginOptions
from app.plugins.llm.calls import split_speakable, parse_calls, new_call_id, tool_start, TAGS, names_tool

log = logging.getLogger("llm")


def to_template_messages(messages):
    """Neutral history -> what the HF chat templates read: an assistant tool call in the OpenAI nested shape
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


class VLLMChatPlugin(LLMPlugin):
    default_model: str = ""
    tool_format: str = "hermes_json"            # calls.TAGS key: the call format the model writes

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        model: str | None = None               # None: the plugin's default_model
        quantization: str | None = None        # e.g. fp8 (at load); unset for a pre-quantized checkpoint
        gpu_memory_utilization: float = 0.45   # vLLM's start-up free-memory check only
        kv_cache_gib: float = 3.0              # the pinned KV budget
        max_model_len: int = 8192
        max_num_seqs: int = 8
        temperature: float = 0.7
        max_tokens: int | None = None          # if set, every request gets it instead of MOUTH's max_new
        # (backchannel / filler limits included): meant for a block that serves only `smart`, like openai-responses'

    def __init__(self, conf: dict, ctx):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        super().__init__(conf, ctx)
        o = self.opts
        self.model = o.model or self.default_model
        self.temperature = o.temperature
        self.max_tokens = o.max_tokens
        args = AsyncEngineArgs(model=self.model, gpu_memory_utilization=o.gpu_memory_utilization,
                               kv_cache_memory_bytes=int(o.kv_cache_gib * 2**30),
                               max_model_len=o.max_model_len, max_num_seqs=o.max_num_seqs,
                               **({"quantization": o.quantization} if o.quantization else {}),
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
        """Text with the model's malformed calls rewritten into its tagged format (`names`: the offered tools).
        Mid-stream it may hold back a trailing word that could still start a call; final=True (the stream ended)
        holds nothing back."""
        return text

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
        visible_now = lambda final=False: self.normalize_calls(raw, names, final)  # noqa: E731
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
        if not saw_tool:              # flush a held-back "<..." that wasn't a tag; a call that shows up only in
            speak, saw_tool = split_speakable(visible, TAGS[self.tool_format])   # this final pass is still a call
            if len(speak) > said:
                yield ("text", speak[said:])
        if saw_tool:
            calls = parse_calls(visible, self.tool_format)
            if not started:               # ended before the name decoded (a cut-off or garbled call)
                yield ("tool_start", {"name": calls[0]["name"] if calls else None})
            yield ("tool_calls", [{"id": new_call_id(), **c} for c in calls], None)

    # warmup: none. MOUTH's own start-up request (config/prompts.yaml `warmup`) pays the first prefill

    def shutdown(self) -> None:
        self.engine.shutdown()
