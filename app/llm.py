"""In-process responder: Qwen3-4B-Instruct-2507 on a vLLM AsyncLLM engine (no OpenAI server, no HTTP).

stream(messages, tools=...) yields ("text", delta) for speakable reply text as it decodes, ("tool_start", None) the
moment a <tool_call> tag appears (the call itself is still decoding), then at most one
("tool_calls", [{"name", "arguments"}, ...]) at the end. Tool calls use the Hermes format the Qwen chat
template emits: <tool_call>{"name": ..., "arguments": {...}}</tool_call>. Text from the first <tool_call> on
is never yielded as speech (a trailing partial "<tool_c" is held back until it's resolved).
"""
from __future__ import annotations
import asyncio, threading, uuid, logging

from app.tools import parse_calls

log = logging.getLogger("llm")
TAG = "<tool_call>"


def split_speakable(raw: str):
    """(speakable_prefix, saw_tool_call) for the raw model text so far."""
    i = raw.find(TAG)
    if i >= 0:
        return raw[:i], True
    for k in range(min(len(TAG) - 1, len(raw)), 0, -1):    # hold back a partial "<tool_c..." suffix
        if TAG.startswith(raw[-k:]):
            return raw[:-k], False
    return raw, False


class LocalLLM:
    def __init__(self, eloop, model, gpu_memory_utilization=0.45, kv_cache_gib=2.0, max_model_len=8192,
                 max_num_seqs=4, temperature=0.7):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        self.eloop = eloop
        self.temperature = float(temperature)
        args = AsyncEngineArgs(model=model, gpu_memory_utilization=float(gpu_memory_utilization),
                               kv_cache_memory_bytes=int(float(kv_cache_gib) * 2**30),
                               max_model_len=int(max_model_len), max_num_seqs=int(max_num_seqs))

        async def make():                                     # construct on the loop that will drive it
            eng = AsyncLLM.from_engine_args(args)
            tok = eng.get_tokenizer()
            return eng, (await tok if asyncio.iscoroutine(tok) else tok)
        self.engine, self.tok = eloop.run(make())
        self._tok_lock = threading.Lock()

    def stream(self, messages, tools=None, max_new=192, temperature=None, stop_flag=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        with self._tok_lock:      # sessions call this from several threads; HF fast tokenizers aren't re-entrant
            prompt = self.tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=True)
        sp = SamplingParams(max_tokens=int(max_new), output_kind=RequestOutputKind.DELTA,
                            temperature=self.temperature if temperature is None else float(temperature))
        rid = f"llm-{uuid.uuid4().hex[:12]}"

        async def deltas():
            async for out in self.engine.generate(prompt, sp, request_id=rid):
                if out.outputs and out.outputs[0].text:
                    yield out.outputs[0].text

        raw, said, saw_tool = "", 0, False
        for d in self.eloop.iterate(deltas, abort=lambda: self.engine.abort(rid), stop_flag=stop_flag):
            raw += d
            if saw_tool:
                continue
            speak, saw_tool = split_speakable(raw)
            if len(speak) > said:
                yield ("text", speak[said:]); said = len(speak)
            if saw_tool:                  # the call's arguments are still decoding; the caller can start a filler
                yield ("tool_start", None)
        if stop_flag and stop_flag():
            return
        if not saw_tool and len(raw) > said:                  # flush a held-back "<..." that wasn't a tag
            yield ("text", raw[said:])
        if saw_tool:
            calls = [c for c in parse_calls(raw) if isinstance(c, dict) and c.get("name")]
            yield ("tool_calls", [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in calls])
