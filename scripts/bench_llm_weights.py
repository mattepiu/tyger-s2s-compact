"""GPU memory per module of a loaded LLM (the tyger plugins' engine args: 3 GiB KV, 8192 tokens, 8 seqs), and
single-stream decode tok/s. Shows which towers / tables are on the GPU (e.g. Gemma's per-layer embeddings, NVFP4
packed weights). Run with the stack stopped: .venv/bin/python scripts/bench_llm_weights.py <hf repo>
Google checkpoints get on-the-fly FP8 like config.yaml; pre-quantized ones (NVFP4) keep their own. Prompts are
synthetic; no text is printed."""
import os, sys, time, collections
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def sizes(model):
    import torch
    groups = collections.defaultdict(int)
    seen = set()
    for kind, it in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
        for n, t in it:
            if t.data_ptr() in seen or t.device.type != "cuda":
                continue
            seen.add(t.data_ptr())
            b = t.numel() * t.element_size()
            if "vision" in n:
                g = "vision tower/embedder"
            elif "audio" in n:
                g = "audio tower/embedder"
            elif "embed_tokens_per_layer" in n:
                g = f"PLE table ({t.dtype})"
            elif "embed_tokens" in n:
                g = f"token embeddings ({t.dtype})"
            elif n == "per_layer_embeddings" or n.endswith(".per_layer_embeddings"):
                g = "PLE input buffer (CUDA-graph)"
            elif t.dtype == torch.float8_e4m3fn:
                g = "decoder linears (fp8)"
            elif t.dtype == torch.uint8:
                g = "NVFP4 packed weights (uint8)"
            else:
                g = f"other text ({kind}, {t.dtype})"
            groups[g] += b
    towers = [type(m).__name__ for n, m in model.named_modules() if n in ("vision_tower", "audio_tower")]
    return dict(groups), towers


def main():
    from vllm import LLM, SamplingParams
    model = sys.argv[1]
    kw = {}
    if "google/" in model:
        kw["quantization"] = "fp8"                     # on-the-fly FP8 (the NVFP4 checkpoints carry their own)
    if "gemma" in model.lower():
        kw["language_model_only"] = True
    llm = LLM(model=model, gpu_memory_utilization=0.5, kv_cache_memory_bytes=3 * 2**30, max_model_len=8192,
              max_num_seqs=8, **kw)
    groups, towers = llm.apply_model(sizes)[0]
    GiB = 2**30
    print(f"\n== {model}: modules on the GPU")
    for g, b in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:36s} {b / GiB:6.3f} GiB")
    print(f"  {'total':36s} {sum(groups.values()) / GiB:6.3f} GiB")
    print("  tower modules:", towers)
    tok = llm.get_tokenizer()
    msgs = [{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper and a storm."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    sp = SamplingParams(temperature=0.7, max_tokens=512, ignore_eos=True)
    llm.generate([prompt], sp, use_tqdm=False)                       # warm-up
    rates = []
    for _ in range(5):
        t = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
        rates.append(len(out.token_ids) / (time.perf_counter() - t))
    rates.sort()
    print(f"  single stream: 512 new tokens, {rates[2]:.0f} tok/s median of 5 (min {rates[0]:.0f}, max {rates[-1]:.0f})")


if __name__ == "__main__":
    main()
