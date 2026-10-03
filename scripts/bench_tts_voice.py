"""I-06 / L-03: what the cloned voice costs VoxCPM2 per request. For each way of passing the reference clip, time
building the prompt and generate() -> first audio output, on short sentences (one TTS chunk each). Needs the GPU
free (stop run.sh) and a clip in voices/. Prints timings only.

  .venv/bin/python scripts/bench_tts_voice.py [--voice en] [--n 12]
"""
import sys, time, uuid, argparse, logging, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.WARNING)
import numpy as np, torch
from app.config import load_config, discover_voices, repo_path
from app.engines import EngineLoop
from app.tts_voxcpm import VoxCPM2TTS

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--voice", default=None); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config(); t = cfg["tts"]
    voices, default = discover_voices(cfg)
    path = voices[a.voice or default]
    tts = VoxCPM2TTS(EngineLoop(), t["model"], deploy_config=repo_path(t["deploy_config"]), out_sr=t.get("out_sr", 24000),
                     ref_max_s=t.get("ref_max_s", 10.0))
    voice = tts.load_voice(path)
    TEXTS = ["Hi there, it's really nice to hear from you today.", "The weather looks lovely this afternoon, honestly.",
             "That sounds like a great plan for the weekend, I think.", "Let me know if there's anything else you need."]


    def first_out(prompt):
        rid = f"tts-{uuid.uuid4().hex[:12]}"

        async def gen():
            async for res in tts.engine.generate(prompt=prompt, request_id=rid, sampling_params_list=tts.params,
                                                 output_modalities=["audio"]):
                mm = getattr(res, "multimodal_output", None) or {}
                if "audio" in mm or "model_outputs" in mm:
                    yield 1
        t0 = time.perf_counter()
        for _ in tts.eloop.iterate(gen, abort=lambda: tts.engine.abort(rid)):
            return time.perf_counter() - t0          # leaving the loop aborts the rest of the request
        return float("nan")


    def bench(label, make_prompt):
        for txt in TEXTS[:2]:                        # warm this path
            first_out(make_prompt(txt))
        pt, fo = [], []
        for i in range(a.n):
            t0 = time.perf_counter(); p = make_prompt(TEXTS[i % len(TEXTS)]); pt.append(time.perf_counter() - t0)
            fo.append(first_out(p))
        tot = [x + y for x, y in zip(pt, fo)]
        print(f"  {label:<34} prompt p50 {1e3 * statistics.median(pt):6.1f} ms | generate->first audio p50 "
              f"{1e3 * statistics.median(fo):6.1f} ms | total p50 {1e3 * statistics.median(tot):6.1f} ms")


    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt
    import soundfile as sf
    kw = dict(hf_config=tts.hf_config, tokenizer=tts.tok, split_map=tts.split_map)
    raw, raw_sr = sf.read(str(path), dtype="float32", always_2d=True)
    raw = raw.mean(1)[:int(tts.ref_max_s * raw_sr)].tolist()
    print(f"voice clip {len(raw) / raw_sr:.1f} s @ {raw_sr} Hz; VAE encodes at {tts.vae_sr} Hz; "
          f"speaker cache: {'yes' if voice.cache_key else 'no (has a transcript)'}")
    bench(f"clip at {raw_sr} Hz, every request (old)",
          lambda txt: build_voxcpm2_prompt(**kw, text=txt, ref_audio=raw, ref_sr=raw_sr))
    bench(f"clip at {tts.vae_sr} Hz, every request",
          lambda txt: build_voxcpm2_prompt(**kw, text=txt, ref_audio=voice.ref, ref_sr=voice.sr, ref_text=voice.text))
    bench("current (tts._prompt: + speaker cache)", lambda txt: tts._prompt(txt, voice))
    bench("zero-shot (no clip)", lambda txt: tts._prompt(txt, None))
    tts.engine.shutdown()
    import os; os._exit(0)


if __name__ == "__main__":
    main()
