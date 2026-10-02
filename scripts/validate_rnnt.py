"""Check the graphed greedy RNN-T decoder (app/ears_core/rnnt.py) against nemotron's HF generate(): same text
on 40 single windows (1-20 s) and 10 padded mixed-length batches, plus timings. Public LibriSpeech test audio, in
memory. Needs the GPU free (stop run.sh) and `datasets` (pip install datasets).

  .venv/bin/python scripts/validate_rnnt.py              # expect "40/40 identical" twice
"""
import sys, time, numpy as np, torch, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); logging.basicConfig(level=logging.ERROR); torch.set_num_threads(4)
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from app.ears_core.rnnt import greedy_rnnt, GraphedRNNT
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sp = np.concatenate([np.asarray(ds[i]["audio"]["array"], np.float32) for i in range(len(ds))])
mid = "nvidia/nemotron-3.5-asr-streaming-0.6b"
proc = AutoProcessor.from_pretrained(mid); asr = AutoModel.from_pretrained(mid, dtype=torch.float32).cuda().eval()
rng = np.random.default_rng(0)
G = GraphedRNNT(asr); G.capture()
def graphed(wavs):
    inp = proc(wavs, sampling_rate=16000, return_tensors="pt").to("cuda")
    return [proc.decode(s, skip_special_tokens=True).strip() for s in G(inp["input_features"], inp.get("attention_mask"))]
def hf(wavs):
    inp = proc(wavs, sampling_rate=16000, return_tensors="pt").to("cuda")
    with torch.no_grad():
        return [t.strip() for t in proc.batch_decode(asr.generate(**inp).sequences, skip_special_tokens=True)]
def lean(wavs):
    inp = proc(wavs, sampling_rate=16000, return_tensors="pt").to("cuda")
    return [t.strip() for t in proc.batch_decode(greedy_rnnt(asr, inp["input_features"], inp.get("attention_mask")),
                                                  skip_special_tokens=True)]
mism = n = 0
for trial in range(40):
    L = int(rng.choice([1.0, 2.5, 4.0, 8.0, 8.0, 8.0, 14.0, 20.0]) * 16000)
    o = int(rng.integers(0, len(sp) - L)); w = sp[o:o + L]
    a, b = hf([w])[0], graphed([w])[0]; n += 1
    if a != b:
        mism += 1; print(f"single MISMATCH L={L/16000}s\n  hf  : {a}\n  lean: {b}")
print(f"single: {n - mism}/{n} identical")
# padded batches of mixed lengths vs each alone (HF, unbatched)
pm = pn = 0
for trial in range(10):
    ws = [sp[(o := int(rng.integers(0, len(sp) - 9 * 16000))): o + int(rng.uniform(1.0, 8.0) * 16000)] for _ in range(4)]
    alone = [hf([w])[0] for w in ws]; bat = graphed(ws); hb = hf(ws)
    for x, y, z in zip(alone, bat, hb):
        pn += 1
        if x != y:
            pm += 1; print(f"padded MISMATCH (hf padded {'==' if z == y else '!='} lean padded)\n  alone: {x}\n  lean : {y}")
print(f"padded batch vs alone: {pn - pm}/{pn} identical")
def t(fn, n=6):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return 1e3 * np.median(ts)
for L, B in ((8, 1), (8, 4), (20, 1)):
    ws = [sp[k * 50000: k * 50000 + L * 16000] for k in range(B)]
    print(f"{L}s x{B}: hf {t(lambda: hf(ws)):.0f} ms | lean {t(lambda: lean(ws)):.0f} ms | graphed {t(lambda: graphed(ws)):.0f} ms")
inp = proc([sp[:8*16000]]*4, sampling_rate=16000, return_tensors="pt").to("cuda")
with torch.no_grad():
    print(f"encoder only 8s x4: {t(lambda: asr.get_audio_features(input_features=inp['input_features'], attention_mask=inp['attention_mask'])):.0f} ms")
