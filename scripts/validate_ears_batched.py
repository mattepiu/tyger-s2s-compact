"""Validate the batched multi-session EARS (app/plugins/ears/nemotron_tyger) against the single-session research
reference (scripts/reference/ears_ref EndpointEars): per-frame head outputs and events, alone and 4 sessions with staggered starts, plus tick timing.
Mic input = public LibriSpeech test clips (hf-internal-testing/librispeech_asr_dummy), in memory; prints numbers
and event kinds only. Needs the GPU free (stop run.sh) and `datasets` (pip install datasets).

  .venv/bin/python scripts/validate_ears_batched.py      # expect "events match: True" for every session
"""
import sys, time, argparse, numpy as np, torch, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset
from app.config import load_config
from app import weights
from scripts.reference.ears_ref.endpoint_ears import EndpointEars
from app.plugins.ears.nemotron_tyger.constants import HOP_N, SEG_S
from app.plugins.ears.nemotron_tyger.engine import EarsEngine
from app.plugins.ears.nemotron_tyger.session import EarsSession

ap = argparse.ArgumentParser(); ap.add_argument("--no-merge", action="store_true"); ap.add_argument("--no-graph", action="store_true")
a = ap.parse_args()
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
clip = lambda i: np.asarray(ds[i]["audio"]["array"], np.float32)
sil = lambda s: np.zeros(int(16000 * s), np.float32)

def scenario(k):
    """listen: speech, pause (endpoint), speech, pause; then speaking: agent talks, user barges in."""
    u_listen = np.concatenate([sil(0.6), clip(k), sil(2.0), clip(k + 1), sil(2.0)])
    agent = clip(k + 2); user_spk = np.concatenate([sil(2.0), clip(k + 3)])
    m = max(len(agent), len(user_spk))
    agent = np.pad(agent, (0, m - len(agent))); user_spk = np.pad(user_spk, (0, m - len(user_spk)))
    hops = []
    for i in range(0, len(u_listen) - HOP_N + 1, HOP_N):
        hops.append((u_listen[i:i + HOP_N], np.zeros(HOP_N, np.float32), "listen"))
    for i in range(0, m - HOP_N + 1, HOP_N):
        hops.append((user_spk[i:i + HOP_N], agent[i:i + HOP_N] * 0.6, "speaking"))
    return hops

cfg = load_config(); e = cfg["ears"]
ckpt = weights.ensure_heads(cfg["ears"]["heads_repo"], cfg["ears"]["heads_file"])
thr = dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
           barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]))

# ---- reference (single-session EndpointEars), one scenario ----
ref = EndpointEars(ckpt, echo=True, theta=0.2, **thr); ref.FALLBACK_SIL_FRAMES = int(e.get("fallback_sil_frames", 15))
ref_trace = []; orig = ref._head_step
def rec(*x):
    r = orig(*x); ref_trace.append(r[1:]); return r      # (haz, yp, bcp, pep)
ref._head_step = rec
def run_ref(hops):
    ref.reset(); ref_trace.clear(); evs = []
    for uh, ah, mode in hops:
        for ev in ref.feed(uh, ah, mode):
            if ev["kind"] != "FRAME":
                evs.append((ev["kind"], len(ref_trace), ev.get("transcript", "")[:40] if ev["kind"] == "ENDPOINT" else "",
                            ev.get("turn_s", 0.0)))
    return list(ref_trace), evs
S = 4
scen = [scenario(4 * k) for k in range(S)]
refs = [run_ref(h) for h in scen]
del ref; torch.cuda.empty_cache()

# ---- batched engine ----
eng = EarsEngine(ckpt, slots=S, merge_lora=not a.no_merge, cuda_graph=not a.no_graph)
def run_eng(sessions, hop_lists, offsets):
    """Run sessions together; session k starts at tick offsets[k]. Returns per-session (trace, events), tick ms."""
    for s in sessions:
        s.trace = []
    out = [[] for _ in sessions]; ticks = []
    T = max(o + len(h) for o, h in zip(offsets, hop_lists))
    for t in range(T):
        items = [(s, *h[t - o]) for s, h, o in zip(sessions, hop_lists, offsets) if 0 <= t - o < len(h)]
        if not items:
            continue
        torch.cuda.synchronize(); t0 = time.perf_counter()
        res = eng.process(items)
        torch.cuda.synchronize(); ticks.append(((time.perf_counter() - t0) * 1e3, len(items)))
        for s, *_ in items:
            events, finals = res[s]
            if finals:                                 # a long turn's earlier segments come from the ASR thread
                for (ev, wav), txt in zip(finals, eng.asr.transcribe([w for _, w in finals])):
                    deadline = time.time() + 10
                    while any(t is None for t in ev.get("seg_texts", [])) and time.time() < deadline:
                        time.sleep(0.01)
                    ev["transcript"] = " ".join(t for t in [*ev.pop("seg_texts", []), txt] if t)
            k = sessions.index(s)
            for ev in events:
                if ev["kind"] != "FRAME":
                    out[k].append((ev["kind"], len(s.trace), (ev.get("transcript") or "")[:40] if ev["kind"] == "ENDPOINT" else "",
                                   ev.get("turn_s", 0.0)))
    return [(list(s.trace), o) for s, o in zip(sessions, out)], ticks

def compare(name, ref_tr, got_tr, ref_ev, got_ev):
    n = min(len(ref_tr), len(got_tr))
    r = np.array(ref_tr[:n]); g = np.array([x[1:] for x in got_tr[:n]])
    d = np.abs(r - g).max(0)
    same = [x[:2] for x in ref_ev] == [x[:2] for x in got_ev]
    # endpoint transcripts: the reference keeps only a turn's last 20 s, the engine transcribes a turn longer
    # than SEG_S in segments (I-05), so only turns up to SEG_S must read the same
    short = [(r[2], g[2]) for r, g in zip(ref_ev, got_ev) if r[0] == "ENDPOINT" and g[3] <= SEG_S]
    n_long = sum(1 for g in got_ev if g[0] == "ENDPOINT" and g[3] > SEG_S)
    print(f"  {name}: frames {len(ref_tr)}/{len(got_tr)} | max|diff| haz {d[0]:.4f} yield {d[1]:.4f} bc {d[2]:.4f} p_end {d[3]:.4f}"
          f" | events match: {same} | transcripts match: {sum(a == b for a, b in short)}/{len(short)}"
          + (f" (+{n_long} turn(s) > {SEG_S:.0f} s, segmented)" if n_long else ""))
    if not same:
        for kind, fi in sorted(set(x[:2] for x in ref_ev) ^ set(x[:2] for x in got_ev), key=lambda x: x[1]):
            src = "ref only" if (kind, fi) in [x[:2] for x in ref_ev] else "got only"
            lo = max(0, fi - 3)
            print(f"    {src}: {kind} at frame {fi} | yield ref {[round(x[1],3) for x in ref_tr[lo:fi]]} got {[round(x[2],3) for x in got_tr[lo:fi]]}")

print("== single session vs reference ==")
s0 = EarsSession(0, **thr); s0.fallback_frames = int(e.get("fallback_sil_frames", 15))
(got,), ticks1 = run_eng([s0], [scen[0]], [0])
compare("session 0", refs[0][0], got[0], refs[0][1], got[1])
print(f"  tick ms p50 {np.median([t for t,_ in ticks1[3:]]):.0f}  p90 {np.percentile([t for t,_ in ticks1[3:]],90):.0f}")

print(f"== {S} sessions together (staggered starts) vs reference ==")
sess = [EarsSession(k, **thr) for k in range(S)]
for s in sess:
    s.fallback_frames = int(e.get("fallback_sil_frames", 15))
gots, ticksN = run_eng(sess, scen, [0, 3, 7, 11])
for k in range(S):
    compare(f"session {k}", refs[k][0], gots[k][0], refs[k][1], gots[k][1])
full = [t for t, b in ticksN if b == S]
print(f"  tick ms with {S} sessions: p50 {np.median(full):.0f}  p90 {np.percentile(full,90):.0f}  max {max(full):.0f}  (n={len(full)})")
print(f"  engine busy over the 4-session run: encoder {eng.t_enc:.2f} s, head {eng.t_head:.2f} s")
print(f"  GPU memory allocated {torch.cuda.memory_allocated()//2**20} MiB, reserved {torch.cuda.memory_reserved()//2**20} MiB")
