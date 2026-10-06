"""R-04b check: the 2nd ear's 24 -> 16 kHz step. MOUTH streams the reply at the 24 kHz wire rate in chunks; EARS
brings it to 16 kHz for the turn-taking model's agent channel. Compares the old per-chunk np.interp (kept here as
the reference) with AgentPlayback's stateful polyphase resampler, at several chunk sizes:
  * fidelity: SNR of the 2nd ear's audio against the original 16 kHz clip (the clip goes 16 -> 24 kHz first, so a
    perfect 24 -> 16 step gives it back);
  * the model: EarsEngine fed as the server feeds it (speaking mode, agent channel = the 2nd ear). Agent alone
    (the user silent): BARGE events (each one false) and the mean yield probability. User cuts in 1.5 s into the
    reply: barges and the median time from the user's start to BARGE.
Agent and user clips are public LibriSpeech (held in memory); prints numbers only. Needs the GPU free (stop run.sh).

  .venv/bin/python scripts/validate_second_ear.py [--trials 12] [--chunk-ms 40,80,160]
"""
import sys, argparse, logging
from pathlib import Path

import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset  # noqa: E402
from app import weights  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.playback import AgentPlayback  # noqa: E402
from app.plugins.ears.nemotron_tyger.constants import HOP_N  # noqa: E402
from app.plugins.ears.nemotron_tyger.engine import EarsEngine  # noqa: E402
from app.plugins.ears.nemotron_tyger.session import EarsSession  # noqa: E402
from app.plugins.tts import WIRE_SR  # noqa: E402
from app.resample import StreamingAudioResampler  # noqa: E402

SR = 16000


def old_per_chunk_interp(a24):
    """The 2nd ear's 24 -> 16 kHz step before R-04b, applied to each chunk on its own."""
    n = int(len(a24) * SR / WIRE_SR)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(a24) - 1, n), np.arange(len(a24)), a24).astype(np.float32)


def to_wire(clip16):
    return StreamingAudioResampler(SR, WIRE_SR).process(clip16, final=True)


def chunks(a24, chunk_ms, rng):
    """Cut the wire audio as a streaming TTS does: chunks around chunk_ms long (+-50%)."""
    out, i = [], 0
    while i < len(a24):
        n = max(1, int(WIRE_SR * chunk_ms / 1e3 * rng.uniform(0.5, 1.5)))
        out.append(a24[i:i + n]); i += n
    return out


def ear(method, pieces):
    if method == "old":
        return np.concatenate([old_per_chunk_interp(c) for c in pieces])
    pb = AgentPlayback(SR); pb.start()
    for c in pieces:
        pb.add(c)
    return pb.take_hop(len(pb.ear))


def snr_db(ref, x):
    n = min(len(ref), len(x)); ref, x = ref[:n], x[:n]
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-12))


def run(eng, thr, user, agent_ear, barge_at_s=None):
    """Feed one reply as the server does; returns the BARGE times (s from the start) and the yield probs."""
    s = EarsSession(0, **thr); s.trace = []
    u = user if user is not None else np.zeros(len(agent_ear), np.float32)
    n = min(len(u), len(agent_ear)) // HOP_N
    barges = []
    for h in range(n):
        sl = slice(h * HOP_N, (h + 1) * HOP_N)
        events, _ = eng.process([(s, u[sl], agent_ear[sl], "speaking")])[s]
        if any(ev["kind"] == "BARGE" for ev in events):
            barges.append((h + 1) * HOP_N / SR)
            if barge_at_s is not None:
                break
    eng.head.reset_slot(0)
    return barges, [t[2] for t in s.trace]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--chunk-ms", default="40,80,160")
    a = ap.parse_args()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    clips = [np.asarray(ds[i]["audio"]["array"], np.float32) for i in range(len(ds))]
    ok = [i for i, c in enumerate(clips) if 4.0 <= len(c) / SR <= 12.0]
    cfg = load_config(); e = cfg["ears"]
    thr = dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
               barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]),
               fallback_frames=int(e.get("fallback_sil_frames", 15)))
    eng = EarsEngine(weights.heads_release(e["heads"], e.get("heads_revision")), slots=1)
    print(f"trials {a.trials} per row; agent level 0.6; user cuts in at 1.5 s")
    print(f"{'chunk':>6} {'method':>6} | {'SNR dB':>7} | {'alone: false barges':>19} {'mean yield':>10} | "
          f"{'cut in: barges':>14} {'median latency':>14}")
    for chunk_ms in (int(x) for x in a.chunk_ms.split(",")):
        latency = {}
        for method in ("old", "new"):
            rng = np.random.default_rng(chunk_ms)               # same cuts for both methods
            snrs, false_barges, ys, lat = [], 0, [], []
            for t in range(a.trials):
                ai = ok[t % len(ok)]; ui = ok[(t + 5) % len(ok)]
                agent16 = 0.6 * clips[ai]
                ear16 = ear(method, chunks(to_wire(agent16), chunk_ms, rng))
                snrs.append(snr_db(agent16, ear16))
                b, y = run(eng, thr, None, ear16)
                false_barges += len(b); ys += y
                user = np.concatenate([np.zeros(int(1.5 * SR), np.float32), clips[ui]])
                user = np.pad(user, (0, max(0, len(ear16) - len(user))))
                b, _ = run(eng, thr, user, np.pad(ear16, (0, max(0, len(user) - len(ear16)))), barge_at_s=1.5)
                lat.append(b[0] - 1.5 if b else None)
            latency[method] = lat
            got = [x for x in lat if x is not None]
            print(f"{chunk_ms:>4}ms {method:>6} | {np.median(snrs):7.1f} | {false_barges:>19d} {np.mean(ys):10.3f} | "
                  f"{len(got):>8d}/{a.trials:<5d} {np.median(got) if got else float('nan'):>12.2f} s")
        both = [(o, n) for o, n in zip(latency["old"], latency["new"]) if o is not None and n is not None]
        hop = HOP_N / SR
        print(f"       paired, trials both barged ({len(both)}): new earlier {sum(n < o for o, n in both)}, same "
              f"{sum(n == o for o, n in both)}, later {sum(n > o for o, n in both)}; mean shift "
              f"{np.mean([n - o for o, n in both]) / hop if both else float('nan'):+.2f} hops | only old barged "
              f"{sum(o is not None and n is None for o, n in zip(latency['old'], latency['new']))}, only new "
              f"{sum(o is None and n is not None for o, n in zip(latency['old'], latency['new']))}")

if __name__ == "__main__":
    main()
