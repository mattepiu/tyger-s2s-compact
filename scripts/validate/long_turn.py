"""I-05 check: a user turn longer than 20 s keeps all of its words, and what the endpoint's ASR costs then.

Runs EarsService (the real GPU + ASR threads) on one session fed as fast as it keeps up: public LibriSpeech clips
joined with short gaps into one ~N s turn, then 5 s of silence. The endpoint head is off while the speech plays (so
it doesn't end the turn at an inner pause) and at the config's threshold from the speech's end, so the head ends the
turn; there is no silence rule. Reports word recall against the clips' reference text, how long after the speech the
head ended the turn, and the endpoint hop's latency (the final ASR) next to an ordinary hop's. Numbers only.
Needs the GPU free (stop run.sh).

  .venv/bin/python scripts/validate/long_turn.py [--seconds 30 45 12]
"""
import sys, re, time, asyncio, argparse, numpy as np, torch, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset
from app.config import load_config
from app.plugins.ears.nemotron_tyger.release import heads_release
from app.plugins.ears.nemotron_tyger.constants import HOP_N
from app.plugins.ears.nemotron_tyger.engine import EarsEngine
from app.plugins.ears.nemotron_tyger.service import EarsService

ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, nargs="+", default=[12.0, 30.0, 45.0])
a = ap.parse_args()
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
words = lambda t: re.findall(r"[a-z']+", t.lower())
cfg = load_config(); e = cfg["ears"]
ckpt = heads_release(cfg["ears"]["heads"], cfg["ears"].get("heads_revision"))
svc = EarsService(EarsEngine(ckpt, slots=4), gather_s=0.0, expect_s=0.0, stats_every_s=1e9)


def turn(seconds, start):
    """Clips from `start` on, 0.25 s apart, until `seconds` of audio; -> (wav, reference words)."""
    parts, ref, i = [np.zeros(8000, np.float32)], [], start
    while sum(len(p) for p in parts) < seconds * 16000:
        parts += [np.asarray(ds[i % len(ds)]["audio"]["array"], np.float32), np.zeros(4000, np.float32)]
        ref += words(ds[i % len(ds)]["text"]); i += 1
    speech = np.concatenate(parts)
    return np.concatenate([speech, np.zeros(16000 * 5, np.float32)]), len(speech), ref


async def run(seconds, start):
    wav, n_speech, ref = turn(seconds, start)
    s = svc.open(ep_thr=2.0, ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]), barge_deb=int(e["barge_deb"]),
                 bc_thr=2.0, bc_deb=int(e["bc_deb"]))   # the head off: the clips have inner pauses
    zeros = np.zeros(HOP_N, np.float32); hop_ms = []; text = None; ep_ms = None
    for h in range(len(wav) // HOP_N):
        if h * HOP_N >= n_speech:
            s.ep_thr = float(e["ep_thr"])      # the speech is over: the head may end the turn
        t0 = time.perf_counter()
        evs = await svc.feed(s, wav[h * HOP_N:(h + 1) * HOP_N], zeros, "listen")
        dt = 1e3 * (time.perf_counter() - t0)
        ep = [x for x in evs if x["kind"] == "ENDPOINT"]
        if ep:
            text, ep_ms, segs, turn_s = ep[0]["transcript"], dt, ep[0].get("segments", 0), ep[0]["turn_s"]
            wait_s = ((h + 1) * HOP_N - n_speech) / 16000
            break
        hop_ms.append(dt)
    svc.close(s); await asyncio.sleep(0.1)
    if text is None:
        print(f"  {seconds:.0f} s turn: the head didn't end it in {(len(wav) - n_speech) / 16000:.1f} s of silence")
        return
    got = set(words(text))
    rec = sum(w in got for w in ref) / len(ref)
    first = sum(w in got for w in ref[:10]) / min(10, len(ref))
    print(f"  {turn_s:5.1f} s turn: {segs} segment(s) | word recall {rec:.2f} (first 10 words {first:.2f}) | "
          f"ended by the head {wait_s:.2f} s after the speech | "
          f"endpoint hop {ep_ms:.0f} ms vs ordinary hop p50 {np.median(hop_ms):.0f} ms")


async def main():
    for k, sec in enumerate(a.seconds):
        await run(sec, 5 * k)
asyncio.run(main())
