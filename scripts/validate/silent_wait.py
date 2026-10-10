"""I-16 check: does a barge fire when the user talks while the agent holds the floor but is silent (waiting on a
tool or the expert after its lead-in)? In speaking mode a turn can only end as a BARGE, so if the yield head stays
quiet while the agent channel is silent, the user's words are dropped.

Drives EarsEngine as the server feeds it: speaking mode from the turn's start, the agent channel the reply audio as
it plays (zeros once nothing plays), the user channel the mic. Per trial, one public LibriSpeech clip as the agent and
another as the user (held in memory; prints numbers only). Cases:
  * talking:       the agent speaks throughout, the user cuts in 1.5 s in (the usual barge-in);
  * wait +X s:     the agent says a short line (its first `--line-s` seconds), then silence; the user starts X s into
                   the silence;
  * wait, no user: the same line and silence, the user silent (any BARGE is false);
  * no agent yet:  the agent hasn't said anything (zeros from the start) and the user talks from 0.3 s, as when an
                   endpoint fires inside the user's utterance and they carry on.
For each: how many trials barged after the user started, false barges before, and the median time to the barge.
Needs the GPU free (stop run.sh).

  .venv/bin/python scripts/validate/silent_wait.py [--trials 24] [--line-s 2.0] [--wait-starts 0.5,1.5]
"""
import sys, argparse, logging
from pathlib import Path

import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset  # noqa: E402
from app.config import load_config  # noqa: E402
from app.plugins.ears.nemotron_tyger.constants import HOP_N  # noqa: E402
from app.plugins.ears.nemotron_tyger.engine import EarsEngine  # noqa: E402
from app.plugins.ears.nemotron_tyger.release import heads_release  # noqa: E402
from app.plugins.ears.nemotron_tyger.session import EarsSession  # noqa: E402

SR = 16000


def run(eng, thr, user, agent, user_from_s):
    """Feed one turn in speaking mode until its first BARGE. Returns (barge time from the user's start, or None;
    False barges before the user started)."""
    s = EarsSession(0, **thr)
    n = min(len(user), len(agent)) // HOP_N
    hit, false = None, 0
    for h in range(n):
        sl = slice(h * HOP_N, (h + 1) * HOP_N)
        events, _ = eng.process([(s, user[sl], agent[sl], "speaking")])[s]
        if any(ev["kind"] == "BARGE" for ev in events):
            t = (h + 1) * HOP_N / SR
            if user_from_s is None or t <= user_from_s:
                false += 1
            else:
                hit = t - user_from_s
                break
    eng.head.reset_slot(0)
    return hit, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--line-s", type=float, default=2.0, help="the agent's line before the silent wait")
    ap.add_argument("--wait-starts", default="0.5,1.5", help="when the user starts, s into the silence")
    a = ap.parse_args()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    clips = [np.asarray(ds[i]["audio"]["array"], np.float32) for i in range(len(ds))]
    ok = [i for i, c in enumerate(clips) if 4.0 <= len(c) / SR <= 12.0]
    cfg = load_config(); e = cfg["ears"]
    thr = dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
               barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]),
               fallback_frames=int(e.get("fallback_sil_frames", 15)))
    eng = EarsEngine(heads_release(e["heads"], e.get("heads_revision")), slots=1)
    sil = lambda s: np.zeros(int(SR * s), np.float32)   # noqa: E731
    starts = [float(x) for x in a.wait_starts.split(",")]
    cases = ["talking"] + [f"wait +{x:g} s" for x in starts] + ["wait, no user", "no agent yet"]
    print(f"trials {a.trials} per case; agent level 0.6; the agent's line {a.line_s:g} s; barge_thr "
          f"{thr['barge_thr']}, barge_deb {thr['barge_deb']}")
    print(f"{'case':>14} | {'barged':>9} | {'false barges':>12} | {'median to barge':>15} | {'p90':>6} | missed trials")
    for case in cases:
        lat, false, n, missed = [], 0, 0, []
        for t in range(a.trials):
            agent = 0.6 * clips[ok[t % len(ok)]]; voice = clips[ok[(t + 5) % len(ok)]]
            line = agent[:int(a.line_s * SR)]
            if case == "talking":
                user_from = 1.5
                user = np.concatenate([sil(user_from), voice])
                ag = np.pad(agent, (0, max(0, len(user) - len(agent))))
            elif case.startswith("wait +"):
                user_from = a.line_s + float(case.split("+")[1].split()[0])
                user = np.concatenate([sil(user_from), voice])
                ag = np.pad(line, (0, len(user) - len(line)))
            elif case == "wait, no user":
                user_from = None
                ag = np.pad(line, (0, int((a.line_s + 6.0) * SR) - len(line)))
                user = np.zeros_like(ag)
            else:                                              # no agent yet
                user_from = 0.3
                user = np.concatenate([sil(user_from), voice])
                ag = np.zeros_like(user)
            hit, f = run(eng, thr, user, ag[:len(user)], user_from)
            false += f
            if user_from is not None:
                n += 1
                if hit is not None:
                    lat.append(hit)
                else:
                    missed.append(t)
        barged = f"{len(lat)}/{n}" if n else "-"
        med = f"{np.median(lat):.2f} s" if lat else "-"
        p90 = f"{np.percentile(lat, 90):.2f}" if lat else "-"
        print(f"{case:>14} | {barged:>9} | {false:>12d} | {med:>15} | {p90:>6} | {missed}")


if __name__ == "__main__":
    main()
