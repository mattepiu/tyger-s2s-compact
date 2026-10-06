"""I-02 check: does the endpoint transcript of a barge-in keep the start of the user's utterance?

Drives EarsEngine directly, with the server's protocol: the agent speaks (speaking mode, agent channel = a public
LibriSpeech clip), the user starts another clip 1.5 s in; when BARGE fires the session resets and switches to
listen (app/ears/conversation.py), and the turn runs to its ENDPOINT. The endpoint transcript is scored against the user clip's
reference text: word recall overall and of its first 3 words. Prints numbers only, never words.
Needs the GPU free (stop run.sh).

  .venv/bin/python scripts/validate_barge.py                  # current server protocol
  .venv/bin/python scripts/validate_barge.py --double-reset   # the old server: reset again when MOUTH's D arrives
"""
import sys, re, argparse, numpy as np, torch, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset
from app.config import load_config
from app import weights
from app.plugins.ears.nemotron_tyger.constants import HOP_N
from app.plugins.ears.nemotron_tyger.engine import EarsEngine
from app.plugins.ears.nemotron_tyger.session import EarsSession

ap = argparse.ArgumentParser()
ap.add_argument("--double-reset", action="store_true", help="emulate the old finish_turn reset after the barge")
ap.add_argument("--trials", type=int, default=12)
a = ap.parse_args()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
words = lambda t: re.findall(r"[a-z']+", t.lower())
cfg = load_config(); e = cfg["ears"]
ckpt = weights.ensure_heads(cfg["ears"]["heads_repo"], cfg["ears"]["heads_file"])
thr = dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
           barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]),
           fallback_frames=int(e.get("fallback_sil_frames", 15)))
eng = EarsEngine(ckpt, slots=1)
sil = lambda s: np.zeros(int(16000 * s), np.float32)
user_ids = [i for i in range(len(ds)) if 4.0 <= len(ds[i]["audio"]["array"]) / 16000 <= 12.0]

recall, first3, barge_at, n_barge = [], [], [], 0
for t in range(a.trials):
    ui = user_ids[t % len(user_ids)]; ai = (ui + 7) % len(ds)
    user = np.asarray(ds[ui]["audio"]["array"], np.float32); ref = words(ds[ui]["text"])
    agent = np.asarray(ds[ai]["audio"]["array"], np.float32)
    u = np.concatenate([sil(1.5), user, sil(3.0)])
    ag = np.pad(agent * 0.6, (0, max(0, len(u) - len(agent))))[:len(u)]
    s = EarsSession(0, **thr); mode = "speaking"; second_reset_at = None; text = None
    for h in range(len(u) // HOP_N):
        uh = u[h * HOP_N:(h + 1) * HOP_N]
        ah = ag[h * HOP_N:(h + 1) * HOP_N] if mode == "speaking" else np.zeros(HOP_N, np.float32)
        if second_reset_at == h:
            s.reset()
        events, finals = eng.process([(s, uh, ah, mode)])[s]
        if mode == "speaking" and any(ev["kind"] == "BARGE" for ev in events):
            n_barge += 1; barge_at.append(((h + 1) * HOP_N / 16000) - 1.5)
            s.reset(); mode = "listen"                        # server: halting, reset, listen
            if a.double_reset:
                second_reset_at = h + 2                       # MOUTH's barged D lands ~1 hop later
        if finals:
            text = eng.asr.transcribe([w for _, w in finals])[0]; break
    if text is None:
        continue
    got = set(words(text))
    recall.append(sum(w in got for w in ref) / len(ref))
    first3.append(sum(w in got for w in ref[:3]) / 3)

print(f"protocol: {'old (double reset)' if a.double_reset else 'current'} | trials {a.trials}, barges {n_barge}, "
      f"endpoints {len(recall)}")
if barge_at:
    print(f"  barge fired {np.median(barge_at):.2f} s (median) after the user started")
if recall:
    print(f"  word recall: mean {np.mean(recall):.2f}, min {np.min(recall):.2f} | "
          f"first 3 words recalled: mean {np.mean(first3):.2f}")
