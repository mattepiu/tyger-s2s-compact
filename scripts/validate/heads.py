"""Check the TYGER-S2S-endpointer release as TygerHeads loads it, on real inputs: nemotron's layer-15 features of public
LibriSpeech clips (one clip as the user, another as the agent), their frame energy, the listening state. Numbers only.
  * it loads: all tensors in their stored dtypes, model_type and the pinned frontend fit this plugin;
  * its outputs are deterministic and well-formed (finite; the four probabilities in [0, 1]);
  * stepping it one frame at a time with a KV cache (as the engine and the reference do) gives what one pass over the
    whole sequence gives;
  * a sequence gives the same outputs alone and inside a batch (as the engine batches conversations).
The last two are bf16 matmuls in a different order: they agree within a tolerance on the probabilities (what the turn
rules read). Random features are no use here: far outside what the heads saw, they push the state logits to |40|+
and amplify bf16 noise (in fp32 the same comparisons agree to ~2e-4). Needs the GPU free (stop run.sh).

  .venv/bin/python scripts/validate/heads.py
"""
import sys, logging
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); logging.basicConfig(level=logging.WARNING)
from app.plugins.ears.nemotron_tyger.release import heads_release  # noqa: E402
from app.config import load_config  # noqa: E402
from app.plugins.ears.nemotron_tyger.heads import TygerHeads  # noqa: E402

TOL = 0.02          # on the four probabilities; bf16 reordering on real features lands well under it


def real_inputs(heads, n=4):
    """(user L15, agent L15, energy, previous state) for n conversations: 8 s of LibriSpeech clip k as the user and of
    clip k+n as the agent, listening. Audio in memory only."""
    import numpy as np
    from datasets import load_dataset
    from app.plugins.ears.nemotron_tyger import rules
    from app.plugins.ears.nemotron_tyger.asr import NemotronAsr
    from app.plugins.ears.nemotron_tyger.constants import LISTEN, SR, WINDOW_S
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    n_win = int(WINDOW_S * SR)
    clip = lambda i: np.pad(np.asarray(ds[i]["audio"]["array"], np.float32), (0, n_win))[:n_win]  # noqa: E731
    fe = heads.frontend
    asr = NemotronAsr(fe["repo_id"], fe["revision"], max_batch=1, cuda_graph=False)
    users, agents = [clip(k) for k in range(n)], [0.6 * clip(k + n) for k in range(n)]
    hu, ha = torch.stack(asr.encode(users)), torch.stack(asr.encode(agents))
    energy = np.stack([rules.frame_energy(u) for u in users])
    t = min(hu.shape[1], ha.shape[1], energy.shape[1])          # the encoder may give one frame more, as in the engine
    act = torch.tensor(energy[:, :t], device="cuda")
    del asr; torch.cuda.empty_cache()
    return hu[:, :t], ha[:, :t], ((act + 50) / 25).clamp(-3, 3), torch.full((n, t), LISTEN, device="cuda")


@torch.no_grad()
def main():
    e = load_config()["ears"]
    heads = TygerHeads.from_pretrained(heads_release(e["heads"], e.get("heads_revision")), device="cuda")
    c = heads.config
    n = sum(p.numel() for p in heads.parameters())
    print(f"release {c.get('version')} @ {(e.get('heads_revision') or 'latest')[:8]}: {n:,} parameters; frontend "
          f"{c['frontend']['repo_id']} @ {c['frontend']['revision'][:8]}, layer {c['frontend']['layer']}")
    full = lambda x: heads.outputs(heads.backbone(inputs_embeds=heads.fuse(*x), use_cache=False).last_hidden_state)  # noqa: E731
    x = real_inputs(heads)
    a, b = full(x), full(x)
    probs = a[..., 4:]
    print(f"deterministic: {torch.equal(a, b)} | finite: {bool(torch.isfinite(a).all())} | "
          f"probabilities in [0, 1]: {bool(((probs >= 0) & (probs <= 1)).all())}")

    cache, steps = None, []                       # frame by frame, as the engine / reference run it
    for i in range(x[0].shape[1]):
        out = heads.backbone(inputs_embeds=heads.fuse(*(v[:1, i:i + 1] for v in x)), past_key_values=cache, use_cache=True)
        cache = out.past_key_values; steps.append(heads.outputs(out.last_hidden_state))
    d_step = (torch.cat(steps, 1) - a[:1])[..., 4:].abs().max().item()
    d_batch = (full(tuple(v[2:3] for v in x)) - a[2:3])[..., 4:].abs().max().item()
    print(f"{x[0].shape[0]} x {x[0].shape[1]} frames | probabilities: frame by frame vs one pass max |diff| {d_step:.4f}, "
          f"alone vs in a batch max |diff| {d_batch:.4f} (tolerance {TOL})")
    ok = torch.equal(a, b) and bool(torch.isfinite(a).all()) and d_step < TOL and d_batch < TOL
    print("heads release: OK" if ok else "heads release: FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
