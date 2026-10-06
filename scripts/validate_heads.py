"""R-06b check: the TYGER-S2S-endpointer release, loaded by TygerHeads, is the research model. The research classes
(scripts/reference/ears_ref: EndpointModel on YieldModel on StateChannelModel) load the source checkpoint the release
was exported from and merge its LoRA as serving always did; TygerHeads loads the release. Compares every weight (bit
for bit) and the five heads' outputs on random frame sequences (no audio). Prints numbers only. Needs the GPU free.

  .venv/bin/python scripts/validate_heads.py          # expect "weights identical: 335 / 335" and max |diff| 0
"""
import sys, logging
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); logging.basicConfig(level=logging.WARNING)
from app import weights  # noqa: E402
from app.config import load_config  # noqa: E402
from app.plugins.ears.nemotron_tyger.heads import TygerHeads  # noqa: E402
from scripts.reference.ears_ref import SOURCE_FILE, SOURCE_REPO  # noqa: E402
from scripts.reference.ears_ref.endpoint_head import EndpointModel  # noqa: E402

RESEARCH_BASE = ("proj_h", "state_emb", "in_norm", "state_head", "hazard_head")   # on EndpointModel.base


@torch.no_grad()
def main():
    e = load_config()["ears"]
    ref = EndpointModel(device="cuda"); ref.load(weights.research_checkpoint(SOURCE_REPO, SOURCE_FILE)); ref.eval()
    ref.base.backbone = ref.base.backbone.merge_and_unload()
    new = TygerHeads.from_pretrained(weights.heads_release(e["heads"], e.get("heads_revision")), device="cuda")

    def research_tensor(name):
        top = name.split(".")[0]
        owner = ref.base if top in RESEARCH_BASE or top == "backbone" else ref
        return owner.get_parameter(name) if name in dict(owner.named_parameters()) else owner.get_buffer(name)
    names = [n for n, _ in new.named_parameters()]
    same = sum(torch.equal(p, research_tensor(n)) and p.dtype == research_tensor(n).dtype
               for n, p in new.named_parameters())
    print(f"weights identical: {same} / {len(names)}")

    g = torch.Generator(device="cuda").manual_seed(0)
    worst = 0.0
    for b, t in ((1, 64), (4, 32), (2, 200)):
        hu = torch.randn(b, t, 1024, device="cuda", generator=g); ha = torch.randn(b, t, 1024, device="cuda", generator=g)
        act = torch.randn(b, t, device="cuda", generator=g).clamp(-3, 3)
        prev = torch.randint(0, 5, (b, t), device="cuda", generator=g)
        st, hz, yi, bc, ep = ref(hu, ha, act, prev)
        want = torch.cat([st, torch.sigmoid(torch.stack([hz, yi, bc, ep], -1))], -1)
        got = new.outputs(new.backbone(inputs_embeds=new.fuse(hu, ha, act, prev), use_cache=False).last_hidden_state)
        d = (want - got).abs().max().item(); worst = max(worst, d)
        print(f"  batch {b} x {t} frames: max |diff| over the 8 outputs {d:.3g}")
    print(f"outputs: max |diff| {worst:.3g}")


if __name__ == "__main__":
    main()
