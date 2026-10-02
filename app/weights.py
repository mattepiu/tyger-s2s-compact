"""Weights: warm the HF cache for public models and fetch the private endpoint-head checkpoint. All downloads
go to HF_HOME (mounted cache); nothing is baked into the image. The MOUTH's models (Qwen3-4B, VoxCPM2) are
downloaded by the vLLM engines themselves on first start."""
from __future__ import annotations
import os, logging
from pathlib import Path

log = logging.getLogger("weights")


def _hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        tf = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "token"
        if tf.exists():
            t = tf.read_text().strip()
    return t or None


# ---- public models: warm the HF cache ----------------------------------------------------------
def ensure_public_cached(repos: list[str]):
    from huggingface_hub import snapshot_download
    for r in repos:
        snapshot_download(r, token=_hf_token())


# ---- private endpoint-head chain ---------------------------------------------------------------
def ensure_heads(repo: str, subpath: str, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    token = token or _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN required for the private endpoint-head repo %r" % repo)
    path = hf_hub_download(repo_id=repo, filename=subpath, token=token)
    verify_head_ckpt(path)
    return path


def verify_head_ckpt(path: str):
    """Confirm the ep checkpoint carries the LoRA + all heads (so bc_s1 isn't needed separately)."""
    import torch
    ck = torch.load(path, map_location="cpu")
    need_top = {"base", "proj_agent", "proj_act", "yield_head", "bc_head", "ep_head"}
    missing = need_top - set(ck.keys())
    if missing:
        raise RuntimeError(f"endpoint ckpt {path} missing keys {missing}; may need bc_s1 too")
    base_keys = " ".join(ck["base"].keys())
    has_lora = "lora" in base_keys.lower()
    log.info("endpoint ckpt OK: heads=%s lora_in_base=%s", sorted(need_top), has_lora)
    if not has_lora:
        log.warning("no LoRA weights found inside ckpt['base'] — endpoint chain expects a LoRA backbone")
    return True
