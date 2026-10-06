"""Weights: fetch the private turn-taking heads release. All downloads go to HF_HOME (mounted cache); nothing is baked
into the image. The other models (nemotron, the
LLM, VoxCPM2) are downloaded by their loaders on first start."""
from __future__ import annotations
import os
from pathlib import Path


def _hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        tf = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "token"
        if tf.exists():
            t = tf.read_text().strip()
    return t or None


# ---- the turn-taking heads ----------------------------------------------------------------------
def heads_release(repo: str, revision: str | None = None) -> str:
    """The local directory of a TYGER-S2S-endpointer release (config.json + model.safetensors), downloaded once into
    HF_HOME. The repo is private: needs HF_TOKEN."""
    from huggingface_hub import snapshot_download
    token = _hf_token()
    if not token:
        raise RuntimeError(f"HF_TOKEN required for the private heads release {repo!r}")
    return snapshot_download(repo, revision=revision, token=token, allow_patterns=["config.json", "model.safetensors"])

