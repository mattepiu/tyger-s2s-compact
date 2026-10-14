"""The tyger bundle (`ears.model` at `ears.revision`; private): one model.safetensors with nemotron-3.5-asr
(`nemotron.*`; its config, processor and tokenizer in nemotron/) and the turn-taker GRU that reads its layer 15
(`turn_taker.*`); config.json describes both. Fetched once into HF_HOME, then loaded from it: nothing else is
downloaded. The checks here make a bundle that needs another audio grid or other turn rules fail at load instead of
misbehaving.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

from .constants import (ENC_HIDDEN, FRAME_S, HOP_N, L15, LISTEN, MIN_SPEECH_MS, SPEAK, SR, STATE_NAMES, VAD_DB,
                        WARMUP_FRAMES, WINDOW_S)

MODEL_TYPE = "tyger_nemotron_gru"
FILES = ["config.json", "model.safetensors", "nemotron/*.json"]


def _hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        tf = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "token"
        if tf.exists():
            t = tf.read_text().strip()
    return t or None


def fetch(repo: str, revision: str | None = None) -> Path:
    """The bundle's local directory: `repo` itself if it is one, else its download into HF_HOME (once). The repo is
    private: needs HF_TOKEN."""
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    token = _hf_token()
    if not token:
        raise RuntimeError(f"HF_TOKEN required for the private bundle {repo!r}")
    return Path(snapshot_download(repo, revision=revision, token=token, allow_patterns=FILES))


def check_config(config: dict):
    """The bundle must be the GRU format this plugin runs, on its audio grid and turn rules (constants.py);
    ValueError naming what differs."""
    if config.get("model_type") != MODEL_TYPE or config.get("format_version") != 1:
        raise ValueError(f"not a {MODEL_TYPE} bundle of format 1 (model_type {config.get('model_type')!r})")
    arch = config["controller"]["architecture"]
    fe, pol = config["frontend"], config["default_policy"]
    want = {"dtype": (config["dtype"], "float32"), "controller": (arch["kind"], "gru"),
            "state_names": (tuple(config["state_names"]), STATE_NAMES),
            "output_order": (tuple(config["output_order"]),
                             ("state_logits", "hazard_logits", "yield_logits", "bc_logits", "endpoint_logits")),
            "live previous state": (config["streaming"]["live_previous_state"], {"listen": LISTEN, "speaking": SPEAK}),
            "layer": (fe["layer"], L15), "hidden_size": (fe["hidden_size"], ENC_HIDDEN),
            "sample_rate": (fe["sample_rate"], SR), "frame_period_s": (fe["frame_period_s"], FRAME_S),
            "live_window_s": (fe["live_window_s"], WINDOW_S), "live_hop_s": (fe["live_hop_s"], HOP_N / SR),
            "num_lookahead_tokens": (fe["num_lookahead_tokens"], 0),
            "energy_norm": (fe["energy_norm"], "clip((dB + 50) / 25, -3, 3)"),
            # the turn rules' fixed guards (rules.py; EarsSession's bc_refractory); its thresholds are ears.* in config
            "vad_db": (pol["vad_db"], VAD_DB), "min_speech_ms": (pol["min_speech_ms"], MIN_SPEECH_MS),
            "warmup_frames": (pol["warmup_frames"], WARMUP_FRAMES),
            "bc_refractory_frames": (pol["bc_refractory_frames"], 15)}
    bad = [f"{k}: bundle {got!r}, plugin {exp!r}" for k, (got, exp) in want.items() if got != exp]
    if bad:
        raise ValueError("the bundle needs another frontend or other rules than this plugin runs: " + "; ".join(bad))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load(folder):
    """-> (nemotron model, its processor, the TurnTaker, config), all fp32 on the CPU. The weights and nemotron's
    files must match the checksums in config.json, and every tensor must load (strict). Turns TF32 off for the whole
    process, as the bundle was validated (its config's "tf32": false; EARS's process runs only EARS)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig
    from .controller import TurnTaker
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    check_config(config)
    for name, digest in config["frontend_file_sha256"].items():
        if _sha256(folder / "nemotron" / name) != digest:
            raise ValueError(f"bundle {folder}: nemotron/{name} doesn't match its checksum")
    if _sha256(folder / "model.safetensors") != config["weights_sha256"]:
        raise ValueError(f"bundle {folder}: model.safetensors doesn't match its checksum")
    weights = load_file(str(folder / "model.safetensors"))
    if len(weights) != config["tensor_count"] or any(t.dtype != torch.float32 for t in weights.values()):
        raise ValueError(f"bundle {folder}: {len(weights)} tensors (want {config['tensor_count']}), all fp32")
    parts = {"nemotron": {}, "turn_taker": {}}
    for k, v in weights.items():
        prefix, _, name = k.partition(".")
        parts[prefix][name] = v
    del weights
    nemotron = AutoModel.from_config(AutoConfig.from_pretrained(folder / "nemotron"), dtype=torch.float32)
    nemotron.load_state_dict(parts["nemotron"], strict=True)
    nemotron.generation_config = GenerationConfig.from_pretrained(folder / "nemotron")
    proc = AutoProcessor.from_pretrained(folder / "nemotron")
    arch = config["controller"]["architecture"]
    turn_taker = TurnTaker(arch["width"], arch["layers"])
    turn_taker.load_state_dict(parts["turn_taker"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    return nemotron.eval(), proc, turn_taker.eval(), config
