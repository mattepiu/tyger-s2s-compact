"""Load config.yaml + v1/.env. Secrets (TAVILY_API_KEY, HF_TOKEN) come from the env / .env and are never
written to logs. Voices are discovered from a directory of *.wav reference clips (outside git)."""
from __future__ import annotations
import os, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # v1/


def load_env(env_path: str | None = None):
    """Load KEY=VALUE lines into os.environ (without overwriting an already-set var). Checks v1/.env first,
    then the repo-root .env (/workspace/tyger/.env) as a fallback."""
    cands = [Path(env_path)] if env_path else [ROOT / ".env", ROOT.parent / ".env"]
    for p in cands:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


def _expand(obj):
    """Expand ${ENV} / $ENV in all string values (so paths like ${HF_HOME}/... work on box + in Docker)."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(path: str | None = None) -> dict:
    import yaml
    p = Path(path) if path else (ROOT / "config.yaml")
    load_env()                       # so ${HF_HOME} etc. are set before expansion
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    return _expand(yaml.safe_load(p.read_text()))


def discover_voices(cfg: dict) -> tuple[dict, str]:
    """Return ({name: wav_path}, default_name). Scans voices.dir for *.wav; adds default_ref if given."""
    v = cfg.get("voices", {})
    voices: dict[str, str] = {}
    vdir = v.get("dir")
    if vdir and Path(vdir).is_dir():
        for w in sorted(glob.glob(os.path.join(vdir, "*.wav"))):
            voices[Path(w).stem] = w
    default_ref = v.get("default_ref")
    if default_ref and Path(default_ref).exists():
        voices.setdefault(Path(default_ref).stem, default_ref)
    default = v.get("default") or (next(iter(voices), None))
    if default not in voices and default_ref and Path(default_ref).exists():
        default = Path(default_ref).stem
    return voices, default
