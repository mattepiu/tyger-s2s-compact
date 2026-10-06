"""Load config/default.yaml + v1/.env and check its plugin shape. Secrets (TAVILY_API_KEY, HF_TOKEN) come from the env /
.env and are never written to logs."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # v1/
CONFIG_DIR = ROOT / "config"                 # the stack's config, prompts and engine configs
DEFAULT = CONFIG_DIR / "default.yaml"         # what run.sh loads unless CONFIG names another file


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


class ConfigError(ValueError):
    pass


# reply: ordinary turns + the tool phase-B answer; filler: backchannels, tool lead-ins, still-looking lines (short,
# low latency); smart: escalated turns and background tasks (optional)
LLM_ROLES = ("reply", "filler", "smart")


def _old_shape(cfg: dict) -> list[str]:
    """What still uses the pre-plugin config shape, each with the key that replaces it."""
    found = []
    if "models" in cfg:
        found.append("top-level `models:` -> the heads release goes in `ears.heads` (it pins its nemotron encoder)")
    ears = cfg.get("ears")
    if isinstance(ears, dict) and ("heads_repo" in ears or "heads_file" in ears):
        found.append("`ears.heads_repo` / `ears.heads_file` (a research .pt) -> `ears.heads: "
                     "nineninesix/TYGER-S2S-endpointer` (the release) and `ears.heads_revision`")
    llm = cfg.get("llm")
    if isinstance(llm, dict) and "model" in llm:
        found.append("`llm.model` and the other flat `llm:` keys -> move them under `llm.reply:` with "
                     "`plugin: gemma-4-e4b`; enable_thinking / language_model_only are now "
                     "part of the plugin")
    if "temperature" in (cfg.get("mouth") or {}):
        found.append("`mouth.temperature` -> `llm.reply.temperature`")
    for kind in ("tts", "ears"):
        if isinstance(cfg.get(kind), dict) and "plugin" not in cfg[kind]:
            found.append(f"`{kind}:` has no `plugin:` -> add `plugin: "
                         f"{'voxcpm2' if kind == 'tts' else 'nemotron-tyger-1.0'}`")
    return found


def validate_config(cfg: dict) -> dict:
    """Check the plugin config shape (PLUGINS.md) and every block's options. The old shape is rejected, never
    reinterpreted."""
    from app.plugins import REGISTRY
    old = _old_shape(cfg)
    if old:
        raise ConfigError("config uses the pre-plugin shape (see PLUGINS.md -> Config migration):\n  - "
                          + "\n  - ".join(old))
    llm = cfg.get("llm")
    if not isinstance(llm, dict) or not isinstance(llm.get("reply"), dict):
        raise ConfigError("`llm.reply:` is missing: a block with `plugin:` and its options")
    for role, block in llm.items():
        if role not in LLM_ROLES:
            raise ConfigError(f"`llm.{role}:` is not a role (roles: {', '.join(LLM_ROLES)})")
        if role != "reply" and isinstance(block, str):
            if block != "reply":
                raise ConfigError(f"`llm.{role}: {block}`: a role can only reuse `reply`")
            continue
        _check_block(REGISTRY, "llm", f"llm.{role}", block)
    seen = {}                       # instance name -> role: two different blocks can't share a name (or instance)
    for role, block in llm.items():
        if isinstance(block, dict):
            name = block.get("name") or block["plugin"]
            if name in seen:
                raise ConfigError(f"`llm.{seen[name]}` and `llm.{role}` are both named `{name}`: give one a "
                                  f"`name:`, or write `{role}: reply` to share the reply instance")
            seen[name] = role
    _check_block(REGISTRY, "tts", "tts", cfg.get("tts"))
    _check_block(REGISTRY, "ears", "ears", cfg.get("ears"))
    return cfg


def _check_block(registry, kind, where, block):
    """A block names a registered plugin, and the rest of it fits that plugin's Options (unknown keys included)."""
    from app.plugins import plugin_class
    if not isinstance(block, dict) or "plugin" not in block:
        raise ConfigError(f"`{where}:` must be a block with `plugin:`")
    if block["plugin"] not in registry[kind]:
        raise ConfigError(f"`{where}.plugin: {block['plugin']}` is not a {kind} plugin "
                          f"(known: {', '.join(sorted(registry[kind]))})")
    try:
        plugin_class(kind, block["plugin"]).options(block)
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"`{where}` (plugin {block['plugin']}): {e.args[0] if e.args else e}") from None


def llm_roles(cfg: dict) -> dict[str, tuple[str, dict]]:
    """{role: (instance name, block)} for the configured LLM roles. `filler` defaults to `reply`; a role that says
    `reply` shares that instance (same name, same block). `smart` appears only if configured."""
    llm = cfg["llm"]
    out = {}
    for role in LLM_ROLES:
        block = llm.get(role, "reply" if role == "filler" else None)
        if block is None:
            continue
        out[role] = out["reply"] if block == "reply" else (block.get("name") or block["plugin"], block)
    return out


def load_config(path: str | None = None) -> dict:
    import yaml
    p = Path(path) if path else DEFAULT
    load_env()                       # so ${HF_HOME} etc. are set before expansion
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    return validate_config(_expand(yaml.safe_load(p.read_text())))


def repo_path(p: str | None) -> Path | None:
    """Config paths may be relative to the repo root (e.g. voices.dir: voices)."""
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q
