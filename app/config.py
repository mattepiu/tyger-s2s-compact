"""Load config/default.yaml + the repo's .env and check its plugin shape. Secrets (HF_TOKEN, OPENAI_API_KEY,
TYGER_API_SECRET) come from the env / .env and are never written to logs. A few `server:` settings can be set by env
vars too (SERVER_ENV), and the env var wins over the file.

`python -m app.config [--config PATH] KEY` prints a setting's effective value (run.sh asks for server.pages)."""
from __future__ import annotations
import difflib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the repo
CONFIG_DIR = ROOT / "config"                 # the stack's config, prompts and engine configs
DEFAULT = CONFIG_DIR / "default.yaml"         # what run.sh loads unless CONFIG names another file


def load_env(env_path: str | None = None):
    """Load KEY=VALUE lines into os.environ (without overwriting an already-set var) from `env_path`, else from the
    repo's own .env."""
    p = Path(env_path) if env_path else ROOT / ".env"
    if not p.exists():
        return
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


# The `server:` settings an env var overrides, as Docker sets them when it starts the container: the env var wins over
# the file, and an empty one counts as unset. TYGER_API_SECRET, the token's secret, is env only (app/ears/access.py).
SERVER_ENV = {"pages": "TYGER_PAGES", "max_session_s": "TYGER_MAX_SESSION_S"}
SERVER_DEFAULTS = {"pages": True,                   # serve the pages (false: API mode)
                   "max_session_s": 180}            # 0: no limit
BOOLEANS = {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False, "off": False}


def server_env(cfg: dict) -> dict:
    """Set cfg's `server:` keys from the env vars in SERVER_ENV that are set."""
    srv = cfg.setdefault("server", {})
    for key, var in SERVER_ENV.items():
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        if key == "pages":
            if raw.lower() not in BOOLEANS:
                raise ConfigError(f"{var}={raw} is not true or false")
            srv[key] = BOOLEANS[raw.lower()]
        else:
            try:
                srv[key] = float(raw)
            except ValueError:
                raise ConfigError(f"{var}={raw} is not a number of seconds") from None
    return cfg


def _check_server(srv: dict):
    """The server settings, defaults filled in: pages a boolean, the session limit seconds (0: none)."""
    for key, default in SERVER_DEFAULTS.items():
        srv.setdefault(key, default)
    if not isinstance(srv["pages"], bool):
        raise ConfigError(f"`server.pages: {srv['pages']}` must be true or false ({SERVER_ENV['pages']})")
    limit = srv["max_session_s"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0:
        raise ConfigError(f"`server.max_session_s: {limit}` must be seconds, 0 for no limit "
                          f"({SERVER_ENV['max_session_s']})")


# reply: ordinary turns + the tool phase-B answer; filler: backchannels, tool lead-ins, still-looking lines (short,
# low latency); smart: escalated turns and background tasks (optional)
LLM_ROLES = ("reply", "filler", "smart")
# the `mouth:` keys MOUTH reads (app/mouth/service.py, app/mouth/__main__.py, app/prompts; EARS reads tasks too):
# any other key is a typo or a removed option, and stops startup like an unknown plugin option
MOUTH_KEYS = ("prompts", "max_gen", "history_turns", "bc_max_new", "bc_min_words", "first_chunk", "cpu_threads",
              "escalation", "tasks")


def _old_shape(cfg: dict) -> list[str]:
    """What still uses the pre-plugin config shape, each with the key that replaces it."""
    found = []
    if "models" in cfg:
        found.append("top-level `models:` -> the turn-taking model goes in `ears.model` (nineninesix/tyger: nemotron "
                     "and the turn-taker in one bundle)")
    ears = cfg.get("ears")
    if isinstance(ears, dict) and ("heads_repo" in ears or "heads_file" in ears or "heads" in ears):
        found.append("`ears.heads` / `ears.heads_repo` / `ears.heads_file` (the tyger 1.0 heads, removed) -> "
                     "`ears.plugin: tyger` with `ears.model: nineninesix/tyger` and `ears.revision`")
    if isinstance(ears, dict) and ears.get("plugin") in ("nemotron-tyger-1.0", "tyger-s2s"):
        found.append(f"`ears.plugin: {ears['plugin']}` -> `tyger` (one EARS plugin since 2026-10-13)")
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
                         f"{'voxcpm2' if kind == 'tts' else 'tyger'}`")
    return found


def validate_config(cfg: dict) -> dict:
    """Check the plugin config shape, every block's options and the `mouth:` keys. The old shape is rejected, never
    reinterpreted."""
    from app.plugins import REGISTRY
    old = _old_shape(cfg)
    if old:
        raise ConfigError("config uses the pre-plugin shape; each old key and what replaces it:\n  - "
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
    _check_mouth(cfg.get("mouth") or {})
    _check_server(cfg.setdefault("server", {}))
    _check_block(REGISTRY, "ears", "ears", cfg.get("ears"))
    return cfg


def _check_mouth(mouth: dict):
    from app.mouth.chunker import FIRST_CHUNK_RULES
    for key in mouth:
        if key not in MOUTH_KEYS:
            near = difflib.get_close_matches(key, MOUTH_KEYS, n=1)
            raise ConfigError(f"unknown option `mouth.{key}`" + (f" (did you mean `mouth.{near[0]}`?)" if near else "")
                              + f"; options: {', '.join(sorted(MOUTH_KEYS))}")
    rule = mouth.get("first_chunk", "sentence")
    if rule not in FIRST_CHUNK_RULES:
        raise ConfigError(f"`mouth.first_chunk: {rule}` is not one of {', '.join(FIRST_CHUNK_RULES)}")


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
    return validate_config(server_env(_expand(yaml.safe_load(p.read_text()))))


def repo_path(p: str | None) -> Path | None:
    """Config paths may be relative to the repo root (e.g. voices.dir: voices)."""
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main():
    import argparse
    ap = argparse.ArgumentParser(description="a setting's effective value (the env vars included), as JSON")
    ap.add_argument("--config", default=None)
    ap.add_argument("key", help="e.g. server.pages")
    a = ap.parse_args()
    value = load_config(a.config)
    for k in a.key.split("."):
        value = value[k]
    print(json.dumps(value))


if __name__ == "__main__":
    main()
