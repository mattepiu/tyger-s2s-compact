"""Plugin registry: kind -> name -> "module:Class", resolved lazily with importlib (PLUGINS.md). No dynamic
loading of third-party packages: a new plugin is a module in app/plugins plus a line here."""
from __future__ import annotations
import importlib

REGISTRY = {
    "llm": {"qwen3-4b-2507": "app.plugins.llm.qwen3_2507:Qwen3Instruct2507",
            "qwen3.5-4b": "app.plugins.llm.qwen35:Qwen35",
            "openai": "app.plugins.llm.openai_compat:OpenAICompatLLM",     # hosted: transcripts leave the box
            "fake": "app.plugins.llm.fake:FakeLLM"},
    "tts": {"voxcpm2": "app.plugins.tts.voxcpm2:VoxCPM2",
            "fake": "app.plugins.tts.fake:FakeTTS"},
    # nemotron ASR encoder + the tyger 1.0 turn-taking heads (exp-12 arm C: endpoint / barge / backchannel)
    "ears": {"nemotron-tyger-1.0": "app.plugins.ears.nemotron:NemotronTyger",
             "fake": "app.plugins.ears.fake:EnergyEars"},
}


def plugin_class(kind: str, name: str):
    """The class registered for (kind, name). Imports only its module; a plugin's heavy dependencies load in its
    constructor."""
    try:
        target = REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY.get(kind, {}))) or "none"
        raise KeyError(f"no {kind} plugin '{name}' (known: {known})") from None
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)


def build_llms(cfg: dict, eloop) -> dict:
    """{role: LLM plugin instance} for llm.reply / filler / smart. Roles that share a block share one instance."""
    import logging
    from app.config import llm_roles
    from app.plugins.base import PluginContext
    log = logging.getLogger("plugins")
    by_name, out = {}, {}
    for role, (name, block) in llm_roles(cfg).items():
        if name not in by_name:
            log.info("loading LLM %s (plugin %s) for %s", name, block["plugin"], role)
            by_name[name] = plugin_class("llm", block["plugin"])(block, PluginContext(eloop, cfg, name))
        out[role] = by_name[name]
    remote = sorted(r for r, p in out.items() if getattr(p, "remote", False))
    if remote:   # transcripts leave the box for these roles: say so once, by role and instance name only
        log.warning("remote LLM roles (transcripts leave the box): %s",
                    ", ".join(f"{r}={n}" for r, (n, _) in llm_roles(cfg).items() if r in remote))
    return out


def build_tts(cfg: dict, eloop):
    """The configured TTS plugin instance, at the 24 kHz wire rate."""
    import logging
    from app.plugins.base import PluginContext
    from app.plugins.tts import AtWireRate, WIRE_SR
    block = cfg["tts"]
    logging.getLogger("plugins").info("loading TTS %s", block["plugin"])
    tts = plugin_class("tts", block["plugin"])(block, PluginContext(eloop, cfg, block.get("name") or block["plugin"]))
    if getattr(tts, "remote", False):
        logging.getLogger("plugins").warning("remote TTS %s: reply text leaves the box", block["plugin"])
    return tts if tts.out_sr == WIRE_SR else AtWireRate(tts)


def build_ears(cfg: dict):
    """The configured EARS plugin instance (EARS process; no engine loop)."""
    from app.plugins.base import PluginContext
    block = cfg["ears"]
    return plugin_class("ears", block["plugin"])(block, PluginContext(None, cfg, block.get("name") or block["plugin"]))
