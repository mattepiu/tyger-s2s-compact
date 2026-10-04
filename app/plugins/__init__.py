"""Plugin registry: kind -> name -> "module:Class", resolved lazily with importlib (PLUGINS.md). No dynamic
loading of third-party packages: a new plugin is a module in app/plugins plus a line here."""
from __future__ import annotations
import importlib

REGISTRY = {
    "llm": {"qwen3-4b-2507": "app.plugins.llm.qwen3_2507:Qwen3Instruct2507",
            "qwen3.5-4b": "app.plugins.llm.qwen35:Qwen35"},
    "tts": {"voxcpm2": "app.plugins.tts.voxcpm2:VoxCPM2"},
    # nemotron ASR encoder + the tyger 1.0 turn-taking heads (exp-12 arm C: endpoint / barge / backchannel)
    "ears": {"nemotron-tyger-1.0": "app.plugins.ears.nemotron:NemotronTyger"},
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
