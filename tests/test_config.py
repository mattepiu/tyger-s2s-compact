"""Plugin config shape and loader (no GPU): the repo config loads, the pre-plugin shape is rejected with the keys
that replace it, roles resolve. Run: .venv/bin/python tests/test_config.py"""
import copy, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import load_config, validate_config, llm_roles, ConfigError  # noqa: E402
from app.plugins import REGISTRY, plugin_class  # noqa: E402

BASE = load_config()


def rejects(cfg, *needles):
    try:
        validate_config(cfg)
    except ConfigError as e:
        msg = str(e)
        missing = [n for n in needles if n not in msg]
        assert not missing, f"message lacks {missing}: {msg}"
        return
    raise AssertionError("accepted")


def test_repo_config_loads():
    assert BASE["llm"]["reply"]["plugin"] in REGISTRY["llm"]
    assert BASE["tts"]["plugin"] in REGISTRY["tts"] and BASE["ears"]["plugin"] in REGISTRY["ears"]


def test_old_shape_rejected_with_new_keys():
    old = {"server": BASE["server"], "models": {"nemotron": "x", "heads_repo": "r", "heads_file": "f"},
           "ears": {"ep_thr": 0.9}, "llm": {"model": "Qwen/Qwen3.5-4B", "enable_thinking": False},
           "tts": {"model": "openbmb/VoxCPM2"}, "mouth": {"temperature": 0.7}}
    rejects(old, "`models:`", "`ears:`", "`llm.reply:`", "plugin: qwen3.5-4b", "`llm.reply.temperature`",
            "plugin: voxcpm2", "plugin: nemotron-tyger-1.0")


def test_filler_defaults_to_reply_instance():
    cfg = copy.deepcopy(BASE); cfg["llm"].pop("filler", None)
    roles = llm_roles(validate_config(cfg))
    assert roles["filler"] is roles["reply"] and "smart" not in roles
    assert roles["reply"][0] == cfg["llm"]["reply"]["plugin"]


def test_separate_filler_block_and_name():
    cfg = copy.deepcopy(BASE)
    cfg["llm"]["filler"] = {"plugin": "qwen3-4b-2507", "name": "small", "kv_cache_gib": 1.0}
    roles = llm_roles(validate_config(cfg))
    assert roles["filler"][0] == "small" and roles["filler"] is not roles["reply"]


def test_same_name_twice_rejected():
    cfg = copy.deepcopy(BASE)
    cfg["llm"]["filler"] = {"plugin": cfg["llm"]["reply"]["plugin"], "kv_cache_gib": 1.0}
    rejects(cfg, "llm.reply", "llm.filler", "`name:`")


def test_bad_roles_and_plugins_rejected():
    cfg = copy.deepcopy(BASE); cfg["llm"]["fast"] = "reply"
    rejects(cfg, "llm.fast", "not a role")
    cfg = copy.deepcopy(BASE); cfg["llm"]["filler"] = "smart"
    rejects(cfg, "can only reuse `reply`")
    cfg = copy.deepcopy(BASE); cfg["tts"]["plugin"] = "nope"
    rejects(cfg, "tts.plugin: nope", "voxcpm2")
    cfg = copy.deepcopy(BASE); del cfg["llm"]["reply"]
    rejects(cfg, "llm.reply")


def test_registry_unknown_name():
    try:
        plugin_class("llm", "nope")
    except KeyError as e:
        assert "qwen3.5-4b" in str(e)
    else:
        raise AssertionError("resolved an unknown plugin")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
