"""Plugin config shape and loader (no GPU): the repo config loads, the pre-plugin shape is rejected with the keys
that replace it, roles resolve. Run: .venv/bin/python tests/test_config.py"""
import copy, os, sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.config import ROOT, load_config, validate_config, llm_roles, ConfigError  # noqa: E402
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
           "ears": {"ep_thr": 0.9}, "llm": {"model": "some/old-model", "enable_thinking": False},
           "tts": {"model": "openbmb/VoxCPM2"}, "mouth": {"temperature": 0.7}}
    rejects(old, "`models:`", "`ears:`", "`llm.reply:`", "plugin: gemma-4-e4b", "`llm.reply.temperature`",
            "plugin: voxcpm2", "plugin: nemotron-tyger-1.0")


def test_research_heads_keys_rejected():
    cfg = copy.deepcopy(BASE)
    cfg["ears"] = {**cfg["ears"], "heads_repo": "r", "heads_file": "f.pt"}
    rejects(cfg, "`ears.heads_repo`", "`ears.heads: nineninesix/TYGER-S2S-endpointer`", "`ears.heads_revision`")


def test_filler_defaults_to_reply_instance():
    cfg = copy.deepcopy(BASE); cfg["llm"].pop("filler", None); cfg["llm"].pop("smart", None)
    roles = llm_roles(validate_config(cfg))
    assert roles["filler"] is roles["reply"] and "smart" not in roles
    assert roles["reply"][0] == cfg["llm"]["reply"]["plugin"]


def test_separate_filler_block_and_name():
    cfg = copy.deepcopy(BASE)
    cfg["llm"]["filler"] = {"plugin": "gemma-4-e4b-nvfp4", "name": "small", "kv_cache_gib": 1.0}
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


def test_fake_config_loads():
    cfg = load_config(str(ROOT / "config/fake.yaml"))
    assert cfg["ears"]["plugin"] == "fake" and cfg["tts"]["plugin"] == "fake"


def test_plugin_options_checked_at_load():
    """A block's keys are its plugin's Options: a typo, a missing required key or a wrong type fails the load with
    the key named (before R-07 a typo was silently ignored and the default used)."""
    cfg = copy.deepcopy(BASE); cfg["llm"]["reply"]["kv_cache_gb"] = 2.0
    rejects(cfg, "`llm.reply`", "unknown option `kv_cache_gb`", "did you mean `kv_cache_gib`")
    cfg = copy.deepcopy(BASE); cfg["ears"]["ring_frame"] = 1024
    rejects(cfg, "`ears`", "`ring_frame`", "`ring_frames`")
    cfg = copy.deepcopy(BASE); cfg["tts"]["allow_zero_shot"] = "yes"
    rejects(cfg, "`tts`", "`allow_zero_shot` must be a bool")
    cfg = copy.deepcopy(BASE); cfg["llm"]["smart"] = {"plugin": "openai", "max_tokens": 512}
    rejects(cfg, "`llm.smart`", "`model` is not set")
    cfg = copy.deepcopy(BASE); del cfg["ears"]["ep_thr"]           # a threshold without a default
    rejects(cfg, "`ears`", "`ep_thr` is not set")
    cfg = copy.deepcopy(BASE); cfg["ears"]["ep_deb"] = "two"
    rejects(cfg, "`ears`")


def test_renamed_option_names_the_new_key():
    """tts.deploy_config became tts.engine_config (R-11, with the engine configs in config/engines/)."""
    cfg = copy.deepcopy(BASE); cfg["tts"]["deploy_config"] = cfg["tts"].pop("engine_config")
    rejects(cfg, "`tts`", "`deploy_config` is now `engine_config`")


def test_config_files_in_config_dir():
    """config/: the default config, the fake one, the prompts and the engine config it names, all repo-relative."""
    from app.config import DEFAULT, repo_path
    from app.prompts import load_prompts
    assert DEFAULT == ROOT / "config" / "default.yaml" and DEFAULT.is_file()
    assert repo_path(BASE["tts"]["engine_config"]).is_file()
    no_paths = {**BASE, "mouth": {k: v for k, v in BASE["mouth"].items() if k not in ("prompts", "prompt_overrides_dir")}}
    assert load_prompts(no_paths, ["x"])["x"].source == "prompts.yaml"          # the default: config/prompts.yaml


def test_options_types():
    """Scalars as YAML leaves them: an int where a float is declared, an all-digit revision hash as a string."""
    from app.plugins.ears.nemotron_tyger.plugin import NemotronTyger
    from app.plugins.llm.vllm_base import VLLMChatPlugin
    o = NemotronTyger.options({**BASE["ears"], "gather_ms": 20, "heads_revision": 1234567})
    assert o.gather_ms == 20.0 and isinstance(o.gather_ms, float) and o.heads_revision == "1234567"
    assert o.heads == BASE["ears"]["heads"] and o.ring_frames == BASE["ears"]["ring_frames"]
    assert VLLMChatPlugin.options({"plugin": "gemma-4-e4b", "name": "x"}).kv_cache_gib == 3.0   # shape keys skipped
    for bad in ({"max_num_seqs": 8.5}, {"max_num_seqs": True}, {"temperature": "hot"}, {"max_model_len": None}):
        try:
            VLLMChatPlugin.options(bad); raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert next(iter(bad)) in str(e)


def test_registry_unknown_name():
    try:
        plugin_class("llm", "nope")
    except KeyError as e:
        assert "gemma-4-e4b" in str(e)
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
