"""Plugin config shape and loader (no GPU): the repo config loads, the pre-plugin shape is rejected with the keys
that replace it, roles resolve; the server settings, their env vars (which win over the file) and their checks.
Run: .venv/bin/python tests/test_config.py"""
import copy, os, subprocess, sys
from contextlib import contextmanager

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
    cfg["llm"]["filler"] = {"plugin": "gemma-4-e4b", "name": "small", "kv_cache_gib": 1.0}   # a 2nd instance
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
    cfg = copy.deepcopy(BASE); cfg["llm"]["smart"] = {"plugin": "openai-responses", "max_tokens": 512}
    rejects(cfg, "`llm.smart`", "`model` is not set")
    cfg = copy.deepcopy(BASE); del cfg["ears"]["ep_thr"]           # a threshold without a default
    rejects(cfg, "`ears`", "`ep_thr` is not set")
    cfg = copy.deepcopy(BASE); cfg["ears"]["ep_deb"] = "two"
    rejects(cfg, "`ears`")


def test_first_chunk_rule_checked():
    cfg = copy.deepcopy(BASE); cfg["mouth"]["first_chunk"] = "word"
    rejects(cfg, "`mouth.first_chunk: word`", "clause", "sentence")
    cfg["mouth"]["first_chunk"] = "sentence"
    validate_config(cfg)


def test_mouth_keys_checked():
    """`mouth:` takes only the keys MOUTH reads: anything else (a typo, an option that was removed) fails the load
    with the key named, instead of being ignored."""
    cfg = copy.deepcopy(BASE); cfg["mouth"]["histroy_turns"] = 4
    rejects(cfg, "unknown option `mouth.histroy_turns`", "did you mean `mouth.history_turns`")
    cfg = copy.deepcopy(BASE); cfg["mouth"]["voice_dir"] = "voices"
    rejects(cfg, "unknown option `mouth.voice_dir`", "options: bc_max_new")


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
    no_paths = {**BASE, "mouth": {k: v for k, v in BASE["mouth"].items() if k != "prompts"}}
    assert load_prompts(no_paths).source == "prompts.yaml"                      # the default: config/prompts.yaml


def test_options_types():
    """Scalars as YAML leaves them: an int where a float is declared, an all-digit revision hash as a string."""
    from app.plugins.ears.nemotron_tyger.plugin import NemotronTyger
    from app.plugins.llm.local.vllm_base import VLLMChatPlugin
    o = NemotronTyger.options({**BASE["ears"], "gather_ms": 20, "heads_revision": 1234567})
    assert o.gather_ms == 20.0 and isinstance(o.gather_ms, float) and o.heads_revision == "1234567"
    assert o.heads == BASE["ears"]["heads"] and o.ring_frames == BASE["ears"]["ring_frames"]
    assert VLLMChatPlugin.options({"plugin": "gemma-4-e4b", "name": "x"}).kv_cache_gib == 3.0   # shape keys skipped
    for bad in ({"max_num_seqs": 8.5}, {"max_num_seqs": True}, {"temperature": "hot"}, {"max_model_len": None}):
        try:
            VLLMChatPlugin.options(bad); raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert next(iter(bad)) in str(e)


def test_env_only_from_the_repo_or_the_path_given():
    """load_env reads the repo's own .env, or the path given, never a .env in the folder above the repo (a
    leftover of the old v1/ layout)."""
    import tempfile
    from pathlib import Path
    from unittest import mock
    from app import config
    keys = ("TYGER_TEST_PARENT_ENV", "TYGER_TEST_REPO_ENV", "TYGER_TEST_GIVEN_ENV")
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"; repo.mkdir()
        (Path(d) / ".env").write_text("TYGER_TEST_PARENT_ENV=1\n")
        (repo / ".env").write_text("TYGER_TEST_REPO_ENV=1\n")
        (Path(d) / "given.env").write_text("TYGER_TEST_GIVEN_ENV=1\n")
        try:
            with mock.patch.object(config, "ROOT", repo):
                config.load_env()
                assert os.environ.get("TYGER_TEST_REPO_ENV") == "1"
                assert "TYGER_TEST_PARENT_ENV" not in os.environ, "read the .env above the repo"
                (repo / ".env").unlink()
                config.load_env()                                   # no .env in the repo: nothing, no fallback
                assert "TYGER_TEST_PARENT_ENV" not in os.environ, "fell back to the .env above the repo"
                config.load_env(str(Path(d) / "given.env"))
                assert os.environ.get("TYGER_TEST_GIVEN_ENV") == "1"
        finally:
            for k in keys:
                os.environ.pop(k, None)


SERVER_VARS = ("TYGER_PAGES", "TYGER_MAX_SESSION_S")


@contextmanager
def env(**values):
    """The server's env vars as given, the others unset, for the block."""
    old = {k: os.environ.get(k) for k in SERVER_VARS}
    for k in SERVER_VARS:
        os.environ.pop(k, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_server_settings_from_the_file_or_the_defaults():
    with env():
        srv, fake = load_config()["server"], load_config(str(ROOT / "config/fake.yaml"))["server"]
    assert (srv["pages"], srv["max_session_s"]) == (True, 180)
    assert (fake["pages"], fake["max_session_s"]) == (True, 180)            # not in the file: the defaults
    assert "allowed_origins" not in srv                                     # removed (the user, 2026-10-13)


def test_env_vars_beat_the_file():
    with env(TYGER_PAGES="false", TYGER_MAX_SESSION_S="2.5"):
        srv = load_config()["server"]
    assert srv["pages"] is False and srv["max_session_s"] == 2.5
    for word, value in (("OFF", False), ("0", False), ("no", False), ("1", True), ("yes", True), ("on", True)):
        with env(TYGER_PAGES=word):
            assert load_config()["server"]["pages"] is value, word
    with env(TYGER_PAGES=" ", TYGER_MAX_SESSION_S="0"):                    # empty: unset; 0: no limit
        srv = load_config()["server"]
    assert srv["pages"] is True and srv["max_session_s"] == 0
    for var, value, needle in (("TYGER_PAGES", "maybe", "TYGER_PAGES=maybe"),
                               ("TYGER_MAX_SESSION_S", "three minutes", "TYGER_MAX_SESSION_S"),
                               ("TYGER_MAX_SESSION_S", "-1", "TYGER_MAX_SESSION_S")):
        with env(**{var: value}):
            try:
                load_config()
            except ConfigError as e:
                assert needle in str(e), str(e)
            else:
                raise AssertionError(f"accepted {var}={value}")
    out = subprocess.run([sys.executable, "-m", "app.config", "server.pages"], cwd=ROOT, capture_output=True,
                         text=True, env={**os.environ, "TYGER_PAGES": "off"}).stdout
    assert out.strip() == "false", out                                     # what run.sh asks


def test_server_settings_checked():
    for key, value, needle in (("pages", "yes", "`server.pages: yes`"), ("max_session_s", -5, "`server.max_session_s"),
                               ("max_session_s", True, "`server.max_session_s")):
        cfg = copy.deepcopy(BASE); cfg["server"][key] = value
        rejects(cfg, needle)


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
