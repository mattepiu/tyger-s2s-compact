"""config/prompts.yaml loader (no GPU). The step-1b byte-identity gate against the pre-move constants (commit
ddb1128) was deleted when the prompts were first edited on purpose (step 11: reply language, tool rules). Run:
.venv/bin/python tests/test_prompts.py
"""
import copy, os, sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.prompts import Prompts, PromptError, default_prompts, deep_merge, load_prompts  # noqa: E402

P = default_prompts()


def test_missing_key_named():
    d = copy.deepcopy(P.d); del d["backchannel"]["user_empty"]
    expect_error(d, "backchannel.user_empty")


def test_template_placeholders_checked():
    d = copy.deepcopy(P.d); d["tool_leadin"]["user"] = 'The user asked: "$question"'
    expect_error(d, "tool_leadin.user", "$transcript")
    d = copy.deepcopy(P.d); d["backchannel"]["no_repeat"] = " You just said $acks for $5."
    expect_error(d, "backchannel.no_repeat")
    d = copy.deepcopy(P.d); d["backchannel"]["no_repeat"] = " You just said $acks (costs $$5)."
    assert Prompts(d).fill("backchannel", "no_repeat", acks="'x'") == " You just said 'x' (costs $5)."


def test_fill_needs_exact_values():
    try:
        P.fill("still_looking", "user", transcript="x")
    except PromptError:
        return
    raise AssertionError("missing value accepted")


def test_shots_and_tools_checked():
    d = copy.deepcopy(P.d); d["backchannel"]["shots"] = [["only one"]]
    expect_error(d, "backchannel.shots")
    d = copy.deepcopy(P.d); d["tools"]["get_current_time"]["params"]["city"] = "x"
    expect_error(d, "tools.get_current_time.params.city")
    d = copy.deepcopy(P.d); d["tools"]["fly"] = {"description": "x"}
    expect_error(d, "tools.fly")


def test_override_merges_and_replaces_lists():
    over = {"backchannel": {"shots": [["a", "b"]], "user_empty": "Nothing yet."}}
    m = Prompts(deep_merge(P.d, over))
    assert m.shots("backchannel") == [("a", "b")] and m.text("backchannel", "user_empty") == "Nothing yet."
    assert m.text("backchannel", "system") == P.text("backchannel", "system")
    assert P.text("backchannel", "user_empty") != "Nothing yet."           # the base is untouched


def test_load_prompts_per_instance(tmp=None):
    import tempfile, yaml
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "small.yaml"), "w") as f:
            yaml.safe_dump({"warmup": {"user": "Hello."}}, f)
        got = load_prompts({"mouth": {"prompt_overrides_dir": d}}, ["gemma-4-e4b", "small"])
    assert got["gemma-4-e4b"].text("warmup", "user") == P.text("warmup", "user")
    assert got["small"].text("warmup", "user") == "Hello."


def test_language_named_in_notes():
    from app.prompts.lang import guess_language
    assert [guess_language(t) for t in ("Какая погода в Казани?", "¿Va a llover mañana?", "Que hora es en Madrid",
                                        "What time is it in Tokyo?", "Wie spät ist es?", "Hmm")] == \
        ["ru", "es", "es", "en", None, None]
    assert "Russian" in P.result_note("Какая погода в Казани?") and P.filler_hint("Какая погода в Казани?")
    assert P.result_note("Hmm") == P.text("language", "tool_note_any") and P.filler_hint("Hmm") == ""
    from app.prompts.lang import language_text
    assert language_text("Actually, never mind, drop that one.", ["Сравни тарифы операторов, пожалуйста"]) \
        .endswith("drop that one.") and guess_language(language_text("Ok, drop it.", ["Сравни тарифы, пожалуйста"])) == "ru"
    assert language_text("What time is it in Tokyo?", ["Привет"]) == "What time is it in Tokyo?"


def test_reply_shots_checked():
    d = copy.deepcopy(P.d); d["reply_shots"].append({"user": "x", "call": {"name": "fly", "arguments": {}},
                                                    "result": {}, "answer": "y"})
    expect_error(d, "reply_shots")


def expect_error(d, *needles):
    try:
        Prompts(d)
    except PromptError as e:
        assert all(n in str(e) for n in needles), str(e)
        return
    raise AssertionError("accepted")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
