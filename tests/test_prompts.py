"""prompts.yaml loader (no GPU), and the step-1b gate: the move out of code was byte-identical.

The gate (test_move_*) compares prompts.yaml with the constants and f-strings of the code BEFORE the move (git
commit PRE_MOVE, read with `git show`; skipped without git). Once prompts.yaml is edited on purpose, those two
tests have done their job: delete them. Run: .venv/bin/python tests/test_prompts.py
"""
import ast, copy, json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.prompts import Prompts, PromptError, default_prompts, deep_merge, load_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402
from app.config import ROOT  # noqa: E402

PRE_MOVE = "ddb1128"
P = default_prompts()


class Skip(Exception):
    pass


def old_constants():
    """Module-level constants of chunker.py, mouth.py and tools.py at PRE_MOVE, evaluated from their source."""
    env = {}
    for f in ("app/chunker.py", "app/mouth.py", "app/tools.py"):
        try:
            src = subprocess.run(["git", "show", f"{PRE_MOVE}:{f}"], cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            raise Skip(f"git show {PRE_MOVE} unavailable")
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("TTS_SYS", "TTS_TOOLS_SYS", "INTERRUPTED", "LEADIN_SYS",
                                               "LEADIN_SHOTS", "STILL_SYS", "STILL_SHOTS", "BC_PROMPT",
                                               "BC_SHOTS", "TOOLS"):
                env[node.targets[0].id] = eval(compile(ast.Expression(node.value), f, "eval"),
                                               {"__builtins__": {}}, env)
    return env


# the f-strings MOUTH built before the move, verbatim (mouth.py at PRE_MOVE)
def old_leadin_user(transcript):
    return f"The user asked: \"{transcript}\""


def old_still_user(transcript, fillers):
    said = " ".join(f"'{f}'" for f in fillers) or "nothing yet"
    return f"The user asked: \"{transcript}\". You already said: {said}"


def old_bc(prompt, acks, user, bc_min_words=4):
    sys_prompt = prompt
    if acks:
        sys_prompt += " You just said " + ", ".join(f"'{a}'" for a in acks) + " -- don't repeat those."
    if not user:
        said = "The user is speaking, but no words have been recognized yet."
    elif len(user.split()) < bc_min_words:
        said = f"The user has only said a few words so far: \"{user}\""
    else:
        said = f"The user is currently saying: \"{user}\""
    return sys_prompt, said


def test_move_is_byte_identical():
    old = old_constants()
    assert P.text("system", "voice_tools") == old["TTS_TOOLS_SYS"]
    assert P.text("tool_leadin", "system") == old["LEADIN_SYS"] and P.shots("tool_leadin") == old["LEADIN_SHOTS"]
    assert P.text("still_looking", "system") == old["STILL_SYS"] and P.shots("still_looking") == old["STILL_SHOTS"]
    assert P.text("backchannel", "system") == old["BC_PROMPT"] and P.shots("backchannel") == old["BC_SHOTS"]
    assert P.text("markers", "interrupted") == old["INTERRUPTED"]
    import app.mouth as M                                # the new code paths, with sample inputs
    for tr in ("what's the weather in Oslo?", "какая погода в Казани", 'he said "hi" for $5 {x}'):
        assert P.fill("tool_leadin", "user", transcript=tr) == old_leadin_user(tr)
        for fillers in ([], ["Checking Oslo."], ["One.", "Two's."]):
            assert M.still_user(P, tr, fillers) == old_still_user(tr, fillers)
    for acks in ([], ["oh nice"], ["wow", "it's $5 {x}"]):
        for user in ("", "so I", "I finally finished my first marathon", "ну вот"):
            assert M.bc_messages(P, list(acks), user, 4)[0]["content"] == old_bc(old["BC_PROMPT"], acks, user)[0]
            assert M.bc_messages(P, list(acks), user, 4)[-1]["content"] == old_bc(old["BC_PROMPT"], acks, user)[1]


def test_move_tools_identical():
    old = old_constants()["TOOLS"]
    assert json.dumps(build_tools(P)) == json.dumps(old)          # same JSON, key order included


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
    d = copy.deepcopy(P.d); d["tools"]["get_weather"]["params"]["city"] = "x"
    expect_error(d, "tools.get_weather.params.city")
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
        got = load_prompts({"mouth": {"prompt_overrides_dir": d}}, ["qwen3.5-4b", "small"])
    assert got["qwen3.5-4b"].text("warmup", "user") == P.text("warmup", "user")
    assert got["small"].text("warmup", "user") == "Hello."


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
            except Skip as e:
                print("skip", name, e)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
