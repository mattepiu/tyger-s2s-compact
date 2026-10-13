"""config/prompts.yaml loader (no GPU). The step-1b byte-identity gate against the pre-move constants (commit
ddb1128) was deleted when the prompts were first edited on purpose (step 11: the tool rules). Run:
.venv/bin/python tests/test_prompts.py
"""
import copy, os, sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.prompts import Prompts, PromptError, default_prompts  # noqa: E402

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


def test_reply_shots_checked():
    d = copy.deepcopy(P.d); d["reply_shots"].append({"user": "x", "call": {"name": "fly", "arguments": {}},
                                                    "result": {}, "answer": "y"})
    expect_error(d, "reply_shots")


def test_persona_checked():
    d = copy.deepcopy(P.d); d["persona"]["characters"] = {}
    expect_error(d, "persona.characters")
    d = copy.deepcopy(P.d); del d["persona"]["characters"]["mira"]["name"]
    expect_error(d, "persona.characters.mira", "{name, text}")


def test_character_opens_every_reply_and_smart_prompt():
    """Every reply system prompt (each mode) and the smart model's open with the character's lines, then the
    prompt as it was; one character at a time. The four characters (2026-10-13): Mira first, the default; Ori, Mike
    and Andrew are gone."""
    from app.mouth.messages import reply_prefix, smart_messages
    assert P.characters() == ["mira", "aria", "kai", "rex"], P.characters()
    assert [P.character_name(c) for c in P.characters()] == ["Mira", "Aria", "Kai", "Rex"]
    assert not {"ori", "mike", "andrew"} & set(P.d["persona"]["characters"])
    for c in P.characters():
        text = P.d["persona"]["characters"][c]["text"]
        systems = {("system", "voice_tools"): reply_prefix(P, c, False)[0]["content"],
                   ("escalation", "system"): reply_prefix(P, c, True)[0]["content"],
                   ("tasks", "reply_system"): reply_prefix(P, c, True, True)[0]["content"],
                   ("smart", "system"): smart_messages(P, c, [], "what's new?", "What's new?")[0]["content"]}
        for (section, key), s in systems.items():
            assert s == text + "\n\n" + P.text(section, key), (c, section, key)
            others = [P.d["persona"]["characters"][o]["text"] for o in P.characters() if o != c]
            assert not any(o in s for o in others), (c, section, key)


def test_characters_by_voice():
    """voices.character is every voice's; voices.characters the exceptions; an unlisted voice (a clip added later
    too) gets the default; a character the prompts file lacks stops start-up with the key named."""
    from app.config import ConfigError
    from app.prompts import Characters
    ch = Characters({"voices": {"character": "kai", "characters": {"breezy": "aria"}}}, P)
    assert (ch.of("breezy"), ch.of("kai"), ch.of("added-later")) == ("aria", "kai", "kai")
    assert ch.label("breezy") == "Aria (breezy)" and ch.label("ref") == "Kai (ref)"
    assert Characters({"voices": {}}, P).default == "mira"                  # unset: the file's first character
    for voices, key in (({"character": "nobody"}, "`voices.character: nobody`"),
                        ({"characters": {"aria": "arya"}}, "`voices.characters.aria: arya`")):
        try:
            Characters({"voices": voices}, P)
        except ConfigError as e:
            assert key in str(e) and "mira" in str(e), str(e)
        else:
            raise AssertionError(f"accepted {voices}")


def test_shipped_voices():
    """config/default.yaml: Mira's voice is the default and Mira every voice's character, and aria, kai and rex each
    speak as their own. A voice it doesn't list (an old clip left in voices/, e.g. ori) speaks as Mira."""
    from app.config import ROOT, load_config
    from app.prompts import Characters
    cfg = load_config(str(ROOT / "config/default.yaml"))
    assert cfg["voices"]["default"] == "mira" and cfg["voices"]["character"] == "mira"
    ch = Characters(cfg, P)
    assert {v: ch.of(v) for v in ("mira", "aria", "kai", "rex", "ori")} == \
        {"mira": "mira", "aria": "aria", "kai": "kai", "rex": "rex", "ori": "mira"}
    assert [ch.label(v) for v in ("mira", "aria", "kai", "rex", "ori")] == \
        ["Mira (mira)", "Aria (aria)", "Kai (kai)", "Rex (rex)", "Mira (ori)"]


def test_missing_character_stops_mouth_start():
    """MOUTH checks the mapping right after the prompts, before it loads any engine."""
    from app.config import ConfigError, ROOT, load_config
    from app.mouth.service import Mouth
    cfg = load_config(str(ROOT / "config/fake.yaml"))
    cfg["voices"]["characters"] = {"fake": "nobody"}
    try:
        Mouth(cfg)
    except ConfigError as e:
        assert "`voices.characters.fake: nobody`" in str(e), str(e)
    else:
        raise AssertionError("MOUTH started with a voice mapped to a character that doesn't exist")


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
