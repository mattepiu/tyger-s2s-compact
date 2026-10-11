"""Plugin contract (PLUGINS.md -> Tests): one suite per kind, run against every plugin that can run here.

  * the `fake` plugins always run (no GPU);
  * the local model plugins (gemma-4-e4b, gemma-4-e4b-nvfp4, voxcpm2, nemotron-tyger-1.0) run with
    TYGER_CONTRACT_LOCAL=1, on the GPU box with the GPU free (stop run.sh first): they load the real models, one at
    a time.

Checks: events in a valid order; stop_flag ends a stream within one item; tool-call IDs unique; text never carries
tool-call or thinking markup; TTS audio is 1-D float32 at out_sr; EARS sessions open / close / reset and return
well-formed events. Prompts are synthetic. Run: [TYGER_CONTRACT_LOCAL=1] .venv/bin/python tests/plugins/test_contract.py
"""
import asyncio, gc, os, sys

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.config import load_config  # noqa: E402
from app.plugins import plugin_class  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.prompts import default_prompts  # noqa: E402
from app.tools import build_tools  # noqa: E402

LOCAL = os.environ.get("TYGER_CONTRACT_LOCAL") == "1"
CFG = load_config()
P = default_prompts()
TOOLS = build_tools(P)
TOOLS_EXPERT = build_tools(P, expert=True)
MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<think>", "</think>")


class Skip(Exception):
    pass


def ctx(name, eloop=None):
    return PluginContext(eloop, CFG, name)


_ELOOP = None


def eloop():
    global _ELOOP
    if _ELOOP is None:
        from app.runtime.event_loop import EngineLoop
        _ELOOP = EngineLoop()
    return _ELOOP


# ---- LLM ------------------------------------------------------------------------------------------------
def check_llm_events(evs):
    kinds = [e[0] for e in evs]
    assert set(kinds) <= {"text", "tool_start", "tool_calls", "refusal"}, kinds
    assert kinds.count("tool_start") <= 1 and kinds.count("tool_calls") <= 1 and kinds.count("refusal") <= 1, kinds
    if "tool_start" in kinds:
        assert "text" not in kinds[kinds.index("tool_start"):], "text after tool_start"
        assert isinstance(evs[kinds.index("tool_start")][1], dict) and "name" in evs[kinds.index("tool_start")][1]
    if "tool_calls" in kinds:
        assert kinds[-1] == "tool_calls" and "tool_start" in kinds, kinds
        ev = evs[-1]; assert len(ev) == 3, "tool_calls carries (calls, native)"
        named = evs[kinds.index("tool_start")][1]["name"]        # the filler is picked by it
        assert named is None or (ev[1] and named == ev[1][0]["name"]), (named, [c["name"] for c in ev[1]])
        ids = [c["id"] for c in ev[1]]
        assert len(ids) == len(set(ids)) and all(isinstance(c["arguments"], dict) and c["name"] for c in ev[1])
    if "refusal" in kinds:
        assert kinds[-1] == "refusal"
    text = "".join(e[1] for e in evs if e[0] == "text")
    assert not any(m in text for m in MARKUP), "markup in spoken text"
    return text


def llm_suite(llm, plain=True, tool_turn=True):
    msgs = [{"role": "system", "content": P.text("system", "voice_tools")},
            {"role": "user", "content": "Can you tell me a short joke?"}]     # nothing to look up: routing is
    # scripts/bench/escalation.py's job (a trivia question went to a lookup 8 / 8 on gemma-4-e4b-nvfp4, NOTES.md)
    text = check_llm_events(list(llm.stream(msgs, tools=TOOLS, max_new=64)))
    assert text.strip() or not plain, "no text for a plain question"
    n = 0                                                    # stop after the first item: at most one more
    for _ in llm.stream(msgs, tools=TOOLS, max_new=64, stop_flag=lambda: n >= 1):
        n += 1
    assert n <= 2, f"{n} items after stop"
    if tool_turn:
        ids = set()
        for city in ("Oslo", "Lima"):
            q = msgs[:1] + [{"role": "user", "content": f"What time is it in {city} right now?"}]
            evs = list(llm.stream(q, tools=TOOLS, max_new=96))
            check_llm_events(evs)
            calls = evs[-1][1] if evs and evs[-1][0] == "tool_calls" else []
            assert calls and calls[0]["name"] == "get_current_time", f"no time call for {city}"
            assert evs[[e[0] for e in evs].index("tool_start")][1]["name"] in (None, "get_current_time")
            ids |= {c["id"] for c in calls}
        assert len(ids) == 2, "tool-call IDs repeat across requests"
        q = [{"role": "system", "content": P.text("system", "voice_tools")},     # ask_expert offered: named
             {"role": "user", "content": "What time is it in Oslo right now?"}]
        evs = list(llm.stream(q, tools=TOOLS_EXPERT, max_new=96))
        check_llm_events(evs)
        assert ("tool_start", {"name": "get_current_time"}) in evs, "tool_start unnamed with ask_expert offered"


def test_llm_fake():
    from app.plugins.llm.fake import FakeLLM
    llm = FakeLLM()
    llm_suite(llm, tool_turn=False)
    tool = FakeLLM(reply='<tool_call>\n{"name": "get_current_time", "arguments": {"timezone": "Europe/Oslo"}}\n</tool_call>')
    llm_suite(tool, plain=False, tool_turn=True)


def test_llm_local():
    if not LOCAL:
        raise Skip("TYGER_CONTRACT_LOCAL=1 to run the Gemma plugins")
    base = dict(CFG["llm"]["reply"])
    # the backup's checkpoint is pre-quantized: no `quantization` at load (config/default.yaml's backup comment)
    for name, over in (("gemma-4-e4b", {}), ("gemma-4-e4b-nvfp4", {"quantization": None})):
        llm = plugin_class("llm", name)({**base, **over, "plugin": name}, ctx(name, eloop()))
        try:
            llm.warmup(); llm_suite(llm)
        finally:
            llm.shutdown(); del llm; gc.collect()


# ---- TTS ------------------------------------------------------------------------------------------------
def tts_suite(tts):
    names, default = tts.list_voices(CFG)             # a classmethod: needs no model (EARS calls it on the class)
    assert isinstance(names, list) and (default is None or default in names)
    voice = tts.load_voice(default) if default else None
    tts.warmup(voice)
    chunks = list(tts.stream("This is a short synthetic test sentence.", voice))
    assert chunks and all(isinstance(c, np.ndarray) and c.dtype == np.float32 and c.ndim == 1 for c in chunks)
    secs = sum(len(c) for c in chunks) / tts.out_sr
    assert 0.3 < secs < 15, secs
    n = 0
    for _ in tts.stream("Another sentence that is long enough to take a while to say out loud.", voice,
                        stop_flag=lambda: n >= 1):
        n += 1
    assert n <= 2, f"{n} chunks after stop"
    assert tts.load_voice("no-such-voice") is None


def test_tts_fake():
    from app.plugins.tts.fake import FakeTTS
    from app.plugins.tts import AtWireRate
    tts_suite(FakeTTS({"per_char_s": 0.06}))
    tts_suite(AtWireRate(FakeTTS({"out_sr": 22050, "per_char_s": 0.06})))


def test_tts_local():
    if not LOCAL:
        raise Skip("TYGER_CONTRACT_LOCAL=1 to run voxcpm2")
    tts = plugin_class("tts", "voxcpm2")(CFG["tts"], ctx("voxcpm2", eloop()))
    try:
        assert tts.caps.cloning and tts.out_sr == 24000
        tts_suite(tts)
    finally:
        tts.shutdown(); del tts; gc.collect()


# ---- EARS -----------------------------------------------------------------------------------------------
def ears_suite(ears, speech):
    """speech: 16 kHz float32 with silence after it. Fed hop by hop, it must yield an ENDPOINT."""
    th = ears.thresholds(s := ears.open())
    assert s is not None and ears.active() == 1
    names = [t.name for t in ears.THRESHOLDS]                 # the plugin's own thresholds, whatever they're called
    assert names and len(set(names)) == len(names) and set(th) == set(names), (names, sorted(th))
    for t in ears.THRESHOLDS:
        assert t.type in (int, float) and isinstance(th[t.name], t.type), t.name
        assert ears.set_threshold(s, t.name, str(th[t.name])) == th[t.name]     # as the page sends it: text
        if t.slider is not None:
            sl = t.slider
            assert sl.lo < sl.hi and sl.step > 0 and sl.meter in (None, "p_end", "yield", "bc"), t.name
            assert sl.off is None or sl.off > sl.hi / sl.scale, t.name               # "off" is outside the range
    for bad in (("nope", 1), (names[0], "x")):
        try:
            ears.set_threshold(s, *bad)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f"set_threshold accepted {bad}")
    hop = ears.hop_samples; zeros = np.zeros(hop, np.float32); kinds = []; ep = None

    async def run():
        nonlocal ep
        x = np.concatenate([speech, np.zeros(ears.sr * 3, np.float32)])
        for h in range(len(x) // hop):
            evs = await ears.feed(s, x[h * hop:(h + 1) * hop], zeros, "listen")
            assert isinstance(evs, list) and all(isinstance(e, dict) and "kind" in e for e in evs)
            kinds.extend(e["kind"] for e in evs)
            for e in evs:
                if e["kind"] == "ENDPOINT" and ep is None:
                    ep = e
    asyncio.run(run())
    assert set(kinds) <= {"FRAME", "ENDPOINT", "BARGE", "BACKCHANNEL"} and "FRAME" in kinds
    assert ep is not None and isinstance(ep.get("transcript"), str) and "turn_s" in ep and "by" in ep, ep
    s.reset()
    ears.close(s)


def test_ears_fake():
    from app.plugins.ears.fake import EnergyEars
    t = np.arange(16000 * 2) / 16000
    ears_suite(EnergyEars({"transcript": "hello there"}, ctx("fake")), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


def test_ears_local():
    if not LOCAL:
        raise Skip("TYGER_CONTRACT_LOCAL=1 to run nemotron-tyger-1.0")
    from datasets import load_dataset                       # public LibriSpeech test clip, in memory
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    a = ds[0]["audio"]; wav = np.asarray(a["array"], np.float32)
    assert a["sampling_rate"] == 16000
    ears = plugin_class("ears", "nemotron-tyger-1.0")(CFG["ears"], ctx("nemotron-tyger-1.0"))
    ears_suite(ears, wav)


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
