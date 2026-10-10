"""EARS floor requests (PLUGINS.md step 13), GPU-free: the Conversation state machine (app/ears/conversation.py)
on the `fake` EARS plugin (energy endpointer), a fake browser WebSocket and a fake MOUTH over real TCP. Covers:
F -> silence -> G, a user turn winning over a waiting request, silence counted only while both sides are quiet, a
backchannel holding the grant, barge-in during an announcement, a lost MOUTH. Audio is synthetic (zeros / a tone).
Run: .venv/bin/python tests/ears/test_floor.py
"""
import os, sys, asyncio, copy

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from app.ears.conversation import Conversation  # noqa: E402
from app.config import load_config  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from fakes import FakePage, MouthServer  # noqa: E402

SIL_MS = 480                                   # announce_after_silence_ms here: 6 frames
HOP = 5120
ROOT = os.path.dirname(TESTS)
EARS = None                                    # the fake EARS plugin of the running test


def silence(hops=1):
    return np.zeros(HOP * hops, np.float32)


def speech(hops=1):
    t = np.arange(HOP * hops) / 16000
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


async def settle(s=0.05):
    await asyncio.sleep(s)


async def setup():
    global EARS
    cfg = load_config(os.path.join(ROOT, "config/fake.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["mouth"]["tasks"] = {"announce_after_silence_ms": SIL_MS}
    mouth = MouthServer(); port = await mouth.start()
    cfg["server"]["mouth_port"] = port
    EARS = EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake"))
    ws = FakePage()
    conv = asyncio.create_task(Conversation(ws, EARS, EARS.open(), cfg).run())
    await asyncio.wait_for(mouth.connected.wait(), 2)
    return ws, mouth, conv


async def teardown(ws, conv):
    ws.q.put_nowait(None)
    await asyncio.wait_for(conv, 2)


async def reply(mouth, seconds=0.1, **d):
    """MOUTH sends a short reply (audio, then D) for the turn EARS just started."""
    await mouth.send(b"A", np.zeros(int(24000 * seconds), np.float32).tobytes())
    await mouth.send(b"D", {"text": "x", "barged": False, **d})


def run(coro):
    asyncio.run(coro)


# ---- tests -----------------------------------------------------------------------------------------------
def test_grant_after_silence():
    async def go():
        ws, mouth, conv = await setup()
        await mouth.send(b"F", {"task": 1}); await settle()
        ws.push(silence(1)); await settle()                      # 4 idle frames < 6: not yet
        assert b"G" not in mouth.kinds()
        ws.push(silence(1)); await settle()                      # 8 frames: granted
        assert (b"G", {"task": 1}) in mouth.got and "task_done" in ws.types()
        await reply(mouth, proactive=1); await settle(0.5)       # announcement plays out -> listening again
        assert ws.types()[-1] == "turn_done"
        ws.push(silence(3)); await settle()
        assert mouth.kinds().count(b"G") == 1                    # one grant per request
        await teardown(ws, conv)
    run(go())


def test_user_turn_wins_and_silence_restarts_after_reply():
    async def go():
        ws, mouth, conv = await setup()
        await mouth.send(b"F", {"task": 2}); await settle()
        ws.push(speech(3)); await settle()                       # the user is talking
        assert b"G" not in mouth.kinds()
        ws.push(silence(2)); await settle()                      # their endpoint: a reply turn, not the grant
        assert b"T" in mouth.kinds() and b"G" not in mouth.kinds()
        await reply(mouth, seconds=0.2); await settle(0.6)       # the reply plays out
        assert "turn_done" in ws.types() and b"G" not in mouth.kinds()   # silence during it didn't count
        ws.push(silence(2)); await settle()
        assert mouth.kinds().index(b"T") < mouth.kinds().index(b"G")
        await teardown(ws, conv)
    run(go())


def test_silence_while_the_agent_speaks_does_not_count():
    """A plugin that reports `idle` frames while the agent talks: they still don't count as silence."""
    async def go():
        ws, mouth, conv = await setup()
        orig = EARS.feed

        async def feed(sess, uh, ah, mode, more=False):
            evs = await orig(sess, uh, ah, mode, more)
            return [{**e, "state": "idle"} if e["kind"] == "FRAME" and mode == "speaking" else e for e in evs]
        EARS.feed = feed
        ws.push(speech(3)); ws.push(silence(2)); await settle()           # a user turn -> a long reply
        assert b"T" in mouth.kinds()
        await mouth.send(b"A", np.zeros(int(24000 * 1.0), np.float32).tobytes())
        await mouth.send(b"F", {"task": 7}); await settle()
        ws.push(silence(3)); await settle()                               # 12 idle frames, all while speaking
        await mouth.send(b"D", {"text": "x", "barged": False}); await settle(1.4)
        assert "turn_done" in ws.types()
        ws.push(silence(1)); await settle()                               # 4 quiet frames since: not yet
        assert b"G" not in mouth.kinds()
        ws.push(silence(1)); await settle()
        assert b"G" in mouth.kinds()
        await teardown(ws, conv)
    run(go())


def test_backchannel_holds_the_grant():
    async def go():
        ws, mouth, conv = await setup()
        orig = EARS.feed
        fired = []

        async def feed(sess, uh, ah, mode, more=False):         # one backchannel cue on the first hop
            evs = await orig(sess, uh, ah, mode, more)
            if not fired:
                fired.append(1); evs = evs + [{"kind": "BACKCHANNEL", "bc_prob": 0.9, "partial": "so"}]
            return evs
        EARS.feed = feed
        await mouth.send(b"F", {"task": 3}); await settle()
        ws.push(silence(3)); await settle()
        assert b"C" in mouth.kinds() and b"G" not in mouth.kinds()   # the ack is still out
        await mouth.send(b"D", {"text": "mm-hm", "bc": True}); await settle()
        ws.push(silence(1)); await settle()
        assert b"G" in mouth.kinds()
        await teardown(ws, conv)
    run(go())


def test_barge_during_announcement():
    async def go():
        ws, mouth, conv = await setup()
        await mouth.send(b"F", {"task": 4}); await settle()
        ws.push(silence(2)); await settle()
        assert b"G" in mouth.kinds()
        await mouth.send(b"A", np.zeros(24000 * 3, np.float32).tobytes()); await settle()   # 3 s announcement
        ws.push(speech(2)); await settle()                        # the user talks over it
        assert b"B" in mouth.kinds() and "barge" in ws.types()
        await mouth.send(b"D", {"text": "x", "barged": True, "proactive": 4}); await settle()
        assert {"type": "turn_done", "barged": True} in ws.sent
        await teardown(ws, conv)
    run(go())


def test_two_requests_take_turns():
    async def go():
        ws, mouth, conv = await setup()
        await mouth.send(b"F", {"task": 5}); await mouth.send(b"F", {"task": 6}); await settle()
        ws.push(silence(2)); await settle()
        assert [d for k, d in mouth.got if k == b"G"] == [{"task": 5}]
        await reply(mouth, proactive=5); await settle(0.5)
        ws.push(silence(2)); await settle()
        assert [d for k, d in mouth.got if k == b"G"] == [{"task": 5}, {"task": 6}]
        await teardown(ws, conv)
    run(go())


def test_lost_mouth_ends_the_conversation():
    """MOUTH restarts mid-conversation: the page is told and the conversation ends, instead of a dead hop loop
    with the mic audio piling up."""
    async def go():
        ws, mouth, conv = await setup()
        ws.push(speech(1)); await settle()
        await mouth.drop()
        await asyncio.wait_for(conv, 2)                           # ended on its own: no teardown needed
        assert {"type": "error", "error": "mouth_lost"} in ws.sent and ws.closed
    run(go())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
