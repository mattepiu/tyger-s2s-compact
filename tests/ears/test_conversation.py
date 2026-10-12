"""EARS's turn-taking state machine (app/ears/conversation.py), GPU-free: the Conversation on the `fake` EARS plugin
(energy endpointer: a barge after 4 voiced frames, an endpoint after 6 silent ones), a fake page and a fake MOUTH
over real TCP. Covers what test_floor.py and test_second_ear.py don't:
  * halting: from a barge until MOUTH's D, no backchannel and no floor grant, and the state is neither speaking nor
    listening to a reply;
  * an endpoint during a halt is held (`pending`) and sent as the next turn the moment the halt ends;
  * the reset after a barge: exactly one, before the hop after the BARGE one (the plugin seeds the new turn with the
    user's start there: tests/ears/test_rules.py), and none when MOUTH's barged D arrives, so the user's speech since
    the barge survives it (I-02);
  * MOUTH's D after the page has played the reply ends the turn at once; a D before it, followed by a barge in the
    tail, closes the turn on the spot (no B: MOUTH has nothing left to stop);
  * MOUTH's L (the links a reply cites) reaches the page after the reply's text.
Audio is synthetic (zeros / a tone).
Run: .venv/bin/python tests/ears/test_conversation.py
"""
import os, sys, asyncio, copy

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears import conversation as C  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402
from fakes import FakePage, MouthServer  # noqa: E402

ROOT = os.path.dirname(TESTS)
HOP = 5120


def silence(hops=1):
    return np.zeros(HOP * hops, np.float32)


def speech(hops=1):
    t = np.arange(HOP * hops) / 16000
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


async def settle(s=0.05):
    await asyncio.sleep(s)


async def start(report=True, silence_ms=480):
    """A conversation on the fake plugin; `calls` records the plugin's resets and hops in order."""
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
    cfg["mouth"]["tasks"] = {"announce_after_silence_ms": silence_ms}
    mouth = MouthServer(); cfg["server"]["mouth_port"] = await mouth.start()
    ears = EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake"))
    page = FakePage(report=report)
    session = ears.open()
    conv = C.Conversation(page, ears, session, cfg)
    calls = []
    orig_feed, orig_reset = ears.feed, session.reset

    async def feed(sess, uh, ah, mode, more=False):
        calls.append(mode)
        return await orig_feed(sess, uh, ah, mode, more)

    def reset():
        calls.append("reset"); orig_reset()
    ears.feed = feed; session.reset = reset
    task = asyncio.create_task(conv.run())
    await asyncio.wait_for(mouth.connected.wait(), 2)
    return page, mouth, conv, task, calls, ears


async def stop(page, task):
    page.q.put_nowait(None)
    await asyncio.wait_for(task, 2)


async def user_turn(page, mouth):
    """The user speaks and stops: the endpoint starts the agent's turn (T)."""
    page.push(speech(2)); page.push(silence(2)); await settle()
    assert mouth.kinds().count(b"T") == 1, mouth.kinds()


async def barge(page, mouth, conv, reply_s=3.0):
    """The agent talks (a long reply, still playing); the user talks over it: BARGE -> B to MOUTH, halting."""
    await mouth.send(b"A", np.zeros(int(24000 * reply_s), np.float32).tobytes()); await settle()
    page.push(speech(1)); await settle()
    assert b"B" in mouth.kinds() and "barge" in page.types(), mouth.kinds()
    assert conv.halting and not conv.speaking


def run(coro):
    asyncio.run(coro)


# ---- halting ------------------------------------------------------------------------------------------------
def test_halting_holds_backchannels_and_the_floor_until_d():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await barge(page, mouth, conv)
        orig = ears.feed

        async def feed(sess, uh, ah, mode, more=False):            # the plugin cues a backchannel every hop
            return await orig(sess, uh, ah, mode, more) + [{"kind": "BACKCHANNEL", "bc_prob": 0.9, "partial": "so"}]
        ears.feed = feed
        await mouth.send(b"F", {"task": 1}); await settle()
        page.push(silence(3)); await settle()                       # 12 quiet frames: enough for a grant
        assert conv.halting and b"C" not in mouth.kinds() and b"G" not in mouth.kinds()
        ears.feed = orig
        await mouth.send(b"D", {"text": "x", "barged": True}); await settle()
        assert not conv.halting and not conv.speaking
        assert {"type": "turn_done", "barged": True} in page.sent
        page.push(silence(2)); await settle()                       # quiet after the halt: now the grant
        assert b"G" in mouth.kinds()
        await stop(page, task)
    run(go())


def test_endpoint_during_a_halt_is_held_then_sent():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await barge(page, mouth, conv)
        page.push(speech(2)); page.push(silence(2)); await settle()  # the user finishes their cut-in: an endpoint
        assert "endpoint" in page.types()[page.types().index("barge"):]
        assert conv.pending is not None and mouth.kinds().count(b"T") == 1   # held: MOUTH hasn't confirmed the stop
        await mouth.send(b"D", {"text": "x", "barged": True}); await settle()
        assert mouth.kinds() == [b"T", b"B", b"T"], mouth.kinds()   # sent the moment the halt ended
        assert conv.pending is None and conv.speaking and not conv.halting
        await stop(page, task)
    run(go())


# ---- the reset after a barge --------------------------------------------------------------------------------
def test_one_reset_after_the_barge_hop_none_on_d():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await barge(page, mouth, conv)
        i = len(calls)
        page.push(speech(2)); await settle()                        # the user goes on talking
        await mouth.send(b"D", {"text": "x", "barged": True}); await settle()   # MOUTH confirms mid-utterance
        page.push(silence(2)); await settle()
        after = calls[i - 1:]
        assert after[:3] == ["speaking", "reset", "listen"], after  # the BARGE hop, one reset, then listening
        assert after.count("reset") == 1, after                     # not again at the barged D
        assert mouth.kinds().count(b"T") == 2                       # the speech before D still made a turn
        await stop(page, task)
    run(go())


def test_a_reply_played_out_resets_before_the_next_turn():
    """The contrast: a turn that ends normally resets the plugin's session (a fresh turn); a barged one doesn't
    (above), since the barge already did."""
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await mouth.send(b"A", np.zeros(2400, np.float32).tobytes())
        await mouth.send(b"D", {"text": "x", "barged": False}); await settle(0.3)
        assert page.types()[-1] == "turn_done"
        i = len(calls)
        page.push(silence(1)); await settle()
        assert calls[i:] == ["reset", "listen"], calls[i:]
        await stop(page, task)
    run(go())


# ---- MOUTH's D before / after the page has played the reply ------------------------------------------------
def test_d_after_the_page_played_everything_ends_the_turn_at_once():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await mouth.send(b"A", np.zeros(2400, np.float32).tobytes()); await settle(0.3)   # played and reported
        assert conv.playback.page_done() and conv.speaking         # MOUTH may still send more: not over yet
        assert "turn_done" not in page.types()
        await mouth.send(b"D", {"text": "x", "barged": False}); await settle(0.02)
        assert page.types()[-1] == "turn_done" and not conv.speaking and conv.done_task is None
        await stop(page, task)
    run(go())


def test_d_before_the_end_then_a_barge_in_the_tail_closes_the_turn():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        await mouth.send(b"A", np.zeros(24000 * 3, np.float32).tobytes())
        await mouth.send(b"D", {"text": "x", "barged": False}); await settle()
        assert conv.mouth_done and conv.speaking and conv.done_task is not None   # waiting for the page
        waiter = conv.done_task
        page.push(speech(1)); await settle()                        # the user talks over the tail
        assert {"type": "turn_done", "barged": True} in page.sent
        assert b"B" not in mouth.kinds()                             # MOUTH is done: nothing to stop, no D to wait for
        await settle(0.01)
        assert not conv.halting and not conv.speaking
        assert waiter.done() and conv.done_task is None             # the playback timeout is gone with the turn
        page.push(speech(2)); page.push(silence(2)); await settle()  # their turn goes straight out, not held
        assert mouth.kinds() == [b"T", b"T"] and conv.pending is None
        await stop(page, task)
    run(go())


# ---- the links a reply cites (I-25) ----------------------------------------------------------------------------
def test_links_reach_the_page():
    async def go():
        page, mouth, conv, task, calls, ears = await start()
        await user_turn(page, mouth)
        link = {"label": "science.nasa.gov", "url": "https://science.nasa.gov/mission/webb/"}
        await mouth.send(b"X", {"text": "Webb found a far-off burst."}); await mouth.send(b"L", {"links": [link]})
        await settle()
        assert {"type": "links", "links": [link]} in page.sent
        assert page.types().index("reply") < page.types().index("links"), page.types()
        await stop(page, task)
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
