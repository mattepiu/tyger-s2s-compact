"""A conversation's limits (app/ears/conversation.py), GPU-free, on the `fake` EARS plugin with a fake page and a fake
MOUTH over real TCP:
  * the session limit (server.max_session_s): `ready` carries it; at the limit a reply in progress finishes, then
    `session_end` and the close, and MOUTH's connection closes too (MOUTH then cancels the conversation's background
    tasks); nothing new starts after the limit (a cut-in's turn isn't sent); a reply that doesn't end is cut after
    REPLY_GRACE_S; an idle conversation ends right at the limit; 0 means no limit;
  * the mic buffer cap: audio sent faster than real time, piling up past MAX_MIC_BUFFER_S, closes the connection.
Audio is synthetic (zeros / a tone).
Run: .venv/bin/python tests/ears/test_limits.py"""
import os, sys, asyncio, copy, time

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


def speech(hops=1):
    t = np.arange(HOP * hops) / 16000
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(hops=1):
    return np.zeros(HOP * hops, np.float32)


async def start(limit, report=True):
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
    cfg["server"]["max_session_s"] = limit
    mouth = MouthServer(); cfg["server"]["mouth_port"] = await mouth.start()
    ears = EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake"))
    page = FakePage(report=report)
    conv = C.Conversation(page, ears, ears.open(), cfg)
    task = asyncio.create_task(conv.run())
    await asyncio.wait_for(mouth.connected.wait(), 2)
    return page, mouth, conv, task


def when(page, kind):
    """The index of the first message of `kind` the page got, or None."""
    return next((i for i, t in enumerate(page.types()) if t == kind), None)


async def ended(page, task, within):
    """The page got session_end and its socket closed, and the conversation is over."""
    await asyncio.wait_for(task, within)
    assert page.closed and page.types()[-1] == "session_end", page.types()
    assert [x for x in page.sent if isinstance(x, dict) and x["type"] == "session_end"][-1]["reason"] == "time_limit"


def test_ready_carries_the_limit():
    async def go():
        for limit in (2, 0):
            page, mouth, conv, task = await start(limit)
            await asyncio.sleep(0.05)
            assert page.sent[0]["type"] == "ready" and page.sent[0]["max_session_s"] == limit, page.sent[0]
            page.q.put_nowait(None); await asyncio.wait_for(task, 2)
    asyncio.run(go())


def test_reply_in_progress_finishes_then_the_session_ends():
    """A 2 s limit; the agent's reply runs past it: it plays to its end (turn_done), then session_end and the close,
    and MOUTH's connection is closed."""
    async def go():
        page, mouth, conv, task = await start(2.0)
        t0 = time.perf_counter()
        page.push(speech(2)); page.push(silence(2)); await asyncio.sleep(0.1)
        assert mouth.kinds().count(b"T") == 1
        await mouth.send(b"A", np.zeros(int(24000 * 2.6), np.float32).tobytes())   # plays until ~2.7 s
        await mouth.send(b"D", {})
        await ended(page, task, 5)
        took = time.perf_counter() - t0
        assert when(page, "turn_done") < when(page, "session_end"), page.types()
        assert 2.5 < took < 3.5, took                                # the reply played out, then it ended
        assert conv.writer.is_closing()                              # MOUTH sees the end: it cancels its tasks
    asyncio.run(go())


def test_nothing_new_starts_after_the_limit():
    """A 1 s limit passes while the agent speaks; the user cuts in (barge) and finishes their sentence: their turn
    isn't sent, and the session ends once MOUTH has halted."""
    async def go():
        page, mouth, conv, task = await start(1.0)
        page.push(speech(2)); page.push(silence(2)); await asyncio.sleep(0.1)
        await mouth.send(b"A", np.zeros(int(24000 * 4), np.float32).tobytes())
        await asyncio.sleep(1.2)
        assert conv.ending and conv.speaking
        page.push(speech(1)); await asyncio.sleep(0.1)                # the cut-in: B to MOUTH, halting
        assert b"B" in mouth.kinds() and conv.halting
        page.push(speech(2)); page.push(silence(3)); await asyncio.sleep(0.2)   # ...and would end a turn
        await mouth.send(b"D", {"barged": True})
        await ended(page, task, 3)
        assert mouth.kinds().count(b"T") == 1, mouth.kinds()         # only the turn from before the limit
    asyncio.run(go())


def test_a_reply_that_doesnt_end_is_cut():
    async def go():
        grace, C.REPLY_GRACE_S = C.REPLY_GRACE_S, 0.5
        try:
            page, mouth, conv, task = await start(1.0, report=False)  # never reports it played; MOUTH sends no D
            t0 = time.perf_counter()
            page.push(speech(2)); page.push(silence(2)); await asyncio.sleep(0.1)
            await mouth.send(b"A", np.zeros(int(24000 * 10), np.float32).tobytes())
            await ended(page, task, 3)
            took = time.perf_counter() - t0
            assert 1.4 < took < 2.0 and "turn_done" not in page.types(), (took, page.types())
        finally:
            C.REPLY_GRACE_S = grace
    asyncio.run(go())


def test_idle_conversation_ends_at_the_limit_and_zero_means_none():
    async def go():
        page, mouth, conv, task = await start(0.5)
        t0 = time.perf_counter()
        await ended(page, task, 2)
        assert 0.45 < time.perf_counter() - t0 < 0.8
        page, mouth, conv, task = await start(0)
        await asyncio.sleep(0.8)
        assert not task.done() and "session_end" not in page.types()
        page.q.put_nowait(None); await asyncio.wait_for(task, 2)
    asyncio.run(go())


def test_audio_faster_than_real_time_closes():
    """4 s of mic audio at once is buffered; 6 s at once (300 blocks of 20 ms, faster than real time) passes the
    5 s cap: the page is told why and the socket closes."""
    async def go():
        page, mouth, conv, task = await start(0)
        for _ in range(200):
            page.push(np.zeros(320, np.float32))
        await asyncio.sleep(0.3)
        assert not page.closed and not task.done()
        for _ in range(300):
            page.push(np.zeros(320, np.float32))
        await asyncio.wait_for(task, 2)
        assert page.closed and {"type": "error", "error": "audio_too_fast"} in page.sent, page.types()
    asyncio.run(go())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
