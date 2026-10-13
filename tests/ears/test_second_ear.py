"""I-17, GPU-free: the 2nd ear on the page's clock.

AgentPlayback builds the agent channel from the mic blocks' playback tags (gaps, a reply-id change, past what has
arrived), says what the user had heard by a mic position, and takes the page's `played` report. The Conversation, on
the fake EARS plugin with a fake page and a fake MOUTH over TCP: cuts tagged blocks that straddle hops into hops with
the 2nd ear lined up; tells MOUTH on a barge what the tags say the user had heard when they started talking; ends the
turn when the page reports it has played everything, even with a downlink slower than the old 0.25 s margin (a
timeout only as the safety net); keeps backchannel audio untagged. Audio is synthetic.
Run: .venv/bin/python tests/ears/test_second_ear.py
"""
import os, sys, asyncio, copy

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
import numpy as np  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears import conversation as C  # noqa: E402
from app.ears.playback import AgentPlayback, BACKCHANNEL_ID, EchoMeter, audio_frame, mic_block  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402
from fakes import FakePage, MouthServer  # noqa: E402

ROOT = os.path.dirname(TESTS)
HOP = 5120


def ramp(seconds):
    """Reply audio whose every sample differs, so a misplaced slice shows (24 kHz)."""
    return np.linspace(-0.9, 0.9, int(24000 * seconds)).astype(np.float32)


def with_reply(seconds=1.0):
    pb = AgentPlayback(16000); rid = pb.start(); pb.add(ramp(seconds), 0)
    return pb, rid, pb.audio[:pb.n16].copy()


# ---- AgentPlayback ----------------------------------------------------------------------------------------
def test_tags_build_the_ear():
    pb, rid, ear = with_reply()
    assert np.array_equal(pb.agent_block(rid, 2400, 320, 1000), ear[1600:1920])      # 0.1 s in: sample 1600
    assert not pb.agent_block(rid, -1, 320, 1320).any()                               # nothing played: silence
    tail = pb.agent_block(rid, 24000 - 150, 320, 1640)                               # runs past what arrived
    k = pb.n16 - 15900
    assert np.array_equal(tail[:k], ear[15900:]) and not tail[k:].any()
    assert not pb.agent_block(rid - 1, 2400, 320, 1960).any()                         # an earlier reply's tag
    rid2 = pb.start(); pb.add(ramp(0.5)[::-1].copy(), 0)                              # a new reply: new id
    assert rid2 == rid + 1 and not pb.agent_block(rid, 2400, 320, 2280).any()         # the old id: silence now
    assert pb.agent_block(rid2, 2400, 320, 2600)[0] == pb.audio[1600]


def test_heard_by_a_mic_position():
    pb, rid, _ = with_reply()
    pb.agent_block(rid, -1, 320, 0)                       # nothing playing yet
    pb.agent_block(rid, 0, 320, 320)                      # playback starts at mic sample 320
    pb.agent_block(rid, 480, 320, 640)
    pb.agent_block(rid - 1, 9000, 320, 960)               # a stray tag of another reply: ignored
    assert pb.heard_s(100) == 0.0 and pb.heard_s(320) == 0.0
    assert pb.heard_s(480) == 160 * 1.5 / 24000           # 160 mic samples into playback
    assert pb.heard_s(5000) == (480 + 320 * 1.5) / 24000  # as far as the last tag reaches
    pb.agent_block(rid, 23900, 320, 1280)                 # near the end: capped at what was sent
    assert pb.heard_s(9999) == 1.0


def test_a_tag_past_what_was_sent_is_nothing_playing():
    """A page that kept an old connection's counts tags this connection's reply 1 from where the old reply 1 ended:
    past what was sent, so none of it is playing (found live: Stop, then Start without a reload)."""
    pb, rid, _ = with_reply(0.5)                          # 12000 samples sent
    assert not pb.agent_block(rid, 12000, 320, 0).any()   # the first sample past what was sent
    assert not pb.agent_block(rid, 30000, 320, 320).any()
    assert pb.heard_s(10 ** 6) == 0.0 and pb.offset_ms() is None


def test_page_report_and_offset():
    pb = AgentPlayback(16000); rid = pb.start()
    assert pb.page_done()                                 # nothing sent: nothing to wait for
    pb.add(ramp(0.5), 5120)                               # arrived as the hop at mic sample 5120 was next
    assert not pb.page_done() and pb.offset_ms() is None
    pb.page_played(rid - 1, 10 ** 6)                      # a report for an earlier reply: ignored
    assert not pb.page_done()
    pb.page_played(rid, 12000)
    assert pb.page_done()
    pb.agent_block(rid, 1500, 320, 5120 + 5600 + 1000)    # the page played sample 0 at mic sample 10720
    assert pb.offset_ms() == 350.0


def test_wire_frames():
    rid, at24, pcm = mic_block(b"\x07\x00\x00\x00\xff\xff\xff\xff" + np.array([16384, -32768], "<i2").tobytes())
    assert (rid, at24) == (7, -1) and np.allclose(pcm, [0.5, -1.0])
    assert mic_block(b"\x00" * 7) is None and mic_block(b"\x00" * 9) is None
    f = audio_frame(3, np.array([0.5, -2.0], np.float32))
    assert f[:4] == b"\x03\x00\x00\x00" and list(np.frombuffer(f, "<i2", offset=4)) == [16384, -32768]


def test_echo_meter():
    """I-26: the agent's voice 35 dB down in the mic over a -90 dB floor reads as -35 dB echo on its voiced frames, the
    agent's silent stretch as the floor; at a barge, the user's speech since the onset (and a margin) is dropped."""
    rng = np.random.default_rng(0)
    ear = (0.1 * rng.standard_normal(HOP * 4)).astype(np.float32)          # -20 dB
    ear[HOP * 3:] = 0                                                       # the last hop: the agent silent
    mic = ear * 10 ** (-35 / 20) + 10 ** (-90 / 20) * rng.standard_normal(len(ear)).astype(np.float32)
    meter = EchoMeter(16000)
    assert meter.summary() is None
    for i in range(4):
        meter.add(mic[i * HOP:(i + 1) * HOP], ear[i * HOP:(i + 1) * HOP])
    s = meter.summary()
    assert s.startswith("echo -35.0 dB") and "over 12 frames" in s, s
    assert abs(float(s.split("mic floor ")[1].split()[0]) + 90) < 0.5, s      # 4 frames of noise: within 0.5 dB
    meter.add(0.3 * np.ones(HOP, np.float32), ear[:HOP])                    # the user cuts in (+ the agent)
    meter.drop_last(0.3)                                                    # onset 0.3 s ago: 0.3 + 0.3 s dropped
    assert len(meter.mic) == 20 - 8 and meter.summary().endswith("over 12 frames; mic floor n/a")
    meter.start()
    assert meter.summary() is None


# ---- the Conversation ---------------------------------------------------------------------------------------
async def start(report=True, downlink_s=0.0):
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
    mouth = MouthServer(); cfg["server"]["mouth_port"] = await mouth.start()
    ears = EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake"))
    page = FakePage(report=report, downlink_s=downlink_s)
    conv = C.Conversation(page, ears, ears.open(), cfg)
    task = asyncio.create_task(conv.run())
    await asyncio.wait_for(mouth.connected.wait(), 2)
    hops = []
    orig = ears.feed

    async def feed(sess, uh, ah, mode, more=False):        # the hops the plugin gets
        hops.append((mode, ah.copy()))
        return await orig(sess, uh, ah, mode, more)
    ears.feed = feed
    return page, mouth, conv, task, hops


async def user_turn(page, mouth):
    """The user speaks and stops: the endpoint starts the agent's turn (T), 5 hops in."""
    t = np.arange(HOP * 3) / 16000
    page.push((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)); page.push(np.zeros(HOP * 2, np.float32))
    await asyncio.sleep(0.05)
    assert mouth.kinds().count(b"T") >= 1


async def stop(page, task):
    page.q.put_nowait(None)
    await asyncio.wait_for(task, 2)


def test_blocks_straddling_hops_line_up():
    """1000-sample blocks (not a divisor of the hop) tagged with continuous playback from mic sample P0, with a
    gap: every speaking hop's 2nd ear is the reply's audio shifted to P0, silent in the gap, resumed after it."""
    async def go():
        page, mouth, conv, task, hops = await start()
        await user_turn(page, mouth)
        await mouth.send(b"A", ramp(3.0).tobytes()); await asyncio.sleep(0.05)
        ear = conv.playback.audio[:conv.playback.n16]; rid = conv.playback.rid
        pos0 = 5 * HOP; p0 = pos0 + 3000; gap = (p0 + 4000, p0 + 6000)
        expect = np.zeros(pos0 + 26 * 1000, np.float32)                 # 26 blocks: 5 hops and a bit
        for pos in range(pos0, pos0 + 26 * 1000, 1000):
            if pos < p0 or gap[0] <= pos < gap[1]:
                tag = -1
            else:
                k = pos - p0 - (gap[1] - gap[0] if pos >= gap[1] else 0)
                tag = k * 3 // 2; expect[pos:pos + 1000] = ear[k:k + 1000]
            page.push(np.zeros(1000, np.float32), tag=(rid, tag))
        await asyncio.sleep(0.1)
        speaking = [(i, ah) for i, (mode, ah) in enumerate(hops) if mode == "speaking"]
        assert len(speaking) == 5 and speaking[0][0] == 5, [m for m, _ in hops]
        for i, ah in speaking:
            assert np.array_equal(ah, expect[i * HOP:(i + 1) * HOP]), i
        assert expect[gap[0]:gap[1]].sum() == 0 and expect[gap[1]:].any()
        await stop(page, task)
    asyncio.run(go())


def test_barge_reports_what_the_tags_say_was_heard():
    """The user cuts in: B carries the reply sample the user was hearing when they started (the plugin's onset,
    counted back from the hop's end), from the tags: 0.5 s here, whenever the barge is detected."""
    async def go():
        page, mouth, conv, task, hops = await start()
        await user_turn(page, mouth)
        await mouth.send(b"A", ramp(3.0).tobytes()); await asyncio.sleep(0.05)
        rid = conv.playback.rid
        page.push(np.zeros(HOP, np.float32), tag=(rid, 0))                     # the reply plays from sample 0
        t = np.arange(HOP) / 16000                                              # the user talks over it: the
        page.push((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),     # fake barges after 4 voiced
                  tag=(rid, 12000))                                             # frames, its onset 0.32 s back
        await asyncio.sleep(0.1)
        b = [d for k, d in mouth.got if k == b"B"]
        assert b == [{"played_s": 0.5}], mouth.got
        assert "barge" in page.types()
        await stop(page, task)
    asyncio.run(go())


def test_turn_ends_on_the_page_report():
    """MOUTH is done long before the page: the turn ends when the page reports it has played everything."""
    async def go():
        page, mouth, conv, task, hops = await start(report=False)
        await user_turn(page, mouth)
        await mouth.send(b"A", ramp(0.2).tobytes()); await mouth.send(b"D", {"text": "x", "barged": False})
        await asyncio.sleep(0.5)                                     # long past the reply, but no report yet
        assert "turn_done" not in page.types() and conv.speaking
        page.command(f"played:{conv.playback.rid}:{len(ramp(0.2))}")
        await asyncio.sleep(0.05)
        assert page.types()[-1] == "turn_done" and not conv.speaking
        await stop(page, task)
    asyncio.run(go())


def test_slow_downlink_no_longer_ends_the_turn_early():
    """A downlink 0.5 s slow: the old end (the server's guess + 0.25 s) came before the page had played the reply;
    now the turn ends at the page's report, after it."""
    async def go():
        page, mouth, conv, task, hops = await start(downlink_s=0.5)
        await user_turn(page, mouth)
        t0 = asyncio.get_running_loop().time()
        await mouth.send(b"A", ramp(0.2).tobytes()); await mouth.send(b"D", {"text": "x", "barged": False})
        await asyncio.sleep(0.6)                      # the old end: ~0.04 + 0.2 + 0.25 = 0.49 s
        assert conv.speaking, "ended before the page played the reply"
        await asyncio.sleep(0.3)                      # the page plays 0.54 .. 0.74 s, then reports
        assert not conv.speaking and page.types()[-1] == "turn_done"
        played_end = page.sched[-1][2] - t0
        assert 0.7 < played_end < 0.8
        await stop(page, task)
    asyncio.run(go())


def test_timeout_when_the_page_never_reports():
    """No report at all (a page that can't): the turn still ends, PLAYBACK_TIMEOUT_S past the server's guess."""
    async def go():
        old = C.PLAYBACK_TIMEOUT_S; C.PLAYBACK_TIMEOUT_S = 0.3
        try:
            page, mouth, conv, task, hops = await start(report=False)
            await user_turn(page, mouth)
            await mouth.send(b"A", ramp(0.1).tobytes()); await mouth.send(b"D", {"text": "x", "barged": False})
            await asyncio.sleep(0.3)
            assert conv.speaking
            await asyncio.sleep(0.3)                  # ~0.04 + 0.1 + 0.3
            assert not conv.speaking and page.types()[-1] == "turn_done"
            await stop(page, task)
        finally:
            C.PLAYBACK_TIMEOUT_S = old
    asyncio.run(go())


def test_backchannel_audio_is_untagged():
    """Backchannel audio goes to the page with id 0, which the page never tags: it stays out of the 2nd ear."""
    async def go():
        page, mouth, conv, task, hops = await start()
        conv.bc_active = True
        await mouth.send(b"K", ramp(0.1).tobytes()); await asyncio.sleep(0.05)
        assert ("audio", BACKCHANNEL_ID, 2400) in page.sent and page.tag_now()[1] == -1
        await stop(page, task)
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
