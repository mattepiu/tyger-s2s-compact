"""Multi-session EARS load test (no browser). Needs run.sh up. Opens N WebSocket conversations at once; each
streams speech at real-time pace, then silence, for a few turns, like a browser mic. Per conversation it
measures:
  * hop lag: how long after a 320 ms hop is sent its `state` event comes back (EARS keeping up with real time);
  * end of speech -> `endpoint` event (detection delay: what live threshold tuning should drive down);
  * end of speech -> first reply audio (the response gap the user feels, minus client playback buffering);
  * endpoint -> first reply audio (onset: the MOUTH + wire part of that gap);
  * turns completed, and the reply audio per turn;
  * turns EARS ended before this page had played the reply (I-17: none, even with --downlink-ms over 250).
End of speech is the last voiced 20 ms frame of the injected utterance, timed on the simulated mic clock (a sample
is "captured" one 100 ms chunk before the chunk carrying it is sent, as in a browser). Each conversation is a
simulated page (scripts/stack/page.py): it plays the reply on the wall clock, tags its mic blocks with what was
playing, and reports when a reply has played out; --downlink-ms delays its playback as a slow network would.
Deterministic, so runs compare: the utterance is a fixed sentence spoken by MOUTH's TTS (the S message: TTS only,
no LLM; VoxCPM2 samples at temperature 0 with a fixed seed), held in memory (nothing is written to disk). It asks
for a reply the LLM can't vary (counting to ten), so every turn gets the same reply text; the test compares the
reply texts by hash (never printed) and warns if they differ. The reply AUDIO still varies a little with load:
VoxCPM2 isn't batch-invariant, so the same text renders 8.8-9.8 s depending on what it is batched with.

  .venv/bin/python scripts/stack/load_ears.py --sessions 4 [--turns 4] [--full-speech]
"""
import sys, json, time, socket, asyncio, hashlib, argparse, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.wire import send_msg, recv_msg
from app.config import load_config
from scripts.stack.page import PageSim

HOP = 5120          # 320 ms @ 16 kHz, the EARS hop
CHUNK = 1600        # client sends 100 ms at a time, like the page
VOICED_DB = -45.0   # a 20 ms frame above this RMS level counts as speech (TTS tails are near-silent)
# The user's utterance: one sentence (no inner pause for the endpoint to fire at) whose reply is fixed: 8/8 fresh
# sessions and 3/3 turns with history gave the same reply, 9.76 s of audio (I-13); with Gemma 4 E4B at temperature
# 0.7 too: one reply text over 4 turns alone and 16 at 4 conversations (2026-10-06), though under 4-way batching it
# has varied now and then (the test warns).
UTTERANCE = "Could you please count from one to ten for me?"
# --full-speech: three sentences with pauses between them, to count endpoints that fire inside an utterance
# (the guard rail for live threshold tuning, O-01). Same request at the end, so the reply is the same.
FULL_SPEECH = "I was at the market this morning. It was really busy today. Anyway, could you please count from " \
              "one to ten for me?"


def _voiced(x, sr=16000):
    f = int(0.02 * sr); n = len(x) // f
    return 20 * np.log10(np.sqrt((x[:n * f].reshape(n, f) ** 2).mean(1)) + 1e-9) > VOICED_DB


def voiced_end(x, sr=16000):
    """Sample index just past the last voiced 20 ms frame of x."""
    v = np.nonzero(_voiced(x, sr))[0]
    return int((v[-1] + 1) * 0.02 * sr) if v.size else len(x)


def longest_pause(x, sr=16000):
    """Longest unvoiced run between the first and last voiced 20 ms frames of x, in seconds."""
    voiced = _voiced(x, sr); v = np.nonzero(voiced)[0]
    best = run = 0
    for flag in voiced[v[0]:v[-1] + 1] if v.size else []:
        run = 0 if flag else run + 1; best = max(best, run)
    return best * 0.02


def speech_from_mouth(cfg, text, voice=None):
    """`text` spoken by MOUTH's TTS (S: no LLM) in `voice` (default: MOUTH's), resampled to 16 kHz, in memory."""
    s = socket.create_connection((cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])), timeout=120)
    if voice:
        send_msg(s, b"V", {"voice": voice})
    send_msg(s, b"S", {"text": text})
    chunks = []
    while True:
        k, p = recv_msg(s)
        if k == b"A":
            chunks.append(np.frombuffer(p, np.float32))
        elif k in (b"D", None):
            break
    s.close()
    a = np.concatenate(chunks)
    return np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / 24000)), np.arange(len(a)), a).astype(np.float32)


async def conversation(idx, port, speech, turns, start_delay, out, downlink_s=0.0):
    import aiohttp
    await asyncio.sleep(start_delay)
    res = {"lag": [], "onset": [], "detect": [], "gap": [], "early": 0, "lost": 0, "by": {}, "turns": 0,
           "busy": False, "reply_s": [], "replies": set(), "bc": 0, "cut_short": 0}
    reply_n = None      # reply audio samples since the endpoint (24 kHz); None outside a reply
    reply_txt = ""      # the reply text so far (compared by hash only, never printed)
    v_end = voiced_end(speech)
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"http://127.0.0.1:{port}/ws", max_msg_size=0) as ws:
            first = await ws.receive()
            if json.loads(first.data).get("type") == "busy":
                res["busy"] = True; out[idx] = res; return
            page = PageSim(ws, downlink_s)
            sent = 0; hop_sent_at = []; states = 0
            phase = {"speech_left": turns, "need_silence": False, "t_endpoint": None, "waiting_reply": False,
                     "t_speech_end": None, "t_gap0": None, "utt_end": None, "last_ep": 0.0, "last_audio": 0.0}
            done = asyncio.Event()

            async def mic():
                nonlocal sent
                pos = 0; t0 = time.perf_counter(); n = 0; k = 0; utt = speech
                while not done.is_set():
                    if phase["speech_left"] > 0 and not phase["waiting_reply"] and pos < len(utt):
                        if pos == 0:
                            # Lead-in silence so the speech ends at a fixed point of EARS's hop grid (hops are cut
                            # from the stream at multiples of HOP): 0, 1/4, 1/2, 3/4 of a hop, cycling over turns
                            # and conversations. Otherwise the hop quantisation of detection (0-320 ms) depends on
                            # wall-clock turn timing and runs don't compare.
                            pad = (((idx + k) % 4) * HOP // 4 - (sent + v_end)) % HOP; k += 1
                            utt = np.concatenate([np.zeros(pad, np.float32), speech])
                            # utterance starts at global sample `sent`; its speech ends pad + v_end later
                            phase["t_speech_end"] = phase["utt_end"] = t0 + (sent + pad + v_end - CHUNK) / 16000
                        chunk = utt[pos:pos + CHUNK]; pos += CHUNK
                        if pos >= len(utt):
                            phase["waiting_reply"] = True; pos = 0
                    else:
                        chunk = np.zeros(CHUNK, np.float32)
                    if len(chunk) < CHUNK:
                        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
                    await page.send_mic(chunk, t0 + (sent - CHUNK) / 16000)   # captured a chunk ago
                    sent += CHUNK
                    while len(hop_sent_at) < sent // HOP:
                        hop_sent_at.append(time.perf_counter())
                    n += 1
                    await asyncio.sleep(max(0.0, t0 + n * CHUNK / 16000 - time.perf_counter()))

            task = asyncio.create_task(mic())
            t_end = time.time() + 40 * turns
            while time.time() < t_end and res["turns"] < turns:
                now = time.perf_counter(); ue = phase["utt_end"]
                if phase["waiting_reply"] and ue is not None and now - ue > 6 and phase["last_ep"] < ue \
                        and now - phase["last_audio"] > 2:   # utterance never got an endpoint (spoken over the
                    res["lost"] += 1; phase["waiting_reply"] = False   # reply, no barge): say it again
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.BINARY:
                    now = time.perf_counter(); phase["last_audio"] = now
                    rid, n = page.on_audio(msg.data)
                    if reply_n is not None and rid:
                        reply_n += n
                    if phase["t_endpoint"] is not None:
                        res["onset"].append(now - phase["t_endpoint"]); phase["t_endpoint"] = None
                    if phase["t_gap0"] is not None:
                        res["gap"].append(now - phase["t_gap0"]); phase["t_gap0"] = None
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m["type"] == "state":
                        if states < len(hop_sent_at):
                            res["lag"].append(time.perf_counter() - hop_sent_at[states])
                        states += 1
                    elif m["type"] in ("barge",):
                        page.stop()
                    elif m["type"] == "endpoint":
                        page.stop()                       # the page never lets a new reply overlap an old one
                        now = time.perf_counter(); phase["t_endpoint"] = phase["last_ep"] = now; reply_n = 0
                        res["by"][m.get("by")] = res["by"].get(m.get("by"), 0) + 1
                        te = phase["t_speech_end"]; phase["t_speech_end"] = None
                        if te is not None and now >= te:
                            res["detect"].append(now - te); phase["t_gap0"] = te
                        elif te is not None:  # fired at a pause inside the utterance: no clean gap to measure
                            res["early"] += 1
                    elif m["type"] == "bc_cue":
                        res["bc"] += 1
                    elif m["type"] == "reply":
                        reply_txt = m.get("text", "")
                    elif m["type"] == "turn_done":
                        if page.ends.get(page.last_id, 0.0) > time.perf_counter() + 0.02:   # still playing here
                            res["cut_short"] += 1
                        res["turns"] += 1; phase["speech_left"] -= 1; phase["waiting_reply"] = False
                        if reply_n is not None:
                            res["reply_s"].append(reply_n / 24000); reply_n = None
                        res["replies"].add(hashlib.sha1(reply_txt.encode()).hexdigest()); reply_txt = ""
                else:
                    break
            done.set(); await task
    out[idx] = res


async def main_async(a, cfg, speech):
    out = {}
    await asyncio.gather(*[conversation(i, int(cfg["server"]["port"]), speech, a.turns, i * 0.9, out,
                                        a.downlink_ms / 1000)
                           for i in range(a.sessions)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4, help="a multiple of 4 covers the hop phases evenly")
    ap.add_argument("--full-speech", action="store_true",
                    help="a three-sentence utterance with inner pauses (counts endpoints fired inside it)")
    ap.add_argument("--downlink-ms", type=float, default=0.0,
                    help="the simulated page plays reply audio this much later (a slow network)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    speech = speech_from_mouth(cfg, FULL_SPEECH if a.full_speech else UTTERANCE)
    out = asyncio.run(main_async(a, cfg, speech))
    lag = [x for r in out.values() for x in r["lag"]]
    onset = [x for r in out.values() for x in r["onset"]]
    detect = [x for r in out.values() for x in r["detect"]]
    gap = [x for r in out.values() for x in r["gap"]]
    by = {}
    for r in out.values():
        for k, v in r["by"].items():
            by[k] = by.get(k, 0) + v
    ms = lambda v: (f"p50 {1e3*statistics.median(v):.0f} / p90 {1e3*np.percentile(v, 90):.0f} / "  # noqa: E731
                    f"max {1e3*max(v):.0f} ms") if v else "n/a"
    reply_s = [x for r in out.values() for x in r["reply_s"]]
    print(f"sessions={a.sessions} (speech {len(speech)/16000:.2f}s, voiced to {voiced_end(speech)/16000:.2f}s, "
          f"longest inner pause {1e3*longest_pause(speech):.0f} ms, {a.turns} turns each)")
    for i in sorted(out):
        r = out[i]
        print(f"  conv {i}: busy={r['busy']} turns={r['turns']} hops={len(r['lag'])} "
              f"lag max {1e3*max(r['lag']) if r['lag'] else -1:.0f} ms")
    print(f"  hop lag (state back after hop sent): {ms(lag)}")
    print(f"  endpoint -> first reply audio:       {ms(onset)}")
    print(f"  end of speech -> endpoint event:     {ms(detect)}")
    print(f"  end of speech -> first reply audio:  {ms(gap)}")
    print(f"  endpoints by {by}; fired inside the utterance: {sum(r['early'] for r in out.values())}; "
          f"utterances lost (no endpoint): {sum(r['lost'] for r in out.values())}")
    print(f"  turns ended before the page had played the reply: {sum(r['cut_short'] for r in out.values())} "
          f"(downlink {a.downlink_ms:g} ms)")
    replies = set().union(*(r["replies"] for r in out.values()))
    print(f"  distinct reply texts: {len(replies)}; reply audio per turn: "
          + (f"{min(reply_s):.2f}-{max(reply_s):.2f} s (n={len(reply_s)})" if reply_s else "n/a")
          + f"; backchannel cues: {sum(r['bc'] for r in out.values())}")
    if len(replies) > 1 and not a.full_speech:      # --full-speech: early endpoints change the transcript
        print("  WARNING: the reply text varied, so this run doesn't compare with others")


if __name__ == "__main__":
    main()
