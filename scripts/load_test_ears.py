"""Multi-session EARS load test (no browser). Needs run.sh up. Opens N WebSocket conversations at once; each
streams speech at real-time pace, then silence, for a few turns, like a browser mic. Per conversation it
measures:
  * hop lag: how long after a 320 ms hop is sent its `state` event comes back (EARS keeping up with real time);
  * endpoint -> first reply audio (onset);
  * turns completed.
The speech is the MOUTH's own TTS reply, held in memory (nothing is written to disk).

  .venv/bin/python scripts/load_test_ears.py --sessions 4 [--turns 2]
"""
import sys, json, time, socket, asyncio, argparse, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.wire import send_msg, recv_msg
from app.config import load_config

HOP = 5120          # 320 ms @ 16 kHz, the EARS hop
CHUNK = 1600        # client sends 100 ms at a time, like the page


def speech_from_mouth(cfg):
    s = socket.create_connection((cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])), timeout=120)
    send_msg(s, b"T", {"transcript": "Say: I would like to know what the weather is like in Berlin today."})
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


async def conversation(idx, port, speech, turns, start_delay, out):
    import aiohttp
    await asyncio.sleep(start_delay)
    res = {"lag": [], "onset": [], "turns": 0, "busy": False}
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"http://127.0.0.1:{port}/ws", max_msg_size=0) as ws:
            first = await ws.receive()
            if json.loads(first.data).get("type") == "busy":
                res["busy"] = True; out[idx] = res; return
            sent = 0; hop_sent_at = []; states = 0
            phase = {"speech_left": turns, "need_silence": False, "t_endpoint": None, "waiting_reply": False}
            done = asyncio.Event()

            async def mic():
                nonlocal sent
                pos = 0; t0 = time.perf_counter(); n = 0
                while not done.is_set():
                    if phase["speech_left"] > 0 and not phase["waiting_reply"] and pos < len(speech):
                        chunk = speech[pos:pos + CHUNK]; pos += CHUNK
                        if pos >= len(speech):
                            phase["waiting_reply"] = True; pos = 0
                    else:
                        chunk = np.zeros(CHUNK, np.float32)
                    if len(chunk) < CHUNK:
                        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
                    await ws.send_bytes(chunk.tobytes())
                    sent += CHUNK
                    while len(hop_sent_at) < sent // HOP:
                        hop_sent_at.append(time.perf_counter())
                    n += 1
                    await asyncio.sleep(max(0.0, t0 + n * CHUNK / 16000 - time.perf_counter()))

            task = asyncio.create_task(mic())
            t_end = time.time() + 40 * turns
            while time.time() < t_end and res["turns"] < turns:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if phase["t_endpoint"] is not None:
                        res["onset"].append(time.perf_counter() - phase["t_endpoint"]); phase["t_endpoint"] = None
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m["type"] == "state":
                        if states < len(hop_sent_at):
                            res["lag"].append(time.perf_counter() - hop_sent_at[states])
                        states += 1
                    elif m["type"] == "endpoint":
                        phase["t_endpoint"] = time.perf_counter()
                    elif m["type"] == "turn_done":
                        res["turns"] += 1; phase["speech_left"] -= 1; phase["waiting_reply"] = False
                else:
                    break
            done.set(); await task
    out[idx] = res


async def main_async(a, cfg, speech):
    out = {}
    await asyncio.gather(*[conversation(i, int(cfg["server"]["port"]), speech, a.turns, i * 0.9, out)
                           for i in range(a.sessions)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    speech = speech_from_mouth(cfg)
    out = asyncio.run(main_async(a, cfg, speech))
    lag = [x for r in out.values() for x in r["lag"]]
    onset = [x for r in out.values() for x in r["onset"]]
    ms = lambda v: (f"p50 {1e3*statistics.median(v):.0f} / p90 {1e3*np.percentile(v, 90):.0f} / "  # noqa: E731
                    f"max {1e3*max(v):.0f} ms") if v else "n/a"
    print(f"sessions={a.sessions} (speech {len(speech)/16000:.1f}s, {a.turns} turns each)")
    for i in sorted(out):
        r = out[i]
        print(f"  conv {i}: busy={r['busy']} turns={r['turns']} hops={len(r['lag'])} "
              f"lag max {1e3*max(r['lag']) if r['lag'] else -1:.0f} ms")
    print(f"  hop lag (state back after hop sent): {ms(lag)}")
    print(f"  endpoint -> first reply audio:       {ms(onset)}")


if __name__ == "__main__":
    main()
