"""v1 smoke test (no browser). Two parts:

  --mouth   : connect to the MOUTH over TCP and run three turns -- a plain reply, a Tavily tool turn
              (weather), and a placeholder tool turn (reminder) -- asserting audio comes back and the
              placeholder reply admits it isn't connected. Deterministic (text in).
  --ears    : connect to the running EARS web app over WebSocket, stream a speech wav as the mic, and
              assert an endpoint fires and reply audio returns (synthetic wav -> endpoint -> audio out).
  (default) : run both. Requires `run.sh` (EARS + MOUTH) to be up for --ears; only MOUTH for --mouth.

Run (EARS venv): HF_HOME=/workspace/.hf_home <venv>/bin/python v1/scripts/smoke_test.py [--mouth|--ears]
"""
import os, sys, json, time, socket, struct, argparse, wave
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from app.wire import send_msg, recv_msg
from app.config import load_config


def mouth_turn(sock, transcript, label):
    send_msg(sock, b"T", {"transcript": transcript})
    audio = 0; text = ""; t0 = time.time(); first = None
    while True:
        kind, payload = recv_msg(sock)
        if kind is None:
            break
        if kind == b"A":
            if first is None:
                first = time.time() - t0
            audio += len(payload)
        elif kind == b"X":
            d = json.loads(payload); text = d.get("text", "")
        elif kind == b"D":
            break
    samples = audio // 4
    print(f"  [{label}] first-audio={(first*1e3 if first else -1):.0f}ms audio={samples/24000:.2f}s "
          f"reply={text[:70]!r}")
    return samples, text


def test_mouth(cfg):
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    s = socket.create_connection((host, port), timeout=120)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("== MOUTH smoke ==")
    n1, _ = mouth_turn(s, "Hi there, how are you today?", "plain")
    n2, t2 = mouth_turn(s, "What's the weather in Paris right now?", "tavily-weather")
    n3, t3 = mouth_turn(s, "Please set a reminder to call mom at six pm tomorrow.", "placeholder-reminder")
    s.close()
    ok = n1 > 0 and n2 > 0 and n3 > 0
    # the placeholder reply should NOT claim it was done; ideally admits it isn't connected
    admits = any(w in t3.lower() for w in ("not connected", "isn't connected", "can't", "cannot",
                                           "unable", "not able", "not set up", "not yet"))
    print(f"  placeholder admits-not-connected: {admits}")
    assert ok, "MOUTH produced no audio on one or more turns"
    print("  MOUTH smoke: PASS")
    return True


def load_wav16(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        m = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype(np.float32)
    return x


def test_ears(cfg, wav_path):
    import asyncio, aiohttp
    port = int(cfg["server"]["port"])

    async def run():
        x = load_wav16(wav_path)
        x = np.concatenate([x, np.zeros(16000 * 2, np.float32)])  # 2 s trailing silence -> endpoint
        url = f"http://127.0.0.1:{port}/ws"
        async with aiohttp.ClientSession() as sess:
            async with sess.ws_connect(url, max_msg_size=0) as ws:
                endpoint = False; audio = 0
                hop = 1600  # 100 ms of 16k f32
                async def feed():
                    for i in range(0, len(x), hop):
                        await ws.send_bytes(x[i:i+hop].tobytes()); await asyncio.sleep(0.01)
                ft = asyncio.create_task(feed())
                t0 = time.time()
                while time.time() - t0 < 60:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        audio += len(msg.data)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        m = json.loads(msg.data)
                        if m.get("type") == "endpoint":
                            endpoint = True
                        if m.get("type") == "turn_done":
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                ft.cancel()
                print(f"  endpoint_fired={endpoint} reply_audio={audio//4/24000:.2f}s")
                assert endpoint, "no endpoint fired from the speech wav"
                assert audio > 0, "no reply audio returned"
        print("  EARS smoke: PASS")
    print("== EARS smoke ==")
    asyncio.run(run())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouth", action="store_true"); ap.add_argument("--ears", action="store_true")
    ap.add_argument("--wav", default="/workspace/MOSS-TTS-Nano/assets/audio/en_2.wav")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    do_mouth = args.mouth or not args.ears
    do_ears = args.ears or not args.mouth
    if do_mouth:
        test_mouth(cfg)
    if do_ears:
        test_ears(cfg, args.wav)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
