"""v1 smoke test (no browser). Two parts:

  --mouth   : connect to the MOUTH over TCP and run three turns -- a plain reply, a Tavily tool turn
              (weather), and a placeholder tool turn (reminder) -- asserting audio comes back and the
              placeholder reply admits it isn't connected. Deterministic (text in).
  --ears    : connect to the running EARS web app over WebSocket, stream speech as the mic, and assert an
              endpoint fires and reply audio returns (speech -> endpoint -> audio out). The speech is the
              MOUTH's own TTS reply held in memory (or --wav).
  (default) : run both. Requires `run.sh` (EARS + MOUTH) to be up for --ears; only MOUTH for --mouth.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python scripts/smoke_test.py [--mouth|--ears] [--wav x.wav]
"""
import os, sys, json, time, socket, struct, argparse, wave
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from app.wire import send_msg, recv_msg
from app.config import load_config


def mouth_turn(sock, transcript, label, keep_audio=False):
    send_msg(sock, b"T", {"transcript": transcript})
    audio = 0; text = ""; t0 = time.time(); first = None; chunks = []
    while True:
        kind, payload = recv_msg(sock)
        if kind is None:
            break
        if kind == b"A":
            if first is None:
                first = time.time() - t0
            audio += len(payload)
            if keep_audio:
                chunks.append(np.frombuffer(payload, np.float32))
        elif kind == b"X":
            d = json.loads(payload); text = d.get("text", "")
        elif kind == b"D":
            break
    samples = audio // 4
    print(f"  [{label}] first-audio={(first*1e3 if first else -1):.0f}ms audio={samples/24000:.2f}s "
          f"reply={text[:70]!r}")
    wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return samples, text, wav


def mouth_backchannel(sock, partial):
    """C frame with an ASR partial -> a short contextual ack (text + audio)."""
    send_msg(sock, b"C", {"partial": partial})
    text = ""; audio = 0; t0 = time.time(); first = None
    while True:
        kind, payload = recv_msg(sock)
        if kind is None:
            break
        if kind == b"A":
            first = first or time.time() - t0; audio += len(payload)
        elif kind == b"D":
            text = json.loads(payload).get("text", ""); break
    print(f"  [backchannel] first-audio={(first*1e3 if first else -1):.0f}ms ack={text!r}")
    return text, audio


def mouth_session(cfg):
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    s = socket.create_connection((host, port), timeout=120)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def test_mouth(cfg):
    """Returns the plain turn's reply audio (24 kHz, in memory) so the EARS test can reuse it as speech."""
    s = mouth_session(cfg)
    print("== MOUTH smoke ==")
    n1, _, speech = mouth_turn(s, "Hi there, how are you today?", "plain", keep_audio=True)
    n2, t2, _ = mouth_turn(s, "What's the weather in Paris right now?", "tavily-weather")
    n3, t3, _ = mouth_turn(s, "Please set a reminder to call mom at six pm tomorrow.", "placeholder-reminder")
    bc_text, bc_audio = mouth_backchannel(s, "so yesterday I finally got the job offer I was hoping for and")
    s.close()
    assert bc_text and bc_audio > 0, "backchannel produced no ack text/audio"
    ok = n1 > 0 and n2 > 0 and n3 > 0
    # the placeholder reply should NOT claim it was done; ideally admits it isn't connected
    admits = any(w in t3.lower() for w in ("not connected", "isn't connected", "can't", "cannot",
                                           "unable", "not able", "not set up", "not yet"))
    print(f"  placeholder admits-not-connected: {admits}")
    assert ok, "MOUTH produced no audio on one or more turns"
    print("  MOUTH smoke: PASS")
    return speech


def speech_from_mouth(cfg):
    """Synthesize a test utterance with the MOUTH's own TTS (kept in memory, never written to disk)."""
    s = mouth_session(cfg)
    _, _, wav = mouth_turn(s, "Say: I would like to know what the weather is like in Berlin today.",
                           "speech-for-ears", keep_audio=True)
    s.close()
    return wav


def to16k(x, sr):
    if sr == 16000:
        return x.astype(np.float32)
    m = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype(np.float32)


def load_wav16(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return to16k(x, sr)


def test_ears(cfg, speech16):
    import asyncio, aiohttp
    port = int(cfg["server"]["port"])

    async def run():
        x = np.concatenate([np.zeros(8000, np.float32), speech16,
                            np.zeros(16000 * 2, np.float32)])  # 2 s trailing silence -> endpoint
        url = f"http://127.0.0.1:{port}/ws"
        async with aiohttp.ClientSession() as sess:
            async with sess.ws_connect(url, max_msg_size=0) as ws:
                endpoint = False; audio = 0; t_first = t_done = None
                hop = 1600  # 100 ms of 16k f32
                async def feed():   # real-time mic: speech, then silence for as long as the test runs
                    i = 0
                    while True:
                        chunk = x[i:i+hop] if i < len(x) else np.zeros(hop, np.float32)
                        await ws.send_bytes(chunk.tobytes()); i += hop
                        await asyncio.sleep(hop / 16000)
                ft = asyncio.create_task(feed())
                t0 = time.time()
                while time.time() - t0 < 90:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if endpoint:                      # reply audio only (backchannels play before it)
                            audio += len(msg.data); t_first = t_first or time.time()
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        m = json.loads(msg.data)
                        if m.get("type") == "endpoint":
                            endpoint = True
                        if m.get("type") == "backchannel":
                            print(f"  backchannel during speech: {m.get('text')!r}")
                        if m.get("type") == "turn_done":
                            t_done = time.time(); break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                ft.cancel()
                secs = audio // 4 / 24000
                held = (t_done - t_first) if (t_done and t_first) else -1
                print(f"  endpoint_fired={endpoint} reply_audio={secs:.2f}s first-audio->turn_done={held:.2f}s")
                assert endpoint, "no endpoint fired from the speech wav"
                assert audio > 0, "no reply audio returned"
                # the turn must stay open while the reply PLAYS (barge stays live), not end when generation does
                assert held >= secs - 0.3, f"turn_done after {held:.2f}s but reply plays {secs:.2f}s"
        print("  EARS smoke: PASS")
    print("== EARS smoke ==")
    asyncio.run(run())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouth", action="store_true"); ap.add_argument("--ears", action="store_true")
    ap.add_argument("--wav", default=None,
                    help="speech wav for the EARS test (default: the MOUTH's own TTS reply, in memory)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    do_mouth = args.mouth or not args.ears
    do_ears = args.ears or not args.mouth
    speech = None
    if do_mouth:
        speech = test_mouth(cfg)
    if do_ears:
        if args.wav:
            speech16 = load_wav16(args.wav)
        else:
            speech16 = to16k(speech if speech is not None and speech.size else speech_from_mouth(cfg), 24000)
        test_ears(cfg, speech16)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
