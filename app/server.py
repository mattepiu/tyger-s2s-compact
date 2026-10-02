"""EARS process + web app (EARS venv, tf 5.17): nemotron L15 -> the exp-10 chain + endpoint head (arm C),
serving the mic page + WebSocket on one port and wiring to the MOUTH over TCP. Adapted from
codebase/exp-12/server/ears_server.py (copy, not import), trimmed to ARM C ONLY: the 4-way endpointer,
SPEAK-lock watchdog, A/B blinding, and code->ckpt map are dropped.

PRIVACY: no audio or transcript is written to disk or logs. We log only timings + event types (endpoint
source, onset ms, barge, bc). One conversation at a time: a second visitor gets a "busy" page.

Run (EARS venv): HF_HOME=/workspace/.hf_home <venv>/bin/python -m app.server
"""
from __future__ import annotations
import os, sys, json, time, asyncio, logging, concurrent.futures
from pathlib import Path
import numpy as np
from aiohttp import web, WSMsgType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v1/
from app.wire import async_send_msg, async_recv_msg
from app.ears_core.endpoint_ears import EndpointEars
from app.ears_core.live_ears import HOP_FRAMES
from app.ears_core.encoder_stream import ENC_SR
from app.ears_core.labeling import FRAME_S
from app.config import load_config, discover_voices
from app import weights

log = logging.getLogger("ears")
HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
HOP_N = int(HOP_FRAMES * FRAME_S * ENC_SR)     # samples per hop (320 ms @ 16 kHz = 5120)
PLAYBACK_MARGIN_S = 0.25                        # slack for network + client scheduling on top of play_until
EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=1)

EARS = None
CFG = None
VOICES = {}
DEFAULT_VOICE = None
BUSY = False                                    # one conversation at a time


def ds24to16(x):
    """assistant audio (MOUTH 24 kHz) -> 16 kHz for the encoder / yield head 2nd ear."""
    n = int(len(x) * ENC_SR / 24000)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


BUSY_HTML = ("<!doctype html><meta charset=utf-8><title>tyger - busy</title>"
             "<body style='font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
             "<h1>tyger is busy</h1><p>Someone else is talking to it right now. "
             "It handles one conversation at a time &mdash; please try again shortly.</p></body>")


async def index(request):
    if BUSY:
        return web.Response(text=BUSY_HTML, content_type="text/html", status=503)
    return web.FileResponse(STATIC / "index.html")


async def voices_handler(request):
    # lets the page populate the voice picker on load (before the WS/Start), so a pre-Start pick is remembered.
    # Re-scans the voices dir so clips added after startup show up on a refresh (no EARS restart needed).
    global VOICES, DEFAULT_VOICE
    try:
        VOICES, DEFAULT_VOICE = discover_voices(CFG)
    except Exception as e:
        log.warning("voice re-discovery failed: %s", e)
    return web.json_response({"voices": sorted(VOICES.keys()), "voice": DEFAULT_VOICE})


async def ws_handler(request):
    global BUSY
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    if BUSY:
        await ws.send_json({"type": "busy"})
        await ws.close()
        return ws
    BUSY = True
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(EXEC, EARS.reset)
    mp = CFG["server"]
    buf = np.zeros(0, np.float32)
    # "speaking" lasts until the client has finished PLAYING the reply, not until the MOUTH has finished
    # generating it: the TTS runs several times faster than real time, so most of a long reply is still
    # queued in the browser when the MOUTH says done (D). play_until estimates the client's playback end;
    # the D is held (done_task) until then so barge detection + the 2nd ear stay live over the audible reply.
    st = {"speaking": False, "reset": False, "halting": False, "pending": None,
          "bc_active": False, "agent_buf": np.zeros(0, np.float32),
          "turn_t0": None, "await_onset": False, "barge_t0": None, "bc_t0": None,
          "play_until": 0.0, "mouth_done": False, "done_task": None}
    try:
        m_reader, m_writer = await asyncio.open_connection(mp["mouth_host"], int(mp["mouth_port"]))
        have_mouth = True
    except OSError as e:
        have_mouth = False; log.warning("no MOUTH: %s", e)

    await ws.send_json({"type": "ready", "hop_ms": int(HOP_FRAMES * FRAME_S * 1e3), "sr": ENC_SR,
                        "mouth": have_mouth,
                        "voices": sorted(VOICES.keys()), "voice": DEFAULT_VOICE,
                        "thresholds": {"barge_thr": EARS.barge_thr, "bc_thr": EARS.bc_thr,
                                       "ep_thr": EARS.ep_thr, "ep_deb": EARS.ep_deb,
                                       "endpoint": "ep", "fallback_frames": EARS.FALLBACK_SIL_FRAMES}})

    async def send_turn(transcript):
        st["speaking"] = True; st["turn_t0"] = time.perf_counter(); st["await_onset"] = True
        st["mouth_done"] = False; st["play_until"] = 0.0
        await async_send_msg(m_writer, b"T", {"transcript": transcript})

    async def finish_turn(barged):
        """The reply is over (played out, or barged): back to listening; send a held turn if any."""
        st["speaking"] = False; st["reset"] = True; st["agent_buf"] = np.zeros(0, np.float32)
        st["mouth_done"] = False; st["done_task"] = None
        if st["barge_t0"] is not None and barged:
            log.info("barge_halted tail %.0f ms", (time.perf_counter() - st["barge_t0"]) * 1e3)
        st["barge_t0"] = None
        log.info("turn_done barged=%s", bool(barged))
        await ws.send_json({"type": "turn_done", "barged": barged})
        if st["halting"]:
            st["halting"] = False
            if st["pending"] is not None:
                tr = st["pending"]; st["pending"] = None
                await send_turn(tr)

    async def finish_after_playback():
        try:
            while (wait := st["play_until"] + PLAYBACK_MARGIN_S - time.perf_counter()) > 0:
                await asyncio.sleep(wait)
            await finish_turn(False)
        except asyncio.CancelledError:
            pass

    async def send_backchannel(partial):
        st["bc_active"] = True; st["bc_t0"] = time.perf_counter()
        await async_send_msg(m_writer, b"C", {"partial": partial or ""})

    async def pump_mouth():
        try:
            while True:
                kind, payload = await async_recv_msg(m_reader)
                if kind == b"A":
                    if not (st["speaking"] or st["bc_active"]):
                        continue
                    if st["await_onset"] and st["speaking"] and st["turn_t0"] is not None:
                        onset = time.perf_counter() - st["turn_t0"]; st["await_onset"] = False
                        log.info("response_onset %.0f ms", onset * 1e3)
                    await ws.send_bytes(payload)
                    if st["speaking"]:
                        a24 = np.frombuffer(payload, np.float32)
                        now = time.perf_counter()        # client queues chunks back to back (+40 ms lead-in)
                        st["play_until"] = max(st["play_until"], now + 0.04) + len(a24) / 24000
                        st["agent_buf"] = np.concatenate([st["agent_buf"], ds24to16(a24)])
                elif kind == b"X":
                    d = json.loads(payload)
                    await ws.send_json({"type": "backchannel" if d.get("bc") else "reply", "text": d["text"]})
                elif kind == b"D":
                    d = json.loads(payload)
                    if d.get("bc"):
                        st["bc_active"] = False
                        await ws.send_json({"type": "backchannel_done", "text": d.get("text", "")})
                        continue
                    if d.get("barged") or not st["speaking"]:       # halted: nothing left to play out
                        await finish_turn(bool(d.get("barged")))
                    else:                                         # generated; hold until it has played
                        st["mouth_done"] = True
                        st["done_task"] = asyncio.create_task(finish_after_playback())
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
    pump = asyncio.create_task(pump_mouth()) if have_mouth else None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                buf = np.concatenate([buf, np.frombuffer(msg.data, dtype=np.float32)])
                while len(buf) >= HOP_N:
                    uh = np.ascontiguousarray(buf[:HOP_N]); buf = buf[HOP_N:]
                    if st["reset"]:
                        await loop.run_in_executor(EXEC, EARS.reset); st["reset"] = False
                    if st["speaking"]:
                        ab = st["agent_buf"]; ah = np.zeros(HOP_N, np.float32)
                        n = min(HOP_N, len(ab)); ah[:n] = ab[:n]; st["agent_buf"] = ab[n:]; mode = "speaking"
                    else:
                        ah = np.zeros(HOP_N, np.float32); mode = "listen"
                    evs = await loop.run_in_executor(EXEC, EARS.feed, uh, ah, mode)
                    yp = bcp = pe = 0.0; last = None
                    for e in evs:
                        if e["kind"] == "FRAME":
                            yp = max(yp, e.get("yield_prob", 0.0) or 0.0)
                            bcp = max(bcp, e.get("bc_prob", 0.0) or 0.0)
                            pe = max(pe, e.get("p_end", 0.0) or 0.0); last = e
                        elif e["kind"] == "ENDPOINT" and not st["speaking"]:
                            log.info("endpoint by=%s turn_s=%.2f", e.get("by"), e.get("turn_s", 0.0))
                            await ws.send_json({"type": "endpoint", "turn_s": round(e["turn_s"], 1),
                                                "by": e.get("by")})
                            if have_mouth and e["transcript"].strip():
                                if st["halting"]:
                                    st["pending"] = e["transcript"]
                                else:
                                    await send_turn(e["transcript"])
                        elif e["kind"] == "BACKCHANNEL" and not st["speaking"] and not st["halting"] \
                                and not st["bc_active"]:
                            log.info("backchannel bc_prob=%.2f partial_chars=%d", e.get("bc_prob", 0.0),
                                     len(e.get("partial") or ""))   # length only: never log the words
                            await ws.send_json({"type": "bc_cue", "bc": e.get("bc_prob")})
                            if have_mouth:
                                await send_backchannel(e.get("partial", ""))
                        elif e["kind"] == "BARGE" and st["speaking"]:
                            st["speaking"] = False; st["halting"] = True; st["reset"] = True
                            st["agent_buf"] = np.zeros(0, np.float32); st["barge_t0"] = time.perf_counter()
                            log.info("barge yield=%.2f", e.get("yield_prob", 0.0))
                            await ws.send_json({"type": "barge", "yield": e["yield_prob"]})  # client stops playback
                            if st["mouth_done"]:
                                # MOUTH already finished generating (its D was held for playback) -> no D will
                                # come back for this turn; close it here instead of waiting.
                                if st["done_task"] is not None:
                                    st["done_task"].cancel()
                                st["halting"] = False
                                await finish_turn(True)
                            elif have_mouth:
                                await async_send_msg(m_writer, b"B")
                    if last is not None:
                        await ws.send_json({"type": "state", "state": last["state"], "yield": yp,
                                            "bc": bcp, "hazard": last.get("hazard", 0.0), "p_end": round(pe, 3)})
            elif msg.type == WSMsgType.TEXT:
                data = msg.data
                if data == "reset":
                    await loop.run_in_executor(EXEC, EARS.reset); buf = np.zeros(0, np.float32)
                elif data.startswith("voice:") and have_mouth:
                    name = data.split(":", 1)[1]
                    if name in VOICES:
                        await async_send_msg(m_writer, b"V", {"voice": name})
                        await ws.send_json({"type": "voice", "voice": name})
                elif data.startswith("set:"):
                    try:
                        _, attr, val = data.split(":", 2)
                        if attr in ("barge_thr", "bc_thr", "ep_thr"):
                            setattr(EARS, attr, float(val))
                        elif attr in ("ep_deb", "bc_deb", "barge_deb"):
                            setattr(EARS, attr, int(val))
                        elif attr == "fallback_frames":
                            EARS.FALLBACK_SIL_FRAMES = int(val)
                        cur = getattr(EARS, "FALLBACK_SIL_FRAMES") if attr == "fallback_frames" else getattr(EARS, attr)
                        await ws.send_json({"type": "threshold", "attr": attr, "value": cur})
                    except (ValueError, AttributeError):
                        pass
    finally:
        if pump:
            pump.cancel()
        if st["done_task"] is not None:
            st["done_task"].cancel()
        if have_mouth:
            try:
                m_writer.close()
            except Exception:
                pass
        BUSY = False
        log.info("client disconnected")
    return ws


def build_ears(cfg):
    heads = weights.ensure_heads(cfg["models"]["heads_repo"], cfg["models"]["heads_file"])
    e = cfg["ears"]
    log.info("loading EARS (arm C) ep_thr=%s ep_deb=%s", e["ep_thr"], e["ep_deb"])
    ears = EndpointEars(heads, ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), echo=True,
                        theta=0.2, barge_thr=float(e["barge_thr"]), barge_deb=int(e["barge_deb"]),
                        bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]))
    ears.FALLBACK_SIL_FRAMES = int(e.get("fallback_sil_frames", 15))
    return ears


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    global EARS, CFG, VOICES, DEFAULT_VOICE
    CFG = load_config(args.config)
    VOICES, DEFAULT_VOICE = discover_voices(CFG)
    EARS = build_ears(CFG)
    log.info("=== EARS ready ===")
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/voices", voices_handler), web.get("/ws", ws_handler)])
    host, port = CFG["server"]["host"], int(CFG["server"]["port"])
    log.info("=== http://%s:%d  (mic needs https OR http://localhost via tunnel) ===", host, port)
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
