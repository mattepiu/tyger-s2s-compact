"""EARS process + web app: nemotron L15 -> the exp-10 chain + endpoint head (arm C), serving the mic page +
WebSocket on one port and wiring each conversation to the MOUTH over its own TCP connection. Adapted from
codebase/exp-12/server/ears_server.py (copy, not import), trimmed to ARM C ONLY: the 4-way endpointer,
SPEAK-lock watchdog, A/B blinding, and code->ckpt map are dropped.

Multi-session: up to server.max_sessions WebSocket conversations at once. The models are shared; each
conversation is an EarsSession, and one GPU thread runs a single batched step per 320 ms hop for every
session with a hop ready (app/ears_core/batched.py). Visitors beyond the cap get a "busy" page.

PRIVACY: no audio or transcript is written to disk or logs. We log only timings + event types (endpoint
source, onset ms, barge, bc).

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.server
"""
from __future__ import annotations
import os, sys, json, time, asyncio, logging
from pathlib import Path
import numpy as np
from aiohttp import web, WSMsgType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from app.wire import async_send_msg, async_recv_msg
from app.ears_core.batched import EarsEngine, EarsService
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

SERVICE: EarsService | None = None
CFG = None
VOICES = {}
DEFAULT_VOICE = None


def ds24to16(x):
    """assistant audio (MOUTH 24 kHz) -> 16 kHz for the encoder / yield head 2nd ear."""
    n = int(len(x) * ENC_SR / 24000)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


BUSY_HTML = ("<!doctype html><meta charset=utf-8><title>tyger - busy</title>"
             "<body style='font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
             "<h1>tyger is busy</h1><p>All conversation slots are in use right now "
             "&mdash; please try again shortly.</p></body>")


async def index(request):
    if SERVICE.full():
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


def session_thresholds(cfg):
    e = cfg["ears"]
    return dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
                barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]),
                fallback_frames=int(e.get("fallback_sil_frames", 15)))


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    ears = SERVICE.open(**session_thresholds(CFG))       # this conversation's EARS state (+ its own thresholds)
    if ears is None:
        await ws.send_json({"type": "busy"})
        await ws.close()
        return ws
    try:
        return await _conversation(ws, ears)
    finally:
        SERVICE.close(ears)


async def _conversation(ws, ears):
    mp = CFG["server"]
    buf = np.zeros(0, np.float32)
    # "speaking" lasts until the client has finished PLAYING the reply, not until the MOUTH has finished
    # generating it: the TTS runs several times faster than real time, so most of a long reply is still
    # queued in the browser when the MOUTH says done (D). play_until estimates the client's playback end;
    # the D is held (done_task) until then so barge detection + the 2nd ear stay live over the audible reply.
    st = {"speaking": False, "reset": False, "halting": False, "pending": None,
          "bc_active": False, "agent_buf": np.zeros(0, np.float32),
          "turn_t0": None, "await_onset": False, "barge_t0": None, "bc_t0": None,
          "play_until": 0.0, "sent_s": 0.0, "mouth_done": False, "done_task": None}
    try:
        m_reader, m_writer = await asyncio.open_connection(mp["mouth_host"], int(mp["mouth_port"]))
        have_mouth = True
    except OSError as e:
        have_mouth = False; log.warning("no MOUTH: %s", e)

    await ws.send_json({"type": "ready", "hop_ms": int(HOP_FRAMES * FRAME_S * 1e3), "sr": ENC_SR,
                        "mouth": have_mouth,
                        "voices": sorted(VOICES.keys()), "voice": DEFAULT_VOICE,
                        "thresholds": {"barge_thr": ears.barge_thr, "bc_thr": ears.bc_thr,
                                       "ep_thr": ears.ep_thr, "ep_deb": ears.ep_deb,
                                       "endpoint": "ep", "fallback_frames": ears.FALLBACK_SIL_FRAMES}})
    log.info("conversation start (active=%d)", SERVICE.active())

    async def send_turn(transcript):
        st["speaking"] = True; st["turn_t0"] = time.perf_counter(); st["await_onset"] = True
        st["mouth_done"] = False; st["play_until"] = 0.0; st["sent_s"] = 0.0
        await async_send_msg(m_writer, b"T", {"transcript": transcript})

    async def finish_turn(barged):
        """The reply is over (played out, or barged): back to listening; send a held turn if any. A barge already
        reset EARS (seeding the new turn with the start of the user's speech, I-02): don't wipe that turn again."""
        st["speaking"] = False; st["agent_buf"] = np.zeros(0, np.float32)
        if not barged:
            st["reset"] = True
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
                if kind == b"K":                          # backchannel audio: only while the user holds the
                    if st["bc_active"] and not st["speaking"]:  # floor (a reply turn supersedes it)
                        await ws.send_bytes(payload)
                    continue
                if kind == b"A":                          # reply audio
                    if not st["speaking"]:
                        continue
                    if st["await_onset"] and st["speaking"] and st["turn_t0"] is not None:
                        onset = time.perf_counter() - st["turn_t0"]; st["await_onset"] = False
                        log.info("response_onset %.0f ms", onset * 1e3)
                    await ws.send_bytes(payload)
                    if st["speaking"]:
                        a24 = np.frombuffer(payload, np.float32)
                        now = time.perf_counter()        # client queues chunks back to back (+40 ms lead-in)
                        st["play_until"] = max(st["play_until"], now + 0.04) + len(a24) / 24000
                        st["sent_s"] += len(a24) / 24000
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

    async def handle_events(evs):
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
                elif have_mouth:            # + how much of the reply had played, so MOUTH remembers what was heard
                    played = st["sent_s"] - max(0.0, st["play_until"] - time.perf_counter())
                    await async_send_msg(m_writer, b"B", {"played_s": round(max(0.0, played), 2)})
        if last is not None:
            await ws.send_json({"type": "state", "state": last["state"], "yield": yp,
                                "bc": bcp, "hazard": last.get("hazard", 0.0), "p_end": round(pe, 3)})

    # The mic reader (below) only appends to buf; this task turns buf into hops. Decoupled so that while a
    # hop is on the GPU the reader keeps draining the socket, and `more` (another hop already buffered, i.e.
    # this conversation is behind) is visible to the batcher, which then waits for its catch-up hop.
    hop_ready = asyncio.Event()

    async def run_hops():
        nonlocal buf
        try:
            while True:
                while len(buf) < HOP_N:
                    hop_ready.clear(); await hop_ready.wait()
                uh = np.ascontiguousarray(buf[:HOP_N]); buf = buf[HOP_N:]
                if st["reset"]:
                    ears.reset(); st["reset"] = False
                if st["speaking"]:
                    ab = st["agent_buf"]; ah = np.zeros(HOP_N, np.float32)
                    n = min(HOP_N, len(ab)); ah[:n] = ab[:n]; st["agent_buf"] = ab[n:]; mode = "speaking"
                else:
                    ah = np.zeros(HOP_N, np.float32); mode = "listen"
                evs = await SERVICE.feed(ears, uh, ah, mode, more=len(buf) >= HOP_N)  # batched across sessions
                await handle_events(evs)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            log.exception("hop loop error: %s", type(e).__name__)
            await ws.close()
    hops = asyncio.create_task(run_hops())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                buf = np.concatenate([buf, np.frombuffer(msg.data, dtype=np.float32)])
                if len(buf) >= HOP_N:
                    hop_ready.set()
            elif msg.type == WSMsgType.TEXT:
                data = msg.data
                if data == "reset":
                    ears.reset(); buf = np.zeros(0, np.float32)
                elif data.startswith("voice:") and have_mouth:
                    name = data.split(":", 1)[1]
                    if name in VOICES:
                        await async_send_msg(m_writer, b"V", {"voice": name})
                        await ws.send_json({"type": "voice", "voice": name})
                elif data.startswith("set:"):
                    try:
                        _, attr, val = data.split(":", 2)
                        if attr in ("barge_thr", "bc_thr", "ep_thr"):      # per conversation (the sliders)
                            setattr(ears, attr, float(val))
                        elif attr in ("ep_deb", "bc_deb", "barge_deb"):
                            setattr(ears, attr, int(val))
                        elif attr == "fallback_frames":
                            ears.FALLBACK_SIL_FRAMES = int(val)
                        cur = getattr(ears, "FALLBACK_SIL_FRAMES") if attr == "fallback_frames" else getattr(ears, attr)
                        await ws.send_json({"type": "threshold", "attr": attr, "value": cur})
                    except (ValueError, AttributeError):
                        pass
    finally:
        hops.cancel()
        if pump:
            pump.cancel()
        if st["done_task"] is not None:
            st["done_task"].cancel()
        if have_mouth:
            try:
                m_writer.close()
            except Exception:
                pass
        log.info("conversation end (active=%d)", SERVICE.active() - 1)
    return ws


def build_service(cfg):
    heads = weights.ensure_heads(cfg["models"]["heads_repo"], cfg["models"]["heads_file"])
    e = cfg["ears"]
    n = int(cfg["server"].get("max_sessions", 4))
    log.info("loading EARS (arm C) for %d sessions, ep_thr=%s ep_deb=%s", n, e["ep_thr"], e["ep_deb"])
    engine = EarsEngine(heads, slots=n, ring_len=int(e.get("ring_frames", 2048)))
    return EarsService(engine, gather_s=float(e.get("gather_ms", 15)) / 1e3,
                       expect_s=float(e.get("expect_ms", 60)) / 1e3,
                       stats_every_s=float(e.get("stats_s", 60)))


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    global SERVICE, CFG, VOICES, DEFAULT_VOICE
    CFG = load_config(args.config)
    import torch
    torch.set_num_threads(int(CFG["ears"].get("cpu_threads", 4)))   # default 48 threads mostly spin-wait
    VOICES, DEFAULT_VOICE = discover_voices(CFG)
    SERVICE = build_service(CFG)
    log.info("=== EARS ready (up to %d conversations) ===", SERVICE.engine.slots)
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/voices", voices_handler), web.get("/ws", ws_handler)])
    host, port = CFG["server"]["host"], int(CFG["server"]["port"])
    log.info("=== http://%s:%d  (mic needs https OR http://localhost via tunnel) ===", host, port)
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
