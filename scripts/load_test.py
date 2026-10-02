"""Concurrency load test (no browser). Opens N sessions against the running MOUTH at once; each runs a few
turns and fires a backchannel in the middle of a turn (separate engine requests, generated concurrently).
Reports per-turn first-audio, backchannel first-audio (backchannel audio arrives as K frames, apart from
reply audio A) and peak GPU memory. Test prompts only; no audio is written anywhere.

  .venv/bin/python scripts/load_test.py --sessions 4 [--turns 3]
"""
import os, sys, json, time, socket, argparse, threading, subprocess, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.wire import send_msg, recv_msg
from app.config import load_config

PROMPTS = ["Tell me something interesting about octopuses.", "How do I keep basil alive indoors?",
           "Can you explain what a black hole is?", "What's a good way to start learning to cook?",
           "Give me a tip for sleeping better.", "Why is the sky blue?"]
PARTIALS = ["so yesterday I finally got the job offer I was hoping for",
            "and then my dog got really sick and we had to rush him to the vet",
            "я вот думаю, может, в выходные съездить на море"]


def gpu_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return -1


def session(idx, cfg, turns, out):
    s = socket.create_connection((cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])), timeout=180)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    res = {"turn_fa": [], "bc_fa": [], "errors": 0}
    for t in range(turns):
        send_msg(s, b"T", {"transcript": PROMPTS[(idx + t) % len(PROMPTS)]})
        t0 = time.time(); fa = None; bc_t0 = None; bc_fa = None; turn_done = bc_done = False; bc_sent = False
        while not (turn_done and (bc_done or not bc_sent)):
            kind, p = recv_msg(s)
            if kind is None:
                res["errors"] += 1; break
            if kind == b"A":
                fa = fa or time.time() - t0
                if not bc_sent:                 # mid-turn: ask for a backchannel while the reply streams
                    send_msg(s, b"C", {"partial": PARTIALS[(idx + t) % len(PARTIALS)]})
                    bc_sent = True; bc_t0 = time.time()
            elif kind == b"K":
                bc_fa = bc_fa or time.time() - bc_t0
            elif kind == b"D":
                d = json.loads(p)
                if d.get("error"):
                    res["errors"] += 1
                if d.get("bc"):
                    bc_done = True
                else:
                    turn_done = True
        res["turn_fa"].append(fa or -1)
        if bc_fa is not None:
            res["bc_fa"].append(bc_fa)
    s.close()
    out[idx] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = {}; peak = [gpu_mib()]; stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_mib()); time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    th = [threading.Thread(target=session, args=(i, cfg, a.turns, out)) for i in range(a.sessions)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.set()
    fa = [x for r in out.values() for x in r["turn_fa"] if x > 0]
    bc = [x for r in out.values() for x in r["bc_fa"]]
    err = sum(r["errors"] for r in out.values())
    ms = lambda v: f"p50 {1e3*statistics.median(v):.0f} / max {1e3*max(v):.0f} ms" if v else "n/a"  # noqa: E731
    print(f"sessions={a.sessions} turns/session={a.turns} wall={time.time()-t0:.1f}s errors={err}")
    print(f"  turn first-audio        {ms(fa)}  (n={len(fa)})")
    print(f"  backchannel first-audio {ms(bc)}  (n={len(bc)}, generated while a reply was streaming)")
    print(f"  GPU memory peak         {peak[0]} MiB")
    assert err == 0, "errors during the load test"


if __name__ == "__main__":
    main()
