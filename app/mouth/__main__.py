"""The MOUTH process: load the shared models (Mouth), then serve EARS connections, one Session thread each.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.mouth [--config PATH]
"""
from __future__ import annotations
import os, sys, socket, signal, threading, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from app.config import load_config  # noqa: E402
from app.mouth.service import Mouth  # noqa: E402
from app.mouth.session import Session  # noqa: E402

log = logging.getLogger("mouth")


class Server:
    """Accepts EARS's connections (one per conversation) and runs a Session for each on its own thread; the
    engines in `m` are shared."""
    def __init__(self, m, host, port):
        self.m = m
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port)); self.sock.listen(16)
        self.active = 0; self.lock = threading.Lock()

    def serve_forever(self):
        while True:
            conn, _ = self.sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _count(self, d):
        with self.lock:
            self.active += d
            return self.active

    def _serve(self, conn):
        n = self._count(+1)
        s = Session(self.m, conn)
        log.info("session start (active=%d, voice=%s)", n, s.voice_name)
        try:
            s.run()
        except Exception as e:
            log.exception("session error: %s", type(e).__name__)
        finally:
            conn.close()
            log.info("session end (active=%d)", self._count(-1))


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import torch
    torch.set_num_threads(int(cfg["mouth"].get("cpu_threads", 4)))   # default 48 threads mostly spin-wait
    if not os.environ.get("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set -> web_search/get_weather will return an error")
    m = Mouth(cfg)

    def stop(signum, frame):
        log.info("MOUTH stopping (signal %d)", signum)
        m.shutdown()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    server = Server(m, host, port)
    log.info("=== MOUTH ready (in-proc %s + %s, sr=%d) listening on %s:%d ===",
             cfg["llm"]["reply"]["plugin"], cfg["tts"]["plugin"], m.sr, host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
