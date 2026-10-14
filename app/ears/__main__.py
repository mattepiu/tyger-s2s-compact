"""The EARS process: load the turn-taking plugin, then serve the page and the conversations' WebSockets.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.ears [--config PATH]
"""
from __future__ import annotations
import os, sys, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aiohttp import web  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.access import AccessLog  # noqa: E402
from app.ears.web import EarsApp  # noqa: E402
from app.plugins import build_ears  # noqa: E402

log = logging.getLogger("ears")


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    app = EarsApp(cfg, build_ears(cfg))
    log.info("=== EARS ready (up to %d conversations) ===", int(cfg["server"].get("max_sessions", 4)))
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    log.info("=== http://%s:%d  (mic needs https OR http://localhost via tunnel) ===", host, port)
    web.run_app(app.application(), host=host, port=port, print=None,
                access_log_class=AccessLog)                  # no query strings in the log: a token is one


if __name__ == "__main__":
    main()
