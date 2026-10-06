"""EarsApp: the web app on one port. The page, the voice list, and a WebSocket per conversation, each a Conversation
on the shared EARS plugin. Visitors beyond the plugin's capacity (server.max_sessions) get a "busy" page."""
from __future__ import annotations
import logging
from pathlib import Path

from aiohttp import web

from app.ears.conversation import Conversation
from app.plugins import plugin_class

log = logging.getLogger("ears")
STATIC = Path(__file__).resolve().parents[2] / "static"
BUSY_HTML = ("<!doctype html><meta charset=utf-8><title>tyger - busy</title>"
             "<body style='font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
             "<h1>tyger is busy</h1><p>All conversation slots are in use right now "
             "&mdash; please try again shortly.</p></body>")


def list_voices(cfg):
    """(names, default) from the configured TTS plugin's class: a light import, no model loaded (EARS never builds
    the TTS)."""
    return plugin_class("tts", cfg["tts"]["plugin"]).list_voices(cfg)


class EarsApp:
    def __init__(self, cfg, ears):
        self.cfg = cfg; self.ears = ears              # ears: the EARS plugin (app/plugins/base.py EarsPlugin)
        self.voices, self.default_voice = list_voices(cfg)

    def application(self):
        app = web.Application()
        app.add_routes([web.get("/", self.index), web.get("/voices", self.voice_list), web.get("/ws", self.websocket)])
        return app

    async def index(self, request):
        if self.ears.full():
            return web.Response(text=BUSY_HTML, content_type="text/html", status=503)
        return web.FileResponse(STATIC / "index.html")

    async def voice_list(self, request):
        """The voice picker's list, so a pick before Start is remembered. Re-scans the voices dir: clips added after
        startup show up on a refresh, no restart needed."""
        try:
            self.voices, self.default_voice = list_voices(self.cfg)
        except Exception as e:
            log.warning("voice re-discovery failed: %s", e)
        return web.json_response({"voices": self.voices, "voice": self.default_voice})

    async def websocket(self, request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        session = self.ears.open()               # this conversation's plugin session (+ its own thresholds)
        if session is None:
            await ws.send_json({"type": "busy"})
            await ws.close()
            return ws
        try:
            await Conversation(ws, self.ears, session, self.cfg, self.voices, self.default_voice).run()
        finally:
            self.ears.close(session)
        return ws
