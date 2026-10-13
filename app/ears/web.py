"""EarsApp: the web app on one port. The pages, the voice list (each voice labelled with its character), and a
WebSocket per conversation, each a Conversation on the shared EARS plugin. Visitors beyond the plugin's capacity
(server.max_sessions) get a "busy" page.

The pages: `/` is the new page, the static build of web/ (`npm run build` exports it to web/out/; run.sh builds it),
with every other file of that build at its own path (/_next/... included). `/classic` is the classic page
(static/index.html). Until the new page is built, `/` serves the classic page too."""
from __future__ import annotations
import logging
from pathlib import Path

from aiohttp import web

from app.config import ROOT
from app.ears.conversation import Conversation
from app.plugins import plugin_class
from app.prompts import Characters, load_prompts

log = logging.getLogger("ears")
CLASSIC = ROOT / "static" / "index.html"
BUILD = ROOT / "web" / "out"                     # the new page's static export
NO_CACHE = {"Cache-Control": "no-cache"}         # the pages: always the current build
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}   # /_next/static/: content-hashed names
BUSY_HTML =("<!doctype html><meta charset=utf-8><title>tyger - busy</title>"
             "<body style='font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
             "<h1>tyger is busy</h1><p>All conversation slots are in use right now "
             "&mdash; please try again shortly.</p></body>")


def list_voices(cfg):
    """(names, default) from the configured TTS plugin's class: a light import, no model loaded (EARS never builds
    the TTS)."""
    return plugin_class("tts", cfg["tts"]["plugin"]).list_voices(cfg)


class EarsApp:
    def __init__(self, cfg, ears, build=BUILD):
        self.cfg = cfg; self.ears = ears              # ears: the EARS plugin (app/plugins/base.py EarsPlugin)
        self.characters = Characters(cfg, load_prompts(cfg))      # the picker's labels: "Mira (mira)"
        self.voices, self.default_voice = list_voices(cfg)
        self.build = Path(build).resolve()
        self.told_unbuilt = False
        self.new_page()                               # says so at startup when it isn't built

    def labels(self):
        """{voice: its picker entry, the character's name then the voice}."""
        return {v: self.characters.label(v) for v in self.voices}

    def application(self):
        app = web.Application()
        app.add_routes([web.get("/", self.index), web.get("/classic", self.classic),
                        web.get("/voices", self.voice_list), web.get("/ws", self.websocket),
                        web.get("/{path:.+}", self.asset)])           # last: the new page's other files
        return app

    def new_page(self):
        """The new page's index.html, or None while it isn't built (logged once, with how to build it). Looked up
        on each visit, so a build made after startup is served without a restart."""
        index = self.build / "index.html"
        if index.is_file():
            return index
        if not self.told_unbuilt:
            self.told_unbuilt = True
            log.warning("the new page isn't built (no %s): / serves the classic page too. Build it with "
                        "`cd web && npm ci && npm run build`, or let run.sh do it (it needs npm)", index)
        return None

    def page(self, path):
        if self.ears.full():
            return web.Response(text=BUSY_HTML, content_type="text/html", status=503)
        return web.FileResponse(path, headers=NO_CACHE)

    async def index(self, request):
        return self.page(self.new_page() or CLASSIC)

    async def classic(self, request):
        return self.page(CLASSIC)

    async def asset(self, request):
        """A file of the new page's build, by its path in web/out/; nothing outside it."""
        rel = request.match_info["path"]
        f = (self.build / rel).resolve()
        if not f.is_relative_to(self.build) or not f.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(f, headers=IMMUTABLE if rel.startswith("_next/static/") else NO_CACHE)

    async def voice_list(self, request):
        """The voice picker's list, so a pick before Start is remembered. Re-scans the voices dir: clips added after
        startup show up on a refresh, no restart needed."""
        try:
            self.voices, self.default_voice = list_voices(self.cfg)
        except Exception as e:
            log.warning("voice re-discovery failed: %s", e)
        return web.json_response({"voices": self.voices, "voice": self.default_voice, "labels": self.labels()})

    async def websocket(self, request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        session = self.ears.open()               # this conversation's plugin session (+ its own thresholds)
        if session is None:
            await ws.send_json({"type": "busy"})
            await ws.close()
            return ws
        try:
            await Conversation(ws, self.ears, session, self.cfg, self.voices, self.default_voice,
                               self.labels()).run()
        finally:
            self.ears.close(session)
        return ws
