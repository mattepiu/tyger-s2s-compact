"""EarsApp: the web app on one port. The pages, the voice list (each voice labelled with its character), and a
WebSocket per conversation, each a Conversation on the shared EARS plugin. Visitors beyond the plugin's capacity
(server.max_sessions) get a "busy" page.

The pages: `/` is the new page, the static build of web/ (`npm run build` exports it to web/out/; run.sh builds it),
with every other file of that build at its own path (/_next/... included). `/classic` is the classic page
(static/index.html). Until the new page is built, `/` serves the classic page too. With server.pages off (API mode,
TYGER_PAGES=false), EARS answers only /ws and /voices.

With TYGER_API_SECRET set, a conversation needs a token (app/ears/access.py), checked before the WebSocket upgrade
(401). /voices needs one too (checking it there doesn't use it up), and answers CORS for any page. There is no origin
check: without the secret, anyone can connect."""
from __future__ import annotations
import logging
from pathlib import Path

from aiohttp import web

from app.config import ROOT
from app.ears.access import SECRET_ENV, TokenRefused, Tokens, secret_from_env
from app.ears.conversation import MAX_MIC_BUFFER_S, Conversation
from app.plugins import plugin_class
from app.prompts import Characters, load_prompts

log = logging.getLogger("ears")
CLASSIC = ROOT / "static" / "index.html"
BUILD = ROOT / "web" / "out"                     # the new page's static export
NO_CACHE = {"Cache-Control": "no-cache"}         # the pages: always the current build
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}   # /_next/static/: content-hashed names
MAX_MSG_BYTES = 64 * 1024                        # a WebSocket message's cap (a 20 ms mic block is 648 bytes)
CORS = {"Access-Control-Allow-Origin": "*"}      # /voices, for another app's page (the token, if any, is in the URL)
BUSY_HTML = ("<!doctype html><meta charset=utf-8><title>tyger - busy</title>"
             "<body style='font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
             "<h1>tyger is busy</h1><p>All conversation slots are in use right now "
             "&mdash; please try again shortly.</p></body>")


def list_voices(cfg):
    """(names, default) from the configured TTS plugin's class: a light import, no model loaded (EARS never builds
    the TTS)."""
    return plugin_class("tts", cfg["tts"]["plugin"]).list_voices(cfg)


class EarsApp:
    def __init__(self, cfg, ears, build=BUILD, secret=None, wakeword_config=None):
        """`secret`: the token's secret; None reads TYGER_API_SECRET (unset or empty: no token check)."""
        self.cfg = cfg; self.ears = ears              # ears: the EARS plugin (app/plugins/base.py EarsPlugin)
        self.wakeword_cfg = wakeword_config           # optional WakeWordConfig
        self.characters = Characters(cfg, load_prompts(cfg))      # the picker's labels: "Mira (mira)"
        self.voices, self.default_voice = list_voices(cfg)
        srv = cfg["server"]
        self.pages = srv["pages"]
        secret = secret_from_env() if secret is None else secret
        self.tokens = Tokens(secret) if secret else None
        self.build = Path(build).resolve()
        self.told_unbuilt = False
        limit = float(srv["max_session_s"])
        log.info("settings: pages %s; conversations end after %s; token check %s; messages up to %d KB, mic audio "
                 "buffered up to %.0f s", "on" if self.pages else "off (API mode: /ws and /voices only)",
                 f"{limit:g} s" if limit else "no limit",
                 f"on ({SECRET_ENV} set)" if self.tokens else f"off ({SECRET_ENV} unset: anyone can connect)",
                 MAX_MSG_BYTES // 1024, MAX_MIC_BUFFER_S)
        if self.pages:
            if self.tokens:
                log.warning("%s is set and the pages are on: the pages EARS serves have no token, so they can't open "
                            "a conversation. Run with TYGER_PAGES=false (server.pages: false)", SECRET_ENV)
            self.new_page()                           # says so at startup when it isn't built

    def labels(self):
        """{voice: its picker entry, the character's name then the voice}."""
        return {v: self.characters.label(v) for v in self.voices}

    def application(self):
        app = web.Application()
        app.add_routes([web.get("/voices", self.voice_list), web.get("/ws", self.websocket)])
        if self.pages:
            app.add_routes([web.get("/", self.index), web.get("/classic", self.classic),
                            web.get("/{path:.+}", self.asset)])       # last: the new page's other files
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

    def refused(self, request, use, headers=None):
        """The 401 that refuses this request's token, or None when it may go on (always, without a secret). The
        reason is logged, never the token."""
        if self.tokens is not None:
            try:
                self.tokens.check(request.query.get("token"), request.headers.get("Origin"), use)
            except TokenRefused as e:
                log.warning("refused %s: token %s", request.path, e.reason)
                return web.Response(status=401, text=f"token refused: {e.reason}", headers=headers)
        return None

    async def voice_list(self, request):
        """The voice picker's list, so a pick before Start is remembered. Re-scans the voices dir: clips added after
        startup show up on a refresh, no restart needed."""
        no = self.refused(request, use=False, headers=CORS)   # the app can read why
        if no is not None:
            return no
        try:
            self.voices, self.default_voice = list_voices(self.cfg)
        except Exception as e:
            log.warning("voice re-discovery failed: %s", e)
        return web.json_response({"voices": self.voices, "voice": self.default_voice, "labels": self.labels()},
                                 headers=CORS)

    async def websocket(self, request):
        no = self.refused(request, use=True)          # before the upgrade: one token opens one conversation
        if no is not None:
            return no
        ws = web.WebSocketResponse(max_msg_size=MAX_MSG_BYTES)   # a bigger message closes the socket (1009)
        await ws.prepare(request)
        session = self.ears.open()               # this conversation's plugin session (+ its own thresholds)
        if session is None:
            await ws.send_json({"type": "busy"})
            await ws.close()
            return ws
        ww_session = self.wakeword_cfg.create_session() if (self.wakeword_cfg and self.wakeword_cfg.enabled) else None
        try:
            await Conversation(ws, self.ears, session, self.cfg, self.voices, self.default_voice,
                               self.labels(), wakeword=ww_session).run()
        finally:
            self.ears.close(session)
        return ws
