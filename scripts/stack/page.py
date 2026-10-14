"""The page, simulated, for the stack scripts: what static/index.html does on EARS's WebSocket (I-17), without a
browser or audio devices. Reply audio "plays" on the wall clock as it arrives, back to back (LEAD_IN_S after it
arrives when nothing is queued; `downlink_s` later still, for a slow network); each mic block is sent as 20 ms
blocks tagged with the reply sample that was playing when the block was captured; when a reply's queue drains the
page reports `played:<id>:<samples>`; a barge or an endpoint stops playback. Backchannel audio (id 0) plays but is
never tagged.

`open_ws` opens the WebSocket as the page does. When the server checks tokens (TYGER_API_SECRET in this env too, or in
the repo's .env, which load_config reads), each conversation gets its own token, signed as an app's backend signs one
(README -> "API mode"). The secret, the tokens and the URLs that hold them are never printed."""
import asyncio, base64, contextlib, hashlib, hmac, json, secrets, time

from app.ears.access import AUDIENCE, secret_from_env
from app.ears.playback import AUDIO_HEADER, BACKCHANNEL_ID, LEAD_IN_S, MIC_HEADER, to_pcm16

BLOCK = 320                      # 20 ms at 16 kHz, as the page's worklet sends them
TOKEN_TTL_S = 60                 # a token's exp, after it's signed (EARS takes up to 300 s)


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_token(secret, ttl_s=TOKEN_TTL_S):
    """A token for one conversation: HS256, aud tyger, exp `ttl_s` ahead, a fresh jti. Never print it."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    claims = _b64url(json.dumps({"aud": AUDIENCE, "iat": now, "exp": now + ttl_s,
                                 "jti": secrets.token_hex(16)}).encode())
    sig = hmac.new(secret.encode(), f"{header}.{claims}".encode("ascii"), hashlib.sha256).digest()
    return f"{header}.{claims}.{_b64url(sig)}"


@contextlib.asynccontextmanager
async def open_ws(sess, port):
    """EARS's WebSocket on this box (an aiohttp ClientSession's), with a token when the server checks them. A refused
    handshake raises without the URL: aiohttp's error names it, token included."""
    import aiohttp
    secret = secret_from_env()
    url = f"http://127.0.0.1:{port}/ws" + (f"?token={sign_token(secret)}" if secret else "")
    try:
        ws = await sess.ws_connect(url, max_msg_size=0)
    except aiohttp.ClientResponseError as e:
        hint = " (no token, or one EARS refused: is TYGER_API_SECRET the server's?)" if e.status == 401 else ""
        raise RuntimeError(f"EARS refused the WebSocket: HTTP {e.status}{hint}") from None
    async with ws:
        yield ws


class PageSim:
    def __init__(self, ws, downlink_s=0.0):
        self.ws = ws; self.downlink_s = downlink_s
        self.sched = []                  # (reply id, t0, t1, its first sample): queued chunks, perf_counter time
        self.next_play = 0.0; self.queued = {}; self.last_id = 0
        self.ends = {}                   # reply id -> when its queued audio ends playing
        self.reports = set()

    def tag(self, t):
        """(reply id, its sample at 24 kHz) playing at time t; (the last id, -1) when nothing is."""
        for rid, t0, t1, first in reversed(self.sched):
            if t0 <= t < t1:
                return rid, first + int((t - t0) * 24000)
        return self.last_id, -1

    async def send_mic(self, x, t_capture):
        """Send 16 kHz float32 mic audio captured from `t_capture` on (perf_counter time), as tagged blocks."""
        for i in range(0, len(x), BLOCK):
            rid, at = self.tag(t_capture + i / 16000)
            await self.ws.send_bytes(MIC_HEADER.pack(rid, at) + to_pcm16(x[i:i + BLOCK]).tobytes())

    def on_audio(self, data):
        """An audio frame from EARS, queued for playback. Returns (id, samples at 24 kHz)."""
        rid = AUDIO_HEADER.unpack_from(data)[0]; n = (len(data) - AUDIO_HEADER.size) // 2
        now = time.perf_counter()
        if self.next_play < now + self.downlink_s:
            self.next_play = now + self.downlink_s + LEAD_IN_S
        t0 = self.next_play; self.next_play = t1 = t0 + n / 24000
        if rid != BACKCHANNEL_ID:
            first = self.queued.get(rid, 0); self.queued[rid] = first + n; self.last_id = rid
            self.sched = [c for c in self.sched if c[2] > now - 5] + [(rid, t0, t1, first)]
            self.ends[rid] = t1
            task = asyncio.ensure_future(self._report(rid, t1))
            self.reports.add(task); task.add_done_callback(self.reports.discard)
        return rid, n

    async def _report(self, rid, end):
        await asyncio.sleep(max(0.0, end - time.perf_counter()))
        if self.next_play <= end + 1e-6:               # nothing queued after it: the reply has played out
            await self.ws.send_str(f"played:{rid}:{self.queued[rid]}")

    def stop(self):
        """A barge or an endpoint: playback stops, nothing more is tagged or reported."""
        self.sched = []; self.next_play = 0.0
        for task in list(self.reports):
            task.cancel()
