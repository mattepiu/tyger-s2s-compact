"""EarsApp's HTTP side, GPU-free, over a real aiohttp test server with the fake EARS plugin: the pages (the new page's
build at /, its files, the classic page at /classic, the classic page at / until the new one is built), the busy
page and the voice list; API mode (no pages), no origin check, the token (401), /voices's CORS headers, the
WebSocket message cap, and no token in the logs. Tokens are signed here as another app's backend would
(tests/fakes.py); the secret is made up.
Run: .venv/bin/python tests/ears/test_web.py"""
import os, sys, asyncio, copy, logging, tempfile, time
from pathlib import Path

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from aiohttp import WSMsgType, WSServerHandshakeError  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from yarl import URL  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.web import EarsApp  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402
from fakes import make_token  # noqa: E402

ROOT = os.path.dirname(TESTS)
NEW_PAGE = "<!doctype html><title>the new page</title>"
SECRET = "web-test-secret-not-a-real-one"
APP_ORIGIN = "https://app.example.com"


def app(max_sessions=1, build=None, secret="", **server):
    """The app on the fake config; `server`: server.* settings; `secret` "" means no token check."""
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
    cfg["server"].update(max_sessions=max_sessions, mouth_port=1, **server)   # no MOUTH: EARS-only conversations
    return EarsApp(cfg, EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake")),
                   build=build or os.path.join(tempfile.mkdtemp(), "out"), secret=secret)


def stub_build(tmp):
    """A build as `npm run build` lays it out, in tmp/out, and a file beside it that must never be served."""
    out = Path(tmp) / "out"
    (out / "_next/static/chunks").mkdir(parents=True)
    (out / "index.html").write_text(NEW_PAGE)
    (out / "index.txt").write_text("rsc payload")
    (out / "_next/static/chunks/main-1a2b.js").write_text("console.log(1)")
    (Path(tmp) / "secret.txt").write_text("not part of the build")
    return out


def run(ears, fn):
    async def go():
        async with TestClient(TestServer(ears.application())) as client:
            await fn(client)
    asyncio.run(go())


async def connects(client, path="/ws", origin=None):
    """The status of a WebSocket handshake: 101 and `ready` on success."""
    try:
        ws = await client.ws_connect(path, origin=origin)
    except WSServerHandshakeError as e:
        return e.status
    msg = await ws.receive_json()
    await ws.close()
    assert msg["type"] == "ready", msg
    return 101


async def text(resp):
    return resp.status, (await resp.text()) if resp.status == 200 else ""


def test_new_page_at_root_and_its_files():
    with tempfile.TemporaryDirectory() as tmp:
        build = stub_build(tmp)

        async def fn(client):
            r = await client.get("/")
            assert await text(r) == (200, NEW_PAGE) and r.headers["Cache-Control"] == "no-cache"
            r = await client.get("/_next/static/chunks/main-1a2b.js")
            assert await text(r) == (200, "console.log(1)")
            assert "immutable" in r.headers["Cache-Control"] and "javascript" in r.headers["Content-Type"]
            r = await client.get("/index.txt")                              # every exported file, not just _next/
            assert await text(r) == (200, "rsc payload") and r.headers["Cache-Control"] == "no-cache"
            assert (await client.get("/_next/static/chunks/nothing.js")).status == 404
        run(app(build=build), fn)


def test_nothing_outside_the_build():
    """Paths that climb out of web/out/ (raw, percent-encoded, through a subfolder) are 404s, whatever is there."""
    with tempfile.TemporaryDirectory() as tmp:
        build = stub_build(tmp)

        async def fn(client):
            for raw in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/_next/../../secret.txt",
                        "/_next/static/%2e%2e/%2e%2e/%2e%2e/secret.txt", "/%2fetc%2fpasswd"):
                r = await client.get(URL(raw, encoded=True))
                body = await r.text()
                assert r.status == 404 and "not part of the build" not in body and "root:" not in body, (raw, r.status)
        run(app(build=build), fn)


def test_classic_page_at_classic():
    with tempfile.TemporaryDirectory() as tmp:
        build = stub_build(tmp)

        async def fn(client):
            status, body = await text(await client.get("/classic"))
            assert status == 200 and "Start listening" in body and "<title>tyger</title>" in body
        run(app(build=build), fn)


def test_classic_page_at_root_until_the_new_one_is_built():
    """No build: / serves the classic page, and EARS says once how to build the new one. A build made after
    startup is served at once."""
    records = []
    handler = logging.Handler(); handler.emit = records.append
    logging.getLogger("ears").addHandler(handler)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ears = app(build=Path(tmp) / "out")

            async def fn(client):
                for _ in range(2):
                    status, body = await text(await client.get("/"))
                    assert status == 200 and "Start listening" in body
                stub_build(tmp)
                assert await text(await client.get("/")) == (200, NEW_PAGE)
            run(ears, fn)
    finally:
        logging.getLogger("ears").removeHandler(handler)
    told = [r.getMessage() for r in records if "isn't built" in r.getMessage()]
    assert len(told) == 1 and "npm run build" in told[0], told


def test_page_busy_page_and_voices():
    async def fn(client):
        assert (await client.get("/")).status == 200
        v = await (await client.get("/voices")).json()
        assert v == {"voices": ["fake"], "voice": "fake", "labels": {"fake": "Mira (fake)"}}   # the picker
        ws1 = await client.ws_connect("/ws")
        ready = await ws1.receive_json()
        assert ready["type"] == "ready" and ready["mouth"] is False and ready["hop_ms"] == 320
        assert ready["voices"] == v["voices"] and ready["labels"] == v["labels"]
        assert [x["name"] for x in ready["sliders"]] == ["endpoint_silence_frames"]   # the plugin's own spec
        assert ready["thresholds"] == {"endpoint_silence_frames": 6}
        assert (await client.get("/")).status == 503                    # the one slot is taken
        assert (await client.get("/classic")).status == 503
        ws2 = await client.ws_connect("/ws")
        assert (await ws2.receive_json()) == {"type": "busy"}
        await ws1.close()
        for _ in range(50):                                             # the slot frees when it ends
            if not ears.ears.full():
                break
            await asyncio.sleep(0.02)
        assert (await client.get("/")).status == 200
    ears = app(max_sessions=1)
    run(ears, fn)


def test_api_mode_serves_no_pages():
    """server.pages off: only /ws and /voices answer."""
    with tempfile.TemporaryDirectory() as tmp:
        build = stub_build(tmp)

        async def fn(client):
            for path in ("/", "/classic", "/index.txt", "/_next/static/chunks/main-1a2b.js"):
                assert (await client.get(path)).status == 404, path
            assert (await client.get("/voices")).status == 200
            assert await connects(client) == 101
        run(app(build=build, pages=False), fn)


def test_no_origin_check():
    """Any page connects, whatever its origin (the token, when there's a secret, is the check)."""
    async def fn(client):
        for origin in (APP_ORIGIN, "https://anywhere.example", "null", None):
            assert await connects(client, origin=origin) == 101, origin
    run(app(), fn)


def test_voices_cors_headers():
    """/voices answers CORS for any page (with a secret, a token too: test_tokens)."""
    async def fn(client):
        for origin in (APP_ORIGIN, "https://anywhere.example"):
            r = await client.get("/voices", headers={"Origin": origin})
            assert r.status == 200 and r.headers["Access-Control-Allow-Origin"] == "*", origin
    run(app(), fn)


def test_message_size_cap():
    """A WebSocket message over 64 KB closes the socket (1009); one under it is read as usual."""
    async def fn(client):
        ws = await client.ws_connect("/ws")
        assert (await ws.receive_json())["type"] == "ready"
        await ws.send_bytes(bytes(8) + bytes(60 * 1024))           # a 1.9 s mic block: fine
        await ws.send_bytes(bytes(8) + bytes(70 * 1024))
        while (await ws.receive(timeout=5)).type not in (WSMsgType.CLOSE, WSMsgType.CLOSED):
            pass
        assert ws.close_code == 1009, ws.close_code
    run(app(), fn)


def test_tokens():
    """With the secret: a valid token opens a conversation, once; every refused kind is a 401 before the upgrade;
    /voices needs a token too, and checking it there doesn't use it up."""
    now = time.time()
    bad = {"missing": None, "expired": make_token(SECRET, exp=now - 5), "bad signature": make_token("another", now),
           "wrong audience": make_token(SECRET, aud="else"), "alg none": make_token(SECRET, header={"alg": "none"},
                                                                                   sign=False),
           "exp too far ahead": make_token(SECRET, exp=now + 3600),
           "origin mismatch": make_token(SECRET, origin="https://other.example.com")}

    async def fn(client):
        token = make_token(SECRET)
        assert (await client.get("/voices")).status == 401
        r = await client.get("/voices", params={"token": token}, headers={"Origin": APP_ORIGIN})
        assert r.status == 200 and r.headers["Access-Control-Allow-Origin"] == "*"
        assert await connects(client, f"/ws?token={token}", origin=APP_ORIGIN) == 101    # /voices didn't use it
        assert await connects(client, f"/ws?token={token}", origin=APP_ORIGIN) == 401    # used: one conversation
        for name, t in bad.items():
            assert await connects(client, "/ws" + (f"?token={t}" if t else ""), origin=APP_ORIGIN) == 401, name
        r = await client.get("/voices", params={"token": bad["expired"]}, headers={"Origin": APP_ORIGIN})
        assert r.status == 401 and r.headers["Access-Control-Allow-Origin"] == "*"    # readable by the app
        assert await connects(client, f"/ws?token={make_token(SECRET, origin=APP_ORIGIN)}", origin=APP_ORIGIN) == 101
        assert await connects(client, f"/ws?token={make_token(SECRET)}", origin="https://anywhere.example") == 101
    run(app(secret=SECRET), fn)


def test_no_secret_no_token_check():
    async def fn(client):
        assert (await client.get("/voices")).status == 200
        assert await connects(client) == 101
    os.environ.pop("TYGER_API_SECRET", None)
    run(app(secret=None), fn)                                       # None: read from the env, which has none


def test_secret_from_the_env():
    async def fn(client):
        assert await connects(client) == 401
        assert await connects(client, f"/ws?token={make_token(SECRET)}") == 101
    os.environ["TYGER_API_SECRET"] = SECRET
    try:
        run(app(secret=None), fn)
    finally:
        del os.environ["TYGER_API_SECRET"]


def test_no_token_in_the_logs():
    """EARS itself (python -m app.ears on the fake config, the secret in its env): its request log has paths without
    query strings, and nothing it logs holds the secret or a token, valid or refused."""
    import socket, subprocess, yaml
    from aiohttp import ClientSession
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    good, expired = make_token(SECRET), make_token(SECRET, exp=time.time() - 5)

    async def requests(base):
        async with ClientSession() as http:
            assert (await http.get(f"{base}/voices", params={"token": good})).status == 200
            for token, status in ((good, 101), (good, 401), (expired, 401)):
                try:
                    ws = await http.ws_connect(f"{base}/ws?token={token}")
                    assert (await ws.receive_json())["type"] == "ready"
                    await ws.close(); got = 101
                except WSServerHandshakeError as e:
                    got = e.status
                assert got == status, (got, status)
            await asyncio.sleep(0.3)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = yaml.safe_load(open(os.path.join(ROOT, "config/fake.yaml")))
        cfg["server"].update(port=port, mouth_port=1, pages=False)
        conf = os.path.join(tmp, "fake.yaml"); yaml.safe_dump(cfg, open(conf, "w"))
        ears = subprocess.Popen([sys.executable, "-m", "app.ears", "--config", conf], cwd=ROOT,
                                env={**os.environ, "TYGER_API_SECRET": SECRET},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            log = []
            for line in ears.stdout:                                 # until it's up, then until it listens
                log.append(line)
                if "EARS ready" in line:
                    break
            for _ in range(100):
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.05)
            asyncio.run(requests(f"http://127.0.0.1:{port}"))
        finally:
            ears.terminate()
            log.append(ears.communicate(timeout=10)[0])
    log = "".join(log)
    assert '"GET /voices" 200' in log and '"GET /ws" 101' in log and '"GET /ws" 401' in log, log[-1500:]
    assert "token used" in log and "token expired" in log, log[-1500:]
    for secret in (SECRET, good, expired) + tuple(good.split(".")) + tuple(expired.split(".")):
        assert secret not in log, secret[:12]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
