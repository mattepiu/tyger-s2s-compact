"""EarsApp's HTTP side, GPU-free, over a real aiohttp test server with the fake EARS plugin: the pages (the new page's
build at /, its files, the classic page at /classic, the classic page at / until the new one is built), the busy
page and the voice list.
Run: .venv/bin/python tests/ears/test_web.py"""
import os, sys, asyncio, copy, logging, tempfile
from pathlib import Path

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from yarl import URL  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.web import EarsApp  # noqa: E402
from app.plugins.base import PluginContext  # noqa: E402
from app.plugins.ears.fake import EnergyEars  # noqa: E402

ROOT = os.path.dirname(TESTS)
NEW_PAGE = "<!doctype html><title>the new page</title>"


def app(max_sessions=1, build=None):
    cfg = copy.deepcopy(load_config(os.path.join(ROOT, "config/fake.yaml")))
    cfg["server"]["max_sessions"] = max_sessions
    cfg["server"]["mouth_port"] = 1                             # nothing there: EARS-only conversations
    return EarsApp(cfg, EnergyEars(cfg["ears"], PluginContext(None, cfg, "fake")),
                   build=build or os.path.join(tempfile.mkdtemp(), "out"))


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


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
