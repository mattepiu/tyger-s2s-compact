"""The `openai` TTS plugin without a network (no GPU, no key, no cost): PCM decoding with odd byte splits, preset
voices only, the request it sends, stop closes the response, errors become TTSError, and an off-rate endpoint is
resampled to the wire. Run: .venv/bin/python tests/test_hosted_tts.py"""
import asyncio, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.plugins.base import PluginContext, TTSError  # noqa: E402
from app.plugins.tts import AtWireRate  # noqa: E402
from app.plugins.tts.openai_compat import OpenAITTS, PCM16Decoder  # noqa: E402
from app.engines import EngineLoop  # noqa: E402

CFG = {"tts": {"voices": ["alloy", "verse"], "default_voice": "verse"}, "voices": {"dir": "voices"}}


def pcm(seconds, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def test_pcm_decoder_carries_odd_bytes():
    raw = pcm(0.1); d = PCM16Decoder(); out = []
    for i in range(0, len(raw), 333):                              # odd-sized reads
        out.append(d.feed(raw[i:i + 333]))
    a = np.concatenate(out)
    assert a.dtype == np.float32 and len(a) == len(raw) // 2 and d.carry == b""
    assert np.allclose(a, np.frombuffer(raw, "<i2") / 32768.0)


class StubResp:
    def __init__(self, data, chunk=1001, delay=0.0):
        self.data, self.chunk, self.delay, self.closed = data, chunk, delay, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        self.closed = True

    async def iter_bytes(self):
        for i in range(0, len(self.data), self.chunk):
            await asyncio.sleep(self.delay)
            yield self.data[i:i + self.chunk]


def make(**conf):
    os.environ["TYGER_TEST_KEY"] = "test-not-a-key"
    cfg = {**CFG, "tts": {**CFG["tts"], **conf.pop("tts", {})}}
    return OpenAITTS({"model": "gpt-4o-mini-tts", "api_key_env": "TYGER_TEST_KEY", "base_url": "http://127.0.0.1:9",
                      **conf}, PluginContext(EngineLoop(), cfg, "openai-tts"))


def test_presets_only_never_clips():
    t = make()
    assert t.list_voices(CFG) == (["alloy", "verse"], "verse") and t.caps.cloning is False and t.remote
    assert t.load_voice("alloy") == "alloy" and t.load_voice("en") is None       # a voices/ clip name: never


def test_streams_pcm_and_sends_options():
    t = make(instructions="Speak warmly.", speed=1.1); sent = []

    def create(**kw):
        sent.append(kw); return StubResp(pcm(0.5))
    t.client.audio.speech.with_streaming_response.create = create
    out = list(t.stream("Hello there, this is a test.", None))
    a = np.concatenate(out)
    assert abs(len(a) - 12000) <= 1 and a.dtype == np.float32
    kw = sent[0]
    assert kw["voice"] == "verse" and kw["response_format"] == "pcm" and kw["model"] == "gpt-4o-mini-tts"
    assert kw["instructions"] == "Speak warmly." and kw["speed"] == 1.1


def test_stop_closes_response():
    t = make(); resps = []

    def create(**kw):
        resps.append(StubResp(pcm(5), chunk=2400, delay=0.01)); return resps[-1]
    t.client.audio.speech.with_streaming_response.create = create
    n = 0
    for _ in t.stream("A long sentence.", "alloy", stop_flag=lambda: n >= 2):
        n += 1
    time.sleep(0.1)
    assert n <= 3 and resps[0].closed


def test_errors_become_ttserror():
    import httpx, openai
    req = httpx.Request("POST", "http://127.0.0.1:9/audio/speech")
    for exc, kind in ((openai.RateLimitError("slow", response=httpx.Response(429, request=req), body=None), "rate_limit"),
                      (openai.APIConnectionError(request=req), "connection")):
        t = make()

        class Failing:
            async def __aenter__(self, exc=exc):
                raise exc

            async def __aexit__(self, *a):
                pass
        t.client.audio.speech.with_streaming_response.create = lambda **kw: Failing()
        try:
            list(t.stream("hi", "alloy"))
        except TTSError as e:
            assert e.kind == kind
        else:
            raise AssertionError("no TTSError")


def test_off_rate_endpoint_resampled():
    t = make(sample_rate=22050)
    t.client.audio.speech.with_streaming_response.create = lambda **kw: StubResp(pcm(1.0, 22050))
    w = AtWireRate(t)
    a = np.concatenate(list(w.stream("hi", "alloy")))
    assert w.out_sr == 24000 and abs(len(a) - 24000) <= 2


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
