"""openai: any OpenAI-compatible /v1/audio/speech endpoint (OpenAI's gpt-4o-mini-tts / tts-1, or a compatible server).

HOSTED: the reply text goes to the endpoint in `base_url`. Allowed only because the config names this plugin. Preset
voices only (`voices` in config): caps.cloning is False and clips in voices/ are NEVER uploaded (a reference clip
is personal data; an endpoint that takes one would need an explicit opt-in this plugin doesn't have).

  * One HTTP request per text chunk, like VoxCPM2: audio.speech.with_streaming_response.create(model, voice, input,
    response_format="pcm", [instructions], [speed]), read as it streams. One client per instance (connections
    reused); `max_concurrency` (default 4) requests in flight.
  * PCM: 16-bit signed little-endian mono at `sample_rate` (OpenAI: 24 kHz); an odd trailing byte is carried to the
    next read. MOUTH resamples to the 24 kHz wire if `sample_rate` differs (AtWireRate).
  * Abort: leaving the response (stop_flag) closes it. A failed request raises TTSError.
  * `min_first_chars` (optional): a hint for MOUTH's chunker if per-request overhead dominates first audio.
"""
from __future__ import annotations
import asyncio, logging, os

import numpy as np

from app.plugins.base import TTSCaps, TTSError

log = logging.getLogger("tts")


class PCM16Decoder:
    """Byte chunks of 16-bit little-endian PCM -> float32 arrays, carrying an odd byte to the next chunk."""
    def __init__(self):
        self.carry = b""

    def feed(self, data: bytes) -> np.ndarray:
        data = self.carry + data
        n = len(data) - len(data) % 2
        self.carry = data[n:]
        return np.frombuffer(data[:n], "<i2").astype(np.float32) / 32768.0


class OpenAITTS:
    remote = True

    @classmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]:
        t = cfg.get("tts", {})
        names = [str(v) for v in t.get("voices") or []]
        default = t.get("default_voice")
        return names, (default if default in names else (names[0] if names else None))

    def __init__(self, conf: dict, ctx):
        from openai import AsyncOpenAI
        import httpx
        self.conf = conf; self.name = ctx.name; self.eloop = ctx.eloop; self.cfg = ctx.cfg
        self.model = conf["model"]
        self.out_sr = int(conf.get("sample_rate", 24000))
        key_env = conf.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"tts {self.name}: {key_env} is not set (put it in .env)")
        if (os.environ.get("OPENAI_LOG") or "").lower() == "debug":
            raise RuntimeError("OPENAI_LOG=debug would log request bodies (reply text): unset it")
        names, _ = self.list_voices(ctx.cfg)
        if not names:
            raise RuntimeError(f"tts {self.name}: list the preset voices in tts.voices")
        self.extra = {k: conf[k] for k in ("instructions", "speed") if k in conf}
        n = int(conf.get("max_concurrency", 4))
        self.caps = TTSCaps(cloning=False, max_batch=n, min_first_chars=conf.get("min_first_chars"))
        timeout = httpx.Timeout(float(conf.get("read_timeout_s", 10.0)), connect=float(conf.get("connect_timeout_s", 2.0)))

        async def make():
            return AsyncOpenAI(api_key=key, base_url=conf.get("base_url"), timeout=timeout,
                               max_retries=int(conf.get("max_retries", 1))), asyncio.Semaphore(n)
        self.client, self.sem = self.eloop.run(make())
        self.wait_s = float(conf.get("queue_timeout_s", 2.0))

    def load_voice(self, name):
        """A preset voice name, or None if this endpoint doesn't offer it. Never a clip from voices/."""
        return name if name in self.list_voices(self.cfg)[0] else None

    def stream(self, text, voice, stop_flag=None):
        voice = voice or self.list_voices(self.cfg)[1]
        kw = dict(model=self.model, voice=voice, input=text, response_format="pcm", **self.extra)

        async def chunks():
            import openai
            try:
                await asyncio.wait_for(self.sem.acquire(), self.wait_s)
            except asyncio.TimeoutError:
                raise TTSError("busy") from None
            try:
                async with self.client.audio.speech.with_streaming_response.create(**kw) as resp:
                    async for b in resp.iter_bytes():           # leaving the block closes the response
                        yield b
            except openai.RateLimitError:
                raise TTSError("rate_limit") from None
            except openai.APIStatusError as e:
                log.warning("tts %s: HTTP %d", self.name, e.status_code)
                raise TTSError("server" if e.status_code >= 500 else "request") from None
            except openai.APITimeoutError:
                raise TTSError("timeout") from None
            except openai.APIConnectionError:
                raise TTSError("connection") from None
            finally:
                self.sem.release()

        dec = PCM16Decoder()
        for b in self.eloop.iterate(chunks, stop_flag=stop_flag):
            a = dec.feed(b)
            if a.size:
                yield a

    def warmup(self, voice=None):
        pass                      # a request would cost money; the first chunk pays the connection set-up

    def shutdown(self):
        try:
            self.eloop.run(self.client.close(), timeout=5)
        except Exception:
            pass
