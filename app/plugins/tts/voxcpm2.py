"""voxcpm2: VoxCPM2 streaming voice-clone TTS on a vLLM-Omni AsyncOmni engine, in-process (no HTTP).

TTSPlugin surface: list_voices(cfg) (scans voices.dir without loading anything), load_voice(name) -> Voice,
stream(text, voice), warmup(voice), shutdown(). Each stream(text, voice) is one VoxCPM2 request for one text
chunk; audio is yielded per decode step as it arrives. VoxCPM2 emits 48 kHz; a stateful polyphase resampler
brings it to out_sr (24 kHz: the EARS<->browser wire and the 2nd ear).

Voice cloning: the clip is resampled once, at load, to the rate VoxCPM2's audio VAE encodes at (the stage worker
otherwise resampled it on the CPU for every request: ~50 ms). If a transcript sits next to the clip
(voices/<name>.txt) it runs in "ultimate cloning" mode (prompt_audio + prompt_text, the full clip, so it matches
the transcript), with the clip in every request. Otherwise reference-only mode, clip trimmed to ref_max_s, and the
stage worker's speaker cache: the first request carries the clip plus a voice_name, and the worker stores the
encoded clip under it; later requests send only voice_name + the encoded length (a voice_profile), so the clip is
neither shipped nor re-encoded (~20 ms more). With both, a cloned voice starts as fast as zero-shot (I-06 / L-03).

Privacy: the clip and transcript are read into memory only (the speaker cache is in the stage worker's memory);
nothing is written to disk or logged.
"""
from __future__ import annotations
import glob, os, threading, uuid, logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import repo_path
from app.plugins.base import TTSCaps, TTSPlugin
from app.plugins.options import PluginOptions
from app.runtime.audio import StreamingAudioResampler

log = logging.getLogger("tts")


def scan_voices(cfg: dict) -> tuple[dict, str | None]:
    """({name: wav_path}, default_name): *.wav in voices.dir (a voices/<name>.txt next to a clip is its
    transcript). Falls back to the first clip if voices.default isn't present."""
    v = cfg.get("voices", {})
    voices: dict[str, str] = {}
    vdir = repo_path(v.get("dir"))
    if vdir and vdir.is_dir():
        for w in sorted(glob.glob(os.path.join(vdir, "*.wav"))):
            voices[Path(w).stem] = w
    default = v.get("default")
    if default not in voices:
        default = next(iter(voices), None)
    return voices, default


@dataclass(frozen=True, eq=False)
class Voice:
    """A loaded reference clip (+ optional transcript). Passed per request, so sessions can use different
    voices concurrently on the one engine."""
    name: str
    ref: list             # mono float samples at sr (the VAE's rate); never mutated. A list: the stage IPC
    sr: int               # passes lists through, while a tensor arrives garbled and kills the stage worker
    text: str | None      # transcript -> "ultimate cloning"; None -> reference-only
    cache_key: str = ""   # speaker-cache name (reference-only voices): changes when the clip file does
    feat_len: int = 0     # encoded clip length, which the prompt needs once the clip itself isn't sent


class VoxCPM2(TTSPlugin):
    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        model: str = "openbmb/VoxCPM2"
        deploy_config: str | None = None   # the vLLM-Omni stage config (repo-relative)
        out_sr: int = 24000                # VoxCPM2's 48 kHz is resampled to this
        ref_max_s: float = 10.0            # reference-only mode: the clip is trimmed to this
        allow_zero_shot: bool = False      # testing only: start without a voice clip

    @classmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]:
        voices, default = scan_voices(cfg)
        return sorted(voices), default

    def __init__(self, conf: dict, ctx):
        from transformers import AutoTokenizer
        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.entrypoints.utils import coerce_param_message_types
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_cjk_split_map
        super().__init__(conf, ctx)
        self.cfg = ctx.cfg; eloop = self.eloop
        model = self.opts.model
        deploy_config = repo_path(self.opts.deploy_config)
        self.out_sr = self.opts.out_sr
        self.ref_max_s = self.opts.ref_max_s
        names, default = scan_voices(self.cfg)
        if names:
            log.info("voices available: %s (default=%s)", sorted(names), default)
        elif self.opts.allow_zero_shot:          # testing only: VoxCPM2's unconditioned voice, no clone
            log.warning("no voice clips in voices.dir -> zero-shot TTS voice (tts.allow_zero_shot)")
        else:
            raise RuntimeError("no voice reference clips found (put a *.wav in voices.dir)")
        self._voices = {}                     # clip path -> (mtime, Voice): each clip is read once
        self._voice_lock = threading.Lock()

        async def make():
            kw = {"deploy_config": str(deploy_config)} if deploy_config else {}
            return AsyncOmni(model=model, **kw)
        self.engine = eloop.run(make())
        mc = self.engine.model_config
        self.hf_config = mc.hf_config
        self.tok = AutoTokenizer.from_pretrained(mc.model, trust_remote_code=True)
        self.split_map = build_cjk_split_map(self.tok)
        self.params = coerce_param_message_types(list(self.engine.default_sampling_params_list), True)
        self._tok_lock = threading.Lock()
        self._cached = set()                  # voice cache keys the stage worker has stored
        vae = self.hf_config.audio_vae_config
        self.vae_sr = int(vae["sample_rate"])
        self.patch_samples = int(self.hf_config.patch_size * np.prod(vae["encoder_rates"]))
        self.max_batch = 1                    # the deploy config's max_num_seqs: batch sizes warmup must cover
        if deploy_config:
            import yaml
            stages = (yaml.safe_load(Path(deploy_config).read_text()) or {}).get("stages") or [{}]
            self.max_batch = int(stages[0].get("max_num_seqs", 1))
        self.caps = TTSCaps(cloning=True, max_batch=self.max_batch)

    def load_voice(self, name) -> Voice | None:
        """The Voice for a clip name (re-scans voices.dir, so clips added after startup work), or None. A clip is
        read once, and again only when its file changes."""
        try:
            voices, _ = scan_voices(self.cfg)
        except Exception as e:
            log.warning("voice discovery failed: %s", e); return None
        path = voices.get(name)
        if path is None:
            return None
        mtime = os.path.getmtime(path)
        with self._voice_lock:
            hit = self._voices.get(path)
            if hit is None or hit[0] != mtime:
                hit = (mtime, self._read_voice(path, name))
                self._voices[path] = hit
            return hit[1]

    def _read_voice(self, ref_path, name=None) -> Voice:
        import soundfile as sf, librosa
        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        txt = Path(ref_path).with_suffix(".txt")
        text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
        if not text:
            text = None
            wav = wav[:int(self.ref_max_s * sr)]
        log.info("voice loaded: %.1fs clip, transcript=%s", len(wav) / sr, text is not None)
        if sr != self.vae_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.vae_sr).astype(np.float32)
        name = name or Path(ref_path).stem
        key = "" if text else f"tyger-{name}-{os.stat(ref_path).st_mtime_ns}"
        return Voice(name, wav.tolist(), self.vae_sr, text, key, -(-len(wav) // self.patch_samples))

    def _prompt(self, text, voice):
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt
        kw = dict(hf_config=self.hf_config, tokenizer=self.tok, split_map=self.split_map, text=text)
        if voice is None:
            return build_voxcpm2_prompt(**kw)
        if voice.cache_key and voice.cache_key in self._cached:          # the worker has the encoded clip
            prompt = build_voxcpm2_prompt(**kw, voice_profile={"mode": "reference", "ref_audio_feat_len": voice.feat_len})
        else:
            prompt = build_voxcpm2_prompt(**kw, ref_audio=voice.ref, ref_sr=voice.sr, ref_text=voice.text)
        if voice.cache_key:                                              # look up, or store on first use
            prompt["additional_information"]["voice_name"] = [voice.cache_key]
        return prompt

    def stream(self, text, voice=None, stop_flag=None):
        """Yield mono float32 chunks @ out_sr for one text chunk, in `voice` (None = zero-shot)."""
        rid = f"tts-{uuid.uuid4().hex[:12]}"
        with self._tok_lock:      # sessions call this from several threads; HF fast tokenizers aren't re-entrant
            prompt = self._prompt(text, voice)

        async def chunks():
            prev = 0
            async for res in self.engine.generate(prompt=prompt, request_id=rid,
                                                  sampling_params_list=self.params, output_modalities=["audio"]):
                mm = getattr(res, "multimodal_output", None) or {}
                key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
                if key is None:
                    continue
                sr = mm.get("sr", 48000)
                sr = sr[-1] if isinstance(sr, list) and sr else sr
                sr = int(sr.item() if hasattr(sr, "item") else sr)
                val = mm[key]
                if isinstance(val, list):                 # cumulative list: only the new tail
                    new, prev = val[prev:], len(val)
                else:
                    new = [val] if val is not None else []
                for c in new:
                    a = c.float().detach().cpu().numpy() if hasattr(c, "float") else np.asarray(c, np.float32)
                    yield a.reshape(-1), sr

        rs = None
        for a, sr in self.eloop.iterate(chunks, abort=lambda: self.engine.abort(rid), stop_flag=stop_flag):
            if rs is None:
                rs = StreamingAudioResampler(sr, self.out_sr)
                if voice is not None and voice.cache_key:   # audio out => prefill done => the worker stored it
                    self._cached.add(voice.cache_key)
            out = rs.process(a)
            if out.size:
                yield np.clip(out, -1.0, 1.0).astype(np.float32)
        if rs is not None and not (stop_flag and stop_flag()):
            tail = rs.process(np.zeros(0, np.float32), final=True)
            if tail.size:
                yield np.clip(tail, -1.0, 1.0).astype(np.float32)

    def prime_voice(self, voice):
        """Get a reference-only voice into the stage worker's speaker cache now (one tiny request), so its first
        real request doesn't pay the clip transfer + encode. No-op for transcript voices and cached ones."""
        if voice is not None and voice.cache_key and voice.cache_key not in self._cached:
            for _ in self.stream("Hi.", voice):
                pass

    def warmup(self, voice=None):
        """Pay VoxCPM2's one-time compiles before serving. It compiles lazily per batch size: the first time two
        (three, four) requests decode together, the WHOLE engine stalls ~25 s mid-conversation (seen live: a
        backchannel next to a turn). So run every batch size up to max_num_seqs concurrently, up front. Then put
        every clip in voices.dir into the speaker cache, so no conversation pays a voice's first use."""
        import time
        for t in ("Warmup.", "This is another warmup sentence to settle the kernels before we go live."):
            for _ in self.stream(t, voice):
                pass
        text = "This sentence warms up batched decoding so concurrent sessions never stall on a compile."
        for b in range(2, self.max_batch + 1):
            t0 = time.perf_counter()
            th = [threading.Thread(target=lambda: [None for _ in self.stream(text, voice)]) for _ in range(b)]
            for x in th:
                x.start()
            for x in th:
                x.join()
            log.info("TTS warmup: batch %d in %.1fs", b, time.perf_counter() - t0)
        for name in self.list_voices(self.cfg)[0]:
            self.prime_voice(self.load_voice(name))

    def shutdown(self):
        self.engine.shutdown()
