"""VoxCPM2 streaming voice-clone TTS on a vLLM-Omni AsyncOmni engine, in-process (no HTTP).

Surface: load_voice(path) -> Voice, then stream(text, voice) / render / warmup. Each stream(text, voice) is
one VoxCPM2 request for one text chunk; audio is yielded per decode step as it arrives. VoxCPM2 emits 48 kHz;
a stateful polyphase resampler brings it to out_sr (24 kHz: the EARS<->browser wire and the 2nd ear).

Voice cloning: the reference clip goes into every request's prompt (VoxCPM2 encodes it per request). If a
transcript sits next to the clip (voices/<name>.txt) it runs in "ultimate cloning" mode (prompt_audio +
prompt_text, the full clip is used so it matches the transcript); otherwise reference-only mode with the clip
trimmed to ref_max_s to bound the per-request prefill.

Privacy: the clip and transcript are read into memory only; nothing is written to disk or logged.
"""
from __future__ import annotations
import threading, uuid, logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("tts")


@dataclass(frozen=True)
class Voice:
    """A loaded reference clip (+ optional transcript). Passed per request, so sessions can use different
    voices concurrently on the one engine."""
    name: str
    ref: tuple            # mono float samples
    sr: int
    text: str | None      # transcript -> "ultimate cloning"; None -> reference-only


class VoxCPM2TTS:
    def __init__(self, eloop, model="openbmb/VoxCPM2", deploy_config=None, out_sr=24000, ref_max_s=10.0):
        from transformers import AutoTokenizer
        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.entrypoints.utils import coerce_param_message_types
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_cjk_split_map
        self.eloop = eloop
        self.out_sr = int(out_sr)
        self.ref_max_s = float(ref_max_s)

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
        self.max_batch = 1                    # the deploy config's max_num_seqs: batch sizes warmup must cover
        if deploy_config:
            import yaml
            stages = (yaml.safe_load(Path(deploy_config).read_text()) or {}).get("stages") or [{}]
            self.max_batch = int(stages[0].get("max_num_seqs", 1))

    def load_voice(self, ref_path, name=None) -> Voice:
        import soundfile as sf
        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        txt = Path(ref_path).with_suffix(".txt")
        text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
        if not text:
            text = None
            wav = wav[:int(self.ref_max_s * sr)]
        log.info("voice loaded: %.1fs clip, transcript=%s", len(wav) / sr, text is not None)
        return Voice(name or Path(ref_path).stem, tuple(wav.tolist()), int(sr), text)

    def _prompt(self, text, voice):
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt
        ref = list(voice.ref) if voice else None
        return build_voxcpm2_prompt(hf_config=self.hf_config, tokenizer=self.tok, split_map=self.split_map,
                                    text=text, ref_audio=ref, ref_sr=voice.sr if voice else None,
                                    ref_text=voice.text if voice else None)

    def stream(self, text, voice=None, stop_flag=None):
        """Yield mono float32 chunks @ out_sr for one text chunk, in `voice` (None = zero-shot)."""
        from vllm_omni.utils.audio_resample import StreamingAudioResampler
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
            out = rs.process(a)
            if out.size:
                yield np.clip(out, -1.0, 1.0).astype(np.float32)
        if rs is not None and not (stop_flag and stop_flag()):
            tail = rs.process(np.zeros(0, np.float32), final=True)
            if tail.size:
                yield np.clip(tail, -1.0, 1.0).astype(np.float32)

    def render(self, text, voice=None):
        chunks = list(self.stream(text, voice))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self, voice=None):
        """Pay VoxCPM2's one-time compiles before serving. It compiles lazily per batch size: the first time two
        (three, four) requests decode together, the WHOLE engine stalls ~25 s mid-conversation (seen live: a
        backchannel next to a turn). So run every batch size up to max_num_seqs concurrently, up front."""
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
