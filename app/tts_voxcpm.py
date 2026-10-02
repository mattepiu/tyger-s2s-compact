"""VoxCPM2 streaming voice-clone TTS on a vLLM-Omni AsyncOmni engine, in-process (no HTTP).

Same surface the MOUTH used for MOSS: set_voice / warmup / stream(text) / render(text). Each stream(text) is
one VoxCPM2 request for one text chunk; audio is yielded per decode step as it arrives. VoxCPM2 emits 48 kHz;
a stateful polyphase resampler brings it to out_sr (24 kHz: the EARS<->browser wire and the 2nd ear).

Voice cloning: the reference clip goes into every request's prompt (VoxCPM2 encodes it per request). If a
transcript sits next to the clip (voices/<name>.txt) it runs in "ultimate cloning" mode (prompt_audio +
prompt_text, the full clip is used so it matches the transcript); otherwise reference-only mode with the clip
trimmed to ref_max_s to bound the per-request prefill.

Privacy: the clip and transcript are read into memory only; nothing is written to disk or logged.
"""
from __future__ import annotations
import uuid, logging
from pathlib import Path

import numpy as np

log = logging.getLogger("tts")


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
        self.ref = self.ref_sr = self.ref_text = None

    def set_voice(self, ref_path):
        import soundfile as sf
        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        txt = Path(ref_path).with_suffix(".txt")
        self.ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
        if not self.ref_text:
            self.ref_text = None
            wav = wav[:int(self.ref_max_s * sr)]
        self.ref, self.ref_sr = wav.tolist(), int(sr)
        log.info("voice set: %.1fs clip, transcript=%s", len(wav) / sr, self.ref_text is not None)

    def _prompt(self, text):
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt
        return build_voxcpm2_prompt(hf_config=self.hf_config, tokenizer=self.tok, split_map=self.split_map,
                                    text=text, ref_audio=self.ref, ref_sr=self.ref_sr if self.ref else None,
                                    ref_text=self.ref_text)

    def stream(self, text, stop_flag=None):
        """Yield mono float32 chunks @ out_sr for one text chunk."""
        from vllm_omni.utils.audio_resample import StreamingAudioResampler
        rid = f"tts-{uuid.uuid4().hex[:12]}"
        prompt = self._prompt(text)

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

    def render(self, text):
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self):
        # first requests pay VoxCPM2's one-time torch.compile / graph capture (cached across restarts)
        for t in ("Warmup.", "This is another warmup sentence to settle the kernels before we go live."):
            for _ in self.stream(t):
                pass
