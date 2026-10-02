"""MOSS-TTS-Nano TTS wrapper for the MOUTH (tf 4.57 venv). Loads the vendored+patched MOSS model, clones the
active voice from a reference clip, and streams audio per text clause. MOSS is 48 kHz stereo natively; we
downmix to mono and resample to the client rate (24 kHz) to match the exp-12 wire + client + EARS 2nd ear.

Streaming design (from the V0 bench): each inference_stream call re-encodes the ref yet still hits ~112 ms
first-audio, so the MOUTH feeds MOSS one short clause at a time (preserving LLM/TTS overlap) without needing
a ref-code cache. No audio is written to disk (output_audio_path=None; weights.py patches MOSS to honor it)."""
from __future__ import annotations
import numpy as np, torch


class MossNanoTTS:
    SRC_SR = 48000   # MOSS-Audio-Tokenizer-Nano native

    def __init__(self, moss_dir, codec_repo, out_sr=24000, device="cuda", dtype="bfloat16", attn="sdpa",
                 max_new_frames=375, vc_max_tokens=100, do_sample=True, audio_temperature=0.8,
                 audio_top_p=0.95, audio_top_k=25, audio_repetition_penalty=1.2):
        from transformers import AutoModelForCausalLM
        import torchaudio
        self._ta = torchaudio
        self.device = device
        self.out_sr = int(out_sr)
        self.codec_repo = codec_repo
        self.max_new_frames = int(max_new_frames)
        self.vc_max_tokens = int(vc_max_tokens)
        self.samp = dict(do_sample=bool(do_sample), audio_temperature=float(audio_temperature),
                         audio_top_p=float(audio_top_p), audio_top_k=int(audio_top_k),
                         audio_repetition_penalty=float(audio_repetition_penalty))
        self.model = AutoModelForCausalLM.from_pretrained(moss_dir, trust_remote_code=True)
        self.model.to(device=device, dtype=getattr(torch, dtype))
        self.model._set_attention_implementation(attn)
        self.model.eval()
        self.ref_path = None

    def set_voice(self, ref_path):
        self.ref_path = ref_path

    def _post(self, wav, sr):
        """(channels, samples) tensor @ sr -> mono float32 np @ out_sr."""
        w = torch.as_tensor(wav).float()
        if w.ndim == 1:
            w = w[None]
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != self.out_sr:
            w = self._ta.functional.resample(w, sr, self.out_sr)
        return np.clip(w.squeeze(0).cpu().numpy().astype(np.float32), -1.0, 1.0)

    @torch.no_grad()
    def stream(self, text, ref_path=None, stop_flag=None):
        """Yield mono float32 np chunks @ out_sr for one text clause."""
        ref = ref_path or self.ref_path
        for ev in self.model.inference_stream(
                text=text, output_audio_path=None, mode="voice_clone", prompt_text=None,
                prompt_audio_path=ref, audio_tokenizer_type="moss-audio-tokenizer-nano",
                audio_tokenizer_pretrained_name_or_path=self.codec_repo, device=torch.device(self.device),
                max_new_frames=self.max_new_frames, voice_clone_max_text_tokens=self.vc_max_tokens, **self.samp):
            if stop_flag and stop_flag():
                break
            if ev.get("type") == "audio":
                out = self._post(ev["waveform"], int(ev.get("sample_rate", self.SRC_SR)))
                if out.size:
                    yield out

    def render(self, text, ref_path=None):
        """Full mono float32 clip @ out_sr (for pre-rendered canned acks + listenable samples)."""
        chunks = [c for c in self.stream(text, ref_path=ref_path)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self, ref_path=None):
        for t in ("Hello there.", "Let me check that for you."):
            for _ in self.stream(t, ref_path=ref_path):
                pass
