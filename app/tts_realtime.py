"""MOSS-TTS-Realtime streaming TTS backend for the MOUTH (venv_moss, tf 5.0.0). Loads the vendored+patched
Realtime model (weights.ensure_moss_realtime applies the static-cache/CUDA-graph patch) and clones the active
voice from a reference clip. 24 kHz native -> no resample to the 24 kHz wire/client.

Drop-in for the Nano backend: same set_voice / warmup / stream(text) / render(text) surface the MOUTH uses.
Each stream(text) runs one MOSS streaming session (begin -> push_text -> end -> drain -> flush) for one
sentence, yielding mono float32 chunks as they decode. The voice prompt is codec-encoded ONCE per set_voice
and reused every turn. With the static-cache patch + the long boot warmup, steady RTF ~0.55-0.7 on a 5090.

Privacy: no audio or transcript is written to disk; the ref clip is read once with soundfile and never copied.
"""
from __future__ import annotations
import numpy as np, torch


def _sanitize(tokens, cbk, eos):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens
    er = (tokens[:, 0] == eos).nonzero(as_tuple=False)
    inv = ((tokens < 0) | (tokens >= cbk)).any(dim=1)
    stop = int(er[0]) if er.numel() else None
    if inv.any():
        i = int(inv.nonzero(as_tuple=False)[0])
        stop = i if stop is None else min(stop, i)
    return tokens[:stop] if stop is not None else tokens


class MossRealtimeTTS:
    SRC_SR = 24000   # MOSS-Audio-Tokenizer native

    def __init__(self, moss_dir, codec_repo, out_sr=24000, device="cuda", dtype="bfloat16", attn="sdpa",
                 chunk_dur=0.24, dec_chunk_frames=4, prefill_text_len=6, prompt_max_s=4.0, max_length=800,
                 channels=16, audio_temperature=0.8, audio_top_p=0.6, audio_top_k=30, do_sample=True,
                 audio_repetition_penalty=1.1, repetition_window=50):
        import soundfile  # noqa: F401  (ensure present)
        from transformers import AutoTokenizer, AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module as G
        self.dev = torch.device(device)
        self.out_sr = int(out_sr)
        self.chunk_dur = float(chunk_dur)
        self.dec_chunk_frames = int(dec_chunk_frames)
        self.prefill_text_len = int(prefill_text_len)
        self.prompt_max_s = float(prompt_max_s)
        self.channels = int(channels)
        self.samp = dict(temperature=float(audio_temperature), top_p=float(audio_top_p),
                         top_k=int(audio_top_k), do_sample=bool(do_sample),
                         repetition_penalty=float(audio_repetition_penalty), repetition_window=int(repetition_window))
        self.tok = AutoTokenizer.from_pretrained(moss_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(moss_dir, trust_remote_code=True, attn_implementation=attn,
                                               torch_dtype=getattr(torch, dtype)).to(self.dev).eval()
        self.codec = AutoModel.from_pretrained(codec_repo, trust_remote_code=True).eval().to(self.dev)
        self.Proc = G("processing_mossttsrealtime.MossTTSRealtimeProcessor", moss_dir)
        Inf = G("streaming_mossttsrealtime.MossTTSRealtimeInference", moss_dir)
        self.Sess = G("streaming_mossttsrealtime.MossTTSRealtimeStreamingSession", moss_dir)
        self.Dec = G("streaming_mossttsrealtime.AudioStreamDecoder", moss_dir)
        self.proc = self.Proc(self.tok)
        self.inf = Inf(self.model, self.tok, max_length=int(max_length), channels=self.channels)
        self.session = self.Sess(self.inf, self.proc, codec=self.codec, codec_sample_rate=self.SRC_SR,
                                 codec_encode_kwargs={"chunk_duration": self.chunk_dur},
                                 prefill_text_len=self.prefill_text_len, **self.samp)
        self._cbk = int(getattr(self.codec, "codebook_size", 1024))
        self._eos = int(getattr(self.inf, "audio_eos_token", 1026))
        self.prompt_tokens = None
        self._pre_ids = self.tok.encode("<|im_end|>\n<|im_start|>assistant\n")

    def set_voice(self, ref_path):
        import soundfile as sf, torchaudio
        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)   # (samp, ch)
        wav = torch.from_numpy(wav.T)                                       # (ch, samp)
        if sr != self.SRC_SR:
            wav = torchaudio.functional.resample(wav, sr, self.SRC_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav[..., :int(self.prompt_max_s * self.SRC_SR)]
        with torch.inference_mode():
            enc = self.codec.encode(wav.unsqueeze(0).to(self.dev), chunk_duration=self.chunk_dur)
        self.prompt_tokens = enc["audio_codes"].cpu().numpy().squeeze(1)
        self.session.set_voice_prompt_tokens(self.prompt_tokens)

    def _post(self, arr):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        return np.clip(a, -1.0, 1.0)

    @torch.inference_mode()
    def stream(self, text, ref_path=None, stop_flag=None):
        """Yield mono float32 np chunks @ out_sr for one sentence (one MOSS streaming session)."""
        if ref_path is not None:
            self.set_voice(ref_path)
        if self.prompt_tokens is None:
            raise RuntimeError("no voice set (call set_voice first)")
        sp = self.proc.make_ensemble(self.prompt_tokens)
        pre = np.full((len(self._pre_ids), sp.shape[1]), self.proc.audio_channel_pad, dtype=np.int64)
        pre[:, 0] = self._pre_ids
        self.session.reset_turn(input_ids=np.concatenate([sp, pre], 0), include_system_prompt=False, reset_cache=True)
        dec = self.Dec(self.codec, chunk_frames=self.dec_chunk_frames, overlap_frames=0,
                       decode_kwargs={"chunk_duration": -1}, device=self.dev)
        cm = self.codec.streaming(batch_size=1); cm.__enter__()
        try:
            def emit(frames):
                for frame in frames:
                    if stop_flag and stop_flag():
                        return
                    t = frame[0] if frame.dim() == 3 else frame
                    t = _sanitize(t, self._cbk, self._eos)
                    if t.numel() == 0:
                        continue
                    dec.push_tokens(t.detach())
                    for wav in dec.audio_chunks():
                        if wav.numel() == 0:
                            continue
                        yield self._post(wav.detach().cpu().numpy())
            yield from emit(self.session.push_text(text + " "))
            if stop_flag and stop_flag():
                return
            yield from emit(self.session.end_text())
            while True:
                if stop_flag and stop_flag():
                    return
                frames = self.session.drain(max_steps=1)
                if not frames:
                    break
                yield from emit(frames)
                if self.inf.is_finished:
                    break
            final = dec.flush()
            if final is not None and final.numel() > 0 and not (stop_flag and stop_flag()):
                yield self._post(final.detach().cpu().numpy())
        finally:
            cm.__exit__(None, None, None)

    def render(self, text, ref_path=None):
        chunks = [c for c in self.stream(text, ref_path=ref_path)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self, ref_path=None):
        # Long sentences first so reduce-overhead captures the full CUDA-graph depth once, up front; later
        # (shorter) sentences then reuse the graphs instead of recapturing mid-conversation.
        for t in ("This is a deliberately long warmup sentence meant to capture the full streaming graph depth "
                  "so that every later reply reuses it instead of recapturing in the middle of a conversation.",
                  "Another warmup sentence to settle the compiled graphs before we go live.",
                  "Okay, ready to go."):
            for _ in self.stream(t, ref_path=ref_path):
                pass
