"""NemotronAsr: everything specific to nemotron-3.5-asr (the copy the tyger bundle carries). Its processor and model,
its log-mel features computed on the GPU, the layer-15 encode the turn-taker reads (equal-length windows batched;
all-zero windows cached), and transcription through its own greedy RNN-T decoder as a CUDA graph (rnnt.py), on a
CUDA stream of its own so it never stalls the hop clock.
"""
from __future__ import annotations
import threading

import numpy as np, torch

from .constants import FRAME_S, L15, TURN_MAX_S
from .rnnt import GraphedRNNT


class _StopAtL15(Exception):
    pass


class NemotronAsr:
    def __init__(self, model, proc, device="cuda", max_batch=4, cuda_graph=True):
        """model, proc: nemotron-3.5-asr's model (fp32) and processor, as bundle.load gives them."""
        self.device = device
        self.proc = proc
        self.fe = self.proc.feature_extractor
        self.fe.mel_filters = self.fe.mel_filters.to(device)   # features are computed on the GPU (self.features)
        self.model = model.to(device).eval()
        self.enc = self.model.encoder
        self._zero_enc = {}                    # window length -> L15 of an all-zero window
        # The heads read layer 15 of 24: their encodes stop there (a hook on that layer raises), ~1/3 less encoder
        # work per tick. Thread-local, so the ASR thread's full-depth encodes of the same module are untouched.
        self._tls = threading.local()
        self.enc.layers[L15 - 1].register_forward_hook(self._l15_hook)
        self.stream = torch.cuda.Stream()      # transcription runs here, beside the hop clock's stream
        self._gen_lock = threading.Lock()      # the decoder's static graph buffers: one transcription at a time
        self.rnnt = GraphedRNNT(self.model, max_batch=max_batch, max_frames=int(TURN_MAX_S / FRAME_S) + 8)
        if cuda_graph:
            self.rnnt.capture()

    def _l15_hook(self, mod, args, out):
        if getattr(self._tls, "stop", False):
            self._tls.h = out[0] if isinstance(out, tuple) else out
            raise _StopAtL15

    @torch.no_grad()
    def features(self, wavs):
        """list of 16 kHz clips -> (log-mel features, attention mask) on the GPU: the processor's feature extraction
        (right-pad, preemphasis, STFT, mel, mask) with the waveform on the GPU. Its CPU STFT was ~20 ms per tick at
        4 conversations (8 windows), more than the encoder; this is ~2 ms. Same values within float noise (~2e-4 on
        log-mels of ~16). The extractor's own device="cuda" leaves the waveform on the CPU and fails."""
        fe, dev = self.fe, self.device
        n = max(len(w) for w in wavs)
        x = np.zeros((len(wavs), n), np.float32)
        for i, w in enumerate(wavs):
            x[i, :len(w)] = w
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        lens = torch.tensor([len(w) for w in wavs], device=dev)
        inside = torch.arange(n, device=dev)[None] < lens[:, None]
        x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], 1).masked_fill(~inside, 0.0)
        f = fe._torch_extract_fbank_features(x, dev, center=True)
        flen = torch.floor_divide(lens + fe.n_fft // 2 * 2 - fe.n_fft, fe.hop_length)
        am = torch.arange(f.shape[1], device=dev)[None] < flen[:, None]
        return f * am.unsqueeze(-1), am

    @torch.no_grad()
    def encode(self, wavs):
        """list of 16 kHz windows -> list of (T, 1024) L15 tensors on the GPU; equal lengths are batched.
        All-zero windows (the agent channel while listening) are encoded once per length and cached: same input,
        same features, half the feature-extraction CPU and encoder work while everyone listens."""
        out = [None] * len(wavs); groups = {}
        for i, w in enumerate(wavs):
            if len(w) in self._zero_enc and not w.any():
                out[i] = self._zero_enc[len(w)]
            else:
                groups.setdefault(len(w), []).append(i)
        for n, idx in groups.items():
            feats, am = self.features([wavs[i] for i in idx])
            self._tls.stop = True
            try:
                self.enc(input_features=feats, attention_mask=am, num_lookahead_tokens=0)
            except _StopAtL15:
                pass
            finally:
                self._tls.stop = False
            hs = self._tls.h.float()
            for j, i in enumerate(idx):
                out[i] = hs[j]
                if n not in self._zero_enc and not wavs[i].any():
                    self._zero_enc[n] = hs[j].clone()
        return out

    @torch.no_grad()
    def transcribe(self, wavs):
        """list of 16 kHz clips -> list of text, on the ASR stream, as one padded batch. Greedy RNN-T as a CUDA
        graph (rnnt.py): same text as nemotron's generate(), ~4x faster and without its per-step Python."""
        if not wavs:
            return []
        with self._gen_lock, torch.cuda.stream(self.stream):
            feats, am = self.features(wavs)
            seqs = self.rnnt(feats, am)
            return [self.proc.decode(q, skip_special_tokens=True).strip() for q in seqs]
