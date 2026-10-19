# Model & Concurrency Reduction Plan for Tyger-S2S

This document specifies the technical plan to downscale Tyger-S2S from an enterprise GPU stack (~28 GB VRAM, requiring an RTX 5090 or RTX PRO 5000) into a lightweight, ultra-low-latency voice agent (~5.0–5.4 GB VRAM) suitable for consumer GPUs (RTX 4060/3060, RTX 4070/3080, laptop GPUs with 6 GB VRAM, or cloud T4/A10G).

---

## 1. Executive Summary & Resource Comparison

The default Tyger-S2S stack allocates pinned GPU memory for 4 concurrent sessions across three heavy in-process engines. By:
1. Reducing **concurrency from 4 to 1 session** (slashing EARS and MOUTH sequence buffers),
2. Replacing the 15 GB **Gemma 4 E4B** LLM with [**`unsloth/Qwen3.5-2B-MTP-GGUF`**](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF) (2B parameters, quantized GGUF with Multi-Token Prediction speculative decoding), and
3. Replacing the 9.2 GB **VoxCPM2** TTS with [**Scylla's Band**](https://github.com/lowkeytea/scyllasband) (rectified-flow continuous acoustic latent model with Vocos 24 kHz neural vocoder),

the overall system VRAM footprint drops from **28.0 GB to ~5.0–5.4 GB (-82%)**, cold-start latency drops from ~11 minutes to < 30 seconds, and token generation speed increases by **1.5x–2x** via MTP.

### Resource Impact Matrix

| Subsystem | Baseline (4 Sessions) | Proposed (1 Session) | VRAM Delta | Latency / Throughput Notes |
|---|---|---|---|---|
| **LLM (`mouth.llm`)** | Gemma 4 E4B (FP8)<br>• Weights: ~10.4 GB<br>• KV Cache: 3.0 GiB (8 seqs)<br>• Graphs: ~1.6 GB<br>**Total: ~15.0 GB** | **`unsloth/Qwen3.5-2B-MTP-GGUF`** (`UD-Q4_K_XL`)<br>• Weights: ~1.4 GB<br>• KV Cache: 0.35 GiB (2 seqs, 4096 ctx)<br>• Working buffers: ~0.15 GB<br>**Total: ~1.90 GB** | **-13.10 GB** | **1.5x–2x faster token generation** via Multi-Token Prediction (MTP) speculative decoding (80–120+ tokens/s). |
| **TTS (`mouth.tts`)** | VoxCPM2 (vLLM-Omni)<br>• Weights: ~4.5 GB<br>• KV Cache: 2.0 GiB (4 seqs)<br>• Multi-stage working: ~2.7 GB<br>**Total: ~9.2 GB** | **Scylla's Band** (`litert` or `onnx`)<br>• Weights: ~164 MB (LiteRT)<br>• Working buffers: ~30 MB<br>**Total: ~0.0 GB (CPU)** or **~0.25 GB (GPU)** | **-9.20 GB** | • Eliminates 25s vLLM-Omni compile stalls.<br>• Native 24 kHz (zero resampling).<br>• Sentence latency: ~24 ms (GPU) or ~140 ms (CPU). |
| **EARS (`ears.tyger`)** | Nemotron-3.5-ASR + GRU<br>• 4 slots / max_batch=4<br>• RNN-T graphs (B=4)<br>**Total: ~3.8 GB** | Nemotron-3.5-ASR + GRU<br>• 1 slot / max_batch=1<br>• RNN-T graphs (B=1)<br>**Total: ~3.1 GB** | **-0.70 GB** | Removing `gather_ms` batch-waiting cuts **15 ms** directly from every 320 ms hop. |
| **Total VRAM** | **~28.0 GB** | **~5.00–5.25 GB** | **-22.75–23.00 GB (-82%)** | **Fits inside 6 GB & 8 GB consumer GPUs.** |
| **Cold Boot** | ~11 min (JIT + engine compiles) | **< 30 seconds** | -10.5 min | Instant GGUF/LiteRT mapping, no multi-stage JIT. |

---

## 2. Reducing EARS & System Concurrency (4 → 1)

### 2.1 EARS Engine Changes
EARS uses [`server.max_sessions`](file:///C:/Temp/tyger-s2s/app/plugins/ears/tyger/plugin.py#L52) to size engine slots:
```python
self.slots = int(ctx.cfg["server"].get("max_sessions", 4))
```
When `server.max_sessions` is set to `1`:
- **`NemotronAsr` & `GraphedRNNT`**: Allocates CUDA graph static execution buffers for `max_batch=1` instead of `max_batch=4` in [`app/plugins/ears/tyger/rnnt.py`](file:///C:/Temp/tyger-s2s/app/plugins/ears/tyger/rnnt.py#L60-L75).
- **`GruBatch`**: Only maintains 1 row of GRU hidden state in [`app/plugins/ears/tyger/gru_batch.py`](file:///C:/Temp/tyger-s2s/app/plugins/ears/tyger/gru_batch.py#L13-L19).
- **`EarsService`**: Only manages a single slot `[0]`. Any concurrent connection receives HTTP 503 "Busy" ([`app/ears/web.py`](file:///C:/Temp/tyger-s2s/app/ears/web.py#L93-L95)).

### 2.2 Latency Optimization: Zero Batch-Wait
In multi-session mode, EARS waits up to `gather_ms: 15` to collect audio hops from other active conversations into a single GPU forward pass:
```yaml
ears:
  gather_ms: 15    # wait up to 15ms to batch other sessions
  expect_ms: 60    # catch-up wait
```
**With `max_sessions: 1`:**
There are no other conversations to batch. Setting `gather_ms: 0` allows the GPU tick loop in [`app/plugins/ears/tyger/service.py`](file:///C:/Temp/tyger-s2s/app/plugins/ears/tyger/service.py#L19-L25) to dispatch immediately whenever the active session submits a 320 ms hop. This removes 15 ms of pure latency on every turn decision.

### 2.3 Cascading MOUTH Reductions
1. **LLM Sequence Concurrency (`max_num_seqs`)**:
   - Baseline: `max_num_seqs: 8` (4 sessions × 1 turn reply + 1 concurrent backchannel ack).
   - Single-Session: `max_num_seqs: 2` (1 turn reply + 1 backchannel ack).
2. **KV Cache Pinned Budget (`kv_cache_gib`)**:
   - Baseline: `3.0 GiB` for 8 sequences at 8192 context.
   - Single-Session: `0.35 GiB` for 2 sequences at 4096 context.
3. **Background Tasks**:
   - Baseline: `mouth.tasks.max_global: 4`.
   - Single-Session: `mouth.tasks.max_global: 1`.

---

## 3. The LLM: `unsloth/Qwen3.5-2B-MTP-GGUF`

### 3.1 Model Characteristics & Multi-Token Prediction (MTP)
[`unsloth/Qwen3.5-2B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF) offers key architectural benefits for real-time speech-to-speech:
1. **Unsloth Dynamic 2.0 Quantization (`UD-Q4_K_XL`)**:
   - Model weights occupy only **~1.4 GB**.
   - Preserves precision on critical attention heads while aggressively compacting linear layers.
2. **MTP (Multi-Token Prediction) Speculative Decoding**:
   - Predicts multiple future tokens simultaneously per step.
   - Using `llama.cpp` (`--spec-type draft-mtp --spec-draft-n-max 6`), it yields **1.5x–2x faster token generation rates** (80–120+ tokens/s).
   - **Crucial for Full Duplex Speech**: Speech synthesis cannot begin until [`app/mouth/chunker.py`](file:///C:/Temp/tyger-s2s/app/mouth/chunker.py) accumulates an entire sentence. Doubling token generation speed cuts the chunking latency in half.
3. **Hermes Tool Calling**:
   - Qwen3.5 natively outputs Hermes tool call blocks:
     `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`
   - Fully compatible with Tyger's existing [`HermesCalls`](file:///C:/Temp/tyger-s2s/app/plugins/llm/fake.py#L24-L45).

### 3.2 Deployment Options in Tyger-S2S

#### Option A: Dedicated `llama-server` Sidecar (Recommended for Full MTP Acceleration)
Run `llama-server` on localhost with MTP speculative decoding enabled, accessed via Tyger's [`HostedLLM`](file:///C:/Temp/tyger-s2s/app/plugins/llm/hosted/base.py) with `remote: false` (zero data leaves the box):

1. **Launch `llama-server`**:
   ```bash
   llama-server -hf unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL \
       -ngl 99 -c 4096 -fa on -np 2 \
       --spec-type draft-mtp --spec-draft-n-max 6 \
       --port 8080 --host 127.0.0.1
   ```
2. **Plugin Implementation (`app/plugins/llm/local/qwen_mtp.py`)**:
   ```python
   """Qwen 3.5 2B MTP GGUF local client for Tyger-S2S."""
   from __future__ import annotations
   from openai import AsyncOpenAI
   from app.plugins.llm.hosted.base import HostedLLM
   from app.plugins.llm.fake import HERMES

   class Qwen35MTPLocal(HostedLLM):
       SDK = "openai"

       def make_client(self, key):
           return AsyncOpenAI(base_url=self.opts.base_url or "http://127.0.0.1:8080/v1",
                              api_key="none",
                              timeout=self.opts.read_timeout_s)
   ```

#### Option B: In-Process via vLLM GGUF Loader
vLLM 0.30+ supports direct GGUF inference via `AsyncLLM`:
```yaml
llm:
  reply:
    plugin: qwen3.5-2b-mtp
    model: unsloth/Qwen3.5-2B-MTP-GGUF
    quantization: gguf
    gpu_memory_utilization: 0.20
    kv_cache_gib: 0.35
    max_num_seqs: 2
```

---

## 4. Replacing the TTS Model: Scylla's Band

Tyger's [`TTSPlugin`](file:///C:/Temp/tyger-s2s/app/plugins/base.py#L93-L110) abstraction allows swapping the voice engine with a single config line (`tts.plugin`).

### 4.1 Overview & Architecture
[**Scylla's Band** (`lowkeytea/scyllasband`)](https://github.com/lowkeytea/scyllasband) is a local text-to-speech runtime designed for managed multi-voice synthesis:
- **Latent Rectified Flow**: Predicts continuous acoustic latents and phone durations.
- **Vocos Vocoder**: Synthesizes speech directly at **24 kHz mono**, matching Tyger's audio wire (`WIRE_SR = 24000`) with zero resampling overhead.
- **Ultra-Lightweight Footprint**:
  - **LiteRT bundle**: **~164 MB** download, runs **>35× faster than real-time on CPU** consuming **0 MB GPU VRAM**.
  - **ONNX Runtime bundle**: **~262 MB** download (< 250 MB VRAM when running on GPU).
- **Process Stability**: Operates 100% in-process. Eliminates `vllm-omni`, its stage worker process, and unkillable background processes during shutdown ([`stop.sh`](file:///C:/Temp/tyger-s2s/stop.sh#L3-L4)).

### 4.2 Fine-Grained Delivery Controls
Scylla's Band provides four continuous 0–4 delivery scales (2 is neutral):
- **Energy**: Voice brightness and projection.
- **Tension**: Tightness and pitch excursions.
- **Valence**: Emotional pitch contour (downcast vs lively).
- **Assertiveness**: Speaking pace and phrase-clipping (strongest control over pace).

These controls can be configured globally or customized per character in `config/small.yaml`.

### 4.3 Voice & Character Mapping

Tyger defines 4 characters in [`config/prompts.yaml`](file:///C:/Temp/tyger-s2s/config/prompts.yaml#L15-L48): Mira, Aria, Kai, and Rex. Scylla's Band provides 10 managed voices across 7 locales, mapping naturally to them:

| Character | Persona Description | Scylla's Band Voice | Tuned Delivery Settings |
|---|---|---|---|
| **Mira** (Default) | Thoughtful, warm woman | `ariadne` | `energy: 1.8, valence: 2.2, tension: 1.5, assertiveness: 2.0` (calm, warm) |
| **Aria** | Quick-witted, bright woman | `scylla` | `energy: 2.2, valence: 2.2, tension: 2.0, assertiveness: 2.2` (bright, witty) |
| **Kai** | Easygoing guy, calm presence | `ink` | `energy: 1.8, tension: 1.4, valence: 2.0, assertiveness: 2.0` (relaxed, grounded) |
| **Rex** | Straightforward, high-energy guy | `rex` | `energy: 2.3, assertiveness: 2.4, tension: 2.2, valence: 2.0` (assertive, energetic) |

*(Note: Scylla's Band natively features a character named `rex`, exactly matching Tyger's persona!)*

### 4.4 Plugin Implementation (`app/plugins/tts/scyllasband.py`)

```python
"""Scylla's Band TTS plugin for Tyger-S2S (https://github.com/lowkeytea/scyllasband)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator
import numpy as np

from app.plugins.base import TTSCaps, TTSPlugin
from app.plugins.options import PluginOptions

VOICE_MAP = {
    "mira": "ariadne",
    "aria": "scylla",
    "kai": "ink",
    "rex": "rex",
}

class ScyllasBandTTS(TTSPlugin):
    out_sr = 24000
    caps = TTSCaps(cloning=False, max_batch=1)

    @dataclass(frozen=True, kw_only=True)
    class Options(PluginOptions):
        bundle: str = "scyllasband/models/litert"  # or "scyllasband/models/onnx"
        backend: str = "litert"                   # "litert", "onnx", "coreml"
        language: str = "en_us"
        energy: float = 2.0                       # 0 to 4 (2 is neutral)
        tension: float = 2.0
        valence: float = 2.0
        assertiveness: float = 2.0
        speed: float = 1.0

    @classmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]:
        return sorted(list(VOICE_MAP.keys())), "mira"

    def __init__(self, conf: dict, ctx):
        super().__init__(conf, ctx)
        from scyllasband import ScyllasBandRuntime
        self.runtime = ScyllasBandRuntime.from_bundle(self.opts.bundle, backend=self.opts.backend)

    def load_voice(self, name: str) -> Any:
        return VOICE_MAP.get(name, "ariadne")

    def stream(self, text: str, voice=None, stop_flag=None) -> Iterator[np.ndarray]:
        if stop_flag and stop_flag():
            return
        from scyllasband import SynthesisRequest
        voice_id = voice if isinstance(voice, str) else "ariadne"
        req = SynthesisRequest(
            text=text,
            voice_id=voice_id,
            language=self.opts.language,
            delivery={
                "energy": self.opts.energy,
                "tension": self.opts.tension,
                "valence": self.opts.valence,
                "assertiveness": self.opts.assertiveness,
            },
            speed=self.opts.speed,
        )
        res = self.runtime.synthesize(req)
        if stop_flag and stop_flag():
            return
        if res.audio is not None and len(res.audio) > 0:
            yield res.audio.astype(np.float32)

    def warmup(self, voice=None):
        self.runtime.warmup()
        for _ in self.stream("Warmup.", voice or "ariadne"):
            pass

    def shutdown(self):
        del self.runtime
```

#### Registry Update ([`app/plugins/__init__.py`](file:///C:/Temp/tyger-s2s/app/plugins/__init__.py#L12-L13))
```python
REGISTRY = {
    ...
    "tts": {
        "voxcpm2": "app.plugins.tts.voxcpm2:VoxCPM2",
        "scyllasband": "app.plugins.tts.scyllasband:ScyllasBandTTS",
        "fake": "app.plugins.tts.fake:FakeTTS",
    },
}
```

---

## 5. Reduced Configuration Profile (`config/small.yaml`)

```yaml
# config/small.yaml - Lightweight single-session stack for consumer GPUs (<= 6-8 GB VRAM)
server:
  host: 0.0.0.0
  port: 7860
  mouth_host: 127.0.0.1
  mouth_port: 8770
  max_sessions: 1         # Single-session concurrency
  pages: true
  max_session_s: 180

ears:
  plugin: tyger
  model: nineninesix/tyger
  revision: cd826db00a73d8d78603be23bb86f1cb02158bf7
  ep_thr: 0.915
  ep_deb: 2
  barge_thr: 0.4
  barge_deb: 2
  bc_thr: 0.4
  bc_deb: 2
  gather_ms: 0            # 0ms: dispatch immediately, no other sessions to batch
  expect_ms: 20
  stats_s: 60
  cpu_threads: 4

llm:
  reply:
    plugin: qwen3.5-2b-mtp
    model: unsloth/Qwen3.5-2B-MTP-GGUF
    base_url: http://127.0.0.1:8080/v1  # local llama-server with MTP speculative decoding
    api_key_env: HF_TOKEN
    remote: false
    max_tokens: 192
    temperature: 0.7
  filler: reply

tts:
  plugin: scyllasband
  bundle: scyllasband/models/litert
  backend: litert                       # runs on CPU (>35x real time, 0 MB GPU VRAM)
  language: en_us
  energy: 2.0
  tension: 2.0
  valence: 2.0
  assertiveness: 2.0
  speed: 1.0
  out_sr: 24000

voices:
  default: mira
  character: mira
  characters:
    aria: aria
    kai: kai
    rex: rex

mouth:
  prompts: config/prompts.yaml
  max_gen: 192
  history_turns: 8
  bc_max_new: 12
  bc_min_words: 4
  first_chunk: sentence
  cpu_threads: 4
  tasks:
    enabled: false
```

---

## 6. Implementation & Verification Roadmap

### Step 1: Environment & Dependencies
```bash
git clone https://github.com/lowkeytea/scyllasband
uv pip install -e ./scyllasband
python -m scyllasband download --flavor litert
```

### Step 2: Implement Code Modules
1. Add `app/plugins/llm/local/qwen_mtp.py`.
2. Add `app/plugins/tts/scyllasband.py`.
3. Register `qwen3.5-2b-mtp` and `scyllasband` in [`app/plugins/__init__.py`](file:///C:/Temp/tyger-s2s/app/plugins/__init__.py).
4. Save `config/small.yaml`.

### Step 3: Run Contract & Regression Tests
```bash
# 1. Verify config parsing & schemas
.venv/bin/python tests/test_config.py

# 2. Verify Hermes call parsing
.venv/bin/python tests/plugins/test_calls_hermes.py

# 3. Verify single-session web & floor handling
.venv/bin/python tests/ears/test_web.py
.venv/bin/python tests/mouth/test_session.py
```

### Step 4: Live Execution
```bash
# Terminal 1: Local MTP speculative inference engine
llama-server -hf unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL \
    -ngl 99 -c 4096 -fa on -np 2 \
    --spec-type draft-mtp --spec-draft-n-max 6 \
    --port 8080

# Terminal 2: Tyger-S2S Stack (with Scylla's Band)
CONFIG=config/small.yaml ./run.sh
```

### Verification Criteria
1. Logs report `loading EARS for 1 sessions` and `loading TTS scyllasband`.
2. Open `http://localhost:7860`:
   - Responses stream with high naturalness and responsive barge-in.
   - Adjusting delivery controls in YAML audibly shifts the character's emotional tone.
   - Total GPU memory footprint remains between **5.0 and 5.3 GB**.
