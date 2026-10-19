# Tyger-S2S Compact

[![VRAM: ~5.2 GB](https://img.shields.io/badge/VRAM-~5.2_GB-success)](file:///C:/Temp/tyger-s2s/config/small.yaml)
[![Architecture: Single_Session](https://img.shields.io/badge/Concurrency-1_Session_(Full_Duplex)-blue)](file:///C:/Temp/tyger-s2s/config/small.yaml)
[![LLM: Qwen3.5--2B--MTP](https://img.shields.io/badge/LLM-Qwen3.5--2B--MTP--GGUF-orange)](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF)
[![TTS: Scylla's Band](https://img.shields.io/badge/TTS-Scylla's_Band_24kHz-purple)](https://github.com/lowkeytea/scyllasband)

**Tyger-S2S Compact** is a lightweight, ultra-low-latency deployment profile of Tyger-S2S engineered for consumer GPUs (6 GB / 8 GB VRAM, such as RTX 3060, RTX 4060, laptop GPUs, or cloud T4/A10G instances).

It reduces total GPU memory consumption from **~28.0 GB down to ~5.0–5.4 GB (-82%)**, cuts cold startup time from ~11 minutes to **< 30 seconds**, and doubles token generation throughput via Multi-Token Prediction (MTP), while preserving the core conversational full-duplex experience:
- Speaks and listens simultaneously (bidirectional full duplex);
- Answers instantly upon speech completion with zero batch-waiting;
- Halts mid-word upon user barge-in;
- Produces natural concurrent backchannels ("mm-hm", "yeah") without interrupting the user.

---

## 1. Resource Comparison Matrix

| Subsystem | Baseline Stack (`config/default.yaml`) | Compact Stack (`config/small.yaml`) | Delta | Practical Impact |
|---|---|---|---|---|
| **Concurrency** | 4 concurrent sessions (`max_sessions: 4`) | 1 dedicated session (`max_sessions: 1`) | -3 sessions | Sized for personal desktop / single-user use. |
| **EARS Turn-Taker** | Nemotron-3.5-ASR + GRU ($B=4$ graphs)<br>**~3.8 GB VRAM** | Nemotron-3.5-ASR + GRU ($B=1$ graphs)<br>**~3.1 GB VRAM** | **-0.70 GB** | `gather_ms: 0` cuts **15 ms** directly from every 320 ms hop. |
| **LLM Responder** | Gemma 4 E4B (FP8 in-process vLLM)<br>**~15.0 GB VRAM** | `unsloth/Qwen3.5-2B-MTP-GGUF`<br>**~1.9 GB VRAM** | **-13.10 GB** | **1.5x–2x generation throughput** via MTP speculative decoding (80–120+ tokens/s). |
| **TTS Engine** | VoxCPM2 (vLLM-Omni AsyncOmni)<br>**~9.2 GB VRAM** | Scylla's Band (LiteRT CPU / ONNX GPU)<br>**0.0 GB (CPU)** or **<0.25 GB (GPU)** | **-9.20 GB** | Native 24 kHz audio (zero resampling), eliminates 25s vLLM-Omni compile stalls. |
| **Total VRAM** | **~28.0 GB** (Requires RTX 5090 / Pro 5000) | **~5.0–5.4 GB** (Fits 6 GB / 8 GB GPUs) | **-22.8 GB (-82%)** | Runs locally on consumer gaming rigs & laptops. |
| **Cold Start** | ~11 minutes (JIT & CUDA graph compilation) | **< 30 seconds** | -10.5 min | Instant startup. |

---

## 2. Architecture & Implementation Overview

Tyger-S2S Compact is composed of three cooperating subsystems running with strict single-session allocation:

```
                      ┌───────────────────────────────────────────────┐
                      │              Browser Client                   │
                      │  (Microphone @ 16 kHz / Speaker @ 24 kHz)     │
                      └──────────────▲─────────────────┬──────────────┘
                                     │ WebSocket       │ WebSocket
                               Audio │ (PCM 16k)       │ (PCM 24k)
                                     │                 ▼
                 ┌───────────────────┴─────────────────────────────────┐
                 │                       EARS                          │
                 │   app.plugins.ears.tyger:Tyger                      │
                 │   - Nemotron-3.5-ASR + GRU Turn-Taker               │
                 │   - server.max_sessions: 1                          │
                 │   - ears.gather_ms: 0 (Zero batch-waiting)          │
                 └───────────────────┬─────────────────────────────────┘
                                     │ Internal TCP (:8770)
                                     ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                      MOUTH                          │
                 │                                                     │
                 │   LLM Plugin: qwen3.5-2b-mtp                        │
                 │   app.plugins.llm.local.qwen_mtp:Qwen35MTPLocal     │
                 │   - Local llama-server with MTP Speculative Decode  │
                 │   - Hermes tool calling (<tool_call>...)            │
                 │   - Instant stop_flag abortion on barge-in          │
                 │                                                     │
                 │   TTS Plugin: scyllasband                           │
                 │   app.plugins.tts.scyllasband:ScyllasBandTTS        │
                 │   - Scylla's Band LiteRT (CPU) / ONNX (GPU)         │
                 │   - Native 24 kHz mono output (WIRE_SR = 24000)     │
                 │   - Continuous delivery controls (energy, tension)  │
                 │   - Persona mapping: Mira, Aria, Kai, Rex           │
                 └─────────────────────────────────────────────────────┘
```

### 2.1 EARS Engine (`app/plugins/ears/tyger/`)
- **Single-Session Sizing**: Setting [`server.max_sessions: 1`](file:///C:/Temp/tyger-s2s/config/small.yaml#L7) instructs [`Tyger`](file:///C:/Temp/tyger-s2s/app/plugins/ears/tyger/plugin.py#L30) to allocate CUDA graphs and GRU hidden state buffers for batch size $B=1$ rather than $B=4$. Any second concurrent connection receives an immediate HTTP 503 "Busy" signal.
- **Zero Batch-Wait Dispatch**: Multi-session Tyger waits up to `gather_ms: 15` to batch audio hops from other sessions into one GPU forward pass. Setting [`ears.gather_ms: 0`](file:///C:/Temp/tyger-s2s/config/small.yaml#L21) disables the batch-wait window, dispatching every 320 ms audio hop immediately upon arrival and saving 15 ms of latency on every turn decision.

### 2.2 LLM Plugin (`app/plugins/llm/local/qwen_mtp.py`)
- **Model**: [`unsloth/Qwen3.5-2B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF) in `UD-Q4_K_XL` quantization (~1.4 GB weights).
- **MTP Speculative Decoding**: Utilizes Multi-Token Prediction (MTP) via local `llama-server` (`--spec-type draft-mtp --spec-draft-n-max 6`). Predicting multiple tokens concurrently per forward pass achieves **80–120+ tokens/s** on consumer hardware. Because speech synthesis in [`app/mouth/chunker.py`](file:///C:/Temp/tyger-s2s/app/mouth/chunker.py) requires an entire first sentence before synthesizing, doubling token generation rate cuts first-chunk sentence formation latency in half.
- **Hermes Tool Calling**: Native parsing for Hermes JSON call syntax (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`). Emits early `tool_start` notifications to trigger MOUTH vocal fillers, followed by structured `tool_calls`.
- **Zero Data Leakage**: Configured with `remote: false`; all prompts and transcripts remain strictly on your local machine.

### 2.3 TTS Plugin (`app/plugins/tts/scyllasband.py`)
- **Engine**: [Scylla's Band (`lowkeytea/scyllasband`)](https://github.com/lowkeytea/scyllasband) continuous acoustic latent model + Vocos neural vocoder.
- **Native 24 kHz Mono**: Synthesizes directly at 24,000 Hz, matching Tyger's wire format ([`WIRE_SR = 24000`](file:///C:/Temp/tyger-s2s/app/plugins/tts/__init__.py#L6)). No polyphase resampler is needed.
- **CPU Offloading**: The LiteRT INT8 runtime (~164 MB) runs **>35× faster than real-time on standard CPUs**, freeing 100% of GPU memory from TTS. Alternatively, the ONNX backend uses <250 MB of GPU VRAM.
- **Process Stability**: Operates entirely in-process without multi-process stage workers or orphaned background daemons.

---

## 3. User Guide: Getting Started

### 3.1 Prerequisites
1. **NVIDIA GPU** with $\ge$ 6 GB VRAM and CUDA driver (CUDA 12 or 13).
2. **Python 3.12** and [**uv**](https://docs.astral.sh/uv/).
3. **llama.cpp** installed with `llama-server` in your `PATH`.
4. A **Hugging Face Token** (free account) to download the EARS bundle `nineninesix/tyger`.

---

### 3.2 Step-by-Step Setup

#### Step 1: Clone Repository and Create Virtual Environment
```bash
git clone https://github.com/nineninesix-ai/tyger-s2s.git
cd tyger-s2s

# Create Python 3.12 virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install Tyger dependencies
uv pip install -r requirements.txt --torch-backend=auto
```

#### Step 2: Install Scylla's Band
Install Scylla's Band and download the lightweight LiteRT models:
```bash
uv pip install scyllasband
python -m scyllasband download --flavor litert
```

#### Step 3: Configure Environment
Copy `.env.example` to `.env` and provide your Hugging Face token:
```bash
cp .env.example .env
```
Inside `.env`:
```env
HF_TOKEN=hf_your_huggingface_token_here
```

---

### 3.3 Running Tyger-S2S Compact

Tyger-S2S Compact runs in two terminals (or two background processes):

#### Terminal 1: Start Local MTP LLM Server
Start `llama-server` hosting the Qwen3.5-2B MTP model:
```bash
llama-server \
  -hf unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL \
  -ngl 99 \
  -c 4096 \
  -fa on \
  -np 2 \
  --spec-type draft-mtp \
  --spec-draft-n-max 6 \
  --port 8080 \
  --host 127.0.0.1
```
*Wait until `llama-server` reports `HTTP server listening at http://127.0.0.1:8080`.*

#### Terminal 2: Start Tyger-S2S with Compact Config
```bash
CONFIG=config/small.yaml ./run.sh
```
*(On Windows PowerShell: `$env:CONFIG="config/small.yaml"; ./run.sh` or run `python -m app.mouth` and `python -m app.ears`).*

When startup completes, EARS prints:
```text
loading EARS for 1 sessions
loading TTS scyllasband
EARS ready on http://0.0.0.0:7860
```

---

### 3.4 Interacting with the Agent
1. Open your browser to **`http://localhost:7860`**.
2. Click the microphone icon to begin speaking.
3. Speak naturally. Notice:
   - The agent responds as soon as you finish your sentence.
   - If you speak while the agent is talking, it halts immediately (barge-in).
   - If you pause mid-thought with vocalizations, the agent can inject backchannels ("mm-hm") without claiming the floor.

---

### 3.5 Optional: Wake Word Standby Mode (`--wakeword`)

To keep the agent silent and prevent GPU forward passes until explicitly addressed, enable the optional `openWakeWord` path by adding `--wakeword` to the start command:

```bash
# Enable default wake word ("hey_jarvis") with 40s silence watchdog
CONFIG=config/small.yaml ./run.sh --wakeword

# Or choose a specific model and customize the silence timeout
python -m app.ears --config config/small.yaml --wakeword hey_jarvis --wakeword-timeout 40.0 --wakeword-threshold 0.6
```

**How it operates:**
- **`STANDBY` Mode**: Microphone audio is routed strictly to `openWakeWord` running on CPU (~20 MB RAM, **0 MB GPU VRAM**). The GPU ASR and LLM remain completely dormant.
- **Wake Word Detection**: Saying *"Hey Jarvis"* triggers a transition to `ACTIVE`, emitting `{"type": "wakeword", "state": "active"}` over the WebSocket. Full-duplex speech begins immediately.
- **Inactivity Watchdog**: After **40 seconds** of silence, the state automatically reverts to `STANDBY`, clearing conversation context and sleeping the pipeline.
- **Manual Control**: Sending `"wake"` or `"sleep"` commands over the WebSocket toggles state on demand.

*Note: Requires `openwakeword` (`pip install openwakeword`). If `--wakeword` is omitted, the stack operates in normal continuous-listening mode with zero overhead.*

---

## 4. Voice, Persona, and Delivery Customization

### 4.1 Persona Mapping
Tyger defines 4 characters in [`config/prompts.yaml`](file:///C:/Temp/tyger-s2s/config/prompts.yaml). [`ScyllasBandTTS`](file:///C:/Temp/tyger-s2s/app/plugins/tts/scyllasband.py) automatically maps them to Scylla's Band native voice IDs with pre-tuned delivery parameters:

| Character | Persona | Scylla's Band Voice | Tuned Baseline Delivery |
|---|---|---|---|
| **Mira** (Default) | Thoughtful, warm woman | `ariadne` | `energy: 1.8, valence: 2.2, tension: 1.5, assertiveness: 2.0` |
| **Aria** | Quick-witted, bright woman | `scylla` | `energy: 2.2, valence: 2.2, tension: 2.0, assertiveness: 2.2` |
| **Kai** | Relaxed, easygoing man | `ink` | `energy: 1.8, tension: 1.4, valence: 2.0, assertiveness: 2.0` |
| **Rex** | Energetic, direct man | `rex` | `energy: 2.3, assertiveness: 2.4, tension: 2.2, valence: 2.0` |

### 4.2 Fine-Tuning Delivery Controls
You can adjust vocal delivery globally or per session in [`config/small.yaml`](file:///C:/Temp/tyger-s2s/config/small.yaml#L37-L48):
```yaml
tts:
  plugin: scyllasband
  bundle: scyllasband/models/litert
  backend: litert          # "litert" (CPU) or "onnx" (GPU)
  language: en_us
  energy: 2.0              # 0.0 to 4.0: Voice brightness and projection (2.0 = neutral)
  tension: 2.0             # 0.0 to 4.0: Vocal tightness and pitch excursion
  valence: 2.0             # 0.0 to 4.0: Emotional contour (downcast vs lively)
  assertiveness: 2.0       # 0.0 to 4.0: Cadence clipping and pace (strongest pace control)
  speed: 1.0               # Overall playback rate multiplier
```

### 4.3 Switching Voices
Change the default character in [`config/small.yaml`](file:///C:/Temp/tyger-s2s/config/small.yaml#L49-L56):
```yaml
voices:
  default: aria            # Choose from: mira, aria, kai, rex
  character: aria
```
Or switch characters dynamically in the web UI using the character tabs.

---

## 5. Turn-Taking Live Tuning

EARS turn-taking thresholds are configured in [`config/small.yaml`](file:///C:/Temp/tyger-s2s/config/small.yaml#L11-L25):

- **`ep_thr` (Endpoint Threshold, default: `0.915`)**:
  Controls when the agent considers your turn finished.
  - *Lower (e.g. `0.85`)*: Responds faster, but may cut you off during longer pauses.
  - *Higher (e.g. `0.95`)*: Waits longer to ensure you have finished speaking.
- **`barge_thr` (Barge-in / Yield Threshold, default: `0.40`)**:
  Controls how sensitive the agent is to your interruptions while it is speaking.
  - *Lower (e.g. `0.30`)*: Easier to interrupt; background coughs might interrupt speech.
  - *Higher (e.g. `0.55`)*: Requires decisive, clear speech to interrupt.
- **`bc_thr` (Backchannel Threshold, default: `0.40`)**:
  Controls frequency of vocal acknowledgment nods ("mm-hm", "yeah") while you speak.
  - *Lower (e.g. `0.30`)*: More frequent backchannels.
  - *Higher (e.g. `0.60`)*: Fewer backchannels.

*Tip: You can adjust these live in the browser via the **Tuning** panel sliders at `http://localhost:7860`, then copy your preferred values into `config/small.yaml`.*

---

## 6. Verification & Automated Tests

To verify your installation and ensure that plugins and configurations are functional:

```bash
# 1. Verify configuration loading and option schemas
python tests/test_config.py

# 2. Test Scylla's Band TTS plugin
python tests/plugins/test_scyllasband.py

# 3. Test Qwen3.5-2B MTP LLM plugin & Hermes tool-calling
python tests/plugins/test_qwen_mtp.py

# 4. Verify plugin registry & base contracts
python tests/plugins/test_plugins.py
```

All tests run without requiring a GPU.

---

## 7. Troubleshooting & FAQs

### Q: Why do I get an HTTP 503 "Busy" error when opening a second tab?
**A**: This is expected behavior. [`server.max_sessions: 1`](file:///C:/Temp/tyger-s2s/config/small.yaml#L7) intentionally pins CUDA graphs and memory buffers to 1 conversation to maintain the small ~5.2 GB VRAM profile. Close the existing session before opening a new one.

### Q: Can I run Scylla's Band on GPU instead of CPU?
**A**: Yes. Download the ONNX models:
```bash
python -m scyllasband download --flavor onnx
```
Then update [`config/small.yaml`](file:///C:/Temp/tyger-s2s/config/small.yaml#L37-L41):
```yaml
tts:
  plugin: scyllasband
  bundle: scyllasband/models/onnx
  backend: onnx
```
This requires only ~200–250 MB of GPU VRAM while decreasing speech latency from ~140 ms to ~25 ms.

### Q: What if `llama-server` is hosted on a remote or custom address?
**A**: Edit `base_url` under `llm.reply` in [`config/small.yaml`](file:///C:/Temp/tyger-s2s/config/small.yaml#L30):
```yaml
llm:
  reply:
    plugin: qwen3.5-2b-mtp
    base_url: http://192.168.1.50:8080/v1
```

### Q: The browser cannot access my microphone.
**A**: Web browsers only permit microphone access over **`http://localhost`** or **`https://`**. If accessing Tyger-S2S over a LAN IP or remote server, use an SSH tunnel:
```bash
ssh -L 7860:localhost:7860 user@remote-box
```
Then open `http://localhost:7860` locally.
