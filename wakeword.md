Here is a complete, step-by-step implementation plan for integrating **openWakeWord** into your low-VRAM, Italian **Tyger-S2S** pipeline.

---

# Architecture Plan: Tyger-S2S with openWakeWord & Inactivity Watchdog

```text
                     ┌───────────────────────────────────┐
                     │         Microphone (16kHz)        │
                     └─────────────────┬─────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼ (in STANDBY)                                ▼ (in ACTIVE)
    ┌──────────────────────┐                      ┌──────────────────────┐
    │     openWakeWord     │                      │      Tyger EARS      │
    │   (Runs on CPU)      │                      │  - Nemotron 3.5 ASR  │
    │  ~20 MB RAM, 0 VRAM  │                      │  - 14M TYGER GRU     │
    └───────────┬──────────┘                      └──────────┬───────────┘
                │ Wake word detected                         │
                ▼                                            ▼
   [State Switch: STANDBY -> ACTIVE]              [State: ACTIVE]
   - Play chime / "Dimmi!"                        - Stream text to Quantized LLM
   - Unmute Tyger pipeline                        - Stream audio via OpenAI TTS
   - Start 40s Watchdog Timer                     - Watchdog resets on speech
                                                             │
                                                  Silence > 40 seconds
                                                             │
                                                             ▼
                                              [State Switch: ACTIVE -> STANDBY]
                                              - Clear conversation context
                                              - Mute Tyger / Stop inference
                                              - Resume openWakeWord
```

---

## 1. Component & Memory Blueprint

| Component | Role | Runtime / Device | Estimated VRAM |
| :--- | :--- | :--- | :--- |
| **openWakeWord** | Wake word detection (Standby) | CPU (ONNX / TFLite) | **0 MB** |
| **Nemotron 3.5 ASR** | Cache-aware streaming ASR (`it-IT`) | GPU (FP16 or INT8) | **~1.2 GB** |
| **TYGER GRU** | 14M turn-taking / barge-in detector | GPU (FP16) | **~30 MB** |
| **EARS Buffers** | Streaming cache (**Concurrency = 1**) | GPU | **~300 MB** |
| **Quantized LLM** | Italian-capable LLM (e.g. Qwen 2.5 7B 4-bit) | GPU | **~4.5 GB** |
| **TTS Engine** | Remote OpenAI endpoint (`/v1/audio/speech`) | Cloud / External | **0 MB** |
| **Total System VRAM** | | | **~6.0 GB** |

---

## 2. Step-by-Step Implementation

### Phase 1: Set Up openWakeWord
Both Nemotron and openWakeWord consume **16 kHz, 16-bit mono PCM** audio, so you can feed the same microphone stream directly into both without resampling.

1. **Install dependency:**
   ```bash
   pip install openwakeword
   ```
2. **Choose or train a model:**
   * **Ready-to-use:** Default models include `"hey_jarvis"`, `"alexa"`, etc.
   * **Custom Italian Wake Word:** (e.g., *"Ciao Tigre"* or *"Ehi Assistente"*). You can generate an openWakeWord custom model using their [automated synthetic-training Colab notebook](https://github.com/dscripka/openWakeWord) in about 30 minutes without recording real voices.

---

### Phase 2: Implement the State Machine & Watchdog

Create a controller (`session_manager.py`) managing the transitions between `STANDBY` and `ACTIVE`.

```python
import time
from openwakeword.model import Model

class SessionManager:
    def __init__(self, wake_model_name="hey_jarvis", silence_timeout=40.0):
        self.state = "STANDBY"
        self.silence_timeout = silence_timeout
        self.last_speech_time = 0.0
        
        # Load wake-word engine on CPU
        self.oww = Model(wakeword_models=[wake_model_name], inference_framework="onnx")

    def process_frame(self, pcm_chunk_16k):
        """
        pcm_chunk_16k: raw audio chunk from microphone (16kHz, 16-bit mono)
        """
        now = time.time()

        if self.state == "STANDBY":
            # Pass frame to CPU wake-word detector
            prediction = self.oww.predict(pcm_chunk_16k)
            for model_name, score in prediction.items():
                if score > 0.6:  # Threshold for detection
                    self.activate_session()
                    break

        elif self.state == "ACTIVE":
            # Check 40-second silence watchdog
            if (now - self.last_speech_time) >= self.silence_timeout:
                self.deactivate_session()

    def notify_user_speech(self):
        """Called whenever Tyger EARS or ASR detects actual voice activity."""
        self.last_speech_time = time.time()

    def activate_session(self):
        print("��� [WAKE WORD DETECTED] Session Active.")
        self.state = "ACTIVE"
        self.last_speech_time = time.time()
        self.oww.reset()
        # Trigger activation sound or greeting (e.g. "Dimmi!")

    def deactivate_session(self):
        print("��� [INACTIVITY TIMEOUT] Going to standby.")
        self.state = "STANDBY"
        # 1. Cancel any active LLM generation
        # 2. Reset conversational KV cache / Tyger turn states
        # 3. Optional farewell: "A presto!"
```

---

### Phase 3: Wire into Tyger-S2S Audio Loop

In Tyger-S2S, replace the continuous audio ingestion loop with the controller checks:

1. **Routing Audio by State:**
   * When `state == "STANDBY"`: Route microphone chunks **only** to `openWakeWord.predict()`. Do not send chunks to the LLM or TTS queue.
   * When `state == "ACTIVE"`: Route microphone chunks to Nemotron ASR + Tyger GRU.
2. **Hooking the Inactivity Reset:**
   * In EARS, listen to the predicted turn state. Whenever the Tyger GRU outputs a user-speaking frame (`state == "speak"` or `state == "hold"`), or when Nemotron produces non-empty text, invoke:
     ```python
     session_manager.notify_user_speech()
     ```
   * As long as the user and agent continue talking back and forth, `last_speech_time` keeps refreshing.
   * If 40 continuous seconds pass with no user speech, `session_manager.deactivate_session()` automatically drops back to standby.

---

### Phase 4: Adapt for Low-VRAM Italian Execution

To make the active session fit in ~6 GB VRAM and speak Italian:

1. **Limit Concurrency to 1:**
   In Tyger-S2S configuration files, change the batch/conversation capacity:
   ```yaml
   max_conversations: 1   # Drops Nemotron streaming cache from ~4 GB down to ~1.5 GB
   ```
2. **Configure Nemotron for Italian:**
   Set the language decoding attribute on the FastConformer model:
   ```python
   asr_model.change_decoding_strategy(language_id="it-IT")
   ```
3. **Swap LLM for Quantized Italian Model:**
   In the MOUTH worker, configure a 4-bit AWQ or GGUF quantized model (such as `Qwen/Qwen2.5-7B-Instruct-AWQ` or `Llama-3.1-8B-Instruct-GPTQ`) with an Italian system prompt:
   ```text
   Sei un assistente vocale intelligente e rispondi sempre in modo naturale, conciso e in italiano.
   ```
4. **Swap Local VoxCPM2 with Remote OpenAI TTS:**
   In the TTS synthesis loop, replace the local model call with an asynchronous client call:
   ```python
   from openai import AsyncOpenAI
   client = AsyncOpenAI()

   async def synthesize_chunk(text_chunk):
       response = await client.audio.speech.create(
           model="tts-1",
           voice="alloy",  # or any natural Italian voice
           input=text_chunk,
           response_format="pcm"
       )
       return response.content
   ```
   *Remember to feed the returned playback audio back into Tyger’s agent echo-reference buffer to ensure interruption/barge-in detection stays calibrated.*

---

## 3. Verification & Testing Checklist

- [ ] **Wake Word Sensitivity:** Test `openWakeWord` with background television/noise to tune the threshold (default `0.5`–`0.6` usually eliminates false triggers).
- [ ] **Dual-Stream Feedback:** Confirm that incoming OpenAI TTS audio chunks are piped into the Tyger echo-cancellation channel during playback.
- [ ] **Interruption (Barge-in):** While the agent is speaking via OpenAI TTS, speak over it to confirm Tyger issues the `stop` command, aborts the audio queue, and resets the timer.
- [ ] **Inactivity Watchdog:** Let the conversation idle in a quiet room and confirm that exactly at 40 seconds the state flips to `STANDBY` and GPU consumption drops to idle.

