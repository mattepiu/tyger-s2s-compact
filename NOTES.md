# tyger v1 — build notes

## Box
NVIDIA RTX 5090 (32 607 MiB, Blackwell sm_120), driver 590.48 / CUDA 13.1; AMD 96-core CPU, 188 GB RAM,
100 GB disk. Fresh container each boot → full rebuild. `HF_HOME=/workspace/.hf_home`, `HF_HUB_DISABLE_XET=1`,
torch **2.9.1+cu130**.

## Architecture decision — three venvs, three processes (V1.2: LLM on vLLM, TTS = MOSS-TTS-Realtime)
A single venv is impossible: **nemotron-3.5-asr-streaming** needs **transformers ≥ 5.13** (its AutoProcessor
isn't in 4.57); **MOSS-TTS-Realtime**'s `trust_remote_code` modeling targets **transformers 5.0.0**; and vLLM
pins its own **torch 2.13**. None can share a venv, so:
- **RESPONDER** venv (`venv_vllm`, vLLM 0.30 / torch 2.13): Qwen3-4B-Instruct-2507 served by **vLLM** on an
  OpenAI API (`:8001`, `--enable-auto-tool-choice --tool-call-parser hermes`). The LLM runs here.
- **MOUTH** venv (`venv_moss`, tf 5.0.0 / torch 2.9.1+cu130): the orchestrator + **in-process** MOSS-TTS-Realtime.
  Talks to the responder over HTTP, chunks the reply into **sentences**, streams TTS audio to EARS over TCP (`:8770`).
- **EARS** venv (`venv`, tf 5.17): nemotron L15 + the exp-10 chain + endpoint head (arm C). Serves the web app.
- Wired by the exp-12 TCP protocol (`app/wire.py`, frames `T/C/B/V → X/A/D`). `run.sh` starts all three.

**Why the LLM is on vLLM.** The old MOUTH ran the responder in-process with a greedy token-by-token loop — the
latency drag the user flagged. On this box Qwen3-4B on vLLM gives TTFT **28 ms**, **~150 tok/s**, with structured
Hermes `tool_calls`. (vLLM-Omni was tried for TTS and was *slower* than in-process, so TTS is NOT on vLLM.)

**Why MOSS-TTS-Realtime, and the fast-path patch.** The user found MOSS-TTS-Nano's voice "weird" and asked to
go back to Realtime (the exp-10/11 TTS). Realtime sounds better but its stock remote code decodes each audio
frame through the Qwen3 backbone with a **DynamicCache + growing mask + no cache_position** — shapes change every
step, so it can't be CUDA-graphed and runs at **RTF ~1.6–2.0** on this 5090 (unusable for streaming). Fix
(`weights.ensure_moss_realtime` patches the vendored `streaming_mossttsrealtime.py`, idempotent): a **bs=1 fast
path** that prefills/steps through a preallocated **StaticCache** with explicit `cache_position` and a pure-causal
mask, and runs the single-token backbone step through **`torch.compile(mode="reduce-overhead")`** (CUDA graphs).
With a long boot warmup (to capture the graph depth once, up front) this holds a stable **RTF ~0.55–0.7**
(DEC=4 ≈ 0.67, DEC=8 ≈ 0.56); audio stays finite/clean. End-to-end plain first-audio **~480 ms** (vs Nano's
261 ms — the quality trade the user chose; lower `prefill_text_len`/`dec_chunk_frames` at V2 to cut it).
(Alternatives weighed earlier: Nano is faster-onset but "weird"; vLLM-Omni slower; OmniVoice non-commercial.)

## What was copied from exp-12, and what was dropped
EARS (`app/ears_core/`) is exp-12 **arm C** copied and flattened (no `codebase/` imports): nemotron encoder,
StateChannelModel/YieldModel/EndpointModel, LiveEars, EndpointEars, plus a minimal `policy._decide` (the full
`fdb_eval`/VAD harness isn't needed live). **Dropped vs exp-12**: the 4-way endpointer, the SPEAK-lock
watchdog, affect/gender, A/B blinding + the sealed code→ckpt map, and all per-session transcript/frame logging
(privacy). MOUTH keeps exp-12's Hermes lead-in + phase-B flow but the responder is now an **HTTP client to the
vLLM server** (`app/mouth.py` `LLMClient`, OpenAI `/v1/chat/completions` streaming; tool calls arrive as
structured `tool_calls`, not inline `<tool_call>` text), the **TTS is in-process MOSS-TTS-Realtime** (see below),
and the reply is chunked into **whole sentences** (`chunker.next_sentence`). **Backchannels are contextual LLM
acks** (`mouth.contextual_acks: true`, the user's choice): a short ack the LLM generates from the ASR partial
transcript (EARS forwards the partial on the backchannel event), so it reacts to what the user is saying;
canned acks remain as a `false` fallback. reminder/message tools return a "not connected" demo marker so the
model says so.

## MOSS-TTS-Realtime integration (app/tts_realtime.py + app/weights.py; vendor the HF snapshot and patch it,
## since trust_remote_code re-downloads and would revert HF-cache edits)
- Model `OpenMOSS-Team/MOSS-TTS-Realtime` + codec `OpenMOSS-Team/MOSS-Audio-Tokenizer` (Apache-2.0), **24 kHz
  native** (no resample to the 24 kHz wire). Backbone is Qwen3-1.7B + a local (depth) transformer over 16 RVQ
  codebooks. Streaming session: `begin → push_text(sentence) → end → drain → flush`; `tts_realtime.stream(text)`
  runs one session per sentence and yields float32 chunks as they decode.
- The ref clip is codec-encoded **once per `set_voice`** (trimmed to `prompt_max_s`=4 s) and reused every turn;
  read with **soundfile** (not torchaudio.load → avoids torchcodec). No audio/transcript to disk.
- **Static-cache / CUDA-graph fast path** (`weights._patch_realtime_streaming`, idempotent; marker
  `tyger: static-cache`): rewrites the Inference `prefill`/`step` to use a preallocated `StaticCache` +
  `cache_position` + pure-causal mask (bs=1) and compiles the single-token backbone step with
  `torch.compile(mode="reduce-overhead")`. Eager was RTF ~1.6–2.0; this is **~0.55–0.7**. **Boot warmup must run
  2–3 long sentences** (`MossRealtimeTTS.warmup`) so the CUDA-graph depth is captured once up front — otherwise a
  reply longer than anything seen so far triggers a one-off recapture spike (RTF ~3) mid-conversation.
- Levers: `dec_chunk_frames` (4 → RTF ~0.67 + lower onset; 8 → ~0.56), `prefill_text_len` (6; lower = sooner
  first audio), `max_length` (StaticCache length, 800). bf16/sdpa.
- `app/tts_nano.py` (the Nano backend) is kept but **dormant** (not imported); it needs the old tf-4.57 venv.

## V0 — MOSS-TTS-Nano bench (bundled `en_2` ref, bf16/sdpa, co-loaded w/ 2507 + nemotron resident)
| metric | MOSS-Nano | bar | MOSS-Realtime (exp-12) | Chatterbox (exp-12) |
|---|---|---|---|---|
| first-audio p50 / p90 | **112 / 117 ms** ✅ | ≤250 | ~1036 ms | ~781 ms |
| CER (nemotron round-trip) | **1.17% / 0.0% med** ✅ | ≤~6.4% | ~5.4% | — |
| RTF | 0.70 non-stream / 0.94 stream ❌ | ≤0.5 | ~0.78 | ~0.40 |

Onset is the win (1036 → 112 ms) and CER improves; **RTF misses 0.5**, bottlenecked in the AR audio-token
generation (codec fp16 vs fp32 = no change), so not cheaply fixable (torch.compile untried). RTF < 1.0, so
streaming stays ahead of playback. **User decision: accept Nano, proceed.**

## V1.2 — build + run on the box (vLLM LLM + MOSS-TTS-Realtime)
- Three processes boot from fresh venvs built off `requirements-{vllm,moss,ears}.txt`. `run.sh` starts the
  vLLM responder (`:8001`), waits for `/v1/models`, starts MOUTH (`:8770`), then EARS + web app (`:7860`).
- **smoke_test.py PASS**: MOUTH plain reply **first-audio ~480 ms**; **Tavily** weather turn (first-audio
  ~577 ms); a **placeholder** reminder turn that admits it isn't connected; contextual acks
  ("walking downtown…"→"oh cool", "I got the job"→"Wow, really?"); EARS WS wav → endpoint fired → reply audio.
  Steady TTS RTF ~0.56–0.7 (Realtime static-cache fast path).
- **GPU memory** (warm, all three loaded): **~30.5 GB / 32 GB** — MOUTH/Realtime ~12.7 GB (CUDA-graph pools),
  vLLM-Qwen ~11.8 GB @ `--gpu-memory-utilization 0.37` + `--max-model-len 4096`, EARS 5.4 GB. **Tight** (~2 GB
  headroom); stable across turns (all pre-allocated). `VLLM_GPU_FRAC` must stay ≤ ~0.37 (0.28 → negative KV).
- **Startup** (warm cache): vLLM ~1–5 min, MOUTH ~2.5 min (Realtime load + codec + 3-sentence CUDA-graph warmup;
  the first warmup sentence triggers a ~30–60 s graph capture), EARS ~15 s. Cold cache adds the model download.

## Still open (later milestones)
- V2: retune `ep_thr`/`ep_deb` + barge/bc live with `?debug=1` (exp-12's 0.9146 was never retuned live; exp-11
  found offline thresholds ~2× high on the browser mic). Freeze into `config.yaml`.
- V3: Docker build/run + container smoke.
