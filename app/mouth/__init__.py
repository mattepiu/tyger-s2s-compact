"""MOUTH: the responder LLM and the TTS, both in-process on vLLM engines, plus the turn orchestration.

Architecture:
  * The models are plugins (app/plugins, PLUGINS.md), chosen in config. LLM roles: `reply` (turns and the tool
    phase-B answer) and `filler` (backchannels, tool lead-ins, still-looking lines), the same instance unless
    configured apart. Local plugins run in-process on a shared asyncio loop (app/runtime/event_loop.py): the local LLMs on
    vLLM AsyncLLM, VoxCPM2 on vLLM-Omni AsyncOmni. No HTTP between MOUTH and the models.
    They can't be one vLLM-Omni pipeline: MOUTH has to sit between them (chunking, tools + lead-in,
    backchannels, barge-in).
  * The LLM reply is chunked by chunker.next_chunk: the FIRST chunk may end at a clause boundary (comma,
    semicolon, dash) so playback doesn't wait for a long opening sentence, then whole sentences. No chunk
    is cut below MIN_CHARS (40), so short replies/sentences go to the TTS whole. Each chunk is one VoxCPM2
    request whose audio streams out per decode step, so playback starts on chunk 1 while the rest render.
  * Concurrency: the engines are shared; each EARS connection is its own Session (history, voice, recent
    acks) on its own thread, and within a Session turns and backchannels run on separate workers, each as
    their own engine requests (vLLM batches them; llm.max_num_seqs, config/engines/voxcpm2.yaml max_num_seqs).
  * Barge-in (B on the wire) stops the session's current turn; its live LLM/TTS request is aborted.
  * reminder/message tools return a "not connected" demo marker (app/tools/stubs.py) and the model says so.
  * Escalation (only with an `llm.smart` block): the reply model gets the ask_expert tool. Its lead-in fits the
    tool (the plugins name it at tool_start), the smart request starts as soon as the call parses, and the smart
    model's answer streams straight to the TTS (its own lookups covered by still-looking lines). Only escalated
    turns and their last mouth.escalation.context_turns turns of transcript go to the smart model.
  * Backchannels are always contextual LLM acks off the ASR partial: a reaction to the content, or a neutral
    hum when the partial is too short (< mouth.bc_min_words) to react to. No canned/pre-rendered acks.
EARS<->MOUTH wire: T/C/B/V in; X/A/D out, plus K = backchannel audio (kept apart from reply audio A, since the
two can now be generated at the same time). Background tasks (step 13): F out = a finished task asks for the floor;
G in = EARS granted it, and the announcement runs as a turn (A audio, then D with "proactive": task id). S in = speak this text, TTS only (no LLM, no history): EARS never
sends it; test tools use it for a fixed utterance (scripts/stack/load_ears.py). Audio is 24 kHz float32. No audio or
transcript is written to disk or logs; only timings and event types.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.mouth

Modules (app/mouth/):
  __main__  the process: load Mouth, accept EARS connections        service  Mouth: plugins, prompts, tool lists
  session   Session: one conversation's state, wire, workers        history  History: turns, trimmed whole
  turns     Turn base, AnnouncementTurn, SayTurn                     reply    ReplyTurn: plain, tools, escalation
  speech    the path to the TTS, Speaker (one reply's audio)          threads  Prefetch, spawn (helper threads)

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.mouth
"""
from app.mouth.service import Mouth
from app.mouth.session import Session

__all__ = ["Mouth", "Session"]
