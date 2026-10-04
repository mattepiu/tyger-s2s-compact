"""Plugin interfaces (PLUGINS.md). MOUTH owns the conversation; a plugin owns one model.

Neutral formats, owned by MOUTH:
  * messages: OpenAI-style dicts. {"role": "system" | "user" | "assistant", "content": str}; an assistant tool
    call {"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "arguments": dict}],
    "native": opaque (optional, written and read by the same plugin)}; a tool result {"role": "tool",
    "tool_call_id", "content": str}. Tool-call IDs come from the plugin and are unique for the session.
  * tools: the OpenAI function schema of app/tools.py; plugins convert it.
  * LLM events, yielded by LLMPlugin.stream:
      ("text", delta)                               speakable text, never tool-call or thinking markup
      ("tool_start", {"name": str | None})          the model committed to a tool call (MOUTH starts the filler)
      ("tool_calls", [{"id", "name", "arguments"}], native)   once, at the end
      ("refusal", category | None)                  the model declined (hosted plugins only)

Streams are plain generators that poll stop_flag between items; a plugin aborts its own request when it turns
true. Constructors import their heavy dependencies (vllm, vllm_omni, SDKs) lazily, so a config that uses only
hosted plugins never loads vLLM; a plugin CLASS must stay light to import (EARS imports the TTS class only for
list_voices).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np

LLMEvent = tuple


@dataclass(frozen=True)
class TTSCaps:
    cloning: bool = False                  # takes reference clips from voices/ (never sent off the box)
    max_batch: int = 1                     # requests it decodes together
    languages: list[str] | None = None     # None = whatever the model speaks
    min_first_chars: int | None = None     # chunker hint for hosted TTS (request overhead); None = default


@dataclass
class PluginContext:
    """What every plugin constructor gets besides its config block."""
    eloop: Any                             # app.engines.EngineLoop: MOUTH's asyncio loop for engines and clients
    cfg: dict                              # the whole config (paths, voices.dir, ...)
    name: str                              # instance name: the block's `name:`, else the plugin name
    extra: dict = field(default_factory=dict)


class LLMPlugin(Protocol):
    remote: bool                           # True: transcripts leave the box (logged once at startup, by name)

    def stream(self, messages, tools=None, max_new=192, stop_flag=None) -> Iterator[LLMEvent]: ...
    def warmup(self) -> None: ...
    def shutdown(self) -> None: ...


class TTSPlugin(Protocol):
    out_sr: int                            # native rate; MOUTH resamples to the 24 kHz wire if different
    caps: TTSCaps
    remote: bool

    @classmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]: ...   # (names, default) WITHOUT loading the model
    def load_voice(self, name) -> Any: ...                            # opaque handle; None if unknown
    def stream(self, text, voice, stop_flag=None) -> Iterator[np.ndarray]: ...   # float32 mono at out_sr
    def warmup(self, voice) -> None: ...
    def shutdown(self) -> None: ...


class EarsPlugin(Protocol):
    """Turn-taking for up to N conversations. A session (from open) has reset(): start a new turn before its next
    hop. feed() takes one hop of user mic audio and of the agent's own audio (the 2nd ear; zeros when it isn't
    speaking) with mode "listen" | "speaking", and returns events: {"kind": "FRAME" | "ENDPOINT" | "BARGE" |
    "BACKCHANNEL", ...} (ENDPOINT carries "transcript", "turn_s", "by"; BACKCHANNEL "partial")."""
    hop_samples: int                       # user/agent samples per feed() call
    sr: int                                # their sample rate

    def open(self, **thresholds) -> Any | None: ...                   # a session, or None when full
    def close(self, session) -> None: ...
    async def feed(self, session, user_hop, agent_hop, mode, more=False) -> list[dict]: ...
    def full(self) -> bool: ...
    def active(self) -> int: ...
    def thresholds(self, session) -> dict: ...                         # name -> value, this conversation's
    def set_threshold(self, session, name, value) -> Any: ...          # KeyError / ValueError if not settable
