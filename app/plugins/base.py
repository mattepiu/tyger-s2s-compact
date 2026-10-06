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
    A failed request raises LLMError. A tool call whose arguments don't fit its schema carries "error": MOUTH
    returns that as the tool's result and never runs it.

Streams are plain generators that poll stop_flag between items; a plugin aborts its own request when it turns
true. Constructors import their heavy dependencies (vllm, vllm_omni, SDKs) lazily, so a config that uses only
hosted plugins never loads vLLM; a plugin CLASS must stay light to import (EARS imports the TTS class only for
list_voices).

A plugin subclasses one of the ABC bases below (LLMPlugin, TTSPlugin, EarsPlugin), which hold the shared defaults,
and declares its config block as `Options` (options.py): config.py checks every block against it when the config
loads, so a typo fails at startup with the key named.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

import numpy as np

from app.plugins.options import PluginOptions

LLMEvent = tuple


class LLMError(Exception):
    """A request failed in a way MOUTH answers with a short spoken apology (hosted plugins: rate limit, timeout,
    connection, HTTP error, too many requests in flight). `kind` is for logs; never the request or response."""
    def __init__(self, kind: str):
        super().__init__(kind); self.kind = kind


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


class Plugin(ABC):
    """What every plugin shares. The constructor takes the plugin's config block and a PluginContext (tests may
    pass neither): the block becomes `self.opts`, an instance of the class's `Options`. Nothing leaves the box
    unless `remote` says so; warmup and shutdown do nothing unless overridden."""
    Options: ClassVar[type[PluginOptions]] = PluginOptions
    remote: bool = False                   # True: transcripts leave the box (logged once at startup, by name)

    def __init__(self, conf: dict | None = None, ctx: PluginContext | None = None):
        self.opts = self.options(conf)
        self.ctx = ctx
        self.name = ctx.name if ctx else type(self).__name__
        self.eloop = ctx.eloop if ctx else None

    @classmethod
    def options(cls, block: dict | None) -> PluginOptions:
        """`block` as this plugin's Options; ValueError (or KeyError) naming the key that doesn't fit. config.py
        calls it for every block when the config loads."""
        return cls.Options.from_config(block)

    def shutdown(self) -> None:
        pass


class LLMPlugin(Plugin):
    @abstractmethod
    def stream(self, messages, tools=None, max_new=192, stop_flag=None) -> Iterator[LLMEvent]: ...

    def warmup(self) -> None:
        pass


class TTSPlugin(Plugin):
    out_sr: int                            # native rate; MOUTH resamples to the 24 kHz wire if different
    caps: TTSCaps

    @classmethod
    @abstractmethod
    def list_voices(cls, cfg) -> tuple[list[str], str | None]: ...   # (names, default) WITHOUT loading the model

    @abstractmethod
    def load_voice(self, name) -> Any: ...                            # opaque handle; None if unknown

    @abstractmethod
    def stream(self, text, voice, stop_flag=None) -> Iterator[np.ndarray]: ...   # float32 mono at out_sr

    def warmup(self, voice=None) -> None:
        pass


@dataclass(frozen=True)
class Slider:
    """How the page's live-tuning panel shows a threshold. lo / hi / step are in display units: display = value *
    `scale` (e.g. frames -> seconds), shown with `decimals` and `unit`. `meter` ("p_end", "yield" or "bc": the
    page's meters, fed by FRAME events) gets an orange mark at the value. With `off` set, a checkbox switches the
    threshold off by setting it to that value (and any value >= off reads as off)."""
    label: str
    lo: float
    hi: float
    step: float
    decimals: int = 2
    scale: float = 1.0
    unit: str = ""
    meter: str | None = None
    off: float | None = None


@dataclass(frozen=True)
class Threshold:
    """A per-conversation threshold an EARS plugin lets the page tune: its name (the `set:` key), type, the config
    key its default comes from (the name if None) and the default if config lacks it (None: config must set it), and
    how the page shows it (None: settable, but no slider)."""
    name: str
    type: type                             # int or float
    config_key: str | None = None
    default: float | None = None
    slider: Slider | None = None

    @property
    def key(self) -> str:
        """Its key in the config block."""
        return self.config_key or self.name

    def from_config(self, conf: dict):
        if self.key not in conf and self.default is None:
            raise KeyError(f"`{self.key}` is not set")
        return self.type(conf.get(self.key, self.default))

    def parse(self, value):
        """The value as this threshold's type; ValueError if it isn't one."""
        return self.type(value)

    def slider_spec(self) -> dict:
        """The slider as the page reads it (in the `ready` message)."""
        return {"name": self.name, "type": self.type.__name__, **self.slider.__dict__}


def find_threshold(thresholds, name) -> Threshold:
    """The Threshold called `name`; KeyError if the plugin has none (an unknown `set:` from the page)."""
    for t in thresholds:
        if t.name == name:
            return t
    raise KeyError(name)


class EarsPlugin(Plugin):
    """Turn-taking for up to N conversations. A session (from open) has reset(): start a new turn before its next
    hop. feed() takes one hop of user mic audio and of the agent's own audio (the 2nd ear; zeros when it isn't
    speaking) with mode "listen" | "speaking", and returns events: {"kind": "FRAME" | "ENDPOINT" | "BARGE" |
    "BACKCHANNEL", ...} (ENDPOINT carries "transcript", "turn_s", "by"; BACKCHANNEL "partial"; FRAME "state" and
    the meters' "p_end", "yield_prob", "bc_prob"). Its tunable thresholds are its THRESHOLDS: the page builds its
    sliders from them, so EARS's server and the page know no plugin's threshold names. Their defaults come from
    the config block (beside the Options), and a session holds its own values as attributes of those names."""
    hop_samples: int                       # user/agent samples per feed() call
    sr: int                                # their sample rate
    THRESHOLDS: ClassVar[tuple[Threshold, ...]] = ()

    def __init__(self, conf: dict | None = None, ctx: PluginContext | None = None):
        super().__init__(conf, ctx)
        self.defaults = {t.name: t.from_config(conf or {}) for t in self.THRESHOLDS}

    @classmethod
    def options(cls, block):
        for t in cls.THRESHOLDS:
            t.from_config(block or {})
        return cls.Options.from_config(block, ignore={t.key for t in cls.THRESHOLDS})

    def open(self, **thresholds) -> Any | None:
        """A new conversation's session with the configured thresholds (overridable), or None when full."""
        return self.open_session({**self.defaults, **thresholds})

    def thresholds(self, session) -> dict:
        """name -> value, this conversation's."""
        return {t.name: getattr(session, t.name) for t in self.THRESHOLDS}

    def set_threshold(self, session, name, value):
        """Set one of THRESHOLDS for this conversation; returns the value now in effect. KeyError for an unknown
        name, ValueError for a bad value."""
        v = find_threshold(self.THRESHOLDS, name).parse(value)
        setattr(session, name, v)
        return v

    @abstractmethod
    def open_session(self, thresholds: dict) -> Any | None: ...       # a session with these values, or None

    @abstractmethod
    def close(self, session) -> None: ...

    @abstractmethod
    async def feed(self, session, user_hop, agent_hop, mode, more=False) -> list[dict]: ...

    @abstractmethod
    def full(self) -> bool: ...

    @abstractmethod
    def active(self) -> int: ...
