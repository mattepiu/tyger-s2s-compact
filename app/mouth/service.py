"""Mouth: what every conversation shares. The model plugins (LLM roles, TTS) on one engine loop, the prompts per
role, the tool lists, and the turn settings from config. Built once at MOUTH start-up, in an order that matters
(prompts first, then the TTS, then the LLMs, then warm-ups); per-conversation state lives in Session.
"""
from __future__ import annotations
import logging

from app.config import llm_roles
from app.runtime.event_loop import EngineLoop
from app.mouth.messages import reply_prefix
from app.plugins import build_llms, build_tts
from app.prompts import load_prompts
from app.mouth.tasks import TaskSlots
from app.tools import build_tools

log = logging.getLogger("mouth")


class Mouth:
    """Shared by all sessions: the model plugins, prompts and voice loading. Per-conversation state lives in
    Session."""
    def __init__(self, cfg):
        roles = llm_roles(cfg)
        m = cfg["mouth"]
        esc = m.get("escalation") or {}
        self.expert = "smart" in roles and bool(esc.get("enabled", True))   # ask_expert offered, smart answers it
        if "smart" in roles and not self.expert:
            log.warning("llm.smart is configured but mouth.escalation.enabled is false: it is never used")
        self.tasks_cfg = m.get("tasks") or {}                     # background tasks need escalation (the smart model)
        self.tasks_on = self.expert and bool(self.tasks_cfg.get("enabled", False))
        if self.tasks_cfg.get("enabled") and not self.tasks_on:
            log.warning("mouth.tasks.enabled needs llm.smart (and escalation): background tasks are off")
        self.task_slots = TaskSlots(self.tasks_cfg.get("max_global", 4))
        self.context_turns = int(esc.get("context_turns", 4))     # history turns sent along with an escalated turn
        self.max_steps = int(esc.get("max_steps", 3))             # the smart model's own tool rounds per turn
        self.cfg = cfg
        # prompts first: a bad config/prompts.yaml stops start-up before minutes of engine loading
        by_name = load_prompts(cfg, {name for name, _ in roles.values()})
        self.prompts = {role: by_name[name] for role, (name, _) in roles.items()}   # role -> Prompts
        self.tools = build_tools(self.prompts["reply"], expert=self.expert, tasks=self.tasks_on)
        self.smart_tools = build_tools(self.prompts["smart"]) if self.expert else None   # lookups, no ask_expert
        self.task_tools = build_tools(self.prompts["smart"], finish=True) if self.tasks_on else None
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.bc_min_words = int(m.get("bc_min_words", 4))
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.first_chunk = m.get("first_chunk", "sentence")         # chunker.FIRST_CHUNK_RULES (checked at config load)
        self.eloop = EngineLoop()
        # TTS first: its engine config pins a small KV budget; the LLM's start-up check then sees what's left.
        self.tts = build_tts(cfg, self.eloop)            # at the 24 kHz wire rate
        self.llms = build_llms(cfg, self.eloop)          # role -> LLM plugin instance
        self.sr = self.tts.out_sr
        default, voice = self.default_voice()
        log.info("warming TTS (voice=%s)", default)
        self.tts.warmup(voice)
        for llm in {id(x): x for x in self.llms.values()}.values():
            llm.warmup()
        # first request with the tools prompt is cold (~490 ms to first token vs ~55 ms after) -> pay it here
        p = self.prompts["reply"]
        warm = reply_prefix(p, self.expert, self.tasks_on) + [{"role": "user", "content": p.text("warmup", "user")}]
        for _ in self.llms["reply"].stream(warm, tools=self.tools, max_new=8):
            pass

    def voice(self, name):
        """The TTS plugin's voice handle for `name`, or None (clips added after startup work)."""
        return self.tts.load_voice(name)

    def default_voice(self):
        _, default = self.tts.list_voices(self.cfg)
        return default, (self.voice(default) if default else None)

    def shutdown(self):
        """Shut every engine down, so their worker processes exit too (a bare SIGTERM orphans VoxCPM2's stage
        process). Each LLM instance once, even when roles share it."""
        for shutdown in [self.tts.shutdown] + [x.shutdown for x in {id(x): x for x in self.llms.values()}.values()]:
            try:
                shutdown()
            except Exception as e:
                log.warning("engine shutdown: %s", type(e).__name__)
