"""MOUTH process: the responder LLM and the TTS, both in-process on vLLM engines, plus the turn orchestration.

Architecture:
  * The models are plugins (app/plugins, PLUGINS.md), chosen in config. LLM roles: `reply` (turns and the tool
    phase-B answer) and `filler` (backchannels, tool lead-ins, still-looking lines), the same instance unless
    configured apart. Local plugins run in-process on a shared asyncio loop (app/engines.py): the Qwen LLMs on
    vLLM AsyncLLM, VoxCPM2 on vLLM-Omni AsyncOmni. No HTTP between MOUTH and the models.
    They can't be one vLLM-Omni pipeline: MOUTH has to sit between them (chunking, tools + lead-in,
    backchannels, barge-in).
  * The LLM reply is chunked by chunker.next_chunk: the FIRST chunk may end at a clause boundary (comma,
    semicolon, dash) so playback doesn't wait for a long opening sentence, then whole sentences. No chunk
    is cut below MIN_CHARS (40), so short replies/sentences go to the TTS whole. Each chunk is one VoxCPM2
    request whose audio streams out per decode step, so playback starts on chunk 1 while the rest render.
  * Concurrency: the engines are shared; each EARS connection is its own Session (history, voice, recent
    acks) on its own thread, and within a Session turns and backchannels run on separate workers, each as
    their own engine requests (vLLM batches them; llm.max_num_seqs, deploy/voxcpm2.yaml max_num_seqs).
  * Barge-in (B on the wire) stops the session's current turn; its live LLM/TTS request is aborted.
  * reminder/message tools return a "not connected" demo marker (tools.py) and the model says so.
  * Escalation (only with an `llm.smart` block): the reply model gets the ask_expert tool. Its lead-in fits the
    tool (the plugins name it at tool_start), the smart request starts as soon as the call parses, and the smart
    model's answer streams straight to the TTS (its own lookups covered by still-looking lines). Only escalated
    turns and their last mouth.escalation.context_turns turns of transcript go to the smart model.
  * Backchannels are always contextual LLM acks off the ASR partial: a reaction to the content, or a neutral
    hum when the partial is too short (< mouth.bc_min_words) to react to. No canned/pre-rendered acks.
EARS<->MOUTH wire: T/C/B/V in; X/A/D out, plus K = backchannel audio (kept apart from reply audio A, since the
two can now be generated at the same time). S in = speak this text, TTS only (no LLM, no history): EARS never
sends it; test tools use it for a fixed utterance (load_test_ears.py). Audio is 24 kHz float32. No audio or
transcript is written to disk or logs; only timings and event types.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.mouth
"""
from __future__ import annotations
import os, sys, socket, signal, threading, queue, json, time, logging
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from app.wire import send_msg, recv_msg
from app.tools import build_tools, execute_tool, EXPERT
from app.chunker import next_chunk, say_clean
from app.config import load_config, llm_roles
from app.prompts import load_prompts
from app.engines import EngineLoop
from app.plugins import build_llms, build_tts
from app.plugins.base import LLMError

log = logging.getLogger("mouth")

# Every prompt is in prompts.yaml (app/prompts.py): system prompts, few-shot turns, the templates below, the
# history markers and the tool descriptions. None of that text belongs in code.
FILLER_GAP_S = 0.8         # tools still out this long after the filler has played -> one more "still looking" line
MAX_STILL = 2              # at most this many "still looking" lines per tool turn
RECENT_ACKS = 3
_END = object()


def _with_shots(system, shots, user):
    msgs = [{"role": "system", "content": system}]
    for q, a in shots:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": user}]


def still_user(p, transcript, fillers, section="still_looking"):
    """A still-looking request's user line (`section`: still_looking or expert_still): what was asked and the
    lines already spoken."""
    said = " ".join(f"'{f}'" for f in fillers) or p.text(section, "said_none")
    return p.fill(section, "user", transcript=transcript, said=said)


def bc_messages(p, recent_acks, partial, bc_min_words):
    """The backchannel request: the system prompt (+ the acks not to repeat), few-shot turns, and the partial
    framed as empty / only a few words / a reaction-worthy utterance."""
    system = p.text("backchannel", "system")
    if recent_acks:
        system += p.fill("backchannel", "no_repeat", acks=", ".join(f"'{a}'" for a in recent_acks))
    user = (partial or "").strip()
    if not user:
        said = p.text("backchannel", "user_empty")
    elif len(user.split()) < bc_min_words:
        said = p.fill("backchannel", "user_short", partial=user)
    else:
        said = p.fill("backchannel", "user", partial=user)
    return _with_shots(system, p.shots("backchannel"), said)


def _text(events):
    return "".join(e[1] for e in events if e[0] == "text")


def expert_result(p):
    """The ask_expert tool result as kept in history (the expert's answer follows it)."""
    return _compact_result({"result": p.text("escalation", "tool_result")})


def reply_prefix(p, expert):
    """The reply request before the conversation: the system prompt, and with escalation on, its few-shot turns
    (prompts.yaml escalation.shots) as earlier turns in the neutral format. A fixed prefix, so it stays cached."""
    if not expert:
        return [{"role": "system", "content": p.text("system", "voice_tools")}]
    msgs = [{"role": "system", "content": p.text("escalation", "system")}]
    for i, t in enumerate(p.turns("escalation"), 1):
        msgs.append({"role": "user", "content": t["user"]})
        if "reply" in t:
            msgs.append({"role": "assistant", "content": t["reply"]})
            continue
        cid = f"call_shot{i:04d}"
        msgs += [{"role": "assistant", "content": None,
                  "tool_calls": [{"id": cid, "name": EXPERT, "arguments": {"question": t["expert"]}}]},
                 {"role": "tool", "tool_call_id": cid, "content": expert_result(p)},
                 {"role": "assistant", "content": t["answer"]}]
    return msgs


def smart_messages(p, turns, transcript, question, others=()):
    """The smart model's request for an escalated turn: its system prompt, the given recent turns as plain text
    (tool calls and results dropped: provider-neutral, and less leaves the box), then the user's words with the
    reply model's restated question as a note, plus what other tools called in the same turn returned."""
    ctx = [{"role": x["role"], "content": x["content"]} for turn in turns for x in turn
           if x["role"] in ("user", "assistant") and not x.get("tool_calls") and x.get("content")]
    user = p.fill("smart", "user", transcript=transcript, question=question or transcript)
    if others:
        user += p.fill("smart", "results", results="; ".join(f"{n}: {_compact_result(r)}" for n, r in others))
    return [{"role": "system", "content": p.text("smart", "system")}] + ctx + [{"role": "user", "content": user}]


def round_messages(content, calls, native, results, compact=False):
    """One tool round in the neutral format: the assistant's calls (plugin-issued IDs; `native` passed through)
    and a result per call, full or compacted for history."""
    asst = {"role": "assistant", "content": (content or "").strip() or None,
            "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c.get("arguments", {})} for c in calls]}
    if native is not None:
        asst["native"] = native
    res = (lambda r: _compact_result(r)) if compact else \
        (lambda r: json.dumps(r if r is not None else {"error": "no result"}))
    return [asst] + [{"role": "tool", "tool_call_id": c["id"], "content": res(r)} for c, r in zip(calls, results)]


class _Prefetch:
    """An LLM stream running on a helper thread from now on, so it is under way while this one speaks; iterate it
    here, in order. `ready` is set at its first event (or when it ends), `t_first` is when that event came. An
    exception in the stream is raised to the reader; a stop ends the iteration."""
    def __init__(self, gen_fn, stop_flag):
        self.q: queue.Queue = queue.Queue(); self.ready = threading.Event(); self.stop_flag = stop_flag
        self.t_first = None
        threading.Thread(target=self._run, args=(gen_fn,), daemon=True).start()

    def _run(self, gen_fn):
        try:
            for x in gen_fn():
                if self.t_first is None:
                    self.t_first = time.perf_counter()
                self.q.put(x); self.ready.set()
        except Exception as e:
            self.q.put(e)
        finally:
            self.q.put(_END); self.ready.set()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                x = self.q.get(timeout=0.01)
            except queue.Empty:
                if self.stop_flag():
                    raise StopIteration
                continue
            if x is _END:
                self.q.put(_END)                       # stays ended
                raise StopIteration
            if isinstance(x, Exception):
                raise x
            return x


def _compact_result(r, cap=600):
    """A tool result as kept in history: the answer without the raw search hits, capped. Enough for the model to
    see that it called a tool and what came back, so later turns keep calling tools instead of answering live
    questions from memory (I-12)."""
    if isinstance(r, dict):
        r = {k: v for k, v in r.items() if k not in ("results", "sources")}
    txt = json.dumps(r if r is not None else {"error": "no result"}, ensure_ascii=False)
    return txt if len(txt) <= cap else txt[:cap] + "…"


class Mouth:
    """Shared by all sessions: the model plugins, prompts and voice loading. Per-conversation state lives in
    Session."""
    def __init__(self, cfg):
        roles = llm_roles(cfg)
        m = cfg["mouth"]
        if (m.get("tasks") or {}).get("enabled"):
            raise NotImplementedError("background tasks (mouth.tasks) come in PLUGINS.md steps 12-14")
        esc = m.get("escalation") or {}
        self.expert = "smart" in roles and bool(esc.get("enabled", True))   # ask_expert offered, smart answers it
        if "smart" in roles and not self.expert:
            log.warning("llm.smart is configured but mouth.escalation.enabled is false: it is never used")
        self.context_turns = int(esc.get("context_turns", 4))     # history turns sent along with an escalated turn
        self.max_steps = int(esc.get("max_steps", 3))             # the smart model's own tool rounds per turn
        self.smart_max_new = int(esc.get("max_new", 384))
        self.cfg = cfg
        # prompts first: a bad prompts.yaml stops start-up before minutes of engine loading
        by_name = load_prompts(cfg, {name for name, _ in roles.values()})
        self.prompts = {role: by_name[name] for role, (name, _) in roles.items()}   # role -> Prompts
        self.tools = build_tools(self.prompts["reply"], expert=self.expert)
        self.smart_tools = build_tools(self.prompts["smart"]) if self.expert else None   # lookups, no ask_expert
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.bc_min_words = int(m.get("bc_min_words", 4))
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.eloop = EngineLoop()
        # TTS first: its deploy config pins a small KV budget; the LLM's start-up check then sees what's left.
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
        warm = reply_prefix(p, self.expert) + [{"role": "user", "content": p.text("warmup", "user")}]
        for _ in self.llms["reply"].stream(warm, tools=self.tools, max_new=8):
            pass

    def voice(self, name):
        """The TTS plugin's voice handle for `name`, or None (clips added after startup work)."""
        return self.tts.load_voice(name)

    def default_voice(self):
        _, default = self.tts.list_voices(self.cfg)
        return default, (self.voice(default) if default else None)


class Session:
    """One conversation = one EARS connection: history, active voice, recent acks. Turns and backchannels run
    on separate worker threads, so a backchannel is generated while a turn is (their own engine requests);
    several Sessions run at once on the shared engines. Barge (B) stops the current turn only."""
    def __init__(self, m: Mouth, conn):
        self.m = m; self.conn = conn
        self.send_lock = threading.Lock()
        self.turns = []                    # conversation history, one message list per turn (trimmed whole)
        self.recent_acks = deque(maxlen=RECENT_ACKS)   # in memory only; so backchannels don't repeat
        self.voice_name, self.voice = m.default_voice()  # every session starts on the configured default
        self.turn_stop = threading.Event()
        self.closed = threading.Event()
        self.turn_seq = 0                  # T frames seen; a backchannel queued before the latest T is stale
        self.barge_played_s = None         # reply audio the client had played at the barge (from EARS's B)
        self._said = []                    # this turn's spoken chunks: (text, audio offset in s when it started)
        self._audio_s = 0.0                # reply audio generated so far this turn, in s
        self.turn_q: queue.Queue = queue.Queue()
        self.bc_q: queue.Queue = queue.Queue()

    def send(self, kind, payload=b""):
        if self.closed.is_set():
            return
        try:
            with self.send_lock:
                send_msg(self.conn, kind, payload)
        except OSError:                                # EARS went away mid-stream -> stop this session
            self.closed.set(); self.turn_stop.set()

    def set_voice(self, name):
        if name == self.voice_name:
            return
        v = self.m.voice(name)
        if v is None:
            log.warning("voice '%s' not found (keeping %s)", name, self.voice_name); return
        self.voice_name, self.voice = name, v
        log.info("voice -> %s", name)

    @property
    def history(self):
        return [msg for turn in self.turns for msg in turn]

    def _remember(self, user, reply, tool_msgs=()):
        """One turn into history: the user's words, the tool call + compact results if any, the reply."""
        if user.strip():
            self.turns.append([{"role": "user", "content": user.strip()}, *tool_msgs,
                               {"role": "assistant",
                                "content": reply or self.m.prompts["reply"].text("markers", "empty_reply")}])
            self.turns = self.turns[-self.m.history_turns:]

    def _remember_barged(self, user, tool_msgs=()):
        """A barged turn still enters history: the user's words, and the reply up to the chunk that was playing
        when they cut in (all generated chunks if EARS didn't say how much had played), marked interrupted."""
        played = self.barge_played_s
        heard = [t for t, at in self._said if played is None or at < played]
        mark = self.m.prompts["reply"].text("markers", "interrupted")   # the system prompt explains it
        self._remember(user, (" ".join(t.strip() for t in heard) + mark).strip(), tool_msgs)

    def _say(self, text, voice, stop_flag):
        """TTS one text chunk as reply audio; logs it in self._said once its first audio is out."""
        clean = say_clean(text); first = True
        if not clean:
            return
        for w in self.m.tts.stream(clean, voice, stop_flag=stop_flag):
            if first:
                self._said.append((text, self._audio_s)); first = False
            self._audio(w)
            yield ("audio", w)

    def _audio(self, w):
        """Book-keep one reply audio chunk: total generated, and when the client will have played it all."""
        d = len(w) / self.m.sr
        self._audio_s += d
        self._play_end = max(self._play_end, time.perf_counter()) + d

    # ---- fillers: short lines spoken over tool latency, made on a helper thread ----
    def _spawn(self, gen_fn):
        """Run the generator gen_fn() on a daemon thread; returns a queue of its items, then _END."""
        q: queue.Queue = queue.Queue()

        def run():
            try:
                for x in gen_fn():
                    q.put(x)
            except Exception as e:
                log.warning("filler error: %s", type(e).__name__)
            finally:
                q.put(_END)
        threading.Thread(target=run, daemon=True).start()
        return q

    def _filler(self, section, user, voice, stop_flag):
        """One filler line from the LLM (prompts.yaml `section`: system prompt + few-shot turns, then `user`),
        then its TTS: ("text", line), ("audio", chunk)..."""
        p = self.m.prompts["filler"]
        msgs = _with_shots(p.text(section, "system"), p.shots(section), user)
        line = _text(self.m.llms["filler"].stream(msgs, max_new=24, stop_flag=stop_flag)).strip()
        if not line or stop_flag():
            return
        yield ("text", line)
        clean = say_clean(line)
        if clean:
            yield from (("audio", w) for w in self.m.tts.stream(clean, voice, stop_flag=stop_flag))

    def _relay(self, q, stop_flag):
        """Yield a filler's items from its helper thread as this turn's reply; returns the filler line."""
        line = None; first = True
        while not stop_flag():
            try:
                x = q.get(timeout=0.005)
            except queue.Empty:
                continue
            if x is _END:
                break
            kind, val = x
            if kind == "text":
                line = val
            else:
                if first:
                    self._said.append((line, self._audio_s)); first = False
                self._audio(val)
            yield x
        return line

    # ---- speak an LLM text stream: a short first chunk (clause), then whole sentences ----
    def _speak_stream(self, events, voice, stop_flag):
        """Speak the text of an LLM event stream as it arrives. Yields MOUTH events; returns (text, calls, native,
        started, refused). Returns at "tool_start" (started = its payload, {"name": ...}) without consuming the
        rest: the call is still decoding, and the caller finishes `events` itself. Text before the tag that didn't
        close a chunk is never spoken. A refusal (hosted models) returns refused=True with nothing more spoken."""
        content = ""; calls = None; native = None; done = 0
        for ev in events:
            kind, val = ev[0], ev[1]
            if kind == "tool_start":
                return content, None, None, val or {"name": None}, False
            if kind == "refusal":
                log.info("reply refused (category=%s)", val)
                return content, None, None, None, True
            if kind == "text":
                content += val
                yield ("text", content)
                safe = content[done:]
                while True:
                    chunk, rest = next_chunk(safe, first=done == 0)
                    if chunk is None:
                        break
                    done += len(safe) - len(rest); safe = rest
                    if stop_flag():
                        break
                    yield from self._say(chunk, voice, stop_flag)
            elif kind == "tool_calls":
                calls, native = val, ev[2]
            if stop_flag():
                return content, calls, native, None, False
        if not calls and not stop_flag():
            yield from self._say(content[done:], voice, stop_flag)
        return content, calls, native, None, False

    def _speak_line(self, key, transcript, stop_flag):
        """End the turn with one of MOUTH's own lines (prompts.yaml `spoken`): the apology after a failed LLM
        request, or the neutral line after a refusal. The user's turn stays in history."""
        line = self.m.prompts["reply"].text("spoken", key)
        yield ("text", line)
        yield from self._say(line, self.voice, stop_flag)
        if stop_flag():
            self._remember_barged(transcript)
        else:
            self._remember(transcript, line)
        yield ("done", line)

    # ---- tool rounds: run the calls, and cover the wait with short lines ----
    def _run_calls(self, calls, expert=False):
        """Run one round of tool calls at once; results in call order. A call the plugin flagged (bad arguments)
        is never run; with `expert`, ask_expert isn't run either: the smart model answers it."""
        results = [None] * len(calls)
        p = self.m.prompts["reply"]

        def _run(i, c):
            if c.get("error"):
                results[i] = {"error": c["error"]}
            elif expert and c["name"] == EXPERT:
                results[i] = {"result": p.text("escalation", "tool_result")}
            else:
                results[i] = execute_tool(c["name"], c.get("arguments", {}))
        workers = [threading.Thread(target=_run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return results

    def _cover(self, done, section, transcript, fillers, voice, stop_flag):
        """Until done(): whenever the last line has played out FILLER_GAP_S ago, one more short line from
        prompts.yaml `section` (still_looking or expert_still), up to MAX_STILL. Appends the lines to `fillers`."""
        n = 0
        while not done() and not stop_flag():
            if n < MAX_STILL and time.perf_counter() >= self._play_end + FILLER_GAP_S:
                n += 1
                user = still_user(self.m.prompts["filler"], transcript, fillers, section)
                q = self._spawn(lambda user=user: self._filler(section, user, voice, stop_flag))
                line = yield from self._relay(q, stop_flag)
                if line:
                    fillers.append(line)
            else:
                time.sleep(0.01)

    # ---- a full turn (may call tools) ----
    def stream_turn(self, transcript, stop_flag):
        try:
            yield from self._turn(transcript, stop_flag)
        except LLMError as e:                    # hosted LLM failed: say so briefly instead of going silent
            log.warning("turn: LLM request failed (%s)", e.kind)
            yield from self._speak_line("error", transcript, stop_flag)

    def _turn(self, transcript, stop_flag):
        m = self.m; voice = self.voice                 # one voice per reply (a switch applies to the next one)
        self._said = []; self._audio_s = 0.0; self._play_end = time.perf_counter()
        msgs = reply_prefix(m.prompts["reply"], m.expert) + list(self.history) + \
            [{"role": "user", "content": transcript}]
        events = m.llms["reply"].stream(msgs, tools=m.tools, max_new=m.max_gen, stop_flag=stop_flag)
        content, calls, native, started, refused = yield from self._speak_stream(events, voice, stop_flag)
        if refused and not stop_flag():
            yield from self._speak_line("refusal", transcript, stop_flag); return
        if stop_flag():
            self._remember_barged(transcript)
            yield ("done", content.strip()); return
        if not calls and not started:                       # plain reply
            reply = content.strip(); self._remember(transcript, reply)
            yield ("done", reply); return

        # ---- tool path (L-04). The call's name just decoded: a helper thread finishes decoding the call and runs
        # the tools the moment it parses, while this one speaks a lead-in that fits the tool, made alongside (vLLM
        # batches the two requests). If the tools are still out FILLER_GAP_S after the last line has played, one
        # more short "still looking" line, up to MAX_STILL. For ask_expert the smart request starts right after the
        # other tools, still under the lead-in. Everything stops on barge (stop_flag); helper threads are daemons.
        box = {"calls": calls or [], "native": native, "results": [], "error": None, "smart": None}
        tools_done = threading.Event()

        def finish_call_and_run_tools():
            try:
                for ev in events:                           # no-op if _speak_stream already drained it
                    if ev[0] == "tool_calls":
                        box["calls"], box["native"] = ev[1], ev[2]
                if stop_flag() or not box["calls"]:
                    return
                log.info("tool calls: %s", [c["name"] for c in box["calls"]])
                box["results"] = self._run_calls(box["calls"], expert=m.expert)
                ask = next((c for c in box["calls"] if c["name"] == EXPERT and not c.get("error")), None)
                if ask is not None and m.expert and not stop_flag():
                    others = [(c["name"], r) for c, r in zip(box["calls"], box["results"]) if c["name"] != EXPERT]
                    recent = self.turns[-m.context_turns:] if m.context_turns > 0 else []
                    box["smart_msgs"] = smart_messages(m.prompts["smart"], recent, transcript,
                                                       (ask.get("arguments") or {}).get("question"), others)
                    box["smart_t0"] = time.perf_counter()
                    box["smart"] = _Prefetch(lambda: m.llms["smart"].stream(
                        box["smart_msgs"], tools=m.smart_tools, max_new=m.smart_max_new, stop_flag=stop_flag),
                        stop_flag)
            except Exception as e:
                box["error"] = e
                log.warning("tool call error: %s", type(e).__name__)
            finally:
                tools_done.set()
        threading.Thread(target=finish_call_and_run_tools, daemon=True).start()

        tool = started["name"] if started else (calls[0]["name"] if calls else None)
        to_expert = m.expert and tool == EXPERT
        fillers = []
        if not self._said:                     # the model's own words before the tag already did the job
            section = "expert_leadin" if to_expert else "tool_leadin"
            user = m.prompts["filler"].fill(section, "user", transcript=transcript)
            q = self._spawn(lambda: self._filler(section, user, voice, stop_flag))
            line = yield from self._relay(q, stop_flag)
            if line:
                fillers.append(line)
        yield from self._cover(lambda: tools_done.is_set() and (box["smart"] is None or box["smart"].ready.is_set()),
                               "expert_still" if to_expert else "still_looking", transcript, fillers, voice, stop_flag)
        if stop_flag():
            self._remember_barged(transcript)
            yield ("done", " ".join(fillers)); return
        if isinstance(box["error"], LLMError):       # the call never finished decoding (hosted LLM failed)
            raise box["error"]
        calls, results = box["calls"], box["results"]
        results = results if len(results) == len(calls) else [None] * len(calls)
        if not calls:                                       # a tag with no parseable call: nothing to answer from
            lead = " ".join(fillers)
            self._remember(transcript, lead)
            yield ("done", lead); return

        kept = round_messages(content, calls, box["native"], results, compact=True)
        if box["smart"] is not None:                       # escalated: the smart model answers, straight to TTS
            full2, refused = yield from self._expert_answer(box["smart"], box["smart_msgs"], box["smart_t0"],
                                                            transcript, voice, stop_flag, fillers)
        else:
            msgs2 = msgs + round_messages(content, calls, box["native"], results)
            full2, _, _, _, refused = yield from self._speak_stream(
                m.llms["reply"].stream(msgs2, max_new=m.max_gen, stop_flag=stop_flag), voice, stop_flag)
        if refused and not stop_flag():
            yield from self._speak_line("refusal", transcript, stop_flag); return
        lead = " ".join(fillers)
        reply = (lead + " " + full2).strip()
        if stop_flag():
            self._remember_barged(transcript, kept)
        else:
            self._remember(transcript, full2.strip() or lead, kept)
        yield ("done", reply)

    def _expert_answer(self, stream, msgs, t0, transcript, voice, stop_flag, fillers):
        """Speak the smart model's answer as it streams (`stream`, already under way), running its own tool rounds:
        up to max_steps, each covered by still-looking lines and the next request started as soon as the results
        are in. Past max_steps its calls get the smart.limit note instead of results, and a model that still calls
        tools fails the turn (LLMError: MOUTH apologises). Returns (text, refused). Logs timings and counts only."""
        m = self.m; said = []; rounds = 0; t_first = stream.t_first
        while True:
            text, calls, native, started, refused = yield from self._speak_stream(stream, voice, stop_flag)
            if text.strip():
                said.append(text.strip())
            if refused or stop_flag() or not (calls or started):
                log.info("escalated turn: smart first event %s, %d tool rounds%s",
                         f"{1e3 * (t_first - t0):.0f} ms" if t_first else "none", rounds,
                         " (barged)" if stop_flag() else "")
                return " ".join(said), refused
            if rounds > m.max_steps:
                log.warning("escalated turn: smart model kept calling tools past max_steps")
                raise LLMError("max_steps")
            box = {"calls": calls or [], "native": native, "next": None, "msgs": None, "error": None}
            done = threading.Event()

            def finish_round(stream=stream, msgs=msgs, text=text, limit=rounds >= m.max_steps):
                try:
                    for ev in stream:
                        if ev[0] == "tool_calls":
                            box["calls"], box["native"] = ev[1], ev[2]
                    if stop_flag() or not box["calls"]:
                        return
                    log.info("escalated turn: tool calls %s", [c["name"] for c in box["calls"]])
                    results = [{"error": m.prompts["smart"].text("smart", "limit")}] * len(box["calls"]) if limit \
                        else self._run_calls(box["calls"])
                    box["msgs"] = msgs + round_messages(text, box["calls"], box["native"], results)
                    box["next"] = _Prefetch(lambda: m.llms["smart"].stream(
                        box["msgs"], tools=m.smart_tools, max_new=m.smart_max_new, stop_flag=stop_flag), stop_flag)
                except Exception as e:
                    box["error"] = e
                finally:
                    done.set()
            threading.Thread(target=finish_round, daemon=True).start()
            yield from self._cover(lambda: done.is_set() and (box["next"] is None or box["next"].ready.is_set()),
                                   "still_looking", transcript, fillers, voice, stop_flag)
            if stop_flag():
                return " ".join(said), False
            if box["error"] is not None:
                raise box["error"]
            if box["next"] is None:                         # a tool call that didn't parse: nothing more to say
                return " ".join(said), False
            rounds += 1; stream, msgs = box["next"], box["msgs"]

    # ---- say (S): TTS of a given text only; no LLM, not remembered ----
    def stream_say(self, text, stop_flag):
        self._said = []; self._audio_s = 0.0; self._play_end = time.perf_counter()
        yield from self._say(text, self.voice, stop_flag)
        yield ("done", text)

    # ---- backchannel: always a contextual LLM ack off the ASR partial ----
    def stream_bc(self, partial, stop_flag):
        """The LLM picks the ack: a reaction to the content, or a neutral hum when the partial is too short
        (< bc_min_words, flagged to it explicitly) or gives nothing to react to."""
        m = self.m
        messages = bc_messages(m.prompts["filler"], list(self.recent_acks), partial, m.bc_min_words)
        text = _text(m.llms["filler"].stream(messages, max_new=m.bc_max_new, stop_flag=stop_flag)).strip()
        if text:
            self.recent_acks.append(text)
            yield ("text", text)
            for w in m.tts.stream(say_clean(text), self.voice, stop_flag=stop_flag):
                yield ("audio", w)
        yield ("done", text)

    # ---- workers ----
    def _turn_worker(self):
        while (turn := self.turn_q.get()) is not None:
            self.turn_stop.clear(); self.barge_played_s = None
            stop = lambda: self.turn_stop.is_set() or self.closed.is_set()   # noqa: E731
            t0 = time.perf_counter(); t_txt = t_aud = None
            gen = self.stream_say(turn["say"], stop) if "say" in turn else self.stream_turn(turn["transcript"], stop)
            try:
                for kind, val in gen:
                    if kind == "text":
                        t_txt = t_txt or time.perf_counter() - t0
                        self.send(b"X", {"text": val})
                    elif kind == "audio":
                        if t_aud is None:
                            t_aud = time.perf_counter() - t0
                            log.info("onset: first-text %.0fms first-audio %.0fms", (t_txt or 0) * 1e3, t_aud * 1e3)
                        self.send(b"A", val.tobytes())
                    elif kind == "done":
                        self.send(b"D", {"text": val, "barged": self.turn_stop.is_set()})
            except Exception as e:
                if not self.closed.is_set():
                    if isinstance(e, LLMError):           # a hosted request failed past the turn's own handling
                        log.warning("turn: LLM request failed (%s)", e.kind)
                    else:
                        log.exception("turn error: %s", type(e).__name__)
                    self.send(b"D", {"text": "", "error": type(e).__name__})

    def _bc_worker(self):
        while (item := self.bc_q.get()) is not None:
            seq, bc = item                                  # a reply turn (T) after this C makes it moot: EARS
            stop = lambda seq=seq: self.closed.is_set() or self.turn_seq != seq  # noqa: E731  drops its K audio
            t0 = time.perf_counter(); t_aud = None; partial = bc.get("partial", "")
            try:
                for kind, val in self.stream_bc(partial, stop):
                    if kind == "text":
                        self.send(b"X", {"text": val, "bc": True})
                    elif kind == "audio":                   # K = backchannel audio, kept apart from replies
                        t_aud = t_aud or time.perf_counter() - t0
                        self.send(b"K", val.tobytes())
                    elif kind == "done":
                        aborted = stop()
                        self.send(b"D", {"text": val, "bc": True, "aborted": aborted})
                        # counts + timing only, never the words
                        log.info("backchannel: partial_words=%d ack_chars=%d first-audio %.0fms%s",
                                 len(partial.split()), len(val), (t_aud or 0) * 1e3,
                                 " (aborted: reply turn started)" if aborted else "")
            except Exception as e:
                if not self.closed.is_set():
                    if isinstance(e, LLMError):              # hosted LLM failed: no ack this time, no traceback
                        log.warning("backchannel: LLM request failed (%s)", e.kind)
                    else:
                        log.exception("backchannel error: %s", type(e).__name__)
                    self.send(b"D", {"text": "", "bc": True, "error": type(e).__name__})

    def run(self):
        workers = [threading.Thread(target=self._turn_worker, daemon=True),
                   threading.Thread(target=self._bc_worker, daemon=True)]
        for w in workers:
            w.start()
        try:
            while True:
                kind, payload = recv_msg(self.conn)
                if kind is None:
                    break
                if kind == b"T":
                    self.turn_seq += 1                      # stops a backchannel still being made
                    self.turn_q.put(json.loads(payload))
                elif kind == b"S":                          # test tools only: speak a fixed text
                    self.turn_q.put({"say": json.loads(payload).get("text", "")})
                elif kind == b"C":
                    self.bc_q.put((self.turn_seq, json.loads(payload)))
                elif kind == b"V":
                    self.set_voice(json.loads(payload).get("voice", ""))
                elif kind == b"B":                          # EARS may say how much reply audio had played
                    self.barge_played_s = (json.loads(payload) if payload else {}).get("played_s")
                    self.turn_stop.set()
        finally:                                            # disconnect: stop + abort whatever is live
            self.closed.set(); self.turn_stop.set()
            self.turn_q.put(None); self.bc_q.put(None)
            for w in workers:
                w.join(timeout=5)


_ACTIVE = 0
_ACTIVE_LOCK = threading.Lock()


def serve_conn(m, conn):
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE += 1; n = _ACTIVE
    s = Session(m, conn)
    log.info("session start (active=%d, voice=%s)", n, s.voice_name)
    try:
        s.run()
    except Exception as e:
        log.exception("session error: %s", type(e).__name__)
    finally:
        conn.close()
        with _ACTIVE_LOCK:
            _ACTIVE -= 1; n = _ACTIVE
        log.info("session end (active=%d)", n)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import torch
    torch.set_num_threads(int(cfg["mouth"].get("cpu_threads", 4)))   # default 48 threads mostly spin-wait
    if not os.environ.get("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set -> web_search/get_weather will return an error")
    m = Mouth(cfg)

    def stop(signum, frame):                      # shut the engines down so their worker processes exit too
        log.info("MOUTH stopping (signal %d)", signum)    # (a bare SIGTERM orphans the VoxCPM2 stage process)
        for shutdown in [m.tts.shutdown] + [x.shutdown for x in {id(x): x for x in m.llms.values()}.values()]:
            try:
                shutdown()
            except Exception as e:
                log.warning("engine shutdown: %s", type(e).__name__)
        os._exit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(16)
    log.info("=== MOUTH ready (in-proc %s + %s, sr=%d) listening on %s:%d ===",
             cfg["llm"]["reply"]["plugin"], cfg["tts"]["plugin"], m.sr, host, port)
    while True:                                   # one thread per EARS session; engines are shared
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve_conn, args=(m, conn), daemon=True).start()


if __name__ == "__main__":
    main()
