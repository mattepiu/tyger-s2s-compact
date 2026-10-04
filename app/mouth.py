"""MOUTH process: the responder LLM and the TTS, both in-process on vLLM engines, plus the turn orchestration.

Architecture:
  * The models are plugins (app/plugins, PLUGINS.md), chosen in config. LLM roles: `reply` (turns and the tool
    phase-B answer) and `filler` (backchannels, tool lead-ins, still-looking lines), the same instance unless
    configured apart. Local plugins run in-process on a shared asyncio loop (app/engines.py): the Qwen LLMs on
    vLLM AsyncLLM, VoxCPM2 on vLLM-Omni AsyncOmni (app/tts_voxcpm.py). No HTTP between MOUTH and the models.
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
from app.tools import build_tools, execute_tool
from app.chunker import next_chunk, say_clean
from app.config import load_config, discover_voices, repo_path, llm_roles
from app.prompts import load_prompts
from app.engines import EngineLoop
from app.plugins import build_llms
from app.tts_voxcpm import VoxCPM2TTS

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


def still_user(p, transcript, fillers):
    """The still-looking request's user line: what was asked and the lines already spoken."""
    said = " ".join(f"'{f}'" for f in fillers) or p.text("still_looking", "said_none")
    return p.fill("still_looking", "user", transcript=transcript, said=said)


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
        if "smart" in roles:
            raise NotImplementedError("the smart role (escalation, background tasks) comes in PLUGINS.md step 10")
        m = cfg["mouth"]; t = cfg["tts"]
        self.cfg = cfg
        # prompts first: a bad prompts.yaml stops start-up before minutes of engine loading
        by_name = load_prompts(cfg, {name for name, _ in roles.values()})
        self.prompts = {role: by_name[name] for role, (name, _) in roles.items()}   # role -> Prompts
        self.tools = build_tools(self.prompts["reply"])
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.bc_min_words = int(m.get("bc_min_words", 4))
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.eloop = EngineLoop()
        # TTS first: its deploy config pins a small KV budget; the LLM's start-up check then sees what's left.
        log.info("loading TTS %s", t["model"])
        self.tts = VoxCPM2TTS(self.eloop, t["model"], deploy_config=repo_path(t.get("deploy_config")),
                              out_sr=t.get("out_sr", 24000), ref_max_s=t.get("ref_max_s", 10.0))
        self.llms = build_llms(cfg, self.eloop)          # role -> LLM plugin instance
        self.sr = self.tts.out_sr
        self._voices = {}                     # clip path -> (mtime, Voice): each clip is read once
        self._voice_lock = threading.Lock()
        names, default = discover_voices(cfg)
        if names:
            log.info("voices available: %s (default=%s)", sorted(names), default)
        elif t.get("allow_zero_shot"):          # testing only: VoxCPM2's unconditioned voice, no clone
            log.warning("no voice clips in voices.dir -> zero-shot TTS voice (tts.allow_zero_shot)")
        else:
            raise RuntimeError("no voice reference clips found (put a *.wav in voices.dir)")
        log.info("warming TTS (voice=%s)", default)
        self.tts.warmup(self.default_voice()[1])
        for name in sorted(names):                     # every clip into VoxCPM2's speaker cache before serving
            self.tts.prime_voice(self.voice(name))
        for llm in {id(x): x for x in self.llms.values()}.values():
            llm.warmup()
        # first request with the tools prompt is cold (~490 ms to first token vs ~55 ms after) -> pay it here
        p = self.prompts["reply"]
        for _ in self.llms["reply"].stream([{"role": "system", "content": p.text("system", "voice_tools")},
                                  {"role": "user", "content": p.text("warmup", "user")}], tools=self.tools, max_new=8):
            pass

    def voice(self, name):
        """The Voice for a clip name (re-scans voices.dir, so clips added after startup work), or None."""
        try:
            voices, _ = discover_voices(self.cfg)
        except Exception as e:
            log.warning("voice discovery failed: %s", e); return None
        path = voices.get(name)
        if path is None:
            return None
        mtime = os.path.getmtime(path)
        with self._voice_lock:
            hit = self._voices.get(path)
            if hit is None or hit[0] != mtime:
                hit = (mtime, self.tts.load_voice(path, name))
                self._voices[path] = hit
            return hit[1]

    def default_voice(self):
        _, default = discover_voices(self.cfg)
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
        started). Returns at "tool_start" (started=True) without consuming the rest: the call is still decoding,
        and the caller finishes `events` itself. Text before the tag that didn't close a chunk is never spoken."""
        content = ""; calls = None; native = None; done = 0
        for ev in events:
            kind, val = ev[0], ev[1]
            if kind == "tool_start":
                return content, None, None, True
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
                return content, calls, native, False
        if not calls and not stop_flag():
            yield from self._say(content[done:], voice, stop_flag)
        return content, calls, native, False

    # ---- a full turn (may call tools) ----
    def stream_turn(self, transcript, stop_flag):
        m = self.m; voice = self.voice                 # one voice per reply (a switch applies to the next one)
        self._said = []; self._audio_s = 0.0; self._play_end = time.perf_counter()
        msgs = [{"role": "system", "content": m.prompts["reply"].text("system", "voice_tools")}] + list(self.history) + \
               [{"role": "user", "content": transcript}]
        events = m.llms["reply"].stream(msgs, tools=m.tools, max_new=m.max_gen, stop_flag=stop_flag)
        content, calls, native, started = yield from self._speak_stream(events, voice, stop_flag)
        if stop_flag():
            self._remember_barged(transcript)
            yield ("done", content.strip()); return
        if not calls and not started:                       # plain reply
            reply = content.strip(); self._remember(transcript, reply)
            yield ("done", reply); return

        # ---- tool path (L-04). The <tool_call> tag just appeared: a helper thread finishes decoding the call
        # and runs the tools the moment it parses, while this one speaks a filler made alongside (vLLM batches the
        # two requests). If the tools are still out FILLER_GAP_S after the filler has played, one more short
        # "still looking" line, up to MAX_STILL. Everything stops on barge (stop_flag); tool threads are daemons.
        box = {"calls": calls or [], "native": native, "results": []}; tools_done = threading.Event()

        def finish_call_and_run_tools():
            try:
                for ev in events:                           # no-op if _speak_stream already drained it
                    if ev[0] == "tool_calls":
                        box["calls"], box["native"] = ev[1], ev[2]
                if stop_flag() or not box["calls"]:
                    return
                log.info("tool calls: %s", [c["name"] for c in box["calls"]])
                results = [None] * len(box["calls"])

                def _run(i, c):
                    results[i] = execute_tool(c["name"], c.get("arguments", {}))
                workers = [threading.Thread(target=_run, args=(i, c), daemon=True)
                           for i, c in enumerate(box["calls"])]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                box["results"] = results
            except Exception as e:
                log.warning("tool call error: %s", type(e).__name__)
            finally:
                tools_done.set()
        threading.Thread(target=finish_call_and_run_tools, daemon=True).start()

        fillers = []; n_still = 0
        if not self._said:                     # the model's own words before the tag already did the job
            user = m.prompts["filler"].fill("tool_leadin", "user", transcript=transcript)
            q = self._spawn(lambda: self._filler("tool_leadin", user, voice, stop_flag))
            line = yield from self._relay(q, stop_flag)
            if line:
                fillers.append(line)
        while not tools_done.is_set() and not stop_flag():
            if n_still < MAX_STILL and time.perf_counter() >= self._play_end + FILLER_GAP_S:
                n_still += 1
                user = still_user(m.prompts["filler"], transcript, fillers)
                q = self._spawn(lambda user=user: self._filler("still_looking", user, voice, stop_flag))
                line = yield from self._relay(q, stop_flag)
                if line:
                    fillers.append(line)
            else:
                tools_done.wait(0.01)
        lead = " ".join(f for f in fillers if f)
        if stop_flag():
            self._remember_barged(transcript)
            yield ("done", lead); return
        calls, results = box["calls"], box["results"]
        results = results if len(results) == len(calls) else [None] * len(calls)
        if not calls:                                       # a tag with no parseable call: nothing to answer from
            self._remember(transcript, lead)
            yield ("done", lead); return

        # neutral format: plugin-issued IDs, unique for the session; `native` is the plugin's own, passed through
        asst = {"role": "assistant", "content": content.strip() or None,
                "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c.get("arguments", {})}
                               for c in calls]}
        if box["native"] is not None:
            asst["native"] = box["native"]
        msgs2 = msgs + [asst] + [{"role": "tool", "tool_call_id": c["id"],
                                  "content": json.dumps(r if r is not None else {"error": "no result"})}
                                 for c, r in zip(calls, results)]
        kept = [asst] + [{"role": "tool", "tool_call_id": c["id"], "content": _compact_result(r)}
                         for c, r in zip(calls, results)]
        full2, _, _, _ = yield from self._speak_stream(
            m.llms["reply"].stream(msgs2, max_new=m.max_gen, stop_flag=stop_flag), voice, stop_flag)
        reply = (lead + " " + full2).strip()
        if stop_flag():
            self._remember_barged(transcript, kept)
        else:
            self._remember(transcript, full2.strip() or lead, kept)
        yield ("done", reply)

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
        for shutdown in [m.tts.engine.shutdown] + [x.shutdown for x in {id(x): x for x in m.llms.values()}.values()]:
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
