"""MOUTH process: the responder LLM and the TTS, both in-process on vLLM engines, plus the turn orchestration.

Architecture:
  * One process hosts two in-process engines on a shared asyncio loop (app/engines.py): Qwen3-4B-Instruct-2507
    on a vLLM AsyncLLM (app/llm.py) and VoxCPM2 on a vLLM-Omni AsyncOmni (app/tts_voxcpm.py). No HTTP between
    MOUTH and the models. They can't be one vLLM-Omni pipeline: stages chain stage 0 -> stage 1, and MOUTH
    has to sit between them (chunking, tools + lead-in, backchannels, barge-in).
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
two can now be generated at the same time). Audio is 24 kHz float32. No audio or transcript is written to disk
or logs; only timings and event types.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.mouth
"""
from __future__ import annotations
import os, sys, socket, threading, queue, json, time, logging
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from app.wire import send_msg, recv_msg
from app.tools import TOOLS, execute_tool
from app.chunker import next_chunk, say_clean, TTS_SYS
from app.config import load_config, discover_voices, repo_path
from app.engines import EngineLoop
from app.llm import LocalLLM
from app.tts_voxcpm import VoxCPM2TTS

log = logging.getLogger("mouth")

TTS_TOOLS_SYS = TTS_SYS + (
    " You can call tools when they help. For anything time-sensitive or that you may not know -- current "
    "weather, prices, news, sports scores, live facts, or the current time -- call the appropriate tool "
    "directly instead of guessing, asking permission, or telling the user to check a website. After a tool "
    "returns, give a short spoken answer in one or two sentences using the result; never read out raw data, "
    "URLs, code, or JSON. If a tool reports it is not connected yet, tell the user that plainly instead of "
    "pretending it was done.")

LEADIN_SYS = (
    "You are a voice assistant and you are about to look something up for the user right now. Say ONE short, "
    "natural spoken sentence that tells them you're checking or looking it up (for example 'Sure, let me "
    "check that for you.' or 'One sec, let me look that up.'). Do NOT answer the question, do NOT mention "
    "tools or say the word tool, do NOT ask anything. Plain spoken words only, at most about ten words.")

BC_PROMPT = (
    "You are a voice assistant listening to the user, who is STILL speaking. You have NOT been given the "
    "floor. Read their partial words so far and emit ONLY a brief, natural spoken reaction (one to four words) "
    "to WHAT they are saying -- match its content and emotion, for example 'oh nice', 'oh no', 'wow, really?', "
    "'that makes sense', 'got it', 'right', 'sure', 'ha, yeah', 'that's great', 'oh, I see'. If they have said "
    "too little so far to react to, or nothing in their words calls for a specific reaction, use a neutral hum "
    "such as 'mm-hm' or 'mhm'. Do NOT answer, do NOT ask a real question, do NOT continue their sentence, do "
    "NOT take the floor. Always react in the same language the user is speaking. Plain spoken words only, "
    "no markdown or symbols. Just the short reaction.")
RECENT_ACKS = 3
# Few-shot turns for the backchannel prompt. A 4B model copies the English examples in BC_PROMPT over an
# abstract "use the user's language" rule; showing reactions in several languages fixes that.
BC_SHOTS = [
    ("The user is currently saying: \"I finally finished my first marathon last weekend\"", "wow, nice!"),
    ("The user is currently saying: \"у меня сегодня машина сломалась прямо по дороге на работу\"", "ой, нет"),
    ("The user has only said a few words so far: \"so I\"", "mm-hm"),
    ("The user has only said a few words so far: \"ну вот\"", "угу"),
    ("The user is currently saying: \"ayer por fin aprobé el examen de conducir\"", "¡qué bien!"),
]


def _text(events):
    return "".join(v for k, v in events if k == "text")


class Mouth:
    """Shared by all sessions: the two engines and voice loading. Per-conversation state lives in Session."""
    def __init__(self, cfg):
        m = cfg["mouth"]; lc = cfg["llm"]; t = cfg["tts"]
        self.cfg = cfg
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.bc_min_words = int(m.get("bc_min_words", 4))
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.temperature = float(m.get("temperature", 0.7))
        self.eloop = EngineLoop()
        # TTS first: its deploy config pins a small KV budget; the LLM's start-up check then sees what's left.
        log.info("loading TTS %s", t["model"])
        self.tts = VoxCPM2TTS(self.eloop, t["model"], deploy_config=repo_path(t.get("deploy_config")),
                              out_sr=t.get("out_sr", 24000), ref_max_s=t.get("ref_max_s", 10.0))
        log.info("loading responder %s", lc["model"])
        self.llm = LocalLLM(self.eloop, lc["model"], gpu_memory_utilization=lc.get("gpu_memory_utilization", 0.45),
                            kv_cache_gib=lc.get("kv_cache_gib", 3.0), max_model_len=lc.get("max_model_len", 8192),
                            max_num_seqs=lc.get("max_num_seqs", 8), temperature=self.temperature)
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
        # first request with the tools prompt is cold (~490 ms to first token vs ~55 ms after) -> pay it here
        for _ in self.llm.stream([{"role": "system", "content": TTS_TOOLS_SYS}, {"role": "user", "content": "Hi."}],
                                 tools=TOOLS, max_new=8):
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
        self.history = []
        self.recent_acks = deque(maxlen=RECENT_ACKS)   # in memory only; so backchannels don't repeat
        self.voice_name, self.voice = m.default_voice()  # every session starts on the configured default
        self.turn_stop = threading.Event()
        self.closed = threading.Event()
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

    def _remember(self, user, reply):
        if user.strip():
            self.history.append({"role": "user", "content": user.strip()})
            self.history.append({"role": "assistant", "content": reply or "..."})
            self.history = self.history[-2 * self.m.history_turns:]

    # ---- speak an LLM text stream: a short first chunk (clause), then whole sentences ----
    def _speak_stream(self, events, voice, stop_flag):
        """Speak the text of an LLM event stream as it arrives. Yields MOUTH events; returns (text, calls)."""
        tts = self.m.tts
        content = ""; calls = None; done = 0
        for kind, val in events:
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
                    clean = say_clean(chunk)
                    if clean:
                        for w in tts.stream(clean, voice, stop_flag=stop_flag):
                            yield ("audio", w)
            elif kind == "tool_calls":
                calls = val
            if stop_flag():
                return content, calls
        tail = say_clean(content[done:])
        if not calls and tail and not stop_flag():
            for w in tts.stream(tail, voice, stop_flag=stop_flag):
                yield ("audio", w)
        return content, calls

    # ---- a full turn (may call tools) ----
    def stream_turn(self, transcript, stop_flag):
        m = self.m; voice = self.voice                 # one voice per reply (a switch applies to the next one)
        msgs = [{"role": "system", "content": TTS_TOOLS_SYS}] + list(self.history) + \
               [{"role": "user", "content": transcript}]
        content, calls = yield from self._speak_stream(
            m.llm.stream(msgs, tools=TOOLS, max_new=m.max_gen, stop_flag=stop_flag), voice, stop_flag)
        if stop_flag():
            yield ("done", content.strip()); return
        if not calls:                                       # plain reply
            reply = content.strip(); self._remember(transcript, reply)
            yield ("done", reply); return

        # ---- tool path: run tool(s) in BG threads, speak a lead-in over them, then phase B ----
        log.info("tool calls: %s", [c["name"] for c in calls])
        results = [None] * len(calls)

        def _run(i, c):
            results[i] = execute_tool(c["name"], c.get("arguments", {}))
        workers = [threading.Thread(target=_run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
        for w in workers:
            w.start()

        lead = _text(m.llm.stream([{"role": "system", "content": LEADIN_SYS},
                                   {"role": "user", "content": transcript}],
                                  max_new=24, stop_flag=stop_flag)).strip()
        if lead and not stop_flag():
            yield ("text", lead)
            for w in m.tts.stream(say_clean(lead), voice, stop_flag=stop_flag):
                yield ("audio", w)
        for w in workers:
            w.join()
        if stop_flag():
            yield ("done", lead); return

        asst = {"role": "assistant", "content": content.strip() or None,
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                               for i, c in enumerate(calls)]}
        msgs2 = msgs + [asst] + [{"role": "tool", "tool_call_id": f"call_{i}",
                                  "content": json.dumps(r if r is not None else {"error": "no result"})}
                                 for i, r in enumerate(results)]
        full2, _ = yield from self._speak_stream(
            m.llm.stream(msgs2, max_new=m.max_gen, stop_flag=stop_flag), voice, stop_flag)
        reply = (lead + " " + full2).strip(); self._remember(transcript, full2.strip() or lead)
        yield ("done", reply)

    # ---- backchannel: always a contextual LLM ack off the ASR partial ----
    def stream_bc(self, partial, stop_flag):
        """The LLM picks the ack: a reaction to the content, or a neutral hum when the partial is too short
        (< bc_min_words, flagged to it explicitly) or gives nothing to react to."""
        m = self.m
        user = (partial or "").strip()
        sys_prompt = BC_PROMPT
        if self.recent_acks:
            sys_prompt += " You just said " + ", ".join(f"'{a}'" for a in self.recent_acks) + \
                          " -- don't repeat those."
        if not user:
            said = "The user is speaking, but no words have been recognized yet."
        elif len(user.split()) < m.bc_min_words:
            said = f"The user has only said a few words so far: \"{user}\""
        else:
            said = f"The user is currently saying: \"{user}\""
        messages = [{"role": "system", "content": sys_prompt}]
        for q, a in BC_SHOTS:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": said})
        text = _text(m.llm.stream(messages, max_new=m.bc_max_new, stop_flag=stop_flag)).strip()
        if text:
            self.recent_acks.append(text)
            yield ("text", text)
            for w in m.tts.stream(say_clean(text), self.voice, stop_flag=stop_flag):
                yield ("audio", w)
        yield ("done", text)

    # ---- workers ----
    def _turn_worker(self):
        while (turn := self.turn_q.get()) is not None:
            self.turn_stop.clear()
            stop = lambda: self.turn_stop.is_set() or self.closed.is_set()   # noqa: E731
            t0 = time.perf_counter(); t_txt = t_aud = None
            try:
                for kind, val in self.stream_turn(turn["transcript"], stop):
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
        while (bc := self.bc_q.get()) is not None:
            stop = self.closed.is_set
            t0 = time.perf_counter(); t_aud = None; partial = bc.get("partial", "")
            try:
                for kind, val in self.stream_bc(partial, stop):
                    if kind == "text":
                        self.send(b"X", {"text": val, "bc": True})
                    elif kind == "audio":                   # K = backchannel audio, kept apart from replies
                        t_aud = t_aud or time.perf_counter() - t0
                        self.send(b"K", val.tobytes())
                    elif kind == "done":
                        self.send(b"D", {"text": val, "bc": True})
                        # counts + timing only, never the words
                        log.info("backchannel: partial_words=%d ack_chars=%d first-audio %.0fms",
                                 len(partial.split()), len(val), (t_aud or 0) * 1e3)
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
                    self.turn_q.put(json.loads(payload))
                elif kind == b"C":
                    self.bc_q.put(json.loads(payload))
                elif kind == b"V":
                    self.set_voice(json.loads(payload).get("voice", ""))
                elif kind == b"B":
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
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(16)
    log.info("=== MOUTH ready (in-proc %s + %s, sr=%d) listening on %s:%d ===",
             cfg["llm"]["model"], cfg["tts"]["model"], m.sr, host, port)
    while True:                                   # one thread per EARS session; engines are shared
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve_conn, args=(m, conn), daemon=True).start()


if __name__ == "__main__":
    main()
