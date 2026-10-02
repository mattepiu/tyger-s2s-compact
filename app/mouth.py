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
  * Barge-in (B on the wire) sets the stop flag; the live LLM/TTS request is aborted in the engine.
  * reminder/message tools return a "not connected" demo marker (tools.py) and the model says so.
  * Backchannels are always contextual LLM acks off the ASR partial: a reaction to the content, or a neutral
    hum when the partial is too short (< mouth.bc_min_words) to react to. No canned/pre-rendered acks.
Same EARS<->MOUTH wire (T/C/B/V -> X/A/D); audio on the wire is 24 kHz float32. No audio or transcript is
written to disk or logs; only timings and event types.

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
    def __init__(self, cfg):
        m = cfg["mouth"]; lc = cfg["llm"]; t = cfg["tts"]
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.bc_min_words = int(m.get("bc_min_words", 4))
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.eloop = EngineLoop()
        # TTS first: its deploy config pins a small KV budget; the LLM's start-up check then sees what's left.
        log.info("loading TTS %s", t["model"])
        self.tts = VoxCPM2TTS(self.eloop, t["model"], deploy_config=repo_path(t.get("deploy_config")),
                              out_sr=t.get("out_sr", 24000), ref_max_s=t.get("ref_max_s", 10.0))
        log.info("loading responder %s", lc["model"])
        self.llm = LocalLLM(self.eloop, lc["model"], gpu_memory_utilization=lc.get("gpu_memory_utilization", 0.45),
                            kv_cache_gib=lc.get("kv_cache_gib", 2.0), max_model_len=lc.get("max_model_len", 8192),
                            max_num_seqs=lc.get("max_num_seqs", 4), temperature=m.get("temperature", 0.7))
        self.cfg = cfg
        self.voices, self.voice = discover_voices(cfg)
        if self.voices:
            log.info("voices available: %s (default=%s)", sorted(self.voices), self.voice)
            self.tts.set_voice(self.voices[self.voice])
        elif t.get("allow_zero_shot"):          # testing only: VoxCPM2's unconditioned voice, no clone
            log.warning("no voice clips in voices.dir -> zero-shot TTS voice (tts.allow_zero_shot)")
        else:
            raise RuntimeError("no voice reference clips found (put a *.wav in voices.dir)")
        self.sr = self.tts.out_sr
        self.history = []
        self.recent_acks = deque(maxlen=RECENT_ACKS)   # in memory only; so backchannels don't repeat
        log.info("warming TTS (voice=%s)", self.voice)
        self.tts.warmup()
        # first request with the tools prompt is cold (~490 ms to first token vs ~55 ms after) -> pay it here
        for _ in self.llm.stream([{"role": "system", "content": TTS_TOOLS_SYS}, {"role": "user", "content": "Hi."}],
                                 tools=TOOLS, max_new=8):
            pass

    # ---- voice ----
    def set_voice(self, name):
        if name not in self.voices:                 # re-scan the dir in case a clip was added after startup
            try:
                self.voices, _ = discover_voices(self.cfg)
            except Exception as e:
                log.warning("voice re-discovery failed: %s", e)
        if name not in self.voices:
            log.warning("voice '%s' not found; available=%s (keeping %s)", name, sorted(self.voices), self.voice)
            return False
        if name == self.voice:
            return False
        self.voice = name
        self.tts.set_voice(self.voices[name])
        log.info("voice -> %s", name)
        return True

    def reset_voice(self):
        """New session -> back to the configured default voice (the client then applies its own pick), so a
        voice chosen in a previous session never leaks into this one."""
        try:
            self.voices, default = discover_voices(self.cfg)
        except Exception as e:
            log.warning("voice re-discovery failed: %s", e); return
        if default and default != self.voice:
            self.set_voice(default)

    # ---- history ----
    def reset_history(self):
        self.history = []
        self.recent_acks.clear()

    def _remember(self, user, reply):
        if user.strip():
            self.history.append({"role": "user", "content": user.strip()})
            self.history.append({"role": "assistant", "content": reply or "..."})
            self.history = self.history[-2 * self.history_turns:]

    # ---- TTS of the untts'd tail: a short first chunk (clause), then whole sentences ----
    def _tts_tail(self, full, stop_flag):
        safe = full[self._tts_done:]
        while True:
            chunk, rest = next_chunk(safe, first=self._tts_done == 0)
            if chunk is None:
                break
            self._tts_done += len(safe) - len(rest)
            safe = rest
            if stop_flag and stop_flag():
                break
            clean = say_clean(chunk)
            if clean:
                for w in self.tts.stream(clean, stop_flag=stop_flag):
                    yield ("audio", w)

    def _flush_tail(self, full, stop_flag):
        tail = say_clean(full[self._tts_done:])
        if tail and not (stop_flag and stop_flag()):
            for w in self.tts.stream(tail, stop_flag=stop_flag):
                yield ("audio", w)

    def _speak_stream(self, events, stop_flag):
        """Speak the text of an LLM event stream as it arrives. Yields MOUTH events; returns (text, calls)."""
        content = ""; calls = None; self._tts_done = 0
        for kind, val in events:
            if kind == "text":
                content += val
                yield ("text", content)
                yield from self._tts_tail(content, stop_flag)
            elif kind == "tool_calls":
                calls = val
            if stop_flag and stop_flag():
                return content, calls
        if not calls:
            yield from self._flush_tail(content, stop_flag)
        return content, calls

    # ---- a full turn (may call tools) ----
    def stream_turn(self, transcript, stop_flag=None):
        msgs = [{"role": "system", "content": TTS_TOOLS_SYS}] + list(self.history) + \
               [{"role": "user", "content": transcript}]
        content, calls = yield from self._speak_stream(
            self.llm.stream(msgs, tools=TOOLS, max_new=self.max_gen, stop_flag=stop_flag), stop_flag)
        if stop_flag and stop_flag():
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

        lead = _text(self.llm.stream([{"role": "system", "content": LEADIN_SYS},
                                      {"role": "user", "content": transcript}],
                                     max_new=24, stop_flag=stop_flag)).strip()
        if lead and not (stop_flag and stop_flag()):
            yield ("text", lead)
            for w in self.tts.stream(say_clean(lead), stop_flag=stop_flag):
                yield ("audio", w)
        for w in workers:
            w.join()
        if stop_flag and stop_flag():
            yield ("done", lead); return

        asst = {"role": "assistant", "content": content.strip() or None,
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                               for i, c in enumerate(calls)]}
        msgs2 = msgs + [asst] + [{"role": "tool", "tool_call_id": f"call_{i}",
                                  "content": json.dumps(r if r is not None else {"error": "no result"})}
                                 for i, r in enumerate(results)]
        full2, _ = yield from self._speak_stream(
            self.llm.stream(msgs2, max_new=self.max_gen, stop_flag=stop_flag), stop_flag)
        reply = (lead + " " + full2).strip(); self._remember(transcript, full2.strip() or lead)
        yield ("done", reply)

    # ---- backchannel: always a contextual LLM ack off the ASR partial ----
    def stream_bc(self, partial, stop_flag=None):
        """The LLM picks the ack: a reaction to the content, or a neutral hum when the partial is too short
        (< bc_min_words, flagged to it explicitly) or gives nothing to react to."""
        user = (partial or "").strip()
        sys_prompt = BC_PROMPT
        if self.recent_acks:
            sys_prompt += " You just said " + ", ".join(f"'{a}'" for a in self.recent_acks) + \
                          " -- don't repeat those."
        if not user:
            said = "The user is speaking, but no words have been recognized yet."
        elif len(user.split()) < self.bc_min_words:
            said = f"The user has only said a few words so far: \"{user}\""
        else:
            said = f"The user is currently saying: \"{user}\""
        messages = [{"role": "system", "content": sys_prompt}]
        for q, a in BC_SHOTS:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": said})
        text = _text(self.llm.stream(messages, max_new=self.bc_max_new, stop_flag=stop_flag)).strip()
        if text:
            self.recent_acks.append(text)
            yield ("text", text)
            for w in self.tts.stream(say_clean(text), stop_flag=stop_flag):
                yield ("audio", w)
        yield ("done", text)


def serve(m, conn):
    turns = queue.Queue(); stop = threading.Event()
    m.reset_history(); m.reset_voice(); log.info("session start: history reset, voice=%s", m.voice)

    def reader():
        while True:
            kind, payload = recv_msg(conn)
            if kind is None:
                turns.put(None); break
            if kind == b"T":
                turns.put(("turn", json.loads(payload)))
            elif kind == b"C":
                turns.put(("bc", json.loads(payload)))
            elif kind == b"V":
                turns.put(("voice", json.loads(payload)))
            elif kind == b"B":
                stop.set()
    threading.Thread(target=reader, daemon=True).start()

    while True:
        item = turns.get()
        if item is None:
            break
        tag, turn = item
        if tag == "voice":
            m.set_voice(turn.get("voice", "")); continue
        stop.clear()
        try:
            if tag == "bc":
                t0 = time.perf_counter(); t_aud = None; partial = turn.get("partial", "")
                for kind, val in m.stream_bc(partial, stop_flag=stop.is_set):
                    if kind == "text":
                        send_msg(conn, b"X", {"text": val, "bc": True})
                    elif kind == "audio":
                        t_aud = t_aud or time.perf_counter() - t0
                        send_msg(conn, b"A", val.tobytes())
                    elif kind == "done":
                        send_msg(conn, b"D", {"text": val, "bc": True})
                        # counts + timing only, never the words
                        log.info("backchannel: partial_words=%d ack_chars=%d first-audio %.0fms",
                                 len(partial.split()), len(val), (t_aud or 0) * 1e3)
            else:
                t0 = time.perf_counter(); t_txt = t_aud = None
                for kind, val in m.stream_turn(turn["transcript"], stop_flag=stop.is_set):
                    if kind == "text":
                        if t_txt is None:
                            t_txt = time.perf_counter() - t0
                        send_msg(conn, b"X", {"text": val})
                    elif kind == "audio":
                        if t_aud is None:
                            t_aud = time.perf_counter() - t0
                            log.info("onset: first-text %.0fms first-audio %.0fms", (t_txt or 0) * 1e3, t_aud * 1e3)
                        send_msg(conn, b"A", val.tobytes())
                    elif kind == "done":
                        send_msg(conn, b"D", {"text": val, "barged": stop.is_set()})
        except Exception as e:
            log.exception("turn error: %s", type(e).__name__)
            send_msg(conn, b"D", {"text": "", "error": type(e).__name__})


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set -> web_search/get_weather will return an error")
    m = Mouth(cfg)
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(1)
    log.info("=== MOUTH ready (in-proc %s + %s, sr=%d, voice=%s) listening on %s:%d ===",
             cfg["llm"]["model"], cfg["tts"]["model"], m.sr, m.voice, host, port)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("EARS connected")
        try:
            serve(m, conn)
        except Exception as e:
            log.exception("conn error: %s", type(e).__name__)
        finally:
            conn.close(); log.info("EARS disconnected")


if __name__ == "__main__":
    main()
