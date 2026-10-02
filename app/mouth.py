"""MOUTH process (tf 4.57 venv): responder on an external vLLM server + MOSS-TTS-Nano voice-clone TTS.

Architecture (milestone V1.1, "LLM on vLLM"):
  * The responder (Qwen3-4B-Instruct-2507) runs on a separate **vLLM** OpenAI server; this process talks to
    it over HTTP (/v1/chat/completions, streaming, Hermes tool calls). vLLM gives ~28 ms TTFT and ~150 tok/s,
    far faster than the old in-process greedy decode -- that was the latency win.
  * TTS stays **in-process** MOSS-TTS-Nano (app/tts_nano.py): measured 112 ms first-audio here vs ~290 ms when
    MOSS was served through vLLM-Omni, so TTS is NOT on vLLM.
  * The LLM reply is chunked into **whole sentences** (chunker.next_sentence) and each sentence is sent to
    MOSS as soon as it closes, so playback starts on sentence 1 while the rest render.
  * reminder/message tools return a "not connected" demo marker (tools.py) and the model says so.
  * Backchannels are contextual LLM acks off the ASR partial by default (mouth.contextual_acks).
Same EARS<->MOUTH wire (T/C/B/V -> X/A/D). No audio or transcript is written to disk; only event logs.

Run (MOUTH venv): HF_HOME=/workspace/.hf_home <venv_moss>/bin/python -m app.mouth
(The vLLM responder server must already be up -- run.sh starts it first.)
"""
from __future__ import annotations
import os, sys, socket, threading, queue, json, time, random, logging
import numpy as np, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v1/ on path
from app.wire import send_msg, recv_msg
from app.tools import TOOLS, execute_tool
from app.chunker import next_sentence, say_clean, TTS_SYS
from app.tts_realtime import MossRealtimeTTS
from app.config import load_config, discover_voices
from app import weights

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
    "floor. Read their partial words so far and emit ONLY a brief, natural spoken backchannel that fits what "
    "they are saying -- a few words at most, for example 'mm-hm', 'right', 'oh nice', 'oh no', 'wow, really?', "
    "'that makes sense', 'got it'. Do NOT answer, do NOT ask a question, do NOT continue their sentence, do "
    "NOT take the floor. Plain spoken words only, no markdown or symbols. Just the short reaction.")


class LLMClient:
    """Streaming chat client to the external vLLM OpenAI server (Hermes tool calls)."""
    def __init__(self, base_url, model, temperature=0.7, timeout=60):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.base = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout = timeout
        self.sess = requests.Session()

    def wait_ready(self, tries=600, delay=1.0):
        for _ in range(tries):
            try:
                if self.sess.get(self.base + "/v1/models", timeout=2).status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(delay)
        return False

    def stream_deltas(self, messages, tools=None, max_new=192, temperature=None, stop_flag=None):
        """Yield raw OpenAI streaming chunk dicts (one per SSE 'data:' line). Stops early on stop_flag."""
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": int(max_new),
                "temperature": self.temperature if temperature is None else float(temperature)}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        with self.sess.post(self.url, json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop_flag and stop_flag():
                    return
                if not line:
                    continue
                s = line.decode("utf-8", "ignore")
                if not s.startswith("data: "):
                    continue
                s = s[6:]
                if s == "[DONE]":
                    return
                try:
                    yield json.loads(s)
                except Exception:
                    continue


class Mouth:
    def __init__(self, cfg):
        m = cfg["mouth"]; mdl = cfg["models"]; t = cfg["tts"]; v = cfg.get("vllm", {})
        self.max_gen = int(m.get("max_gen", 192))
        self.history_turns = int(m.get("history_turns", 8))
        self.contextual_acks = bool(m.get("contextual_acks", False))
        self.canned_texts = list(m.get("canned_acks", ["mm-hm", "right", "yeah", "okay"]))
        base_url = v.get("base_url", "http://127.0.0.1:8001")
        log.info("responder via vLLM at %s (model=%s)", base_url, mdl["responder"])
        self.llm = LLMClient(base_url, mdl["responder"],
                             temperature=float(m.get("temperature", 0.7)),
                             timeout=int(v.get("timeout", 60)))
        log.info("waiting for vLLM responder to be ready ...")
        if not self.llm.wait_ready():
            raise RuntimeError(f"vLLM responder not reachable at {base_url}")
        log.info("loading MOSS-TTS-Realtime from %s", mdl["moss_dir"])
        self.tts = MossRealtimeTTS(mdl["moss_dir"], mdl["moss_codec"], out_sr=t.get("out_sr", 24000),
                                   dtype=t.get("dtype", "bfloat16"), attn=t.get("attn", "sdpa"),
                                   chunk_dur=t.get("chunk_dur", 0.24), dec_chunk_frames=t.get("dec_chunk_frames", 4),
                                   prefill_text_len=t.get("prefill_text_len", 6), prompt_max_s=t.get("prompt_max_s", 4.0),
                                   max_length=t.get("max_length", 800), channels=t.get("channels", 16),
                                   do_sample=True, audio_temperature=t.get("audio_temperature", 0.8),
                                   audio_top_p=t.get("audio_top_p", 0.6), audio_top_k=t.get("audio_top_k", 30),
                                   audio_repetition_penalty=t.get("audio_repetition_penalty", 1.1),
                                   repetition_window=t.get("repetition_window", 50))
        self.cfg = cfg
        self.voices, self.voice = discover_voices(cfg)
        if not self.voices:
            raise RuntimeError("no voice reference clips found (voices.dir / voices.default_ref)")
        log.info("voices available: %s (default=%s)", sorted(self.voices), self.voice)
        self.tts.set_voice(self.voices[self.voice])
        self.sr = self.tts.out_sr
        self.bc_max_new = int(m.get("bc_max_new", 12))
        self.history = []
        self.acks = {}
        log.info("warming TTS (voice=%s, contextual_acks=%s)", self.voice, self.contextual_acks)
        self.tts.warmup()
        if not self.contextual_acks:      # canned acks are a fallback only; skip the pre-render otherwise
            self._render_acks()

    # ---- voice + acks ----
    def _render_acks(self):
        self.acks = {}
        for txt in self.canned_texts:
            w = self.tts.render(txt)
            if w.size:
                self.acks[txt] = w
        log.info("pre-rendered %d canned acks", len(self.acks))

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
        if not self.contextual_acks:
            self._render_acks()
        log.info("voice -> %s", name)
        return True

    # ---- history ----
    def reset_history(self):
        self.history = []

    def _remember(self, user, reply):
        if user.strip():
            self.history.append({"role": "user", "content": user.strip()})
            self.history.append({"role": "assistant", "content": reply or "..."})
            self.history = self.history[-2 * self.history_turns:]

    # ---- TTS of the untts'd tail, one whole sentence at a time ----
    def _tts_tail(self, full, stop_flag):
        safe = full[self._tts_done:]
        while True:
            chunk, rest = next_sentence(safe)
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

    # ---- a full turn (may call tools) ----
    def stream_turn(self, transcript, stop_flag=None):
        msgs = [{"role": "system", "content": TTS_TOOLS_SYS}] + list(self.history) + \
               [{"role": "user", "content": transcript}]
        content = ""; self._tts_done = 0; tool_acc = {}; stopped = False
        for d in self.llm.stream_deltas(msgs, tools=TOOLS, max_new=self.max_gen, stop_flag=stop_flag):
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
                yield ("text", content)
                for ev in self._tts_tail(content, stop_flag):
                    yield ev
            for tc in (delta.get("tool_calls") or []):
                i = tc.get("index", 0)
                e = tool_acc.setdefault(i, {"name": "", "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    e["name"] = fn["name"]
                if fn.get("arguments"):
                    e["args"] += fn["arguments"]
            if stop_flag and stop_flag():
                stopped = True; break
        if stopped:
            yield ("done", content.strip()); return

        if not tool_acc:                                    # plain reply
            for ev in self._flush_tail(content, stop_flag):
                yield ev
            reply = content.strip(); self._remember(transcript, reply)
            yield ("done", reply); return

        # ---- tool path: run tool(s) in BG threads, speak a lead-in over them, then phase B ----
        calls = []
        for i in sorted(tool_acc):
            e = tool_acc[i]
            if not e["name"]:
                continue
            try:
                args = json.loads(e["args"]) if e["args"].strip() else {}
            except Exception:
                args = {}
            calls.append({"name": e["name"], "arguments": args})
        log.info("tool calls: %s", [c["name"] for c in calls])
        results = [None] * len(calls)

        def _run(i, c):
            results[i] = execute_tool(c["name"], c.get("arguments", {}))
        workers = [threading.Thread(target=_run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
        for w in workers:
            w.start()

        lead_ids = []
        for d in self.llm.stream_deltas([{"role": "system", "content": LEADIN_SYS},
                                         {"role": "user", "content": transcript}],
                                        tools=None, max_new=24, stop_flag=stop_flag):
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                lead_ids.append(delta["content"])
        lead = "".join(lead_ids).strip()
        if lead and not (stop_flag and stop_flag()):
            yield ("text", lead)
            for w in self.tts.stream(say_clean(lead), stop_flag=stop_flag):
                yield ("audio", w)
        for w in workers:
            w.join()
        if stop_flag and stop_flag():
            yield ("done", lead); return

        asst = {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                               for i, c in enumerate(calls)]}
        msgs2 = msgs + [asst] + [{"role": "tool", "tool_call_id": f"call_{i}",
                                  "content": json.dumps(r if r is not None else {"error": "no result"})}
                                 for i, r in enumerate(results)]
        full2 = ""; self._tts_done = 0; stopped2 = False
        for d in self.llm.stream_deltas(msgs2, tools=None, max_new=self.max_gen, stop_flag=stop_flag):
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                full2 += delta["content"]
                yield ("text", full2)
                for ev in self._tts_tail(full2, stop_flag):
                    yield ev
            if stop_flag and stop_flag():
                stopped2 = True; break
        if not stopped2:
            for ev in self._flush_tail(full2, stop_flag):
                yield ev
        reply = (lead + " " + full2).strip(); self._remember(transcript, full2.strip() or lead)
        yield ("done", reply)

    # ---- backchannel: contextual LLM ack (default) or canned (fallback) ----
    def stream_bc(self, partial, stop_flag=None):
        if not self.contextual_acks and self.acks:
            txt = random.choice(list(self.acks.keys()))
            yield ("text", txt)
            yield ("audio", self.acks[txt])
            yield ("done", txt); return
        user = (partial or "").strip() or "(the user is speaking)"
        messages = [{"role": "system", "content": BC_PROMPT},
                    {"role": "user", "content": f"The user is currently saying: \"{user}\""}]
        parts = []
        for d in self.llm.stream_deltas(messages, tools=None, max_new=self.bc_max_new, stop_flag=stop_flag):
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                parts.append(delta["content"])
        text = "".join(parts).strip()
        if text:
            yield ("text", text)
            for w in self.tts.stream(say_clean(text), stop_flag=stop_flag):
                yield ("audio", w)
        yield ("done", text)


def serve(m, conn):
    turns = queue.Queue(); stop = threading.Event()
    m.reset_history(); log.info("history reset")

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
                for kind, val in m.stream_bc(turn.get("partial", ""), stop_flag=stop.is_set):
                    if kind == "text":
                        send_msg(conn, b"X", {"text": val, "bc": True})
                    elif kind == "audio":
                        send_msg(conn, b"A", val.tobytes())
                    elif kind == "done":
                        send_msg(conn, b"D", {"text": val, "bc": True})
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
                            log.info("onset: first-text %.0fms first-audio %.0fms", t_txt * 1e3, t_aud * 1e3)
                        send_msg(conn, b"A", val.tobytes())
                    elif kind == "done":
                        send_msg(conn, b"D", {"text": val, "barged": stop.is_set()})
        except Exception as e:
            log.exception("turn error: %s", e)
            send_msg(conn, b"D", {"text": "", "error": str(e)})


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # ensure MOSS-Realtime is vendored+patched (static-cache/CUDA-graph patch; idempotent)
    weights.ensure_moss_realtime(cfg["models"]["moss_repo"], cfg["models"]["moss_dir"])
    if not os.environ.get("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set -> web_search/get_weather will return an error")
    m = Mouth(cfg)
    host, port = cfg["server"]["mouth_host"], int(cfg["server"]["mouth_port"])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(1)
    log.info("=== MOUTH ready (vLLM responder + in-proc MOSS-Nano, sr=%d, voice=%s) listening on %s:%d ===",
             m.sr, m.voice, host, port)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("EARS connected")
        try:
            serve(m, conn)
        except Exception as e:
            log.exception("conn error: %s", e)
        finally:
            conn.close(); log.info("EARS disconnected")


if __name__ == "__main__":
    main()
