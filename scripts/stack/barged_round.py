"""I-16 live check: a round the user cuts in on goes on as a background task. Calls the `smart` model (hosted:
needs the user's OK for each run). Needs run.sh up with a config that has llm.smart and mouth.tasks.

One WebSocket conversation, its speech made by MOUTH's TTS (S: no LLM) and held in memory: ask a question the reply
model hands to the expert, cut in on the expert's lead-in with a quick question of its own, stay quiet after its
answer, and wait for the expert's answer to come back as a background task (task_done, then its audio). The user's
lines are spoken in another voice than the agent's (`--user-voice`, a clip in voices/): in the agent's own voice a
cut-in sounds like its echo, and the turn-taking heads don't barge. Prints event types, timings and yes / no checks
on the replies (keywords), never their words. Which path MOUTH took comes from its log (`--stack-log`, counts
only): the round went on as a task, or the user had already heard part of the answer when the barge fired (then
nothing is said later, by design).

  .venv/bin/python scripts/stack/barged_round.py --stack-log STACK_LOG [--user-voice NAME] [--cut-in-after 0.0]
"""
import sys, json, time, asyncio, argparse, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import load_config  # noqa: E402
from scripts.stack.load_ears import speech_from_mouth, CHUNK  # noqa: E402
from scripts.stack.page import PageSim, open_ws  # noqa: E402

ASK = "If I save three hundred dollars every month for fifteen years at five percent interest, how much will I have?"
CUT_IN = "Wait, sorry, quick one first: what is the capital of Australia?"
NUMBER = re.compile(r"\d|thousand|hundred", re.I)          # the expert's answer gives an amount


def other_voice(cfg):
    """The first voice clip that isn't the agent's: the user's lines must not sound like the agent's own echo."""
    from app.plugins import plugin_class
    names, agent = plugin_class("tts", cfg["tts"]["plugin"]).list_voices(cfg)
    rest = [n for n in names if n != agent]
    if not rest:
        sys.exit("the user's lines need a second voice clip in voices/ (or --user-voice)")
    return rest[0]


async def run(port, ask, cut_in, cut_in_after, timeout_s, barge_thr=None, turns=3):
    import aiohttp
    ev = []                                     # (t, kind) from the conversation's start
    texts = {"reply": [], "announce": []}        # kept in memory for the keyword checks; never printed
    yields = []                                  # (t, the yield head's peak in a hop) while the agent speaks
    t0 = time.perf_counter()
    mark = lambda kind: ev.append((time.perf_counter() - t0, kind))   # noqa: E731
    queue = [np.zeros(int(0.5 * 16000), np.float32), ask]
    state = {"phase": "ask", "audio_after_ask": False, "announce": False, "done": 0}
    async with aiohttp.ClientSession() as sess:
        async with open_ws(sess, port) as ws:
            assert json.loads((await ws.receive()).data)["type"] == "ready"
            if barge_thr is not None:                       # the page's live-tuning command, this conversation only
                await ws.send_str(f"set:barge_thr:{barge_thr}")
            stop = asyncio.Event(); page = PageSim(ws)

            async def mic():
                n = 0; buf = np.zeros(0, np.float32); start = time.perf_counter()
                while not stop.is_set():
                    if len(buf) < CHUNK and queue:
                        seg = queue.pop(0)
                        if seg is cut_in:
                            mark("cut-in speech starts")
                        buf = np.concatenate([buf, seg])
                    chunk = buf[:CHUNK]; buf = buf[CHUNK:]
                    await page.send_mic(np.pad(chunk, (0, CHUNK - len(chunk))), time.perf_counter() - CHUNK / 16000)
                    n += 1
                    await asyncio.sleep(max(0.0, start + n * CHUNK / 16000 - time.perf_counter()))

            task = asyncio.create_task(mic())
            while time.perf_counter() - t0 < timeout_s and state["done"] < turns:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.BINARY:
                    rid, _ = page.on_audio(msg.data)
                    if not rid:
                        continue                                # a backchannel
                    if state["phase"] == "leadin" and not state["audio_after_ask"]:
                        state["audio_after_ask"] = True; mark("lead-in audio")
                        await asyncio.sleep(cut_in_after)
                        queue.append(cut_in)
                    elif state["announce"]:
                        state["announce"] = False; mark("announcement audio")
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                m = json.loads(msg.data)
                if m["type"] == "state" and state["phase"] == "leadin":
                    yields.append((time.perf_counter() - t0, m.get("yield") or 0.0))
                elif m["type"] == "endpoint":
                    mark(f"endpoint ({m.get('by')})"); page.stop()
                    if state["phase"] == "ask":
                        state["phase"] = "leadin"
                elif m["type"] == "barge":
                    mark("barge"); page.stop()
                elif m["type"] == "task_done":
                    mark(f"task_done #{m.get('task')}"); state["announce"] = True; state["phase"] = "announce"
                elif m["type"] == "reply":
                    texts["announce" if state["phase"] == "announce" else "reply"].append(m.get("text", ""))
                elif m["type"] == "turn_done":
                    mark(f"turn_done{' (barged)' if m.get('barged') else ''}"); state["done"] += 1
                    if state["phase"] == "leadin":
                        state["phase"] = "after"
            stop.set(); await task
    return ev, texts, yields


def mouth_path(log_path, since):
    """What the stack's log (counts only) says about the barged round, from line `since` on."""
    keys = ("barged tool round", "tool calls:", "task #", "floor granted", "barge yield")
    out = []
    for ln in Path(log_path).read_text(errors="replace").splitlines()[since:]:
        m = re.search(r" (mouth|tasks|ears): (.*)$", ln)
        if m and any(k in m.group(2) for k in keys):
            out.append(f"{m.group(1)}: {m.group(2)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-log", required=True, help="run.sh's log, to read which path MOUTH took")
    ap.add_argument("--cut-in-after", type=float, default=0.0, help="s after the lead-in's first audio")
    ap.add_argument("--user-voice", default=None, help="the voice (a clip in voices/) the user's lines are said in "
                    "(default: the first clip that isn't the agent's)")
    ap.add_argument("--ask", default=ASK, help="the first question (default: one the reply model escalates)")
    ap.add_argument("--barge-thr", type=float, default=None,
                    help="this conversation's barge threshold (the page's slider; default: the config's)")
    ap.add_argument("--turns", type=int, default=3, help="stop after this many turns (3: with the announcement)")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    voice = a.user_voice or other_voice(cfg)
    ask, cut_in = (speech_from_mouth(cfg, x, voice=voice) for x in (a.ask, CUT_IN))
    since = len(Path(a.stack_log).read_text(errors="replace").splitlines())
    ev, texts, yields = asyncio.run(run(int(cfg["server"]["port"]), ask, cut_in, a.cut_in_after, a.timeout,
                                        a.barge_thr, a.turns))
    print(f"speech: the question {len(ask) / 16000:.1f} s, the cut-in {len(cut_in) / 16000:.1f} s")
    for t, kind in ev:
        print(f"  {t:6.2f} s  {kind}")
    when = {k: t for t, k in ev}
    if "cut-in speech starts" in when:
        c0 = when["cut-in speech starts"]
        ys = [(t - c0, y) for t, y in yields if t >= c0]
        if ys:
            print("yield head per hop after the cut-in started (s: peak): "
                  + " ".join(f"{t:.1f}:{y:.2f}" for t, y in ys[:20]))
    if "barge" in when and "cut-in speech starts" in when:
        print(f"cut-in start -> barge: {when['barge'] - when['cut-in speech starts']:.2f} s")
    task = next((t for t, k in ev if k.startswith("task_done")), None)
    if task is not None and "barge" in when:
        print(f"barge -> task_done (the announcement takes the floor): {task - when['barge']:.1f} s")
    print(f"the cut-in question answered (Canberra): {any('canberra' in x.lower() for x in texts['reply'])}")
    print(f"the announcement gives an amount: {any(NUMBER.search(x) for x in texts['announce'])}")
    print("MOUTH's log for this run:")
    for ln in mouth_path(a.stack_log, since):
        print("  " + ln)


if __name__ == "__main__":
    main()
