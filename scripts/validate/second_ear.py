"""The 2nd ear, checked on the EARS engine. Agent and user clips are public LibriSpeech, the acknowledgements VoxCPM2's
zero-shot voice (a clone of no one); all held in memory, numbers printed only. Needs the GPU free (stop run.sh).

R-04b, the default: the 2nd ear's 24 -> 16 kHz step. MOUTH streams the reply at the 24 kHz wire rate in chunks;
EARS brings it to 16 kHz for the turn-taking model's agent channel. Compares the old per-chunk np.interp (kept here
as the reference) with AgentPlayback's stateful polyphase resampler, at several chunk sizes:
  * fidelity: SNR of the 2nd ear's audio against the original 16 kHz clip (the clip goes 16 -> 24 kHz first, so a
    perfect 24 -> 16 step gives it back);
  * the model: EarsEngine fed as the server feeds it (speaking mode, agent channel = the 2nd ear). Agent alone
    (the user silent): BARGE events (each one false) and the mean yield probability. User cuts in 1.5 s into the
    reply: barges and the median time from the user's start to BARGE.

I-17, with --offset-ms: the 2nd ear runs ahead of what the user heard. EARS pairs the agent's audio with the mic in
arrival order, while the page plays it later and the mic's audio of that moment arrives later still, so live the
agent channel leads the user's by ~150-700 ms. Each offset shifts the agent channel that much earlier than the
user's. The reply is three sentences (2-6 s clips) with --pause-s between them, and the mic carries a residual echo
of what the user hears (--echo-db). Three cases:
  * the agent alone (the mic: the echo only): every BARGE is false;
  * the user cuts in 1.5 s into the reply: how many barge, and the time from the user's start to the BARGE;
  * short acknowledgements ("mm-hm", "yeah", "right") in the agent's pauses, as the user hears them: none should
    barge.
--pairing tags builds the agent channel as EARS has since I-17 step 2, from the page's tags, so the offset drops
out. --echo-db and --barge-thr take comma lists (a sweep in one load; write --echo-db=-30,-40 for the parser).

  .venv/bin/python scripts/validate/second_ear.py [--trials 12] [--chunk-ms 40,80,160]
  .venv/bin/python scripts/validate/second_ear.py --offset-ms 0,100,200,400 [--trials 24] [--echo-db -30]
  .venv/bin/python scripts/validate/second_ear.py --offset-ms 0 --pairing tags --echo-db=-30,-120 --barge-thr 0.5,0.7
"""
import sys, argparse, logging
from pathlib import Path

import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); torch.set_num_threads(4)
logging.basicConfig(level=logging.WARNING)
from datasets import load_dataset  # noqa: E402
from app.plugins.ears.nemotron_tyger.release import heads_release  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ears.playback import AgentPlayback  # noqa: E402
from app.plugins.ears.nemotron_tyger.constants import HOP_N  # noqa: E402
from app.plugins.ears.nemotron_tyger.engine import EarsEngine  # noqa: E402
from app.plugins.ears.nemotron_tyger.session import EarsSession  # noqa: E402
from app.plugins.tts import WIRE_SR  # noqa: E402
from app.runtime.audio import StreamingAudioResampler  # noqa: E402

SR = 16000


def old_per_chunk_interp(a24):
    """The 2nd ear's 24 -> 16 kHz step before R-04b, applied to each chunk on its own."""
    n = int(len(a24) * SR / WIRE_SR)
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(a24) - 1, n), np.arange(len(a24)), a24).astype(np.float32)


def to_wire(clip16):
    return StreamingAudioResampler(SR, WIRE_SR).process(clip16, final=True)


def chunks(a24, chunk_ms, rng):
    """Cut the wire audio as a streaming TTS does: chunks around chunk_ms long (+-50%)."""
    out, i = [], 0
    while i < len(a24):
        n = max(1, int(WIRE_SR * chunk_ms / 1e3 * rng.uniform(0.5, 1.5)))
        out.append(a24[i:i + n]); i += n
    return out


def ear(method, pieces):
    if method == "old":
        return np.concatenate([old_per_chunk_interp(c) for c in pieces])
    pb = AgentPlayback(SR); pb.start()
    for c in pieces:
        pb.add(c, 0)
    return pb.audio[:pb.n16].copy()


def snr_db(ref, x):
    n = min(len(ref), len(x)); ref, x = ref[:n], x[:n]
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-12))


def run(eng, thr, user, agent_ear, barge_at_s=None, stop_after_s=None):
    """Feed one reply as the server does; returns the BARGE times (s from the start) and the yield probs. Stops at
    the first BARGE with `barge_at_s`, or at the first one after `stop_after_s`."""
    s = EarsSession(0, **thr); s.trace = []
    u = user if user is not None else np.zeros(len(agent_ear), np.float32)
    n = min(len(u), len(agent_ear)) // HOP_N
    barges = []
    for h in range(n):
        sl = slice(h * HOP_N, (h + 1) * HOP_N)
        events, _ = eng.process([(s, u[sl], agent_ear[sl], "speaking")])[s]
        if any(ev["kind"] == "BARGE" for ev in events):
            barges.append((h + 1) * HOP_N / SR)
            if barge_at_s is not None or (stop_after_s is not None and barges[-1] > stop_after_s):
                break
    eng.head.reset_slot(0)
    return barges, [t[2] for t in s.trace]


# ---- I-17: the agent channel ahead of the user's ---------------------------------------------------------------
ACKS = ("Mm-hm.", "Yeah.", "Right.")
ACK_DELAY_S = 0.1                # an acknowledgement starts this long into the pause the user hears
ACK_WINDOW_S = 1.0               # a barge this long after an acknowledgement's end still counts against it
CUT_IN_S = 1.5


def trim(x):
    """x without its quiet ends (20 ms frames more than 35 dB under its loudest)."""
    f = int(0.02 * SR); n = len(x) // f
    if n == 0:
        return x
    e = 20 * np.log10(np.sqrt((x[:n * f].reshape(n, f) ** 2).mean(1)) + 1e-9)
    v = np.nonzero(e > e.max() - 35)[0]
    return x[v[0] * f:(v[-1] + 1) * f]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def make_acks(cfg, level):
    """The acknowledgements in VoxCPM2's zero-shot voice (no clip: a clone of no one), 16 kHz, trimmed, at `level`
    RMS; in memory. Loads VoxCPM2 and shuts it down again before EARS loads."""
    from app.plugins import build_tts
    from app.runtime.event_loop import EngineLoop
    tts = build_tts(cfg, EngineLoop())
    out = []
    for text in ACKS:
        a = trim(StreamingAudioResampler(WIRE_SR, SR).process(np.concatenate(list(tts.stream(text, None))),
                                                                 final=True))
        out.append((a * level / rms(a)).astype(np.float32))
    tts.shutdown()
    return out


def reply_with_pauses(clips, idx, pause_s):
    """Three sentences (trimmed clips at the agent's level) with `pause_s` between them: the reply at 16 kHz and the
    times (s) its pauses start."""
    parts, pauses, t = [], [], 0.0
    for k, i in enumerate(idx):
        s = (0.6 * trim(clips[i])).astype(np.float32)
        parts.append(s); t += len(s) / SR
        if k < len(idx) - 1:
            pauses.append(t); parts.append(np.zeros(int(pause_s * SR), np.float32)); t += pause_s
    return np.concatenate(parts), pauses


def place(base, x, at_s):
    out = base.copy(); i = int(at_s * SR)
    n = min(len(x), len(out) - i)
    out[i:i + n] += x[:n]
    return out


def tagged_ear(pieces, n16, block=320):
    """The 2nd ear EARS builds from the page's tags (I-17 step 2): the reply's chunks as MOUTH sends them, then mic
    blocks (20 ms, as the page sends them) tagged with the reply sample the user hears at each block's start. The
    network offset drops out: the tags carry the timing."""
    pb = AgentPlayback(SR); rid = pb.start()
    for c in pieces:
        pb.add(c, 0)
    out = []
    for pos in range(0, n16, block):
        at24 = pos * WIRE_SR // SR
        out.append(pb.agent_block(rid, at24 if at24 < pb.sent24 else -1, min(block, n16 - pos), pos))
    return np.concatenate(out)


def agent_channel(pieces, ear16, offset_ms, pairing):
    """The agent channel EARS feeds the heads while the user hears ear16 from t = 0: arrival-order pairing puts it
    `offset_ms` ahead; the page's tags put it where it played."""
    if pairing == "tags":
        return tagged_ear(pieces, len(ear16))
    d = int(offset_ms * SR / 1000)
    return np.concatenate([ear16[d:], np.zeros(d, np.float32)])


def offset_study(eng, thr, clips, acks, a, echo_db):
    sent = [i for i, c in enumerate(clips) if 2.0 <= len(c) / SR <= 6.0]
    users = [i for i, c in enumerate(clips) if 4.0 <= len(c) / SR <= 12.0]
    offsets = [int(x) for x in a.offset_ms.split(",")]
    print(f"I-17, the 2nd ear by {'arrival order' if a.pairing == 'arrival' else 'the page tags'}: trials {a.trials} "
          f"per row; replies of 3 sentences, {a.pause_s:g} s pauses; agent level 0.6; echo {echo_db:g} dB in the mic; "
          f"barge_thr {thr['barge_thr']}, barge_deb {thr['barge_deb']}; acks "
          + ", ".join(f"{len(x) / SR:.2f} s" for x in acks))
    print("(a barge ends the agent's turn live, so: trials with a false barge; acks followed by a barge within "
          f"{ACK_WINDOW_S:g} s of their end)")
    print(f"{'agent ahead':>11} | {'alone: trials with a false barge':>32} | {'cut in: barged':>14} {'median':>7} "
          f"{'p90':>6} | {'acks in pauses: barged':>22}")
    for off in offsets:
        rng = np.random.default_rng(17)                       # the same replies and cuts at every offset
        false_alone, lat, n_cut, ack_barges, n_acks = 0, [], 0, 0, 0
        by_ack = {k: [0, 0] for k in range(len(acks))}        # per acknowledgement: [barged, said]
        for t in range(a.trials):
            idx = [sent[(3 * t + k) % len(sent)] for k in range(3)]
            agent16, pauses = reply_with_pauses(clips, idx, a.pause_s)
            pieces = chunks(to_wire(agent16), 80, rng)
            ear16 = ear("new", pieces)[:len(agent16)]
            ear16 = np.pad(ear16, (0, len(agent16) - len(ear16)))
            echo = (ear16 * 10 ** (echo_db / 20)).astype(np.float32)
            ch = agent_channel(pieces, ear16, off, a.pairing)
            b, _ = run(eng, thr, echo, ch)                                  # the agent alone
            false_alone += bool(b)
            user = place(echo, clips[users[t % len(users)]], CUT_IN_S)      # the user cuts in
            b, _ = run(eng, thr, user, ch, stop_after_s=CUT_IN_S)
            n_cut += 1
            hit = [x for x in b if x > CUT_IN_S]
            if hit:
                lat.append(hit[0] - CUT_IN_S)
            user, spans = echo, []
            for k, p in enumerate(pauses):                                  # acknowledgements in the pauses
                j = (t + k) % len(acks)
                user = place(user, acks[j], p + ACK_DELAY_S); n_acks += 1
                spans.append((j, p + ACK_DELAY_S, p + ACK_DELAY_S + len(acks[j]) / SR + ACK_WINDOW_S))
            b, _ = run(eng, thr, user, ch)
            for j, lo, hi in spans:
                hit = any(lo < x <= hi for x in b)
                ack_barges += hit; by_ack[j][0] += hit; by_ack[j][1] += 1
        med = f"{np.median(lat):.2f}" if lat else "-"
        p90 = f"{np.percentile(lat, 90):.2f}" if lat else "-"
        print(f"{off:>8} ms | {false_alone:>23d} / {a.trials:<6d} | {len(lat):>6d}/{n_cut:<7d} {med:>7} {p90:>6} | "
              f"{ack_barges:>13d} / {n_acks:<6d} ("
              + ", ".join(f"{ACKS[j]} {h}/{n}" for j, (h, n) in by_ack.items()) + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--chunk-ms", default="40,80,160")
    ap.add_argument("--offset-ms", default=None, help="I-17: the agent channel this far ahead of the user's")
    ap.add_argument("--echo-db", default="-30", help="I-17: the agent's residual echo in the mic (a comma list sweeps)")
    ap.add_argument("--pause-s", type=float, default=0.5, help="I-17: the pause between the reply's sentences")
    ap.add_argument("--pairing", choices=("arrival", "tags"), default="arrival",
                    help="I-17: the 2nd ear by arrival order (before) or from the page's tags (step 2)")
    ap.add_argument("--barge-thr", default=None,
                    help="override the config's barge threshold (I-17: a comma list sweeps)")
    a = ap.parse_args()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    clips = [np.asarray(ds[i]["audio"]["array"], np.float32) for i in range(len(ds))]
    ok = [i for i, c in enumerate(clips) if 4.0 <= len(c) / SR <= 12.0]
    cfg = load_config(); e = cfg["ears"]
    thr = dict(ep_thr=float(e["ep_thr"]), ep_deb=int(e["ep_deb"]), barge_thr=float(e["barge_thr"]),
               barge_deb=int(e["barge_deb"]), bc_thr=float(e["bc_thr"]), bc_deb=int(e["bc_deb"]))
    barge_thrs = [float(x) for x in a.barge_thr.split(",")] if a.barge_thr else [thr["barge_thr"]]
    acks = make_acks(cfg, np.median([rms(c) for c in clips])) if a.offset_ms else None
    eng = EarsEngine(heads_release(e["heads"], e.get("heads_revision")), slots=1)
    if a.offset_ms:
        for echo_db in (float(x) for x in a.echo_db.split(",")):
            for b in barge_thrs:
                offset_study(eng, dict(thr, barge_thr=b), clips, acks, a, echo_db)
        return
    thr["barge_thr"] = barge_thrs[0]
    print(f"trials {a.trials} per row; agent level 0.6; user cuts in at 1.5 s")
    print(f"{'chunk':>6} {'method':>6} | {'SNR dB':>7} | {'alone: false barges':>19} {'mean yield':>10} | "
          f"{'cut in: barges':>14} {'median latency':>14}")
    for chunk_ms in (int(x) for x in a.chunk_ms.split(",")):
        latency = {}
        for method in ("old", "new"):
            rng = np.random.default_rng(chunk_ms)               # same cuts for both methods
            snrs, false_barges, ys, lat = [], 0, [], []
            for t in range(a.trials):
                ai = ok[t % len(ok)]; ui = ok[(t + 5) % len(ok)]
                agent16 = 0.6 * clips[ai]
                ear16 = ear(method, chunks(to_wire(agent16), chunk_ms, rng))
                snrs.append(snr_db(agent16, ear16))
                b, y = run(eng, thr, None, ear16)
                false_barges += len(b); ys += y
                user = np.concatenate([np.zeros(int(1.5 * SR), np.float32), clips[ui]])
                user = np.pad(user, (0, max(0, len(ear16) - len(user))))
                b, _ = run(eng, thr, user, np.pad(ear16, (0, max(0, len(user) - len(ear16)))), barge_at_s=1.5)
                lat.append(b[0] - 1.5 if b else None)
            latency[method] = lat
            got = [x for x in lat if x is not None]
            print(f"{chunk_ms:>4}ms {method:>6} | {np.median(snrs):7.1f} | {false_barges:>19d} {np.mean(ys):10.3f} | "
                  f"{len(got):>8d}/{a.trials:<5d} {np.median(got) if got else float('nan'):>12.2f} s")
        both = [(o, n) for o, n in zip(latency["old"], latency["new"]) if o is not None and n is not None]
        hop = HOP_N / SR
        print(f"       paired, trials both barged ({len(both)}): new earlier {sum(n < o for o, n in both)}, same "
              f"{sum(n == o for o, n in both)}, later {sum(n > o for o, n in both)}; mean shift "
              f"{np.mean([n - o for o, n in both]) / hop if both else float('nan'):+.2f} hops | only old barged "
              f"{sum(o is not None and n is None for o, n in zip(latency['old'], latency['new']))}, only new "
              f"{sum(o is None and n is not None for o, n in zip(latency['old'], latency['new']))}")

if __name__ == "__main__":
    main()
