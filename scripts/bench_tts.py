"""MOSS-TTS-Nano bench (MOUTH venv): per-voice first-audio (p50/p90) + RTF over a set of session-style
English replies, and optional sample rendering for listening. CER is measured separately with nemotron in
the EARS venv (see NOTES.md); kept out of here to avoid loading two transformers versions in one process.

Run (MOUTH venv):
  HF_HOME=/workspace/.hf_home <venv457>/bin/python v1/scripts/bench_tts.py [--samples N] [--out DIR]
"""
import os, sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from app.config import load_config, discover_voices
from app.tts_nano import MossNanoTTS
from app import weights
from app.chunker import next_chunk, say_clean

REPLIES = [
    "Sure, I can help with that.", "Got it, let me take care of it.", "That sounds like a great plan.",
    "Hmm, let me think about that for a second.", "No problem at all, happy to help.",
    "Right, so the first step is to open the settings menu.",
    "The meeting is scheduled for three o'clock this afternoon.",
    "Honestly, that's a really interesting question.", "Let me check the latest numbers and get back to you.",
    "It looks like the weather will be clear and mild tomorrow.",
    "If you want, I can walk you through it step by step.",
    "I'm really sorry to hear that, that sounds frustrating.", "Congratulations, that's wonderful news.",
    "The train leaves from platform four in about ten minutes.",
    "Here's a quick summary of what we covered today.",
    "Let me pull up the directions to the nearest station.",
    "We can always adjust the plan if something comes up.", "Good morning, I hope you slept well.",
    "Absolutely, I'll send that over to you right away.",
    "The short answer is yes, and here's the reasoning behind it.",
]
SAMPLE_TEXTS = [
    "Hey, it's good to hear from you. What would you like to do today?",
    "Let me look that up for you real quick.",
    "That's a great question, and the answer is a little surprising.",
    "No worries at all, we can take this one step at a time.",
    "Alright, I think we're all set. Anything else you need?",
]


def pctl(xs, p):
    return float(np.percentile(np.array(xs), p)) if xs else float("nan")


def bench_voice(tts, ref, name):
    # warmup
    for w in ("Hello there.", "Let me check that now."):
        for _ in tts.stream(w, ref_path=ref):
            pass
    fa, rtf = [], []
    for txt in REPLIES:
        # mirror the live path: app-level clause chunking, time to first audio of the reply
        t0 = time.time(); first = None; nsamp = 0; pend = txt; done = False
        while not done:
            chunk, rest = next_chunk(pend)
            if chunk is None:
                chunk, rest, done = say_clean(pend), "", True
            else:
                pend = rest
            chunk = say_clean(chunk)
            if not chunk:
                continue
            for wv in tts.stream(chunk):
                if first is None:
                    first = time.time() - t0
                nsamp += wv.shape[-1]
        gen = time.time() - t0
        audio_s = nsamp / tts.out_sr
        if first is not None and audio_s > 0:
            fa.append(first * 1e3); rtf.append(gen / audio_s)
    return dict(voice=name, n=len(fa), first_audio_ms_p50=pctl(fa, 50), first_audio_ms_p90=pctl(fa, 90),
                rtf_p50=pctl(rtf, 50), rtf_p90=pctl(rtf, 90), rtf_mean=float(np.mean(rtf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0, help="render N listenable samples per voice to --out")
    ap.add_argument("--out", default="/workspace/tyger_samples")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    weights.ensure_moss(cfg["models"]["moss_repo"], cfg["models"]["moss_dir"])
    t = cfg["tts"]; mdl = cfg["models"]
    tts = MossNanoTTS(mdl["moss_dir"], mdl["moss_codec"], out_sr=t["out_sr"], dtype=t["dtype"], attn=t["attn"],
                      max_new_frames=t["max_new_frames"], vc_max_tokens=t["vc_max_tokens"])
    voices, _ = discover_voices(cfg)
    results = []
    for name, ref in voices.items():
        tts.set_voice(ref)
        print(f"== voice {name} ({ref}) ==", flush=True)
        r = bench_voice(tts, ref, name); results.append(r)
        print(json.dumps(r, indent=2), flush=True)
        if args.samples:
            import soundfile as sf
            d = Path(args.out) / name; d.mkdir(parents=True, exist_ok=True)
            for i, txt in enumerate(SAMPLE_TEXTS[:args.samples]):
                w = tts.render(txt, ref_path=ref)
                sf.write(str(d / f"sample_{i:02d}.wav"), w, tts.out_sr)
            print(f"  wrote {args.samples} samples to {d}", flush=True)
    print("\n=== BENCH SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
