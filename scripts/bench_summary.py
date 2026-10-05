"""The comparison numbers from scripts/bench_stack.sh's logs (bench_runs/<tag>/), numbers only.
Run: .venv/bin/python scripts/bench_summary.py TAG [TAG...]"""
import re, sys, glob, os, statistics

S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_runs")


def grab(path, pat, n=1):
    m = re.search(pat, open(path).read())
    return tuple(int(m.group(i)) for i in range(1, n + 1)) if m else None


for tag in sys.argv[1:]:
    d = f"{S}/{tag}"
    print(f"=== {tag}")
    print("  smoke:", " / ".join(re.findall(r"first-audio=(\d+)ms", open(f"{d}/smoke.log").read())[:4]),
          "ms;", "PASS" if "SMOKE OK" in open(f"{d}/smoke.log").read() else "FAIL")
    for kind in ("plain", "tools"):
        fa, bc, peak, dead, err = [], [], [], [], []
        for f in sorted(glob.glob(f"{d}/lt_{kind}_*.log")):
            fa.append(grab(f, r"turn first-audio\s+p50 (\d+) / max (\d+)", 2))
            bc.append(grab(f, r"backchannel first-audio p50 (\d+)")[0])
            peak.append(grab(f, r"GPU memory peak\s+(\d+)")[0])
            dead.append(grab(f, r"dead air / turn p50 (\d+) / max (\d+)", 2))
            err.append(grab(f, r"errors=(\d+)")[0])
        print(f"  load_test {kind}: turn first-audio p50 {' / '.join(str(x[0]) for x in fa)} "
              f"(median {statistics.median(x[0] for x in fa):.0f}), max {max(x[1] for x in fa)}; "
              f"bc p50 {' / '.join(map(str, bc))}; dead air max {max(x[1] for x in dead)}; GPU peak {max(peak)} MiB; "
              f"errors {sum(err)}")
    for n in (1, 4):
        rows = []
        for f in sorted(glob.glob(f"{d}/ears{n}_*.log")):
            t = open(f).read()
            rows.append((grab(f, r"end of speech -> endpoint event:\s+p50 (\d+) / p90 (\d+)", 2),
                         grab(f, r"endpoint -> first reply audio:\s+p50 (\d+)")[0],
                         grab(f, r"end of speech -> first reply audio:\s+p50 (\d+) / p90 (\d+)", 2),
                         grab(f, r"hop lag.*?p50 (\d+) / p90 (\d+)", 2),
                         re.search(r"distinct reply texts: (\d+)", t).group(1),
                         re.search(r"fired inside the utterance: (\d+)", t).group(1)))
        print(f"  ears {n} conv: detect p50/p90 {', '.join(f'{r[0][0]}/{r[0][1]}' for r in rows)}; "
              f"ep->audio p50 {', '.join(str(r[1]) for r in rows)}; "
              f"GAP p50/p90 {', '.join(f'{r[2][0]}/{r[2][1]}' for r in rows)} "
              f"(median p50 {statistics.median(r[2][0] for r in rows):.0f}); "
              f"hop lag p50/p90 {', '.join(f'{r[3][0]}/{r[3][1]}' for r in rows)}; "
              f"distinct replies {','.join(r[4] for r in rows)}; inside {','.join(r[5] for r in rows)}")
    for f in ("gpu_idle.txt", "gpu_after_lt.txt"):
        print(f"  {f}: {open(f'{d}/{f}').read().strip()}")
