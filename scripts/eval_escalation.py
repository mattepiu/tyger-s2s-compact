"""Escalation routing eval (PLUGINS.md step 11): does the `reply` model answer locally, call a lookup tool, or
call ask_expert, on the labelled synthetic set in scripts/escalation_set.yaml?

The decision is the reply model's alone, so only that model runs: no smart model, no remote calls, no cost. Each
case is sent exactly as MOUTH would send it with a smart model configured (mouth.reply_prefix: the escalation
system prompt + few-shot turns, then the case's history, then the user turn; tools with ask_expert), at the
model's configured temperature, `--samples` times. Needs the GPU free (stop run.sh). Prints routes and rates;
the set is synthetic, so misrouted cases are listed with their text.

  .venv/bin/python scripts/eval_escalation.py [--plugin qwen3-4b-2507] [--samples 3] [--split heldout]
      [--prompts prompts.yaml,/path/candidate.yaml]      several prompt files are compared on one model load
"""
import sys, json, argparse, logging, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.WARNING)
import yaml
from app.config import load_config, repo_path
from app.engines import EngineLoop
from app.plugins import plugin_class
from app.plugins.base import PluginContext
from app.prompts import load_prompts
from app.tools import build_tools, EXPERT
from app.mouth import reply_prefix, turns_messages

ROUTES = ("local", "tool", "expert")
HERE = Path(__file__).resolve().parent


def load_set(path):
    cases = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cases"]
    seen = collections.Counter()
    for c in cases:                                  # held out: every third case of each route, in file order
        seen[c["route"]] += 1
        c["split"] = "heldout" if seen[c["route"]] % 3 == 0 else "tune"
        assert c["route"] in ROUTES, c["id"]
    return cases


def route_of(events):
    """The route a reply stream took: ask_expert called -> expert; any other tool -> tool; else local. A
    <tool_call> that didn't parse -> broken."""
    started = calls = None
    for ev in events:
        if ev[0] == "tool_start":
            started = ev[1]
        elif ev[0] == "tool_calls":
            calls = ev[1]
    names = [c["name"] for c in calls or []]
    if EXPERT in names:
        return "expert"
    if names:
        return "tool"
    return "broken" if started else "local"


def evaluate(llm, p, cases, samples, max_new, workers):
    tools = build_tools(p, expert=True)
    prefix = reply_prefix(p, True)
    jobs = [(c, k) for c in cases for k in range(samples)]

    def run(job):
        c, _ = job
        msgs = prefix + turns_messages(p, c.get("history", []), f"h{c['id']}_") + \
            [{"role": "user", "content": c["user"]}]
        return c, route_of(llm.stream(msgs, tools=tools, max_new=max_new))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, jobs))


def report(results, title):
    print(f"\n=== {title} ===")
    by = collections.defaultdict(collections.Counter)
    for c, got in results:
        by[c["route"]][got] += 1
    cols = ROUTES + ("broken",)
    print(f"{'expected / got':<16}" + "".join(f"{x:>8}" for x in cols) + f"{'n':>6}")
    for r in ROUTES:
        n = sum(by[r].values())
        if n:
            print(f"{r:<16}" + "".join(f"{by[r][x]:>8}" for x in cols) + f"{n:>6}")

    def rate(exp, got):
        n = sum(by[exp].values())
        return (by[exp][got] / n if n else 0.0), n
    missed, n_e = rate("expert", "local")
    unneeded, n_l = rate("local", "expert")
    print(f"missed escalations (expert -> local): {100 * missed:5.1f}%  (target <= 10%, n={n_e})")
    print(f"unneeded escalations (local -> expert): {100 * unneeded:5.1f}%  (target <= 15%, n={n_l})")
    print(f"expert -> tool {100 * rate('expert', 'tool')[0]:.1f}%, tool -> expert {100 * rate('tool', 'expert')[0]:.1f}%, "
          f"tool -> local {100 * rate('tool', 'local')[0]:.1f}%")
    return {"missed": missed, "unneeded": unneeded, "matrix": {r: dict(by[r]) for r in ROUTES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plugin", default=None, help="override llm.reply.plugin (e.g. qwen3-4b-2507)")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt files to compare (default: config's)")
    ap.add_argument("--set", default=str(HERE / "escalation_set.yaml"))
    ap.add_argument("--split", choices=("all", "tune", "heldout"), default="all")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--json", default=None, help="write the rates and per-case routes here")
    a = ap.parse_args()
    cfg = load_config(a.config)
    block = dict(cfg["llm"]["reply"])
    if a.plugin:
        block["plugin"] = a.plugin; block.pop("model", None)
    name = block.get("name") or block["plugin"]
    cases = [c for c in load_set(a.set) if a.split == "all" or c["split"] == a.split]
    llm = plugin_class("llm", block["plugin"])(block, PluginContext(EngineLoop(), cfg, name))
    out = {}
    for pf in (a.prompts.split(",") if a.prompts else [cfg["mouth"].get("prompts") or "prompts.yaml"]):
        cfg["mouth"]["prompts"] = str(repo_path(pf))
        p = load_prompts(cfg, [name])[name]
        results = evaluate(llm, p, cases, a.samples, int(cfg["mouth"].get("max_gen", 192)),
                           int(block.get("max_num_seqs", 8)))
        title = f"{name}, prompts {Path(pf).name}, split {a.split}, {len(cases)} cases x {a.samples}"
        out[pf] = report(results, title)
        for split in ("tune", "heldout") if a.split == "all" else ():
            out[pf][split] = report([r for r in results if r[0]["split"] == split], f"{title}: {split} only")
        wrong = collections.defaultdict(list)
        for c, got in results:
            if got != c["route"]:
                wrong[c["id"]].append(got)
        out[pf]["wrong"] = wrong
        print("\nmisrouted (id, expected, got per sample):")
        for c in cases:
            if c["id"] in wrong:
                print(f"  {c['id']} [{c['split']}] {c['route']} -> {','.join(wrong[c['id']])}: {c['user']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    llm.shutdown()


if __name__ == "__main__":
    main()
