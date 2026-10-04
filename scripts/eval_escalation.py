"""Escalation routing eval (PLUGINS.md step 11): does the `reply` model answer locally, call a lookup tool, or
call ask_expert, on the labelled synthetic set in scripts/escalation_set.yaml?

The decision is the reply model's alone, so only that model runs: no smart model, no remote calls, no cost. Each
case is sent exactly as MOUTH would send it with a smart model configured (mouth.reply_prefix: the escalation
system prompt + few-shot turns, then the case's history, then the user turn; tools with ask_expert), at the
model's configured temperature, `--samples` times. Needs the GPU free (stop run.sh). Prints routes and rates;
the set is synthetic, so misrouted cases are listed with their text.

Also scores what the model would say aloud (its text before a call, or its whole local reply): replies of 4+
words in a language other than the case's (script + stopword guess), and narration: words about tools / the
expert, or call markup leaking into speech. For tool-route samples it also runs the answer after the tool
result (phase B), with fixed English stub results (the real reminder / message stubs; a canned answer for
lookups), and scores that answer's speech the same way: English tool output must not pull the reply language.

  .venv/bin/python scripts/eval_escalation.py [--plugin qwen3-4b-2507] [--samples 3] [--split heldout]
      [--prompts prompts.yaml,/path/candidate.yaml]      several prompt files are compared on one model load
"""
import sys, re, json, argparse, logging, collections
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
from app.tools import build_tools, execute_tool, EXPERT
from app.mouth import reply_prefix, turns_messages, round_messages
from app.lang import guess_language

ROUTES = ("local", "tool", "expert")
HERE = Path(__file__).resolve().parent


def load_set(path):
    cases = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cases"]
    seen = collections.Counter()
    for c in cases:                # held out: every third case of each route, in file order; `fresh` as marked
        assert c["route"] in ROUTES, c["id"]
        if "split" in c:
            continue
        seen[c["route"]] += 1
        c["split"] = "heldout" if seen[c["route"]] % 3 == 0 else "tune"
    return cases


STUB = {"set_reminder": {"demo": True, "note": "Reminders aren't connected yet."},
        "send_message": {"demo": True, "note": "Messaging isn't connected yet."}}
LOOKUP = {"get_weather": {"answer": "Partly cloudy, 14 degrees Celsius, light wind from the north-west; no rain "
                                    "expected today."},
          "web_search": {"answer": "Recent sources give no single clear answer; reports differ."}}   # fits any query


def collect(events):
    text, calls = "", []
    for ev in events:
        if ev[0] == "text":
            text += ev[1]
        elif ev[0] == "tool_calls":
            calls = ev[1]
    return text, calls


def route_of(events):
    """(route, spoken text) of a reply stream: ask_expert called -> expert; any other tool -> tool; else local;
    a call that didn't parse -> broken. The text is what MOUTH would speak (before any call)."""
    started = calls = None; text = ""
    for ev in events:
        if ev[0] == "text":
            text += ev[1]
        elif ev[0] == "tool_start":
            started = ev[1]
        elif ev[0] == "tool_calls":
            calls = ev[1]
    names = [c["name"] for c in calls or []]
    if EXPERT in names:
        return "expert", text
    if names:
        return "tool", text
    return ("broken" if started else "local"), text


NARRATE = re.compile(r"\b(ask_expert|tools?|experts?|инструмент\w*|эксперт\w*|herramientas?|expert[oa]s?)\b|<\||\|>"
                     r"|call:|<tool_call|</?function", re.I)


def speech_issues(case, text):
    """[] or the problems with what would be spoken: 'language', 'narration'."""
    out = []
    if NARRATE.search(text):
        out.append("narration")
    if len(text.split()) >= 4:
        got = guess_language(text)
        if got and got != case["lang"]:
            out.append("language")
    return out


def evaluate(llm, p, cases, samples, max_new, workers, expert=True):
    tools = build_tools(p, expert=expert)
    prefix = reply_prefix(p, expert)
    jobs = [(c, k) for c in cases for k in range(samples)]

    def run(job):
        c, _ = job
        msgs = prefix + turns_messages(p, c.get("history", []), f"h{c['id']}_") + \
            [{"role": "user", "content": c["user"]}]
        evs = list(llm.stream(msgs, tools=tools, max_new=max_new))
        route, text = route_of(evs)
        phase_b = None
        if route == "tool":                       # the answer after the (stub) tool result, as MOUTH asks for it
            calls = next(e[1] for e in evs if e[0] == "tool_calls")
            results = [STUB.get(x["name"]) or LOOKUP.get(x["name"]) or execute_tool(x["name"], x.get("arguments", {}))
                       for x in calls]                         # get_current_time runs locally
            phase_b, _ = collect(llm.stream(msgs + round_messages(text, calls, None, results,
                                                                  note=p.result_note(c["user"])), max_new=max_new))
        return c, route, text, phase_b
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, jobs))


def report(results, title):
    print(f"\n=== {title} ===")
    by = collections.defaultdict(collections.Counter)
    for c, got, *_ in results:
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
    issues = collections.Counter(i for c, _, text, _ in results for i in speech_issues(c, text))
    spoken = sum(1 for _, _, text, _ in results if text.strip())
    print(f"speech: wrong language {issues['language']}, narration / markup {issues['narration']} "
          f"(of {spoken} samples that spoke)")
    pb = [(c, b) for c, _, _, b in results if b is not None]
    pb_issues = collections.Counter(i for c, b in pb for i in speech_issues(c, b))
    print(f"after the tool result: wrong language {pb_issues['language']}, narration / markup "
          f"{pb_issues['narration']}, empty {sum(1 for _, b in pb if not b.strip())} (of {len(pb)})")
    return {"missed": missed, "unneeded": unneeded, "matrix": {r: dict(by[r]) for r in ROUTES},
            "language": issues["language"], "narration": issues["narration"], "spoke": spoken,
            "pb_language": pb_issues["language"], "pb_narration": pb_issues["narration"], "pb_n": len(pb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plugin", default=None, help="override llm.reply.plugin (e.g. qwen3-4b-2507)")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt files to compare (default: config's)")
    ap.add_argument("--set", default=str(HERE / "escalation_set.yaml"))
    ap.add_argument("--split", choices=("all", "tune", "heldout", "fresh"), default="all")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--json", default=None, help="write the rates and per-case routes here")
    ap.add_argument("--no-expert", action="store_true", help="the prompt and tools without a smart model (no "
                    "ask_expert); expert cases are skipped")
    ap.add_argument("--opt", action="append", default=[], help="override a reply-block option, e.g. "
                    "gpu_memory_utilization=0.8 (YAML value)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    block = dict(cfg["llm"]["reply"])
    if a.plugin:
        block["plugin"] = a.plugin; block.pop("model", None)
    for o in a.opt:
        k, v = o.split("=", 1); block[k] = yaml.safe_load(v)
    name = block.get("name") or block["plugin"]
    cases = [c for c in load_set(a.set) if (a.split == "all" or c["split"] == a.split)
             and not (a.no_expert and c["route"] == "expert")]
    llm = plugin_class("llm", block["plugin"])(block, PluginContext(EngineLoop(), cfg, name))
    out = {}
    for pf in (a.prompts.split(",") if a.prompts else [cfg["mouth"].get("prompts") or "prompts.yaml"]):
        cfg["mouth"]["prompts"] = str(repo_path(pf))
        p = load_prompts(cfg, [name])[name]
        results = evaluate(llm, p, cases, a.samples, int(cfg["mouth"].get("max_gen", 192)),
                           int(block.get("max_num_seqs", 8)), expert=not a.no_expert)
        title = f"{name}, prompts {Path(pf).name}{' (no expert)' if a.no_expert else ''}, split {a.split}, " \
            f"{len(cases)} cases x {a.samples}"
        out[pf] = report(results, title)
        for split in ("tune", "heldout", "fresh") if a.split == "all" else ():
            out[pf][split] = report([r for r in results if r[0]["split"] == split], f"{title}: {split} only")
        wrong = collections.defaultdict(list); said = collections.defaultdict(list)
        for c, got, text, pbtext in results:
            for issue in speech_issues(c, pbtext or ""):
                said[c["id"]].append(f"after tool, {issue}: {pbtext.strip()[:90]!r}")
            if got != c["route"]:
                wrong[c["id"]].append(got)
                if text.strip() and len(said[c["id"] + ":misrouted"]) < 2:
                    said[c["id"] + ":misrouted"].append(text.strip()[:90])
            for issue in speech_issues(c, text):
                said[c["id"]].append(f"{issue}: {text.strip()[:90]!r}")
        out[pf]["wrong"] = wrong; out[pf]["speech"] = said
        print("\nmisrouted (id, expected, got per sample):")
        for c in cases:
            if c["id"] in wrong:
                print(f"  {c['id']} [{c['split']}] {c['route']} -> {','.join(wrong[c['id']])}: {c['user']}")
                for t in said.get(c["id"] + ":misrouted", []):
                    print(f"      said: {t!r}")
        print("\nspeech issues (id: issue: what it would say):")
        for c in cases:
            for line in said.get(c["id"], [])[:2]:  # (misrouted samples' text is listed above)
                print(f"  {c['id']} [{c['lang']}]: {line}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    llm.shutdown()


if __name__ == "__main__":
    main()
