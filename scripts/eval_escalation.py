"""Escalation routing eval (PLUGINS.md steps 11, 14): does the `reply` model answer locally, call a lookup tool,
call ask_expert, or start a background task, on the labelled synthetic set in scripts/escalation_set.yaml?

Routes: local (no call), tool (any lookup / action, cancel_task included), expert (ask_expert), background
(start_task). A case may list `also` routes that count as right too (planning: answered now by the expert, or
researched as a task). The decision is the reply model's alone, so only that model runs: no smart model, no
remote calls, no cost. Each case is sent exactly as MOUTH would send it (messages.reply_prefix for the mode, then the
case's history, then the user turn, with the case's `status` note in front as MOUTH prepends it), at the model's
configured temperature, `--samples` times. The mode follows config.yaml (tasks on if mouth.tasks.enabled);
--no-tasks / --no-expert evaluate the prompts without them (cases needing the missing route are skipped). Needs the GPU free (stop run.sh). Prints routes and rates;
the set is synthetic, so misrouted cases are listed with their text.

Also scores what the model would say aloud (its text before a call, or its whole local reply): replies of 4+
words in a language other than the case's (script + stopword guess), and narration: words about tools / the
expert, or call markup leaking into speech. For tool-route samples it also runs the answer after the tool
result (phase B), with fixed English stub results (the real reminder / message stubs; a canned answer for
lookups), and scores that answer's speech the same way: English tool output must not pull the reply language.

  .venv/bin/python scripts/eval_escalation.py [--plugin gemma-4-e4b-nvfp4 --opt quantization=null] [--samples 3]
      [--split heldout] [--prompts prompts.yaml,/path/candidate.yaml]   several prompt files: one model load
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
from app.tools import build_tools, execute_tool, EXPERT, START, CANCEL
from app.messages import reply_prefix, turns_messages, round_messages, with_status
from app.lang import guess_language, language_text

ROUTES = ("local", "tool", "expert", "background")
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
        "send_message": {"demo": True, "note": "Messaging isn't connected yet."},
        START: {"task_id": 1, "status": "started"}, CANCEL: {"task_id": 1, "status": "cancelled"}}
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
    """(route, spoken text) of a reply stream: start_task called -> background; ask_expert -> expert; any other
    tool -> tool; else local; a call that didn't parse -> broken. The text is what MOUTH would speak (before any
    call)."""
    started = calls = None; text = ""
    for ev in events:
        if ev[0] == "text":
            text += ev[1]
        elif ev[0] == "tool_start":
            started = ev[1]
        elif ev[0] == "tool_calls":
            calls = ev[1]
    names = [c["name"] for c in calls or []]
    if START in names:
        return "background", text
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


def right(case, got):
    return got == case["route"] or got in case.get("also", [])


def evaluate(llm, p, cases, samples, max_new, workers, expert=True, tasks=False):
    tools = build_tools(p, expert=expert, tasks=tasks)
    prefix = reply_prefix(p, expert, tasks)
    jobs = [(c, k) for c in cases for k in range(samples)]

    def run(job):
        c, _ = job
        lang = language_text(c["user"], [t["user"] for t in c.get("history", [])])    # as MOUTH guesses it
        user = with_status(p, c.get("status"), c["user"], lang)      # MOUTH's status note, as MOUTH builds it
        msgs = prefix + turns_messages(p, c.get("history", []), f"h{c['id']}_") + [{"role": "user", "content": user}]
        evs = list(llm.stream(msgs, tools=tools, max_new=max_new))
        route, text = route_of(evs)
        phase_b = None
        if route in ("tool", "background"):       # the answer after the (stub) tool result, as MOUTH asks for it
            calls = next(e[1] for e in evs if e[0] == "tool_calls")
            results = [STUB.get(x["name"]) or LOOKUP.get(x["name"]) or execute_tool(x["name"], x.get("arguments", {}))
                       for x in calls]                         # get_current_time runs locally
            phase_b, _ = collect(llm.stream(msgs + round_messages(text, calls, None, results,
                                                                  note=p.result_note(lang)), max_new=max_new))
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
        n = sum(sum(by[e].values()) for e in exp)
        return (sum(by[e][g] for e in exp for g in got) / n if n else 0.0), n
    up = ("expert", "background")
    missed, n_e = rate(up, ("local",))
    unneeded, n_l = rate(("local",), up)
    ok = sum(right(c, got) for c, got, *_ in results)
    print(f"right (incl. `also`): {ok}/{len(results)} = {100 * ok / max(1, len(results)):.1f}%")
    print(f"missed escalations (expert / background -> local): {100 * missed:5.1f}%  (target <= 10%, n={n_e})")
    print(f"unneeded escalations (local -> expert / background): {100 * unneeded:5.1f}%  (target <= 15%, n={n_l})")
    eb = sum(1 for c, got, *_ in results if c["route"] == "expert" and got == "background")
    be = sum(1 for c, got, *_ in results if c["route"] == "background" and got == "expert")
    eb_ok = sum(1 for c, got, *_ in results if c["route"] == "expert" and got == "background" and right(c, got))
    be_ok = sum(1 for c, got, *_ in results if c["route"] == "background" and got == "expert" and right(c, got))
    print(f"expert -> background {eb} ({eb_ok} allowed by `also`), background -> expert {be} ({be_ok} allowed)")
    print(f"expert -> tool {100 * rate(('expert',), ('tool',))[0]:.1f}%, tool -> expert / background "
          f"{100 * rate(('tool',), up)[0]:.1f}%, tool -> local {100 * rate(('tool',), ('local',))[0]:.1f}%")
    issues = collections.Counter(i for c, _, text, _ in results for i in speech_issues(c, text))
    spoken = sum(1 for _, _, text, _ in results if text.strip())
    print(f"speech: wrong language {issues['language']}, narration / markup {issues['narration']} "
          f"(of {spoken} samples that spoke)")
    pb = [(c, b) for c, _, _, b in results if b is not None]
    pb_issues = collections.Counter(i for c, b in pb for i in speech_issues(c, b))
    print(f"after the tool / task result: wrong language {pb_issues['language']}, narration / markup "
          f"{pb_issues['narration']}, empty {sum(1 for _, b in pb if not b.strip())} (of {len(pb)})")
    return {"missed": missed, "unneeded": unneeded, "right": ok, "n": len(results), "expert_to_bg": eb,
            "bg_to_expert": be, "matrix": {r: dict(by[r]) for r in ROUTES},
            "language": issues["language"], "narration": issues["narration"], "spoke": spoken,
            "pb_language": pb_issues["language"], "pb_narration": pb_issues["narration"], "pb_n": len(pb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plugin", default=None, help="override llm.reply.plugin (e.g. gemma-4-e4b-nvfp4)")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt files to compare (default: config's)")
    ap.add_argument("--set", default=str(HERE / "escalation_set.yaml"))
    ap.add_argument("--split", choices=("all", "tune", "heldout", "fresh"), default="all")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--json", default=None, help="write the rates and per-case routes here")
    ap.add_argument("--no-expert", action="store_true", help="the prompt and tools without a smart model (no "
                    "ask_expert); expert cases are skipped")
    ap.add_argument("--no-tasks", action="store_true", help="the prompt and tools without background tasks, "
                    "even if config.yaml has them on; background cases are skipped")
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
    expert = not a.no_expert
    tasks = expert and not a.no_tasks and bool((cfg["mouth"].get("tasks") or {}).get("enabled"))
    cases = [c for c in load_set(a.set) if (a.split == "all" or c["split"] == a.split)
             and not (not expert and c["route"] == "expert") and not (not tasks and c["route"] == "background")
             and not (not tasks and (c.get("status") or any("call" in t and t["call"]["name"] in (START, CANCEL)
                                                              for t in c.get("history", []))))]
    llm = plugin_class("llm", block["plugin"])(block, PluginContext(EngineLoop(), cfg, name))
    out = {}
    for pf in (a.prompts.split(",") if a.prompts else [cfg["mouth"].get("prompts") or "prompts.yaml"]):
        cfg["mouth"]["prompts"] = str(repo_path(pf))
        p = load_prompts(cfg, [name])[name]
        results = evaluate(llm, p, cases, a.samples, int(cfg["mouth"].get("max_gen", 192)),
                           int(block.get("max_num_seqs", 8)), expert=expert, tasks=tasks)
        mode = "tasks" if tasks else "expert" if expert else "no expert"
        title = f"{name}, prompts {Path(pf).name} ({mode}), split {a.split}, {len(cases)} cases x {a.samples}"
        out[pf] = report(results, title)
        for split in ("tune", "heldout", "fresh") if a.split == "all" else ():
            out[pf][split] = report([r for r in results if r[0]["split"] == split], f"{title}: {split} only")
        wrong = collections.defaultdict(list); said = collections.defaultdict(list)
        for c, got, text, pbtext in results:
            for issue in speech_issues(c, pbtext or ""):
                said[c["id"]].append(f"after tool, {issue}: {pbtext.strip()[:90]!r}")
            if not right(c, got):
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
