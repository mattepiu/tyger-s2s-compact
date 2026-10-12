"""Style probe for the reply prompts (DEVELOPING.md -> "Checking a change"): how the `reply` model talks on synthetic
conversational turns, per prompt file and character. Of the replies answered locally it counts:
  * markup: digits, symbols outside letters and basic punctuation, emoji, parentheticals or action lines;
  * long: more than three sentences;
  * human: a disfluency or a filler word (um, uh, you know, I mean, a repeated word, a false start, trailing off);
and the routes taken (a turn answered with a tool call is counted, not scored). Only the reply model runs, sent each
turn as MOUTH sends it (messages.reply_prefix in the config's mode, with its tools), so there are no hosted calls and
no cost. Needs the GPU free (stop run.sh). The turns are synthetic; `--show` prints some replies to read.

  .venv/bin/python scripts/bench/style.py [--samples 3] [--prompts a.yaml,b.yaml] [--character ori] [--no-expert]
      [--show 8]
"""
import sys, re, argparse, collections, logging, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.WARNING)
from app.config import load_config, repo_path
from app.runtime.event_loop import EngineLoop
from app.plugins import plugin_class
from app.plugins.base import PluginContext
from app.prompts import Characters, load_prompts
from app.tools import build_tools
from app.mouth.messages import reply_prefix

TURNS = [
    "Hey, how's it going?", "What have you been up to today?", "I just got back from a run and I'm exhausted.",
    "I had a really rough day at work.", "Do you like jazz?", "What do you think about working from home?",
    "I'm thinking about learning to play the guitar.", "My cat knocked my coffee off the table this morning.",
    "Is it weird that I talk to my plants?", "What's your favorite season?",
    "I can't decide between pizza and sushi for dinner.", "Tell me something interesting.",
    "How many legs does a spider have?", "What year did the Berlin Wall fall?", "How long should I boil pasta?",
    "How many minutes are in a day?", "What temperature does water boil at?", "How far away is the moon, roughly?",
    "What's the capital of Australia?", "How do I make cold brew coffee?", "Any tips for falling asleep faster?",
    "Why is the sky blue?", "Can you recommend a good book?", "I'm nervous about a job interview tomorrow.",
    "My best friend is moving to another city.", "Do you ever get bored?", "Are you a real person?",
    "What makes a good friend, in your opinion?", "I've been procrastinating all week.",
    "What do you think of pineapple on pizza?", "Say something to cheer me up.", "I finally fixed my bike today.",
    "How many ounces are in a pound?", "What's a good way to start journaling?", "Who wrote Pride and Prejudice?",
    "I think I'm going to adopt a dog.",
]

BASIC = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n.,?!;:'\"-’‘“”—–…")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
ASIDE = re.compile(r"[()\[\]]|\*[^*\n]+\*")                    # a parenthetical, or an *action line*
FILLER = re.compile(r"\b(?:u+m+|u+h+|h+m+|e+rm*|a+h+|you know|i mean|kind of|sort of)\b|\b(?:like|well|oh),", re.I)
REPEAT = re.compile(r"\b(\w+)(?:,|\.\.\.|…|\s*[—-])?\s+\1\b", re.I)    # "I, I", "the the", "it's... it's"
BREAK = re.compile(r"\.\.\.|…|—|\s-\s")              # trailing off, a false start or a revision


def issues(text):
    """{"digits", "symbols", "emoji", "aside"} found in a reply, and the symbol characters."""
    sym = {ch for ch in text if ch not in BASIC and not ch.isdigit() and not EMOJI.match(ch)}
    found = {k for k, hit in (("digits", re.search(r"\d", text)), ("symbols", sym), ("emoji", EMOJI.search(text)),
                              ("aside", ASIDE.search(text))) if hit}
    return found, sym


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"\w", s)]


def human(text):
    return bool(FILLER.search(text) or REPEAT.search(text) or BREAK.search(text))


def run(llm, p, character, expert, tasks, samples, max_new, workers):
    tools = build_tools(p, expert=expert, tasks=tasks)
    prefix = reply_prefix(p, character, expert, tasks)

    def one(turn):
        text, calls = "", []
        for ev in llm.stream(prefix + [{"role": "user", "content": turn}], tools=tools, max_new=max_new):
            if ev[0] == "text":
                text += ev[1]
            elif ev[0] == "tool_calls":
                calls = [c["name"] for c in ev[1]]
        return turn, text.strip(), calls
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, [t for t in TURNS for _ in range(samples)]))


def report(results, title, show):
    print(f"\n=== {title} ===")
    routes = collections.Counter(",".join(c) if c else "local" for _, _, c in results)
    print("routes:", ", ".join(f"{r} {n}" for r, n in routes.most_common()))
    local = [(t, x) for t, x, c in results if not c]
    n = max(1, len(local))
    kinds = collections.Counter(); syms = collections.Counter()
    for _, x in local:
        found, sym = issues(x)
        kinds.update(found); syms.update(sym)
    marked = sum(1 for _, x in local if issues(x)[0])
    long = sum(1 for _, x in local if len(sentences(x)) > 3)
    hum = sum(1 for _, x in local if human(x))
    words = sum(len(x.split()) for _, x in local) / n
    print(f"local replies {len(local)}: markup {marked} ({100 * marked / n:.0f}%; "
          + ", ".join(f"{k} {kinds[k]}" for k in ("digits", "symbols", "emoji", "aside")) + ")"
          + (f" symbols seen {''.join(sorted(syms))!r}" if syms else ""))
    print(f"  over three sentences {long} ({100 * long / n:.0f}%); disfluency or filler {hum} ({100 * hum / n:.0f}%); "
          f"{words:.0f} words on average")
    for t, x in [(t, x) for t, x in local if issues(x)[0]][:show]:
        print(f"  markup ({', '.join(sorted(issues(x)[0]))}) > {t}\n    {x}")
    picks = random.Random(0).sample(local, min(show, len(local)))
    for t, x in picks:
        print(f"  > {t}\n    {x}")
    return {"local": len(local), "markup": marked, "long": long, "human": hum, "routes": dict(routes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None, help="comma-separated prompt files to compare (default: config's)")
    ap.add_argument("--character", default=None, help="default: the config's default character (voices.character)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--no-expert", action="store_true", help="the prompt and tools without a smart model")
    ap.add_argument("--show", type=int, default=8, help="replies to print per prompt file")
    a = ap.parse_args()
    cfg = load_config(a.config)
    block = dict(cfg["llm"]["reply"])
    name = block.get("name") or block["plugin"]
    expert = not a.no_expert
    tasks = expert and bool((cfg["mouth"].get("tasks") or {}).get("enabled"))
    llm = plugin_class("llm", block["plugin"])(block, PluginContext(EngineLoop(), cfg, name))
    for pf in (a.prompts.split(",") if a.prompts else [cfg["mouth"].get("prompts") or "config/prompts.yaml"]):
        cfg["mouth"]["prompts"] = str(repo_path(pf))
        p = load_prompts(cfg)
        character = a.character or Characters(cfg, p).default
        results = run(llm, p, character, expert, tasks, a.samples, int(cfg["mouth"].get("max_gen", 192)),
                      int(block.get("max_num_seqs", 8)))
        mode = "tasks" if tasks else "expert" if expert else "no expert"
        report(results, f"{name}, prompts {Path(pf).name}, character {character} ({mode}), {len(TURNS)} turns x "
                        f"{a.samples}", a.show)
    llm.shutdown()


if __name__ == "__main__":
    main()
