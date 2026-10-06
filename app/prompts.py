"""Prompts: every piece of text MOUTH sends to an LLM lives in prompts.yaml (PLUGINS.md -> "Prompts in one file").

Read once at MOUTH start and validated: required keys present, each template's placeholders exactly the ones the
code passes, shots a list of [user, assistant] pairs, tool descriptions only for tools and parameters that exist.
An invalid file stops start-up with the key named. Per-instance overrides: prompts/<instance>.yaml, deep-merged
over the base file (dicts key by key; lists such as `shots` and scalars replaced whole) for the requests that LLM
instance serves. Not reloaded live: restart MOUTH after editing.
"""
from __future__ import annotations
import copy
from pathlib import Path
from string import Template

# templates: (section, key) -> the placeholders the code passes
TEMPLATES = {
    ("tool_leadin", "user"): {"transcript"},
    ("still_looking", "user"): {"transcript", "said"},
    ("backchannel", "no_repeat"): {"acks"},
    ("backchannel", "user_short"): {"partial"},
    ("backchannel", "user"): {"partial"},
    ("expert_leadin", "user"): {"transcript"},
    ("expert_still", "user"): {"transcript", "said"},
    ("smart", "user"): {"transcript", "question"},
    ("smart", "results"): {"results"},
    ("language", "tool_note"): {"language"},
    ("language", "filler"): {"language"},
    ("language", "status_note"): {"language"},
    ("tasks", "user"): {"goal", "language", "transcript"},
    ("tasks", "status_block"): {"tasks"},
    ("tasks", "status_running"): {"id", "goal", "seconds"},
    ("tasks", "status_done"): {"id", "goal", "summary"},
    ("tasks", "status_announced"): {"id", "goal", "summary"},
    ("tasks", "status_partial"): {"id", "goal", "summary"},
    ("tasks", "status_failed"): {"id", "goal"},
    ("tasks", "status_cancelled"): {"id", "goal"},
    ("tasks", "result_note"): {"id", "goal", "result"},
    ("tasks", "failed_note"): {"id", "goal"},
    ("tasks", "expert_context"): {"results"},
    ("task_failed", "user"): {"goal"},
}
# plain text, sent as written
TEXTS = [("system", "voice_tools"), ("tool_leadin", "system"), ("still_looking", "system"),
         ("still_looking", "said_none"), ("backchannel", "system"), ("backchannel", "user_empty"),
         ("markers", "interrupted"), ("markers", "empty_reply"), ("warmup", "user"),
         ("spoken", "error"), ("spoken", "refusal"),
         ("escalation", "system"), ("escalation", "tool_result"), ("expert_leadin", "system"),
         ("expert_still", "system"), ("expert_still", "said_none"), ("smart", "system"), ("smart", "limit"),
         ("language", "tool_note_any"), ("language", "unknown"), ("tasks", "reply_system"), ("tasks", "system"),
         ("tasks", "limit"), ("tasks", "cap_concurrent"), ("tasks", "cap_session"), ("tasks", "cap_global"),
         ("tasks", "unknown_task"), ("task_failed", "system")]
SHOTS = ["tool_leadin", "still_looking", "backchannel", "expert_leadin", "expert_still", "task_failed"]


class PromptError(ValueError):
    pass


def deep_merge(base, over):
    """`over` on top of `base`: dicts merge key by key; anything else (lists, strings) is replaced whole."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


class Prompts:
    """One LLM instance's prompts (the base file plus its override, if any), validated."""
    def __init__(self, data: dict, source: str = "prompts.yaml"):
        self.d = data; self.source = source
        self._validate()

    def _fail(self, key, what):
        raise PromptError(f"{self.source}: `{key}` {what}")

    def _get(self, section, key=None):
        node = self.d.get(section)
        if key is None:
            return node
        return node.get(key) if isinstance(node, dict) else None

    def _validate(self):
        from app.tools import CANCEL, START, TOOLS, names
        if self.d.get("version") != 1:
            self._fail("version", "must be 1")
        for sec, key in TEXTS:
            if not isinstance(self._get(sec, key), str):
                self._fail(f"{sec}.{key}", "is missing or not text")
        for (sec, key), want in TEMPLATES.items():
            t = self._get(sec, key)
            if not isinstance(t, str):
                self._fail(f"{sec}.{key}", "is missing or not text")
            tpl = Template(t)
            if not tpl.is_valid():
                self._fail(f"{sec}.{key}", "has a bad placeholder (write $$ for a literal $)")
            got = set(tpl.get_identifiers())
            if got != want:
                self._fail(f"{sec}.{key}", f"must use exactly {sorted('$' + w for w in want)}, "
                                           f"uses {sorted('$' + g for g in got)}")
        for sec in SHOTS:
            shots = self._get(sec, "shots")
            if not isinstance(shots, list) or not all(
                    isinstance(p, list) and len(p) == 2 and all(isinstance(s, str) for s in p) for p in shots):
                self._fail(f"{sec}.shots", "must be a list of [user, assistant] text pairs")
        shots = self._get("escalation", "shots")
        if not isinstance(shots, list) or not all(
                isinstance(t, dict) and isinstance(t.get("user"), str) and
                (set(t) == {"user", "reply"} and isinstance(t["reply"], str) or
                 set(t) == {"user", "expert", "answer"} and isinstance(t["expert"], str) and isinstance(t["answer"], str))
                for t in shots):
            self._fail("escalation.shots", "must be a list of {user, reply} or {user, expert, answer} turns (text)")
        lookups = set(names("base"))
        ts = self.d.get("reply_shots")
        if not isinstance(ts, list) or not all(
                isinstance(t, dict) and isinstance(t.get("user"), str) and (
                    set(t) == {"user", "reply"} and isinstance(t["reply"], str) or
                    set(t) == {"user", "call", "result", "answer"} and isinstance(t["answer"], str)
                    and isinstance(t["call"], dict) and t["call"].get("name") in lookups
                    and isinstance(t["call"].get("arguments"), dict)) for t in ts):
            self._fail("reply_shots", "must be a list of {user, reply} or {user, call: {name, arguments}, result, "
                                      "answer} turns (calls to the lookup / action tools of app/tools/)")
        ts = self._get("tasks", "shots")
        if not isinstance(ts, list) or not all(
                isinstance(t, dict) and isinstance(t.get("user"), str) and (
                    set(t) == {"user", "reply"} and isinstance(t["reply"], str) or
                    set(t) == {"user", "call", "result", "answer"} and isinstance(t["answer"], str)
                    and isinstance(t["call"], dict) and t["call"].get("name") in (START, CANCEL)
                    and isinstance(t["call"].get("arguments"), dict)) for t in ts):
            self._fail("tasks.shots", "must be a list of {user, reply} or {user, call: {name: start_task|cancel_task, "
                                      "arguments}, result, answer} turns")
        langs = self._get("language", "names")
        if not isinstance(langs, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in langs.items()):
            self._fail("language.names", "must map language codes to names (text)")
        tools = self.d.get("tools")
        if not isinstance(tools, dict):
            self._fail("tools", "is missing")
        props_of = {n: t.parameters["properties"] for n, t in TOOLS.items()}
        for name, props in props_of.items():
            entry = tools.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("description"), str):
                self._fail(f"tools.{name}.description", "is missing")
            for p, desc in (entry.get("params") or {}).items():
                if p not in props or not isinstance(desc, str):
                    self._fail(f"tools.{name}.params.{p}", "is not a text description of a parameter of the tool")
        for name in tools:
            if name not in props_of:
                self._fail(f"tools.{name}", "is not a tool in app/tools/")

    def text(self, section, key) -> str:
        return self.d[section][key]

    def shots(self, section) -> list[tuple[str, str]]:
        return [tuple(p) for p in self.d[section]["shots"]]

    def fill(self, section, key, **values) -> str:
        """A template with the code's values; a missing or extra value fails loudly."""
        want = TEMPLATES[(section, key)]
        if set(values) != want:
            raise PromptError(f"`{section}.{key}` takes {sorted(want)}, got {sorted(values)}")
        return Template(self.d[section][key]).substitute(values)

    def language_name(self, transcript):
        from app.lang import guess_language
        return self.d["language"]["names"].get(guess_language(transcript) or "")

    def result_note(self, transcript) -> str:
        """The note appended to tool results: names the reply language when app/lang.py can tell it."""
        name = self.language_name(transcript)
        return self.fill("language", "tool_note", language=name) if name else self.text("language", "tool_note_any")

    def filler_hint(self, transcript) -> str:
        """Appended to lead-in / still-looking requests: the language to say it in, or "" if unknown."""
        name = self.language_name(transcript)
        return self.fill("language", "filler", language=name) if name else ""

    def turns(self, section) -> list[dict]:
        """Structured few-shot turns: reply_shots (user + reply, or user + call + result + answer),
        escalation.shots (user + expert + answer, or user + reply) or tasks.shots (user + call + result + answer)."""
        node = self.d[section]
        return [dict(t) for t in (node if isinstance(node, list) else node["shots"])]

    def language_or_unknown(self, transcript) -> str:
        """The user's language by name, or language.unknown ("the same language they spoke in")."""
        return self.language_name(transcript) or self.text("language", "unknown")

    def tool_text(self) -> dict:
        return self.d["tools"]


def _read(path: Path) -> dict:
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PromptError(f"prompts file not found: {path}") from None
    if not isinstance(data, dict):
        raise PromptError(f"{path}: not a mapping")
    return data


def load_prompts(cfg: dict, names) -> dict[str, Prompts]:
    """{instance name: Prompts}: the base file (mouth.prompts, default prompts.yaml), with
    <mouth.prompt_overrides_dir>/<name>.yaml merged over it where one exists."""
    from app.config import repo_path
    m = cfg.get("mouth") or {}
    base_path = repo_path(m.get("prompts") or "prompts.yaml")
    over_dir = repo_path(m.get("prompt_overrides_dir") or "prompts")
    base = _read(base_path)
    out = {}
    for name in names:
        over = over_dir / f"{name}.yaml"
        if over.is_file():
            out[name] = Prompts(deep_merge(base, _read(over)), f"{base_path.name} + {over.name}")
        else:
            out[name] = Prompts(base, base_path.name)
    return out


def default_prompts() -> Prompts:
    """The repo's prompts.yaml with no overrides (tests and tools)."""
    from app.config import ROOT
    return Prompts(_read(ROOT / "prompts.yaml"))
