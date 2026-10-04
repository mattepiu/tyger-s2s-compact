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
}
# plain text, sent as written
TEXTS = [("system", "voice_tools"), ("tool_leadin", "system"), ("still_looking", "system"),
         ("still_looking", "said_none"), ("backchannel", "system"), ("backchannel", "user_empty"),
         ("markers", "interrupted"), ("markers", "empty_reply"), ("warmup", "user"),
         ("spoken", "error"), ("spoken", "refusal"),
         ("escalation", "system"), ("escalation", "tool_result"), ("expert_leadin", "system"),
         ("expert_still", "system"), ("expert_still", "said_none"), ("smart", "system"), ("smart", "limit")]
SHOTS = ["tool_leadin", "still_looking", "backchannel", "expert_leadin", "expert_still"]


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
        from app.tools import ALL_SCHEMAS
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
        tools = self.d.get("tools")
        if not isinstance(tools, dict):
            self._fail("tools", "is missing")
        names = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in ALL_SCHEMAS}
        for name, props in names.items():
            entry = tools.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("description"), str):
                self._fail(f"tools.{name}.description", "is missing")
            for p, desc in (entry.get("params") or {}).items():
                if p not in props or not isinstance(desc, str):
                    self._fail(f"tools.{name}.params.{p}", "is not a text description of a parameter of the tool")
        for name in tools:
            if name not in names:
                self._fail(f"tools.{name}", "is not a tool in app/tools.py")

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

    def turns(self, section) -> list[dict]:
        """Structured few-shot turns (escalation.shots): dicts with user + reply, or user + expert + answer."""
        return [dict(t) for t in self.d[section]["shots"]]

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
