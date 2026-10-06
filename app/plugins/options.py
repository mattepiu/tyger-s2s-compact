"""A plugin's config block as a typed, frozen dataclass: its `Options`.

Each plugin class declares `Options`, a subclass of PluginOptions whose fields are the keys its config block may set
(with their types and defaults; a field without a default must be set). `from_config` builds it from the block and
rejects what doesn't fit, naming the key: a renamed key (`RENAMED`: the message names the new one), an unknown key
(a typo is an error, never silently ignored), a missing required key, a value of the wrong type. config.py runs it
for every block when the config loads, so a bad block fails at startup, before any model loads. `plugin:` and
`name:` belong to the config shape, not to the plugin.

Scalars are coerced the way YAML leaves them: an int where a float is declared, a number where a string is (an
all-digit revision hash). A bool must be a bool.
"""
from __future__ import annotations
import dataclasses, difflib, types, typing
from dataclasses import dataclass
from typing import ClassVar

SHAPE_KEYS = ("plugin", "name")              # every block's; config.py reads them


@dataclass(frozen=True, kw_only=True)
class PluginOptions:
    RENAMED: ClassVar[dict] = {}             # old key -> new key: an old key is rejected with the new one named

    @classmethod
    def from_config(cls, block: dict | None, ignore=()) -> PluginOptions:
        """The options in `block` (`ignore`: keys someone else reads, e.g. an EARS plugin's thresholds). ValueError
        naming the key if one is unknown, missing or of the wrong type."""
        block = {k: v for k, v in (block or {}).items() if k not in SHAPE_KEYS and k not in ignore}
        fields = {f.name: f for f in dataclasses.fields(cls)}
        for key in block:
            if key in cls.RENAMED:
                raise ValueError(f"`{key}` is now `{cls.RENAMED[key]}`")
            if key not in fields:
                near = difflib.get_close_matches(key, fields, n=1)
                raise ValueError(f"unknown option `{key}`" + (f" (did you mean `{near[0]}`?)" if near else "")
                                 + f"; options: {', '.join(sorted(fields)) or 'none'}")
        missing = [n for n, f in fields.items() if n not in block and f.default is dataclasses.MISSING
                   and f.default_factory is dataclasses.MISSING]
        if missing:
            raise ValueError(f"`{missing[0]}` is not set")
        hints = typing.get_type_hints(cls)
        return cls(**{k: _coerce(k, v, hints[k]) for k, v in block.items()})


def _coerce(key, value, hint):
    """`value` as the declared type `hint` (simple types and `X | None`); other hints pass the value through."""
    args = typing.get_args(hint) if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union \
        else (hint,)
    if value is None:
        if type(None) in args:
            return None
        raise ValueError(f"`{key}` must be set (got null)")
    kinds = [a for a in args if a is not type(None)]
    if len(kinds) != 1:
        return value
    kind = typing.get_origin(kinds[0]) or kinds[0]
    if kind is bool:
        ok = isinstance(value, bool)
    elif kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind is str:
        ok = isinstance(value, (str, int, float)) and not isinstance(value, bool)
    elif kind in (dict, list, tuple):
        ok = isinstance(value, (list, tuple)) if kind in (list, tuple) else isinstance(value, dict)
    else:
        return value
    if not ok:
        raise ValueError(f"`{key}` must be a {kind.__name__} (got {value!r})")
    return kind(value)
