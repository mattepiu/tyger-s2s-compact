"""A cheap guess at the language of a transcript: the 3 languages the prompts and tests cover (ru / es / en), from
the script and common words. Used to name the reply language next to tool output, which is mostly English and
otherwise pulls a small model's answer into English (PLUGINS.md step 11). Text only; nothing is logged."""
from __future__ import annotations
import re

ES = set("el la los las de que y en un una es por para con se lo su al del más pero como muy está son te tu ya "
         "hay qué cómo también puedes tienes mañana hoy dónde cuándo cuánto".split())
EN = set("the an is are and of to in you your it i that this for with on be can it's at or not do have was "
         "what how there today tomorrow where when".split())


def guess_language(text: str) -> str | None:
    """"ru", "es", "en", or None if it can't tell (too short, or another language: Spanish or English needs at
    least two signs, so e.g. German "Wie spät ist es?" stays None)."""
    t = (text or "").lower()
    cyr, lat = len(re.findall(r"[а-яё]", t)), len(re.findall(r"[a-zñáéíóúü]", t))
    if not cyr and not lat:
        return None
    if cyr >= lat:
        return "ru"
    words = re.findall(r"[a-zñáéíóúü']+", t)
    es = sum(w in ES for w in words) + 2 * len(re.findall(r"[¿¡ñáéíóú]", t))
    en = sum(w in EN for w in words)
    if max(es, en) < 2 or es == en:
        return None
    return "es" if es > en else "en"
