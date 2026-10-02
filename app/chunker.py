"""Streaming sentence/clause chunker + TTS-safe prompt + sanitizer (copied/adapted from
codebase/exp-12/server/mouth_cbx_server.py). As the LLM streams tokens, cut the reply at sentence/clause
boundaries (capped length) and TTS each chunk as soon as it closes, so playback starts on chunk 1 while
the rest render. MOSS-TTS-Nano has no paralinguistic tags, so (unlike the Chatterbox MOUTH) the prompt does
NOT invite [laugh]/[sigh] and the sanitizer strips all bracketed tags."""
import re

# TTS-safe system prompt: the reply is read aloud, so plain spoken words only.
TTS_SYS = ("You are a helpful voice assistant talking with the user by voice. Reply briefly and naturally, "
           "the way you would in spoken conversation. Your reply will be read aloud by a text-to-speech "
           "voice, so use plain spoken words only: no markdown or formatting, no asterisks, bullet points, "
           "headings, numbered lists, or code, no emoji, and no special symbols. Use only ordinary sentence "
           "punctuation (period, comma, question mark, apostrophe).")

# Sanitizer: keep letters of ANY script (Cyrillic, CJK, Latin, ...) + digits + basic speakable punctuation,
# strip markdown/symbols/emoji. \w is Unicode-aware in Python 3, so Russian etc. survive for multilingual TTS
# (the old [A-Za-z] whitelist silently dropped all non-Latin text before it reached the TTS).
_KEEP = re.compile(r"[^\w\s.,!?;:'\"()\-]", re.UNICODE)
# Clause dashes -> comma (a pause cue). "Las-Palmas" (unspaced hyphen) and ranges like "5 - 10" / "5–10"
# (digit on both sides) are left alone.
_DASH = re.compile(r"\s*[—–]\s*|\s+-(?:\s+|$)")
def _dash_to_comma(m):
    left, right = m.string[:m.start()].rstrip(), m.string[m.end():].lstrip()
    if left[-1:].isdigit() and right[:1].isdigit():
        return "-"                                  # range "5 – 10" -> "5-10" (en dash isn't kept below)
    return ", "
def say_clean(text: str) -> str:
    s = _DASH.sub(_dash_to_comma, text)
    s = _KEEP.sub(" ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return re.sub(r",(?=[,.!?;:])", "", s).strip().lstrip(",").strip()

# ---- streaming chunker -------------------------------------------------------------------------
# The LLM text arrives token by token, so a punctuation mark only counts as a boundary once the NEXT char
# is known to be whitespace (after any closing quotes/brackets). That keeps in-word punctuation intact --
# "0.50$", "3,000", "1:30", "example.com" -- and never cuts a buffer that merely ends in "0." because
# "50" hasn't streamed yet. The end-of-reply tail is flushed by the caller (Mouth._flush_tail).
# ASCII hyphen is a boundary only when spaced (" - "), so "Las-Palmas" / "well-known" never split.
# Full-width CJK punctuation is always a boundary (CJK text has no spaces after it).
_STRONG = ".!?…"
_WEAK = ",;:—–"
_CLOSERS = '"\')]»”’'
_STRONG_FW = "。！？"
_WEAK_FW = "，；：、"
# A "." after these (lower-cased, without the dot) is not a sentence end; neither is a single-letter
# initial ("J. K. Rowling") or a dotted acronym ("U.S.", "e.g.").
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "approx", "mt", "ft", "no", "fig"}
_ACRONYM = re.compile(r"^(?:\w\.)+$")

MIN_CHARS = 40         # never cut a chunk shorter than this: short replies / short sentences go out whole
                       # (merged with what follows), which reads more naturally than many tiny TTS calls
FIRST_MAX_WORDS = 12   # first chunk: cut at a word boundary if no punctuation shows up by this many words
SENT_MAX = 220         # later chunks are whole sentences; a run-on still flushes at SENT_MAX on a space


def _is_abbrev(core: str, dot: int) -> bool:
    start = max(core.rfind(" ", 0, dot), core.rfind("\n", 0, dot)) + 1
    tok = core[start:dot + 1].lstrip("\"'([«“‘")
    if _ACRONYM.match(tok):                     # "U.S.", "e.g.", "J."
        return True
    return tok[:-1].lower() in _ABBREV


def _boundaries(core: str):
    """Yield (end, strong) for each confirmed boundary; core[:end] is the chunk incl. punctuation/closers."""
    n = len(core)
    i = 0
    while i < n:
        ch = core[i]
        if ch in _STRONG_FW or ch in _WEAK_FW:
            j = i
            while j + 1 < n and core[j + 1] in _STRONG_FW + _WEAK_FW + _CLOSERS:
                j += 1
            yield j + 1, ch in _STRONG_FW
            i = j + 1
            continue
        spaced_hyphen = ch == "-" and i > 0 and core[i - 1] == " "
        if (spaced_hyphen or ch == "–") and core[:i].rstrip()[-1:].isdigit():   # "5 - 10": a range
            i += 1
            continue
        if ch in _STRONG or ch in _WEAK or spaced_hyphen:
            j = i
            while j + 1 < n and core[j + 1] in _STRONG + _CLOSERS:
                j += 1
            if j + 1 < n and core[j + 1].isspace():          # confirmed by the following whitespace
                strong = any(c in _STRONG for c in core[i:j + 1])
                if not (ch == "." and j == i and _is_abbrev(core, i)):
                    yield j + 1, strong
            i = j + 1
            continue
        i += 1


def _word_cut(core: str, max_words: int):
    """Index just past the max_words-th COMPLETE word (one already followed by whitespace), else None."""
    words = 0
    for m in re.finditer(r"\S+(?=\s)", core):
        words += 1
        if words >= max_words:
            return m.end()
    return None


def next_chunk(pending: str, first: bool = False):
    """Return (chunk, rest) if a chunk is ready to TTS, else (None, pending). Operates on the untts'd tail.

    A boundary only counts once the chunk is >= MIN_CHARS, so short text is never fragmented.
    first=True (the first chunk of a spoken LLM stream): end at any boundary (sentence OR clause: , ; : dash),
    or at a word boundary after FIRST_MAX_WORDS words -- so playback starts without waiting for a long
    opening sentence. first=False: whole sentences only.
    """
    lead = len(pending) - len(pending.lstrip())
    core = pending[lead:]
    if not core:
        return None, pending
    for end, strong in _boundaries(core):
        if not first and end > SENT_MAX:
            break
        if len(core[:end].strip()) >= MIN_CHARS and (strong or first):
            return core[:end].strip(), core[end:]
    if first:
        cut = _word_cut(core, FIRST_MAX_WORDS)
        if cut is not None:
            return core[:cut].strip(), core[cut:]
    elif len(core) >= SENT_MAX:
        cut = core.rfind(" ", SENT_MAX // 2, SENT_MAX)
        if cut == -1:
            cut = SENT_MAX
        return core[:cut].strip(), core[cut:]
    return None, pending


def next_sentence(pending: str):
    """Whole-sentence chunk (see next_chunk)."""
    return next_chunk(pending, first=False)
