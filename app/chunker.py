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
def say_clean(text: str) -> str:
    s = _KEEP.sub(" ", text)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s+([.,!?;:])", r"\1", s).strip()

# ---- streaming sentence/clause chunker ---------------------------------------------------------
_STRONG = ".!?"
_WEAK = ",;:"
_CLOSERS = '"\')]'
MIN_WEAK = 18       # only split on a comma/clause once the pending chunk is at least this long
MAX_CHARS = 90      # force a split beyond this (keeps every TTS call short -> low, flat first-audio)


def next_chunk(pending: str):
    """Return (chunk, rest) if a chunk is ready to TTS, else (None, pending). Operates on the untts'd tail."""
    s = pending
    lead = len(s) - len(s.lstrip())
    core = s[lead:]
    if not core:
        return None, pending
    for i, ch in enumerate(core[:MAX_CHARS]):
        if ch in _STRONG:
            j = i
            while j + 1 < len(core) and core[j + 1] in _STRONG + _CLOSERS:
                j += 1
            return core[:j + 1].strip(), core[j + 1:]
    if len(core) >= MIN_WEAK:
        for i, ch in enumerate(core):
            if ch in _WEAK and i >= MIN_WEAK:
                return core[:i + 1].strip(), core[i + 1:]
    if len(core) >= MAX_CHARS:
        cut = core.rfind(" ", MIN_WEAK, MAX_CHARS)
        if cut == -1:
            cut = MAX_CHARS
        return core[:cut].strip(), core[cut:]
    return None, pending


# ---- SENTENCE chunker: split only on sentence-final punctuation (.!?), never mid-sentence on a comma.
# Used by the MOUTH orchestrator (LLM on vLLM streams fast enough that whole sentences keep prosody clean
# while still starting playback on sentence 1). A very long run-on still flushes at SENT_MAX on a space so
# one TTS call never grows unbounded.
SENT_MAX = 220
def next_sentence(pending: str):
    """Return (sentence, rest) if a sentence has closed, else (None, pending). Operates on the untts'd tail."""
    s = pending
    lead = len(s) - len(s.lstrip())
    core = s[lead:]
    if not core:
        return None, pending
    for i, ch in enumerate(core[:SENT_MAX]):
        if ch in _STRONG:
            j = i
            while j + 1 < len(core) and core[j + 1] in _STRONG + _CLOSERS:
                j += 1
            return core[:j + 1].strip(), core[j + 1:]
    if len(core) >= SENT_MAX:
        cut = core.rfind(" ", SENT_MAX // 2, SENT_MAX)
        if cut == -1:
            cut = SENT_MAX
        return core[:cut].strip(), core[cut:]
    return None, pending
