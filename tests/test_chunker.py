"""Chunker tests. Run: .venv/bin/python -m pytest tests/  (or plain: .venv/bin/python tests/test_chunker.py)

Replays text through the chunker the way Mouth._tts_tail does while the LLM streams (a few chars at a time),
then flushes the tail like Mouth._flush_tail, so streaming-only bugs (cutting "0." before "50" arrives) show up.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.chunker import next_chunk, next_sentence, say_clean  # noqa: E402


def stream_chunks(text, step=3):
    """Feed `text` in `step`-char deltas; return the chunks Mouth would TTS (first chunk in first mode)."""
    full, done, out = "", 0, []
    for i in range(0, len(text), step):
        full += text[i:i + step]
        safe = full[done:]
        while True:
            chunk, rest = next_chunk(safe, first=done == 0)
            if chunk is None:
                break
            done += len(safe) - len(rest)
            safe = rest
            out.append(chunk)
    tail = full[done:].strip()
    if tail:
        out.append(tail)
    return out


def assert_lossless(text, chunks):
    assert " ".join(chunks).split() == text.split(), chunks


# ---- in-word punctuation ------------------------------------------------------------------------
def test_in_word_punctuation_never_splits():
    for text in ("It costs 0.50$ today at the market downtown, which is cheap. Then we go home.",
                 "We sold 3,000 units at 1:30 in Las-Palmas yesterday, a well-known spot for that.",
                 "Visit example.com for all of the details and the prices, they said."):
        for step in (1, 2, 3, 7):
            chunks = stream_chunks(text, step)
            assert_lossless(text, chunks)
            for c in chunks:
                assert not c.endswith(("0.", "3,", "1:", "Las-", "example.")), (step, chunks)


def test_waits_for_lookahead():
    long_ = "That is everything I could find about the weather."
    # "0." at the end of the buffer may be "0.50" -- the cut must be the comma before it, never "0."
    assert next_chunk("It costs about as much as it did last year, roughly 0.", first=True) == \
        ("It costs about as much as it did last year,", " roughly 0.")
    assert next_chunk("It costs about as much as it did last year roughly 0.", first=True)[0] is None
    assert next_chunk(long_, first=False) == (None, long_)        # flushed by the caller at end of reply
    assert next_chunk(long_ + " ", first=False)[0] == long_
    q = 'He looked at me and said, very quietly, "go now." Then'
    assert next_chunk(q, first=False)[0] == 'He looked at me and said, very quietly, "go now."'


def test_number_ranges_are_not_clauses():
    for dash in ("-", "–"):
        text = f"Cook the pasta for 5 {dash} 10 minutes in salted water, then drain it and serve it warm."
        chunks = stream_chunks(text)
        assert chunks[0] == f"Cook the pasta for 5 {dash} 10 minutes in salted water,", chunks
        assert say_clean(chunks[0]) == "Cook the pasta for 5-10 minutes in salted water,"
    assert say_clean("5–10 minutes") == "5-10 minutes"


def test_abbreviations_and_initials():
    text = ("I spoke with Dr. Smith about the new plan yesterday. "
            "She lives in the U.S. now, e.g. near J. K. Rowling's old house. Bye.")
    s1, rest = next_sentence(text)
    assert s1 == "I spoke with Dr. Smith about the new plan yesterday."
    assert next_sentence(rest)[0] == "She lives in the U.S. now, e.g. near J. K. Rowling's old house."


# ---- the MIN_CHARS window: short text is never fragmented --------------------------------------
def test_short_reply_goes_whole():
    assert stream_chunks("Привет, как у тебя сегодня дела? Всё хорошо.") == \
        ["Привет, как у тебя сегодня дела? Всё хорошо."]
    assert stream_chunks("Hi! How can I help you today?") == ["Hi! How can I help you today?"]
    assert stream_chunks("你好，我是你的助手。今天天气很好！") == ["你好，我是你的助手。今天天气很好！"]


def test_short_sentences_merge_with_the_next():
    text = ("Well, octopuses are fascinating creatures, they have three hearts and blue blood. "
            "They are smart. They can even open jars and solve simple puzzles.")
    assert stream_chunks(text) == [
        "Well, octopuses are fascinating creatures,",                            # first: clause, >= 40
        "they have three hearts and blue blood. They are smart.",                # 38 chars -> merged
        "They can even open jars and solve simple puzzles.",
    ]


# ---- first chunk vs later chunks ----------------------------------------------------------------
def test_first_chunk_is_a_clause_later_are_sentences():
    text = ("Sure, here is a simple recipe you can try tonight, it only takes a few minutes. "
            "First, boil the water, then add the pasta and a little salt. Enjoy it.")
    assert stream_chunks(text) == [
        "Sure, here is a simple recipe you can try tonight,",
        "it only takes a few minutes. First, boil the water, then add the pasta and a little salt.",
        "Enjoy it.",
    ]
    assert stream_chunks("Ну что ж, давай посмотрим, что там у нас есть. Да.") == \
        ["Ну что ж, давай посмотрим, что там у нас есть.", "Да."]


def test_first_chunk_word_cap_without_punctuation():
    text = "the quick brown fox jumps over the lazy dog again and again and again until it is tired."
    chunks = stream_chunks(text)
    assert len(chunks[0].split()) == 12
    assert_lossless(text, chunks)


def test_long_runon_sentence_flushes():
    text = "Okay. " + ("and " * 80).strip() + "."
    chunks = stream_chunks(text)
    assert all(len(c) <= 220 for c in chunks)
    assert_lossless(text, chunks)


# ---- say_clean ----------------------------------------------------------------------------------
def test_dashes():
    assert stream_chunks("Paris — the capital of France, is lovely in the spring. Yes.")[0] == \
        "Paris — the capital of France, is lovely in the spring."
    assert say_clean("Paris — the capital - is lovely") == "Paris, the capital, is lovely"
    assert say_clean("Las-Palmas, a well-known spot,") == "Las-Palmas, a well-known spot,"
    assert say_clean("Привет — мир") == "Привет, мир"


def test_keeps_dollar_and_percent():
    assert say_clean("It costs 0.50$ or $2, about 50% off.") == "It costs 0.50$ or $2, about 50% off."
    assert say_clean("**Bold** #tag 🎉 done") == "Bold tag done"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
