"""From text to the agent's voice: the one path to the TTS (speech_audio), and Speaker, the book-keeping of one
reply's audio (what was said, and when the client will have played it), made fresh for every turn."""
from __future__ import annotations
import time

from app.mouth.chunker import say_clean


def speech_audio(tts, text, voice, stop_flag):
    """The TTS audio of `text` as it is said aloud (say_clean). No request at all when nothing speakable is left
    (markup or punctuation only): every path from text to the TTS goes through here."""
    clean = say_clean(text)
    if clean:
        yield from tts.stream(clean, voice, stop_flag=stop_flag)


class Speaker:
    """One reply's audio as it is generated. `said`: each spoken line with the reply-audio offset (s) at which it
    started, so a barge-in can tell what the user heard; `audio_s`: reply audio generated so far; `play_end`: when
    the client will have played all of it (it plays chunks back to back, in real time)."""
    def __init__(self, tts, sr):
        self.tts = tts; self.sr = sr
        self.said: list[tuple[str, float]] = []
        self.audio_s = 0.0
        self.play_end = time.perf_counter()

    def say(self, text, voice, stop_flag):
        """TTS one line as reply audio: yields ("audio", chunk); the line counts as said from its first audio."""
        first = True
        for w in speech_audio(self.tts, text, voice, stop_flag):
            if first:
                self.line_started(text); first = False
            self.add_audio(w)
            yield ("audio", w)

    def line_started(self, text):
        """A line's first audio is out (fillers relayed from a helper thread call this themselves)."""
        self.said.append((text, self.audio_s))

    def add_audio(self, w):
        d = len(w) / self.sr
        self.audio_s += d
        self.play_end = max(self.play_end, time.perf_counter()) + d

    def played_out_for(self, seconds):
        """Whether everything said so far has finished playing at least `seconds` ago."""
        return time.perf_counter() >= self.play_end + seconds

    def heard(self, played_s, since=0):
        """The lines the user heard (at least the start of) when they cut in after `played_s` seconds of reply
        audio; every line said if that isn't known (None). `since`: only lines from that index on (len(said) when a
        part of the reply began, e.g. the answer after the fillers)."""
        return [t for t, at in self.said[since:] if played_s is None or at < played_s]
