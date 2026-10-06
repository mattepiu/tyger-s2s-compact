"""The agent's turns. Every turn is a generator of MOUTH events, consumed by the session's turn worker:
("text", the reply text so far), ("audio", a 24 kHz chunk), then one ("done", final text).

Turn holds what every kind of turn shares: its Speaker (this reply's audio book-keeping), speaking an LLM event
stream chunk by chunk, MOUTH's own spoken lines, fillers over tool latency, and remembering the turn. The kinds:
  * ReplyTurn (reply.py): an answer to the user's words, with tools and escalation;
  * AnnouncementTurn: a finished background task speaks up once EARS gives it the floor;
  * SayTurn: speak a given text, TTS only (test tools use it for a fixed utterance; never remembered).
"""
from __future__ import annotations
import logging, queue, time

from app.mouth.chunker import next_chunk
from app.mouth.messages import with_shots, still_user, compact_result
from app.mouth.speech import Speaker, speech_audio
from app.mouth.threads import END, spawn, text_of

log = logging.getLogger("mouth")

FILLER_GAP_S = 0.8         # tools still out this long after the filler has played -> one more "still looking" line
MAX_STILL = 2              # at most this many "still looking" lines per tool turn


class Turn:
    def __init__(self, session, stop_flag):
        self.session = session; self.m = session.m; self.stop = stop_flag
        self.voice = session.voice               # one voice per turn: a switch applies to the next one
        self.speaker = Speaker(self.m.tts, self.m.sr)
        self.lang = ""                           # what the language hints are guessed from (ReplyTurn sets it)

    def run(self):
        raise NotImplementedError

    # ---- history ----
    def remember(self, user, reply, tool_msgs=()):
        self.session.history.add(user, reply, tool_msgs)

    def remember_barged(self, user, tool_msgs=()):
        """The reply up to the line that was playing when the user cut in (every line said, if EARS didn't say how
        much had played)."""
        self.session.history.add_barged(user, self.speaker.heard(self.session.barge_played_s), tool_msgs)

    # ---- speak an LLM text stream: the first chunk by mouth.first_chunk, then whole sentences ----
    def speak_stream(self, events):
        """Speak the text of an LLM event stream as it arrives. Yields MOUTH events; returns (text, calls, native,
        started, refused). Returns at "tool_start" (started = its payload, {"name": ...}) without consuming the
        rest: the call is still decoding, and the caller finishes `events` itself. Text before the tag that didn't
        close a chunk is never spoken. A refusal (hosted models) returns refused=True with nothing more spoken."""
        content = ""; calls = None; native = None; done = 0
        early = self.m.first_chunk == "clause"          # the first chunk may end at a clause (chunker.py)
        for ev in events:
            kind, val = ev[0], ev[1]
            if kind == "tool_start":
                return content, None, None, val or {"name": None}, False
            if kind == "refusal":
                log.info("reply refused (category=%s)", val)
                return content, None, None, None, True
            if kind == "text":
                content += val
                yield ("text", content)
                safe = content[done:]
                while True:
                    chunk, rest = next_chunk(safe, first=early and done == 0)
                    if chunk is None:
                        break
                    done += len(safe) - len(rest); safe = rest
                    if self.stop():
                        break
                    yield from self.speaker.say(chunk, self.voice, self.stop)
            elif kind == "tool_calls":
                calls, native = val, ev[2]
            if self.stop():
                return content, calls, native, None, False
        if not calls and not self.stop():
            yield from self.speaker.say(content[done:], self.voice, self.stop)
        return content, calls, native, None, False

    def speak_line(self, key, transcript):
        """End the turn with one of MOUTH's own lines (config/prompts.yaml `spoken`): the apology after a failed LLM
        request, or the neutral line after a refusal. The user's turn stays in history."""
        line = self.m.prompts["reply"].text("spoken", key)
        yield ("text", line)
        yield from self.speaker.say(line, self.voice, self.stop)
        if self.stop():
            self.remember_barged(transcript)
        else:
            self.remember(transcript, line)
        yield ("done", line)

    # ---- fillers: short lines spoken over tool latency, made on a helper thread ----
    def filler(self, section, user):
        """Speak one filler line, made on a helper thread (config/prompts.yaml `section`: system prompt + few-shot
        turns, then `user`). Yields its events as this turn's reply; returns the line (None if there was none)."""
        return (yield from self._relay(spawn(lambda: self._make_filler(section, user))))

    def _make_filler(self, section, user):
        """("text", line), then ("audio", chunk)...: the line from the filler LLM, then its TTS."""
        p = self.m.prompts["filler"]
        msgs = with_shots(p.text(section, "system"), p.shots(section), user)
        line = text_of(self.m.llms["filler"].stream(msgs, max_new=24, stop_flag=self.stop)).strip()
        if not line or self.stop():
            return
        yield ("text", line)
        yield from (("audio", w) for w in speech_audio(self.m.tts, line, self.voice, self.stop))

    def _relay(self, q):
        """Yield a filler's items from its helper thread as this turn's reply; returns the filler line."""
        line = None; first = True
        while not self.stop():
            try:
                x = q.get(timeout=0.005)
            except queue.Empty:
                continue
            if x is END:
                break
            kind, val = x
            if kind == "text":
                line = val
            else:
                if first:
                    self.speaker.line_started(line); first = False
                self.speaker.add_audio(val)
            yield x
        return line

    def cover(self, finished, section, transcript, fillers):
        """Until finished(): whenever the last line has played out FILLER_GAP_S ago, one more short line from
        config/prompts.yaml `section` (still_looking or expert_still), up to MAX_STILL. Appends the lines to
        `fillers`."""
        n = 0
        while not finished() and not self.stop():
            if n < MAX_STILL and self.speaker.played_out_for(FILLER_GAP_S):
                n += 1
                fp = self.m.prompts["filler"]
                line = yield from self.filler(section, still_user(fp, transcript, fillers, section) +
                                              fp.filler_hint(self.lang))
                if line:
                    fillers.append(line)
            else:
                time.sleep(0.01)


class SayTurn(Turn):
    """Speak a given text, TTS only: no LLM, not remembered (the S wire message; test tools only)."""
    def __init__(self, session, stop_flag, text):
        super().__init__(session, stop_flag)
        self.text = text

    def run(self):
        yield from self.speaker.say(self.text, self.voice, self.stop)
        yield ("done", self.text)


class AnnouncementTurn(Turn):
    """A finished task's summary (or, if it failed, one sentence saying so), spoken as a reply once EARS gives it the
    floor, chunked like one. History keeps a note with the compacted result, then what was heard; a barge marks the
    task as announced partially, so the reply model can repeat it when asked."""
    def __init__(self, session, stop_flag, task_id):
        super().__init__(session, stop_flag)
        self.task_id = task_id

    def run(self):
        m, s = self.m, self.session
        p = m.prompts["reply"]
        task = s.tasks.get(self.task_id) if s.tasks is not None else None
        if task is None or task.status not in ("done", "failed") or task.announced == "yes":
            yield ("done", ""); return                     # cancelled meanwhile, or already told
        if task.status == "done":
            line = task.summary or task.result or ""
            note = p.fill("tasks", "result_note", id=task.id, goal=task.goal,
                          result=compact_result(task.result or task.summary or ""))
        else:
            fp = m.prompts["filler"]
            user = fp.fill("task_failed", "user", goal=task.goal) + fp.filler_hint(task.lang_text)
            line = text_of(m.llms["filler"].stream(with_shots(fp.text("task_failed", "system"),
                                                              fp.shots("task_failed"), user),
                                                   max_new=40, stop_flag=self.stop)).strip()
            note = p.fill("tasks", "failed_note", id=task.id, goal=task.goal)
        yield from self.speak_stream(iter([("text", line)]))
        barged = self.stop()
        task.announced = "partial" if barged else "yes"
        log.info("task #%d announced%s", task.id, " partially (barged)" if barged else "")
        if barged:
            self.remember_barged(note)
        else:
            self.remember(note, line)
        yield ("done", line)
