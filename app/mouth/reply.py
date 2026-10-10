"""ReplyTurn: the agent's answer to the user's words.

A plain reply is spoken as it decodes. On a tool call (L-04) the call's name has just decoded: a helper thread
finishes decoding the call and runs the tools the moment it parses (a ToolRound), while this thread speaks a lead-in
that fits the tool, made alongside (vLLM batches the two requests). If the tools are still out FILLER_GAP_S after
the last line has played, one more short "still looking" line, up to MAX_STILL. Then the reply model answers from
the results (phase B). Escalation (ask_expert, only with an `llm.smart` block): the smart request starts right after
the other tools, still under the lead-in, and its answer streams straight to the TTS, with its own tool rounds
covered the same way. Everything said stops on barge-in (the stop flag); helper threads are daemons.

A barge before any of the answer was heard doesn't lose the round (I-16). A call that has started decoding decodes
to the end and its tools run as they would have (a barge stops what is said, not what was asked; no new model
request starts after it). The round then goes on as a background task of this conversation (app/mouth/tasks.py),
announced at the next pause; where no task can start, history keeps its results.
"""
from __future__ import annotations
import logging, threading, time
from dataclasses import dataclass, field
from typing import Any

from app import tools
from app.prompts.lang import language_text
from app.mouth.messages import compact_result, reply_prefix, with_status, smart_messages, round_messages
from app.mouth.threads import Prefetch
from app.mouth.turns import Turn
from app.plugins.base import LLMError
from app.tools import CANCEL, EXPERT, START

log = logging.getLogger("mouth")
CALL_WAIT_S = 2.0          # a barge while a call's arguments decode: wait this long at most for the call to parse
RESULT_WAIT_S = 60.0       # a barged round's results still out: history takes them if they come back within this


@dataclass
class ToolRound:
    """One round of tool calls, finished on a helper thread while the turn speaks over it. `next` is the stream that
    answers from the results when it was started there already (the smart model's), with `next_msgs` its request."""
    calls: list
    native: Any = None
    results: list = field(default_factory=list)
    error: Exception | None = None
    next: Prefetch | None = None
    next_msgs: list | None = None
    t0: float | None = None                      # when `next` was requested
    parsed: threading.Event = field(default_factory=threading.Event)   # the calls have decoded (or never will)
    done: threading.Event = field(default_factory=threading.Event)     # the tools are back

    def finished(self):
        """The tools are back and, if an answer was started, its first event too: time to stop covering."""
        return self.done.is_set() and (self.next is None or self.next.ready.is_set())

    def back(self):
        """[(call, result)] once the tools are back; [] before, or if they never ran."""
        return list(zip(self.calls, self.results)) if self.done.is_set() else []


class ReplyTurn(Turn):
    def __init__(self, session, stop_flag, transcript):
        super().__init__(session, stop_flag)
        self.transcript = transcript
        self.lang = language_text(transcript, session.history.user_words())
        self.calling = False                     # a tool call has started decoding
        self.rounds: list[ToolRound] = []        # this turn's tool rounds: the reply model's, then the smart model's

    def run(self):
        try:
            yield from self._reply()
        except LLMError as e:                    # hosted LLM failed: say so briefly instead of going silent
            log.warning("turn: LLM request failed (%s)", e.kind)
            yield from self.speak_line("error", self.transcript)

    def _reply(self):
        m, s, t = self.m, self.session, self.transcript
        status = s.tasks.status_block() if s.tasks is not None else ""   # tasks: a note on this message only
        msgs = reply_prefix(m.prompts["reply"], m.expert, m.tasks_on) + s.history.messages + \
            [{"role": "user", "content": with_status(m.prompts["reply"], status, t, self.lang)}]
        events = m.llms["reply"].stream(msgs, tools=m.tools, max_new=m.max_gen, stop_flag=self._stop_before_call)
        content, calls, native, started, refused = yield from self.speak_stream(events)
        self.calling = bool(calls or started)
        if refused and not self.stop():
            yield from self.speak_line("refusal", t); return
        if not self.calling:                                # plain reply
            if self.stop():
                self.remember_barged(t)
                yield ("done", content.strip()); return
            reply = content.strip(); self.remember(t, reply)
            yield ("done", reply); return
        yield from self._tool_turn(msgs, events, content, calls, native, started)

    def _stop_before_call(self):
        """The reply stream's stop flag: a barge stops it, unless a tool call has started decoding. The call then
        decodes to the end (a few dozen tokens), so a barged round can go on without the user (I-16)."""
        return self.stop() and not self.calling

    # ---- the tool path ----
    def _round(self, calls, native=None):
        rnd = ToolRound(calls or [], native)
        self.rounds.append(rnd)
        return rnd

    def _tool_turn(self, msgs, events, content, calls, native, started):
        m, t = self.m, self.transcript
        rnd = self._round(calls, native)
        threading.Thread(target=self._finish_call_and_run_tools, args=(events, rnd), daemon=True).start()
        tool = started["name"] if started else (calls[0]["name"] if calls else None)
        to_expert = m.expert and tool == EXPERT
        fillers = []
        if not self.speaker.said:              # the model's own words before the tag already did the job
            section = "expert_leadin" if to_expert else "tool_leadin"
            fp = m.prompts["filler"]
            line = yield from self.filler(section, fp.fill(section, "user", transcript=t) + fp.filler_hint(self.lang))
            if line:
                fillers.append(line)
        yield from self.cover(rnd.finished, "expert_still" if to_expert else "still_looking", t, fillers)
        if self.stop():
            yield from self._barged_round(content, fillers); return
        if isinstance(rnd.error, LLMError):          # the call never finished decoding (hosted LLM failed)
            raise rnd.error
        calls = rnd.calls
        results = rnd.results if len(rnd.results) == len(calls) else [None] * len(calls)
        if not calls:                                       # a tag with no parseable call: nothing to answer from
            lead = " ".join(fillers)
            self.remember(t, lead)
            yield ("done", lead); return

        kept = round_messages(content, calls, rnd.native, results, compact=True)
        answer_from = len(self.speaker.said)                # the answer's lines start here
        if rnd.next is not None:                            # escalated: the smart model answers, straight to TTS
            answer, refused = yield from self._expert_answer(rnd, fillers)
        else:
            msgs2 = msgs + round_messages(content, calls, rnd.native, results,
                                          note=m.prompts["reply"].result_note(self.lang))
            answer, _, _, _, refused = yield from self.speak_stream(
                m.llms["reply"].stream(msgs2, max_new=m.max_gen, stop_flag=self.stop))
        if refused and not self.stop():
            yield from self.speak_line("refusal", t); return
        lead = " ".join(fillers)
        if self.stop():
            if not self._answer_heard(answer_from, fillers):
                yield from self._barged_round(content, fillers); return
            played = self.session.barge_played_s
            log.info("barged tool round: the user had heard part of the answer (played %s s, the answer from %.1f s), "
                     "results kept in history", "?" if played is None else f"{played:.1f}",
                     self.speaker.said[answer_from][1] if len(self.speaker.said) > answer_from else 0.0)
            self.remember_barged(t, kept)
        else:
            self.remember(t, answer.strip() or lead, kept)
        yield ("done", (lead + " " + answer).strip())

    def _finish_call_and_run_tools(self, events, rnd):
        """(helper thread) Finish decoding the call, run the tools, and for ask_expert start the smart model. After a
        barge the call still decodes and the tools still run (the round may go on as a task); the smart model isn't
        asked."""
        try:
            for ev in events:                               # no-op if speak_stream already drained it
                if ev[0] == "tool_calls":
                    rnd.calls, rnd.native = ev[1], ev[2]
            rnd.parsed.set()
            if not rnd.calls:
                return
            log.info("tool calls: %s", [c["name"] for c in rnd.calls])
            rnd.results = self.run_calls(rnd.calls, expert=self.m.expert, transcript=self.transcript)
            ask = next((c for c in rnd.calls if c["name"] == EXPERT and not c.get("error")), None)
            if ask is not None and self.m.expert and not self.stop():
                self._start_smart(rnd, ask)
        except Exception as e:
            rnd.error = e
            log.warning("tool call error: %s", type(e).__name__)
        finally:
            rnd.parsed.set(); rnd.done.set()

    # ---- a barge before the answer: the round goes on in the background (I-16) ----
    def _answer_heard(self, answer_from, fillers):
        """Whether the user had heard any of the answer when they cut in: a line from `answer_from` on that was
        playing by then (EARS's played_s), still-looking lines over the smart model's own lookups aside."""
        return any(line not in fillers for line in self.speaker.heard(self.session.barge_played_s, since=answer_from))

    def _barged_round(self, content, fillers):
        """The user cut in before any of the answer was heard. The round goes on as a background task of this
        conversation: the expert's question, or the user's words with what the round looked up (the task waits for
        results still out, and doesn't redo them). History keeps the call with a note that it goes on in the
        background, so the reply model sees the task in its status note (and can cancel it); it is announced at the
        next pause like any task. Where no task can start (tasks off, a cap), history keeps the round's results."""
        t, rnd = self.transcript, self.rounds[0]
        rnd.parsed.wait(CALL_WAIT_S)                        # cut in while the call's arguments were decoding
        if not rnd.calls:
            self.remember_barged(t)
        else:
            task = self._continue_as_task(rnd)
            self.remember_barged(t, self._barged_history(rnd, content, task))
        yield ("done", " ".join(fillers))

    def _continue_as_task(self, rnd):
        """The task the round goes on as, or None: tasks off, a cap reached, or only MOUTH's task tools in it
        (start_task / cancel_task have done their job). Logs counts only."""
        s = self.session
        if all(c["name"] in (START, CANCEL) for c in rnd.calls):
            return None
        if s.tasks is None:
            log.info("barged tool round: %d call(s), results kept in history (tasks off)", len(rnd.calls))
            return None
        ask = next((c for c in rnd.calls if c["name"] == EXPERT), None)
        question = (ask.get("arguments") or {}).get("question") if ask is not None else None
        return s.tasks.continue_round(question or self.transcript, self.transcript, self.lang, self._looked_up)

    def _looked_up(self, stop):
        """(the task's thread) What this turn's rounds looked up, as [(tool name, result)]. Waits for the reply model's
        round to come back (or stop()); the smart model's own rounds count where they came back before the barge."""
        while not self.rounds[0].done.wait(0.05) and not stop():
            pass
        return [(c["name"], r) for rnd in self.rounds for c, r in rnd.back() if tools.runs_for_real(c["name"])]

    def _barged_history(self, rnd, content, task):
        """The barged round as history keeps it: the calls, and for each the task's note (it goes on in the
        background), or for an expert who never answered a note saying so, or its compact result. A result still
        out is written in when it comes back."""
        p = self.m.prompts["reply"]

        def stand_in(c):                    # what history says instead of the call's result (None: the result)
            if task is not None and c["name"] not in (START, CANCEL):
                return {"task_id": task.id, "status": p.text("tasks", "continued")}
            if c["name"] == EXPERT:
                return {"error": p.text("escalation", "interrupted")}
            return None
        stand = [stand_in(c) for c in rnd.calls]
        msgs = round_messages(content, rnd.calls, rnd.native, stand, compact=True)
        real = [(msg, i) for i, msg in enumerate(msgs[1:]) if stand[i] is None]

        def write_results():
            for msg, i in real:
                msg["content"] = compact_result(rnd.results[i] if i < len(rnd.results) else None)
        if real and rnd.done.is_set():
            write_results()
        elif real:
            for msg, _ in real:
                msg["content"] = compact_result({"status": p.text("tasks", "result_pending")})
            threading.Thread(target=lambda: rnd.done.wait(RESULT_WAIT_S) and write_results(), daemon=True).start()
        return msgs

    def _start_smart(self, rnd, ask):
        """Request the smart model's answer to an escalated turn: the recent turns as plain text, the user's words,
        the reply model's restated question, the other tools' results, and finished tasks' full results."""
        m, s = self.m, self.session
        others = [(c["name"], r) for c, r in zip(rnd.calls, rnd.results) if c["name"] != EXPERT]
        rnd.next_msgs = smart_messages(m.prompts["smart"], s.history.recent(m.context_turns), self.transcript,
                                       (ask.get("arguments") or {}).get("question"), others)
        done = s.tasks.done_results() if s.tasks is not None else []
        if done:                                            # deeper follow-ups on a task get its full result
            rnd.next_msgs[-1]["content"] += m.prompts["smart"].fill(
                "tasks", "expert_context", results="; ".join(f'#{t.id} "{t.goal}": {r}' for t, r in done))
        rnd.t0 = time.perf_counter()
        rnd.next = Prefetch(lambda: m.llms["smart"].stream(rnd.next_msgs, tools=m.smart_tools, max_new=m.max_gen,
                                                           stop_flag=self.stop), self.stop)

    def run_calls(self, calls, expert=False, transcript=""):
        """Run one round of tool calls at once; results in call order (app/tools). With `expert`, ask_expert gets a
        note (the smart model answers the turn); start_task / cancel_task go to this conversation's task runner."""
        ctx = tools.ToolContext(prompts=self.m.prompts["reply"], transcript=transcript, lang=self.lang,
                                tasks=self.session.tasks, expert=expert)
        return tools.run_round(calls, ctx)

    # ---- escalation: the smart model answers, with its own tool rounds ----
    def _expert_answer(self, rnd, fillers):
        """Speak the smart model's answer as it streams (`rnd.next`, already under way), running its own tool rounds:
        up to max_steps, each covered by still-looking lines and the next request started as soon as the results
        are in. Past max_steps its calls get the smart.limit note instead of results, and a model that still calls
        tools fails the turn (LLMError: MOUTH apologises). Returns (text, refused). Logs timings and counts only."""
        m = self.m; said = []; rounds = 0
        stream, msgs, t0 = rnd.next, rnd.next_msgs, rnd.t0
        t_first = stream.t_first
        while True:
            text, calls, native, started, refused = yield from self.speak_stream(stream)
            if text.strip():
                said.append(text.strip())
            if refused or self.stop() or not (calls or started):
                log.info("escalated turn: smart first event %s, %d tool rounds%s",
                         f"{1e3 * (t_first - t0):.0f} ms" if t_first else "none", rounds,
                         " (barged)" if self.stop() else "")
                return " ".join(said), refused
            if rounds > m.max_steps:
                log.warning("escalated turn: smart model kept calling tools past max_steps")
                raise LLMError("max_steps")
            nxt = self._round(calls, native)
            threading.Thread(target=self._finish_smart_round, args=(nxt, stream, msgs, text, rounds >= m.max_steps),
                             daemon=True).start()
            yield from self.cover(nxt.finished, "still_looking", self.transcript, fillers)
            if self.stop():
                return " ".join(said), False
            if nxt.error is not None:
                raise nxt.error
            if nxt.next is None:                            # a tool call that didn't parse: nothing more to say
                return " ".join(said), False
            rounds += 1; stream, msgs = nxt.next, nxt.next_msgs

    def _finish_smart_round(self, rnd, stream, msgs, text, limit):
        """(helper thread) Finish decoding the smart model's calls, run them (or, past max_steps, answer each with
        the limit note), and start its next request."""
        m = self.m
        try:
            for ev in stream:
                if ev[0] == "tool_calls":
                    rnd.calls, rnd.native = ev[1], ev[2]
            rnd.parsed.set()
            if self.stop() or not rnd.calls:
                return
            log.info("escalated turn: tool calls %s", [c["name"] for c in rnd.calls])
            if limit:
                results = [{"error": m.prompts["smart"].text("smart", "limit")}] * len(rnd.calls)
            else:
                results = rnd.results = self.run_calls(rnd.calls)     # kept: a barged round's task reuses them
            rnd.next_msgs = msgs + round_messages(text, rnd.calls, rnd.native, results,
                                                  note=m.prompts["smart"].result_note(self.lang))
            rnd.next = Prefetch(lambda: m.llms["smart"].stream(rnd.next_msgs, tools=m.smart_tools, max_new=m.max_gen,
                                                               stop_flag=self.stop), self.stop)
        except Exception as e:
            rnd.error = e
        finally:
            rnd.parsed.set(); rnd.done.set()
