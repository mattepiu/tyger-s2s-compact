"""ReplyTurn: the agent's answer to the user's words.

A plain reply is spoken as it decodes. On a tool call (L-04) the call's name has just decoded: a helper thread
finishes decoding the call and runs the tools the moment it parses (a ToolRound), while this thread speaks a lead-in
that fits the tool, made alongside (vLLM batches the two requests). If the tools are still out FILLER_GAP_S after
the last line has played, one more short "still looking" line, up to MAX_STILL. Then the reply model answers from
the results (phase B). Escalation (ask_expert, only with an `llm.smart` block): the smart request starts right after
the other tools, still under the lead-in, and its answer streams straight to the TTS, with its own tool rounds
covered the same way. Everything stops on barge-in (the stop flag); helper threads are daemons.
"""
from __future__ import annotations
import logging, threading, time
from dataclasses import dataclass, field
from typing import Any

from app import tools
from app.lang import language_text
from app.messages import reply_prefix, with_status, smart_messages, round_messages
from app.mouth.threads import Prefetch
from app.mouth.turns import Turn
from app.plugins.base import LLMError
from app.tools import EXPERT, START, CANCEL

log = logging.getLogger("mouth")


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
    done: threading.Event = field(default_factory=threading.Event)

    def finished(self):
        """The tools are back and, if an answer was started, its first event too: time to stop covering."""
        return self.done.is_set() and (self.next is None or self.next.ready.is_set())


class ReplyTurn(Turn):
    def __init__(self, session, stop_flag, transcript):
        super().__init__(session, stop_flag)
        self.transcript = transcript
        self.lang = language_text(transcript, session.history.user_words())

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
        events = m.llms["reply"].stream(msgs, tools=m.tools, max_new=m.max_gen, stop_flag=self.stop)
        content, calls, native, started, refused = yield from self.speak_stream(events)
        if refused and not self.stop():
            yield from self.speak_line("refusal", t); return
        if self.stop():
            self.remember_barged(t)
            yield ("done", content.strip()); return
        if not calls and not started:                       # plain reply
            reply = content.strip(); self.remember(t, reply)
            yield ("done", reply); return
        yield from self._tool_turn(msgs, events, content, calls, native, started)

    # ---- the tool path ----
    def _tool_turn(self, msgs, events, content, calls, native, started):
        m, t = self.m, self.transcript
        rnd = ToolRound(calls or [], native)
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
            self.remember_barged(t)
            yield ("done", " ".join(fillers)); return
        if isinstance(rnd.error, LLMError):          # the call never finished decoding (hosted LLM failed)
            raise rnd.error
        calls = rnd.calls
        results = rnd.results if len(rnd.results) == len(calls) else [None] * len(calls)
        if not calls:                                       # a tag with no parseable call: nothing to answer from
            lead = " ".join(fillers)
            self.remember(t, lead)
            yield ("done", lead); return

        kept = round_messages(content, calls, rnd.native, results, compact=True)
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
            self.remember_barged(t, kept)
        else:
            self.remember(t, answer.strip() or lead, kept)
        yield ("done", (lead + " " + answer).strip())

    def _finish_call_and_run_tools(self, events, rnd):
        """(helper thread) Finish decoding the call, run the tools, and for ask_expert start the smart model."""
        try:
            for ev in events:                               # no-op if speak_stream already drained it
                if ev[0] == "tool_calls":
                    rnd.calls, rnd.native = ev[1], ev[2]
            if self.stop() or not rnd.calls:
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
            rnd.done.set()

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
        """Run one round of tool calls at once; results in call order. A call the plugin flagged (bad arguments)
        is never run; with `expert`, ask_expert isn't run either: the smart model answers it. start_task /
        cancel_task go to this conversation's task runner."""
        results = [None] * len(calls)
        p = self.m.prompts["reply"]
        tasks = self.session.tasks

        def run(i, c):
            args = c.get("arguments") or {}
            if c.get("error"):
                results[i] = {"error": c["error"]}
            elif expert and c["name"] == EXPERT:
                results[i] = {"result": p.text("escalation", "tool_result")}
            elif tasks is not None and c["name"] == START:
                results[i] = tasks.start(args.get("goal"), transcript, self.lang)
            elif tasks is not None and c["name"] == CANCEL:
                try:
                    results[i] = tasks.cancel(int(args.get("task_id")))
                except (TypeError, ValueError):
                    results[i] = {"error": p.text("tasks", "unknown_task")}
            else:
                results[i] = tools.execute_tool(c["name"], c.get("arguments", {}))
        workers = [threading.Thread(target=run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return results

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
            nxt = ToolRound(calls or [], native)
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
            if self.stop() or not rnd.calls:
                return
            log.info("escalated turn: tool calls %s", [c["name"] for c in rnd.calls])
            results = [{"error": m.prompts["smart"].text("smart", "limit")}] * len(rnd.calls) if limit \
                else self.run_calls(rnd.calls)
            rnd.next_msgs = msgs + round_messages(text, rnd.calls, rnd.native, results,
                                                  note=m.prompts["smart"].result_note(self.lang))
            rnd.next = Prefetch(lambda: m.llms["smart"].stream(rnd.next_msgs, tools=m.smart_tools, max_new=m.max_gen,
                                                               stop_flag=self.stop), self.stop)
        except Exception as e:
            rnd.error = e
        finally:
            rnd.done.set()
