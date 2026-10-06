"""Background tasks (PLUGINS.md steps 12-14): work that takes minutes runs on the `smart` model while the
conversation carries on.

The `reply` model calls start_task(goal) / cancel_task(task_id); MOUTH answers those calls here. Each task is an
agentic loop on the smart plugin in its own thread: stream, run the tool calls (lookups only; finish_task ends it
with a spoken summary and the full result), append the round, repeat. The task's message list is append-only, so a
hosted model's own content (`native`, e.g. Claude's thinking) stays valid within it. Limits: mouth.tasks.max_steps
tool rounds (then the `limit` note instead of results, one more round), max_wall_s, the plugin's own max_tokens;
max_concurrent and max_per_session per conversation, max_global across them (TaskSlots, shared by all sessions).

A conversation's tasks live in memory only and are cancelled when it ends. The reply model sees them through a
status note on the user's latest message (status_block). Logs carry task IDs, timings, counts and status: never
goals, results or summaries.
"""
from __future__ import annotations
import logging, threading, time

from app import tools
from app.mouth.messages import plain_context, round_messages
from app.plugins.base import LLMError
from app.tools import FINISH

log = logging.getLogger("tasks")


class TaskSlots:
    """The cap on tasks running at once across all conversations (mouth.tasks.max_global)."""
    def __init__(self, n):
        self.n = int(n); self.used = 0; self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if self.used >= self.n:
                return False
            self.used += 1
            return True

    def give(self):
        with self.lock:
            self.used = max(0, self.used - 1)


class Task:
    def __init__(self, tid, goal, transcript="", lang_text=None):
        self.id = tid; self.goal = goal; self.t0 = time.monotonic(); self.t_end = None
        self.transcript = transcript    # the words that started it
        self.lang_text = lang_text or transcript   # what its language is guessed from (app/prompts/lang.language_text)
        self.status = "running"         # running / done / failed / cancelled
        self.announced = "no"           # no / yes / partial (step 13)
        self.summary = self.result = None
        self.why = None                 # failure kind (logged): timeout / max_steps / refused / empty / LLMError kind
        self.steps = 0
        self.stop = threading.Event()


class TaskRunner:
    """One conversation's background tasks. `m`: the Mouth (llms, prompts, tool lists, task config and slots);
    `context`: a callable giving the conversation turns to send along; on_done(task) is called when a task
    finishes, fails or times out (not when cancelled)."""
    def __init__(self, m, context=lambda: [], on_done=None):
        self.m = m; self.context = context; self.on_done = on_done
        c = m.tasks_cfg
        self.max_concurrent = int(c.get("max_concurrent", 1)); self.max_per_session = int(c.get("max_per_session", 5))
        self.max_wall_s = float(c.get("max_wall_s", 300)); self.max_steps = int(c.get("max_steps", 8))
        self.result_chars = int(c.get("result_chars", 4000))
        self.tasks: list[Task] = []; self.lock = threading.Lock(); self.closed = False

    # ---- the reply model's tools ----
    def start(self, goal, transcript="", lang_text=None):
        """start_task: the tool result for the reply model ({task_id, status} or {error})."""
        p = self.m.prompts["reply"]
        goal = (goal or "").strip() or (transcript or "").strip()
        with self.lock:
            if self.closed:
                return {"error": p.text("tasks", "cap_global")}
            running = sum(t.status == "running" for t in self.tasks)
            if running >= self.max_concurrent:
                return {"error": p.text("tasks", "cap_concurrent")}
            if len(self.tasks) >= self.max_per_session:
                return {"error": p.text("tasks", "cap_session")}
            if not self.m.task_slots.take():
                log.info("task refused: global cap (%d) reached", self.m.task_slots.n)
                return {"error": p.text("tasks", "cap_global")}
            task = Task(len(self.tasks) + 1, goal, transcript, lang_text)
            self.tasks.append(task)
        log.info("task #%d started (this conversation: %d running)", task.id, running + 1)
        threading.Thread(target=self._run, args=(task, list(self.context())), daemon=True).start()
        return {"task_id": task.id, "status": "started"}

    def cancel(self, task_id):
        """cancel_task: the tool result for the reply model."""
        with self.lock:
            task = next((t for t in self.tasks if t.id == task_id and t.status == "running"), None)
        if task is None:
            return {"error": self.m.prompts["reply"].text("tasks", "unknown_task")}
        task.stop.set(); task.status = "cancelled"
        log.info("task #%d cancelled by the user", task.id)
        return {"task_id": task.id, "status": "cancelled"}

    def close(self):
        """Conversation over: cancel every running task (nothing persists)."""
        with self.lock:
            self.closed = True
            live = [t for t in self.tasks if t.status == "running"]
        for t in live:
            t.stop.set(); t.status = "cancelled"
        if live:
            log.info("conversation ended: %d task(s) cancelled", len(live))

    # ---- what the reply model sees ----
    def status_block(self):
        """The note prepended to the user's message, or "" if this conversation has no tasks."""
        p = self.m.prompts["reply"]
        with self.lock:
            tasks = list(self.tasks)
        if not tasks:
            return ""
        lines = []
        for t in tasks:
            if t.status == "running":
                lines.append(p.fill("tasks", "status_running", id=t.id, goal=t.goal,
                                    seconds=int(time.monotonic() - t.t0)))
            elif t.status == "done":
                key = {"no": "status_done", "yes": "status_announced", "partial": "status_partial"}[t.announced]
                lines.append(p.fill("tasks", key, id=t.id, goal=t.goal, summary=t.summary))
            else:
                lines.append(p.fill("tasks", f"status_{t.status}", id=t.id, goal=t.goal))
        return p.fill("tasks", "status_block", tasks="; ".join(lines))

    def done_results(self, n=2):
        """The full results of the last `n` finished tasks, for an escalation's context: [(task, result)]."""
        with self.lock:
            return [(t, t.result) for t in self.tasks if t.status == "done" and t.result][-n:]

    def get(self, task_id):
        with self.lock:
            return next((t for t in self.tasks if t.id == task_id), None)

    # ---- the agentic loop, on its own thread ----
    def _messages(self, task, turns):
        p = self.m.prompts["smart"]
        user = p.fill("tasks", "user", goal=task.goal, transcript=task.transcript or task.goal,
                      language=p.language_or_unknown(task.lang_text))
        return [{"role": "system", "content": p.text("tasks", "system")}] + plain_context(turns) + \
            [{"role": "user", "content": user}]

    def _run(self, task, turns):
        m = self.m; smart = m.llms["smart"]; p = m.prompts["smart"]
        deadline = task.t0 + self.max_wall_s
        stop = lambda: task.stop.is_set() or time.monotonic() > deadline   # noqa: E731
        msgs = self._messages(task, turns)
        try:
            for rnd in range(self.max_steps + 2):
                text, calls, native = "", [], None
                for ev in smart.stream(msgs, tools=m.task_tools, max_new=m.max_gen, stop_flag=stop):
                    if ev[0] == "text":
                        text += ev[1]
                    elif ev[0] == "tool_calls":
                        calls, native = ev[1], ev[2]
                    elif ev[0] == "refusal":
                        return self._end(task, "failed", why="refused")
                if task.stop.is_set():
                    return                                     # cancelled: cancel() / close() set the status
                if stop():
                    return self._end(task, "failed", why="timeout")
                fin = next((c for c in calls if c["name"] == FINISH and not c.get("error")), None)
                if fin is not None:
                    a = fin.get("arguments") or {}
                    return self._end(task, "done", summary=str(a.get("summary") or "").strip(),
                                     result=str(a.get("result") or a.get("summary") or "").strip())
                if not calls:                                  # answered in text instead of finish_task
                    if text.strip():
                        return self._end(task, "done", summary=text.strip(), result=text.strip())
                    return self._end(task, "failed", why="empty")
                if rnd > self.max_steps:
                    return self._end(task, "failed", why="max_steps")
                task.steps += 1
                limit = rnd >= self.max_steps
                log.info("task #%d round %d: %s", task.id, task.steps, [c["name"] for c in calls])
                results = [{"error": p.text("tasks", "limit")} if limit or c["name"] == FINISH else
                           tools.run_call(c, tools.ToolContext(prompts=p)) for c in calls]
                msgs = msgs + round_messages(text, calls, native, results,
                                               note=None if limit else p.result_note(task.lang_text))
            return self._end(task, "failed", why="max_steps")
        except LLMError as e:
            return self._end(task, "failed", why=e.kind)
        except Exception as e:                                 # a task must never take the conversation down
            log.warning("task #%d error: %s", task.id, type(e).__name__)
            return self._end(task, "failed", why=type(e).__name__)
        finally:
            m.task_slots.give()

    def _end(self, task, status, summary=None, result=None, why=None):
        if task.status != "running":                       # cancelled meanwhile
            return
        task.summary = summary; task.result = (result or "")[:self.result_chars] or None
        task.why = why; task.t_end = time.monotonic(); task.status = status
        log.info("task #%d %s after %.1f s, %d tool rounds%s", task.id, status, task.t_end - task.t0, task.steps,
                 f" ({why})" if why else "")
        if self.on_done:
            try:
                self.on_done(task)
            except Exception as e:
                log.warning("task #%d on_done error: %s", task.id, type(e).__name__)
