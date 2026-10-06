"""The neutral message format MOUTH speaks to every LLM plugin (app/plugins/base.py has the format), and every
builder of it: the reply request's fixed prefix and few-shot turns, tool rounds, the smart model's request, the
backchannel and filler requests. Pure functions over a Prompts (app/prompts/): no I/O, no model, no state, so
MOUTH, background tasks, tests and scripts/bench/escalation.py build exactly the same requests.
"""
from __future__ import annotations
import json

from app.tools import EXPERT


def with_shots(system, shots, user):
    msgs = [{"role": "system", "content": system}]
    for q, a in shots:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": user}]


def still_user(p, transcript, fillers, section="still_looking"):
    """A still-looking request's user line (`section`: still_looking or expert_still): what was asked and the
    lines already spoken."""
    said = " ".join(f"'{f}'" for f in fillers) or p.text(section, "said_none")
    return p.fill(section, "user", transcript=transcript, said=said)


def bc_messages(p, recent_acks, partial, bc_min_words):
    """The backchannel request: the system prompt (+ the acks not to repeat), few-shot turns, and the partial
    framed as empty / only a few words / a reaction-worthy utterance."""
    system = p.text("backchannel", "system")
    if recent_acks:
        system += p.fill("backchannel", "no_repeat", acks=", ".join(f"'{a}'" for a in recent_acks))
    user = (partial or "").strip()
    if not user:
        said = p.text("backchannel", "user_empty")
    elif len(user.split()) < bc_min_words:
        said = p.fill("backchannel", "user_short", partial=user)
    else:
        said = p.fill("backchannel", "user", partial=user)
    return with_shots(system, p.shots("backchannel"), said)


def expert_result(p):
    """The ask_expert tool result as kept in history (the expert's answer follows it)."""
    return compact_result({"result": p.text("escalation", "tool_result")})


def turns_messages(p, turns, id_prefix="shot"):
    """Structured turns -> neutral messages: {user, reply} is answered locally; {user, expert, answer} is an
    ask_expert call (with `expert` as its question, ID call_<id_prefix><n>), its tool result, then the answer;
    {user, call: {name, arguments}, result, answer} is any other tool call with its (compacted) result. Used for
    the escalation few-shot turns and for the history of scripts/bench/escalation.py cases."""
    msgs = []
    for i, t in enumerate(turns, 1):
        msgs.append({"role": "user", "content": t["user"]})
        if "reply" in t:
            msgs.append({"role": "assistant", "content": t["reply"]})
            continue
        cid = f"call_{id_prefix}{i:04d}"
        if "call" in t:
            call, result = {"id": cid, **t["call"]}, compact_result(t["result"])
        else:
            call, result = {"id": cid, "name": EXPERT, "arguments": {"question": t["expert"]}}, expert_result(p)
        msgs += [{"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": cid, "content": result},
                 {"role": "assistant", "content": t["answer"]}]
    return msgs


def reply_prefix(p, expert, tasks=False):
    """The reply request before the conversation: the system prompt, the few-shot turns (reply_shots: local answers
    and lookups), with escalation on the hand-off turns (escalation.shots), with tasks on the start_task turns
    (tasks.shots), as earlier turns in the neutral format. A fixed prefix, so it stays cached."""
    system = p.text("tasks", "reply_system") if tasks else \
        p.text("escalation", "system") if expert else p.text("system", "voice_tools")
    msgs = [{"role": "system", "content": system}] + turns_messages(p, p.turns("reply_shots"), "rshot")
    msgs += turns_messages(p, p.turns("escalation")) if expert else []
    return msgs + (turns_messages(p, p.turns("tasks"), "tshot") if tasks else [])


def with_status(p, status, transcript, lang_text=None):
    """The user's message as the reply model gets it while background tasks exist: the status note (+ the user's
    language named, guessed from `lang_text` (default: their words), as the note is English: step 14 eval), then
    their words."""
    if not status:
        return transcript
    name = p.language_name(lang_text or transcript)
    lang = " " + p.fill("language", "status_note", language=name) if name else ""
    return f"{status}{lang}\n\n{transcript}"


def plain_context(turns):
    """Conversation turns as plain user / assistant text for a smart-model request (escalation, tasks): tool calls
    and results dropped (provider-neutral, no foreign tool IDs, and less leaves the box)."""
    return [{"role": x["role"], "content": x["content"]} for turn in turns for x in turn
            if x["role"] in ("user", "assistant") and not x.get("tool_calls") and x.get("content")]


def smart_messages(p, turns, transcript, question, others=()):
    """The smart model's request for an escalated turn: its system prompt, the given recent turns as plain text
    (tool calls and results dropped: provider-neutral, and less leaves the box), then the user's words with the
    reply model's restated question as a note, plus what other tools called in the same turn returned."""
    ctx = plain_context(turns)
    user = p.fill("smart", "user", transcript=transcript, question=question or transcript)
    if others:
        user += p.fill("smart", "results", results="; ".join(f"{n}: {compact_result(r)}" for n, r in others))
    return [{"role": "system", "content": p.text("smart", "system")}] + ctx + [{"role": "user", "content": user}]


def round_messages(content, calls, native, results, compact=False, note=None):
    """One tool round in the neutral format: the assistant's calls (plugin-issued IDs; `native` passed through)
    and a result per call, full or compacted for history. `note` (Prompts.result_note) is appended to each full
    result: tool output is usually English, and a small model otherwise answers in its language."""
    asst = {"role": "assistant", "content": (content or "").strip() or None,
            "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c.get("arguments", {})} for c in calls]}
    if native is not None:
        asst["native"] = native
    res = (lambda r: compact_result(r)) if compact else \
        (lambda r: json.dumps(r if r is not None else {"error": "no result"}) + (f"\n\n{note}" if note else ""))
    return [asst] + [{"role": "tool", "tool_call_id": c["id"], "content": res(r)} for c, r in zip(calls, results)]


def compact_result(r, cap=600):
    """A tool result as kept in history: the answer without the raw search hits, capped. Enough for the model to
    see that it called a tool and what came back, so later turns keep calling tools instead of answering live
    questions from memory (I-12)."""
    if isinstance(r, dict):
        r = {k: v for k, v in r.items() if k not in ("results", "sources")}
    txt = json.dumps(r if r is not None else {"error": "no result"}, ensure_ascii=False)
    return txt if len(txt) <= cap else txt[:cap] + "…"
