"""One conversation's memory, as the reply model sees it: a message list per turn (the user's words, any tool round
with compacted results, the reply), trimmed by whole turns so a tool call never loses its result. In memory only."""
from __future__ import annotations


class History:
    def __init__(self, max_turns, prompts):
        self.turns: list[list[dict]] = []
        self.max_turns = int(max_turns)
        self.p = prompts                      # the reply model's Prompts: the history markers

    @property
    def messages(self):
        return [msg for turn in self.turns for msg in turn]

    def add(self, user, reply, tool_msgs=()):
        """One turn: the user's words, the tool call + compact results if any, the reply. A turn with no words from
        the user isn't kept."""
        if user.strip():
            self.turns.append([{"role": "user", "content": user.strip()}, *tool_msgs,
                               {"role": "assistant", "content": reply or self.p.text("markers", "empty_reply")}])
            self.turns = self.turns[-self.max_turns:]

    def add_barged(self, user, heard, tool_msgs=()):
        """A barged turn still counts: the user's words and the reply lines they `heard`, marked interrupted (the
        system prompt explains the marker)."""
        mark = self.p.text("markers", "interrupted")
        self.add(user, (" ".join(t.strip() for t in heard) + mark).strip(), tool_msgs)

    def recent(self, n):
        """The last `n` turns (none if n <= 0): the context sent along to the smart model."""
        return self.turns[-n:] if n > 0 else []

    def user_words(self):
        """The user's earlier messages (MOUTH's bracketed notes left out), oldest first."""
        return [x["content"] for x in self.messages if x["role"] == "user" and not x["content"].startswith("[")]
