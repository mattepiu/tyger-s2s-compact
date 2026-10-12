// What the page shows, from the conversation's events: a pure reducer (every event comes with the time it came, ms on
// performance.now()) and the phase the animation and the label follow.
import type { Message } from "./conversation"

export type Link = { label: string; url: string }
export type Reply = { text: string; links: Link[]; task: number | null }
export type Phase = "off" | "connecting" | "listening" | "thinking" | "lookup" | "speaking" | "barge"

export type Talk = {
  conn: "off" | "connecting" | "on"
  note: string            // why the conversation ended or couldn't start, or what's missing (no MOUTH)
  turn: boolean           // the agent's turn: from an endpoint (or a task speaking up) to turn_done or a barge
  turnAt: number          // when it began
  speakAt: number         // EARS's last `state` SPEAK: every hop while the agent has the floor (the model's own state
                          // head can say SPEAK too, so it only confirms a turn, never starts one)
  bargeAt: number         // when the user last cut in
  playing: boolean        // reply audio is playing
  quietSince: number      // when reply audio last stopped
  spoke: boolean          // reply audio has played in this turn
  replies: Reply[]        // the replies so far, oldest first (kept across Stop / Start)
  open: boolean           // the last reply is still being said: `reply` updates its text, `links` add to it
  shown: boolean          // the last reply is on screen: from its first text to the user's next turn
  task: number | null     // the background task whose result is said next
}

export const initial: Talk = {
  conn: "off", note: "", turn: false, turnAt: -Infinity, speakAt: -Infinity, bargeAt: -Infinity, playing: false,
  quietSince: -Infinity, spoke: false, replies: [], open: false, shown: false, task: null,
}

export type Action =
  | { kind: "connecting" }
  | { kind: "opened" }
  | { kind: "closed"; why: string }
  | { kind: "message"; m: Message; now: number }
  | { kind: "playing"; on: boolean; now: number }

export function reduce(s: Talk, a: Action): Talk {
  switch (a.kind) {
    case "connecting": return { ...initial, conn: "connecting", replies: s.replies }
    case "opened": return { ...s, conn: "on" }
    case "closed": return { ...initial, replies: s.replies, note: a.why }
    case "playing": return a.on ? { ...s, playing: true, spoke: true } : { ...s, playing: false, quietSince: a.now }
    case "message": return onMessage(s, a.m, a.now)
  }
}

function onMessage(s: Talk, m: Message, now: number): Talk {
  switch (m.type) {
    case "ready":
      return { ...s, note: m.mouth ? "" : "Ears only: MOUTH isn't running, so tyger can't answer." }
    case "state": return m.state === "SPEAK" ? { ...s, speakAt: now } : s
    case "endpoint": return { ...s, turn: true, turnAt: now, spoke: false, open: false, shown: false }
    case "reply": {
      const replies = s.open ? [...s.replies.slice(0, -1), { ...s.replies[s.replies.length - 1], text: m.text }]
                             : [...s.replies, { text: m.text, links: [], task: s.task }]
      return { ...s, replies, open: true, shown: true }
    }
    case "links": {                // the sources a reply cited (the expert's web search): shown, never spoken
      if (!s.open) return s
      const last = s.replies[s.replies.length - 1], links = [...last.links]
      for (const l of m.links || [])
        if (/^https?:\/\//i.test(l.url || "") && !links.some((x) => x.url === l.url)) links.push({ label: l.label || l.url, url: l.url })
      return { ...s, replies: [...s.replies.slice(0, -1), { ...last, links }] }
    }
    case "turn_done": {
      // after a barge the user may have finished speaking before MOUTH halted: EARS sends that turn right after this
      const pending = !!m.barged && s.turnAt > s.bargeAt
      return { ...s, turn: pending, turnAt: pending ? now : s.turnAt, open: false, task: null, spoke: false }
    }
    case "task_done": return { ...s, turn: true, turnAt: now, spoke: false, open: false, task: m.task }   // speaks up
    case "barge": return { ...s, turn: false, bargeAt: now }
    default: return s
  }
}

export const ENDPOINT_GRACE_MS = 1000  // EARS says SPEAK a hop or two after a turn goes to MOUTH (never, if nothing was heard)
export const SPEAK_STALE_MS = 600      // no SPEAK this long in a turn: EARS isn't giving the agent the floor
export const LOOKUP_AFTER_MS = 700     // the agent's turn, its voice quiet this long after it spoke: a tool, the expert
export const BARGE_FLASH_MS = 900

export function phaseOf(s: Talk, now: number): Phase {
  if (s.conn !== "on") return s.conn
  if (now - s.bargeAt < BARGE_FLASH_MS) return "barge"
  if (s.playing) return "speaking"
  const live = s.turn && (now - s.turnAt < ENDPOINT_GRACE_MS || now - s.speakAt < SPEAK_STALE_MS)
  if (!live) return "listening"
  if (!s.spoke) return "thinking"
  return now - s.quietSince < LOOKUP_AFTER_MS ? "speaking" : "lookup"
}
