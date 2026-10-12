// What the page shows, from the conversation's events: a pure reducer (every event comes with the time it came, ms on
// performance.now()) and the phase the animation and the label follow.
import type { Message } from "./conversation"

export type Link = { label: string; url: string }
// One line of the conversation: the user's words (the endpoint's) with the agent's backchannels while they spoke,
// or the agent's (a reply, or a background task's result) with the sources it cited and whether the user cut it off.
export type Said =
  | { who: "user"; text: string; acks: string[] }
  | { who: "agent"; text: string; links: Link[]; task: number | null; cut: boolean }
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
  log: Said[]             // the conversation so far, oldest first (kept across Stop / Start)
  open: boolean           // the agent's last line is still being said: `reply` updates its text, `links` add to it
  from: number            // the exchange on screen is log[from..]: from the user's words to the next turn
  words: string           // the user's words so far in their turn (EARS's partial; the whole of them come with the endpoint)
  acks: string[]          // backchannels in the user's turn so far
  ack: Ack | null         // the latest of them, shown for a moment beside the user's words
  task: number | null     // the background task whose result is said next
}
export type Ack = { text: string; at: number }

export const initial: Talk = {
  conn: "off", note: "", turn: false, turnAt: -Infinity, speakAt: -Infinity, bargeAt: -Infinity, playing: false,
  quietSince: -Infinity, spoke: false, log: [], open: false, from: 0, words: "", acks: [], ack: null, task: null,
}

// The user's turn has shown on screen already (its words so far, or a backchannel to them)
const userTurnShown = (s: Talk) => s.words !== "" || s.acks.length > 0

export type Action =
  | { kind: "connecting" }
  | { kind: "opened" }
  | { kind: "closed"; why: string }
  | { kind: "message"; m: Message; now: number }
  | { kind: "playing"; on: boolean; now: number }

export function reduce(s: Talk, a: Action): Talk {
  switch (a.kind) {
    case "connecting": return { ...initial, conn: "connecting", log: s.log, from: s.log.length }
    case "opened": return { ...s, conn: "on" }
    case "closed": return { ...initial, log: s.log, from: s.log.length, note: a.why }
    case "playing": return a.on ? { ...s, playing: true, spoke: true } : { ...s, playing: false, quietSince: a.now }
    case "message": return onMessage(s, a.m, a.now)
  }
}

function onMessage(s: Talk, m: Message, now: number): Talk {
  switch (m.type) {
    case "ready":
      return { ...s, note: m.mouth ? "" : "Ears only: MOUTH isn't running, so tyger can't answer." }
    case "state": return m.state === "SPEAK" ? { ...s, speakAt: now } : s
    case "endpoint": {             // the user's turn is over: their words start the exchange on screen (none heard: none)
      const text = (m.transcript || "").trim()
      const log: Said[] = text ? [...s.log, { who: "user", text, acks: s.acks }] : s.log
      return { ...s, turn: true, turnAt: now, spoke: false, open: false, log, from: text ? log.length - 1 : log.length,
               words: "", acks: [], ack: null }
    }
    case "partial":                // the user's words so far: their turn takes the screen
      return { ...s, words: m.text || "", from: userTurnShown(s) ? s.from : s.log.length }
    case "reply": {                // the text so far
      const last = s.log[s.log.length - 1]
      const log: Said[] = s.open && last.who === "agent" ? [...s.log.slice(0, -1), { ...last, text: m.text }]
                                  : [...s.log, { who: "agent", text: m.text, links: [], task: s.task, cut: false }]
      return { ...s, log, open: true }
    }
    case "links": {                // the sources a reply cited (the expert's web search): shown, never spoken
      const last = s.log[s.log.length - 1]
      if (!s.open || last.who !== "agent") return s
      const links = [...last.links]
      for (const l of m.links || [])
        if (/^https?:\/\//i.test(l.url || "") && !links.some((x) => x.url === l.url)) links.push({ label: l.label || l.url, url: l.url })
      return { ...s, log: [...s.log.slice(0, -1), { ...last, links }] }
    }
    case "backchannel":            // the agent acks while the user speaks: the user's new turn takes the screen
      return { ...s, acks: [...s.acks, m.text], ack: { text: m.text, at: now },
               from: userTurnShown(s) ? s.from : s.log.length }
    case "backchannel_done": {     // dropped unvoiced (the user's turn ended first): take it back, wherever it went
      if (!m.aborted) return s
      const drop = (acks: string[]) => { const i = acks.lastIndexOf(m.text); return i < 0 ? acks : acks.filter((_, k) => k !== i) }
      if (s.acks.includes(m.text)) return { ...s, acks: drop(s.acks), ack: s.ack?.text === m.text ? null : s.ack }
      const i = s.log.map((x) => x.who).lastIndexOf("user"), u = s.log[i]
      return u && u.who === "user" && u.acks.includes(m.text)
        ? { ...s, log: s.log.map((x, k) => (k === i ? { ...u, acks: drop(u.acks) } : x)) } : s
    }
    case "turn_done": {
      // after a barge the user may have finished speaking before MOUTH halted: EARS sends that turn right after this
      const pending = !!m.barged && s.turnAt > s.bargeAt
      return { ...s, turn: pending, turnAt: pending ? now : s.turnAt, open: false, task: null, spoke: false }
    }
    case "task_done":              // a background task speaks up: a new exchange
      return { ...s, turn: true, turnAt: now, spoke: false, open: false, task: m.task, from: s.log.length }
    case "barge": {                // the line being said is cut off (it stays open: its last text may still come)
      const last = s.log[s.log.length - 1]
      const log = s.open && last.who === "agent" ? [...s.log.slice(0, -1), { ...last, cut: true }] : s.log
      return { ...s, turn: false, bargeAt: now, log }
    }
    default: return s
  }
}

export const ENDPOINT_GRACE_MS = 1000  // EARS says SPEAK a hop or two after a turn goes to MOUTH (never, if nothing was heard)
export const SPEAK_STALE_MS = 600      // no SPEAK this long in a turn: EARS isn't giving the agent the floor
export const LOOKUP_AFTER_MS = 700     // the agent's turn, its voice quiet this long after it spoke: a tool, the expert
export const BARGE_FLASH_MS = 900
export const ACK_SHOW_MS = 2000        // a backchannel shows this long beside the user's words

export function ackShown(s: Talk, now: number): Ack | null {
  return s.ack && now - s.ack.at < ACK_SHOW_MS ? s.ack : null
}

export function phaseOf(s: Talk, now: number): Phase {
  if (s.conn !== "on") return s.conn
  if (now - s.bargeAt < BARGE_FLASH_MS) return "barge"
  if (s.playing) return "speaking"
  const live = s.turn && (now - s.turnAt < ENDPOINT_GRACE_MS || now - s.speakAt < SPEAK_STALE_MS)
  if (!live) return "listening"
  if (!s.spoke) return "thinking"
  return now - s.quietSince < LOOKUP_AFTER_MS ? "speaking" : "lookup"
}
