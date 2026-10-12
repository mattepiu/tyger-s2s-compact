"use client"
// tyger's page: talk, and see what it's doing. The conversation (lib/conversation.ts) is the classic page's
// (static/index.html), its protocol unchanged; this page shows it differently: an orb in the middle whose state follows
// the conversation (lib/talk.ts has the phases), the phase as a word, the exchange as text (the user's words, the
// agent's backchannels and reply, the sources it cited) and all of it in the history, the characters, one mic button.
// The classic page, with the turn-taking meters and the live tuning, is one link away.
import { useCallback, useEffect, useReducer, useRef, useState } from "react"
import { AnimatedText } from "@/components/animated-text"
import { Characters, MicButton, type Character } from "@/components/controls"
import { Orb } from "@/components/orb"
import { Exchange, History } from "@/components/transcript"
import { Conversation } from "@/lib/conversation"
import { ackShown, initial, phaseOf, reduce, type Phase } from "@/lib/talk"

const LABELS: Record<Phase, string> = {
  off: "",                     // "Talk to <name>"
  connecting: "Connecting",
  listening: "Listening",
  thinking: "Thinking",
  lookup: "Looking it up",     // a tool or the expert, after a filler line
  speaking: "",                // the character's name
  barge: "Go ahead",
}
const CLASSIC = process.env.NEXT_PUBLIC_CLASSIC || "/"

type Voices = { list: string[]; labels: Record<string, string> }
const nameOf = (label: string | undefined, voice: string) => (label || voice).replace(/\s*\([^)]*\)$/, "")  // "Ori (ori)"

function useTicks(on: boolean) {              // re-render 10 times a second while talking: the phases have timers
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!on) return
    const t = setInterval(() => setTick((n) => n + 1), 100)
    return () => clearInterval(t)
  }, [on])
}

export default function Page() {
  const [talk, dispatch] = useReducer(reduce, initial)
  const [voices, setVoices] = useState<Voices>({ list: [], labels: {} })
  const [voice, setVoice] = useState("")
  const [history, setHistory] = useState(false)
  const conv = useRef<Conversation | null>(null)
  useTicks(talk.conn === "on")

  // The picker: the saved pick if it's still offered, else EARS's default. Returns the voice picked.
  const populate = useCallback((list: string[], cur: string | null, labels: Record<string, string>) => {
    if (!list || !list.length) return ""
    let saved: string | null = null
    try { saved = localStorage.getItem("tyger_voice") } catch {}
    const want = saved && list.includes(saved) ? saved : cur || list[0]
    setVoices({ list, labels: labels || {} }); setVoice(want)
    return want
  }, [])

  useEffect(() => {             // the picker before Start, so a pick is shown and remembered without a conversation
    fetch("/voices").then((r) => r.json()).then((d) => populate(d.voices, d.voice, d.labels)).catch(() => {})
  }, [populate])

  useEffect(() => {
    const c = conv.current = new Conversation({
      opened: () => dispatch({ kind: "opened" }),
      closed: (why) => dispatch({ kind: "closed", why }),
      failed: (why) => dispatch({ kind: "closed", why }),
      playing: (on) => dispatch({ kind: "playing", on, now: performance.now() }),
      message: (m) => {
        // always send the picked voice at session start: MOUTH keeps its voice across sessions, so EARS's default
        // (m.voice) could leave it on another voice than the one shown
        if (m.type === "ready") { const v = populate(m.voices, m.voice, m.labels); if (v) c.send("voice:" + v) }
        dispatch({ kind: "message", m, now: performance.now() })
      },
    })
    return () => c.stop()
  }, [populate])

  const meter = useCallback(() => conv.current ? conv.current.meter() : { mic: 0, out: 0 }, [])
  const pick = (v: string) => {
    setVoice(v)
    try { localStorage.setItem("tyger_voice", v) } catch {}
    conv.current?.send("voice:" + v)
  }
  const toggle = () => {
    if (talk.conn === "off") { dispatch({ kind: "connecting" }); conv.current?.start() }
    else if (talk.conn === "on") conv.current?.stop()
  }

  const now = performance.now()
  const phase = phaseOf(talk, now)
  const characters: Character[] = voices.list.map((v) => ({ voice: v, name: nameOf(voices.labels[v], v) }))
  const me = characters.find((c) => c.voice === voice)
  const name = me ? me.name : "tyger"
  const label = phase === "off" ? `Talk to ${name}` : phase === "speaking" ? name : LABELS[phase]
  const shown = talk.conn === "on" ? talk.log.slice(talk.from) : []
  const exchange = shown.length > 0 || (talk.conn === "on" && (talk.words !== "" || talk.acks.length > 0))

  return (
    <>
      <main className="relative z-10 h-dvh w-full overflow-hidden text-white">
        <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between gap-2 px-4 pt-[max(1rem,env(safe-area-inset-top))] sm:px-6">
          <span className="text-sm font-semibold tracking-wide text-white/80">tyger</span>
          <nav className="flex items-center gap-1 text-sm">
            <button onClick={() => setHistory(true)} className="rounded-full px-3 py-1.5 text-white/70 hover:bg-white/10 hover:text-white">
              History{talk.log.length ? ` · ${talk.log.length}` : ""}
            </button>
            <a href={CLASSIC} className="rounded-full px-3 py-1.5 text-white/70 hover:bg-white/10 hover:text-white">Classic view</a>
          </nav>
        </header>

        <section className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center-safe gap-4 px-4 pb-56 pt-16">
          <Orb phase={phase} voice={voice} meter={meter} small={exchange} />
          <h1 className={`text-center font-bold transition-[font-size] duration-300 ${
                exchange ? "text-2xl md:text-4xl" : "text-4xl md:text-6xl"}`}>
            <AnimatedText key={label} text={label} />
          </h1>
          {phase === "off" && <p className="text-white/60">Tap the mic to start</p>}
          {exchange && (
            <div className="pointer-events-auto flex w-full justify-center">
              <Exchange items={shown} from={talk.from} words={talk.words} acks={talk.acks} ack={ackShown(talk, now)}
                        open={talk.open} />
            </div>
          )}
        </section>

        <footer className="absolute inset-x-0 bottom-0 z-20 flex flex-col items-center gap-4 px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
          <Characters list={characters} voice={voice} onPick={pick} />
          <MicButton conn={talk.conn} onClick={toggle} />
          <p className="min-h-5 max-w-md text-center text-sm text-white/65" role="status">{talk.note}</p>
        </footer>
      </main>
      <History open={history} log={talk.log} name={name} onClose={() => setHistory(false)} />
    </>
  )
}
