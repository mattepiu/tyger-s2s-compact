"use client"
// tyger's page: talk, and see what it's doing. The conversation (lib/conversation.ts) is the classic page's
// (static/index.html), its protocol unchanged; this page shows it differently: a full-screen animation per phase
// (lib/talk.ts), the reply as text with the sources it cited, the characters, one mic button. The classic page, with
// the turn-taking meters and the live tuning, is one link away.
import { useCallback, useEffect, useReducer, useRef, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { AnimatedText } from "@/components/animated-text"
import { Backdrop, type AnimationName } from "@/components/backdrop"
import { Characters, MicButton, type Character } from "@/components/controls"
import { History, ReplyCard } from "@/components/replies"
import { Conversation } from "@/lib/conversation"
import { SHAPES } from "@/lib/shapes"
import { initial, phaseOf, reduce, type Phase } from "@/lib/talk"

const LOOKS: Record<Phase, { animation: AnimationName; label: string; dim?: boolean }> = {
  off: { animation: "think", label: "", dim: true },             // "Talk to <name>"
  connecting: { animation: "think", label: "Connecting", dim: true },
  listening: { animation: "think", label: "Listening" },
  thinking: { animation: "analyze", label: "Thinking" },
  lookup: { animation: "deep", label: "Looking it up" },         // a tool or the expert, after a filler line
  speaking: { animation: "transfer", label: "" },                // the character's name
  barge: { animation: "intense", label: "Go ahead" },
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
  const [level, setLevel] = useState(0)
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
      closed: (why) => { setLevel(0); dispatch({ kind: "closed", why }) },
      failed: (why) => { setLevel(0); dispatch({ kind: "closed", why }) },
      level: (peak) => setLevel(Math.min(1, peak * 3)),
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

  const pick = (v: string) => {
    setVoice(v)
    try { localStorage.setItem("tyger_voice", v) } catch {}
    conv.current?.send("voice:" + v)
  }
  const toggle = () => {
    if (talk.conn === "off") { dispatch({ kind: "connecting" }); conv.current?.start() }
    else if (talk.conn === "on") conv.current?.stop()
  }

  const phase = phaseOf(talk, performance.now())
  const look = LOOKS[phase]
  const characters: Character[] = voices.list.map((v, i) => ({ voice: v, name: nameOf(voices.labels[v], v), shape: SHAPES[i % SHAPES.length] }))
  const me = characters.find((c) => c.voice === voice)
  const name = me ? me.name : "tyger"
  const label = phase === "off" ? `Talk to ${name}` : phase === "speaking" ? name : look.label
  const reply = talk.conn === "on" && talk.shown ? talk.replies[talk.replies.length - 1] : null

  return (
    <main className="relative h-dvh w-full overflow-hidden text-white">
      <Backdrop name={look.animation} shape={me ? me.shape : "circle"} dim={!!look.dim} />

      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between gap-2 px-4 pt-[max(1rem,env(safe-area-inset-top))] sm:px-6">
        <span className="text-sm font-semibold tracking-wide text-white/80">tyger</span>
        <nav className="flex items-center gap-1 text-sm">
          <button onClick={() => setHistory(true)} className="rounded-full px-3 py-1.5 text-white/70 hover:bg-white/10 hover:text-white">
            History{talk.replies.length ? ` · ${talk.replies.length}` : ""}
          </button>
          <a href={CLASSIC} className="rounded-full px-3 py-1.5 text-white/70 hover:bg-white/10 hover:text-white">Classic view</a>
        </nav>
      </header>

      <section className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-6 px-4 pb-52 pt-20">
        <h1 className="text-center text-5xl font-bold drop-shadow-[0_2px_24px_rgba(0,0,0,0.35)] md:text-7xl">
          <AnimatedText key={label} text={label} />
        </h1>
        {phase === "off" && <p className="text-white/60">Tap the mic to start</p>}
        <AnimatePresence>
          {reply && (
            <motion.div key={talk.replies.length} className="pointer-events-auto flex w-full justify-center" aria-live="polite"
                        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.35 }}>
              <ReplyCard reply={reply} live={talk.open} />
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      <footer className="absolute inset-x-0 bottom-0 z-20 flex flex-col items-center gap-4 px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
        <Characters list={characters} voice={voice} onPick={pick} />
        <MicButton conn={talk.conn} level={level} onClick={toggle} />
        <p className="min-h-5 max-w-md text-center text-sm text-white/65" role="status">{talk.note}</p>
      </footer>

      <History open={history} replies={talk.replies} name={name} onClose={() => setHistory(false)} />
    </main>
  )
}
