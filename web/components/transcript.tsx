"use client"
// The conversation as text: the exchange on screen (the user's words, the agent's backchannels while they spoke, its
// reply with the sources it cited, never spoken), and the whole of it in the history drawer.
import { useEffect, useRef } from "react"
import { AnimatePresence, motion } from "framer-motion"
import type { Link, Said } from "@/lib/talk"

function Sources({ links }: { links: Link[] }) {
  if (!links.length) return null
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {links.map((l) => (
        <a key={l.url} href={l.url} target="_blank" rel="noopener noreferrer" title={l.url}
           className="max-w-full truncate rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs text-white/85 hover:bg-white/20">
          {l.label} <span aria-hidden>↗</span>
        </a>
      ))}
    </div>
  )
}

// The agent's backchannels while the user spoke ("Ori: Mm-hm."), under the user's words.
function Acks({ acks, name }: { acks: string[]; name: string }) {
  if (!acks.length) return null
  return <div className="px-1 text-xs text-white/55"><span className="font-medium">{name}:</span> {acks.join(" ")}</div>
}

const TAIL = 100      // characters of a turn shown while it's spoken: its newest words, within the bubble's 3 lines

function tailOf(text: string) {
  if (text.length <= TAIL) return text
  const cut = text.slice(-TAIL), sp = cut.indexOf(" ")
  return "…" + (sp >= 0 ? cut.slice(sp + 1) : cut)
}

function UserLine({ text, acks, name, small, live }: {
  text: string; acks: string[]; name: string; small?: boolean; live?: boolean
}) {
  return (
    <div className="flex flex-col items-end gap-1">
      <div className={`max-w-[85%] rounded-2xl rounded-br-md bg-white/12 text-white/90 backdrop-blur-xl ${
             small ? "px-3 py-2 text-sm" : "line-clamp-3 px-4 py-2.5 text-base md:text-lg"}`}>
        {(live ? tailOf(text) : text) || <span className="animate-pulse" aria-label="speaking">…</span>}
      </div>
      <Acks acks={acks} name={name} />
    </div>
  )
}

function AgentLine({ said, live, small }: { said: Extract<Said, { who: "agent" }>; live?: boolean; small?: boolean }) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-black/35 backdrop-blur-xl transition-opacity duration-500 ${
           small ? "px-4 py-3" : "px-5 py-4"} ${live || small ? "opacity-100" : "opacity-75"}`}>
      {said.task !== null && <div className="mb-1 text-xs text-white/55">Task #{said.task} done</div>}
      <p className={small ? "text-sm leading-relaxed text-white/90"
                          : "max-h-[26vh] overflow-y-auto text-lg leading-relaxed text-white/95 md:text-xl"}>
        {said.text}{said.cut && <span className="text-white/45"> (cut off)</span>}
      </p>
      <Sources links={said.links} />
    </div>
  )
}

// The exchange on screen: `items` (the log from the user's words on), then, while the user is still speaking, their
// turn so far: their words so far ("…" before any) with the backchannels it got. Keys are log positions, so the live
// turn becomes its words in place.
export function Exchange({ items, from, words, acks, open, name }: {
  items: Said[]; from: number; words: string; acks: string[]; open: boolean; name: string
}) {
  const live = words !== "" || acks.length > 0
  return (
    <div className="flex w-full max-w-2xl flex-col gap-3" aria-live="polite">
      <AnimatePresence initial={false}>
        {items.map((it, i) => (
          <motion.div key={from + i} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}>
            {it.who === "user" ? <UserLine text={it.text} acks={it.acks} name={name} />
                               : <AgentLine said={it} live={open && i === items.length - 1} />}
          </motion.div>
        ))}
        {live && (
          <motion.div key={from + items.length} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <UserLine text={words} acks={acks} name={name} live />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function History({ open, log, name, onClose }: { open: boolean; log: Said[]; name: string; onClose: () => void }) {
  const list = useRef<HTMLDivElement>(null)
  useEffect(() => { if (open && list.current) list.current.scrollTop = list.current.scrollHeight }, [open, log])  // the newest
  return (
    <AnimatePresence>
      {open && (
        <motion.aside key="history" initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
                      transition={{ type: "tween", duration: 0.25 }} aria-label="The conversation"
                      className="fixed inset-y-0 right-0 z-30 flex w-full flex-col border-l border-white/10 bg-black/75 backdrop-blur-2xl sm:w-[28rem]">
          <div className="flex items-center justify-between px-5 pb-3 pt-[max(1.25rem,env(safe-area-inset-top))]">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-white/60">Conversation</h2>
            <button onClick={onClose} className="rounded-full px-3 py-1 text-sm text-white/70 hover:bg-white/10 hover:text-white">Close</button>
          </div>
          <div ref={list} className="flex-1 space-y-3 overflow-y-auto px-5 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
            {log.length === 0 && <p className="text-sm italic text-white/50">Nothing yet.</p>}
            {log.map((it, i) => (
              <div key={i}>
                <div className={`mb-1 px-1 text-[11px] uppercase tracking-wider text-white/40 ${it.who === "user" ? "text-right" : ""}`}>
                  {it.who === "user" ? "You" : name}
                </div>
                {it.who === "user" ? <UserLine text={it.text} acks={it.acks} name={name} small /> : <AgentLine said={it} small />}
              </div>
            ))}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
