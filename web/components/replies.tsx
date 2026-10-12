"use client"
// A reply as text (what the voice says), with the sources it cited as links (never spoken); and the history drawer.
import { AnimatePresence, motion } from "framer-motion"
import type { Link, Reply } from "@/lib/talk"

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

export function ReplyCard({ reply, live }: { reply: Reply; live: boolean }) {
  return (
    <div className={`w-full max-w-2xl rounded-2xl border border-white/10 bg-black/35 px-5 py-4 backdrop-blur-xl transition-opacity duration-500 ${
           live ? "opacity-100" : "opacity-70"}`}>
      {reply.task !== null && <div className="mb-1 text-xs text-white/55">Task #{reply.task} done</div>}
      <p className="max-h-[30vh] overflow-y-auto text-lg leading-relaxed text-white/95 md:text-xl">{reply.text}</p>
      <Sources links={reply.links} />
    </div>
  )
}

export function History({ open, replies, name, onClose }: { open: boolean; replies: Reply[]; name: string; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.aside key="history" initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
                      transition={{ type: "tween", duration: 0.25 }} aria-label="What tyger said"
                      className="fixed inset-y-0 right-0 z-30 flex w-full flex-col border-l border-white/10 bg-black/70 backdrop-blur-2xl sm:w-[26rem]">
          <div className="flex items-center justify-between px-5 pb-3 pt-[max(1.25rem,env(safe-area-inset-top))]">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-white/60">What {name} said</h2>
            <button onClick={onClose} className="rounded-full px-3 py-1 text-sm text-white/70 hover:bg-white/10 hover:text-white">Close</button>
          </div>
          <div className="flex-1 space-y-3 overflow-y-auto px-5 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
            {replies.length === 0 && <p className="text-sm italic text-white/50">Nothing yet.</p>}
            {[...replies].reverse().map((r, i) => (
              <div key={replies.length - i} className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                {r.task !== null && <div className="mb-1 text-xs text-white/55">Task #{r.task} done</div>}
                <p className="text-sm leading-relaxed text-white/90">{r.text}</p>
                <Sources links={r.links} />
              </div>
            ))}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
