"use client"
// The live-tuning panel (lib/tuning.ts): the turn-taking heads' meters with their thresholds marked, and the plugin's
// sliders. It floats over the page without blocking it, so the user can tune while they talk.
import { useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { isOff, markOf, positionOf, shown, valueOf, type Meter, type Slider, type Tuning } from "@/lib/tuning"

const METERS: [Meter, string, string][] = [   // the classic page's names and colours
  ["p_end", "p_end", "bg-sky-400"], ["yield", "yield", "bg-red-400"], ["bc", "backchan", "bg-violet-400"],
]

function MeterRow({ label, value, mark, color }: { label: string; value: number; mark: number | null; color: string }) {
  const pct = (v: number) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(1)}%`
  return (
    <div className="grid grid-cols-[4.5rem_1fr_2.5rem] items-center gap-3 text-xs text-white/60">
      <span>{label}</span>
      <div className="relative h-2 overflow-hidden rounded-full bg-white/10">
        <div className={`absolute inset-y-0 left-0 ${color} transition-[width] duration-100`} style={{ width: pct(value) }} />
        {mark !== null && <div className="absolute -inset-y-1 w-0.5 bg-orange-400" style={{ left: pct(mark) }} />}
      </div>
      <span className="text-right tabular-nums">{value.toFixed(2)}</span>
    </div>
  )
}

// One threshold. A slider with an "off" value has a checkbox; while off, the slider keeps its place for when it's on.
function SliderRow({ s, value, onSet }: { s: Slider; value: number; onSet: (name: string, v: number) => void }) {
  const off = isOff(s, value)
  const [kept, setKept] = useState<number | null>(null)          // its place while off
  const pos = off ? (kept ?? s.lo) : positionOf(s, value)
  const toggle = (on: boolean) => {
    if (on) onSet(s.name, valueOf(s, kept ?? s.lo))
    else { setKept(pos); onSet(s.name, s.off as number) }
  }
  return (
    <div className="grid grid-cols-[7.5rem_1fr_3.5rem] items-center gap-3 text-xs">
      <label className="flex items-center gap-1.5 text-white/75">
        {s.off !== null && <input type="checkbox" checked={!off} onChange={(e) => toggle(e.target.checked)} className="accent-orange-400" />}
        {s.label}
      </label>
      <input type="range" min={s.lo} max={s.hi} step={s.step} value={pos} aria-label={s.label}
             onChange={(e) => off ? setKept(+e.target.value) : onSet(s.name, valueOf(s, +e.target.value))}
             className="w-full accent-orange-400" />
      <span className="text-right font-semibold tabular-nums text-orange-300">{shown(s, value)}</span>
    </div>
  )
}

export function TuningPanel({ open, tuning, onSet, onClose }: {
  open: boolean; tuning: Tuning; onSet: (name: string, v: number) => void; onClose: () => void
}) {
  return (
    <AnimatePresence>
      {open && (
        <motion.aside key="tuning" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.2 }} aria-label="Tuning"
                      className="fixed left-4 top-[max(3.5rem,calc(env(safe-area-inset-top)+2.5rem))] z-30 max-h-[calc(100dvh-9rem)] w-[min(24rem,calc(100vw-2rem))] overflow-y-auto rounded-2xl border border-white/10 bg-black/75 p-4 text-white shadow-2xl backdrop-blur-2xl">
          <div className="mb-1 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-white/60">Tuning</h2>
            <button onClick={onClose} className="rounded-full px-3 py-1 text-sm text-white/70 hover:bg-white/10 hover:text-white">Close</button>
          </div>
          {tuning.sliders.length === 0 ? (
            <p className="text-sm italic text-white/50">Start a conversation to tune its turn-taking.</p>
          ) : (
            <>
              <p className="mb-3 text-xs text-white/45">For this conversation only. Orange line: the threshold.</p>
              <div className="grid gap-2.5">
                {METERS.map(([m, label, color]) => (
                  <MeterRow key={m} label={label} value={tuning.meters[m]} mark={markOf(tuning, m)} color={color} />
                ))}
              </div>
              <div className="mt-4 grid gap-3 border-t border-white/10 pt-4">
                {tuning.sliders.map((s) => <SliderRow key={s.name} s={s} value={tuning.values[s.name] ?? valueOf(s, s.lo)} onSet={onSet} />)}
              </div>
            </>
          )}
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
