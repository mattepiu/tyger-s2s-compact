"use client"
// The backdrop: ASCII art after the template's animation states, over each state's colours. One <pre> of characters,
// redrawn ~20 times a second from lib/fields.ts; a change of pattern blends the old one into the new. Think (listening)
// follows the user's voice at the mic, Transfer (speaking) the agent's as it plays.
import { useEffect, useRef } from "react"
import { BLEND_S, fill, PATTERNS, RAMP, type Pattern } from "@/lib/fields"

// The template's colours for each state (its gradients, without the blur)
const TINTS: Record<Pattern, [string, string]> = {
  analyze: ["bg-gradient-to-br from-purple-500/30 via-pink-500/40 to-indigo-900/50", "bg-gradient-to-tr from-indigo-950/60 via-transparent to-purple-900/40"],
  think: ["bg-gradient-to-br from-teal-400/30 via-cyan-500/40 to-blue-900/50", "bg-gradient-to-tr from-slate-950/60 via-transparent to-teal-800/40"],
  connect: ["bg-gradient-to-br from-emerald-400/30 via-green-500/40 to-slate-900/50", "bg-gradient-to-tr from-slate-950/60 via-transparent to-emerald-800/40"],
  deep: ["bg-gradient-to-br from-indigo-950/40 via-purple-950/50 to-violet-950/60", "bg-gradient-to-tr from-black/80 via-transparent to-indigo-950/30"],
  transfer: ["bg-gradient-to-br from-amber-400/30 via-orange-500/40 to-slate-900/50", "bg-gradient-to-tr from-slate-950/60 via-transparent to-amber-800/40"],
  intense: ["bg-gradient-to-br from-red-500/30 via-orange-500/40 to-slate-900/50", "bg-gradient-to-tr from-slate-950/60 via-transparent to-red-800/40"],
}
const FPS = 20                // 4 with prefers-reduced-motion
const CALM = "radial-gradient(ellipse 40% 34% at 50% 42%, rgba(0,0,0,0.4), #000)"   // fainter behind the text
const GAIN = 5 * 3.1          // voice RMS -> 0..1

export function Backdrop({ pattern, meter }: { pattern: Pattern; meter: () => { mic: number; out: number } }) {
  const pre = useRef<HTMLPreElement>(null)
  const want = useRef(pattern), voices = useRef(meter)
  want.current = pattern; voices.current = meter

  useEffect(() => {             // one drawing loop for the page's life: the pattern and the voices are read from refs
    const el = pre.current!
    let grid = { cols: 0, rows: 0, aspect: 2 }, now = new Float32Array(0), old = now
    let cur = want.current, prev = cur, since = -Infinity, voice = 0, last = 0, raf = 0
    const measure = () => {     // the grid: as many character cells as cover the window
      const probe = document.createElement("span"); probe.textContent = "M".repeat(40)
      el.appendChild(probe); const box = probe.getBoundingClientRect(); el.removeChild(probe)
      const w = box.width / 40 || 7, h = parseFloat(getComputedStyle(el).lineHeight) || 12   // a span's box is taller
      grid = { cols: Math.ceil(window.innerWidth / w), rows: Math.ceil(window.innerHeight / h), aspect: h / w }
      now = new Float32Array(grid.cols * grid.rows); old = new Float32Array(grid.cols * grid.rows)
    }
    const slow = matchMedia("(prefers-reduced-motion: reduce)").matches
    const frame = (ms: number) => {
      raf = requestAnimationFrame(frame)
      if (ms - last < 1000 / (slow ? 4 : FPS)) return
      last = ms
      const t = ms / 1000
      if (want.current !== cur) { prev = cur; cur = want.current; since = t }
      const m = voices.current(), level = Math.min(1, GAIN * (cur === "transfer" ? m.out : m.mic))
      voice += (level - voice) * (level > voice ? 0.5 : 0.12)              // quick to rise, slow to settle
      fill(cur, grid, t, voice, now)
      const k = (t - since) / BLEND_S
      if (k < 1) {
        fill(prev, grid, t, voice, old)
        for (let c = 0; c < now.length; c++) now[c] = old[c] + (now[c] - old[c]) * k
      }
      const top = RAMP.length - 1
      let text = ""
      for (let j = 0, c = 0; j < grid.rows; j++) {
        for (let i = 0; i < grid.cols; i++, c++) text += RAMP[Math.round(now[c] * top)]
        text += "\n"
      }
      el.textContent = text
    }
    measure()
    window.addEventListener("resize", measure)
    raf = requestAnimationFrame(frame)
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", measure) }
  }, [])

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#05060a]" aria-hidden>
      {PATTERNS.map((p) => (
        <div key={p} className={`absolute inset-0 transition-opacity duration-700 ${p === pattern ? "opacity-100" : "opacity-0"}`}>
          <div className={`absolute inset-0 ${TINTS[p][0]}`} />
          <div className={`absolute inset-0 ${TINTS[p][1]}`} />
        </div>
      ))}
      <pre ref={pre} style={{ maskImage: CALM, WebkitMaskImage: CALM }}
           className="absolute inset-0 m-0 select-none overflow-hidden whitespace-pre font-mono text-[12px] leading-[12px] text-white/70" />
    </div>
  )
}
