"use client"
// The voice glow (voice-glow's VoiceBeam), a layer behind the page along the bottom of the screen. It rises with whoever
// is talking: the user at the mic, or the agent as its reply plays. While the agent is thinking or looking something up,
// it gathers into one beam that sweeps side to side. That sweep is our own, on the package's `motion` prop: its
// ready-made "processing" state is a paid add-on. Silent, it rests on the package's own breathing (`idle`).
import { useCallback, useEffect, useRef, useState } from "react"
import { VoiceBeam, type VoiceBeamMotion } from "voice-glow"

const GAIN = 5 * 3.1      // RMS -> level as voice-glow scales its own mic input (x5, sensitivity 3.1)
const IDLE = 0.4          // the resting presence (the package's default 0.23 is barely visible on the first screen)
const EASE_S = 0.3        // the beam gathers and lets go in about this long
const SWEEP = 2.2         // rad/s: a round trip in ~2.9 s

// The glow's size, known only in the browser: voice-glow's "mobile" preset is tuned for a phone's width, wider
// screens scale it up. null until measured, so the glow never shows at a wrong size first (its stylesheet is
// generated from the scale: a change redraws it at once).
function useScale() {
  const [scale, setScale] = useState<number | null>(null)
  useEffect(() => {
    const fit = () => setScale(Math.round(Math.min(2.4, Math.max(1, window.innerWidth / 430)) * 10) / 10)
    fit(); window.addEventListener("resize", fit)
    return () => window.removeEventListener("resize", fit)
  }, [])
  return scale
}

// The thinking sweep, read by voice-glow every frame (no re-render): gathered while `on`, eased in and out.
function useSweep(on: boolean) {
  const st = useRef({ on, gather: 0, t: 0, last: 0 })
  st.current.on = on
  return useCallback((): VoiceBeamMotion => {
    const s = st.current, now = performance.now() / 1000
    const dt = s.last ? Math.min(0.05, now - s.last) : 0
    s.last = now
    s.gather += ((s.on ? 1 : 0) - s.gather) * (1 - Math.exp(-dt / EASE_S))
    if (s.gather > 0.01) s.t += dt; else s.t = 0
    const g = s.gather
    return { gather: g, offset: 0.7 * Math.sin(s.t * SWEEP) * g, stretch: 0.35 * Math.abs(Math.cos(s.t * SWEEP)) * g,
             heldLevel: 0.5 * g }
  }, [])
}

export function Glow({ meter, thinking }: { meter: () => { mic: number; out: number }; thinking: boolean }) {
  const scale = useScale()
  const motion = useSweep(thinking)
  const level = useCallback(() => { const m = meter(); return Math.min(1, Math.max(m.mic, m.out) * GAIN) }, [meter])
  if (scale === null) return null
  // a fixed layer behind the page: inline, as the package's own CSS sets its wrapper `position: relative`
  return (
    <VoiceBeam type="mobile" theme="dark" scale={scale} idle={IDLE} level={level} motion={motion}
               style={{ position: "fixed", inset: 0, pointerEvents: "none" }} aria-hidden>
      <div className="h-dvh w-full" />
    </VoiceBeam>
  )
}
