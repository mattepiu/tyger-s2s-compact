"use client"
// The orb in the middle of the page: OrbKit's Ion (from the Agents Kit registry, MIT; the files and their notices are
// in components/voice-agents/orbkit). Its three states follow the conversation. While the user talks their voice drives
// its input, while the agent talks the agent's voice drives its output; otherwise it makes up its own motion.
import { useEffect, useState } from "react"
import type { OrbState } from "@/components/voice-agents/orbkit/core"
import { Shdr13 } from "@/components/voice-agents/orbkit/shdr-13"
import type { Phase } from "@/lib/talk"

const STATES: Record<Phase, OrbState> = {
  off: "idle", connecting: "idle", listening: "idle", barge: "idle",
  thinking: "thinking", lookup: "thinking",
  speaking: "speaking",
}
const GAIN = 5 * 3.1          // voice RMS -> 0..1
const SIZE = 300              // px, at full size
const SMALL = 0.5             // its scale while text is on screen

type Volumes = { input?: number; output?: number }

// Whoever is talking, how loud: polled ~15 times a second while it matters (the orb eases between values itself).
function useVolumes(phase: Phase, meter: () => { mic: number; out: number }) {
  const [vol, setVol] = useState<Volumes | undefined>(undefined)
  const agent = phase === "speaking", user = phase === "listening" || phase === "barge"
  useEffect(() => {
    if (!agent && !user) { setVol(undefined); return }
    const t = setInterval(() => {
      const m = meter()
      setVol(agent ? { output: Math.min(1, m.out * GAIN) } : { input: Math.min(1, m.mic * GAIN) })
    }, 66)
    return () => clearInterval(t)
  }, [agent, user, meter])
  return vol
}

export function Orb({ phase, meter, small }: { phase: Phase; meter: () => { mic: number; out: number }; small: boolean }) {
  const volumes = useVolumes(phase, meter)
  const k = small ? SMALL : 1
  return (   // the box shrinks with the orb, so the text below moves up with it
    <div className="flex justify-center transition-[height] duration-500 ease-out" style={{ height: SIZE * k }}>
      <div className="origin-top transition-transform duration-500 ease-out" style={{ transform: `scale(${k})` }}>
        <Shdr13 size={SIZE} state={STATES[phase]} volumes={volumes} ariaLabel={`tyger is ${STATES[phase]}`} />
      </div>
    </div>
  )
}
