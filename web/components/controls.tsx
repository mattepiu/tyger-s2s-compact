"use client"
// The bottom controls: the character tabs (the template's shape tabs: each character draws its own shape) and the mic.
import { GLYPHS, type ShapeType } from "@/lib/shapes"

export type Character = { voice: string; name: string; shape: ShapeType }

export function Characters({ list, voice, onPick }: { list: Character[]; voice: string; onPick: (v: string) => void }) {
  if (list.length < 2) return null
  return (
    <div role="tablist" aria-label="Character"
         className="inline-flex max-w-full flex-wrap justify-center gap-1 rounded-full border border-white/10 bg-white/5 p-1 backdrop-blur-xl">
      {list.map((c) => (
        <button key={c.voice} role="tab" aria-selected={c.voice === voice} onClick={() => onPick(c.voice)}
                className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                  c.voice === voice ? "bg-white text-black" : "text-white/75 hover:text-white"}`}>
          <span className="mr-1.5 text-xs" aria-hidden>{GLYPHS[c.shape]}</span>{c.name}
        </button>
      ))}
    </div>
  )
}

export function MicButton({ conn, level, onClick }: { conn: "off" | "connecting" | "on"; level: number; onClick: () => void }) {
  const on = conn !== "off"
  return (
    <button onClick={onClick} aria-label={on ? "Stop" : "Start talking"}
            className={`grid size-18 place-items-center rounded-full transition-[transform,background-color,box-shadow] duration-150 ${
              on ? "border border-white/25 bg-white/10 text-white backdrop-blur-xl"
                 : "bg-white text-black shadow-[0_0_40px_rgba(255,255,255,0.25)] hover:scale-105"} ${
              conn === "connecting" ? "animate-pulse" : ""}`}
            style={conn === "on" ? { boxShadow: `0 0 0 ${Math.round(level * 22)}px rgba(255,255,255,0.14)` } : undefined}>
      {on ? <StopIcon /> : <MicIcon />}
    </button>
  )
}

function MicIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
         strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8" />
    </svg>
  )
}

function StopIcon() {
  return <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden><rect x="5" y="5" width="14" height="14" rx="3" fill="currentColor" /></svg>
}
