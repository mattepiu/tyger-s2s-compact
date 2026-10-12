// Live tuning, as on the classic page: the EARS plugin declares its tunable thresholds. `ready` carries their sliders
// (label, range and step in display units, display = value * scale, the meter they mark, an optional "off" value) and
// this conversation's values; `set:<name>:<value>` changes one (for this conversation only) and EARS echoes it back
// (`threshold`); every hop's `state` carries the meters (the heads' p_end, yield and backchannel probabilities).
// A reducer of its own over the same actions as talk.ts: the values are the conversation's, so they go with it.
import type { Action } from "./talk"

export type Meter = "p_end" | "yield" | "bc"
export type Slider = {
  name: string; type: "int" | "float"; label: string
  lo: number; hi: number; step: number; decimals: number; scale: number; unit: string
  meter: Meter | null     // gets a mark at the value
  off: number | null      // any value from this one on reads as off (a checkbox sets it)
}
export type Tuning = {
  sliders: Slider[]; values: Record<string, number>; meters: Record<Meter, number>
  sent: Record<string, number>   // the last value the user set, until EARS echoes it back
}

export const noTuning: Tuning = { sliders: [], values: {}, meters: { p_end: 0, yield: 0, bc: 0 }, sent: {} }

// The user moved a slider: shown at once, sent to EARS by the caller
export type TuneAction = Action | { kind: "set"; name: string; value: number }

export function tune(t: Tuning, a: TuneAction): Tuning {
  if (a.kind === "connecting" || a.kind === "closed") return noTuning
  if (a.kind === "set") return { ...t, values: { ...t.values, [a.name]: a.value }, sent: { ...t.sent, [a.name]: a.value } }
  if (a.kind !== "message") return t
  const m = a.m
  switch (m.type) {
    case "ready": return { ...t, sliders: m.sliders || [], values: { ...(m.thresholds || {}) } }
    case "threshold": {          // an echo; while a drag's are still coming back, only the last one counts
      if (m.attr in t.sent && t.sent[m.attr] !== m.value) return t
      const { [m.attr]: _, ...sent } = t.sent
      return { ...t, values: { ...t.values, [m.attr]: m.value }, sent }
    }
    case "state": return { ...t, meters: { p_end: m.p_end || 0, yield: m.yield || 0, bc: m.bc || 0 } }
    default: return t
  }
}

export const isOff = (s: Slider, v: number) => s.off !== null && v >= s.off
export const shown = (s: Slider, v: number) => isOff(s, v) ? "off" : (v * s.scale).toFixed(s.decimals) + s.unit
export const positionOf = (s: Slider, v: number) => v * s.scale          // value -> the slider's place
export function valueOf(s: Slider, pos: number) {                         // the slider's place -> value
  const v = pos / s.scale
  return s.type === "int" ? Math.round(v) : v
}

// Where a meter's threshold is (0..1): the value of the slider that marks it, or null (none, or off)
export function markOf(t: Tuning, meter: Meter): number | null {
  const s = t.sliders.find((x) => x.meter === meter), v = s ? t.values[s.name] : undefined
  return s && v !== undefined && !isOff(s, v) ? v : null
}
