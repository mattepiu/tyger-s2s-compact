// The backdrop's patterns: ASCII-art versions of the template's animation states (ai-animation-states). fill() gives
// every character cell a brightness, 0..1, at time t; the backdrop draws it with RAMP. Pure (no DOM), so
// tests/page/web_check.mjs checks them.
export type Pattern = "analyze" | "think" | "connect" | "deep" | "transfer" | "intense"
export const PATTERNS: Pattern[] = ["analyze", "think", "connect", "deep", "transfer", "intense"]

export type Grid = { cols: number; rows: number; aspect: number }   // aspect: a cell's height / its width

export const RAMP = " .:-=+*#%@"          // darkest to brightest
export const BLEND_S = 0.6                // a change of pattern blends the old one into the new over this long

const TAU = Math.PI * 2

function hash(a: number, b: number, c: number) {    // integers -> [0, 1)
  let h = Math.imul(a, 374761393) ^ Math.imul(b, 668265263) ^ Math.imul(c, 1274126177)
  h = Math.imul(h ^ (h >>> 13), 1103515245)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

// Each cell's own slow flicker (the template's particles), 0..1: noise eased between random values ~`rate` per s.
function flicker(i: number, j: number, t: number, rate: number) {
  const x = t * rate + hash(i, j, 7) * 9, n = Math.floor(x), f = x - n, e = f * f * (3 - 2 * f)
  const a = hash(i, j, n), b = hash(i, j, n + 1)
  return a + (b - a) * e
}

const CENTRES = [[0.25, 0.3], [0.72, 0.38], [0.45, 0.72], [0.82, 0.75]]    // Intense's pulses, as fractions of the grid
const ring = (phase: number, sharp: number) => Math.pow(0.5 + 0.5 * Math.cos(TAU * phase), sharp)
const clamp = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

// Analyze's signals: a few lines running along the grid (across, then up or down) to a random cell, fading as they
// finish; where they are follows from t alone.
const SIGNALS = 8, SIGNAL_S = 2.4

function signals(g: Grid, t: number, out: Float32Array) {
  const { cols, rows } = g
  for (let k = 0; k < SIGNALS; k++) {
    const x = t / SIGNAL_S + k / SIGNALS, e = Math.floor(x), p = x - e
    const x0 = Math.floor(hash(k, e, 1) * cols), y0 = Math.floor(hash(k, e, 2) * rows)
    const x1 = Math.floor(hash(k, e, 3) * cols), y1 = Math.floor(hash(k, e, 4) * rows)
    const len = Math.abs(x1 - x0) + Math.abs(y1 - y0), head = Math.floor(p * 1.25 * len)   // done at 80%, then fades
    const fade = p < 0.8 ? 1 : (1 - p) / 0.2
    for (let s = 0; s <= Math.min(head, len); s++) {
      const across = Math.abs(x1 - x0)
      const i = s <= across ? x0 + Math.sign(x1 - x0) * s : x1
      const j = s <= across ? y0 : y0 + Math.sign(y1 - y0) * (s - across)
      const v = (s >= head - 1 ? 1 : 0.45) * fade, c = j * cols + i
      if (v > out[c]) out[c] = v
    }
  }
}

// Fill `out` (cols * rows, row by row) with `pattern` at time t (s). `voice` (0..1, smoothed) is how loud whoever is
// talking is: Think's ripples follow the user, Transfer's waves the agent.
export function fill(pattern: Pattern, g: Grid, t: number, voice: number, out: Float32Array) {
  const { cols, rows, aspect } = g, cx = (cols - 1) / 2, cy = (rows - 1) / 2
  const R = Math.sqrt(cx * cx + cy * aspect * cy * aspect) || 1
  for (let j = 0, c = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++, c++) {
      const dx = i - cx, dy = (j - cy) * aspect, r = Math.sqrt(dx * dx + dy * dy), fall = Math.max(0, 1 - r / R)
      let v = 0
      switch (pattern) {
        case "analyze":         // purple: a flickering grid, signals running through it
          v = 0.08 + 0.5 * Math.pow(flicker(i, j, t, 1.6), 2.2)
          break
        case "think": {         // teal: ripples out from the centre, every 0.8 s
          const amp = 0.55 + 0.45 * voice
          v = 0.04 + 0.1 * flicker(i, j, t, 1) + (0.2 + 0.85 * ring(r / 16 - t * 1.25, 6)) * fall * amp
          break
        }
        case "connect":         // green: waves sweeping left to right
          v = 0.04 + 0.1 * flicker(i, j, t, 1) + 0.9 * ring(i / 26 - t * 0.85, 5) * (0.3 + 0.7 * (1 - i / cols))
          break
        case "deep": {          // violet: a turning spiral
          const a = Math.atan2(dy, dx)
          v = 0.04 + 0.95 * Math.pow(0.5 + 0.5 * Math.cos(3 * a - r / 5 + t * 2.2), 3) * Math.pow(fall, 0.7)
          break
        }
        case "transfer": {      // amber: bands rolling down, swaying
          const amp = 0.5 + 0.5 * voice
          v = 0.04 + 0.1 * flicker(i, j, t, 1)
            + 0.9 * amp * ring(j / 11 - t * 1.3 + 0.2 * Math.sin(i / 14 + t * 0.8), 4) * (1 - 0.6 * j / rows)
          break
        }
        case "intense": {       // red: pulses from four centres, and a fast one from the middle every 3 s
          v = 0.04
          for (let k = 0; k < CENTRES.length; k++) {
            const qx = i - CENTRES[k][0] * cols, qy = (j - CENTRES[k][1] * rows) * aspect
            const rk = Math.sqrt(qx * qx + qy * qy)
            v += 0.7 * ring(rk / 12 - t * 1.4 - k / 4, 8) * Math.max(0, 1 - rk / (R * 0.7))
          }
          const pulse = (t % 3) * 60
          v += 0.9 * Math.exp(-(((r - pulse) / 3) ** 2))
          break
        }
      }
      out[c] = clamp(v)
    }
  }
  if (pattern === "analyze") signals(g, t, out)
}
