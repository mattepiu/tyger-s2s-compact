// The new page (web/) against the running stack, in headless Chromium: one real spoken turn, with the Tuning panel
// open. The mic is a fake getUserMedia stream that plays load_ears's fixed utterance (MOUTH's TTS, piped here in
// memory, never on disk). It checks:
//   * the turn: the user's words as they speak (`partial`) before the endpoint, the endpoint's transcript, the reply
//     (its text and audio), the page's `played` report, `turn_done`, and the user's line on screen;
//   * the tuning: the plugin's sliders in the panel, a slider moved with the keyboard reaching EARS (`set:`) and
//     echoed back (`threshold`), the meters moving while the user speaks;
//   * no page errors.
// Logs message types, counts and lengths only: never the transcript or the reply text.
// Needs the stack (./run.sh) and the new page (cd web && npm run dev) running. Once:
//   cd scripts/web && npm install && npx playwright install --with-deps chromium
// then: node scripts/web/live_check.mjs [page URL, default http://127.0.0.1:3000/]
import { chromium } from "playwright"
import { execFileSync } from "child_process"
import { dirname, resolve } from "path"
import { fileURLToPath } from "url"

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const URL = process.argv[2] || "http://127.0.0.1:3000/"
let checks = 0, fails = 0
const ok = (cond, msg) => { checks++; if (!cond) fails++; console.log(cond ? "ok  " : "FAIL", msg) }

const py = `
import sys, base64; sys.path.insert(0, ".")
from app.config import load_config
from scripts.stack.load_ears import speech_from_mouth, UTTERANCE
x = speech_from_mouth(load_config(), UTTERANCE)
sys.stdout.write(base64.b64encode(x.astype("float32").tobytes()).decode())`
const b64 = execFileSync(`${REPO}/.venv/bin/python`, ["-c", py], { cwd: REPO, maxBuffer: 64 << 20 }).toString()
console.log("utterance:", (Buffer.from(b64, "base64").length / 4 / 16000).toFixed(2), "s")

// software WebGL, so the orb runs as it would in a browser
const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream",
                                               "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] })
const p = await browser.newPage({ viewport: { width: 1280, height: 760 } })
const errors = []
p.on("pageerror", (e) => errors.push(e.message))
await p.addInitScript(() => {
  window._log = []                                 // [ms, kind, type, size]: types and sizes only
  const t0 = performance.now(), note = (k, type, size = 0) => window._log.push([Math.round(performance.now() - t0), k, type, size])
  const mic = new AudioContext({ sampleRate: 16000 }), dest = mic.createMediaStreamDestination()
  navigator.mediaDevices.getUserMedia = async () => dest.stream
  window._say = async (b64) => {
    await mic.resume()
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)), x = new Float32Array(bytes.buffer)
    const buf = mic.createBuffer(1, x.length, 16000); buf.getChannelData(0).set(x)
    const s = mic.createBufferSource(); s.buffer = buf; s.connect(dest); s.start(); note("said", "utterance", x.length)
  }
  const Real = window.WebSocket
  window.WebSocket = function (u, proto) {         // the page's /ws, watched; any other socket (Next's HMR) as is
    const ws = new Real(u, proto)
    if (!/\/ws$/.test(u)) return ws
    const send = ws.send.bind(ws)
    ws.send = (x) => { if (typeof x === "string") note("sent", x.split(":").slice(0, 2).join(":")); return send(x) }
    ws.addEventListener("message", (e) => {
      if (typeof e.data !== "string") { note("audio", "chunk", e.data.byteLength || e.data.size); return }
      const m = JSON.parse(e.data)
      if (m.type !== "state") note("msg", m.type, (m.text ?? m.transcript ?? "").length)
    })
    return ws
  }
  window.WebSocket.prototype = Real.prototype
})
const log = () => p.evaluate(() => window._log)

await p.goto(URL, { waitUntil: "networkidle" })
await p.click("button[aria-label='Start talking']")
await p.waitForFunction(() => window._log.some((e) => e[2] === "ready"), null, { timeout: 15000 })
await p.click("header button:has-text('Tuning')")
const panel = "aside[aria-label=Tuning]"
await p.waitForSelector(`${panel} input[type=range]`, { timeout: 5000 })
const sliders = await p.$$eval(`${panel} input[type=range]`, (els) => els.map((e) => ({
  label: e.getAttribute("aria-label"), value: +e.value, min: +e.min, max: +e.max })))
ok(sliders.length > 0, `the plugin's sliders in the panel: ${sliders.length}`)
await p.waitForTimeout(1000)

await p.evaluate((d) => window._say(d), b64)
await p.waitForTimeout(1500)                       // mid-utterance: move the first slider one step
const s0 = sliders[0], down = s0.value > s0.min
await p.focus(`${panel} input[type=range] >> nth=0`)
await p.keyboard.press(down ? "ArrowLeft" : "ArrowRight")
const meters = { max: [0, 0, 0] }
for (let t = 0; t < 300; t++) {                    // up to 30 s for the turn to finish
  await p.waitForTimeout(100)
  const m = await p.$$eval(`${panel} .tabular-nums`, (els) => els.slice(0, 3).map((e) => +e.textContent))
  meters.max = meters.max.map((v, i) => Math.max(v, m[i] || 0))
  if ((await log()).some((e) => e[2] === "turn_done")) break
}
await p.waitForTimeout(500)
const userLines = await p.$$eval("[aria-live=polite] .rounded-br-md", (els) => els.length)   // the user's bubble
const after = await p.$eval(`${panel} input[type=range] >> nth=0`, (e) => +e.value)
const L = await log()
await p.click("button[aria-label='Stop']")
await browser.close()

for (let i = 0; i < L.length;) {                   // the timeline; a run of the same message as one line
  let j = i + 1
  while (j < L.length && L[j][1] === L[i][1] && L[j][2] === L[i][2]) j++
  const e = L[i], n = j - i
  console.log(String(e[0]).padStart(6), e[1].padEnd(6), e[2], n > 1 ? `x${n} (last ${L[j - 1][3]})` : e[3] || "")
  i = j
}
const at = (kind, type) => L.find((e) => e[1] === kind && e[2] === type)
const all = (kind, type) => L.filter((e) => e[1] === kind && e[2] === type)
const ep = at("msg", "endpoint"), partials = all("msg", "partial")
ok(partials.length > 0 && ep && partials[0][0] < ep[0], `the user's words as they speak, before the endpoint: ${partials.length}`)
ok(ep && ep[3] > 0, `the endpoint's transcript: ${ep ? ep[3] : "none"} chars`)
ok(all("msg", "reply").some((e) => e[3] > 0), "the reply's text")
ok(all("audio", "chunk").length > 0, `the reply's audio: ${all("audio", "chunk").length} chunks`)
ok(L.some((e) => e[1] === "sent" && e[2].startsWith("played:")), "the page reported the reply played")
ok(!!at("msg", "turn_done"), "turn_done")
ok(userLines > 0, `the user's line on screen: ${userLines}`)
ok(L.some((e) => e[1] === "sent" && e[2].startsWith("set:")) && !!at("msg", "threshold"), "a slider moved: sent and echoed")
ok(after !== s0.value, `the slider shows its new value: ${s0.label} ${s0.value} -> ${after}`)
ok(meters.max.some((v) => v > 0), `the meters moved (max p_end / yield / bc): ${meters.max.join(" / ")}`)
ok(errors.length === 0, `no page errors: ${JSON.stringify(errors)}`)
console.log(`${checks - fails}/${checks} checks passed`)
process.exit(fails ? 1 : 0)
