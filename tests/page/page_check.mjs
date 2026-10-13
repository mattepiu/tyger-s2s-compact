// The page's own check (I-17), GPU-free and without a browser: static/index.html in jsdom, with fake audio APIs.
// The mic worklet's 16 kHz resampling and block times; the playback tags around chunks, the lead-in and a
// backchannel; the int16 frames; the `played` report only once the queue drains (after the output latency), none
// after a stop; Stop -> Start without a reload (EARS counts reply ids from 1 on each connection); the voice picker;
// and the links a reply cites. Audio is synthetic (a sine, silence).
//   cd tests/page && npm install && node page_check.mjs [path/to/index.html]     # needs node >= 18
import { JSDOM } from "jsdom";
import fs from "fs";
import vm from "vm";

const page = process.argv[2] || new URL("../../static/index.html", import.meta.url);
const html = fs.readFileSync(page, "utf8");
let checks = 0, fails = 0;
const ok = (cond, msg) => { checks++; if (!cond) { fails++; console.log("FAIL", msg); } };
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dom = new JSDOM(html, {
  runScripts: "dangerously", url: "http://localhost:7860/",
  beforeParse(W) {
    W.fetch = async () => ({ json: async () => ({ voices: [], voice: null }) });
    W.URL.createObjectURL = () => "blob:mic";
    W.Blob = class { constructor(parts) { this.parts = parts; } };
    W.WebSocket = class { constructor(url) { this.url = url; this.readyState = 1; this.sent = []; W._ws = this;
      setTimeout(() => this.onopen && this.onopen(), 0); } send(x) { this.sent.push(x); } close() {} };
    W.AudioWorkletNode = class { constructor(ctx, name) { this.name = name; this.port = {}; } connect() {} disconnect() {} };
    W.AudioContext = class {
      constructor() { this.currentTime = 10.0; this.sampleRate = 48000; this.baseLatency = 0.01; this.outputLatency = 0.02;
        this.destination = {}; this.audioWorklet = { addModule: async () => {} }; W._ctx = this; W._sources = []; }
      createBuffer(ch, n, sr) { const d = new Float32Array(n); return { duration: n / sr, getChannelData: () => d }; }
      createBufferSource() { const s = { connect() {}, start(t) { s.t = t; }, stop() { s.onended && s.onended(); } };
        W._sources.push(s); return s; }
      createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
      createGain() { return { gain: {}, connect() {}, disconnect() {} }; }
      close() {}
    };
    W.navigator.mediaDevices = { getUserMedia: async () => ({
      getAudioTracks: () => [{ getSettings: () => ({ latency: 0.01 }), stop() {} }], getTracks: () => [] }) };
  },
});
const W = dom.window;

// ---- the worklet, in a fake AudioWorkletGlobalScope -------------------------------------------------------
function runWorklet(sr, startFrame, seconds, hz) {
  const posted = [];
  const scope = { sampleRate: sr, currentFrame: 0, Float32Array, Math,
    registerProcessor: (n, cls) => { scope.cls = cls; },
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: m => posted.push(m) }; } } };
  vm.createContext(scope); vm.runInContext(W.eval("MIC_WORKLET"), scope);
  const proc = new scope.cls();
  for (let f = 0; f < sr * seconds; f += 128) {
    scope.currentFrame = startFrame + f;
    const x = new Float32Array(128);
    for (let i = 0; i < 128; i++) x[i] = Math.sin(2 * Math.PI * hz * (startFrame + f + i) / sr);
    proc.process([[x]]);
  }
  return posted;
}
for (const sr of [48000, 44100]) {
  const blocks = runWorklet(sr, 4800, 1.0, 440);
  ok(blocks.length >= 48 && blocks.length <= 50, `${sr}: ${blocks.length} blocks of 20 ms in 1 s`);
  let maxErr = 0, maxDt = 0;
  blocks.forEach((b, k) => {
    ok(b.x.length === 320, `${sr}: block ${k} has 320 samples`);
    const t0 = 4800 / sr + k * 320 / 16000;           // continuous: block k starts 20 ms after block k-1
    maxDt = Math.max(maxDt, Math.abs(b.t - t0));
    b.x.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - Math.sin(2 * Math.PI * 440 * (b.t + j / 16000)))); });
  });
  ok(maxDt < 1e-9, `${sr}: block times on the context clock (max error ${maxDt})`);
  ok(maxErr < (sr === 48000 ? 1e-6 : 3e-3), `${sr}: samples match the 16 kHz sine (max error ${maxErr.toExponential(1)})`);
}

// ---- the page: tags, frames, reports ------------------------------------------------------------------------
await sleep(10);
await W.eval("start()");
await sleep(10);
const ws = W._ws, ctx = W._ctx;
ok(ws && ws.binaryType === "arraybuffer", "a WebSocket in arraybuffer mode");
const frame = (id, n) => { const ab = new ArrayBuffer(4 + 2 * n); new DataView(ab).setInt32(0, id, true); return ab; };
const playChunk = W.eval("playChunk"), micFrame = W.eval("micFrame"), stopPlayback = W.eval("stopPlayback");
playChunk(frame(1, 2400));                            // 0.1 s, queued at 10.04 (the 40 ms lead-in)
playChunk(frame(1, 2400));                            // back to back: 10.14 .. 10.24
const hdr = buf => { const dv = new DataView(buf); return [dv.getInt32(0, true), dv.getInt32(4, true)]; };
const lat = 0.01 + 0.02 + 0.01;                       // base + output + input latency
const x = new Float32Array([0.5, -1.0, 2.0]);
let [id, at] = hdr(micFrame(10.10 + lat, x));
ok(id === 1 && Math.abs(at - 1440) <= 1, `tag in the first chunk: ${id} ${at} (want 1 1440)`);
[id, at] = hdr(micFrame(10.16 + lat, x));
ok(id === 1 && Math.abs(at - 2880) <= 1, `tag in the second chunk: ${id} ${at} (want 1 2880)`);
[id, at] = hdr(micFrame(10.30 + lat, x));
ok(id === 1 && at === -1, `nothing playing after it: ${id} ${at}`);
[id, at] = hdr(micFrame(10.02 + lat, x));
ok(at === -1, `nothing playing before the lead-in ends: ${at}`);
const pcm = new Int16Array(micFrame(10.0, x), 8);
ok(pcm[0] === 16384 && pcm[1] === -32767 && pcm[2] === 32767, `int16 mic samples ${Array.from(pcm)}`);
ok(micFrame(10.0, x).byteLength === 8 + 6, "8-byte header + 2 bytes a sample");

ctx.currentTime = 10.5;                               // a backchannel: played, never tagged
playChunk(frame(0, 2400));
[id, at] = hdr(micFrame(10.56 + lat, x));
ok(at === -1, `a backchannel is untagged: ${at}`);

ctx.currentTime = 11.0;                               // reply 2: the reports
W._sources.length = 0; ws.sent.length = 0;
playChunk(frame(2, 4800)); playChunk(frame(2, 2400));
W._sources[0].onended();                              // the first chunk ends: more is queued, no report
await sleep(40);
ok(!ws.sent.some(m => typeof m === "string" && m.startsWith("played")), "no report while more is queued");
W._sources[1].onended();                              // the last one ends: report (after the output latency)
await sleep(5);
ok(!ws.sent.includes("played:2:7200"), "the report waits for the output latency");
await sleep(40);
ok(ws.sent.includes("played:2:7200"), `the report: ${ws.sent.filter(m => typeof m === "string")}`);

ctx.currentTime = 12.0; ws.sent.length = 0; W._sources.length = 0;   // a barge: stopped, no report, no tags
playChunk(frame(3, 24000));
stopPlayback();
await sleep(40);
ok(!ws.sent.some(m => typeof m === "string"), "a stopped reply sends no report");
[id, at] = hdr(micFrame(12.5 + lat, x));
ok(at === -1, `nothing tagged after a stop: ${at}`);

W.eval("onEvent")({ data: JSON.stringify({ type: "ready", hop_ms: 320, sr: 16000, mouth: true, voices: [], sliders: [], thresholds: {} }) });
ok(W.document.getElementById("rt").textContent.includes("audio latency 40ms"), W.document.getElementById("rt").textContent);

// ---- Stop, then Start again without a reload: EARS counts reply ids from 1 on the new connection -------------
ctx.currentTime = 13.0;
playChunk(frame(1, 24000));                           // something still queued when the user presses Stop
W.eval("stop()");
await W.eval("start()");
await sleep(10);
const ctx2 = W._ctx;                                   // a new AudioContext, its clock from 10.0 again
ok(ctx2 !== ctx && W._ws !== ws, "a new context and connection");
W._sources.length = 0;
playChunk(frame(1, 2400));                            // the new connection's reply 1
ok(Math.abs(W._sources[0].t - 10.04) < 1e-9, `reply 1 starts after the lead-in, not where the old queue ended: ${W._sources[0].t}`);
[id, at] = hdr(micFrame(10.10 + lat, x));
ok(id === 1 && Math.abs(at - 1440) <= 1, `reply 1 is tagged from its own start: ${id} ${at} (want 1 1440)`);
W._sources[0].onended();
await sleep(40);
ok(W._ws.sent.includes("played:1:2400"), `the new reply 1's report counts only its own samples: ${W._ws.sent.filter(m => typeof m === "string")}`);

// ---- the voice picker: shown for everyone (each voice is a character), the sliders only in debug ---------------
const VOICES = { voices: ["aria", "mira"], voice: "mira", labels: { aria: "Aria (aria)", mira: "Mira (mira)" } };
async function pickerPage(query) {
  const d = new JSDOM(html, { runScripts: "dangerously", url: "http://localhost:7860/" + query,
    beforeParse(W) {
      W.fetch = async () => ({ json: async () => VOICES });
      W.WebSocket = class { constructor() { this.readyState = 1; this.sent = []; W._ws = this; } send(x) { this.sent.push(x); } close() {} };
    } });
  await sleep(20);
  return d.window;
}
for (const query of ["", "?nodebug"]) {
  const P = await pickerPage(query), sel = P.document.getElementById("voice"), at = query || "(no query)";
  ok(!sel.hidden, `the voice picker shows ${at}`);
  ok([...sel.options].map(o => o.textContent).join() === "Aria (aria),Mira (mira)",
     `entries name the character, then the voice ${at}: ${[...sel.options].map(o => o.textContent)}`);
  ok(P.document.getElementById("tune").hidden === (query === "?nodebug"), `the sliders show only in debug ${at}`);
  P.eval("ws = new WebSocket('ws://localhost:7860/ws')");
  P.eval("onEvent")({ data: JSON.stringify({ type: "ready", hop_ms: 320, sr: 16000, mouth: true, sliders: [], thresholds: {}, ...VOICES }) });
  ok(P._ws.sent.includes("voice:" + sel.value), `the picked voice is sent at the start ${at}: ${P._ws.sent}`);
  sel.value = "aria"; sel.dispatchEvent(new P.Event("change"));
  ok(P._ws.sent.at(-1) === "voice:aria", `picking a voice sends voice:<name> ${at}: ${P._ws.sent}`);
}

// ---- the links a reply cites (I-25): under it, http(s) only, as text; a new reply starts without them -----------
{
  const P = await pickerPage(""), turns = P.document.getElementById("turns");
  const ev = m => P.eval("onEvent")({ data: JSON.stringify(m) });
  const NASA = "https://science.nasa.gov/mission/webb/?a=1&b=2";
  ev({ type: "links", links: [{ label: "early", url: "https://early.example/" }] });     // no reply yet: dropped
  ev({ type: "reply", text: "Webb found a far-off burst." });
  ev({ type: "links", links: [{ label: "science.nasa.gov", url: NASA }, { label: "bad", url: "javascript:alert(1)" },
                              { label: "<b>x</b>", url: "https://x.example/" }] });
  ev({ type: "links", links: [{ label: "science.nasa.gov", url: NASA }] });              // cited twice: shown once
  ev({ type: "reply", text: "Webb found a far-off burst. And more." });                  // the text so far: links stay
  const a = [...turns.querySelectorAll(".turn.asst .src a")];
  ok(a.length === 2 && a[0].href === NASA && a[0].textContent === "science.nasa.gov",
     `the reply's links, once each, http(s) only: ${a.map(x => x.href)}`);
  ok(a[0]?.target === "_blank" && a[0]?.rel.includes("noopener"), `a link opens apart from the page: ${a[0]?.target}`);
  ok(a[1]?.textContent === "<b>x</b>" && !turns.querySelector(".src b"), "a label is text, never markup");
  ok(!turns.innerHTML.includes("javascript:") && !turns.innerHTML.includes("early"), "no script link, no stray link");
  ok(turns.querySelector(".turn.asst .txt").textContent === "Webb found a far-off burst. And more.",
     "the text stays the text");
  ev({ type: "turn_done" }); ev({ type: "reply", text: "Next one." });
  ok(turns.querySelectorAll(".turn.asst").length === 2 && !turns.firstElementChild.querySelector(".src"),
     "a new reply starts without links");
}

console.log(`${checks - fails}/${checks} checks passed`);
process.exit(fails ? 1 : 0);
