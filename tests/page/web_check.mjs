// The new page's own check (web/), GPU-free and without a browser: web/lib/conversation.ts with page_check.mjs's fake
// audio APIs and its cases (the worklet's 16 kHz blocks, the playback tags, the int16 frames, the `played` report only
// once the queue drains, none after a stop, Stop -> Start without a reload), plus what's new there (reply audio
// playing / drained, the voices' meters, the reason a conversation closed, an old socket's late close); and
// web/lib/talk.ts, what the page shows (the phases; the user's words, the backchannels, the replies and the links they
// cite). Audio is synthetic (a sine, silence).
//   node tests/page/web_check.mjs        # node >= 22.18 (runs the TypeScript as is); no npm install needed
import vm from "vm";

let checks = 0, fails = 0;
const ok = (cond, msg) => { checks++; if (!cond) { fails++; console.log("FAIL", msg); } };
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- the browser, faked as page_check.mjs does -------------------------------------------------------------------
const G = globalThis;
G.location = { protocol: "http:", host: "localhost:7860" };
G.WebSocket = class { constructor(url) { this.url = url; this.readyState = 1; this.sent = []; G._ws = this;
  setTimeout(() => this.onopen && this.onopen(), 0); } send(x) { this.sent.push(x); } close() { this.readyState = 3; } };
G.AudioWorkletNode = class { constructor(ctx, name) { this.name = name; this.port = {}; } connect() {} disconnect() {} };
G.AudioContext = class {
  constructor() { this.currentTime = 10.0; this.sampleRate = 48000; this.baseLatency = 0.01; this.outputLatency = 0.02;
    this.destination = {}; this.audioWorklet = { addModule: async () => {} }; G._ctx = this; G._sources = []; }
  createBuffer(ch, n, sr) { const d = new Float32Array(n); return { duration: n / sr, getChannelData: () => d }; }
  createBufferSource() { const s = { connect(n) { s.to = n; }, start(t) { s.t = t; }, stop() { s.onended && s.onended(); } };
    G._sources.push(s); return s; }
  createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  createGain() { const g = { gain: {}, to: [], connect(n) { g.to.push(n); }, disconnect() {} }; return g; }
  createAnalyser() { const a = { fftSize: 2048, value: 0, connect() {}, disconnect() {},     // a steady signal of `value`
    getFloatTimeDomainData(buf) { buf.fill(a.value); } }; (G._analysers ||= []).push(a); return a; }
  close() {}
};
Object.defineProperty(G, "navigator", { configurable: true, value: { mediaDevices: { getUserMedia: async () => ({
  getAudioTracks: () => [{ getSettings: () => ({ latency: 0.01 }), stop() {} }], getTracks: () => [] }) } } });

const { Conversation, MIC_WORKLET } = await import("../../web/lib/conversation.ts");
const T = await import("../../web/lib/talk.ts");

// ---- the worklet, in a fake AudioWorkletGlobalScope -------------------------------------------------------------
function runWorklet(sr, startFrame, seconds, hz) {
  const posted = [];
  const scope = { sampleRate: sr, currentFrame: 0, Float32Array, Math,
    registerProcessor: (n, cls) => { scope.cls = cls; },
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: m => posted.push(m) }; } } };
  vm.createContext(scope); vm.runInContext(MIC_WORKLET, scope);
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
    const t0 = 4800 / sr + k * 320 / 16000;
    maxDt = Math.max(maxDt, Math.abs(b.t - t0));
    b.x.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - Math.sin(2 * Math.PI * 440 * (b.t + j / 16000)))); });
  });
  ok(maxDt < 1e-9, `${sr}: block times on the context clock (max error ${maxDt})`);
  ok(maxErr < (sr === 48000 ? 1e-6 : 3e-3), `${sr}: samples match the 16 kHz sine (max error ${maxErr.toExponential(1)})`);
}

// ---- the conversation: tags, frames, reports ------------------------------------------------------------------
const log = [];                                       // the page's side: every event, in order
const conv = new Conversation({
  opened: () => log.push(["opened"]), closed: why => log.push(["closed", why]), failed: why => log.push(["failed", why]),
  message: m => log.push(["message", m.type]), playing: on => log.push(["playing", on]),
});
const playing = () => log.filter(e => e[0] === "playing").map(e => e[1]);
await conv.start();
await sleep(10);
const ws = G._ws, ctx = G._ctx;
ok(ws && ws.binaryType === "arraybuffer" && ws.url === "ws://localhost:7860/ws", `a WebSocket in arraybuffer mode: ${ws?.url}`);
ok(log.some(e => e[0] === "opened"), "opened");
const frame = (id, n) => { const ab = new ArrayBuffer(4 + 2 * n); new DataView(ab).setInt32(0, id, true); return ab; };
const recv = data => ws.onmessage({ data });
recv(frame(1, 2400));                                 // 0.1 s, queued at 10.04 (the 40 ms lead-in)
recv(frame(1, 2400));                                 // back to back: 10.14 .. 10.24
ok(playing().join() === "true", `reply audio playing, said once: ${playing()}`);
const hdr = buf => { const dv = new DataView(buf); return [dv.getInt32(0, true), dv.getInt32(4, true)]; };
const lat = 0.01 + 0.02 + 0.01;                       // base + output + input latency
const x = new Float32Array([0.5, -1.0, 2.0]);
let [id, at] = hdr(conv.micFrame(10.10 + lat, x));
ok(id === 1 && Math.abs(at - 1440) <= 1, `tag in the first chunk: ${id} ${at} (want 1 1440)`);
[id, at] = hdr(conv.micFrame(10.16 + lat, x));
ok(id === 1 && Math.abs(at - 2880) <= 1, `tag in the second chunk: ${id} ${at} (want 1 2880)`);
[id, at] = hdr(conv.micFrame(10.30 + lat, x));
ok(id === 1 && at === -1, `nothing playing after it: ${id} ${at}`);
[id, at] = hdr(conv.micFrame(10.02 + lat, x));
ok(at === -1, `nothing playing before the lead-in ends: ${at}`);
const pcm = new Int16Array(conv.micFrame(10.0, x), 8);
ok(pcm[0] === 16384 && pcm[1] === -32767 && pcm[2] === 32767, `int16 mic samples ${Array.from(pcm)}`);
ok(conv.micFrame(10.0, x).byteLength === 8 + 6, "8-byte header + 2 bytes a sample");
ws.sent.length = 0;
for (let k = 0; k < 5; k++) conv.onMic(10.0 + k * 0.02, x);   // the mic: every block sent
ok(ws.sent.length === 5 && ws.sent.every(b => b instanceof ArrayBuffer), `mic blocks sent: ${ws.sent.length}`);
const [micA, outA] = [conv.micMeter, conv.outMeter];             // the meters: the mic's, and the reply as it plays
micA.value = 0.1; outA.value = -0.25;
const lv = conv.meter();
ok(Math.abs(lv.mic - 0.1) < 1e-6 && Math.abs(lv.out - 0.25) < 1e-6, `the voices' loudness (RMS): ${JSON.stringify(lv)}`);
ok(micA.fftSize === 512 && G._analysers.length === 2, "two meters of 512 samples");
ok(G._sources[0].to === conv.out && conv.out.to.includes(ctx.destination) && conv.out.to.includes(outA),
   "the replies play through the out meter to the speakers");

ctx.currentTime = 10.5;                               // a backchannel: played, never tagged, not "playing"
conv.stopPlayback(); log.length = 0;
recv(frame(0, 2400));
[id, at] = hdr(conv.micFrame(10.56 + lat, x));
ok(at === -1, `a backchannel is untagged: ${at}`);
G._sources.at(-1).onended();
ok(playing().length === 0, `a backchannel isn't reply audio: ${playing()}`);

ctx.currentTime = 11.0;                               // reply 2: the reports
G._sources.length = 0; ws.sent.length = 0; log.length = 0;
recv(frame(2, 4800)); recv(frame(2, 2400));
G._sources[0].onended();                              // the first chunk ends: more is queued, no report
await sleep(40);
ok(!ws.sent.some(m => typeof m === "string" && m.startsWith("played")), "no report while more is queued");
ok(playing().join() === "true", `still playing while more is queued: ${playing()}`);
G._sources[1].onended();                              // the last one ends: report (after the output latency)
ok(playing().join() === "true,false", `drained: ${playing()}`);
await sleep(5);
ok(!ws.sent.includes("played:2:7200"), "the report waits for the output latency");
await sleep(40);
ok(ws.sent.includes("played:2:7200"), `the report: ${ws.sent.filter(m => typeof m === "string")}`);

ctx.currentTime = 12.0; ws.sent.length = 0; G._sources.length = 0; log.length = 0;   // a barge: stopped, no report
recv(frame(3, 24000));
recv(JSON.stringify({ type: "barge", yield: 0.8 }));
await sleep(40);
ok(!ws.sent.some(m => typeof m === "string"), "a stopped reply sends no report");
ok(playing().join() === "true,false", `a barge stops the playing: ${playing()}`);
ok(log.some(e => e[0] === "message" && e[1] === "barge"), "the barge goes on to the page");
[id, at] = hdr(conv.micFrame(12.5 + lat, x));
ok(at === -1, `nothing tagged after a stop: ${at}`);
recv(frame(4, 2400)); recv(JSON.stringify({ type: "endpoint", turn_s: 1 }));
ok(G._sources.at(-1).t !== undefined && conv.active.length === 0, "an endpoint stops what's playing");

// ---- Stop, then Start again without a reload: EARS counts reply ids from 1 on the new connection -------------
ctx.currentTime = 13.0; log.length = 0;
recv(frame(1, 24000));                                // something still queued when the user presses Stop
conv.stop();
ok(log.some(e => e[0] === "closed" && e[1] === ""), `Stop: closed, no reason: ${JSON.stringify(log)}`);
ok(ws.readyState === 3, "Stop closes the socket");
ok(JSON.stringify(conv.meter()) === '{"mic":0,"out":0}', `no conversation, no sound: ${JSON.stringify(conv.meter())}`);
await conv.start();
await sleep(10);
const ctx2 = G._ctx, ws2 = G._ws;
ok(ctx2 !== ctx && ws2 !== ws, "a new context and connection");
log.length = 0;
ws.onclose();                                         // the old socket's close comes in late: ignored
ok(log.length === 0 && conv.open, `an old socket's late close doesn't end the new conversation: ${JSON.stringify(log)}`);
G._sources.length = 0;
ws2.onmessage({ data: frame(1, 2400) });              // the new connection's reply 1
ok(Math.abs(G._sources[0].t - 10.04) < 1e-9, `reply 1 starts after the lead-in, not where the old queue ended: ${G._sources[0].t}`);
[id, at] = hdr(conv.micFrame(10.10 + lat, x));
ok(id === 1 && Math.abs(at - 1440) <= 1, `reply 1 is tagged from its own start: ${id} ${at} (want 1 1440)`);
G._sources[0].onended();
await sleep(40);
ok(ws2.sent.includes("played:1:2400"), `the new reply 1's report counts only its own samples: ${ws2.sent.filter(m => typeof m === "string")}`);

// ---- why a conversation closed ----------------------------------------------------------------------------------
log.length = 0;
ws2.onmessage({ data: JSON.stringify({ type: "busy" }) }); ws2.onclose();
ok(log.at(-1)[0] === "closed" && /busy/.test(log.at(-1)[1]), `busy: ${log.at(-1)}`);
await conv.start(); await sleep(10); log.length = 0;
G._ws.onmessage({ data: JSON.stringify({ type: "error", error: "mouth_lost" }) }); G._ws.onclose();
ok(/went away/.test(log.at(-1)[1]), `MOUTH gone: ${log.at(-1)}`);
await conv.start(); await sleep(10); log.length = 0;
G._ws.onclose();
ok(/connection closed/.test(log.at(-1)[1]), `closed unasked: ${log.at(-1)}`);
G.navigator.mediaDevices.getUserMedia = async () => { throw new Error("Permission denied"); };
log.length = 0; await conv.start();
ok(log.at(-1)[0] === "failed" && /Permission denied/.test(log.at(-1)[1]), `no mic: ${log.at(-1)}`);

// ---- what the page shows (talk.ts): the phases --------------------------------------------------------------------
let s = T.initial, now = 1000;
const ev = (m, dt = 0) => { now += dt; s = T.reduce(s, { kind: "message", m, now }); };
const play = (on, dt = 0) => { now += dt; s = T.reduce(s, { kind: "playing", on, now }); };
const hops = (ms, state = "SPEAK") => { for (let t = 0; t < ms; t += 80) ev({ type: "state", state }, 80); };  // EARS: a state a hop
const phase = (dt = 0) => T.phaseOf(s, now + dt);
ok(phase() === "off", `at first: ${phase()}`);
s = T.reduce(s, { kind: "connecting" }); ok(phase() === "connecting", "connecting");
s = T.reduce(s, { kind: "opened" }); ev({ type: "ready", mouth: true }); hops(400, "LISTEN");
ok(phase() === "listening" && s.note === "", `connected: ${phase()}`);
hops(400);                                            // the model's own state head says SPEAK: no turn without an endpoint
ok(phase() === "listening", `SPEAK before any endpoint: ${phase()}`);
ev({ type: "endpoint", turn_s: 2 }, 100);
ok(phase() === "thinking", `the user's turn ended: ${phase()}`);
hops(5000);
ok(phase() === "thinking", `the agent has the floor, nothing said yet: ${phase()}`);
ev({ type: "reply", text: "One moment." }, 80); play(true, 50);
ok(phase() === "speaking", `speaking: ${phase()}`);
hops(1200); play(false);
hops(T.LOOKUP_AFTER_MS - 100);
ok(phase() === "speaking", `a short gap is still speaking: ${phase()}`);
hops(200);
ok(phase() === "lookup", `quiet in the agent's turn: ${phase()}`);
hops(3000); play(true);
ok(phase() === "speaking", "the answer comes");
hops(2000); play(false); ev({ type: "turn_done", barged: false }, 100);
ok(phase() === "listening", `the turn is over: ${phase()}`);
hops(600);                                            // live 2026-10-12: SPEAK went on for ~600 ms after turn_done
ok(phase() === "listening", `SPEAK after turn_done doesn't open a turn: ${phase()}`);
ev({ type: "endpoint", turn_s: 1 }, 1000); hops(400, "LISTEN");   // nothing heard: no turn goes to MOUTH
ok(phase() === "thinking" && phase(T.ENDPOINT_GRACE_MS) === "listening", "an endpoint with no turn goes back to listening");
hops(1000, "idle"); ev({ type: "endpoint", turn_s: 1 }); hops(800);
ok(phase() === "thinking" && phase(T.SPEAK_STALE_MS + 10) === "listening", "EARS stops saying SPEAK: the turn is gone");
ev({ type: "endpoint", turn_s: 1 }, 2000); hops(300); play(true, 100);
ev({ type: "barge", yield: 0.8 }, 300); play(false);
ok(phase() === "barge" && phase(T.BARGE_FLASH_MS + 10) === "listening", "a barge flashes, then listening");
ev({ type: "turn_done", barged: true }, 200);         // MOUTH halted; the user is still talking
hops(1000, "LISTEN");
ok(phase() === "listening", `after the barge, the user's turn: ${phase()}`);
ev({ type: "endpoint", turn_s: 1 }, 300); hops(400, "LISTEN"); play(true); play(false, 100);   // a barge, then an
ev({ type: "barge", yield: 0.9 }, 100); ev({ type: "endpoint", turn_s: 1 }, 900);           // endpoint before MOUTH halted:
ev({ type: "turn_done", barged: true }, 100); hops(160, "LISTEN");                           // EARS sends that turn next
ok(phase() === "thinking", `the held turn after a barge: ${phase()}`);
ev({ type: "task_done", task: 7 }, 3000); hops(160);
ok(phase() === "thinking" && s.task === 7, `a background task speaks up: ${phase()}`);
ev({ type: "ready", mouth: false });
ok(/MOUTH/.test(s.note), `ears only: ${s.note}`);

// ---- the conversation as text: the user's words, backchannels, replies and their links (I-25) ---------------------
s = T.reduce(T.initial, { kind: "opened" });
const NASA = "https://science.nasa.gov/mission/webb/?a=1&b=2";
const screen = () => s.log.slice(s.from).map(x => `${x.who}:${x.text}` + (x.acks?.length ? `[${x.acks}]` : ""));
ev({ type: "endpoint", turn_s: 2, transcript: " Any news about Webb? " });
ok(screen().join() === "user:Any news about Webb?", `the user's words on screen: ${screen()}`);
ev({ type: "links", links: [{ label: "early", url: "https://early.example/" }] });     // no reply yet: dropped
ev({ type: "reply", text: "Webb found a far-off burst." });
ev({ type: "links", links: [{ label: "science.nasa.gov", url: NASA }, { label: "bad", url: "javascript:alert(1)" },
                            { label: "", url: "https://x.example/" }] });
ev({ type: "links", links: [{ label: "science.nasa.gov", url: NASA }] });                // cited twice: shown once
ev({ type: "reply", text: "Webb found a far-off burst. And more." });                    // the text so far: links stay
let r = s.log.at(-1);
ok(s.log.length === 2 && r.who === "agent" && r.text === "Webb found a far-off burst. And more.", `the reply, updated: ${r.text}`);
ok(r.links.map(l => l.url).join() === `${NASA},https://x.example/`, `links, once each, http(s) only: ${r.links.map(l => l.url)}`);
ok(r.links[1].label === "https://x.example/", "a link with no label shows its URL");
ok(screen().length === 2, `the exchange on screen: the words, then the reply: ${screen()}`);
ev({ type: "turn_done" });
ok(screen().length === 2, "after the turn, the exchange stays on screen");
ev({ type: "backchannel", text: "Mm-hm." });                                             // the user speaks again
ok(screen().length === 0 && s.acks.join() === "Mm-hm.", `a backchannel: the user's new turn takes the screen: ${screen()}`);
ev({ type: "backchannel_done", text: "Mm-hm.", aborted: false }); ev({ type: "backchannel", text: "Right." });
ev({ type: "endpoint", turn_s: 3, transcript: "And the rover?" });
ok(screen().join() === "user:And the rover?[Mm-hm.,Right.]" && s.acks.length === 0, `the turn's acks go with its words: ${screen()}`);
ev({ type: "backchannel_done", text: "Right.", aborted: true });                         // dropped unvoiced, after the endpoint
ok(screen().join() === "user:And the rover?[Mm-hm.]", `an unvoiced backchannel is taken back: ${screen()}`);
ev({ type: "reply", text: "Next one." });
ok(s.log.at(-1).links.length === 0, "a new reply starts without links");
ev({ type: "barge", yield: 0.8 }); ev({ type: "reply", text: "Next one, and" });
ok(s.log.at(-1).cut && s.log.at(-1).text === "Next one, and", "cut off: marked, and its last text still lands");
ev({ type: "turn_done", barged: true });
ev({ type: "backchannel", text: "Okay." }); ev({ type: "backchannel_done", text: "Okay.", aborted: true });
ok(s.acks.length === 0, "an unvoiced backchannel before the endpoint is taken back too");
ev({ type: "endpoint", turn_s: 1, transcript: "" });                                     // nothing heard
ok(screen().length === 0 && s.log.length === 4, `no words: nothing added, the screen cleared: ${screen()}`);
ev({ type: "task_done", task: 3 }); ev({ type: "reply", text: "Your timer is done." });
ok(screen().join() === "agent:Your timer is done." && s.log.at(-1).task === 3, `a background task speaks up: ${screen()}`);
s = T.reduce(s, { kind: "closed", why: "" });
ok(s.conn === "off" && s.log.length === 5 && screen().length === 0, "Stop keeps the history, clears the screen");
ok(s.log.map(x => x.who).join() === "user,agent,user,agent,agent", `the history, in order: ${s.log.map(x => x.who)}`);

console.log(`${checks - fails}/${checks} checks passed`);
process.exit(fails ? 1 : 0);
