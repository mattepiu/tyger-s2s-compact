// tyger's client: a conversation with tyger's server (EARS) from any web page, as tyger's own page has it. No
// dependencies, no framework: copy this folder into your app, or import it (README -> "Using tyger from your own app").
//
//   const tyger = new TygerClient({
//     url: "wss://tyger.example.com/ws",                                 // default: this page's origin, /ws
//     token: async () => (await fetch("/api/tyger-token")).text(),     // when the server checks tokens
//     on: { reply: (text) => show(text), close: (why) => done(why) },
//   })
//   await tyger.start()        // asks for the mic; then talk. tyger.stop() ends it, tyger.setVoice("aria") switches
//
// What it does for you, and what a client must do for barge-in to work (DEVELOPING.md -> "The 2nd ear"):
//   * the mic goes up in 20 ms blocks at 16 kHz, each tagged with the reply sample the user was hearing when it was
//     captured: [int32 reply id][int32 reply sample, or -1][int16 samples];
//   * it plays the agent's audio itself, on the same AudioContext as the mic capture, which is what makes the tags
//     right (an app that plays the audio another way breaks them, and barge-in then takes the agent's echo for the
//     user);
//   * once a reply has played out it reports `played:<id>:<samples>` (that ends the agent's turn), and on `barge` it
//     stops playback at once.
// The mic needs HTTPS (or localhost) on your page, and then wss:// to the server.
//
// Plain TypeScript (erasable syntax only, no imports), so tests/page/web_check.mjs runs it in Node with fake audio APIs.

export const TARGET_SR = 16000, WIRE_SR = 24000, BLOCK = 320   // mic blocks of 20 ms at 16 kHz; replies at 24 kHz

// The mic and the agent's voice share one AudioContext clock (I-17), so every mic block can say which reply sample
// the user was hearing when it was captured. The worklet resamples the mic to 16 kHz, continuously across render
// quanta (linear interpolation), and posts 20 ms blocks with the context time of each block's first sample. It is
// loaded from this string, so there's no asset file to serve.
export const MIC_WORKLET = `
class Mic16k extends AudioWorkletProcessor {
  constructor(){ super(); this.step = sampleRate / ${TARGET_SR}; this.pos = -1; this.prev = 0;
    this.out = new Float32Array(${BLOCK}); this.n = 0; this.t0 = 0; }
  process(inputs){
    const x = inputs[0] && inputs[0][0];
    if (!x || !x.length) return true;
    const f0 = currentFrame;                          // the context frame of x[0]
    if (this.pos < f0 - 1) this.pos = f0;             // the first quantum, or a gap: start over here
    for (;;) {
      const p = Math.floor(this.pos), i = p - f0;     // the input sample at or before the next output sample
      if (i + 1 >= x.length) break;                   // its successor comes with the next quantum
      const a = i < 0 ? this.prev : x[i], f = this.pos - p;
      if (this.n === 0) this.t0 = this.pos / sampleRate;
      this.out[this.n++] = a + (x[i + 1] - a) * f;
      if (this.n === ${BLOCK}) { this.port.postMessage({t: this.t0, x: this.out.slice()}); this.n = 0; }
      this.pos += this.step;
    }
    this.prev = x[x.length - 1];
    return true;
  }
}
registerProcessor('mic16k', Mic16k);`

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Message = { type: string; [key: string]: any }
export type Link = { label: string; url: string }

// What happens in the conversation. Every one is optional.
export interface TygerEvents {
  open(): void                                    // connected; `ready` comes next
  ready(m: Message): void                         // voices, labels, the session limit (max_session_s, 0: none), sliders
  partial(text: string): void                     // the user's words so far, while they speak
  transcript(text: string): void                  // the user's turn, whole: the agent's turn starts ("" if none heard)
  reply(text: string): void                       // the agent's reply so far: each one replaces the last
  links(links: Link[]): void                      // the sources the reply cites: show them, they're never spoken
  backchannel(text: string): void                 // the agent's "mm-hm" while the user speaks
  backchannelDone(text: string, aborted: boolean): void   // aborted: dropped unvoiced, the user's turn ended first
  barge(): void                                   // the user cut in: the agent stops (its playback is stopped here)
  turnDone(barged: boolean): void                 // the agent's turn is over
  taskDone(task: number): void                    // a background task's result is said next, as a new reply
  sessionEnd(reason: string): void                // "time_limit": the server ends the conversation; `close` follows
  playing(on: boolean): void                      // the agent's audio started / all of it has played
  close(why: string): void                        // the conversation is over: "" after stop(), else why
  error(why: string): void                        // it couldn't start: the mic, the audio, the token
  message(m: Message): void                       // every JSON message from the server, as it came
}

export type TokenSource = string | (() => string | Promise<string>)

export interface TygerOptions {
  url?: string                // the server's WebSocket, wss://host/ws (http(s):// works too); default: this page's /ws
  token?: TokenSource         // when the server checks tokens: a string, or a function called at each start
  voice?: string              // the voice (and so the character) to talk to; default: the server's
  on?: Partial<TygerEvents>
}

type Chunk = { id: number; t0: number; t1: number; first: number }
type Source = AudioBufferSourceNode & { stopped?: boolean }

export class TygerClient {
  on: Partial<TygerEvents>
  url: string | undefined
  token: TokenSource | undefined
  voice: string | null
  ws: WebSocket | null = null
  ctx: AudioContext | null = null
  stream: MediaStream | null = null
  nodes: AudioNode[] = []
  out: GainNode | null = null          // the replies play through it, so outMeter hears them
  micMeter: AnalyserNode | null = null
  outMeter: AnalyserNode | null = null
  buf = new Float32Array(512)
  inLatency = 0
  closeReason = ""
  sentVoice: string | null = null      // the voice already sent on this connection
  // Reply chunks as scheduled on the context clock: {id, t0, t1, first: its first sample in the reply}. Backchannel
  // audio (id 0) plays but is never here, so it stays out of the 2nd ear.
  sched: Chunk[] = []
  queued: Record<number, number> = {}
  lastId = 0
  nextPlay = 0
  active: Source[] = []
  replies = 0                          // reply chunks (not backchannels) scheduled and not yet played

  constructor(opts: TygerOptions = {}) {
    this.on = opts.on || {}; this.url = opts.url; this.token = opts.token; this.voice = opts.voice || null
  }

  get open() { return this.ws !== null && this.ws.readyState === 1 }
  send(text: string) { if (this.open) this.ws!.send(text) }      // a command: set:<threshold>:<value>, reset

  // Switch voices, and with it the character, from the next turn; remembered for the next start too.
  setVoice(name: string) {
    this.voice = name
    if (this.open && this.sentVoice !== name) { this.sentVoice = name; this.ws!.send("voice:" + name) }
  }

  // The mic, the audio, then the token and the connection. Reports `error` when one can't start.
  async start() {
    try {   // full browser DSP on (echo / noise / gain), so the agent's own voice doesn't leak into the mic
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    } catch (e) { this.on.error?.("The mic is blocked: " + (e as Error).message); return }
    const ctx = this.ctx = new AudioContext()
    try {
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([MIC_WORKLET], { type: "application/javascript" })))
    } catch (e) { this.stopAudio(); this.on.error?.("This browser has no AudioWorklet: " + (e as Error).message); return }
    try { this.inLatency = (this.stream.getAudioTracks()[0].getSettings() as { latency?: number }).latency || 0 }
    catch { this.inLatency = 0 }
    this.resetPlayback()                                 // Stop, then Start without a reload: nothing of the old queue
    let url: URL
    try { url = await this.address() }                   // last, right before connecting: tokens are short-lived
    catch (e) { this.stopAudio(); this.on.error?.("Couldn't get a token for tyger: " + (e as Error).message); return }
    const ws = this.ws = new WebSocket(url.toString())
    ws.binaryType = "arraybuffer"
    this.sentVoice = null
    let opened = false
    ws.onopen = () => { opened = true; this.on.open?.() }
    ws.onclose = () => {
      if (this.ws !== ws) return                         // stopped here (stop() said so), or an older conversation's
      const why = this.closeReason || (opened ? "The connection closed." : this.token
        ? "Can't reach tyger, or it refused the token." : "Can't reach tyger.")
      this.closeReason = ""; this.ws = null; this.stopAudio(); this.on.close?.(why)
    }
    ws.onmessage = (ev) => { if (ev.data instanceof ArrayBuffer) this.playChunk(ev.data); else this.onMessage(JSON.parse(ev.data)) }
    const source = ctx.createMediaStreamSource(this.stream!)
    const mic = new AudioWorkletNode(ctx, "mic16k")
    mic.port.onmessage = (e) => this.onMic(e.data.t, e.data.x)
    const gain = ctx.createGain(); gain.gain.value = 0   // keeps the worklet pulled by the graph, silently
    source.connect(mic); mic.connect(gain); gain.connect(ctx.destination)
    const micMeter = this.micMeter = ctx.createAnalyser(), out = this.out = ctx.createGain()
    const outMeter = this.outMeter = ctx.createAnalyser()
    micMeter.fftSize = outMeter.fftSize = this.buf.length
    source.connect(micMeter); out.connect(ctx.destination); out.connect(outMeter)
    this.nodes = [mic, source, gain, micMeter, out, outMeter]
  }

  // The WebSocket's URL: the option's (http(s) taken as ws(s)) or this page's /ws, with the token if there is one.
  async address() {
    const page = typeof location !== "undefined" ? location : null
    const url = new URL(this.url || `${page && page.protocol === "https:" ? "wss" : "ws"}://${page ? page.host : ""}/ws`)
    if (url.protocol === "https:") url.protocol = "wss:"
    else if (url.protocol === "http:") url.protocol = "ws:"
    const token = typeof this.token === "function" ? await this.token() : this.token
    if (token) url.searchParams.set("token", token)
    return url
  }

  // How loud each voice is right now (RMS of the last ~10 ms): the user's at the mic, the agent's as it plays. 0 when
  // there's no conversation.
  meter() {
    const rms = (a: AnalyserNode | null) => {
      if (!a) return 0
      a.getFloatTimeDomainData(this.buf)
      let sum = 0
      for (let i = 0; i < this.buf.length; i++) sum += this.buf[i] * this.buf[i]
      return Math.sqrt(sum / this.buf.length)
    }
    return { mic: rms(this.micMeter), out: rms(this.outMeter) }
  }

  stop() {                                               // the user stopped: close("") now, not when the socket closes
    const ws = this.ws; this.ws = null
    try { ws?.close() } catch {}
    this.stopAudio(); this.closeReason = ""
    this.on.close?.("")
  }

  onMessage(m: Message) {
    const on = this.on
    switch (m.type) {
      case "busy": this.closeReason = "tyger is busy: someone else is talking. Try again shortly."; break
      case "error": this.closeReason = m.error === "audio_too_fast"
        ? "The mic audio came in faster than real time, so tyger closed the connection."
        : "tyger went away. Start again to reconnect."; break                          // e.g. MOUTH restarted
      case "session_end":
        this.closeReason = m.reason === "time_limit" ? "Time's up — start again." : "The conversation ended."
        on.sessionEnd?.(m.reason); break
      case "ready": on.ready?.(m); break
      case "partial": on.partial?.(m.text || ""); break
      case "endpoint": this.stopPlayback(); on.transcript?.(m.transcript || ""); break   // a new reply never overlaps
      case "reply": on.reply?.(m.text || ""); break
      case "links": on.links?.(m.links || []); break
      case "backchannel": on.backchannel?.(m.text || ""); break
      case "backchannel_done": on.backchannelDone?.(m.text || "", !!m.aborted); break
      case "barge": this.stopPlayback(); on.barge?.(); break
      case "turn_done": on.turnDone?.(!!m.barged); break
      case "task_done": on.taskDone?.(m.task); break
    }
    on.message?.(m)
    // the voice picked, sent at every start: the server's default (m.voice) may not be the one this page shows
    if (m.type === "ready" && this.voice && (m.voices || []).includes(this.voice)) this.setVoice(this.voice)
  }

  latency() {                                            // graph -> ear -> graph
    const c = this.ctx!
    return (c.baseLatency || 0) + (c.outputLatency || 0) + this.inLatency
  }

  tagAt(t: number): [number, number] {                   // [reply id, its sample the user heard at context time t]
    for (const c of this.sched) if (c.t0 <= t && t < c.t1) return [c.id, c.first + Math.floor((t - c.t0) * WIRE_SR)]
    return [this.lastId, -1]
  }

  micFrame(t: number, x: Float32Array) {                 // [int32 id][int32 agent sample, -1: none][int16 x...]
    const [id, at] = this.tagAt(t - this.latency())
    const buf = new ArrayBuffer(8 + 2 * x.length), dv = new DataView(buf), pcm = new Int16Array(buf, 8)
    dv.setInt32(0, id, true); dv.setInt32(4, at, true)
    for (let i = 0; i < x.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)))
    return buf
  }

  onMic(t: number, x: Float32Array) {
    if (this.open) this.ws!.send(this.micFrame(t, x))
  }

  playChunk(ab: ArrayBuffer) {                           // [int32 id][int16 samples at 24 kHz]
    const ctx = this.ctx
    if (!ctx) return
    const id = new DataView(ab).getInt32(0, true), pcm = new Int16Array(ab, 4)
    const b = ctx.createBuffer(1, pcm.length, WIRE_SR), ch = b.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768
    const s: Source = ctx.createBufferSource(); s.buffer = b; s.connect(this.out || ctx.destination)
    const t = ctx.currentTime
    if (this.nextPlay < t) this.nextPlay = t + 0.04
    s.start(this.nextPlay)
    if (id !== 0) {
      const first = this.queued[id] || 0; this.queued[id] = first + pcm.length; this.lastId = id
      this.sched.push({ id, t0: this.nextPlay, t1: this.nextPlay + b.duration, first })
      this.sched = this.sched.filter((c) => c.t1 > t - 2)
      if (this.replies++ === 0) this.on.playing?.(true)
    }
    this.nextPlay += b.duration
    const end = this.nextPlay
    this.active.push(s)
    s.onended = () => {
      const i = this.active.indexOf(s); if (i >= 0) this.active.splice(i, 1)
      if (id === 0 || s.stopped) return
      if (--this.replies === 0) this.on.playing?.(false)
      // the last chunk queued has played: tell the server how much of this reply the user heard (it ends the turn)
      if (this.nextPlay <= end + 1e-6) setTimeout(() => this.send(`played:${id}:${this.queued[id]}`),
                                                  1000 * (ctx.outputLatency || 0))
    }
  }

  stopPlayback() {
    this.active.forEach((s) => { s.stopped = true; try { s.stop() } catch {} })
    this.active = []; this.nextPlay = 0; this.sched = []
    if (this.replies) { this.replies = 0; this.on.playing?.(false) }
  }

  resetPlayback() { this.stopPlayback(); this.queued = {}; this.lastId = 0 }   // reply ids count from 1 per connection

  stopAudio() {
    this.stopPlayback()
    try { this.nodes.forEach((n) => n.disconnect()) } catch {}
    try { this.stream?.getTracks().forEach((t) => t.stop()) } catch {}
    try { this.ctx?.close() } catch {}
    this.nodes = []; this.stream = null; this.ctx = null; this.out = this.micMeter = this.outMeter = null
  }
}
