// The conversation with EARS, as the classic page (static/index.html) has it; the protocol is in README.md ("The web
// page"). One WebSocket per conversation. The mic goes up in 20 ms blocks at 16 kHz, each tagged with the reply sample
// the user was hearing when it was captured (I-17); the reply audio comes down and plays on the same AudioContext
// clock; once a reply has played out, `played:<id>:<samples>` tells EARS (that ends the agent's turn). The JSON
// messages are passed on to the page as they come.
//
// Plain TypeScript (erasable syntax only, no imports), so tests/page/web_check.mjs runs it in Node with fake audio APIs.

export const TARGET_SR = 16000, WIRE_SR = 24000, BLOCK = 320   // mic blocks of 20 ms at 16 kHz; replies at 24 kHz

// The mic and the agent's voice share one AudioContext clock (I-17), so every mic block can say which reply sample
// the user was hearing when it was captured. The worklet resamples the mic to 16 kHz, continuously across render
// quanta (linear interpolation), and posts 20 ms blocks with the context time of each block's first sample.
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

export interface Events {
  opened(): void
  closed(reason: string): void        // "" unless EARS said why (busy, MOUTH gone)
  failed(why: string): void           // the mic or the audio couldn't start
  message(m: Message): void           // every JSON message from EARS
  level(peak: number): void           // the mic's peak (0..1), every 100 ms
  playing(on: boolean): void          // reply audio started / all of it has played (backchannels aside)
}

type Chunk = { id: number; t0: number; t1: number; first: number }
type Source = AudioBufferSourceNode & { stopped?: boolean }

export class Conversation {
  on: Events
  ws: WebSocket | null = null
  ctx: AudioContext | null = null
  stream: MediaStream | null = null
  nodes: AudioNode[] = []
  inLatency = 0
  closeReason = ""
  // Reply chunks as scheduled on the context clock: {id, t0, t1, first: its first sample in the reply}. Backchannel
  // audio (id 0) plays but is never here, so it stays out of the 2nd ear.
  sched: Chunk[] = []
  queued: Record<number, number> = {}
  lastId = 0
  micBlocks = 0
  nextPlay = 0
  active: Source[] = []
  replies = 0                          // reply chunks (not backchannels) scheduled and not yet played

  constructor(on: Events) { this.on = on }

  get open() { return this.ws !== null && this.ws.readyState === 1 }
  send(text: string) { if (this.open) this.ws!.send(text) }

  async start() {
    try {   // full browser DSP on (echo / noise / gain), so the agent's own voice doesn't leak into the mic
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    } catch (e) { this.on.failed("The mic is blocked: " + (e as Error).message); return }
    const ctx = this.ctx = new AudioContext()
    try {
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([MIC_WORKLET], { type: "application/javascript" })))
    } catch (e) { this.stopAudio(); this.on.failed("This browser has no AudioWorklet: " + (e as Error).message); return }
    try { this.inLatency = (this.stream.getAudioTracks()[0].getSettings() as { latency?: number }).latency || 0 }
    catch { this.inLatency = 0 }
    this.resetPlayback()                                 // Stop, then Start without a reload: nothing of the old queue
    const ws = this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`)
    ws.binaryType = "arraybuffer"
    let opened = false
    ws.onopen = () => { opened = true; this.on.opened() }
    ws.onclose = () => {
      if (this.ws !== ws) return                         // stopped here (stop() said so), or an older conversation's
      const why = this.closeReason || (opened ? "The connection closed. Tap the mic to talk again." : "Can't reach tyger.")
      this.closeReason = ""; this.ws = null; this.stopAudio(); this.on.closed(why)
    }
    ws.onmessage = (ev) => { if (ev.data instanceof ArrayBuffer) this.playChunk(ev.data); else this.onMessage(JSON.parse(ev.data)) }
    const source = ctx.createMediaStreamSource(this.stream)
    const mic = new AudioWorkletNode(ctx, "mic16k")
    mic.port.onmessage = (e) => this.onMic(e.data.t, e.data.x)
    const gain = ctx.createGain(); gain.gain.value = 0   // keeps the worklet pulled by the graph, silently
    source.connect(mic); mic.connect(gain); gain.connect(ctx.destination)
    this.nodes = [mic, source, gain]
  }

  stop() {                                               // the user stopped: closed("") now, not when the socket closes
    const ws = this.ws; this.ws = null
    try { ws?.close() } catch {}
    this.stopAudio(); this.closeReason = ""
    this.on.closed("")
  }

  onMessage(m: Message) {
    if (m.type === "busy") this.closeReason = "tyger is busy: someone else is talking. Try again shortly."
    else if (m.type === "error") this.closeReason = "tyger went away. Tap the mic to reconnect."   // e.g. MOUTH restarted
    else if (m.type === "endpoint" || m.type === "barge") this.stopPlayback()  // never let a new reply overlap an old one
    this.on.message(m)
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
    if (!this.open) return
    this.ws!.send(this.micFrame(t, x))
    if (++this.micBlocks % 5 === 0) {
      let peak = 0
      for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a }
      this.on.level(peak)
    }
  }

  playChunk(ab: ArrayBuffer) {                           // [int32 id][int16 samples at 24 kHz]
    const ctx = this.ctx
    if (!ctx) return
    const id = new DataView(ab).getInt32(0, true), pcm = new Int16Array(ab, 4)
    const b = ctx.createBuffer(1, pcm.length, WIRE_SR), ch = b.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768
    const s: Source = ctx.createBufferSource(); s.buffer = b; s.connect(ctx.destination)
    const t = ctx.currentTime
    if (this.nextPlay < t) this.nextPlay = t + 0.04
    s.start(this.nextPlay)
    if (id !== 0) {
      const first = this.queued[id] || 0; this.queued[id] = first + pcm.length; this.lastId = id
      this.sched.push({ id, t0: this.nextPlay, t1: this.nextPlay + b.duration, first })
      this.sched = this.sched.filter((c) => c.t1 > t - 2)
      if (this.replies++ === 0) this.on.playing(true)
    }
    this.nextPlay += b.duration
    const end = this.nextPlay
    this.active.push(s)
    s.onended = () => {
      const i = this.active.indexOf(s); if (i >= 0) this.active.splice(i, 1)
      if (id === 0 || s.stopped) return
      if (--this.replies === 0) this.on.playing(false)
      // the last chunk queued has played: tell EARS how much of this reply the user heard (it ends the turn)
      if (this.nextPlay <= end + 1e-6) setTimeout(() => this.send(`played:${id}:${this.queued[id]}`),
                                                  1000 * (ctx.outputLatency || 0))
    }
  }

  stopPlayback() {
    this.active.forEach((s) => { s.stopped = true; try { s.stop() } catch {} })
    this.active = []; this.nextPlay = 0; this.sched = []
    if (this.replies) { this.replies = 0; this.on.playing(false) }
  }

  resetPlayback() { this.stopPlayback(); this.queued = {}; this.lastId = 0 }   // EARS counts reply ids from 1 per connection

  stopAudio() {
    this.stopPlayback()
    try { this.nodes.forEach((n) => n.disconnect()) } catch {}
    try { this.stream?.getTracks().forEach((t) => t.stop()) } catch {}
    try { this.ctx?.close() } catch {}
    this.nodes = []; this.stream = null; this.ctx = null
  }
}
