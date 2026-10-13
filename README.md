# tyger v1 — a deployable full-duplex voice agent on one GPU

Real-time spoken conversation in English, in the browser: your mic → live turn-taking → a reasoning LLM with tools
and a hosted expert that searches the web → a cloned voice, all on a single GPU. The agent knows when you've finished
a turn, when you're interrupting it, and when to drop in a quick "mm-hm" — then answers in a voice cloned from a
reference clip you provide.

This repo is the **inference / serving** project. The turn-taking model itself (the endpoint head and the
chain it sits on) is trained elsewhere; v1 only *runs* it.

**The docs:**
- `README.md` (this file): what it is, and how to install, configure and run it;
- `DEVELOPING.md`: how it works inside, the rules for changing it, and how to check a change;
- `CHANGELOG.md`: what changed, in short;
- `ISSUES.md`: what is left to do, by ID;
- `HANDOFF.md`: the GPU-box agent's current round.

```
browser mic ──ws──▶ EARS    nemotron L15 → endpoint head: endpoint / barge / backchannel decisions
                     │                                    also serves the web app on :7860
                     ▼ tcp :8770
                    MOUTH   orchestrator + two in-process vLLM engines on one asyncio loop:
                     │        • Gemma 4 E4B FP8, text only, no thinking (vLLM AsyncLLM, tool calls)
                     │        • VoxCPM2 voice clone (vLLM-Omni AsyncOmni, 48 kHz → 24 kHz)
                     ▼ ws
browser speaker ◀────┘      24 kHz audio, streamed as it decodes
```

Two processes, **one venv** (`.venv`, `requirements.txt`). **One GPU, up to 4 conversations at once**
(`server.max_sessions`); the next visitor gets a "busy" page, and a conversation ends after 3 minutes. Another app
can use it too, with its own page and a token from its backend ("API mode", "Using tyger from your own app").

## How it works

1. **EARS** runs the streaming ASR encoder (nemotron) and, on top of its L15 hidden state, a set of per-frame
   heads that decide, every 80 ms: has the user's turn **ended** (take the floor), is the user **barging in**
   over the agent (stop talking), and should the agent emit a **backchannel** ("right", "mm-hm") while still
   listening. EARS serves the web page and the WebSocket. To judge a barge-in, the heads hear the agent's own
   reply next to the mic (the "2nd ear"). The page plays the reply and records the mic on one audio clock and
   tags every mic block with the reply sample the user was hearing, so the 2nd ear lines up with what the user
   actually heard, whatever the network and audio latencies. The agent's turn ends when the page reports it has
   played the whole reply.
2. **MOUTH** is the orchestrator, and hosts both models in-process (no HTTP hop to either). On an endpoint it
   streams the LLM reply, cuts it into text chunks, and sends each chunk to VoxCPM2 as soon as it closes;
   audio streams back to EARS (and on to the browser) per decode step. Chunks are whole sentences: each chunk is
   its own TTS request, so the intonation restarts at a chunk boundary, and at a sentence end that sounds natural.
   (`mouth.first_chunk: clause` lets the first chunk end at a comma instead: audio ~30–95 ms sooner, but the
   intonation restarts mid-sentence.) No chunk is cut below 40 characters, so short replies go to the TTS
   whole. On a tool call
   it speaks a short contextual filler ("Checking the time in Tokyo.") the moment the call starts
   decoding, runs the tool as soon as the call parses, adds a "still looking" line if the tool is slow, then
   speaks the result. On a barge it stops within one audio chunk, aborts the live LLM/TTS requests, and keeps
   what the user heard of the reply in the conversation history.

## Hardware

Built on one **NVIDIA RTX 5090** (32 GB, Blackwell sm_120), CUDA-13 driver (≥ 590) for the cu130 torch wheels.
It also runs on an **RTX PRO 5000** (48 GB), with the same memory but ~20% slower LLM decoding (DEVELOPING.md →
"Benchmarks").

GPU memory in use ≈ **30 GB** with everything loaded, warm, and under a 4-conversation load: Gemma 4 E4B ~15 GB
with FP8 weights (3 GiB pinned KV, `llm.reply.kv_cache_gib`; bf16 weights would not fit), VoxCPM2 ~9.2 GB (2 GiB
pinned KV, `config/engines/voxcpm2.yaml`), EARS ~6 GB. Both engines pin their KV budgets, so the total is
predictable. DEVELOPING.md → "GPU memory" has the breakdown.

## Setup

### 1. Install (one venv)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=auto
```

### 2. Secrets — `.env`

Copy `.env.example` to `.env` (gitignored) and fill in:

- **`OPENAI_API_KEY`** — the `smart` model (`llm.smart` in `config/default.yaml`): escalations, background tasks,
  and live facts (weather, news, prices) through OpenAI's web search. There is no local web search or weather tool.
  Without the key, comment out `llm.smart`: the agent then says it can't check live facts.
- **`ANTHROPIC_API_KEY`** — only if `llm.smart` is the `anthropic` plugin instead.
- **`HF_TOKEN`** — pulls the **private** turn-taking heads release (`ears.heads` in `config/default.yaml`). If a token
  already exists at `$HF_HOME/token`, leave this blank.
- **`TYGER_API_SECRET`** — only for API mode: the secret your app's backend signs conversation tokens with ("API
  mode"). Unset, there's no token check. A long random string, e.g. `openssl rand -base64 32`.

Never commit `.env`.

### 3. Voices

Put one or more reference `.wav` clips in `voices/` (`voices.dir` in `config/default.yaml`; gitignored). The stem is the
voice name; `voices.default` picks the active one (falls back to the first clip). Optionally add
`voices/<name>.txt` with the clip's exact transcript: VoxCPM2 then clones from audio + transcript (better
quality, uses the whole clip); without it the clip is trimmed to `tts.ref_max_s` (10 s). **Reference clips and
transcripts are personal data: keep them on the box, and never commit, upload, or redistribute them.**

**Each voice has its own character.** A character is a few lines on who the agent is: its name, personality and
humour (`persona.characters` in `config/prompts.yaml`). There are four, one per voice: **Mira** (the default) and
**Aria**, with women's voices, and **Kai** and **Rex**, with men's. Their clips are `voices/mira.wav`, `aria.wav`,
`kai.wav` and `rex.wav`. `voices.character` gives every voice its character (`mira`), and `voices.characters` maps a
voice to another one (`aria: aria`); any other clip in `voices/` speaks as Mira. The page shows the characters as
tabs (the classic page as a picker, "Mira (mira)"). Picking another one switches the character from the next turn,
and the conversation goes on.

### 4. Configure

Everything you edit to configure the stack is in `config/` (no secrets: those are in `.env`):

```
config/
  default.yaml        # the stack: plugins and their options, thresholds, voices, MOUTH's limits (run.sh's default)
  fake.yaml           # the stack on the fake plugins: no GPU, no downloads (CONFIG=config/fake.yaml ./run.sh)
  prompts.yaml        # every prompt MOUTH sends to an LLM
  engines/
    voxcpm2.yaml      # vLLM-Omni's own engine config for VoxCPM2 (batch, KV budget), named by tts.engine_config
```

Paths inside a config are relative to the repo root. `CONFIG=<file> ./run.sh` runs another config. Worth knowing:

- **Prompts** live in `config/prompts.yaml`, not in code: system prompts, few-shot turns, the small templates MOUTH
  fills in, and the tool descriptions. It is checked at MOUTH start (a bad key or placeholder stops start-up with
  the key named) and read once, so restart MOUTH after editing. `mouth.prompts` can point at a copy for an
  experiment. Keep few-shot examples synthetic: the file is committed.
- **Endpoint thresholds** (`ears.ep_thr` etc.) ship at the training-time defaults and have **not** been
  retuned on a live browser mic — offline thresholds have been seen to run ~2× too high on real mic signal.
  Expect to tune these live with the page's sliders, then freeze the values back into `config/default.yaml`.
- **Backchannels** are always **contextual LLM acks**: a short reaction the LLM generates from the running
  ASR partial ("oh nice", "wow, really?"). When you've said too little to react to (`mouth.bc_min_words`),
  it uses a neutral "mm-hm". How *often* they fire is EARS's `ears.bc_thr` (a slider on the page).

## Run & reach it

```bash
./run.sh     # builds the page if needed, starts MOUTH (both engines + TTS warmup), waits for it, then EARS on :7860
CONFIG=/path/to/other.yaml ./run.sh   # alternate config for both processes
```

Everything is on **one port, :7860**: the page at `/`, the classic page at `/classic`, the voice list at `/voices`
and the conversations' WebSocket at `/ws`.

Startup from a warm cache is ~2.1 min (VoxCPM2 ~45 s, Gemma 4 E4B ~40 s, TTS warmup incl. batch sizes 2–4
~25 s, EARS ~15 s). The very first start also downloads the models (~25 GB) and pays one-time compiles (VoxCPM2's
first warmup ~2 min, Gemma's torch.compile + CUDA-graph capture ~1.5 min), which are cached for later starts.

Browsers only allow the mic over **HTTPS or localhost**, so either:

- **SSH port-forward** (simplest): `ssh -L 7860:localhost:7860 <box>`, then open `http://localhost:7860`.
- **HTTPS** behind a reverse proxy (e.g. Caddy: `tyger.example.com { reverse_proxy localhost:7860 }`). The
  page upgrades the WebSocket to `wss` automatically when served over HTTPS.

**The page** (`/`, built from `web/`):
- one mic button, and the characters as tabs;
- an orb in the middle ([OrbKit](https://github.com/zzzzshawn/orbkit)'s Ion, MIT, from the
  [Agents Kit](https://agents-ui.github.io/agents-kit/voice) registry). Its state follows the conversation (idle,
  thinking, speaking), and it reacts to your voice while you talk and to the agent's while it speaks. Each
  character has its own colours;
- the exchange as text: your words as you say them, the agent's backchannels in a bubble beside them, its reply with
  the sources it cited; and the whole conversation in a history;
- the live tuning in a panel over the page ("Tuning"): the meters with their thresholds and the sliders (endpoint τ /
  debounce, barge, backchannel), so you can tune while you talk (O-01).

It's a Next.js app exported to static files (`web/out/`), which EARS serves. **`run.sh` builds it** (`npm ci`, then
`npm run build`) when the build is missing or older than the page's sources, and installs the page's packages only
when they're missing or the lockfile changed. That needs Node ≥ 22 with `npm` on the PATH; without it, `run.sh` says
so and starts anyway, and `/` serves the classic page until the page is built. To build it by hand:
`cd web && npm ci && npm run build`. EARS picks up a new build without a restart.

**The classic page** (`/classic`, `static/index.html`, linked from the page as "Classic view") is the plain one: a
start button, the voice picker, the meters and the tuning sliders (`?nodebug` hides the sliders). Both pages run the
same audio code and the same WebSocket, with the browser's echo cancellation, noise suppression and auto gain on.

**Working on the page:** run it in dev mode beside the stack, with hot reload:

```bash
cd web && npm install && npm run dev   # http://localhost:3000; /ws, /voices and /classic pass through to EARS on :7860
```

Forward port 3000 the same way as 7860. Run `npm run build` (or restart `run.sh`) when you're done, so :7860 serves
the change.

## API mode

To use tyger from another app (its own page, e.g. a Next.js app, calling this server), run it without its pages and
let only that app in, with a token its backend hands out. Two settings are in `config/default.yaml`'s `server:` block
and each has an env var, which wins over the file (Docker sets them when it starts the container); the token's secret
is an env var only. EARS logs the values it runs with at startup.

| Setting | Env var | Default | What it does |
|---|---|---|---|
| `server.pages` | `TYGER_PAGES` | `true` | `false`: no pages, only `/ws` and `/voices` (`/` and `/classic` are 404s), and `run.sh` builds nothing, so no Node is needed |
| `server.max_session_s` | `TYGER_MAX_SESSION_S` | `180` | a conversation's length in seconds; `0`: no limit |
| — | `TYGER_API_SECRET` | unset | the token's secret; env only, never in the config file, never logged |

For example: `TYGER_PAGES=false TYGER_API_SECRET=... ./run.sh`, or the same as `-e` options to `docker run`.

**The token is the only protection.** EARS doesn't check which site a page comes from: any page (and any script)
can connect, and `/voices` answers CORS for any page. **Without `TYGER_API_SECRET`, anyone who can reach the port can
open a conversation, so set the secret before the server is reachable from the internet.**

**The token.** With `TYGER_API_SECRET` set, every conversation needs a token, made by your app's backend for a user it
has signed in, and handed to the browser:
- a JWT signed with HS256 and the same secret, carrying `aud: "tyger"`, `exp` at most 5 minutes ahead and a unique
  `jti`; optionally `origin`, which must then match the page's origin exactly (it ties a token to your app's page);
- sent as `?token=…` on `/ws` (browsers can't set headers on a WebSocket). EARS never logs it, nor any query string;
- one token opens one conversation (EARS remembers its `jti` until it expires); a refused one gets a 401 before the
  upgrade (bad signature, another algorithm or `none`, wrong audience, expired, `exp` too far ahead, origin mismatch,
  `jti` used);
- `/voices` needs a valid token too (`/voices?token=…`); checking it there doesn't use it up. The `ready` message
  carries the voice list anyway.

The pages EARS serves have no way to get a token, so with the secret set, run with `TYGER_PAGES=false` (EARS warns at
startup otherwise). For development, leave the secret unset: then there's no token check.

**The session limit.** `ready` carries `max_session_s`, so a page can show a countdown. At the limit no new turn
starts, a reply in progress may finish (10 s at most), then EARS sends `{"type": "session_end", "reason":
"time_limit"}` and closes. The conversation's background tasks end with it, as on any end. Both pages and the client
say "Time's up — start again", and a new conversation can start. Your own test sessions are limited too: turn it off
with `TYGER_MAX_SESSION_S=0`.

**Size limits.** A WebSocket message over 64 KB closes the connection (1009); a 20 ms mic block is 648 bytes. Mic
audio piling up more than 5 s ahead of real time (sent faster than it's spoken) closes it too, with
`{"type": "error", "error": "audio_too_fast"}`.

## Using tyger from your own app

**The client.** `web/lib/tyger/client.ts` is tyger's browser client: one file, no dependencies, no framework, used by
tyger's own page too. Copy the `web/lib/tyger/` folder into your app (or import it). It captures the mic, plays the
agent, and talks to the server:

```ts
import { TygerClient } from "@/lib/tyger/client"

const tyger = new TygerClient({
  url: "wss://tyger.example.com/ws",                          // default: the page's own origin, /ws
  token: async () => {                                        // called at each start; omit it without a secret
    const r = await fetch("/api/tyger-token", { method: "POST" })
    if (!r.ok) throw new Error(`token: ${r.status}`)
    return (await r.json()).token
  },
  voice: "mira",                                              // the character to talk to (optional)
  on: {
    ready: (m) => showVoices(m.voices, m.labels, m.max_session_s),   // m.max_session_s: for a countdown
    partial: (words) => showUserWords(words),                 // the user's words so far
    transcript: (text) => addUserTurn(text),                  // the user's turn, whole
    reply: (text) => showReply(text),                         // the reply so far: replaces the last
    links: (links) => showSources(links),                     // {label, url}: shown, never spoken
    close: (why) => ended(why),                               // "" after stop(); else why ("Time's up — start again.")
    error: (why) => couldNotStart(why),                       // no mic, no token...
  },
})
await tyger.start()      // on a click: the browser asks for the mic
tyger.setVoice("rex")    // switch characters from the next turn
tyger.stop()
```

The events: `open`, `ready` (the voices, their labels, the session limit), `partial`, `transcript`, `reply`, `links`,
`backchannel` and `backchannelDone` (the agent's "mm-hm"s), `barge` (the user cut in), `turnDone`, `taskDone` (a
background task's result is said next), `sessionEnd`, `playing` (the agent's audio started / ended), `close`,
`error`, and `message` with every JSON message as it came. The voice list is in `ready`; before a conversation,
`GET /voices` (with `?token=…` when the secret is set) returns `{"voices", "voice", "labels"}`.

**What a client must do,** and this one does all three:
- tag every 20 ms mic block with the reply sample it was playing when the block was captured;
- report `played:<id>:<samples>` once a reply has played out (that ends the agent's turn);
- stop playback at once on `barge`.

**It must play the agent's audio itself,** on the same AudioContext as its mic capture. That is what makes the tags
right (DEVELOPING.md → "The 2nd ear"). An app that plays the audio another way (an `<audio>` element, another
context) breaks them, and barge-in then takes the agent's own echo for the user cutting in.

**HTTPS.** Browsers give the mic only to pages on HTTPS (or localhost), and an HTTPS page can only open `wss://`, so
put the server behind TLS too (e.g. Caddy: `tyger.example.com { reverse_proxy localhost:7860 }`).

**The token, from your backend.** A Next.js route handler that checks your app's own session, then signs a token with
[`jose`](https://github.com/panva/jose) (`npm install jose`). The secret is a server-only env var with the same value
as the server's `TYGER_API_SECRET`:

```ts
// app/api/tyger-token/route.ts
import { SignJWT } from "jose"
import { getSignedInUser } from "@/lib/auth"                    // your app's own session check

const secret = new TextEncoder().encode(process.env.TYGER_API_SECRET)   // server-only: never NEXT_PUBLIC_

export async function POST() {
  const user = await getSignedInUser()
  if (!user) return new Response("Sign in first", { status: 401 })
  const token = await new SignJWT({ origin: "https://app.example.com" })   // optional: only this page may use it
    .setProtectedHeader({ alg: "HS256" })
    .setAudience("tyger")
    .setJti(crypto.randomUUID())                                 // one token, one conversation
    .setIssuedAt()
    .setExpirationTime("60s")                                    // at most 5 minutes
    .sign(secret)
  return Response.json({ token })
}
```

**The protocol,** for a client of your own (`app/ears/playback.py` has the details). The WebSocket is `/ws`
(`?token=…` when the server checks tokens). All binary frames are little-endian, with int16 audio:
- **mic → EARS:** `[int32 reply id][int32 reply sample, or -1][int16 mic samples at 16 kHz]`. The tag is the reply
  sample (24 kHz) the user was hearing at the block's first mic sample, or -1 when no reply was playing. The page
  sends 20 ms blocks, captured with an AudioWorklet on the same AudioContext that plays the reply.
- **audio → page:** `[int32 reply id][int16 samples at 24 kHz]`. Replies and announcements are numbered from 1
  on each connection, so a client clears its per-reply counts when it connects again. Id 0 is a backchannel,
  which the page plays but never tags.
- **text → EARS:** `played:<id>:<samples>` when a reply's queue has drained; `reset`, `voice:<name>`,
  `set:<threshold>:<value>`.
- **JSON → page:** `ready` (with `voices`, `labels` and `max_session_s`), `state`, `partial` (the user's words so far
  in their turn, when they change), `endpoint` (with the user's `transcript`), `reply`, `links`, `backchannel` /
  `backchannel_done` (`aborted`: dropped unvoiced), `barge`, `turn_done`, `task_done`, `session_end` (`reason`:
  `time_limit`; the socket closes next), `busy` and `error` (`mouth_lost`, `audio_too_fast`; the socket closes next),
  and a few more.

## Plugins

The LLM, the TTS and the turn-taking model are **plugins chosen in `config/default.yaml`** (`app/plugins/`). Each
block names its `plugin:`; the rest of the block is that plugin's options.

| kind | plugin | runs | notes |
|---|---|---|---|
| LLM | `gemma-4-e4b` | local, vLLM in MOUTH | **the responder**, `google/gemma-4-E4B-it` with `quantization: fp8`: thinking off, text only, Gemma tool-call tokens |
| LLM | `openai-responses` | **hosted** | **the `smart` role**: OpenAI's Responses API with its web search (escalations, live facts, background tasks) |
| LLM | `anthropic` | **hosted** | the alternative `smart`: the Claude Messages API; default `claude-opus-5-5`, `effort: low`, server-side refusal fallback |
| TTS | `voxcpm2` | local, vLLM-Omni in MOUTH | cloned voices from `voices/` |
| EARS | `nemotron-tyger-1.0` | local, in EARS | nemotron ASR + the tyger 1.0 turn-taking heads |
| all three | `fake` | no model, CPU | scripted / energy-based; tests and `CONFIG=config/fake.yaml ./run.sh` |

Ten other responders were measured against Gemma 4 E4B and lost (DEVELOPING.md → "Benchmarks"); their plugins are
in git history.

**LLM roles.** `llm.reply` answers turns, and the answer after a tool round. `llm.filler` makes backchannels, tool
lead-ins and still-looking lines; `filler: reply` (the default) shares one instance. `llm.smart` (optional) answers
escalated turns and runs background tasks. Any role can use any LLM plugin; two blocks with the same plugin need a
`name:` each. Keys live in `.env`, under the variable the block's `api_key_env` names.

Hosted `reply` and `filler` roles were measured and left local: a hosted `reply` added ~0.35 s to first audio, and a
hosted `filler` made backchannels ~0.55 s slower (2026-10-04). A hosted TTS was tried and removed: every
text chunk paid a request's time to first audio. A hosted request that fails (rate limit, timeout, network) ends
the turn with a short spoken apology, and a refusal with a short neutral line (both in `config/prompts.yaml` →
`spoken`).

Adding a plugin or a tool: DEVELOPING.md.

## Escalation and background tasks

With an `llm.smart` block, the local `reply` model can hand work to a bigger hosted model. It decides itself, by
calling a tool, so easy turns pay nothing extra:

- **`ask_expert`** for a question that needs care now: maths, code, health / money / legal advice, comparisons,
  planning, long explanations, or "think carefully"; and for live facts (weather, news, prices, scores), which the
  smart model looks up with OpenAI's web search (`openai-responses`, `web_search: true`). The lead-in fits ("let me
  think about that", "let me pull up the Lisbon forecast"), the smart
  request starts the moment the call is read, and its answer streams straight to the voice (live: answer audio at
  ~1.0–1.3 s, under the lead-in). Follow-ups on the same topic go to it too. The web search's citations are cut
  out of the answer: they show as links under it on the page and are never read out.
- **`start_task`** for work that takes minutes: research, comparing many options, a multi-day plan, a long draft.
  The local model says it'll look into it and the conversation carries on. The task runs on the smart model in the
  background (its own web searches, `mouth.tasks.max_steps` rounds, `max_wall_s`). When it finishes, EARS gives it
  the floor once the user has been quiet for `announce_after_silence_ms`, and it speaks a short summary; the user
  can barge in as usual. "How's it going?" is answered locally from a status note; "drop it" cancels it.
- **A round the user cuts in on** before hearing any of its answer (during the lead-in or a still-looking line)
  isn't lost: it goes on as a background task of the conversation, counted against the task caps. The task gets
  the expert's question, or the user's words with what the round already looked up, and is announced at the next
  pause. Where no task can start (tasks off, a cap reached), history keeps the round's results instead.

```yaml
llm:
  smart: {plugin: openai-responses, model: gpt-5.4-mini, max_tokens: 1024, web_search: true, reasoning_effort: low,
          api_key_env: OPENAI_API_KEY}
mouth:
  escalation: {context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_global: 4, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

Routing is measured by `scripts/bench/escalation.py` (local model only, no cost): with Gemma 4 E4B, all 119
synthetic cases × 8 samples go the right way, and every live-fact question goes to the expert (DEVELOPING.md →
"Benchmarks"). Cost on ordinary turns: the longer prompt adds ~30 ms (expert) and ~90 ms
(expert + tasks) to first audio. Without `llm.smart` neither tool is offered and the prompt is the plain one.
Tasks and live facts need the smart model's web search (`openai-responses`); each search is billed on top of the
tokens. Without `llm.smart` the agent answers what it knows and says plainly that it can't check live facts.

## Models & licences

| component | model | licence |
|---|---|---|
| ASR encoder | `nvidia/nemotron-3.5-asr-streaming-0.6b` @ `ea30d66` (the revision the heads release pins) | OpenMDW-1.1 |
| turn-taking heads | `nineninesix/TYGER-S2S-endpointer` v0.1.0-rc1 (private; Qwen3-0.6B decoder with its LoRA merged + the heads) | Apache-2.0 (modified Qwen3-0.6B: see its NOTICE) |
| responder | `google/gemma-4-E4B-it`, FP8 at load (was `Qwen/Qwen3.5-4B`, before that `Qwen/Qwen3-4B-Instruct-2507`) | Apache-2.0 |
| TTS | `openbmb/VoxCPM2` | Apache-2.0 |
| expert, background tasks, web search | OpenAI `gpt-5.4-mini` with OpenAI's web search (`llm.smart`) | commercial API (your key) |

## Privacy

No audio or transcript is written to disk or logs. Logs carry only timings and event types (endpoint source,
onset ms, barge, backchannel, tool name). EARS's request log has paths, never query strings, so a conversation's
token isn't logged either. Voice reference clips and their transcripts stay on the box. The page gets
the user's own words back (as they speak, on `partial`, and each turn's transcript, on `endpoint`) to show them.

**Hosted plugins send text off the box**, and only because the config names one: a hosted `reply` gets every
turn's transcript, the conversation history and tool results; a hosted `filler` gets the user's question (tool
lead-ins) and the running ASR partial (backchannels). A `smart` model gets only escalated turns and background
tasks: the user's words, the question or goal, and the last `context_turns` turns as plain text (tool results
left out), plus its own tool results, and for a round the user cut in on, what that round looked up. Audio and
voice clips are never sent. The startup log names
the remote roles; request and response bodies are never logged, and MOUTH refuses to start with the SDKs' debug
logging on (`OPENAI_LOG` / `ANTHROPIC_LOG` = `debug`). Run hosted plugins only with the provider terms you accept
for that data.

## Known limits (v1)

- **Up to 4 conversations** per GPU (`server.max_sessions`); more visitors get a busy page. On a healthy 5090, 4
  conversations keep EARS real time (hop lag p50 47–87 / p90 124–170 ms on two boxes).
- **A conversation ends after 3 minutes** by default (`TYGER_MAX_SESSION_S`, 0 for no limit).
- **No per-user or per-IP cap:** whoever holds tokens (or reaches a server without a secret) can take all 4 slots.
  Your app's backend decides who gets a token.
- **English only:** every prompt asks for English, so the agent answers in English whatever language it hears.
- `set_reminder` / `send_message` are **placeholders** — the model says they aren't connected yet.
- The user waits **~0.8–1.0 s** from the end of their speech to the first reply audio at 1 conversation and
  **~1.0–1.1 s** at 4 (two RTX 5090 boxes, `scripts/stack/load_ears.py`), and ~600–770 ms of that is endpoint
  detection. Live threshold tuning should cut it (ISSUES.md, O-01). MOUTH's part is small: first audio ~175–235 ms
  after the endpoint on a plain turn (smoke test, cloned voice), 400–520 ms p50 with 4 busy sessions.
- Endpoint thresholds are not yet live-tuned (see Configure).
- The `Dockerfile` has not yet been built and run end to end (the box used so far is itself a container, with no
  Docker daemon). The non-Docker fresh-box path above has been run end to end on a clean box.
