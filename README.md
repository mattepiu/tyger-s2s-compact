# Tyger-S2S

[![Discord](https://img.shields.io/badge/Discord-Join_the_community-5865F2)](https://discord.gg/NzP3rjB4SB)
[![Website](https://img.shields.io/badge/Website-NineNineSix-111827)](https://www.nineninesix.ai/)
[![Model: nineninesix/tyger](https://img.shields.io/badge/Model-nineninesix%2Ftyger-FFD21E)](https://huggingface.co/nineninesix/tyger)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

https://github.com/user-attachments/assets/24bba1bb-94f2-4f3b-b592-595e552c5fc0

Tyger-S2S is a voice agent: an AI you talk to out loud in your browser, like on a phone call. It keeps listening while it talks, so the conversation feels natural:

- it answers as soon as you've finished;
- it stops when you interrupt;
- it says "mm-hm" while you're talking.

Everything else runs on one graphics card (GPU).

Its turn-taking model, [`nineninesix/tyger`](https://huggingface.co/nineninesix/tyger), is trained elsewhere and downloaded from Hugging Face.

**More docs:** [DEVELOPING.md](DEVELOPING.md) (how it works inside, and how to change it),
[CHANGELOG.md](CHANGELOG.md) (what changed), and [ISSUES.md](ISSUES.md) (what's left to do).

## How it works

Tyger-S2S has two parts, EARS and MOUTH, both on the same GPU:

```
your mic ──▶ EARS    listens: turns your speech into text, and decides when to answer,
              │      when to stop, and when to say "mm-hm". It also serves the web page.
              ▼
             MOUTH   talks: writes the reply (Gemma 4 E4B) and speaks it (VoxCPM2)
              │
your speaker ◀┘      the reply plays while it's still being made
```

- **EARS** checks several times a second. It also hears the agent's own voice, so the agent's echo in your mic isn't mistaken for you.
- **MOUTH** speaks one sentence at a time, so you hear the start before the rest is written. Before using a tool, it says a short line ("Checking the time in Tokyo"). If you interrupt, it stops at once and remembers only what you heard.

DEVELOPING.md has the details.

## What you need

- **An NVIDIA GPU with ~28 GB of memory** for the default models at 4 conversations (Gemma 4 E4B FP8 ~15 GB, VoxCPM2
  ~9 GB, EARS ~4 GB), so a 32 GB card is enough. Other models need more or less. Tested on an RTX 5090, RTX PRO 5000
  and RTX PRO 4500 (slower: see "Known limits"); other GPUs should work, but aren't tested.
- **An NVIDIA driver for CUDA 13** (version 590 or newer). That's what we test on; CUDA 12 should work too, but isn't
  tested.
- **About 60 GB of free disk.** The models take about 25 GB of it.
- **Python 3.12 and [uv](https://docs.astral.sh/uv/)**, a fast Python installer. **Node 22 or newer** is optional:
  it builds the new web page.
- **A Hugging Face token** (a free account is enough) and, if you want the expert, an **OpenAI key**.

## Quick start

### 1. Install

In the repo's folder:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=auto
```

### 2. Keys

Copy `.env.example` to `.env` and fill it in.

| Variable | What it's for |
|---|---|
| `HF_TOKEN` | Downloads [`nineninesix/tyger`](https://huggingface.co/nineninesix/tyger). The model is public, but Tyger-S2S still asks for a token for now: any account's token works. Not needed if a token is already saved in `$HF_HOME/token`. |
| `OPENAI_API_KEY` | The expert: hard questions, live facts from the web, and longer background tasks. No key? Remove `llm.smart` from the settings, and the agent says it can't look things up. |
| `ANTHROPIC_API_KEY` | Only if you use Claude as the expert instead. |
| `TYGER_API_SECRET` | Only if Tyger-S2S runs behind your own app ("API mode", below). |

### 3. Voices

The repo comes without voice clips: add your own `.wav` files to `voices/`. The file name becomes the voice's name
(`voices/mira.wav` is `mira`). For a better copy of the voice, write the clip's exact words in a text file next to it
(`voices/mira.txt`). Without one, only the first 10 seconds of the clip are used.

Each voice speaks as a character, described in `config/prompts.yaml`: **Mira** (the default), **Aria**, **Kai** and
**Rex**, for `mira.wav`, `aria.wav`, `kai.wav` and `rex.wav`.

**A real person's voice is personal data.** Keep their clip on your own machine, and don't commit or share it. (The
four voices in the public Docker image are synthetic, not real people.)

### 4. Run

```bash
./run.sh
```

Then open http://localhost:7860 and click the mic. To stop Tyger-S2S, run `./stop.sh`.

- **The first start** downloads the models (~25 GB) and prepares them for your GPU: about 10 minutes or more. After
  that, Tyger-S2S starts in about 2 minutes.
- **With Node installed,** `run.sh` also builds the new web page. Without Node, you get the classic page.
- **No GPU?** `CONFIG=config/fake.yaml ./run.sh` runs everything on fake models, for testing, with nothing to
  download.

**Running it on another computer, like a server?** Browsers only allow the mic on `https://` pages and on
`localhost`. So either forward the port over SSH (`ssh -L 7860:localhost:7860 <server>`) and open
http://localhost:7860 on your own computer, or put Tyger-S2S behind HTTPS: a web server such as Caddy
(`tyger.example.com { reverse_proxy localhost:7860 }`), or a Cloudflare Tunnel (see "RunPod + Cloudflare").

## The page

Everything is on port 7860.

- **The main page** (`/`) has a mic button and the characters as tabs. Your words appear as you speak, with the agent's "mm-hm"s beside them. **Tuning** opens live meters and sliders for the turn-taking settings (see "Settings").
- **The classic page** (`/classic`) is a simpler one: a start button, a voice picker, meters and sliders. Add
  `?nodebug` to its address to hide the sliders.

Also on the port: the list of voices at `/voices`, and the conversations themselves at `/ws`.

The orb is [OrbKit](https://github.com/zzzzshawn/orbkit)'s Ion (MIT licence), via
[Agents Kit](https://agents-ui.github.io/agents-kit/voice). The page is a Next.js app in `web/`. To change it and see your edits live while Tyger-S2S runs:

```bash
cd web && npm install && npm run dev    # http://localhost:3000, talking to EARS on :7860
```

When you're done, run `npm run build` (or restart `run.sh`), so that port 7860 serves your change.

## Settings

Settings are in `config/`; keys stay in `.env`.

```
config/
  default.yaml           models, turn-taking, voices, limits (what run.sh uses)
  fake.yaml              fake models: no GPU, no downloads
  prompts.yaml           everything the AI models are told
  engines/voxcpm2.yaml   the voice engine
```

To use another settings file: `CONFIG=<file> ./run.sh`. If a setting is wrong, Tyger-S2S doesn't start, and it tells you which one.

- **Prompts** are all in `config/prompts.yaml`. Restart server after editing them.
- **Turn-taking:** three numbers between 0 and 1. Lower makes the agent quicker to act; higher makes it wait until
  it's surer.
  - `ears.ep_thr`: when to answer. Lower answers sooner, but may cut you off mid-sentence.
  - `ears.barge_thr`: when to stop for you. Lower lets you interrupt more easily, but small sounds may stop it too.
  - `ears.bc_thr`: when to say "mm-hm". Lower means more "mm-hm"s.

  They aren't fully tuned for real microphones yet. The easiest way to tune them: move the page's sliders while you talk, then copy the values you like into `config/default.yaml`.
- **"Mm-hm"s** are short reactions written from what you've said so far ("oh nice", "wow, really?"), or a plain "mm-hm" when you haven't said enough yet (`mouth.bc_min_words`).
- **Speaking sooner:** `mouth.first_chunk: clause` starts the voice at the first comma instead of the end of the first sentence. That's 30–95 ms sooner, but the tone of voice can change mid-sentence.

## Docker

The easiest way to run Tyger-S2S on a server. The public image comes with the four synthetic voices:

```bash
docker run -d --name tyger --restart unless-stopped --gpus all \
  --env-file .env -p 127.0.0.1:7860:7860 \
  -v tyger-hf:/hf -v tyger-cache:/root/.cache -v tyger-triton:/root/.triton \
  ylankgz/tyger-s2s-vllm:36da1bd
docker logs -f tyger      # ready when it says "EARS ready"
```

- **You need** an NVIDIA GPU with a CUDA 13 driver, Docker with the NVIDIA Container Toolkit, and ~60 GB of disk.
- **`.env`** needs `HF_TOKEN` and `OPENAI_API_KEY`. No OpenAI key? Copy `config/default.yaml` to `local.yaml`, delete
  `llm.smart` and `mouth.tasks`, and add `-v "$PWD/local.yaml":/app/config/local.yaml:ro -e CONFIG=config/local.yaml`.
- **The volumes** keep the models, so later starts take ~2 minutes instead of ~11.
- **The port** only opens on this machine. To reach it from elsewhere, put HTTPS in front ("RunPod + Cloudflare").
- **Test:** `docker exec tyger .venv/bin/python scripts/stack/smoke.py`. **Stop:** `docker stop -t 30 tyger`.
- The image has only the classic page (the new one needs Node to build).

**Your own image,** for example with your own voices from `voices/`:

```bash
docker build -t tyger .
docker login -u <you>
docker tag tyger <you>/tyger-s2s-vllm:<tag>
docker push <you>/tyger-s2s-vllm:<tag>
```

New Docker Hub repositories are public by default: if your image has a real person's voice, make it private first.

## RunPod + Cloudflare

How we run Tyger-S2S online (tested on an RTX PRO 4500): the Docker image on a RunPod GPU, a Cloudflare Tunnel for a
fixed HTTPS address with no open ports, and your app on Cloudflare Workers. RunPod's own proxy won't do: it drops
connections after 100 seconds.

```
browser ── https ──▶ your app (Cloudflare Workers): makes a token
   └────── wss://tyger.example.com/ws?token=… ──▶ Cloudflare ── tunnel ──▶ the GPU, port 7860
```

1. **Create the tunnel** (Cloudflare: Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared). Copy its token
   (after `--token` in the install command), keep it secret, and add a public hostname: `tyger.example.com`, type
   `HTTP`, URL `localhost:7860`.
2. **Rent a GPU** (RunPod: Pods → Deploy): an RTX 5090 or RTX PRO 4500 with CUDA 13, near your users; the image
   `ylankgz/tyger-s2s-vllm:36da1bd`; a ~30 GB container disk and a ~60 GB volume at `/workspace`; **no exposed
   ports**; and these variables (secrets go in RunPod Secrets, used as `{{ RUNPOD_SECRET_<name> }}`):

   | Variable | Value |
   |---|---|
   | `CLOUDFLARE_TUNNEL_TOKEN` | the tunnel's token (secret): Tyger-S2S starts the tunnel itself |
   | `TYGER_API_SECRET` | the token secret (secret), the same as your app's |
   | `HF_TOKEN`, `OPENAI_API_KEY` | your keys (secret) |
   | `TYGER_PAGES` | `false` |
   | `HF_HOME` | `/workspace/.hf_home` |
   | `XDG_CACHE_HOME` | `/workspace/.cache` |
   | `FLASHINFER_WORKSPACE_BASE` | `/workspace` |
   | `TRITON_CACHE_DIR` | `/workspace/.cache/triton` |

   The last four keep the models on the volume: the first start takes ~11 minutes, later ones ~2.
3. **Check it** once the logs say "Registered tunnel connection":
   `curl -s -o /dev/null -w '%{http_code}\n' https://tyger.example.com/voices`. `401` means it works (it wants a
   token); `530`: the tunnel isn't up, or its token is wrong; `502`: the tunnel can't reach port 7860.
4. **Connect your app** ("API mode" below). On Workers, set the secret with
   `npx wrangler secret put TYGER_API_SECRET` (`.dev.vars` locally) and read it with
   `getCloudflareContext().env.TYGER_API_SECRET`.
5. **Optional:** limit how often one IP starts conversations, with a Cloudflare rate limiting rule on `/ws` (Security →
   WAF).

Good to know:
- Keep `cloudflared` at its default log level: at debug level it logs URLs, and they contain the tokens.
- A stopped pod keeps its volume (and its bill), but may not get its GPU back.

## API mode: Tyger-S2S behind your own app

Make Tyger-S2S the voice of your own web app: its pages are off, and only your signed-in users can talk to it.

| Setting | Variable | Default | What it does |
|---|---|---|---|
| `server.pages` | `TYGER_PAGES` | `true` | `false`: no pages, only `/ws` and `/voices` (and no need for Node) |
| `server.max_session_s` | `TYGER_MAX_SESSION_S` | `180` | how long a conversation can last, in seconds; `0`: no limit |
| — | `TYGER_API_SECRET` | not set | the secret that tokens are signed with |

The variables override `config/default.yaml`, e.g. `TYGER_PAGES=false TYGER_API_SECRET=... ./run.sh`. The secret is
never logged.

**Set `TYGER_API_SECRET` before Tyger-S2S is reachable from the internet:** the token is the only lock, so without a
secret anyone who finds the address can talk to it.

**Tokens.** Your app's server signs a token for each signed-in user, and the browser connects to `/ws?token=…`. A
token is a JWT signed with HS256 and the same secret, with:
- `aud: "tyger"`;
- `exp` at most 5 minutes ahead;
- a unique `jti`: one token, one conversation;
- optionally `origin`: then it only works from that website.

A bad token gets a `401`; `/voices` needs one too, but doesn't use it up. The built-in pages can't get tokens, so set
`TYGER_PAGES=false` along with the secret. While developing, leave the secret unset.

**Limits.** When time's up, a reply already playing may finish, then Tyger-S2S sends `session_end` and hangs up. A
message over 64 KB, or mic audio more than 5 s ahead of real time, also ends the connection.

### The client

`web/lib/tyger/client.ts` is Tyger-S2S's browser code: one file with no dependencies, and Tyger-S2S's own page uses it too.
Copy the `web/lib/tyger/` folder into your app:

```ts
import { TygerClient } from "@/lib/tyger/client"

const tyger = new TygerClient({
  url: "wss://tyger.example.com/ws",                  // default: this page's own /ws
  token: async () => {                                // called at each start; leave it out without a secret
    const r = await fetch("/api/tyger-token", { method: "POST" })
    if (!r.ok) throw new Error(`token: ${r.status}`)
    return (await r.json()).token
  },
  voice: "mira",                                      // the character to talk to (optional)
  on: {
    ready: (m) => showVoices(m.voices, m.labels, m.max_session_s),
    partial: (words) => showUserWords(words),         // the user's words so far
    transcript: (text) => addUserTurn(text),          // the user's whole turn
    reply: (text) => showReply(text),                 // the reply so far: each one replaces the last
    links: (links) => showSources(links),             // {label, url}: shown, never spoken
    close: (why) => ended(why),                       // "" after stop(); else why ("Time's up — start again.")
    error: (why) => couldNotStart(why),               // no mic, no token...
  },
})
await tyger.start()      // on a click: the browser asks for the mic
tyger.setVoice("rex")    // switch characters from the next turn
tyger.stop()
```

Its events: `open`, `ready` (the voices and the time limit), `partial`, `transcript`, `reply`, `links`,
`backchannel` and `backchannelDone` (the "mm-hm"s), `barge` (the user interrupted), `turnDone`, `taskDone`,
`sessionEnd`, `playing`, `close`, `error`, and `message` (every message, as it came).

**Let the client play the agent's voice itself.** It plays the audio in step with the mic, and tells Tyger-S2S what the
user was hearing at each moment: that's how Tyger-S2S tells the user's voice from the agent's echo. If you play the audio
another way (an `<audio>` element, another AudioContext), this breaks, and the agent keeps cutting itself off.

**HTTPS:** the mic only works on an `https://` page, and an `https://` page can only connect to `wss://`, so Tyger-S2S
needs HTTPS too.

### The token route

A Next.js route that checks your app's own login, then signs a token with [`jose`](https://github.com/panva/jose). The
secret is a server-only setting, with the same value as Tyger-S2S's:

```ts
// app/api/tyger-token/route.ts
import { SignJWT } from "jose"
import { getSignedInUser } from "@/lib/auth"                    // your app's own login check

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

### The protocol

Only needed if you write your own client instead of `client.ts` (`app/ears/playback.py` has the details). Binary
messages are little-endian, with 16-bit audio:
- **mic → Tyger-S2S:** `[int32 reply id][int32 reply sample, or -1][int16 mic audio at 16 kHz]`, in 20 ms blocks. The tag
  is the reply sample (24 kHz) the user was hearing when the block started, or -1 when nothing was playing.
- **Tyger-S2S → speaker:** `[int32 reply id][int16 audio at 24 kHz]`. Replies are numbered from 1 on each connection.
  Id 0 is a "mm-hm": play it, but never tag it.
- **text → Tyger-S2S:** `played:<id>:<samples>` once a reply has finished playing (this ends the agent's turn); also
  `reset`, `voice:<name>` and `set:<threshold>:<value>`.
- **JSON → page:** `ready` (`voices`, `labels`, `max_session_s`), `state`, `partial`, `endpoint` (with the user's
  `transcript`), `reply`, `links`, `backchannel`, `backchannel_done`, `barge`, `turn_done`, `task_done`, and a few
  more. After `session_end`, `busy` or `error`, the connection closes.

On `barge`, a client must stop playing the reply at once.

## Plugins

The language model (LLM), the voice and the turn-taking model are plugins: you pick them in `config/default.yaml`,
and their code is in `app/plugins/`. Each block names its `plugin:`, then that plugin's options.

| Kind | Plugin | Runs on | What it is |
|---|---|---|---|
| LLM | `gemma-4-e4b` | your GPU | **answers you:** `google/gemma-4-E4B-it` in FP8, text only, no "thinking" step |
| LLM | `openai-responses` | online | **the expert** (`smart`): OpenAI's Responses API, with web search |
| LLM | `anthropic` | online | another expert: Claude (`claude-opus-5-5` by default) |
| Voice | `voxcpm2` | your GPU | voices cloned from `voices/` |
| EARS | `tyger` | your GPU | speech recognition (nemotron) + a small turn-taking model, in one file (`nineninesix/tyger`) |
| all three | `fake` | CPU, no model | for tests and `config/fake.yaml` |

Gemma 4 E4B did better than the 10 other models we measured for answering (DEVELOPING.md → "Benchmarks").

**The LLM's three jobs:**
- `llm.reply` answers each turn.
- `llm.filler` writes the "mm-hm"s and the short lines around tool use. By default it's the same model as `reply`.
- `llm.smart` (optional) is the expert.

Any job can use any LLM plugin; if two blocks use the same plugin, give each a `name:`. Keys go in `.env`, under the
name the block's `api_key_env` gives. `reply` and `filler` stay on your GPU because online models were slower: the
voice started ~0.35 s later, and "mm-hm"s ~0.55 s later. If an online request fails, the agent says a short apology;
if the model refuses, a short neutral line.

To add a plugin or a tool, see DEVELOPING.md.

## The expert and background tasks

With an `llm.smart` block, the local model can hand work to a bigger online model, the expert, only when it needs to.
Easy turns cost nothing extra.

- **`ask_expert`:** hard questions (maths, code, health, money, legal, planning) and live facts it looks up on the web
  (weather, news, prices). The agent says a short line first ("let me pull up the Lisbon forecast"), then speaks the
  answer. Sources show as links on the page, never read out.
- **`start_task`:** work that takes minutes (research, a plan, a long draft). The conversation goes on, and the agent tells you a summary at the next quiet moment. "How's it going?" gets an update; "drop it" cancels it.
- **If you interrupt** a lookup, it carries on in the background and comes up at the next pause.

```yaml
llm:
  smart: {plugin: openai-responses, model: gpt-5.4-mini, max_tokens: 1024, web_search: true, reasoning_effort: low,
          api_key_env: OPENAI_API_KEY}
mouth:
  escalation: {context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_global: 4, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

`scripts/bench/escalation.py` checks which questions go to the expert (local model only, no API cost): about 99% of
119 made-up cases go the right way, and every live-facts question goes to the expert. The longer prompt adds
~30–90 ms before the voice starts on ordinary turns. Web searches cost extra, on top of the tokens.

## Models and licences

| Part | Model | Licence |
|---|---|---|
| Speech recognition + turn-taking | [`nineninesix/tyger`](https://huggingface.co/nineninesix/tyger) | OpenMDW-1.1 |
| Answers | `google/gemma-4-E4B-it`, FP8 | Apache-2.0 |
| Voice | `openbmb/VoxCPM2` | Apache-2.0 |
| Expert, tasks, web search | OpenAI `gpt-5.4-mini`, with web search | paid API (your key) |

`config/default.yaml` pins each model to an exact version.

## Privacy

- **Nothing you say is saved.** No audio or text of the conversation goes to disk or into the logs, which only hold
  timings and event types. Tokens aren't logged either. The page gets your words back only to show them.
- **Voice clips stay where you put them:** in `voices/`, or in an image you build.
- **Online models only get text, and only if your settings use one.** The expert gets the question, your words and
  the last few turns (`context_turns`), but not tool results. An online `reply` or `filler` model would get every
  turn. Audio and voice clips are never sent.
- **What's sent to online models is never logged,** and MOUTH won't start if the OpenAI or Anthropic SDK's debug
  logging is on (`OPENAI_LOG` / `ANTHROPIC_LOG` = `debug`). Only use online models under terms you accept.

## Known limits

The numbers here come from the current models (Gemma 4 E4B FP8, VoxCPM2 and tyger) on an RTX 5090, unless a line
says otherwise. Other models or GPUs will give different ones.

- **4 conversations per GPU.** The next person gets a "busy" message.
- **Conversations end after 3 minutes** by default (`TYGER_MAX_SESSION_S`; `0` means no limit).
- **The server doesn't limit each user:** whoever has tokens can take all 4 places. Your app decides who gets a token,
  and a Cloudflare rule can limit how often one IP address starts conversations.
- **English only.**
- **`set_reminder` and `send_message` don't work yet:** the agent says they aren't connected.
- **Response time,** from when you stop talking to when the agent's voice starts: about 0.7 s on an RTX 5090 (0.75 s
  with 4 conversations), and about 0.8 s on an RTX PRO 4500 (1.15 s with 4). About 0.5 s of that is spent
  making sure you've finished; better-tuned settings should cut it.

## Licence

Tyger-S2S's code is under the [Apache License 2.0](LICENSE). The models keep their own licences (see "Models and
licences"), and the orb's files keep OrbKit's MIT licence (`web/components/voice-agents/orbkit/`).
