# tyger

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/nineninesix/tyger)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

tyger is a voice agent you talk to in the browser, as on a phone call. It listens while it talks: it knows when
you've finished, stops when you cut in, and says "mm-hm" while you speak. It answers in a cloned voice, can use tools,
and passes hard questions to a bigger hosted model. It all runs on one GPU, for up to 4 conversations at once. English
only.

This repo runs tyger. The turn-taking model is trained elsewhere.

**Docs:** this README (set it up and run it), [DEVELOPING.md](DEVELOPING.md) (how it works inside, and how to change
it), [CHANGELOG.md](CHANGELOG.md), and [ISSUES.md](ISSUES.md) (what's left to do).

## How it works

```
browser mic ──▶ EARS    speech recognition + turn-taking: answer now, stop, or say "mm-hm"?
                 │      also serves the web page, on port 7860
                 ▼
                MOUTH   runs the conversation: the LLM (Gemma 4 E4B) and the voice (VoxCPM2)
                 │
browser speaker ◀┘      the reply's audio streams back as it's made
```

- **EARS** transcribes you as you speak. Several times a second it decides whether you've finished, whether you're
  cutting in on the agent, or whether the agent should say "mm-hm". It also hears the agent's own voice, so the
  agent's echo in your mic isn't taken for you.
- **MOUTH** writes the reply with the LLM and speaks it sentence by sentence, so you hear it before it's all written.
  If it needs a tool, it says a short line first ("Checking the time in Tokyo"), then the result. If you cut in, it
  stops at once and remembers only what you heard.

Both models run inside MOUTH, on the same GPU as EARS. DEVELOPING.md has the details.

## What you need

- **An NVIDIA Blackwell GPU with 32 GB or more.** tyger is built on an RTX 5090. It also runs on an RTX PRO 5000
  (48 GB) and an RTX PRO 4500 (32 GB), with slower replies ("Known limits"). Other GPUs aren't tested.
- **A driver for CUDA 13** (590 or newer).
- **~60 GB of disk.** The models take ~25 GB.
- **Python 3.12 and [uv](https://docs.astral.sh/uv/).** Node 22 or newer builds the web page (optional).
- **A Hugging Face token** (any account's), and an **OpenAI key** for the expert (optional).

At 4 conversations the GPU holds ~28 GB: the LLM ~15 GB, the voice ~9 GB, EARS ~4 GB.

## Quick start

### 1. Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=auto
```

### 2. Keys

Copy `.env.example` to `.env` and fill it in. It's gitignored: never commit it.

| Variable | What it's for |
|---|---|
| `HF_TOKEN` | Downloads the turn-taking model, [`nineninesix/tyger`](https://huggingface.co/nineninesix/tyger). The model is public, but tyger's loader still asks for a token: any account's will do. Not needed if a token is already saved in `$HF_HOME/token`. |
| `OPENAI_API_KEY` | The expert: hard questions, live facts through web search, and background tasks. Without a key, remove `llm.smart` from the config: the agent then says it can't check live facts. |
| `ANTHROPIC_API_KEY` | Only if the expert runs on Claude instead. |
| `TYGER_API_SECRET` | Only for API mode (below). |

### 3. Voices

The repo has no voice clips: put your own `.wav` files in `voices/`. The file name is the voice's name
(`voices/mira.wav` is `mira`). For a better clone, write the clip's exact words in `voices/<name>.txt`. Without them,
only the first 10 seconds are used.

Each voice speaks as a character, written in `config/prompts.yaml`: **Mira** (the default), **Aria**, **Kai** and
**Rex**, for `mira.wav`, `aria.wav`, `kai.wav` and `rex.wav`. Any other clip speaks as Mira. Picking another voice on
the page switches the character from the next turn, and the conversation goes on.

**A real person's voice is personal data:** keep the clip on your machine, and don't commit or share it. (The four
clips in the public Docker image are synthetic.)

### 4. Run

```bash
./run.sh
```

Then open http://localhost:7860.

- The first start downloads the models (~25 GB) and compiles kernels: ~10 minutes or more. Later starts take ~2
  minutes.
- With Node installed, `run.sh` also builds the web page. Without it, you get the classic page.
- No GPU? `CONFIG=config/fake.yaml ./run.sh` runs everything on fake models, with no downloads.

**Browsers allow the mic only on HTTPS or localhost.** On a remote machine, forward the port
(`ssh -L 7860:localhost:7860 <server>`) and open http://localhost:7860. Or serve it over HTTPS: behind a reverse
proxy such as Caddy (`tyger.example.com { reverse_proxy localhost:7860 }`), or a Cloudflare Tunnel ("RunPod +
Cloudflare").

## The page

Everything is on port 7860: the page at `/`, the classic page at `/classic`, the voice list at `/voices`, and the
conversations at `/ws`.

- **The page (`/`)** has a mic button, the characters as tabs, and an orb that follows the conversation
  ([OrbKit](https://github.com/zzzzshawn/orbkit)'s Ion, MIT, via
  [Agents Kit](https://agents-ui.github.io/agents-kit/voice)). Your words show as you speak, with the agent's
  "mm-hm"s beside them. Its reply shows with the links it cited, and a history keeps the whole conversation.
  **Tuning** opens live meters and sliders for the turn-taking thresholds.
- **The classic page (`/classic`)** is the plain one: a start button, a voice picker, meters and sliders
  (`?nodebug` hides the sliders).

The page is a Next.js app in `web/`. To work on it with hot reload, next to a running stack:

```bash
cd web && npm install && npm run dev    # http://localhost:3000, talking to EARS on :7860
```

When you're done, run `npm run build` (or restart `run.sh`), so that port 7860 serves the change.

## Configure

Settings are in `config/`; keys stay in `.env`.

```
config/
  default.yaml           models, thresholds, voices, limits (what run.sh uses)
  fake.yaml              fake models: no GPU, no downloads
  prompts.yaml           every prompt the LLMs get
  engines/voxcpm2.yaml   the voice engine
```

Use another config with `CONFIG=<file> ./run.sh`. A mistake in a config stops the start and names the key.

- **Prompts** are all in `config/prompts.yaml`. Restart after editing.
- **Turn-taking thresholds** set how soon the agent answers (`ears.ep_thr`), how easily you can cut in
  (`ears.barge_thr`), and how often it says "mm-hm" (`ears.bc_thr`). They aren't fully tuned on live mics yet: tune
  them with the page's sliders while you talk, then copy the values into `config/default.yaml`.
- **"Mm-hm"s** are short reactions the LLM writes from what you've said so far ("oh nice", "wow, really?"), or a plain
  "mm-hm" when you've said too little (`mouth.bc_min_words`).
- **Speaking sooner:** `mouth.first_chunk: clause` starts the voice at the first comma instead of the first full
  stop. It's 30–95 ms sooner, but the tone can reset mid-sentence.

## API mode: tyger behind your own app

Run tyger as the voice backend of your own web app: no pages, and only your app's users get in.

| Setting | Env var | Default | What it does |
|---|---|---|---|
| `server.pages` | `TYGER_PAGES` | `true` | `false`: no pages, only `/ws` and `/voices` (and no Node needed) |
| `server.max_session_s` | `TYGER_MAX_SESSION_S` | `180` | a conversation's length in seconds; `0`: no limit |
| — | `TYGER_API_SECRET` | not set | the secret that tokens are signed with |

The env vars override `config/default.yaml`, e.g. `TYGER_PAGES=false TYGER_API_SECRET=... ./run.sh`. The secret is
only ever an env var, and is never logged.

**Set `TYGER_API_SECRET` before the server can be reached from the internet.** Tokens are the only check: tyger
accepts connections from any site, and without the secret anyone who reaches the port can talk to it.

**Tokens.** Your app's backend signs a token for each signed-in user, and the browser opens `/ws?token=…`. A token is
a JWT, signed with HS256 and the same secret, with:
- `aud: "tyger"`;
- `exp` at most 5 minutes ahead;
- a unique `jti`: one token opens one conversation;
- optionally `origin`: then the token works only from that page's origin.

A bad token gets a 401. `/voices` takes a token too, without using it up. The built-in pages can't get tokens, so run
with `TYGER_PAGES=false` when the secret is set. For development, leave the secret unset.

**Limits.** At the session limit, a reply in progress may finish; then tyger sends `session_end` and closes, and the
page says "Time's up — start again". A message over 64 KB, or mic audio more than 5 s ahead of real time, closes the
connection.

### The client

`web/lib/tyger/client.ts` is tyger's browser client: one file, no dependencies, and tyger's own page uses it. Copy
`web/lib/tyger/` into your app:

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

Its events: `open`, `ready` (the voices and the session limit), `partial`, `transcript`, `reply`, `links`,
`backchannel` and `backchannelDone` (the "mm-hm"s), `barge` (the user cut in), `turnDone`, `taskDone`, `sessionEnd`,
`playing`, `close`, `error`, and `message` (every message, as it came).

**The client must play the agent's voice itself.** It plays it on the same audio clock as the mic, and tags each mic
block with what the user was hearing: that's how tyger tells the user from the agent's echo. Playing the audio any
other way (an `<audio>` element, another AudioContext) breaks this, and the agent then cuts itself off.

**HTTPS:** the mic needs your page on HTTPS, and an HTTPS page needs `wss://`, so put tyger behind TLS too.

### The token route

A Next.js route handler that checks your app's own session, then signs a token with
[`jose`](https://github.com/panva/jose). The secret is a server-only env var, with the same value as tyger's:

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

### The protocol

For writing a client of your own (`app/ears/playback.py` has the details). Binary frames are little-endian, with
16-bit audio:
- **mic → tyger:** `[int32 reply id][int32 reply sample, or -1][int16 mic audio at 16 kHz]`, in 20 ms blocks. The tag
  is the reply sample (24 kHz) the user was hearing at the block's start, or -1 when nothing was playing.
- **tyger → speaker:** `[int32 reply id][int16 audio at 24 kHz]`. Replies are numbered from 1 on each connection.
  Id 0 is a "mm-hm": play it, but never tag it.
- **text → tyger:** `played:<id>:<samples>` once a reply has played out (this ends the agent's turn); also `reset`,
  `voice:<name>` and `set:<threshold>:<value>`.
- **JSON → page:** `ready` (`voices`, `labels`, `max_session_s`), `state`, `partial`, `endpoint` (with the user's
  `transcript`), `reply`, `links`, `backchannel`, `backchannel_done`, `barge`, `turn_done`, `task_done`, and a few
  more. After `session_end`, `busy` or `error`, the connection closes.

On `barge`, a client must stop playback at once.

## Docker

The public image comes with the four synthetic voices:

```bash
docker run -d --name tyger --restart unless-stopped --gpus all \
  --env-file .env -p 127.0.0.1:7860:7860 \
  -v tyger-hf:/hf -v tyger-cache:/root/.cache -v tyger-triton:/root/.triton \
  ylankgz/tyger-s2s-vllm:36da1bd
docker logs -f tyger      # ready at "EARS ready"
```

- **The host** needs an NVIDIA GPU ("What you need"), Docker with the NVIDIA Container Toolkit, and ~60 GB of disk.
  The image is 8.4 GB to download, ~17 GB unpacked.
- **`.env`** holds `HF_TOKEN` and `OPENAI_API_KEY` (MOUTH won't start without it on the default config), and the
  API-mode settings.
- **The port** is open on 127.0.0.1 only. Put HTTPS in front: a reverse proxy, or a Cloudflare Tunnel that the
  container starts itself ("RunPod + Cloudflare").
- **The volumes** keep the models and compiled kernels, so a new container starts in ~2 minutes, not ~11.
  `--restart unless-stopped` also covers a first download that outlasts `run.sh`'s 15-minute wait.
- **Without OpenAI,** mount a copy of `config/default.yaml` without `llm.smart` and `mouth.tasks`:
  `-v "$PWD/local.yaml":/app/config/local.yaml:ro -e CONFIG=config/local.yaml`.
- **Check it:** `docker exec tyger .venv/bin/python scripts/stack/smoke.py` runs a few turns, a cut-in, and speech
  through EARS. In API mode it signs its own token.
- **Stop it** with `docker stop -t 30 tyger`.
- The image serves the classic page: the new page needs Node to build, and the image has none.

**Your own image,** for example with your own voices (the clips in `voices/` go into it):

```bash
docker build -t tyger .                      # on an Apple Silicon Mac, add --platform linux/amd64
docker login -u <you>                        # with a Docker Hub access token
docker tag tyger <you>/tyger-s2s-vllm:<tag>
docker push <you>/tyger-s2s-vllm:<tag>
```

Docker Hub makes a new repository public unless your account says otherwise. If the image holds a real person's
voice, create the repository as private first.

## RunPod + Cloudflare

The setup we use: the image on a RunPod GPU pod, a Cloudflare Tunnel from the pod to your domain, and your app on
Cloudflare Workers. It's been tested on an RTX PRO 4500 pod.

```
browser ── https ──▶ your app (Cloudflare Workers): signs a token
   └────── wss://tyger.example.com/ws?token=… ──▶ Cloudflare ── tunnel ──▶ the pod, port 7860
```

The tunnel gives the pod a fixed HTTPS address, with no open ports. RunPod's own proxy won't do: it cuts connections
after 100 seconds, and a conversation lasts up to 3 minutes.

1. **Create the tunnel** in the Cloudflare dashboard: Zero Trust → Networks → Tunnels → Create a tunnel →
   Cloudflared. Copy its token (the string after `--token` in the install command), and add a public hostname:
   `tyger.example.com`, type `HTTP`, URL `localhost:7860`. Keep the token secret; if it leaks, refresh it on the
   tunnel's page. If Terraform manages your DNS, keep the tunnel's CNAME out of it.
2. **Deploy a pod** (Pods → Deploy):
   - an RTX 5090 (an RTX PRO 4500 works, more slowly) on a host with CUDA 13, in a region near your users;
   - the image `ylankgz/tyger-s2s-vllm:36da1bd`, or your own (a private one needs Settings → Container Registry
     Auth, with a read-only token);
   - a ~30 GB container disk, and a ~60 GB volume disk at `/workspace`;
   - **no exposed ports**;
   - these env vars, with the secrets in RunPod Secrets (used as `{{ RUNPOD_SECRET_<name> }}`):

   | Env var | Value |
   |---|---|
   | `CLOUDFLARE_TUNNEL_TOKEN` | the tunnel's token (secret): `run.sh` starts the tunnel once EARS is up |
   | `TYGER_API_SECRET` | the token secret (secret), the same as your app's |
   | `HF_TOKEN` | for the turn-taking model (secret) |
   | `OPENAI_API_KEY` | for the expert (secret). Without it: your own image, with a config that has no `llm.smart`, named in `CONFIG` |
   | `TYGER_PAGES` | `false` |
   | `HF_HOME` | `/workspace/.hf_home` |
   | `XDG_CACHE_HOME` | `/workspace/.cache` |
   | `FLASHINFER_WORKSPACE_BASE` | `/workspace` |
   | `TRITON_CACHE_DIR` | `/workspace/.cache/triton` |

   The last four keep the models and kernels on the volume, so the pod's later starts take ~2 minutes. A new pod's
   first start takes ~11 minutes.
3. **Check it.** The logs show "EARS ready", then "Registered tunnel connection". Then, from anywhere:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' https://tyger.example.com/voices
   ```

   `401` means it works (it wants a token). `530`: the tunnel isn't up yet, or its token is wrong (see the logs).
   `502`: the tunnel can't reach port 7860. Then run the smoke test in the pod's terminal:
   `cd /app && .venv/bin/python scripts/stack/smoke.py`.
4. **Connect your app:** copy the client, point it at `wss://tyger.example.com/ws`, and add the token route ("The
   token route"). On Workers, set the secret with `npx wrangler secret put TYGER_API_SECRET` (in `.dev.vars` for
   `next dev`), and read it inside the handler:

   ```ts
   import { getCloudflareContext } from "@opennextjs/cloudflare"
   const secret = new TextEncoder().encode(getCloudflareContext().env.TYGER_API_SECRET)
   ```
5. **Optional: a per-IP limit,** with a Cloudflare rate limiting rule (Security → WAF → Rate limiting rules) on the
   path `/ws`, counted by IP. For example, more than 3 requests a minute blocks the IP for 10 minutes (Pro plan; the
   Free plan counts over 10 seconds only). Each conversation opens `/ws` once, so this limits how often an IP starts
   conversations, not how many it holds. Past the limit, Cloudflare answers `429`.

Good to know:
- 64 MB of `/dev/shm` is enough: the stack doesn't need Docker's `--ipc=host`.
- Keep `cloudflared` at its default log level. At debug level it logs request URLs, and they hold the tokens.
- A stopped pod keeps its volume (and bills for it), but may not get its GPU back.

## Plugins

The LLM, the voice and the turn-taking model are plugins, picked in `config/default.yaml` (code in `app/plugins/`).
Each block names its `plugin:`, then that plugin's options.

| Kind | Plugin | Runs | What it is |
|---|---|---|---|
| LLM | `gemma-4-e4b` | locally | **the responder:** `google/gemma-4-E4B-it` in FP8, text only, no thinking |
| LLM | `openai-responses` | hosted | **the expert** (`smart`): OpenAI's Responses API, with web search |
| LLM | `anthropic` | hosted | another expert: Claude (`claude-opus-5-5` by default) |
| TTS | `voxcpm2` | locally | voices cloned from `voices/` |
| EARS | `tyger` | locally | nemotron speech recognition + a small GRU turn-taker, in one model (`nineninesix/tyger`) |
| all three | `fake` | CPU, no model | for tests and `config/fake.yaml` |

Gemma 4 E4B beat the ten other responders we measured (DEVELOPING.md → "Benchmarks").

**LLM roles:**
- `llm.reply` answers each turn.
- `llm.filler` writes the "mm-hm"s and the short lines around tool calls. By default it's the same model as `reply`.
- `llm.smart` (optional) is the expert.

Any role can use any LLM plugin; two blocks with the same plugin need a `name:` each. Keys go in `.env`, under the
name the block's `api_key_env` gives. `reply` and `filler` stay local because hosted ones were slower: first audio
came ~0.35 s later, and "mm-hm"s ~0.55 s later. If a hosted request fails, the agent says a short apology; if the
model refuses, a short neutral line.

To add a plugin or a tool, see DEVELOPING.md.

## The expert and background tasks

With an `llm.smart` block, the local model can hand work to the expert. It decides by calling a tool, so easy turns
cost nothing extra.

- **`ask_expert`** for questions that need care: maths, code, health, money or legal questions, comparisons,
  planning, long explanations. Also for live facts (weather, news, prices, scores), which the expert looks up on the
  web. The agent says a fitting line first ("let me pull up the Lisbon forecast"), and the expert's answer streams
  straight into the voice. Its sources show as links on the page, and are never read out.
- **`start_task`** for work that takes minutes: research, comparing many options, a plan, a long draft. The agent
  says it'll look into it, and the conversation goes on. When the task is done and you've been quiet for a moment,
  the agent tells you a short summary. "How's it going?" gets a status; "drop it" cancels the task.
- **If you cut in** while the agent is still looking something up, the lookup isn't lost: it carries on as a
  background task, and is announced at the next pause.

```yaml
llm:
  smart: {plugin: openai-responses, model: gpt-5.4-mini, max_tokens: 1024, web_search: true, reasoning_effort: low,
          api_key_env: OPENAI_API_KEY}
mouth:
  escalation: {context_turns: 4, max_steps: 3}
  tasks: {enabled: true, max_concurrent: 1, max_per_session: 5, max_global: 4, max_wall_s: 300, max_steps: 8,
          announce_after_silence_ms: 1200}
```

Routing is tested by `scripts/bench/escalation.py` (local model only, no API cost): about 99% of 119 synthetic cases
go the right way, and every live-fact question goes to the expert. The longer prompt adds ~30–90 ms to first audio
on ordinary turns. Web searches are billed on top of the tokens.

## Models and licences

| Part | Model | Licence |
|---|---|---|
| Speech recognition + turn-taking | [`nineninesix/tyger`](https://huggingface.co/nineninesix/tyger) (public): `nvidia/nemotron-3.5-asr-streaming-0.6b`, unchanged, + a GRU turn-taker | Apache-2.0 (the GRU), OpenMDW-1.1 (nemotron) |
| Responder | `google/gemma-4-E4B-it`, FP8 | Apache-2.0 |
| Voice | `openbmb/VoxCPM2` | Apache-2.0 |
| Expert, tasks, web search | OpenAI `gpt-5.4-mini`, with web search | commercial API (your key) |

`config/default.yaml` pins each model's revision.

## Privacy

- **Nothing you say is stored.** No audio or transcript goes to disk or into the logs, which hold timings and event
  types only. Tokens aren't logged either. The page gets your words back, to show them.
- **Voice clips stay where you put them:** in `voices/`, or in an image you build.
- **Hosted models get text only, and only if the config names one.** The expert gets the question, your words and the
  last few turns (`context_turns`), without tool results. A hosted `reply` or `filler` would get every turn. Audio and
  voice clips are never sent.
- **Request and response bodies are never logged,** and MOUTH won't start with the OpenAI or Anthropic SDK's debug
  logging on (`OPENAI_LOG` / `ANTHROPIC_LOG` = `debug`). Use hosted models only under provider terms you accept.

## Known limits

- **4 conversations per GPU.** The next visitor gets a "busy" message.
- **Conversations end after 3 minutes** by default (`TYGER_MAX_SESSION_S`; `0` for no limit).
- **No per-user limit in the server:** whoever has tokens can take all 4 places. Your app decides who gets a token,
  and a Cloudflare rule can limit how often one IP starts conversations.
- **English only.**
- **`set_reminder` and `send_message` are placeholders:** the agent says they aren't connected yet.
- **Response time,** from the end of your speech to the agent's voice: ~0.7 s on an RTX 5090 (~0.75 s at 4
  conversations), and ~0.8 s on an RTX PRO 4500 (~1.15 s at 4). About half a second of that is detecting that
  you've finished; tuning the thresholds should cut it.

## Licence

tyger's code is under the [Apache License 2.0](LICENSE). The models keep their own licences ("Models and
licences"), and the orb's files keep OrbKit's MIT licence (`web/components/voice-agents/orbkit/`).
