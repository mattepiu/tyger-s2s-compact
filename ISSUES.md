# tyger — open issues and planned work

What is left to do, by ID. IDs are stable: put them in commit subjects ("Fix I-20").
- When an issue is fixed, it leaves this file. Its fix gets a short line in `CHANGELOG.md`, and its commit message
  has the detail and the numbers.
- The write-ups of fixed issues (I-01 to I-17, L-03, L-04, O-04) are in git history: this file at `4d4c2e3`. I-19,
  I-22 and I-23 left when the agent became English only: this file at `1b8e6ff`. O-06 (run.sh now limits the
  kernel compilers): this file at `9caeca9`. I-21 was dropped by the user on 2026-10-13: the voice reads digits and
  "°C" well. Its write-up: this file at `ce52f66`. I-11 (no auth, no limits on :7860) left on 2026-10-13 with API
  mode: a token, the session limit and the size caps. Its write-up: this file at `17e88cd`.
- Each issue says how sure it is: **confirmed** (in code or on the box), **likely**, or **hypothesis** (measure
  before fixing).

| ID | Area | Severity | Status | One line |
|---|---|---|---|---|
| I-20 | smart | medium | open | The expert calls `get_current_time` before most answers; once an empty answer, once a "Dummy" reminder |
| I-18 | EARS | medium | accepted for now | With a clean mic, an "mm-hm" in the agent's pause barges |
| I-26 | EARS | medium | open | With echo in the mic (−30 dB), tyger barges on the agent's own voice |
| I-27 | EARS | low | open | One slow tick (0.27–0.39 s) soon after start |
| I-24 | tests | low | open | `test_barged_round.py` is timing-based and failed once under CPU load |
| I-09 | EARS | low | check on the stack | A lost MOUTH ends the conversation on the fakes; not checked on the real stack |
| I-14 | ops | low | open (P fixes it) | `run.sh` giving up on a slow first start orphans the LLM engine on the GPU |
| I-15 | tests | low | open | The local contract test aborts at exit after the EARS engine ran in the process |
| L-01 | perf | lever | partly done | The encoder re-encodes the full 8 s window, twice, every hop |
| L-02 | latency | lever | open | Decisions land on 320 ms hops |
| O-01 | ops | — | needs the user | The endpoint thresholds were never tuned live |
| O-02 | ops | — | blocked | The Dockerfile was never built |
| O-03 | ops | — | needs the user | `main` is over 130 commits behind `plugins` |
| O-05 | ops | — | partly done | No CI |
| P | packaging | — | planned (parked) | The pip package `tyger-s2s` |

---

## The expert (`smart`)

Found by the live check of `openai-responses` on 2026-10-11 and the plugin-level probes of 2026-10-12, with the
characters (`gpt-5.4-mini`, web search on, synthetic questions, 10 requests). A live run against OpenAI needs the
user's OK each time.

### I-20 — The expert calls `get_current_time` before most answers (medium, confirmed)
**What happens** (`gpt-5.4-mini`, `reasoning_effort: low`):
- 7 of 12 escalated turns on the stack had `get_current_time` rounds first, one of them 3. Each is an extra round
  trip before the answer.
- Once it looped through four `get_current_time` rounds and then gave an empty answer.
- Once it called `set_reminder` with the text "Dummy" for a savings question.
- The probes of 2026-10-12: one `get_current_time` round before each of the 5 weather answers (they name the date),
  none before the 5 others. No empty answers, no action calls.

**Where:** `app/plugins/llm/hosted/openai_responses.py`, and the smart model's tools: `Mouth.smart_tools` in
`app/mouth/service.py` gives it every tool except the ones MOUTH answers itself.

**Fix options:**
- **Send the reasoning items back** with the tool results. With `store: false`, OpenAI's docs ask for
  `include: ["reasoning.encrypted_content"]` in stateless function calling, or the model loses why it made the call.
  A likely cause, untested.
- **Offer the expert only what it needs:** `get_current_time` and its web search. `set_reminder` and
  `send_message` belong to the reply model (and are placeholders anyway).
- **Then decide `reasoning_effort`.** `none` works with web search and is ~0.6 s faster on live facts, mostly
  because it doesn't check the time first. But it got a savings sum wrong once in 3. `low` is set.

**Verify:** the live check's questions as plugin-level probes, with the user's OK: time rounds per answer, no empty
answers, no action calls.

---

## EARS

### I-18 — With a clean mic, an "mm-hm" in the agent's pause barges (medium, confirmed; accepted for now)
**Where:** the yield head and the barge rule (`app/plugins/ears/tyger/rules.py`, `barge_thr`).
tyger (2026-10-13, I-26's run): with no echo every ack barges (48 / 48 at `barge_thr` 0.4, 0.5 and 0.8); at −40 dB
29 / 48 (0.4), at −30 dB 8 / 48. The numbers below are the tyger 1.0 heads'.

**What happens** (`scripts/validate/second_ear.py --pairing tags`, 24 trials):
- With the 2nd ear aligned and no echo in the mic (headphones, or echo cancellation that leaves nothing), a short
  "Mm-hm.", "Yeah." or "Right." 0.1 s into a 0.5 s pause between the agent's sentences barges 46 of 48 times.
- The heads read user speech in the agent's silence as taking the turn. The silent-wait study (I-16) found the same.
- Residual echo makes the heads cautious: 13 / 48 at −40 dB, 2 / 48 at −30 dB.
- The threshold can't separate an ack from a cut-in: at `barge_thr` 0.8, 38 / 48 acks still barge, while cut-ins
  over the agent's speech fall to 3 / 24.
- Live (the user, 2026-10-10): at 0.4, the default, an "mhm" stops the agent, which the user accepts for now. It also
  reaches I-16's carry-over: with headphones, a short "okay" during an expert's lead-in barges and sends the answer
  to a later announcement.

**Fix options (not decided):**
- Measure acks live, with headphones and with speakers. The stack's pauses are the TTS's own silence at sentence
  edges, which hasn't been measured.
- Hold a barge until the user has spoken longer than an acknowledgement, ducking the agent meanwhile, and resume if
  the user stops.
- In the head repo: train the yield head on acknowledgements in pauses. That fixes it at the source.

### I-09 — A lost MOUTH: check it on the real stack (low)
On a lost MOUTH, EARS sends the page `{"type": "error", "error": "mouth_lost"}` and closes the WebSocket; the page
says "the agent went away — press Start to reconnect". It seems to have been fixed by R-01 (`d901503`), and
`tests/ears/test_floor.py::test_lost_mouth_ends_the_conversation` passes on the fakes.

**Verify:** restart MOUTH mid-conversation on the real stack. The page should say the agent went away.

### L-01 — The encoder re-encodes the full 8 s window, twice, every hop (lever, partly done)
Each hop re-encodes ~25× the new audio, for both channels, in fp32.
- **Done (2026-10-03):** features on the GPU; the encoder went from ~33% to ~22% busy at 4 conversations. bf16 was
  rejected: no speedup (the encoder is launch-bound), head differences up to 0.14, and an event changed.
- **Option 2, not tried:** compute the agent channel's features from the TTS audio while it speaks, indexed by the
  page's playback tags (I-17), not by arrival.
- **Option 3, for the head repo:** a cache-aware streaming encode would cut the encoder to the new 320 ms per hop,
  but the heads would need validating or retraining on streaming layer-15 features.

### L-02 — Decisions land on 320 ms hops (lever)
Endpoint and barge decisions land on hop boundaries, which adds ~160 ms on average to both. A 160 ms hop (2 frames)
would halve that.
- GPU time is there now: the encoder was ~22% busy at 4 conversations, and tyger's GRU steps all 4 in ~0.5 ms a
  frame (the tyger 1.0 heads were another ~22%).
- Measure the hop lag at 4 conversations, and the detection delay, before and after.

---

## Tests and ops

### I-26 — With echo in the mic, tyger barges on the agent's own voice (medium, confirmed offline)
**What happens** (`scripts/validate/second_ear.py --pairing tags`, 24 trials, 2026-10-13): the agent talks alone, its
echo in the mic at a fixed level, the user silent. A trial with a BARGE is a reply cut off by the agent's own voice.

| echo in the mic | `barge_thr` | replies cut off (agent alone) | cut-ins that barge, median | acks in pauses that barge |
|---|---|---|---|---|
| −30 dB | 0.4 | 16 / 24 | 22 / 24, 1.38 s | 8 / 48 |
| −30 dB | 0.5 | 5 / 24 | 8 / 24, 3.62 s | 6 / 48 |
| −30 dB | 0.8 | 0 / 24 | 1 / 24 | 2 / 48 |
| −40 dB | 0.4 | 1 / 24 | 21 / 24, 1.38 s | 29 / 48 |
| −40 dB | 0.5 | 0 / 24 | 10 / 24, 2.34 s | 27 / 48 |
| none | 0.4 | 0 / 24 | 23 / 24, 1.06 s | 48 / 48 |
| none | 0.5 | 0 / 24 | 17 / 24, 1.06 s | 48 / 48 |

- The tyger 1.0 heads, with the same tags (2026-10-10): 0 / 24 replies cut off at every echo level.
- At −30 dB no threshold works: 0.4 cuts off 2 replies in 3, 0.5 misses 2 cut-ins in 3. At −40 dB and below, 0.4
  (the user's live setting) is fine.
- The page asks the browser for echo cancellation. How much echo it leaves live hasn't been measured; the user
  hasn't reported a reply cut off at 0.4 (2026-10-13).
- I-16's silent wait is fine on tyger (`silent_wait.py`, 24 trials): every case barges 24 / 24 with no false barges,
  0.66–0.70 s (median) after the user starts in the agent's silence, 1.54 s over its speech.

**Fix options:**
- Measure the echo the browser leaves live (the mic's level while the agent plays, through speakers), then decide.
- In the model repo: train the yield head with the agent's echo in the user channel.

### I-27 — One slow tick soon after start (low, confirmed)
One tick takes 0.27–0.39 s (the hop is 0.32 s), once: the first time the encoder meets a new shape at 4
conversations (`ears_batched.py`: 267 ms at the first 4-session tick of that size; the stack's first-minute tick
log: 394 ms; `load_ears.py --sessions 4` right after a restart: a 268 ms hop lag). It is the encoder, not the GRU,
and not TF32 (the same with TF32 on). Every tick after it stays under ~100 ms. The tyger 1.0 heads didn't show it,
likely because their CUDA graphs left the allocator spare memory.

**Fix option:** warm the encoder at start on the window lengths and batch sizes it will see.

### I-24 — `test_barged_round.py` is timing-based (low, confirmed)
It failed once while vLLM was compiling on the same CPUs; it passes 3 / 3 on an idle box. Run the GPU-free tests on
an idle box, or make its timing robust. Separately, one vLLM start-up hung with nothing on the GPU (a probe, killed
after 11 min); not seen again.

### I-14 — `run.sh` giving up on a slow first start orphans the LLM engine (low, confirmed)
**What happened** (box of 2026-10-05): the first start downloaded the models slowly, and VoxCPM2's first compile was
still running when `run.sh` stopped waiting ("MOUTH did not come up"). `cleanup` killed MOUTH's PID, but Gemma's
vLLM EngineCore, a child of MOUTH, survived with 14.7 GB of GPU memory. `stop.sh` finds stacks by their `run.sh`
process, which had exited, so it said "not running".

**Fix options:** signal MOUTH's whole process group in `cleanup`, as `stop.sh` does; keep waiting while MOUTH is
alive, since `wait_tcp` already fails fast when MOUTH exits. P's `tyger serve` does both.

### I-15 — The local contract test aborts at exit (low, confirmed)
`TYGER_CONTRACT_LOCAL=1 tests/plugins/test_contract.py` passes every check, then prints `terminate called without
an active exception` and exits 134. It is the EARS engine alone (4 / 4 runs): likely its two daemon threads, with
CUDA streams and graphs, still alive at interpreter teardown. The served stack runs EARS in its own process and
isn't affected.

**Fix options:** find the thread (`threading.enumerate()` and `faulthandler` at exit; this container has no ptrace).
As a stopgap, end the test's `__main__` with `os._exit(code)` after flushing.

### O-01 — The endpoint thresholds were never tuned live (needs the user)
tyger ships `ep_thr` 0.915 (the user's pick, by the bundle's 0.91), `ep_deb` 2 and `bc_thr` 0.4, untuned
live; `barge_thr` 0.4 is the user's live call (2026-10-13; also 0.4 for the tyger 1.0 heads, 2026-10-10). It's the
biggest risk to how the product feels.
- The user drives the mic with the page's sliders (shown unless `?nodebug`); a slider change is logged
  (`threshold <name> = <value>`).
- Target: end of speech → endpoint from ~530 ms p50 (tyger, `load_ears.py`, 1 and 4 conversations) to
  ≤ 450 ms (p90 ≤ 550), with no more endpoints inside utterances.
- Afterwards, freeze the values into `config/default.yaml` and re-run `scripts/stack/load_ears.py`.

### O-02 — The Dockerfile was never built (blocked)
The GPU boxes so far are themselves Docker containers with no daemon, so nothing can be built there. The
non-Docker fresh-box install has been run end to end. It needs a host with Docker and the NVIDIA runtime, and is
best done after P, from the Dockerfile P leaves.

The user plans to run the server in Docker, called from another Next.js app, and will build the image on their own
side (2026-10-13). The API-mode settings (pages, session limit) and the token's secret are env vars
for that reason (README → "API mode"); the Dockerfile's header shows them.

The steps (README → "Docker"):
1. Build the image, with the clips in `voices/` (they go into it).
2. Run it with the GPU, `--ipc=host`, `.env` and the three volumes.
3. Run the smoke test inside it: `docker exec tyger .venv/bin/python scripts/stack/smoke.py`.

The target is a RunPod GPU Pod behind a Cloudflare Tunnel (README → "RunPod + Cloudflare", 2026-10-13), not run
there yet either. Open on RunPod: whether the pod's `/dev/shm` is big enough for vLLM without `--ipc=host`.

### O-03 — `main` is over 130 commits behind `plugins` (needs the user)
All work since 2026-10-02 is on `plugins`. `main` is an ancestor of it (a fast-forward), and `voxcpm2` and
`origin/setup/new-box` are already merged into it. Fast-forward `main` and delete the two old branches when the
user says so. The user pushes.

### O-05 — No CI (partly done)
The GPU-free tests and the page check cover MOUTH's session, tasks, the EARS state machine, the 2nd ear and the
plugins, and run in about a minute. Nothing runs them automatically. A CI workflow (no GPU) could run both on every
push.

---

## Planned

### P — the pip package `tyger-s2s` (user, 2026-10-07; final shape 2026-10-10; parked 2026-10-11)
Do it on a new branch, `pip-package`, made from `plugins`. Work queued after it continues on that branch, so it
doesn't conflict with the rename.

**Goal.** On a fresh Linux box with an NVIDIA driver and no repo checkout, two commands bring the stack up:
`uv pip install tyger-s2s --torch-backend=auto`, then `tyger serve`. A pip install can't start a server by itself,
so the second command stays.

**One package (user, 2026-10-10).** `tyger-s2s` is this repo: the core and every plugin, with vLLM and vLLM-Omni as
ordinary dependencies. There are no extras and no engine packages: tyger targets vLLM on NVIDIA GPUs. The name
`tyger` is taken on PyPI; `tyger-s2s` was free on 2026-10-10.
- **Rename.** The import package is `tyger`.
  - Rename `app/` to `tyger/` with `git mv`, so history follows.
  - Make it its own commit, holding only the rename and the import fixes, so the user can review it alone.
  - Update the imports in the code, `scripts/` and `tests/`, and the paths in the docs.
- **`pyproject.toml`** replaces `requirements.txt`; delete the latter.
  - Build backend: hatchling.
  - Console script: `tyger = tyger.cli:main`.
  - Version: `1.0.0.dev0`, until the user picks the release number.
- **Dependencies.** Keep what requirements.txt has, with its exact pins (`vllm==0.30.0`, `vllm-omni==0.30.0`,
  `voxcpm==2.0.3`, `accelerate==1.12.0`, …). Exact pins clash with other packages, so the README says to install
  into a fresh venv.
- **Plugins** stay in the static registry.

**CLI (`tyger/cli.py`).**
- **`tyger serve [--config NAME|PATH] [--voices DIR] [--env-file PATH]`** is run.sh rewritten in Python:
  1. Load `.env`.
  2. Start MOUTH (`python -m tyger.mouth`) in its own process group.
  3. Wait for its port, checking that the process is still alive.
  4. Start EARS with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  5. On exit, SIGINT or SIGTERM, send SIGTERM to both process groups, wait 30 s, then SIGKILL whatever is left, as
     `stop.sh` does now. This fixes **I-14**.
- **`tyger stop`** stops a running `serve`, using a pidfile in the user's state directory.
- **`tyger init [DIR]`** copies the bundled `default.yaml` and `prompts.yaml`, plus an `.env.example`, into
  `~/.config/tyger/` (or DIR), and creates `voices/` there. It never overwrites a file.
- **`tyger check`** is a preflight check. `serve` also runs it before loading the models. It checks:
  - Linux x86_64;
  - `nvidia-smi` runs and the driver is ≥ 590 (the cu130 wheels need it);
  - free GPU memory against the budget in DEVELOPING.md → "GPU memory";
  - a C compiler is present, for Triton and FlashInfer's kernel builds (the Dockerfile installs `build-essential`
    for this reason);
  - Hugging Face access to the turn-taking model (the tyger bundle, `nineninesix/tyger`). On a 401 or 404, say
    it is private and `HF_TOKEN` needs access to it;
  - every key the config needs, e.g. `OPENAI_API_KEY` for `smart`.

  Each failure prints one line that names the fix. The fake config skips the GPU checks.

**Files and paths.**
- **Package data.** `config/` and `static/` move into the package (`tyger/config/`, `tyger/static/`) and are read
  with `importlib.resources`. A bundled config can be picked by name: `--config default` or `--config fake`.
- **Config lookup order:** `--config`, then `$TYGER_CONFIG`, then `~/.config/tyger/config.yaml` (written by `init`),
  then the bundled default.
  - Relative paths inside a user's config resolve against that config's own directory.
  - `ROOT` and `repo_path` go away.
- **`.env` lookup order:** `--env-file`, then `./.env`, then `~/.config/tyger/.env`. It is never shipped and never
  printed.
- **Voices.** `voices.dir` defaults to `~/.config/tyger/voices`, and `--voices` overrides it. Voice clips never go
  into the wheel or the sdist.
- **Hugging Face cache.** Remove the `/workspace/.hf_home` default and use the standard HF cache. The box keeps its
  own `HF_HOME` through `.env` or the shell.
- **`run.sh` and `stop.sh`** stay as thin wrappers around `tyger serve` and `tyger stop`, so the commands in the docs
  keep working.
- **`scripts/` and `tests/`** stay out of the wheel; the sdist may include them. They run from an editable install,
  `uv pip install -e .`. `tests/page/page_check.mjs` reads `static/index.html` by default; point it at
  `tyger/static/`.
- **The new page** (`web/`, served by EARS from its static build): the wheel ships the build, so building it needs
  Node at release time only, never at install.
- **Dockerfile.** Install the wheel and set `ENTRYPOINT ["tyger", "serve"]`. A box that is itself a Docker container
  can't build it (O-02).

**Not part of this task:**
- **Publishing.** Don't upload anywhere, not even TestPyPI: the user publishes.
- **Making the heads public.** That is the user's decision. While the release stays private, the package works only
  for HF accounts that have access, and `tyger check` must say so plainly.
- **API mode** (the token, the limits: README → "API mode"). `serve` keeps them as they are, env
  vars included.

**Done when:**
1. **The build is clean.** `uv build` makes a wheel and an sdist.
   - Neither contains `.env`, `voices/`, audio files, `agent-memory/` or `tests/page/node_modules/`.
   - The wheel has `tyger/config/*.yaml` and `tyger/static/index.html`.
2. **The stack works from the wheel.** Take a baseline on the box first. Then install the wheel into a fresh venv
   with `--torch-backend=auto`, and work from a directory outside the repo, so the checkout isn't on `sys.path`.
   1. `tyger serve --config fake` comes up, and `scripts/stack/load_ears.py` passes against it.
   2. Run `tyger init`, put the voice clip in `~/.config/tyger/voices/`, and run `tyger serve`. The smoke test
      passes.
   3. On a no-`smart` config, `load.py --sessions 4 --turns 6` ×3 and `load_ears.py --sessions 1` stay within noise
      of the baseline.
   4. The GPU peak is unchanged.
3. **Find out whether plain pip works.** In one more fresh venv, install with plain `pip install` (no uv). Record
   which torch build pip picks for vllm 0.30.0 and whether it runs on the box's GPU. This decides the README's
   install line.
4. **Nothing is left on the GPU.**
   - After `tyger stop`, and after Ctrl-C, `nvidia-smi` shows no GPU process.
   - Kill `serve` while MOUTH is still starting, as a slow first download would; no `VLLM::EngineCore` remains.
   - I-14 leaves this file.
5. **The usual checks pass** in an editable install: the GPU-free tests, the page check, the local contract test and
   `uvx pyflakes tyger/ scripts/ tests/`.
6. **The docs are updated.**
   - **README:** the install line and the driver requirement; the first start downloads ~25 GB; the private-heads
     note; `tyger init` / `serve` / `stop` / `check`; the config lookup order.
   - **DEVELOPING.md and the other docs:** `app/` → `tyger/`, plus the new config paths.
   - **CHANGELOG.md:** a short line under Unreleased → Added.

**Disk space:** each fresh venv with vllm is ~10 GB or more. Share the uv cache between them, and delete the test
venvs when you're done.
