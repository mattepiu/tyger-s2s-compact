# tyger — open issues and planned work

What is left to do, by ID. IDs are stable: put them in commit subjects ("Fix I-19").
- When an issue is fixed, it leaves this file. Its fix gets a short line in `CHANGELOG.md`, and its commit message
  has the detail and the numbers.
- The write-ups of fixed issues (I-01 to I-17, L-03, L-04, O-04) are in git history: this file at `4d4c2e3`.
- Each issue says how sure it is: **confirmed** (in code or on the box), **likely**, or **hypothesis** (measure
  before fixing).

| ID | Area | Severity | Status | One line |
|---|---|---|---|---|
| I-19 | smart | medium | open | After a language switch, the expert answers in the previous turn's language |
| I-20 | smart | medium | open | The expert calls `get_current_time` before most answers; once an empty answer, once a "Dummy" reminder |
| I-21 | smart | low | open | The expert writes unit symbols ("19°C") into speech |
| I-22 | prompts | medium | open | Without an expert, E4B answers Russian and Spanish live-fact questions in English |
| I-23 | routing set | low | needs the user | b10's label; b15 drifts to the expert |
| I-18 | EARS | medium | accepted for now | With a clean mic, an "mm-hm" in the agent's pause barges |
| I-24 | tests | low | open | `test_barged_round.py` is timing-based and failed once under CPU load |
| I-09 | EARS | low | check on the stack | A lost MOUTH ends the conversation on the fakes; not checked on the real stack |
| I-14 | ops | low | open (P fixes it) | `run.sh` giving up on a slow first start orphans the LLM engine on the GPU |
| I-15 | tests | low | open | The local contract test aborts at exit after the EARS engine ran in the process |
| I-11 | security | low (demo) / high (public) | open | No auth and no limits on :7860 |
| L-01 | perf | lever | partly done | The encoder re-encodes the full 8 s window, twice, every hop |
| L-02 | latency | lever | open | Decisions land on 320 ms hops |
| O-01 | ops | — | needs the user | The endpoint thresholds were never tuned live |
| O-02 | ops | — | blocked | The Dockerfile was never built |
| O-03 | ops | — | needs the user | `main` is 128 commits behind `plugins` |
| O-05 | ops | — | partly done | No CI |
| O-06 | ops | — | needs the user | A first model load can run the box out of RAM |
| P | packaging | — | planned (parked) | The pip package `tyger-s2s` |

---

## The expert (`smart`)

All three were found by the live check of `openai-responses` on 2026-10-11 (`gpt-5.4-mini`, web search on,
synthetic questions). A live run against OpenAI needs the user's OK each time.

### I-19 — After a language switch, the expert answers in the previous turn's language (medium, confirmed)
**What happens:** on the stack (the default config, 3 runs of 6 turns), an English question right after a Russian
turn ("Who won the last Formula One race?") was answered in Russian in 2 of 3 runs. A plugin-level replay with a
shorter context answered in English 4 / 4. The real request isn't logged (privacy).

**Where:** the smart request (`app/mouth/messages.py`) and the `smart` prompts in `config/prompts.yaml`.

**Fix:** name the reply language in the smart request, as MOUTH already does after tool results (`result_note`,
with `app/prompts/lang.py`).

**Verify:** a GPU-free test that the smart request names the user's language; then the stack run that found it,
with the user's OK.

### I-20 — The expert calls `get_current_time` before most answers (medium, confirmed)
**What happens** (`gpt-5.4-mini`, `reasoning_effort: low`):
- 7 of 12 escalated turns on the stack had `get_current_time` rounds first, one of them 3. Each is an extra round
  trip before the answer.
- Once it looped through four `get_current_time` rounds and then gave an empty answer.
- Once it called `set_reminder` with the text "Dummy" for a savings question.

**Where:** `app/plugins/llm/openai_responses.py`, and the smart model's tools: `Mouth.smart_tools` in
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

### I-21 — The expert writes unit symbols into speech (low, confirmed)
**What happens:** weather answers said "19°C, or 66°F" (4 of 6), though the smart prompt asks for spoken words only.

**Fix:** spell units out in the smart prompt, with an example ("nineteen degrees Celsius").

**Verify:** the weather questions of the live check, with the user's OK.

---

## Routing and prompts

### I-22 — Without an expert, E4B answers Russian and Spanish live-fact questions in English (medium, confirmed)
**What happens:** with no `llm.smart`, E4B says plainly it can't check live facts (66 / 66, no guessed values). But
all 24 Russian and Spanish samples were in English ("I'm sorry, I can't check the weather right now."). It copies
the English wording of `system.voice_tools`. The routing eval doesn't see this: it skips expert cases when there is
no expert.

**Fix:** reword the sentence. Two drafts, untested:
- "say plainly, in the language of the user's latest message, that you can't check that right now";
- "You have no way to look up anything else that is live, such as weather, prices, news or sports scores: when the
  user asks for one, tell them so honestly, and never guess a current value or tell the user to check a website."

**Verify:** the probe of 2026-10-11 (the 22 live-fact questions without history, 3 samples each, default prompts, no
`smart`), and `scripts/bench/escalation.py --no-expert` unchanged.

### I-23 — The routing set: b10's label, and b15's drift (low)
- **b10** ("Сравни тарифы… и скажи, какой выгоднее": compare… and tell me which is cheaper) has no "later", and
  goes to the expert 8 / 8 on every prompt tried. An expert answer is a fair reading of it. **Needs the user:**
  label it `also: [expert]`? (Recommended: yes.)
- **b15** (Spanish "compare the online banks… and let me know what you find") goes to the expert 3–4 of 8 since
  `b28f889`, against 0 / 8 after `79fbfa5`. That's a fair answer now that the expert can search, but not the
  background task the case asks for. Watch it on the next prompt change.

---

## EARS

### I-18 — With a clean mic, an "mm-hm" in the agent's pause barges (medium, confirmed; accepted for now)
**Where:** the yield head and the barge rule (`app/plugins/ears/nemotron_tyger/rules.py`, `barge_thr`).

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
- GPU time is there now: the encoder and the heads are each ~22% busy at 4 conversations.
- Measure the hop lag at 4 conversations, and the detection delay, before and after.

---

## Tests and ops

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

### I-11 — No auth and no limits on :7860 (low for the SSH-tunnel demo, high if public)
Anyone who reaches the port can hold all 4 slots and spend the OpenAI key, grow memory without bound by sending audio
faster than real time or in huge frames, and list voice names at `/voices`.

**Fix, before any exposure beyond an SSH tunnel:**
- token or basic auth, at the proxy or in the app;
- a WebSocket message-size limit around 64 KB;
- drop or close a connection whose buffer exceeds a few seconds;
- a per-IP session cap.

### O-01 — The endpoint thresholds were never tuned live (needs the user)
`ep_thr` 0.9146 and `ep_deb` 2 ship at the training defaults; offline thresholds ran ~2× high on a live mic.
`barge_thr` was set to 0.4 live on 2026-10-10. It's the biggest risk to how the product feels.
- The user drives the mic with `?debug=1`; a slider change is logged (`threshold <name> = <value>`).
- Target: end of speech → endpoint from ~600–775 ms p50 to ≤ 450 ms (p90 ≤ 550), with no more endpoints inside
  utterances.
- Afterwards, freeze the values into `config/default.yaml` and re-run `scripts/stack/load_ears.py`.

### O-02 — The Dockerfile was never built (blocked)
The GPU boxes so far are themselves Docker containers with no daemon, so nothing can be built there. The
non-Docker fresh-box install has been run end to end. It needs a host with Docker and the NVIDIA runtime, and is
best done after P, from the Dockerfile P leaves:
1. Build the image.
2. Run it with the mounts from the Dockerfile header.
3. Run `scripts/stack/smoke.py` against it.

### O-03 — `main` is 128 commits behind `plugins` (needs the user)
All work since 2026-10-02 is on `plugins`. `main` is an ancestor of it (a fast-forward), and `voxcpm2` and
`origin/setup/new-box` are already merged into it. Fast-forward `main` and delete the two old branches when the
user says so. The user pushes.

### O-05 — No CI (partly done)
The GPU-free tests and the page check cover MOUTH's session, tasks, the EARS state machine, the 2nd ear and the
plugins, and run in about a minute. Nothing runs them automatically. A CI workflow (no GPU) could run both on every
push.

### O-06 — A first model load can run the box out of RAM (needs the user)
The first load of a model or quantization JIT-compiles FlashInfer kernels, one compiler per CPU thread at 3–6 GB
each. On 2026-10-11 that twice ran the 64.6 GB container out of memory and restarted it. With `MAX_JOBS=4
FLASHINFER_NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4` the peak was 32 GiB.

**Fix:** `run.sh` exports those three unless they're already set. **Needs the user:** do it? (Recommended: yes.)

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
  - Hugging Face access to the heads release. On a 401 or 404, say the release is private and `HF_TOKEN` needs
    access to it;
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
- **Dockerfile.** Install the wheel and set `ENTRYPOINT ["tyger", "serve"]`. A box that is itself a Docker container
  can't build it (O-02).

**Not part of this task:**
- **Publishing.** Don't upload anywhere, not even TestPyPI: the user publishes.
- **Making the heads public.** That is the user's decision. While the release stays private, the package works only
  for HF accounts that have access, and `tyger check` must say so plainly.
- **I-11 (auth).** `serve` keeps binding to 127.0.0.1 by default, as the config does now.

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
