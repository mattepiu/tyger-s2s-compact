# Backlog — parked for a later session

These items were moved out of HANDOFF.md by the user: items 1–6 on 2026-10-10, and the pip package (item 7) on
2026-10-11.
- When a session takes one up, move it back into HANDOFF.md's work queue.
- ISSUES.md has the detail behind each ID.
- Record the work in CHANGELOG.md as usual.

1. **L-02 — 160 ms hops.**
   - GPU time is freed now: the encoder and the head are each ~22% busy at 4 conversations.
   - Halving the hop saves ~80 ms of detection on average.
   - Measure hop lag at 4 conversations, and the detection delay.
   - The 2026-10-05 box's slow detection at 1 conversation (803–883 ms p50) was that box; the 2026-10-06 box
     measured 641 ms (NOTES.md → "Box (2026-10-06)").
2. **I-15 (low, tests).** The local contract test aborts at exit after VoxCPM2 and EARS ran in one process, though
   all checks pass first. ISSUES.md has the isolation and the options.
3. **L-01 option 2.** Precompute the agent channel's features from the TTS audio while it speaks. After I-17,
   index them by the page's playback tags, not by arrival.
4. **O-02 — Docker.** It needs a host with Docker and the NVIDIA runtime.
   1. Build the image from the Dockerfile that the pip-package work (item 7) leaves.
   2. Run it with the mounts from the Dockerfile header.
   3. Run `scripts/stack/smoke.py` against it.
5. **I-09 — confirm the fix** (low).
   - It was marked fixed on 2026-10-10: R-01 (`d901503`) seems to have fixed it, and
     `test_lost_mouth_ends_the_conversation` passes on the fakes.
   - To check it on the real stack, restart MOUTH mid-conversation: the page should say the agent went away.
   - I-08 and I-10 were fixed by I-17 on 2026-10-10, with the page's playback tags and its `played` report.
6. **I-11 (auth and limits).** It is mandatory before anything is exposed beyond an SSH tunnel.
7. **P — pip package `tyger-s2s`** (user, 2026-10-07; final shape 2026-10-10; parked 2026-10-11). Do it on a new
   branch, `pip-package`, made from `plugins`. Work queued after it continues on that branch, so it doesn't
   conflict with the rename.

   **Goal.** On a fresh Linux box with an NVIDIA driver and no repo checkout, two commands bring the stack up:
   `uv pip install tyger-s2s --torch-backend=auto`, then `tyger serve`. A pip install can't start a server by
   itself, so the second command stays.

   **One package (user, 2026-10-10).** `tyger-s2s` is this repo: the core and every plugin, with vLLM and vLLM-Omni
   as ordinary dependencies. There are no extras and no engine packages: tyger targets vLLM on NVIDIA GPUs. The name
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
     `voxcpm==2.0.3`, `accelerate==1.12.0`, …).
     - Exact pins clash with other packages, so the README says to install into a fresh venv.
     - `peft` is listed, but nothing in `app/` imports it, since the heads release has the LoRA merged in. Confirm
       that and drop it.
   - **Plugins** stay in the static registry.

   **CLI (`tyger/cli.py`).**
   - **`tyger serve [--config NAME|PATH] [--voices DIR] [--env-file PATH]`** is run.sh rewritten in Python:
     1. Load `.env`.
     2. Start MOUTH (`python -m tyger.mouth`) in its own process group.
     3. Wait for its port, checking that the process is still alive.
     4. Start EARS with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
     5. On exit, SIGINT or SIGTERM, send SIGTERM to both process groups, wait 30 s, then SIGKILL whatever is left,
        as `stop.sh` does now. This fixes **I-14**.
   - **`tyger stop`** stops a running `serve`, using a pidfile in the user's state directory.
   - **`tyger init [DIR]`** copies the bundled `default.yaml` and `prompts.yaml`, plus an `.env.example`, into
     `~/.config/tyger/` (or DIR), and creates `voices/` there. It never overwrites a file.
   - **`tyger check`** is a preflight check. `serve` also runs it before loading the models. It checks:
     - Linux x86_64;
     - `nvidia-smi` runs and the driver is ≥ 590 (the cu130 wheels need it);
     - free GPU memory against the budget in HANDOFF's Rules;
     - a C compiler is present, for Triton and FlashInfer's kernel builds (the Dockerfile installs
       `build-essential` for this reason);
     - Hugging Face access to the heads release. On a 401 or 404, say the release is private and `HF_TOKEN` needs
       access to it;
     - every key the config needs, e.g. `OPENAI_API_KEY` for `smart`.

     Each failure prints one line that names the fix. The fake config skips the GPU checks.

   **Files and paths.**
   - **Package data.** `config/` and `static/` move into the package (`tyger/config/`, `tyger/static/`) and are
     read with `importlib.resources`. A bundled config can be picked by name: `--config default` or
     `--config fake`.
   - **Config lookup order:** `--config`, then `$TYGER_CONFIG`, then `~/.config/tyger/config.yaml` (written by
     `init`), then the bundled default.
     - Relative paths inside a user's config resolve against that config's own directory.
     - `ROOT` and `repo_path` go away.
   - **`.env` lookup order:** `--env-file`, then `./.env`, then `~/.config/tyger/.env`. It is never shipped and
     never printed.
   - **Voices.** `voices.dir` defaults to `~/.config/tyger/voices`, and `--voices` overrides it. Voice clips never go
     into the wheel or the sdist.
   - **Hugging Face cache.** Remove the `/workspace/.hf_home` default and use the standard HF cache. The box keeps
     its own `HF_HOME` through `.env` or the shell.
   - **`run.sh` and `stop.sh`** stay as thin wrappers around `tyger serve` and `tyger stop`, so the commands in
     HANDOFF keep working.
   - **`scripts/` and `tests/`** stay out of the wheel; the sdist may include them. They run from an editable
     install, `uv pip install -e .`. `tests/page/page_check.mjs` reads `static/index.html` by default; point it at
     `tyger/static/`.
   - **Dockerfile.** Install the wheel and set `ENTRYPOINT ["tyger", "serve"]`. A box that is itself a Docker
     container can't build it (item 4).

   **Not part of this task:**
   - **Publishing.** Don't upload anywhere, not even TestPyPI: the user publishes.
   - **Making the heads public.** That is the user's decision. While the release stays private, the package works
     only for HF accounts that have access, and `tyger check` must say so plainly.
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
      3. On a no-`smart` config, `load.py --sessions 4 --turns 6` ×3 and `load_ears.py --sessions 1` stay within
         noise of the baseline.
      4. The GPU peak is unchanged.
   3. **Find out whether plain pip works.** In one more fresh venv, install with plain `pip install` (no uv).
      Record which torch build pip picks for vllm 0.30.0 and whether it runs on the box's GPU. This decides the
      README's install line.
   4. **Nothing is left on the GPU.**
      - After `tyger stop`, and after Ctrl-C, `nvidia-smi` shows no GPU process.
      - Kill `serve` while MOUTH is still starting, as a slow first download would; no `VLLM::EngineCore` remains.
      - Mark I-14 fixed in ISSUES.md.
   5. **The usual checks pass** in an editable install: the GPU-free tests, the page check, the local contract test
      and `uvx pyflakes tyger/ scripts/ tests/`.
   6. **The docs are updated.**
      - **README:**
        - the install line and the driver requirement;
        - the first start downloads ~25 GB;
        - the private-heads note;
        - `tyger init` / `serve` / `stop` / `check`;
        - the config lookup order.
      - **The commands in HANDOFF and the other docs:** `app/` → `tyger/`, plus the new config paths.
      - **CHANGELOG.md:** a dated entry under Unreleased.

   **Disk space:** each fresh venv with vllm is ~10 GB or more. Share the uv cache between them, and delete the
   test venvs when you're done.
