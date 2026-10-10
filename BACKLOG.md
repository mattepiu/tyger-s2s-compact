# Backlog — parked for a later session

These items were moved out of HANDOFF.md on 2026-10-10 (user). That round does I-16, I-17 and the pip package
first.
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
   1. Build the image from the Dockerfile that the pip-package work (HANDOFF item P) leaves.
   2. Run it with the mounts from the Dockerfile header.
   3. Run `scripts/stack/smoke.py` against it.
5. **I-08, I-09, I-10** (low).
6. **I-11 (auth and limits).** It is mandatory before anything is exposed beyond an SSH tunnel.
