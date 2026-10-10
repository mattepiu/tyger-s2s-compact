# The 2nd ear and audio latency: a memo

## What the 2nd ear is

The turn-taking heads listen to two channels at once:

- **the user's mic;**
- **the 2nd ear:** what the agent was saying at that same moment.

From the pair, the yield head decides whether the user is cutting in (a barge).

## The old problem (I-17)

EARS used to pair the two channels **by arrival order**: it fed the agent's audio to the heads as soon as MOUTH
sent it. But the user hears that audio later (the network, the page's buffer, the device's output), and the mic's
recording of that moment reaches EARS later still. So the 2nd ear ran **ahead** of what the user heard:

| setup | how far ahead |
|---|---|
| localhost, scripts | ~290 ms (111–419) |
| your mic, over the SSH tunnel | ~570 ms (316–783) |
| a 500 ms slow network (simulated) | ~780 ms (632–918) |

With any echo of the agent in the mic, this lead made the agent's own voice look like the user cutting in.

## How it works now: tags, not a lag number

The page plays the reply and records the mic **on one audio clock**. Every 20 ms mic block it sends carries a tag:

> "While this was being recorded, the user was hearing sample N of reply #id."

EARS builds the 2nd ear directly from these tags. So:

- the alignment is **per block and per client, and automatic**: the browser never measures or sends a lag
  number, and nothing has to be configured;
- the network delay, the buffering and the jitter all drop out, because the tag says what was really playing.

The EARS log line `2nd ear by arrival order +X ms ahead` is **only a diagnostic**: how far the old method would
have been off. Nothing uses it.

## The device's latency

Even on one clock, sound takes time to reach the ear and to come back from the mic. The page corrects every tag by
the latencies the **browser reports**: `baseLatency` + `outputLatency` + the mic's input latency. It shows the
total as **"audio latency"** in its status line:

| your device | audio latency |
|---|---|
| built-in speakers and mic | 55–61 ms |
| EarPods | 200 ms |

The EarPods' larger delay is reported by the browser, and the page corrects for it.

**Caveat:** a browser that doesn't report one of these latencies counts it as 0, and the tags are then off by the
missing part. That matters most for Bluetooth output (~150–250 ms).

## Do I need to re-test each browser or device?

**No.** An optional sanity check on a new device:

1. Look at the "audio latency" on the page. Does it look plausible for that device?
2. Check that cut-ins feel right.

## Two other things the tags fixed

- **A barge tells MOUTH what the user had really heard** when they started talking, not the server's guess.
  The server's guess ran ~0.3 s ahead.
- **The agent's turn ends when the page reports it has played everything** (`played:<id>:<samples>`), not at a
  server-side guess. A 2 s timeout is only the safety net; it never fired in the live checks.

## Headphones and "mm-hm" (I-18, open)

With headphones the mic hears no echo of the agent. Then an "mm-hm" said in a pause between the agent's sentences
stops the agent almost every time: 46–48 of 48 offline. No barge threshold separates it from a real cut-in.
With speakers, the leftover echo makes the heads more cautious. This is accepted for now (`barge_thr` 0.4).

## Where to look

- **Numbers:** `CHANGELOG.md`, 2026-10-10 (I-17).
- **Issues:** `ISSUES.md`, I-17 and I-18.
- **Code:** `app/ears/playback.py` (EARS side), `static/index.html` (the page).
