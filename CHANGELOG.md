# Changelog

All notable changes to tyger are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Everything since the imported v1 baseline. It becomes the first release.

### Added
- **4 conversations on one GPU**, batched in both processes; the next visitor gets a "busy" page.
- **Voice cloning** with VoxCPM2 from a reference clip in `voices/`, with an optional transcript.
- **Plugins:** the LLM, the TTS and the turn-taking model are chosen in `config/default.yaml`.
- **An expert:** the local model hands hard questions to a hosted model (`ask_expert`), and its answer streams
  straight to the voice.
- **Live facts** (weather, news, prices) through the expert's web search. Its sources show as links under the
  answer on the page and are never read out.
- **Background tasks:** research that takes minutes runs while you keep talking, and the agent tells you the result
  at the next pause.
- **Interrupted lookups carry on:** cut in before hearing an answer, and it's still found and told to you later.
- **`openai-responses` plugin:** OpenAI's Responses API with web search, the default expert.
- **`anthropic` plugin:** Claude as the alternative expert (not yet tested against the API).
- **`fake` plugins** and `config/fake.yaml`: the whole stack runs on a CPU, with no downloads.
- **All prompts in one file,** `config/prompts.yaml`, checked at startup.
- **Config checks:** a typo in the config stops startup and names the key.
- **Tuning sliders** on the page (`?nodebug` hides them), built from the turn-taking plugin's thresholds.
- **A new page** (`web/`): one mic button and the characters as tabs. An orb in the middle (OrbKit's Ion)
  follows the conversation and the voices, in each character's own colours. The exchange shows as text: your words
  as you say them, the agent's backchannels in a bubble beside them, its reply with its sources. A history
  holds the whole conversation. A Tuning panel over the page has the meters and the sliders, to tune while you
  talk. `run.sh` builds it.
- **Tests that need no GPU,** and a check of the web page in jsdom.
- **A routing test** (`scripts/bench/escalation.py`): 119 synthetic cases, no API cost.
- **Measurement scripts:** smoke and load tests on the running stack, GPU parity checks, stack comparisons.

### Changed
- **The new page is the default at `/`; the classic page moved to `/classic`.** Both are on the one port, :7860.
- **The agent is Ori,** a warm, witty character with a chill vibe; the expert answers in the same voice.
- **Each voice can have its own character:** picking a voice on the page switches who you're talking to, and the
  picker shows for everyone.
- **English only:** the agent answers in English, whatever language it hears.
- **Responder: Gemma 4 E4B,** FP8 at load, replaces Qwen3-4B, for reliable tool calls.
- **TTS: VoxCPM2** replaces MOSS-TTS-Realtime: first audio per request ~480 → ~80 ms.
- **Two processes in one venv,** instead of three in three. Both of MOUTH's models run in-process on vLLM.
- **Replies reach the TTS in whole sentences,** so the intonation doesn't restart mid-sentence.
- **Tool lead-ins start sooner and fit the request** ("Checking the time in Tokyo."), and "still looking" lines
  cover slow tools.
- **Backchannels react to what you're saying,** instead of a canned "mm-hm".
- **Barge-in follows what you actually heard:** the page tags the mic with what it was playing.
- **A turn ends when the page has played the whole reply,** not when generation ends.
- **Cut-ins get through sooner:** the barge-in threshold went from 0.5 to 0.4.
- **Half the bandwidth:** int16 audio both ways, and the mic is captured with an AudioWorklet.
- **Faster turn-taking:** audio features on the GPU, and a speech decoder 3–5× faster.
- **The voice clip is encoded once** and kept in memory.
- **The turn-taking heads** load from the `TYGER-S2S-endpointer` release.
- **Code layout:** `app/` is split into `ears/`, `mouth/`, `plugins/`, `prompts/`, `runtime/` and `tools/`, and the
  configs are in `config/`.
- **Docs:** README, DEVELOPING, CHANGELOG, ISSUES and HANDOFF replace the old notes and design docs.

### Fixed
- An interrupted reply stays in the conversation history, cut to what you heard.
- The first words of an interruption are kept: the first 3 words are recognised 83% of the time, up from 14%.
- Interrupting during a tool call stops the agent at once, instead of after up to 15 s.
- Turns longer than 20 s are transcribed whole, not just their last 20 s.
- Tool calls stay in history, so later turns keep using tools instead of making answers up.
- A tool or expert answer is no longer lost when you cut in before hearing it.
- The agent's own voice, echoed into the mic, no longer counts as you interrupting.
- A backchannel that was still being made when the reply started is cancelled.
- When MOUTH goes away, the page says so instead of going silent.
- A first start no longer risks running the machine out of memory while it compiles GPU kernels.
- The expert's typographic apostrophes and quotes (’ “ ”) reach the voice as written, not as spaces ("I’d" was said
  "I d").

### Removed
- **The silence fallback:** only the turn-taking model ends a turn.
- **The weather and web-search tools,** and Tavily: live facts go to the expert's web search.

## v1 baseline — 2026-10-02

Imported from the research code (`07d651e`):
- three processes in three venvs;
- MOSS-TTS-Realtime (first audio ~480 ms), and Qwen3-4B on a vLLM HTTP server;
- one conversation per GPU, using ~30.5 GB.
