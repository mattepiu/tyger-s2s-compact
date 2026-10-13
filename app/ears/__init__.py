"""EARS: the turn-taking side. The configured EARS plugin (ears.plugin; tyger = nemotron-3.5-asr L15 + a GRU
turn-taker, one bundle) listens to every conversation, and this package serves the mic page + WebSocket on one port
and wires each conversation to MOUTH over its own TCP connection. Up to server.max_sessions conversations at once; the plugin batches their hops on the GPU.

Modules (app/ears/):
  __main__      the process: load the plugin, run the web app
  web           EarsApp: the page, the voice list, one Conversation per WebSocket
  conversation  Conversation: the turn-taking state machine (listening / speaking / halting, the floor queue)
  playback      AgentPlayback: the reply as the browser plays it, and the 2nd ear's audio

PRIVACY: no audio or transcript is written to disk or logs; only timings and event types.

Run: HF_HOME=/workspace/.hf_home .venv/bin/python -m app.ears
"""
