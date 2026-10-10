"""nemotron-tyger-1.0's fixed numbers, in one place: the audio grid nemotron-3.5-asr sets, the layer the
tyger 1.0 heads were trained on, the heads' state vocabulary, and the turn rules around them. Light (no imports):
the plugin class and the page's slider spec read it without loading a model."""

# ---- nemotron-3.5-asr: the audio grid -----------------------------------------------------------------------
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"   # the encoder the heads were trained on (the release pins it)
SR = 16000                       # its input rate
FRAME_S = 0.08                   # one encoder frame (12.5 Hz) = one turn-taking decision
HOP_FRAMES = 4                   # frames per hop: EARS steps every 320 ms
HOP_N = int(round(HOP_FRAMES * FRAME_S * SR))           # samples per hop (5120)
WINDOW_S = 8.0                   # the rolling window each hop re-encodes
L15 = 15                         # the encoder layer the heads read (of 24): encodes stop there
ENC_HIDDEN = 1024                # its width

# ---- the tyger 1.0 heads (their release's config.json holds the backbone; these are the inputs' ids) ----------
LISTEN, HOLD, SPEAK, BACKCHANNEL = 0, 1, 2, 3           # the state head's classes
STATE_NAMES = ("LISTEN", "HOLD", "SPEAK", "BACKCHANNEL")
N_STATE = 4
START = 4                        # the previous-state input's begin-of-sequence id
TEXT_PAD = 0                     # the agent-text input when no agent word is active (always, live)

# ---- turn rules around the heads ----------------------------------------------------------------------------
VAD_DB = -50.0                   # a frame above this energy (dB) is voiced
MIN_SPEECH_MS = 350.0            # a turn needs this much voiced audio before it can end or get a backchannel
WARMUP_FRAMES = 6                # frames after a reset before the heads' decisions count
TURN_MAX_S = 20.0                # longest audio one transcription gets (sizes the RNN-T graph); a safety cap
SEG_S = 15.0                     # a turn longer than this is transcribed in blocks as it grows (I-05),
SEG_SEARCH_S = 4.0               #   cut at the quietest frame of the block's last SEG_SEARCH_S
BARGE_PRE_S = 2.0                # user audio kept while the agent speaks: seeds the turn a barge-in starts (I-02)
ONSET_GAP_FRAMES = 8             # while the agent speaks, a pause longer than this (640 ms: more than a pause inside an
                                 # utterance) ends a stretch of the user's speech
ONSET_MAX_S = 4.0                # a barge's onset (when the user started talking) reaches back at most this far (the
                                 # barge's p90 latency is ~3.5 s: scripts/validate/silent_wait.py)
