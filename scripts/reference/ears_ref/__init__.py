"""The research reference for nemotron-tyger-1.0, kept unchanged from the experiments (exp-4 / exp-7 / exp-12): the
single-session EndpointEars (on LiveEars, EncoderStream and the research model classes) whose per-frame decisions
the live engine ports line by line. Not used live: scripts/validate_ears_batched.py checks the live engine
(app/plugins/ears/nemotron_tyger) against it, and scripts/validate_heads.py checks the release's weights against
this code on the checkpoint the release was exported from (its model card's provenance):
"""
SOURCE_REPO = "nineninesix/tyger-exp10-heads"            # private
SOURCE_FILE = "exp12/ep_s0/ep_best.pt"                   # exp-12 arm C, seed 0, step 30,000
