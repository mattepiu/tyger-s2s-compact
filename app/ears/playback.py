"""The agent's reply audio and what the page played of it, on the page's own clock (I-17).

The page plays the reply and records the mic on one AudioContext, so it knows which reply sample the user was
hearing when each mic sample was captured. Every mic block carries that tag: the reply's id and its sample (at the
24 kHz wire rate) at the block's first mic sample, or -1 when nothing of a reply played. AgentPlayback keeps the
current reply's audio at the EARS rate and builds the turn-taking model's 2nd ear from those tags, with silence
exactly where nothing played. The tags also say what the user had heard by the time they started talking (a
barge's played_s), and the page reports when it has played a whole reply (`played:<id>:<samples>`), which ends the
agent's turn. Before, EARS paired the reply with the mic in arrival order and guessed playback from its own clock,
so the 2nd ear ran ~150-700 ms ahead of what the user heard (scripts/validate/second_ear.py --offset-ms).

The reply audio is resampled 24 -> 16 kHz by one stateful polyphase resampler per reply (app/runtime/audio.py), so
the ear doesn't depend on where MOUTH's chunks end. play_until, the server's guess at when the page will have played
everything, stays only as the safety net when no report comes.

Wire (both little-endian): a mic block is [int32 reply id][int32 agent sample, -1: none][int16 PCM at the EARS
rate]; audio to the page is [int32 reply id, 0: a backchannel][int16 PCM at 24 kHz]. Backchannel audio is never
tagged, so it stays out of the 2nd ear.
"""
from __future__ import annotations
import struct, time
from collections import deque

import numpy as np

from app.plugins.tts import WIRE_SR
from app.runtime.audio import StreamingAudioResampler

LEAD_IN_S = 0.04                 # the page starts a chunk this long after it arrives when nothing is queued
BACKCHANNEL_ID = 0               # backchannel audio: played by the page, never tagged
MIC_HEADER = struct.Struct("<ii")        # a mic block: reply id, agent sample at 24 kHz (-1: nothing playing)
AUDIO_HEADER = struct.Struct("<i")       # audio to the page: its reply id (BACKCHANNEL_ID for a backchannel)
TAGS_KEPT_S = 10.0               # mic blocks' tags kept this long (a barge's onset reaches back at most ~4 s)


def to_pcm16(x):
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")


def audio_frame(rid, a24):
    """Reply or backchannel audio (float32 at WIRE_SR) as the page gets it: the id, then int16 samples."""
    return AUDIO_HEADER.pack(rid) + to_pcm16(a24).tobytes()


def mic_block(data):
    """A mic block from the page -> (reply id, agent sample at 24 kHz or -1, float32 samples at the EARS rate);
    None if it is too short to be one."""
    if len(data) < MIC_HEADER.size or (len(data) - MIC_HEADER.size) % 2:
        return None
    rid, at24 = MIC_HEADER.unpack_from(data)
    pcm = np.frombuffer(data, "<i2", offset=MIC_HEADER.size).astype(np.float32) / 32768.0
    return rid, at24, pcm


class AgentPlayback:
    def __init__(self, ears_sr):
        self.ears_sr = ears_sr
        self.rid = 0                   # the current reply's id: replies and announcements count up from 1
        self.audio = np.zeros(WIRE_SR, np.float32)   # its audio at the EARS rate (the first n16 samples)
        self.n16 = 0
        self.sent24 = 0                # its samples sent to the page (at 24 kHz)
        self.played24 = 0              # how much of it the page reported played
        self.play_until = 0.0          # the server's guess (perf_counter) at when the page will have played it
        self.arrival_pos = None        # the mic position arrival-order pairing would have put its start at
        self.tag0_pos = None           # the mic position the tags put its start at
        self.tags = deque()            # (mic position, n, reply id, agent sample at 24 kHz) per mic block
        self.to_ears_rate = StreamingAudioResampler(WIRE_SR, ears_sr)

    # ---- the reply, as MOUTH sends it ----
    def start(self):
        """A new reply (or announcement): a new id, nothing of it sent, played or tagged yet. Returns the id."""
        self.rid += 1
        self.n16 = self.sent24 = self.played24 = 0
        self.play_until = 0.0; self.arrival_pos = self.tag0_pos = None
        self.to_ears_rate.reset()
        return self.rid

    def add(self, a24, next_hop_pos):
        """One chunk of the current reply (float32 at WIRE_SR) went to the page. `next_hop_pos`: the mic position
        the next hop starts at, where arrival-order pairing would have put it (logged against the tags)."""
        if self.arrival_pos is None:                   # its first chunk
            self.arrival_pos = next_hop_pos
        self.play_until = max(self.play_until, time.perf_counter() + LEAD_IN_S) + len(a24) / WIRE_SR
        self.sent24 += len(a24)
        x = self.to_ears_rate.process(a24)
        need = self.n16 + len(x)
        if need > len(self.audio):                     # grow by doubling: a long reply arrives in hundreds of chunks
            grown = np.zeros(max(need, 2 * len(self.audio)), np.float32)
            grown[:self.n16] = self.audio[:self.n16]; self.audio = grown
        self.audio[self.n16:need] = x; self.n16 = need

    # ---- the mic, tagged by the page ----
    def agent_block(self, rid, at24, n, pos):
        """The 2nd ear for a mic block of n samples at mic position `pos`, tagged (rid, at24): the current reply's
        audio from the sample the user was hearing, silence where nothing of it played (another reply's tag, -1,
        or past what has arrived). The tag is kept for heard_s. A tag past what was sent can't be playing (a page
        that kept an old connection's counts, where ids also started at 1): it counts as -1."""
        out = np.zeros(n, np.float32)
        if at24 >= self.sent24:
            at24 = -1
        if rid == self.rid and at24 >= 0:
            i = (at24 * self.ears_sr + WIRE_SR // 2) // WIRE_SR
            k = max(0, min(n, self.n16 - i))
            out[:k] = self.audio[i:i + k]
            if self.tag0_pos is None:
                self.tag0_pos = pos - i
        self.tags.append((pos, n, rid, at24))
        while self.tags and self.tags[0][0] < pos - TAGS_KEPT_S * self.ears_sr:
            self.tags.popleft()
        return out

    def heard_s(self, pos):
        """How much of the current reply the user had heard by mic position `pos` (s): the furthest of its samples
        the tags show playing up to there (0 if none)."""
        best = 0.0
        for p, n, rid, at24 in self.tags:
            if rid == self.rid and at24 >= 0 and p <= pos:
                best = max(best, at24 + min(n, pos - p) * WIRE_SR / self.ears_sr)
        return min(best, self.sent24) / WIRE_SR

    def page_played(self, rid, samples):
        """The page's report: it has played the first `samples` (24 kHz) of reply `rid`, and has nothing queued."""
        if rid == self.rid:
            self.played24 = max(self.played24, int(samples))

    def page_done(self):
        """The page has played everything sent of the current reply (its report, not a guess)."""
        return self.played24 >= self.sent24

    def offset_ms(self):
        """How far arrival-order pairing would have put the 2nd ear ahead of what the user heard (ms), once a tag of
        the current reply came; None before."""
        if self.arrival_pos is None or self.tag0_pos is None:
            return None
        return (self.tag0_pos - self.arrival_pos) * 1000.0 / self.ears_sr

    def estimate_s(self):
        """The server's own guess at how much the page has played (s): only for the logs, next to the tags."""
        return max(0.0, self.sent24 / WIRE_SR - max(0.0, self.play_until - time.perf_counter()))
