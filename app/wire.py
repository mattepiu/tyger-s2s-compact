"""Tiny length-prefixed message framing for the EARS<->MOUTH local socket (pure stdlib; works in both
venvs). Frame = 1 kind byte + 4-byte big-endian length + payload. Control payloads are JSON; audio is raw
float32 bytes.  Kinds: T=turn(json), C=backchannel(json), B=barge, A=audio(bytes), X=text(json), D=done(json)."""
import struct, json


def send_msg(sock, kind, payload=b""):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload).encode()
    elif isinstance(payload, str):
        payload = payload.encode()
    sock.sendall(struct.pack(">cI", kind, len(payload)) + payload)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def recv_msg(sock):
    h = _recv_exact(sock, 5)
    if not h:
        return None, None
    kind, ln = struct.unpack(">cI", h)
    payload = _recv_exact(sock, ln) if ln else b""
    if payload is None:
        return None, None
    return kind, payload


# ---- asyncio variants (EARS side) --------------------------------------------------------------
async def async_send_msg(writer, kind, payload=b""):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload).encode()
    elif isinstance(payload, str):
        payload = payload.encode()
    writer.write(struct.pack(">cI", kind, len(payload)) + payload)
    await writer.drain()


async def async_recv_msg(reader):
    h = await reader.readexactly(5)
    kind, ln = struct.unpack(">cI", h)
    payload = await reader.readexactly(ln) if ln else b""
    return kind, payload
