"""Who may open a conversation, checked before the WebSocket upgrade (README -> "API mode").

  * Tokens: with TYGER_API_SECRET set, a short-lived JWT signed with HS256 by the other app's backend: `aud` "tyger",
    `exp` at most 5 minutes ahead, a `jti` that opens one conversation only, and an optional `origin` that must match
    the request's (it ties a token to the app's page). Only HS256 is accepted (`none` included, every other algorithm
    is refused), and the signature is compared in constant time. The stdlib is enough. There is no origin check
    otherwise: without the secret, anyone can connect.
  * AccessLog: the request log, without query strings (the token travels in one). The secret and the token are never
    logged.
"""
from __future__ import annotations
import base64, binascii, hashlib, hmac, json, os, re, time

from aiohttp.abc import AbstractAccessLogger

SECRET_ENV = "TYGER_API_SECRET"
AUDIENCE = "tyger"
MAX_AHEAD_S = 300            # a token's exp may be at most this far ahead
SKEW_S = 10                  # the signer's clock may be this far ahead of EARS's (exp's cap, nbf)
B64URL = re.compile(r"[A-Za-z0-9_-]*")


def secret_from_env():
    return os.environ.get(SECRET_ENV, "").strip() or None


class TokenRefused(Exception):
    def __init__(self, reason):
        super().__init__(reason); self.reason = reason


def _b64(part):
    if not B64URL.fullmatch(part):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


class Tokens:
    """The token check. `used` remembers each conversation's jti until its exp, so a token opens one conversation."""
    def __init__(self, secret, clock=time.time):
        self.key = secret.encode(); self.clock = clock
        self.used = {}                 # jti -> exp

    def check(self, token, origin, use):
        """The token's claims; TokenRefused(reason) when it's refused. `use`: it opens a conversation, so its jti is
        spent. The checks run in order, the signature before any claim is read."""
        try:
            h64, p64, s64 = (token or "").split(".")
            header, claims, sig = json.loads(_b64(h64)), json.loads(_b64(p64)), _b64(s64)
        except (ValueError, binascii.Error):          # json's and the split's errors are ValueErrors too
            raise TokenRefused("malformed") from None
        if not isinstance(header, dict) or header.get("alg") != "HS256":
            raise TokenRefused("algorithm")
        want = hmac.new(self.key, f"{h64}.{p64}".encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(want, sig):
            raise TokenRefused("signature")
        if not isinstance(claims, dict):
            raise TokenRefused("malformed")
        aud = claims.get("aud")
        if aud != AUDIENCE and not (isinstance(aud, list) and AUDIENCE in aud):
            raise TokenRefused("audience")
        now = self.clock()
        exp, nbf = claims.get("exp"), claims.get("nbf")
        if not _number(exp):
            raise TokenRefused("no exp")
        if exp <= now:
            raise TokenRefused("expired")
        if exp > now + MAX_AHEAD_S + SKEW_S:
            raise TokenRefused("exp too far ahead")
        if _number(nbf) and nbf > now + SKEW_S:
            raise TokenRefused("not yet valid")
        jti = claims.get("jti")
        if not isinstance(jti, str) or not jti:
            raise TokenRefused("no jti")
        if "origin" in claims and claims["origin"] != origin:
            raise TokenRefused("origin")
        if use:
            self.used = {j: e for j, e in self.used.items() if e > now}      # spent tokens, until they expire
            if jti in self.used:
                raise TokenRefused("used")
            self.used[jti] = exp
        return claims


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class AccessLog(AbstractAccessLogger):
    """One line per request: who, the method and the path without its query string, the status, the time."""
    def log(self, request, response, time):
        self.logger.info('%s "%s %s" %s %.0f ms', request.remote, request.method, request.path, response.status,
                         time * 1e3)
