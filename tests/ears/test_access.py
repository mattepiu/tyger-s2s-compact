"""Who may open a conversation (app/ears/access.py), GPU-free: the token check (HS256 only, the signature, aud, exp
and its 5-minute cap, the jti used once, the optional origin). Tokens are made here with the stdlib, as the other app's backend would make
them with `jose` (tests/fakes.py). The secrets and claims are made up.
Run: .venv/bin/python tests/ears/test_access.py"""
import os, sys, json

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]                      # the repo (app/) and tests/ (fakes.py)
from app.ears.access import TokenRefused, Tokens  # noqa: E402
from fakes import b64url as b64, make_token as signed  # noqa: E402

SECRET = "test-secret-not-a-real-one"
NOW = 1_800_000_000.0


def make_token(secret=SECRET, header=None, sign=True, **over):
    return signed(secret, NOW, header=header, sign=sign, **{"jti": "jti-1", **over})


def refused(tokens, token, origin=None, use=True):
    try:
        tokens.check(token, origin, use)
    except TokenRefused as e:
        return e.reason
    return None


def test_a_valid_token():
    t = Tokens(SECRET, clock=lambda: NOW)
    claims = t.check(make_token(), None, use=True)
    assert claims["aud"] == "tyger" and claims["jti"] == "jti-1"
    assert refused(Tokens(SECRET, clock=lambda: NOW), make_token(aud=["other", "tyger"])) is None   # aud as a list
    assert refused(Tokens(SECRET, clock=lambda: NOW), make_token(exp=NOW + 300)) is None            # the cap itself


def test_refused_tokens():
    cases = {
        "missing": (None, "malformed"),
        "empty": ("", "malformed"),
        "two parts": ("a.b", "malformed"),
        "not base64": ("@@@.###.$$$", "malformed"),
        "alg none, unsigned": (make_token(header={"alg": "none", "typ": "JWT"}, sign=False), "algorithm"),
        "alg none, signed": (make_token(header={"alg": "none"}), "algorithm"),
        "alg HS512": (make_token(header={"alg": "HS512"}), "algorithm"),
        "alg RS256": (make_token(header={"alg": "RS256"}), "algorithm"),
        "no alg": (make_token(header={"typ": "JWT"}), "algorithm"),
        "bad signature": (make_token(secret="another-secret"), "signature"),
        "no signature": (make_token(sign=False), "signature"),
        "wrong audience": (make_token(aud="someone-else"), "audience"),
        "no audience": (make_token(aud=None), "audience"),
        "expired": (make_token(exp=NOW - 1), "expired"),
        "no exp": (make_token(exp=None), "no exp"),
        "exp a string": (make_token(exp="soon"), "no exp"),
        "exp too far ahead": (make_token(exp=NOW + 3600), "exp too far ahead"),
        "not yet valid": (make_token(nbf=NOW + 60), "not yet valid"),
        "no jti": (make_token(jti=None), "no jti"),
        "empty jti": (make_token(jti=""), "no jti"),
    }
    for name, (token, reason) in cases.items():
        got = refused(Tokens(SECRET, clock=lambda: NOW), token)
        assert got == reason, (name, got)
    tampered = make_token().split(".")                               # a claim changed after signing
    tampered[1] = b64(json.dumps({"aud": "tyger", "exp": NOW + 120, "jti": "jti-2"}).encode())
    assert refused(Tokens(SECRET, clock=lambda: NOW), ".".join(tampered)) == "signature"


def test_origin_claim():
    t = Tokens(SECRET, clock=lambda: NOW)
    token = make_token(origin="https://app.example.com")
    assert refused(t, token, "https://evil.example", use=False) == "origin"
    assert refused(t, token, None, use=False) == "origin"            # no Origin: it can't match
    assert refused(t, token, "https://app.example.com") is None


def test_one_token_one_conversation():
    """A jti opens one conversation; checking without using (/voices) doesn't spend it; once a used token expires,
    it's forgotten (and refused as expired anyway)."""
    now = [NOW]
    t = Tokens(SECRET, clock=lambda: now[0])
    token = make_token(jti="once")
    assert refused(t, token, use=False) is None and refused(t, token, use=False) is None
    assert refused(t, token) is None
    assert refused(t, token) == "used"
    assert refused(t, make_token(jti="another")) is None
    now[0] = NOW + 121
    assert refused(t, make_token(jti="later", exp=now[0] + 60)) is None
    assert "once" not in t.used and refused(t, token) == "expired"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
