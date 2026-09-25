"""(alpha*nu, nu) threshold secret sharing.

Used for exactly one thing: splitting the alternate ciphertext c^-_{i+1} across
the global committee G, so that P_i can only assemble it once at least
alpha*nu members have certified that the preferred candidate is offline
(Sections 3.2, 4.1.2, 4.4).

Shamir over GF(2^8), applied independently to each byte of the secret.  A share
is one x-coordinate byte followed by |secret| body bytes, so share length is a
fixed function of the secret length -- which is what keeps tau_I constant.
"""

from __future__ import annotations

from collections.abc import Sequence

from .primitives import rand

# --------------------------------------------------- GF(2^8) log tables -----
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x ^= (_x << 1) ^ (0x11B if _x & 0x80 else 0)
    _x &= 0xFF
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(256)")
    if a == 0:
        return 0
    return _EXP[_LOG[a] - _LOG[b] + 255]


def split(secret: bytes, n: int, t: int) -> list[bytes]:
    """Split `secret` into `n` shares, any `t` of which reconstruct it."""
    if not 1 <= t <= n <= 255:
        raise ValueError("need 1 <= t <= n <= 255")
    # One random polynomial of degree t-1 per byte, with a_0 = secret byte.
    coeffs = [bytes([s]) + rand(t - 1) for s in secret]
    shares = []
    for x in range(1, n + 1):
        body = bytearray(len(secret))
        for idx, poly in enumerate(coeffs):
            acc = 0
            for c in reversed(poly):          # Horner
                acc = _mul(acc, x) ^ c
            body[idx] = acc
        shares.append(bytes([x]) + bytes(body))
    return shares


def reconstruct(shares: Sequence[bytes]) -> bytes:
    """Lagrange-interpolate the shares at x = 0."""
    if not shares:
        raise ValueError("no shares")
    xs = [s[0] for s in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate share indices")
    bodies = [s[1:] for s in shares]
    length = len(bodies[0])
    if any(len(b) != length for b in bodies):
        raise ValueError("inconsistent share lengths")

    out = bytearray(length)
    for pos in range(length):
        acc = 0
        for j, xj in enumerate(xs):
            num, den = 1, 1
            for m, xm in enumerate(xs):
                if m == j:
                    continue
                num = _mul(num, xm)
                den = _mul(den, xj ^ xm)
            acc ^= _mul(bodies[j][pos], _div(num, den))
        out[pos] = acc
    return bytes(out)


def share_len(secret_len: int) -> int:
    return 1 + secret_len
