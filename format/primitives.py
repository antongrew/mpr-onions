"""Symmetric primitives, all instantiated from the `cryptography` package.

Section 4.5.1: the format deliberately does *not* use a wide-block PRP.  It uses
a nonce-based keystream generator G_k applied as a one-time-pad mask,

    {p}^(n)_k := p XOR G_k(n, |p|),

and closes the resulting malleability with an explicit MAC where tampering has
to be detectable.  Because XOR is its own inverse, forward evaluation {.} and
backward evaluation }.{ are literally the same computation.

    G_k      ChaCha20 (RFC 8439), 256-bit key, 96-bit nonce, 32-bit block counter
    KDF      HKDF-SHA-256 (RFC 5869)
    MAC/PRF  HMAC-SHA-256, truncated to the tag length
    h_K      SHA-256

G_k has to be *seekable*: the header bundle is encrypted as one long unit, but
the vacated K^d / H^d slots are masked at their own offsets inside it.  ChaCha20
gives this directly -- the keystream is a sequence of independent 64-byte
blocks, and the block counter is part of the cipher's input, so starting at
byte `offset` means starting the counter at offset // 64.

The format's authenticated tags are deliberately *not* an AEAD from the
library: an AEAD would add a tag per layer (the onion would change size in
transit) and its decryption fails on a wrong tag, whereas AE.Unwrap must never
fail at a relay.  The only non-layered ciphertexts -- the public-key ones in
pke.py -- do use ChaCha20-Poly1305.
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives import constant_time, hashes
from cryptography.hazmat.primitives import hmac as _hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

STREAM_KEY_LEN = 32     # ChaCha20 key
_NONCE_LEN = 12         # ChaCha20 nonce (RFC 8439)
_BLOCK = 64             # ChaCha20 block
_MAX_BLOCKS = 2 ** 32   # 32-bit block counter


def rand(n: int) -> bytes:
    return os.urandom(n)


def xor(a: bytes, b: bytes) -> bytes:
    """Bytewise XOR of two equal-length strings (used for tag masking)."""
    if len(a) != len(b):
        raise ValueError(f"xor length mismatch: {len(a)} vs {len(b)}")
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(
        len(a), "little")


# -------------------------------------------------------------- G_k ---------
def _nonce(label: bytes) -> bytes:
    """The format's per-field nonces are short ASCII labels; pad them to 96 bits."""
    if len(label) > _NONCE_LEN:
        raise ValueError(f"nonce label {label!r} longer than {_NONCE_LEN} B")
    return label.ljust(_NONCE_LEN, b"\x00")


def _chacha(key: bytes, nonce: bytes, offset: int):
    if len(key) != STREAM_KEY_LEN:
        raise ValueError(f"G_k needs a {STREAM_KEY_LEN}-byte key, got {len(key)} B; "
                         "derive one with stream_key()")
    block, skip = divmod(offset, _BLOCK)
    if block >= _MAX_BLOCKS:
        raise ValueError("offset beyond the ChaCha20 counter range")
    # cryptography's ChaCha20 takes counter (32-bit LE) || nonce (96-bit).
    ctx = Cipher(algorithms.ChaCha20(key, block.to_bytes(4, "little") + _nonce(nonce)),
                 mode=None).encryptor()
    if skip:
        ctx.update(bytes(skip))
    return ctx


def keystream(key: bytes, nonce: bytes, length: int, offset: int = 0) -> bytes:
    """G_k(n, .) -- `length` bytes starting at byte `offset` of the keystream."""
    if length <= 0:
        return b""
    return _chacha(key, nonce, offset).update(bytes(length))


def enc(key: bytes, nonce: bytes, data: bytes, offset: int = 0) -> bytes:
    """{data}^(nonce)_key -- forward evaluation."""
    if not data:
        return b""
    return _chacha(key, nonce, offset).update(data)


# }data{^(nonce)_key -- backward evaluation.  Identical computation; the two
# names exist only to keep the code readable against the paper.
dec = enc


# ------------------------------------------------------------ KDF / MAC -----
def kdf(key: bytes, label: bytes, out_len: int = 16) -> bytes:
    """k^(X)_i := KDF(k_i, X), the domain-separated sub-key for label X."""
    return HKDF(algorithm=hashes.SHA256(), length=out_len, salt=None,
                info=b"MPR-KDF" + label).derive(key)


def stream_key(secret: bytes, label: bytes) -> bytes:
    """A G_k key derived from a kappa-bit secret that travels in the onion.

    Seeds and replication keys are kappa = 16 bytes on the wire; ChaCha20 wants
    256 bits, so they pass through the KDF first.
    """
    return kdf(secret, b"STREAM" + label, STREAM_KEY_LEN)


def prg(seed: bytes, label: bytes, length: int) -> bytes:
    """Expand a kappa-bit seed into `length` pseudorandom bytes."""
    return keystream(stream_key(seed, label), label, length)


def mac(key: bytes, *parts: bytes, out_len: int = 16) -> bytes:
    """MAC_k(.) -- deterministic, existentially unforgeable, fixed-size tag.

    Each part is length-prefixed, so the tag binds field boundaries and not only
    the concatenation.
    """
    h = _hmac.HMAC(key, hashes.SHA256())
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.finalize()[:out_len]


def verify(key: bytes, tag: bytes, *parts: bytes) -> bool:
    """Ver_k(sigma, T) -- recompute and compare in constant time."""
    return constant_time.bytes_eq(mac(key, *parts, out_len=len(tag)), tag)


def hash_kt(data: bytes, out_len: int = 16) -> bytes:
    """h_K, the hash expanding one key-tree node into its children."""
    h = hashes.Hash(hashes.SHA256())
    h.update(b"MPR-KT" + data)
    return h.finalize()[:out_len]
