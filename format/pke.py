"""Enc_pk(.) / Dec_sk(.) -- the public-key encryption used for c_i, c^-_i and
every committee input I_i,j.

An ECIES-style hybrid built entirely from `cryptography`:

    X25519 ephemeral key exchange
    HKDF-SHA-256 over the shared secret, bound to both public keys
    ChaCha20-Poly1305 over the plaintext

    Enc_pk(m) = epk || ChaCha20-Poly1305_k(nonce = 0^96, m)

Every encryption uses a fresh ephemeral key, hence a fresh AEAD key, so the
all-zero nonce is never reused under one key.  Ciphertexts are exactly
|m| + 48 bytes (32-byte epk, 16-byte Poly1305 tag), which is what lets tau_K and
tau_I be protocol-wide constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

EPK_LEN = 32
TAG_LEN = 16
OVERHEAD = EPK_LEN + TAG_LEN
_NONCE = bytes(12)
_INFO = b"MPR-PKE"


def _raw_pk(key: X25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


# ---------------------------------------------------------------- keys ------
@dataclass(frozen=True)
class SecretKey:
    """A party's private key, loaded once and kept in memory, as a real node
    would.  Loading an X25519 key from its raw bytes derives the public key,
    a full scalar multiplication; doing that on every decryption would double
    the public-key cost of a hop."""

    key: X25519PrivateKey
    pk: bytes


@dataclass(frozen=True)
class KeyPair:
    sk: SecretKey
    pk: bytes

    @staticmethod
    def generate() -> KeyPair:
        key = X25519PrivateKey.generate()
        pk = _raw_pk(key)
        return KeyPair(SecretKey(key, pk), pk)


def ct_len(plain_len: int) -> int:
    return plain_len + OVERHEAD


def _aead(shared: bytes, epk: bytes, pk: bytes) -> ChaCha20Poly1305:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
               info=_INFO + epk + pk).derive(shared)
    return ChaCha20Poly1305(key)


def encrypt(pk: bytes, plaintext: bytes) -> bytes:
    """Enc_pk(m) -> epk || ciphertext || tag."""
    esk = X25519PrivateKey.generate()
    epk = _raw_pk(esk)
    shared = esk.exchange(X25519PublicKey.from_public_bytes(pk))
    return epk + _aead(shared, epk, pk).encrypt(_NONCE, plaintext, None)


def decrypt(sk: SecretKey, ciphertext: bytes) -> bytes | None:
    """Dec_sk(c) -> plaintext, or None if the ciphertext is not for us.

    Exactly one X25519 operation: the exchange with the ephemeral key.
    """
    if len(ciphertext) < OVERHEAD:
        return None
    epk, body = ciphertext[:EPK_LEN], ciphertext[EPK_LEN:]
    try:
        # exchange() rejects low-order points (all-zero shared secret).
        shared = sk.key.exchange(X25519PublicKey.from_public_bytes(epk))
        return _aead(shared, epk, sk.pk).decrypt(_NONCE, body, None)
    except (InvalidTag, ValueError):
        return None
