"""The backward payload's encrypt-then-mask construction (Section 4.1.4).

Four operations, all taking nothing but a hop's single key k_i and re-deriving
k^SKE_i and k^PRF_i internally:

  AE.Enc     run once by the replier, under the one key they already hold
  AE.Unwrap  run by *every* hop while peeling, unconditionally
  AE.Wrap    the inverse of Unwrap; only the header's creator ever runs it
  AE.Dec     run once by the header's creator, after undoing every Unwrap

bwd is a pair (ciphertext, tag).  Unwrap peels one keystream layer off the
ciphertext and re-masks the tag; no hop ever checks that tag and no hop's peel
can fail because of it.  Only the party who can regenerate every hop key on the
path can strip the masks back off and verify -- which is precisely the
tamper-evidence property of Section 4.3.3, with no separate integrity key.

Masking convention: the tag is always masked with a PRF value computed over the
ciphertext in its *outer* (more-wrapped) form, which makes Wrap and Unwrap exact
inverses of each other.
"""

from __future__ import annotations

from .params import L_PRF, L_SKE, N_BWD, Params
from .primitives import STREAM_KEY_LEN, enc, kdf, mac, verify, xor


def _subkeys(k: bytes, params: Params):
    return kdf(k, L_SKE, STREAM_KEY_LEN), kdf(k, L_PRF, params.key_len)


def _split(bwd: bytes, params: Params):
    return bwd[:params.bwd_ct_len], bwd[params.bwd_ct_len:]


def encrypt(k: bytes, msg: bytes, params: Params) -> bytes:
    """AE.Enc_k(m') -- what the replier runs, once, under k_Bob.

    m' is exactly beta = params.bwd_ct_len bytes; encoding a shorter reply into
    it is up to the application.
    """
    if len(msg) != params.bwd_ct_len:
        raise ValueError(f"reply content must be exactly beta = {params.bwd_ct_len} B")
    k_ske, k_prf = _subkeys(k, params)
    ct = enc(k_ske, N_BWD, msg)
    return ct + mac(k_prf, ct, out_len=params.tag_len)


def decrypt(k: bytes, bwd: bytes, params: Params) -> bytes | None:
    """AE.Dec_k(bwd) -- returns the plaintext, or None if the tag fails."""
    ct, tag = _split(bwd, params)
    k_ske, k_prf = _subkeys(k, params)
    if not verify(k_prf, tag, ct):
        return None
    return enc(k_ske, N_BWD, ct)


def unwrap(k: bytes, bwd: bytes, params: Params) -> bytes:
    """AE.Unwrap_k(bwd) -- applied by every hop, regardless of role."""
    ct, tag = _split(bwd, params)
    k_ske, k_prf = _subkeys(k, params)
    inner_ct = enc(k_ske, N_BWD, ct)
    inner_tag = xor(tag, mac(k_prf, ct, out_len=params.tag_len))
    return inner_ct + inner_tag


def wrap(k: bytes, bwd: bytes, params: Params) -> bytes:
    """AE.Wrap_k(bwd) -- the exact inverse of unwrap."""
    ct, tag = _split(bwd, params)
    k_ske, k_prf = _subkeys(k, params)
    outer_ct = enc(k_ske, N_BWD, ct)
    outer_tag = xor(tag, mac(k_prf, outer_ct, out_len=params.tag_len))
    return outer_ct + outer_tag
