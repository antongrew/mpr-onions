"""Forward-payload layout and the multicast key tree.

The forward half of U always has the same internal shape:

    SURB slot | content field

Both are fixed-size.  The SURB slot is always present; when no reply block is
attached it holds pseudorandom filler, so the size never reveals whether a
reply path came along.  The content field is exactly mu = params.msg_max bytes
and is opaque to the format: how a message is encoded into it -- its length,
any padding, whether the SURB slot is in use -- is left to the application.
"""

from __future__ import annotations

from .params import Params
from .primitives import STREAM_KEY_LEN, enc, hash_kt, rand

N_MC = b"MC"


def pack(content: bytes, surb: bytes | None, params: Params) -> bytes:
    """Lay out the plaintext of fwd for a recipient hop."""
    if len(content) != params.msg_max:
        raise ValueError(f"content must be exactly mu = {params.msg_max} B, got "
                         f"{len(content)} B; encoding a shorter message is up to "
                         "the application")
    if surb is not None and len(surb) != params.surb_len:
        raise ValueError("SURB has the wrong length")
    slot = surb if surb is not None else rand(params.surb_len)
    return slot + content


def unpack(field: bytes, params: Params) -> tuple[bytes, bytes]:
    """(content, SURB slot) from a recovered forward payload.

    The slot is returned whether or not it holds a SURB; the format cannot tell
    a real reply block from filler, and it does not need to.
    """
    return field[params.surb_len:], field[:params.surb_len]


# ------------------------------------------------------------- key tree -----
def kt_root(seed: bytes, params: Params) -> bytes:
    """K[ ] = h_K(S_mc); the payload key is h_K(K[ ]), sized for G_k."""
    return hash_kt(hash_kt(seed, params.key_len), STREAM_KEY_LEN)


def kt_child(node: bytes, index: int, params: Params) -> bytes:
    """K[x;i] := h_K(K[x] + i)."""
    return hash_kt(node + index.to_bytes(4, "big"), params.key_len)


def mc_encrypt(seed: bytes, plaintext: bytes, params: Params) -> bytes:
    """xi = {m}_{h_K(K[ ])} -- encrypted exactly once, whatever g is."""
    return enc(kt_root(seed, params), N_MC, plaintext)


def mc_decrypt(seed: bytes, xi: bytes, params: Params) -> bytes:
    return mc_encrypt(seed, xi, params)
