"""Header creation (Section 4.2).

An onion is built from the inside out, mirroring the peeling process.  Two
things in here are worth calling out because the paper states them only
implicitly.

1. The filler string.  Hop i vacates K^d_{i+1} and H^d_{i+1} and refills them
   with an encryption of a fixed constant under its own k^SKE_i.  But tau_{i+1}
   binds those slots, and tau_{i+1} is computed by the sender *before* hop i's
   layer exists.  So the sender cannot choose the innermost layer's tail slots
   freely: it has to sample every hop key up front and simulate forward to
   learn what the tail of the innermost layer will contain.  This is Sphinx's
   filler-string argument, applied to a two-stack header; without it the
   fixed-d construction is not forwardable at all.

2. Pinned versus free payload blocks.  An ordinary sender fixes M and fwd at
   the innermost layer and wraps outward.  A SURB creator and a multicast
   sender have to fix them at the *outer* layer instead (the first hop is the
   one that will be handed those exact bytes) and peel forward to learn what
   every inner layer holds.  Both directions are supported here; the tag
   computation is identical either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from ipaddress import IPv4Address

from . import address, pke, sharing
from .onion import Onion, StoredHeader
from .params import (
    DELTA_NONE,
    L_MAC,
    L_SKE,
    N_FFILL,
    N_FWD,
    N_HEADER,
    N_HFILL,
    N_KFILL,
    N_MFILL,
    ROLE_INTERMEDIARY,
    ROLE_RECIPIENT,
    ROLE_REPLICATION,
    Params,
)
from .primitives import STREAM_KEY_LEN, enc, kdf, keystream, mac, prg, rand, xor


def placeholder_id(params: Params) -> bytes:
    """The fixed value in P^+_{i+1} when a hop has no single next hop."""
    return address.PLACEHOLDER.packed


def encode_id(node_id, params: Params) -> bytes:
    """A party's IPv4 address, packed into the four bytes the format reserves."""
    return address.to_bytes(node_id)


def decode_id(raw: bytes):
    return address.from_bytes(raw)


def pad_delta(delta: bytes, params: Params) -> bytes:
    """Delta_i sits in H^1 of a recipient hop, padded out to a full sub-block."""
    if len(delta) > params.tau_H:
        raise ValueError("delta too large")
    return delta + rand(params.tau_H - len(delta))


def empty_delta(params: Params) -> bytes:
    return bytes([DELTA_NONE]) + rand(params.delta_len - 1)


# ------------------------------------------------------------ sub-keys ------
def subkeys(k: bytes, params: Params) -> tuple[bytes, bytes]:
    """(k^SKE_i, k^MAC_i) -- k^PRF_i is derived inside AE.Wrap/AE.Unwrap.

    k^SKE_i is a 256-bit ChaCha20 key.  It never travels, so its length does not
    enter any block size.
    """
    return kdf(k, L_SKE, STREAM_KEY_LEN), kdf(k, L_MAC, params.key_len)


def filler_K(ske: bytes, params: Params) -> bytes:
    return enc(ske, N_KFILL, b"\xff" * params.tau_K)


def filler_H(ske: bytes, params: Params) -> bytes:
    return enc(ske, N_HFILL, b"\x00" * params.tau_H)


def filler_string(skes: Sequence[bytes], params: Params) -> tuple[bytes, bytes]:
    """Accumulated K/H tail as it will look at the innermost layer.

    After hop i has run, the last slot of the next layer's stack holds
    filler_K(k^SKE_i); everything already in the tail has been peeled one more
    time at its own offset inside the bundle.  Simulating that for hops
    0..l-2 yields exactly the last l-1 slots of layer l-1.
    """
    phi_k = b""
    phi_h = b""
    k_end = params.off_K + (params.d - 1) * params.tau_K
    h_end = params.off_H + params.d * params.tau_H
    for ske in skes:
        if phi_k:
            phi_k = xor(phi_k, keystream(ske, N_HEADER, len(phi_k), k_end - len(phi_k)))
        phi_k += filler_K(ske, params)
        if phi_h:
            phi_h = xor(phi_h, keystream(ske, N_HEADER, len(phi_h), h_end - len(phi_h)))
        phi_h += filler_H(ske, params)
    return phi_k, phi_h


def _chunks(data: bytes, size: int, count: int) -> list[bytes]:
    return [data[i * size:(i + 1) * size] for i in range(count)]


# --------------------------------------------------------- path description --
@dataclass
class Hop:
    node_id: IPv4Address
    pk: bytes
    role: int = ROLE_INTERMEDIARY
    alt_id: IPv4Address | None = None   # P^-: the alternate candidate for this hop
    alt_pk: bytes | None = None
    delta: bytes | None = None     # only meaningful for a recipient hop


@dataclass
class BuiltHeader:
    onion: Onion
    keys: list[bytes]              # k_0 .. k_{l-1}
    stored: StoredHeader           # compact form, for a SURB or an M slot
    seed: bytes


def build(
    hops: Sequence[Hop],
    params: Params,
    committee: Sequence[tuple[int, bytes]],
    *,
    keys: Sequence[bytes] | None = None,
    m_last: bytes | None = None,
    fwd_last: bytes | None = None,
    seed: bytes | None = None,
    fwd_first: bytes | None = None,
    bwd: bytes | None = None,
) -> BuiltHeader:
    """Build a complete onion along `hops`.

    Free mode (default): `m_last` and `fwd_last` give the innermost layer's M
    and forward payload, and the layers are wrapped outward from there.

    Pinned mode (`seed` given): the *first* hop's M block is the seed-derived
    filler, and its forward payload is `fwd_first` (also seed-derived if that is
    None).  Used for SURBs and for multicast inner headers, whose stored form
    carries only the two stacks and the seed.
    """
    ell = len(hops)
    if not 1 <= ell <= params.d:
        raise ValueError(f"path length {ell} must satisfy 1 <= l <= d = {params.d}")
    if hops[-1].role == ROLE_INTERMEDIARY:
        raise ValueError("last hop must be a recipient or a replication node")

    seed = seed if seed is not None else rand(params.key_len)
    pinned = fwd_first is not None or m_last is None

    keys = list(keys) if keys is not None else [rand(params.key_len) for _ in range(ell)]
    sk = [subkeys(k, params) for k in keys]
    skes = [s[0] for s in sk]
    macs = [s[1] for s in sk]

    # -- per-layer M and fwd -------------------------------------------------
    if pinned:
        m_first = prg(seed, N_MFILL, params.tau_M)
        f_first = (fwd_first if fwd_first is not None
                   else prg(seed, N_FFILL, params.tau_fwd))
        m_layers = [m_first]
        f_layers = [f_first]
        for i in range(ell - 1):
            m_layers.append(xor(m_layers[-1],
                                keystream(skes[i], N_HEADER, params.tau_M, params.off_M)))
            f_layers.append(enc(skes[i], N_FWD, f_layers[-1]))
    else:
        if fwd_last is None:
            raise ValueError("free mode needs fwd_last")
        # m_last / fwd_last are the *plaintexts* the final hop recovers, so
        # they are encrypted once under that hop's own key before wrapping.
        m_layers = [None] * ell
        f_layers = [None] * ell
        m_layers[-1] = xor(m_last, keystream(skes[-1], N_HEADER, params.tau_M, params.off_M))
        f_layers[-1] = enc(skes[-1], N_FWD, fwd_last)
        for i in range(ell - 2, -1, -1):
            m_layers[i] = xor(m_layers[i + 1],
                              keystream(skes[i], N_HEADER, params.tau_M, params.off_M))
            f_layers[i] = enc(skes[i], N_FWD, f_layers[i + 1])

    # -- innermost layer -----------------------------------------------------
    phi_k, phi_h = filler_string(skes[:ell - 1], params)
    n_rand = params.d - ell                       # never-read slots, fixed once

    K = ([b""]                                    # K^1 filled in below
         + [rand(params.tau_K) for _ in range(n_rand)]
         + _chunks(phi_k, params.tau_K, ell - 1))
    # K^2..K^d and H^2..H^d of the innermost layer are never read by anyone, so
    # they may be random -- but H^1 carries Delta, which the final hop recovers
    # by decrypting the bundle, so it has to be encrypted at its own offset.
    h1_last = _h1_content(hops, ell - 1, params, committee, keys, macs)
    H = ([xor(h1_last, keystream(skes[-1], N_HEADER, params.tau_H, params.off_H))]
         + [rand(params.tau_H) for _ in range(n_rand)]
         + _chunks(phi_h, params.tau_H, ell - 1))

    bundle_ct = b"".join(K[1:]) + b"".join(H) + m_layers[-1]
    tag = mac(macs[-1], bundle_ct, f_layers[-1], out_len=params.tag_len)
    K[0] = tag + _encrypt_c(hops, ell - 1, params, keys[ell - 1], hops[-1].pk)

    # -- wrap outward --------------------------------------------------------
    for i in range(ell - 2, -1, -1):
        h1 = _h1_content(hops, i, params, committee, keys, macs)
        plain = b"".join(K[:params.d - 1]) + h1 + b"".join(H[:params.d - 1]) + m_layers[i + 1]
        ct = enc(skes[i], N_HEADER, plain)

        K = [b""] + _chunks(ct[:params.off_H], params.tau_K, params.d - 1)
        H = _chunks(ct[params.off_H:params.off_M], params.tau_H, params.d)
        m_i = ct[params.off_M:]
        assert m_i == m_layers[i]

        bundle_ct = b"".join(K[1:]) + b"".join(H) + m_i
        tag = mac(macs[i], bundle_ct, f_layers[i], out_len=params.tag_len)
        K[0] = tag + _encrypt_c(hops, i, params, keys[i], hops[i].pk)

    onion = Onion(K, H, m_layers[0], f_layers[0],
                  bwd if bwd is not None else rand(params.tau_bwd))
    onion.check_shape(params)
    stored = StoredHeader(hops[0].node_id, list(K), list(H), seed)
    return BuiltHeader(onion, list(keys), stored, seed)


def _next_field(hops: Sequence[Hop], i: int, params: Params) -> bytes:
    """P^+_{i+1}, or the fixed placeholder where there is no single next hop."""
    if hops[i].role != ROLE_INTERMEDIARY or i + 1 >= len(hops):
        return placeholder_id(params)
    return encode_id(hops[i + 1].node_id, params)


def _c_plain(hops: Sequence[Hop], i: int, params: Params, key: bytes) -> bytes:
    return key + bytes([hops[i].role]) + _next_field(hops, i, params)


def _encrypt_c(hops, i, params, key, pk) -> bytes:
    return pke.encrypt(pk, _c_plain(hops, i, params, key))


def _h1_content(hops, i, params, committee, keys, macs) -> bytes:
    """What H^1_i decrypts to, which depends on role(P_i)."""
    hop = hops[i]
    if hop.role == ROLE_RECIPIENT:
        return pad_delta(hop.delta if hop.delta is not None else empty_delta(params), params)
    if hop.role == ROLE_REPLICATION:
        # A replication node has no use for committee content.
        return rand(params.tau_H)

    nxt = i + 1
    if nxt >= len(hops) or hops[nxt].role == ROLE_RECIPIENT or hops[nxt].alt_id is None:
        # No alternate for the next hop (the final recipient never has one),
        # so there is nothing for the committee to hold.
        return rand(params.tau_H)

    # c^-_{i+1}: the same plaintext as c^+_{i+1}, under P^-_{i+1}'s public key.
    c_minus = pke.encrypt(hops[nxt].alt_pk, _c_plain(hops, nxt, params, keys[nxt]))
    shares = sharing.split(c_minus, params.nu, params.threshold)

    pref = encode_id(hops[nxt].node_id, params)
    alt = encode_id(hops[nxt].alt_id, params)
    out = []
    for (_gid, gpk), sigma in zip(committee, shares, strict=True):
        t = mac(macs[i], sigma, out_len=params.tag_len)   # same k^MAC_i as tau_i
        out.append(pke.encrypt(gpk, pref + alt + sigma + t))
    return b"".join(out)


def parse_committee_input(plain: bytes, params: Params):
    """(P^+_{i+1}, P^-_{i+1}, sigma_i,j, T_i,j)."""
    n = params.node_id_len
    pref = decode_id(plain[:n])
    alt = decode_id(plain[n:2 * n])
    sigma = plain[2 * n:2 * n + params.share_len]
    tag = plain[2 * n + params.share_len:2 * n + params.share_len + params.tag_len]
    return pref, alt, sigma, tag