"""Message processing at mix nodes (Section 4.4).

A processing party runs a fixed sequence: open K^1 and check the tag, decrypt
the header bundle as one unit, then dispatch on the role recovered from c_i.
The backward half is unwrapped on every path through this code, before the role
is even looked at, so its presence or absence signals nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address

from . import ae, pke, sharing
from .header import (
    decode_id,
    filler_H,
    filler_K,
    parse_committee_input,
    placeholder_id,
    subkeys,
)
from .onion import Onion, StoredHeader
from .params import (
    N_FWD,
    N_HEADER,
    N_REPL,
    ROLE_INTERMEDIARY,
    ROLE_NAMES,
    Params,
)
from .primitives import dec, enc, rand, stream_key, verify


class DropError(Exception):
    """The onion is dropped at this hop."""


@dataclass
class Peeled:
    """Everything hop i holds after Steps 1 and 2."""

    key: bytes                 # k_i
    ske: bytes                 # k^SKE_i
    mac_key: bytes             # k^MAC_i
    role: int
    next_id: IPv4Address | None   # P^+_{i+1}, or None at a non-forwarding hop
    h1: bytes                  # the decrypted H^1_i
    m: bytes                   # the decrypted M_i
    fwd: bytes                 # the peeled forward half
    bwd: bytes                 # the unwrapped backward half
    K_next: list[bytes]
    H_next: list[bytes]

    def next_onion(self) -> Onion:
        return Onion(self.K_next, self.H_next, self.m, self.fwd, self.bwd)


@dataclass
class Node:
    node_id: IPv4Address
    sk: pke.SecretKey          # was: sk: bytes
    pk: bytes
    online: bool = True

    def peel(self, onion: Onion, params: Params) -> Peeled:
        # Step 1 -- open the outer key block and check the tag.
        plain = pke.decrypt(self.sk, onion.c(params))
        if plain is None:
            raise DropError("c_i does not decrypt under this node's secret key")
        key = plain[:params.key_len]
        role = plain[params.key_len]
        raw_next = plain[params.key_len + 1:]
        if role not in ROLE_NAMES:
            raise DropError("unknown role")
        ske, mac_key = subkeys(key, params)

        bundle = onion.bundle(params)
        if not verify(mac_key, onion.K[0][:params.tag_len], bundle, onion.fwd):
            raise DropError("tau_i failed to verify")

        # Unconditional, regardless of role and of whether a reply is in transit.
        bwd = ae.unwrap(key, onion.bwd, params)

        # Step 2 -- decrypt the header bundle as a single unit.
        pt = dec(ske, N_HEADER, bundle)
        k_rest = [pt[i * params.tau_K:(i + 1) * params.tau_K] for i in range(params.d - 1)]
        h_all = [pt[params.off_H + i * params.tau_H:params.off_H + (i + 1) * params.tau_H]
                 for i in range(params.d)]
        m = pt[params.off_M:]
        fwd = dec(ske, N_FWD, onion.fwd)

        next_id = None
        if role == ROLE_INTERMEDIARY and raw_next != placeholder_id(params):
            next_id = decode_id(raw_next)

        return Peeled(
            key=key, ske=ske, mac_key=mac_key, role=role, next_id=next_id,
            h1=h_all[0], m=m, fwd=fwd, bwd=bwd,
            # Shift both stacks down one and refill the vacated slot, so that
            # every onion always has exactly d sub-blocks.
            K_next=k_rest + [filler_K(ske, params)],
            H_next=h_all[1:] + [filler_H(ske, params)],
        )


# ------------------------------------------------------------- committee ----
@dataclass
class CommitteeResponse:
    sigma: bytes
    tag: bytes
    preferred: IPv4Address
    alternate: IPv4Address


@dataclass
class CommitteeMember:
    node_id: IPv4Address
    sk: pke.SecretKey          # was: sk: bytes
    pk: bytes
    available: bool = True     # reachable and willing to answer

    def respond(self, input_ct: bytes, is_online, params: Params):
        """Decrypt I_i,j, check the two conditions, release the share or bottom."""
        plain = pke.decrypt(self.sk, input_ct)
        if plain is None:
            return None
        pref, alt, sigma, tag = parse_committee_input(plain, params)
        # Both conditions must hold: the preferred candidate really is offline
        # and the alternate really is online.  Otherwise the share is withheld.
        if is_online(pref) or not is_online(alt):
            return None
        return CommitteeResponse(sigma, tag, pref, alt)


def split_inputs(h1: bytes, params: Params) -> list[bytes]:
    """H^1_i of an intermediary is I_i = {I_i,j}: nu fixed-size ciphertexts."""
    return [h1[j * params.tau_I:(j + 1) * params.tau_I] for j in range(params.nu)]


def reconstruct_alternate(peeled: Peeled, responses, params: Params) -> bytes:
    """Steps 6 and 7: verify the returned shares, then reconstruct c^-_{i+1}."""
    good = []
    for r in responses:
        if verify(peeled.mac_key, r.tag, r.sigma):
            good.append(r.sigma)
    if len(good) < params.threshold:
        raise DropError(
            f"only {len(good)} of {params.threshold} required shares verified")
    return sharing.reconstruct(good[:params.threshold])


def patch_alternate(onion: Onion, c_minus: bytes, params: Params) -> Onion:
    """Swap only the ciphertext half of K^1_{i+1}; tau_{i+1} stays untouched."""
    if len(c_minus) != params.tau_c:
        raise DropError("reconstructed ciphertext has the wrong length")
    k1 = onion.K[0][:params.tag_len] + c_minus
    return Onion([k1] + onion.K[1:], onion.H, onion.M, onion.fwd, onion.bwd)


# ----------------------------------------------------------- replication ----
def parse_m_block(m: bytes, params: Params):
    """The p (H_h,j , k_h,j) pairs a replication node finds in its own M block."""
    out = []
    for j in range(params.p):
        blob = m[j * params.repl_slot_len:(j + 1) * params.repl_slot_len]
        header = StoredHeader.from_bytes(blob[:params.stored_header_len], params)
        out.append((header, blob[params.stored_header_len:]))
    return out


def replicate(peeled: Peeled, params: Params) -> list[tuple[int, Onion]]:
    """Re-wrap the single shared payload once per slot and forward each copy."""
    xi = peeled.fwd
    copies = []
    for header, k_h in parse_m_block(peeled.m, params):
        xi_j = enc(stream_key(k_h, N_REPL), N_REPL, xi)
        copies.append((header.first_hop,
                       header.to_onion(params, xi_j, rand(params.tau_bwd))))
    return copies
