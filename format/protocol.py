"""Sender- and recipient-side constructions built on top of the packet format.

Covers the four things an MPR onion can be doing: an ordinary forward message,
a forward message carrying a SURB, a reply written into a SURB, and a multicast
message replicated at a sender-designated node.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address

from .. import address, ae
from .. import header as hd
from . import payload as pl
from .network import Network
from .node import Peeled
from .onion import Onion, StoredHeader
from .params import (
    DELTA_MULTICAST,
    DELTA_NONE,
    DELTA_SURB,
    N_FFILL,
    N_FWD,
    N_REPL,
    ROLE_RECIPIENT,
    ROLE_REPLICATION,
    Params,
)
from .primitives import dec, enc, kdf, keystream, prg, rand, stream_key, xor


# ------------------------------------------------------------- key chains ---
def path_keys(secret: bytes, ell: int, params: Params) -> list[bytes]:
    """k_i = KDF(secret, i) -- one retained secret regenerates a whole path.

    This is what omsk_surb does for a reply path (Section 4.3.1), and what the
    per-slot seed does for a multicast recipient's own inner path.
    """
    return [kdf(secret, b"HOP" + i.to_bytes(2, "big"), params.key_len)
            for i in range(ell)]


def accumulated_fwd_mask(secret: bytes, ell: int, params: Params) -> bytes:
    """The XOR of every hop's forward-payload keystream along a derived path."""
    acc = bytes(params.tau_fwd)
    for k in path_keys(secret, ell, params):
        ske, _ = hd.subkeys(k, params)
        acc = xor(acc, keystream(ske, N_FWD, params.tau_fwd))
    return acc


# ----------------------------------------------------------------- delta ----
def delta_surb(identifier: bytes, params: Params) -> bytes:
    return (bytes([DELTA_SURB]) + identifier).ljust(params.delta_len, b"\x00")


def delta_multicast(seed: bytes, k_h: bytes, slot_seed: bytes, ell: int,
                    params: Params) -> bytes:
    """S_mc, the slot's replication key, its path seed, and its path length."""
    return (bytes([DELTA_MULTICAST]) + seed + k_h + slot_seed
            + bytes([ell])).ljust(params.delta_len, b"\x00")


def parse_delta(h1: bytes, params: Params) -> dict:
    kind = h1[0]
    n = params.key_len
    if kind == DELTA_SURB:
        return {"kind": kind, "identifier": h1[1:1 + n]}
    if kind == DELTA_MULTICAST:
        return {"kind": kind, "seed": h1[1:1 + n], "k_h": h1[1 + n:1 + 2 * n],
                "slot_seed": h1[1 + 2 * n:1 + 3 * n], "ell": h1[1 + 3 * n]}
    return {"kind": DELTA_NONE}


# --------------------------------------------------------------- unicast ----
@dataclass
class Sent:
    first_hop: IPv4Address
    onion: Onion
    keys: list[bytes]


def send_unicast(net: Network, hops, content: bytes, surb: bytes | None = None) -> Sent:
    """`content` is the mu-byte forward content field, encoded by the application."""
    if hops[-1].delta is None:
        hops[-1].delta = hd.empty_delta(net.params)
    built = hd.build(
        hops, net.params, net.committee_pks,
        m_last=rand(net.params.tau_M),           # filler: no replication here
        fwd_last=pl.pack(content, surb, net.params),
    )
    return Sent(hops[0].node_id, built.onion, built.keys)


# ------------------------------------------------------------------ SURB ----
@dataclass
class SurbRecord:
    """What the SURB's creator retains: one secret, one identifier, one key."""

    identifier: bytes
    omsk: bytes
    k_bob: bytes
    ell: int


def create_surb(net: Network, back_to, ell: int, k_bob: bytes
                ) -> tuple[bytes, SurbRecord]:
    """Alice builds a reply header addressed back to herself.

    The replier receives the header and nothing else.  Every hop key on the
    reply path is KDF(omsk_surb, i), so Alice regenerates all of them later from
    one retained secret; I_surb rides in her own last-hop Delta so she can
    recognise which SURB a reply belongs to without any coordination.
    """
    params = net.params
    omsk = rand(params.key_len)
    identifier = rand(params.key_len)
    hops = net.sample_path(ell, dest=back_to, dest_role=ROLE_RECIPIENT)
    hops[-1].delta = delta_surb(identifier, params)
    built = hd.build(hops, params, net.committee_pks,
                     keys=path_keys(omsk, ell, params), seed=rand(params.key_len))
    return built.stored.to_bytes(), SurbRecord(identifier, omsk, k_bob, ell)


def reply(surb_bytes: bytes, k_bob: bytes, content: bytes, params: Params) -> Sent:
    """Bob's entire contribution: one AE.Enc under the key he already holds.

    `content` is the beta-byte backward content field, encoded by the application.

    The forward half and the M block are *not* freshly randomised by Bob -- they
    are the exact filler the SURB's creator committed to in every tau_i, and are
    regenerated from the seed shipped with the header.  A reply that reinvented
    them would fail tau verification at the very first hop.
    """
    stored = StoredHeader.from_bytes(surb_bytes, params)
    onion = stored.to_onion(
        params,
        fwd=prg(stored.seed, N_FFILL, params.tau_fwd),
        bwd=ae.encrypt(k_bob, content, params),
    )
    return Sent(stored.first_hop, onion, [])


def recover_reply(record: SurbRecord, peeled: Peeled, params: Params) -> bytes | None:
    """Undo every hop's AE.Unwrap in reverse, then AE.Dec and check the tag."""
    bwd = peeled.bwd
    for k in reversed(path_keys(record.omsk, record.ell, params)):
        bwd = ae.wrap(k, bwd, params)
    return ae.decrypt(record.k_bob, bwd, params)


# -------------------------------------------------------------- multicast ---
@dataclass
class MulticastSent(Sent):
    seed: bytes
    slots: list[dict]


def send_multicast(net: Network, content: bytes, recipients: list,
                   outer_len: int, inner_len: int,
                   surb: bytes | None = None) -> MulticastSent:
    """Encrypt once, replicate at one sender-designated node.

    The p inner headers and their replication keys go into the replication
    hop's own M block; the single shared ciphertext xi travels as that hop's
    ordinary forward payload.  Slots beyond the g real recipients address
    dummies and are indistinguishable from real ones.
    """
    params = net.params
    if len(recipients) > params.p:
        raise ValueError(f"at most p = {params.p} recipients per multicast message")

    seed = rand(params.key_len)
    xi = pl.mc_encrypt(seed, pl.pack(content, surb, params), params)

    # Dummy slots address parties that are not in the group.  Pointing one at a
    # real recipient would send that node a second copy it cannot read, which is
    # wasted work and a needless hint that the two slots belong together.
    wanted = [address.parse(r) for r in recipients]
    spare = [n for n in net.nodes if n not in set(wanted)]
    if len(spare) < params.p - len(wanted):
        raise ValueError("not enough nodes to address the dummy slots")
    net.rng.shuffle(spare)

    slots, blob = [], []
    for j in range(params.p):
        real = j < len(wanted)
        dest = wanted[j] if real else spare.pop()
        slot_seed = rand(params.key_len)
        k_h = rand(params.key_len)
        inner = net.sample_path(inner_len, dest=dest, dest_role=ROLE_RECIPIENT)
        inner[-1].delta = (delta_multicast(seed, k_h, slot_seed, inner_len, params)
                           if real else hd.empty_delta(params))
        # The replication node will compute exactly this value from xi and k_h,j.
        xi_j = enc(stream_key(k_h, N_REPL), N_REPL, xi)
        built = hd.build(inner, params, net.committee_pks,
                         keys=path_keys(slot_seed, inner_len, params),
                         seed=rand(params.key_len), fwd_first=xi_j)
        blob.append(built.stored.to_bytes() + k_h)
        slots.append({"real": real, "dest": dest, "slot_seed": slot_seed,
                      "k_h": k_h, "first_hop": inner[0].node_id})

    outer = net.sample_path(outer_len, dest_role=ROLE_REPLICATION)
    built = hd.build(outer, params, net.committee_pks,
                     m_last=b"".join(blob), fwd_last=xi)
    return MulticastSent(outer[0].node_id, built.onion, built.keys, seed, slots)


# ------------------------------------------------------ recipient recovery --
@dataclass
class Received:
    kind: str                    # "unicast" | "multicast" | "reply"
    message: bytes | None = None  # the mu-byte content field, as the sender wrote it
    surb: bytes | None = None     # the SURB slot; a real SURB only if the sender put one there
    identifier: bytes | None = None


def receive(peeled: Peeled, params: Params) -> Received:
    """What a recipient does after its own K^1/H^1 are open, before touching U.

    Unicast and multicast are told apart locally, with no explicit type field:
    the final routing information says which processing mode applies.
    """
    info = parse_delta(peeled.h1, params)

    if info["kind"] == DELTA_SURB:
        # A reply arriving back at the SURB's creator.
        return Received("reply", identifier=info["identifier"])

    if info["kind"] == DELTA_MULTICAST:
        ell = info["ell"]
        # Undo this slot's own path mask, then the replication node's re-wrap,
        # then the one remaining key-tree layer.
        xi = xor(peeled.fwd, accumulated_fwd_mask(info["slot_seed"], ell, params))
        xi = dec(stream_key(info["k_h"], N_REPL), N_REPL, xi)
        msg, surb = pl.unpack(pl.mc_decrypt(info["seed"], xi, params), params)
        return Received("multicast", msg, surb)

    # A dummy multicast slot lands here too: it peels its layer like any other
    # recipient and obtains mu pseudorandom bytes.  Nothing in the format tells
    # it, or anyone watching, that it was a dummy; recognising a real message
    # is the application's job.
    msg, surb = pl.unpack(peeled.fwd, params)
    return Received("unicast", msg, surb)
