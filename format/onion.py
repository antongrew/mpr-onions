"""O_i = (K_i, H_i, M_i, U_i) and its serialisation.

  K_i   a stack of exactly d sub-blocks; only K^1_i is meaningful at layer i
  H_i   a stack of exactly d sub-blocks; only H^1_i is meaningful at layer i
  M_i   a single fixed-size field, not a stack -- exactly one hop on any path
        ever reads it, so there is nothing to shift layer by layer
  U_i   split into fwd_i and bwd_i, each fixed-size

Every one of these has the same size at every hop of every onion, whatever the
message is doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address

from . import address
from .params import N_MFILL, Params
from .primitives import prg, rand


@dataclass
class Onion:
    K: list[bytes]
    H: list[bytes]
    M: bytes
    fwd: bytes
    bwd: bytes

    # ------------------------------------------------------------ views ----
    @property
    def tau(self) -> bytes:
        """The tag half of K^1_i."""
        return self.K[0][:16]

    def c(self, params: Params) -> bytes:
        """The public-key ciphertext half of K^1_i."""
        return self.K[0][params.tag_len:]

    def bundle(self, params: Params) -> bytes:
        """hdr_i := (K^2_i, ..., K^d_i, H_i, M_i), as received (still encrypted)."""
        return b"".join(self.K[1:]) + b"".join(self.H) + self.M

    def check_shape(self, params: Params) -> None:
        assert len(self.K) == params.d and len(self.H) == params.d
        assert all(len(k) == params.tau_K for k in self.K)
        assert all(len(h) == params.tau_H for h in self.H)
        assert len(self.M) == params.tau_M
        assert len(self.fwd) == params.tau_fwd
        assert len(self.bwd) == params.tau_bwd

    # -------------------------------------------------------- wire form ----
    def to_bytes(self) -> bytes:
        return b"".join(self.K) + b"".join(self.H) + self.M + self.fwd + self.bwd

    @staticmethod
    def from_bytes(data: bytes, params: Params) -> Onion:
        if len(data) != params.onion_len:
            raise ValueError(f"expected {params.onion_len} B, got {len(data)} B")
        off = 0
        K = []
        for _ in range(params.d):
            K.append(data[off:off + params.tau_K])
            off += params.tau_K
        H = []
        for _ in range(params.d):
            H.append(data[off:off + params.tau_H])
            off += params.tau_H
        M = data[off:off + params.tau_M]
        off += params.tau_M
        fwd = data[off:off + params.tau_fwd]
        off += params.tau_fwd
        return Onion(K, H, M, fwd, data[off:])

    def size(self) -> int:
        return len(self.to_bytes())


@dataclass
class StoredHeader:
    """A header in the compact form that travels inside another onion.

    Carried in a SURB slot and in the p replication slots of an M block.  Only
    the two sub-block stacks and a filler seed are stored: the header's own M
    block (and, for a SURB, its forward payload) are regenerated from that seed
    rather than shipped.  Without this, tau_M would be defined in terms of a
    quantity that contains tau_M.
    """

    first_hop: IPv4Address
    K: list[bytes]
    H: list[bytes]
    seed: bytes

    def to_bytes(self) -> bytes:
        return (address.to_bytes(self.first_hop)
                + b"".join(self.K) + b"".join(self.H) + self.seed)

    @staticmethod
    def from_bytes(data: bytes, params: Params) -> StoredHeader:
        if len(data) != params.stored_header_len:
            raise ValueError("bad stored header length")
        first = address.from_bytes(data[:params.node_id_len])
        off = params.node_id_len
        K = []
        for _ in range(params.d):
            K.append(data[off:off + params.tau_K])
            off += params.tau_K
        H = []
        for _ in range(params.d):
            H.append(data[off:off + params.tau_H])
            off += params.tau_H
        return StoredHeader(first, K, H, data[off:])

    def m_block(self, params: Params) -> bytes:
        """The M block this header's first hop must receive."""
        return prg(self.seed, N_MFILL, params.tau_M)

    def to_onion(self, params: Params, fwd: bytes, bwd: bytes | None = None) -> Onion:
        o = Onion(
            list(self.K), list(self.H), self.m_block(params), fwd,
            bwd if bwd is not None else rand(params.tau_bwd),
        )
        o.check_shape(params)
        return o
