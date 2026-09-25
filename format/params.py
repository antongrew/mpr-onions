"""Protocol parameters and derived block sizes for the MPR Onion format.

Every size in the format is a public, network-wide constant.  Nothing here
depends on the actual path length, the number of real multicast recipients, or
on which functionality a given onion happens to be using -- that is the "one
shape, whatever the message is doing" decision from Section 3.2 of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------- roles ----
ROLE_INTERMEDIARY = 0
ROLE_REPLICATION = 1
ROLE_RECIPIENT = 2

ROLE_NAMES = {
    ROLE_INTERMEDIARY: "intermediary",
    ROLE_REPLICATION: "replication",
    ROLE_RECIPIENT: "recipient",
}

# ------------------------------------------------------- delta tag bytes ----
DELTA_NONE = 0      # ordinary forward message -- rest is pseudorandom filler
DELTA_SURB = 1      # last hop of a reply path: carries I_surb
DELTA_MULTICAST = 2  # multicast recipient: carries S_mc and k_h,j

# ------------------------------------------------------------- nonces -------
# Distinct per-field nonces, so that one k^SKE_i can safely encrypt every
# symmetric field at a hop (Section 4.1.1 / 4.5.1).
N_HEADER = b"HEADER"   # the whole header bundle (K^2..K^d, H, M) as one unit
N_FWD = b"FWD"         # forward half of U
N_BWD = b"BWD"         # backward half of U
N_KFILL = b"KFILL"     # vacated K slot filler
N_HFILL = b"HFILL"     # vacated H slot filler
N_REPL = b"REPL"       # replication node's re-wrap of the shared payload
N_MFILL = b"MFILL"     # seed-derived M filler of a stored header
N_FFILL = b"FFILL"     # seed-derived fwd filler of a stored header

# --------------------------------------------------------- KDF labels -------
L_SKE = b"SKE"
L_MAC = b"MAC"
L_PRF = b"PRF"


@dataclass(frozen=True)
class Params:
    """All tunable parameters, plus every size derived from them."""

    d: int = 5            # fixed number of K/H sub-blocks (max path length)
    nu: int = 5           # size of the global committee G
    alpha: float = 0.6    # reconstruction threshold, as a fraction of nu
    p: int = 4            # replication factor (recipient slots per multicast)

    key_len: int = 16     # kappa: hop keys, seeds, identifiers
    tag_len: int = 16     # MAC tag length
    node_id_len: int = 4  # an IPv4 address identifies a party

    msg_max: int = 1024      # mu: the forward content field, exactly this many bytes
    bwd_ct_len: int = 1024   # beta: the backward content field (ciphertext half of bwd)

    # ------------------------------------------------------ thresholds ----
    @property
    def threshold(self) -> int:
        """alpha * nu, rounded up: shares needed to reconstruct c^-."""
        return max(1, math.ceil(self.alpha * self.nu))

    @property
    def f_max(self) -> int:
        """Largest f satisfying f < alpha*nu <= nu - f."""
        return min(self.threshold - 1, self.nu - self.threshold)

    # ---------------------------------------------------- PKE / K block ----
    @property
    def pke_overhead(self) -> int:
        return 32 + 16  # ephemeral X25519 public key + Poly1305 tag (pke.OVERHEAD)

    @property
    def c_plain_len(self) -> int:
        """Plaintext of c_i: (k_i, role(P_i), P^+_{i+1})."""
        return self.key_len + 1 + self.node_id_len

    @property
    def tau_c(self) -> int:
        return self.pke_overhead + self.c_plain_len

    @property
    def tau_K(self) -> int:
        """Size of one K sub-block: a tag and a public-key ciphertext."""
        return self.tag_len + self.tau_c

    # -------------------------------------------- committee / H block ----
    @property
    def share_len(self) -> int:
        """One Shamir share of c^-: an x-coordinate byte plus the share body."""
        return 1 + self.tau_c

    @property
    def input_plain_len(self) -> int:
        """Plaintext of I_i,j: (P^+, P^-, sigma_i,j, T_i,j)."""
        return 2 * self.node_id_len + self.share_len + self.tag_len

    @property
    def tau_I(self) -> int:
        return self.pke_overhead + self.input_plain_len

    @property
    def tau_H(self) -> int:
        """Room for exactly one committee input set I_i (or a padded Delta)."""
        return self.nu * self.tau_I

    @property
    def delta_len(self) -> int:
        """Content of Delta before it is padded out to tau_H.

        Widest case is a multicast recipient: type | S_mc | k_h,j | s_j | l'.
        """
        return 2 + 3 * self.key_len

    # ------------------------------------------- stored headers / M block --
    @property
    def stack_len(self) -> int:
        """A full K stack plus a full H stack."""
        return self.d * self.tau_K + self.d * self.tau_H

    @property
    def stored_header_len(self) -> int:
        """A header in transportable form: both stacks plus a filler seed.

        The M and fwd blocks of a stored header are *not* carried literally;
        they are regenerated from the seed (see header.py).  This is what
        breaks the self-reference in tau_M -- see the README.
        """
        return self.node_id_len + self.stack_len + self.key_len

    @property
    def surb_len(self) -> int:
        return self.stored_header_len

    @property
    def repl_slot_len(self) -> int:
        """One (H_h,j , k_h,j) pair inside the M block."""
        return self.stored_header_len + self.key_len

    @property
    def tau_M(self) -> int:
        return self.p * self.repl_slot_len

    # ------------------------------------------------------- U block ------
    @property
    def tau_fwd(self) -> int:
        """SURB slot | content field.

        No length field and no flag: the format carries a fixed mu-byte content
        field and leaves its encoding (message length, whether the SURB slot is
        in use) to the application.  The only padding the format performs is
        structural.
        """
        return self.surb_len + self.msg_max

    @property
    def tau_bwd(self) -> int:
        return self.bwd_ct_len + self.tag_len

    @property
    def tau_U(self) -> int:
        return self.tau_fwd + self.tau_bwd

    @property
    def reply_max(self) -> int:
        return self.bwd_ct_len

    # ------------------------------------------- header bundle layout -----
    # hdr_i := (K^2_i, ..., K^d_i, H_i, M_i), encrypted as a single unit.
    @property
    def off_K(self) -> int:
        return 0

    @property
    def off_H(self) -> int:
        return (self.d - 1) * self.tau_K

    @property
    def off_M(self) -> int:
        return self.off_H + self.d * self.tau_H

    @property
    def bundle_len(self) -> int:
        return self.off_M + self.tau_M

    @property
    def onion_len(self) -> int:
        return self.d * self.tau_K + self.d * self.tau_H + self.tau_M + self.tau_U

    # ------------------------------------------------------- reporting ----
    def size_table(self) -> str:
        rows = [
            ("d (K/H stack depth)", self.d),
            ("nu (committee size)", self.nu),
            ("alpha * nu (threshold)", self.threshold),
            ("f (tolerated corruptions)", self.f_max),
            ("p (replication factor)", self.p),
            ("", ""),
            ("tau_K (one K sub-block)", self.tau_K),
            ("tau_H (one H sub-block)", self.tau_H),
            ("K block (d sub-blocks)", self.d * self.tau_K),
            ("H block (d sub-blocks)", self.d * self.tau_H),
            ("tau_M", self.tau_M),
            ("tau_fwd", self.tau_fwd),
            ("tau_bwd", self.tau_bwd),
            ("tau_U", self.tau_U),
            ("", ""),
            ("stored header / SURB", self.surb_len),
            ("forward content field (mu)", self.msg_max),
            ("backward content field (beta)", self.reply_max),
            ("TOTAL onion size", self.onion_len),
        ]
        out = []
        for name, val in rows:
            if name == "":
                out.append("")
            else:
                out.append(f"  {name:<32} {val:>10}")
        return "\n".join(out)

    def validate(self) -> None:
        if self.d < 2:
            raise ValueError("d must be at least 2")
        if not (self.f_max < self.threshold <= self.nu - self.f_max):
            raise ValueError("committee threshold violates f < alpha*nu <= nu - f")
        if self.p < 1:
            raise ValueError("p must be at least 1")
        if self.node_id_len != 4:
            raise ValueError("parties are identified by IPv4 addresses, so "
                             "node_id_len must be 4")
        if self.msg_max < 1:
            raise ValueError("msg_max must be positive")


DEFAULT = Params()
