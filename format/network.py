"""A small mix network to run onions through.

This is the surrounding system, not part of the packet format: it owns the node
registry, the fixed global committee G, and who happens to be online.  It also
runs the relay loop, so the format's own processing logic in node.py stays free
of scheduling concerns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv4Network

from . import address, pke
from . import node as nd
from .header import Hop
from .node import DropError
from .onion import Onion
from .params import ROLE_INTERMEDIARY, ROLE_RECIPIENT, Params


@dataclass
class Delivered:
    node_id: IPv4Address
    peeled: nd.Peeled


@dataclass
class RouteStats:
    hops: int = 0
    committee_rounds: int = 0
    detours: int = 0
    drops: list[str] = field(default_factory=list)
    deliveries: list[Delivered] = field(default_factory=list)


class Network:
    def __init__(self, n_nodes: int, params: Params, seed: int | None = None,
                 mix_range: IPv4Network = address.MIX_RANGE,
                 committee_range: IPv4Network = address.COMMITTEE_RANGE):
        params.validate()
        self.params = params
        self.rng = random.Random(seed)

        # Every party is identified by an IPv4 address; mix nodes and committee
        # members are allocated out of separate ranges purely for legibility.
        self.nodes: dict[IPv4Address, nd.Node] = {}
        for addr in address.allocate(mix_range, n_nodes):
            kp = pke.KeyPair.generate()
            self.nodes[addr] = nd.Node(addr, kp.sk, kp.pk)

        # G = (G_1, ..., G_nu): one committee for the entire network, a fixed
        # and publicly known system parameter -- not chosen per hop or per onion.
        self.committee: list[nd.CommitteeMember] = []
        for addr in address.allocate(committee_range, params.nu):
            kp = pke.KeyPair.generate()
            self.committee.append(nd.CommitteeMember(addr, kp.sk, kp.pk))

    # ------------------------------------------------------------ lookup ---
    @property
    def committee_pks(self) -> list[tuple[IPv4Address, bytes]]:
        return [(m.node_id, m.pk) for m in self.committee]

    def node(self, node_id) -> nd.Node:
        """Look a node up by address, accepting a dotted-quad string too."""
        return self.nodes[address.parse(node_id)]

    def pk(self, node_id) -> bytes:
        return self.node(node_id).pk

    def is_online(self, node_id) -> bool:
        n = self.nodes.get(address.parse(node_id))
        return n is not None and n.online

    def set_online(self, node_id, value: bool) -> None:
        self.node(node_id).online = value

    def all_online(self) -> None:
        for n in self.nodes.values():
            n.online = True
        for m in self.committee:
            m.available = True

    def set_committee_available(self, count: int) -> None:
        """Leave exactly `count` members of G reachable; the rest go silent."""
        for i, m in enumerate(self.committee):
            m.available = i < count

    # -------------------------------------------------------- path picking --
    def sample_path(self, length: int, dest=None,
                    dest_role: int = ROLE_RECIPIENT,
                    with_alternates: bool = True) -> list[Hop]:
        """A path of `length` hops: intermediaries plus a final destination."""
        dest = self.rng.choice(list(self.nodes)) if dest is None else address.parse(dest)
        pool = [i for i in self.nodes if i != dest]
        self.rng.shuffle(pool)
        chosen = pool[:length - 1] + [dest]
        used = set(chosen)
        spare = [i for i in self.nodes if i not in used]
        self.rng.shuffle(spare)

        hops = []
        for idx, nid in enumerate(chosen):
            last = idx == length - 1
            role = dest_role if last else ROLE_INTERMEDIARY
            alt_id = alt_pk = None
            if with_alternates and not last and spare:
                alt = spare.pop()
                alt_id, alt_pk = alt, self.pk(alt)
            hops.append(Hop(nid, self.pk(nid), role, alt_id, alt_pk))
        return hops

    # ------------------------------------------------------- the relay loop -
    def route(self, first_hop, onion: Onion, stats: RouteStats | None = None
              ) -> RouteStats:
        st = stats or RouteStats()
        queue = [(address.parse(first_hop), onion)]
        while queue:
            nid, o = queue.pop(0)
            if not self.is_online(nid):
                st.drops.append(f"node {nid} offline")
                continue
            try:
                peeled = self.nodes[nid].peel(o, self.params)
            except DropError as e:
                st.drops.append(f"node {nid}: {e}")
                continue
            st.hops += 1

            if peeled.role == ROLE_RECIPIENT:
                st.deliveries.append(Delivered(nid, peeled))
                continue
            if peeled.role != ROLE_INTERMEDIARY:      # replication
                for target, copy in nd.replicate(peeled, self.params):
                    queue.append((target, copy))
                continue

            nxt = peeled.next_id
            if nxt is not None and self.is_online(nxt):
                queue.append((nxt, peeled.next_onion()))
                continue

            # Preferred candidate unreachable: P_i cannot redirect on its own.
            st.committee_rounds += 1
            try:
                target, patched = self._failover(peeled)
            except DropError as e:
                st.drops.append(f"node {nid}: {e}")
                continue
            st.detours += 1
            queue.append((target, patched))
        return st

    def _failover(self, peeled: nd.Peeled) -> tuple[IPv4Address, Onion]:
        """Steps 5 to 7: ask G, verify the shares, reconstruct, patch, forward."""
        inputs = nd.split_inputs(peeled.h1, self.params)
        responses = []
        for member, ct in zip(self.committee, inputs, strict=True):
            # An unreachable member simply never answers; from P_i's side that
            # is indistinguishable from one that withheld its share.
            if not member.available:
                continue
            r = member.respond(ct, self.is_online, self.params)
            if r is not None:
                responses.append(r)
        if not responses:
            raise DropError("no committee member released a share")
        c_minus = nd.reconstruct_alternate(peeled, responses, self.params)
        alt = responses[0].alternate
        return alt, nd.patch_alternate(peeled.next_onion(), c_minus, self.params)