"""MPR Onions -- Multi Poly Reply Onions.

A churn-resilient, multicast-capable, reply-supporting onion routing format,
implemented as described in the MPR Onions paper.

    from format import Params, Network, protocol

    net = Network(40, Params())
    hops = net.sample_path(4)
    content = b"hello".ljust(net.params.msg_max, b"\0")   # encoding is the app's job
    sent = protocol.send_unicast(net, hops, content)
    stats = net.route(sent.first_hop, sent.onion)
    print(protocol.receive(stats.deliveries[0].peeled, net.params).message.rstrip(b"\0"))
"""

from . import address, ae, header, payload, pke, primitives, protocol, sharing
from .header import Hop
from .network import Network, RouteStats
from .node import CommitteeMember, DropError, Node, Peeled
from .onion import Onion, StoredHeader
from .params import (
    DEFAULT,
    ROLE_INTERMEDIARY,
    ROLE_RECIPIENT,
    ROLE_REPLICATION,
    Params,
)

__all__ = [
    "DEFAULT", "Params", "Hop", "Network", "RouteStats", "Node", "Peeled",
    "CommitteeMember", "DropError", "Onion", "StoredHeader",
    "ROLE_INTERMEDIARY", "ROLE_REPLICATION", "ROLE_RECIPIENT",
    "ae", "address", "header", "payload", "pke", "primitives", "protocol",
    "sharing",
]

__version__ = "1.0.0"
