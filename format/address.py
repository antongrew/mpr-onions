"""IPv4 addressing for the parties in the network.

Every party -- mix node, replication node, recipient, committee member -- is
identified by an IPv4 address, which is exactly the four bytes the format
already reserves for a next-hop field.  Addresses travel on the wire in network
byte order, so what sits inside c_i is a packed address and nothing else.

255.255.255.255 is reserved as the placeholder written into P^+_{i+1} when a hop
has no single next hop: a replication node, whose next hops live inside the p
inner headers in its own M block, and the final recipient, which has none.  It
is never assigned to a party, so a real address can never be mistaken for it.
"""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network

# The address a hop is given when it has no single successor to name.
PLACEHOLDER = IPv4Address("255.255.255.255")

# Default ranges used by the simulated network.  Mix nodes sit in RFC 1918
# space; the global committee G sits in TEST-NET-1, so the two are obvious
# apart at a glance in logs and test output.
MIX_RANGE = IPv4Network("10.0.0.0/8")
COMMITTEE_RANGE = IPv4Network("192.0.2.0/24")

ADDR_LEN = 4


def parse(value) -> IPv4Address:
    """Accept an IPv4Address, a dotted-quad string, an int, or packed bytes."""
    if isinstance(value, IPv4Address):
        return value
    if isinstance(value, (bytes, bytearray)):
        if len(value) != ADDR_LEN:
            raise ValueError(f"an IPv4 address is {ADDR_LEN} bytes, got {len(value)}")
        return IPv4Address(bytes(value))
    return IPv4Address(value)


def to_bytes(value) -> bytes:
    """Pack an address into its four wire bytes, network byte order."""
    return parse(value).packed


def from_bytes(raw: bytes) -> IPv4Address:
    return IPv4Address(bytes(raw))


def is_placeholder(value) -> bool:
    return parse(value) == PLACEHOLDER


def allocate(network: IPv4Network, count: int, start: int = 1) -> list[IPv4Address]:
    """`count` consecutive usable addresses from `network`, skipping the base."""
    base = int(network.network_address)
    last = int(network.broadcast_address)
    out = []
    n = base + start
    while len(out) < count:
        if n >= last:
            raise ValueError(f"{network} cannot hold {count} addresses")
        addr = IPv4Address(n)
        if addr != PLACEHOLDER:
            out.append(addr)
        n += 1
    return out
