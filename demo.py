"""Walks through the four things an MPR onion can be doing.

    python demo.py
"""

from format import Network, Params, address, protocol
from format import header as hd
from format import payload as pl
from format import primitives as pr

P = Params()
net = Network(60, P, seed=2026)
first, last = min(net.nodes), max(net.nodes)
print(f"network: {len(net.nodes)} mix nodes at {first}-{last}, global committee G "
      f"of {P.nu} at {net.committee[0].node_id}-{net.committee[-1].node_id}")
print(f"onion size {P.onion_len} B\n")


def show(title):
    print(f"--- {title} " + "-" * (58 - len(title)))


# The format carries fixed-size content fields (mu bytes forward, beta back) and
# leaves their encoding to the application.  This demo is the application: it
# writes a two-byte length, the text, and zero fill.
def app_encode(text: bytes, size: int) -> bytes:
    return (len(text).to_bytes(2, "big") + text).ljust(size, b"\0")


def app_decode(field: bytes) -> bytes:
    return field[2:2 + int.from_bytes(field[:2], "big")]


def fwd(text):
    return app_encode(text, P.msg_max)


# 1 ---------------------------------------------------------------- unicast --
show("unicast")
hops = net.sample_path(4)
print("path:", " -> ".join(str(h.node_id) for h in hops))
sent = protocol.send_unicast(net, hops, fwd(b"the quick brown fox"))
st = net.route(sent.first_hop, sent.onion)
got = protocol.receive(st.deliveries[0].peeled, P)
print(f"delivered to {st.deliveries[0].node_id} after {st.hops} hops: "
      f"{app_decode(got.message)!r}\n")

# 2 ------------------------------------------------------------------ churn --
show("churn: the preferred candidate is offline")
hops = net.sample_path(5)
sent = protocol.send_unicast(net, hops, fwd(b"routed around a dead node"))
victim, standin = hops[2].node_id, hops[2].alt_id
net.set_online(victim, False)
print(f"node {victim} goes offline; its alternate is {standin}")
st = net.route(sent.first_hop, sent.onion)
print(f"committee rounds: {st.committee_rounds}, detours: {st.detours}")
print(f"delivered: {app_decode(protocol.receive(st.deliveries[0].peeled, P).message)!r}\n")
net.all_online()

# 3 ---------------------------------------------------------- SURB and reply --
show("anonymous reply")
hops = net.sample_path(4)
keys = [pr.rand(P.key_len) for _ in hops]
hops[-1].delta = hd.empty_delta(P)
surb, record = protocol.create_surb(net, back_to=hops[0].node_id, ell=4,
                                    k_bob=keys[-1])
built = hd.build(hops, P, net.committee_pks, keys=keys,
                 m_last=pr.rand(P.tau_M),
                 fwd_last=pl.pack(fwd(b"here is a reply block"), surb, P))
st = net.route(hops[0].node_id, built.onion)
bob = st.deliveries[0]
got = protocol.receive(bob.peeled, P)
print(f"Bob ({bob.node_id}) receives {app_decode(got.message)!r} plus a SURB")
print("Bob holds: the SURB header and k_Bob -- no reply-path key, no seed")
reply = protocol.reply(got.surb, bob.peeled.key,
                       app_encode(b"and here is my answer", P.bwd_ct_len), P)
st2 = net.route(reply.first_hop, reply.onion)
back = st2.deliveries[0].peeled
info = protocol.receive(back, P)
print(f"reply travels {st2.hops} hops, indistinguishable from a forward onion")
print(f"Alice matches I_surb: {info.identifier == record.identifier}")
print(f"Alice recovers: {app_decode(protocol.recover_reply(record, back, P))!r}\n")

# 4 -------------------------------------------------------------- multicast --
show("multicast")
recipients = ["10.0.0.11", "10.0.0.22", "10.0.0.33"]
sent = protocol.send_multicast(net, fwd(b"one payload, many recipients"),
                               recipients, outer_len=2, inner_len=3)
st = net.route(sent.first_hop, sent.onion)
wanted = [address.parse(r) for r in recipients]
print(f"g={len(wanted)} real recipients, p={P.p} slots, {st.hops} hops total")
for d in st.deliveries:
    r = protocol.receive(d.peeled, P)
    if d.node_id in wanted:
        print(f"  real  {str(d.node_id):<15} {r.kind:<10} {app_decode(r.message)!r}")
    else:
        print(f"  dummy {str(d.node_id):<15} {r.kind:<10} "
              f"<{len(r.message)} pseudorandom bytes>")
print("\nevery onion above was exactly", P.onion_len, "bytes at every hop.")
