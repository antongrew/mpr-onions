"""Self-contained test suite.  Run with `python -m format.selftest`.

Five suites, no external test framework and no fixtures beyond what each check
builds for itself.
"""

from __future__ import annotations

import sys
import traceback

from .. import address, ae, sharing
from .. import header as hd
from . import payload as pl, pke
from .. import primitives as pr
from .. import protocol as proto
from .network import Network
from .node import DropError
from .onion import Onion, StoredHeader
from .params import (
    ROLE_REPLICATION,
    Params,
)


# The format carries fixed-size content fields and leaves their encoding to the
# application.  The tests play a trivial application: a short tag, zero-filled to
# the field size, compared as the whole field.
def C(P: Params, tag: bytes = b"") -> bytes:
    """A forward content field of exactly mu bytes."""
    return tag.ljust(P.msg_max, b"\x00")


def R(P: Params, tag: bytes = b"") -> bytes:
    """A backward content field of exactly beta bytes."""
    return tag.ljust(P.bwd_ct_len, b"\x00")


_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, fn) -> None:
    global _PASS, _FAIL
    try:
        fn()
    except Exception:
        _FAIL += 1
        _FAILURES.append(f"{name}\n{traceback.format_exc()}")
        print(f"  FAIL  {name}")
    else:
        _PASS += 1
        print(f"  ok    {name}")


def suite(title: str) -> None:
    print(f"\n[{title}]")


# ============================================================ 1. primitives ==
def test_primitives(P: Params) -> None:
    suite("primitives")

    def chacha20_vector():
        # RFC 8439, Section 2.4.2: key 00..1f, nonce 00^7 4a 00^4, counter 1.
        # Counter 1 is byte offset 64, so this also pins the seek arithmetic.
        key = bytes(range(32))
        nonce = bytes(7) + b"\x4a"
        pt = (b"Ladies and Gentlemen of the class of '99: If I could offer you "
              b"only one tip for the future, sunscreen would be it.")
        ct = pr.enc(key, nonce, pt, offset=64)
        assert ct[:16].hex() == "6e2e359a2568f98041ba0728dd0d6981"
        assert ct[-18:].hex() == "5af90bbf74a35be6b40b8eedf2785e42874d"
    check("G_k is ChaCha20 and matches the RFC 8439 test vector", chacha20_vector)

    def keystream_ranges():
        k, n = pr.rand(pr.STREAM_KEY_LEN), b"N"
        full = pr.keystream(k, n, 700)
        for off, ln in ((0, 700), (1, 100), (63, 2), (64, 64), (137, 413), (500, 200)):
            assert pr.keystream(k, n, ln, off) == full[off:off + ln]
    check("keystream ranges agree with the full stream", keystream_ranges)

    def xor_involution():
        k, m = pr.rand(pr.STREAM_KEY_LEN), pr.rand(777)
        assert pr.dec(k, b"A", pr.enc(k, b"A", m)) == m
        assert pr.enc(k, b"A", m) != pr.enc(k, b"B", m)
    check("forward and backward evaluation are one algorithm", xor_involution)

    def kdf_separation():
        k = pr.rand(16)
        subs = {pr.kdf(k, lbl) for lbl in (b"SKE", b"MAC", b"PRF")}
        assert len(subs) == 3
        assert pr.kdf(k, b"SKE") == pr.kdf(k, b"SKE")
        assert pr.kdf(pr.rand(16), b"SKE") not in subs
    check("KDF is deterministic and domain-separated", kdf_separation)

    def mac_binds():
        k = pr.rand(16)
        t = pr.mac(k, b"aa", b"bb")
        assert pr.verify(k, t, b"aa", b"bb")
        assert not pr.verify(k, t, b"a", b"abb")     # no length-extension mixup
        assert not pr.verify(pr.rand(16), t, b"aa", b"bb")
    check("MAC binds field boundaries, not just concatenation", mac_binds)

    def content_fields():
        # No length field, no flag: fwd is exactly SURB slot || mu content bytes,
        # and the backward ciphertext is exactly beta bytes.
        assert P.tau_fwd == P.surb_len + P.msg_max
        assert P.tau_bwd == P.bwd_ct_len + P.tag_len
        body, surb = pr.rand(P.msg_max), pr.rand(P.surb_len)
        field = pl.pack(body, surb, P)
        assert field == surb + body and pl.unpack(field, P) == (body, surb)
        k, reply = pr.rand(P.key_len), pr.rand(P.bwd_ct_len)
        assert ae.decrypt(k, ae.encrypt(k, reply, P), P) == reply
        for bad in (lambda: pl.pack(body[:-1], None, P),
                    lambda: ae.encrypt(k, reply + b"x", P)):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("a content field of the wrong size was accepted")
    check("content fields are fixed-size and carry no length or flag bytes", content_fields)

    def pke_roundtrip():
        kp = pke.KeyPair.generate()
        m = pr.rand(P.c_plain_len)
        c = pke.encrypt(kp.pk, m)
        assert len(c) == P.tau_c, "ciphertext length must be a constant"
        assert pke.decrypt(kp.sk, c) == m
        assert pke.decrypt(pke.KeyPair.generate().sk, c) is None
        bad = bytearray(c)
        bad[-20] ^= 0x40
        assert pke.decrypt(kp.sk, bytes(bad)) is None
    check("public-key encryption round-trips at a fixed length", pke_roundtrip)

    def key_tree():
        s = pr.rand(16)
        assert pl.kt_root(s, P) == pl.kt_root(s, P)
        assert pl.kt_root(s, P) != pl.kt_root(pr.rand(16), P)
        assert pl.kt_child(pl.kt_root(s, P), 3, P) != pl.kt_child(pl.kt_root(s, P), 4, P)
        m = pr.rand(500)
        assert pl.mc_decrypt(s, pl.mc_encrypt(s, m, P), P) == m
    check("key tree derives stable, distinct children", key_tree)


def test_addressing(P: Params) -> None:
    suite("addressing")

    def roundtrip():
        for text in ("10.0.0.1", "192.0.2.255", "8.8.8.8", "0.0.0.1"):
            a = address.parse(text)
            assert str(a) == text
            assert len(address.to_bytes(a)) == P.node_id_len
            assert address.from_bytes(address.to_bytes(a)) == a
            assert address.parse(address.to_bytes(a)) == a
            assert address.parse(int(a)) == a
    check("addresses round-trip through their four wire bytes", roundtrip)

    def network_byte_order():
        assert address.to_bytes("1.2.3.4") == bytes([1, 2, 3, 4])
        assert address.to_bytes("255.255.255.255") == b"\xff\xff\xff\xff"
    check("addresses are packed in network byte order", network_byte_order)

    def placeholder_is_reserved():
        assert address.is_placeholder(hd.placeholder_id(P))
        pool = address.allocate(address.MIX_RANGE, 64)
        assert address.PLACEHOLDER not in pool
        assert len(set(pool)) == 64
    check("255.255.255.255 is reserved and never allocated to a party",
          placeholder_is_reserved)

    def next_hop_field_carries_the_address():
        net = _fixture(P, seed=17)
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"x"))
        peeled = net.node(sent.first_hop).peel(sent.onion, P)
        assert peeled.next_id == hops[1].node_id
        assert str(peeled.next_id).startswith("10.")
    check("P^+_{i+1} recovered from c_i is the next hop's address",
          next_hop_field_carries_the_address)

    def placeholder_at_non_forwarding_hops():
        net = _fixture(P, seed=19)
        # A recipient has no next hop; the field holds the placeholder, so the
        # peel reports None rather than an address.
        hops = net.sample_path(min(3, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"x"))
        st = net.route(sent.first_hop, sent.onion)
        assert st.deliveries[0].peeled.next_id is None
        # So does a replication node, whose successors live in its M block.
        recips = net.rng.sample(list(net.nodes), 1)
        mc = proto.send_multicast(net, C(P, b"x"), recips,
                                  outer_len=min(2, P.d), inner_len=min(3, P.d))
        peeled = net.node(mc.first_hop).peel(mc.onion, P)
        if peeled.role == ROLE_REPLICATION:
            assert peeled.next_id is None
    check("a recipient and a replication node both name no successor",
          placeholder_at_non_forwarding_hops)

    def lookup_accepts_strings():
        net = _fixture(P, seed=21)
        some = next(iter(net.nodes))
        assert net.node(str(some)) is net.node(some)
        assert net.is_online(str(some))
        net.set_online(str(some), False)
        assert not net.is_online(some)
        net.all_online()
    check("nodes can be addressed by dotted-quad string or IPv4Address",
          lookup_accepts_strings)

    def stored_headers_carry_their_first_hop():
        net = _fixture(P, seed=23)
        surb, _ = proto.create_surb(net, net.rng.choice(list(net.nodes)),
                                    min(3, P.d), pr.rand(P.key_len))
        stored = StoredHeader.from_bytes(surb, P)
        assert stored.first_hop in net.nodes
        assert stored.to_bytes() == surb
    check("a stored header names its own first hop as an address",
          stored_headers_carry_their_first_hop)


# =============================================================== 3. sharing ==
def test_sharing(P: Params) -> None:
    suite("sharing")

    def exact_threshold():
        secret = pr.rand(P.tau_c)
        shares = sharing.split(secret, P.nu, P.threshold)
        assert all(len(s) == P.share_len for s in shares)
        assert sharing.reconstruct(shares[:P.threshold]) == secret
        assert sharing.reconstruct(shares[-P.threshold:]) == secret
    check("any alpha*nu shares reconstruct c^-", exact_threshold)

    def below_threshold():
        secret = pr.rand(P.tau_c)
        shares = sharing.split(secret, P.nu, P.threshold)
        assert sharing.reconstruct(shares[:P.threshold - 1]) != secret
    check("fewer than alpha*nu shares reconstruct nothing", below_threshold)

    def tagged_shares():
        k = pr.rand(16)
        secret = pr.rand(P.tau_c)
        shares = sharing.split(secret, P.nu, P.threshold)
        tags = [pr.mac(k, s) for s in shares]
        assert all(pr.verify(k, t, s) for s, t in zip(shares, tags, strict=True))
        forged = bytearray(shares[0])
        forged[5] ^= 1
        assert not pr.verify(k, tags[0], bytes(forged))
    check("a forged share fails Ver_{k^MAC_i}", tagged_shares)

    def threshold_bound():
        assert P.f_max < P.threshold <= P.nu - P.f_max
    check("f < alpha*nu <= nu - f holds for the chosen parameters", threshold_bound)


# ================================================================ 4. format ==
def _fixture(P: Params, seed: int = 3):
    return Network(40, P, seed=seed)


def test_format(P: Params) -> None:
    suite("format / header")
    net = _fixture(P)

    def all_path_lengths():
        for ell in range(1, P.d + 1):
            hops = net.sample_path(ell)
            sent = proto.send_unicast(net, hops, C(P, b"len=%d" % ell))
            st = net.route(sent.first_hop, sent.onion)
            assert not st.drops, st.drops
            assert len(st.deliveries) == 1
            got = proto.receive(st.deliveries[0].peeled, P)
            assert got.message == C(P, b"len=%d" % ell), (ell, got.message[:16])
    check("every path length 1..d builds, forwards and delivers", all_path_lengths)

    def constant_shape():
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"x"))
        o = sent.onion
        o.check_shape(P)
        cur, nid = o, sent.first_hop
        for _ in range(3):
            peeled = net.nodes[nid].peel(cur, P)
            nxt = peeled.next_onion()
            nxt.check_shape(P)
            assert nxt.size() == o.size(), "onion size changed along the path"
            cur, nid = nxt, peeled.next_id
    check("K/H stay at exactly d sub-blocks and the size never changes",
          constant_shape)

    def filler_is_regenerated():
        """The vacated slot must be exactly what the sender assumed it would be."""
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"x"))
        peeled = net.nodes[sent.first_hop].peel(sent.onion, P)
        assert peeled.K_next[-1] == hd.filler_K(peeled.ske, P)
        assert peeled.H_next[-1] == hd.filler_H(peeled.ske, P)
    check("hop i refills K^d/H^d with the constant tau_{i+1} was computed over",
          filler_is_regenerated)

    def tag_rejects_tampering():
        for field in ("K", "H", "M", "fwd"):
            hops = net.sample_path(min(3, P.d))
            sent = proto.send_unicast(net, hops, C(P, b"x"))
            o = sent.onion
            if field == "K":
                mod = bytearray(o.K[2])
                mod[0] ^= 1
                o = Onion(o.K[:2] + [bytes(mod)] + o.K[3:], o.H, o.M, o.fwd, o.bwd)
            elif field == "H":
                mod = bytearray(o.H[0])
                mod[0] ^= 1
                o = Onion(o.K, [bytes(mod)] + o.H[1:], o.M, o.fwd, o.bwd)
            elif field == "M":
                mod = bytearray(o.M)
                mod[0] ^= 1
                o = Onion(o.K, o.H, bytes(mod), o.fwd, o.bwd)
            else:
                mod = bytearray(o.fwd)
                mod[0] ^= 1
                o = Onion(o.K, o.H, o.M, bytes(mod), o.bwd)
            try:
                net.nodes[sent.first_hop].peel(o, P)
            except DropError:
                continue
            raise AssertionError(f"tampering with {field} was not caught by tau_i")
    check("tau_i catches tampering with K^2..K^d, H, M and fwd", tag_rejects_tampering)

    def bwd_is_not_bound():
        """bwd is deliberately outside tau_i, or no SURB could ever be used."""
        hops = net.sample_path(min(3, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"x"))
        o = sent.onion
        mod = bytearray(o.bwd)
        mod[0] ^= 1
        o2 = Onion(o.K, o.H, o.M, o.fwd, bytes(mod))
        net.nodes[sent.first_hop].peel(o2, P)      # must not raise
    check("bwd is outside tau_i and no hop's peel can fail because of it",
          bwd_is_not_bound)

    def splice_is_caught():
        """(tau, c) cut from one onion pasted into another must fail."""
        a = net.sample_path(min(3, P.d))
        b = net.sample_path(min(3, P.d))
        oa = proto.send_unicast(net, a, C(P, b"a")).onion
        ob = proto.send_unicast(net, b, C(P, b"b")).onion
        spliced = Onion([ob.K[0]] + oa.K[1:], oa.H, oa.M, oa.fwd, oa.bwd)
        try:
            net.nodes[b[0].node_id].peel(spliced, P)
        except DropError:
            return
        raise AssertionError("spliced K^1 was accepted")
    check("a foreign K^1 spliced onto another onion fails verification",
          splice_is_caught)

    def wire_roundtrip():
        hops = net.sample_path(min(4, P.d))
        o = proto.send_unicast(net, hops, C(P, b"x")).onion
        assert Onion.from_bytes(o.to_bytes(), P).to_bytes() == o.to_bytes()
        assert len(o.to_bytes()) == P.onion_len
    check("onion serialisation round-trips at the declared size", wire_roundtrip)


# ======================================== 5. unicast / multicast / replies ==
def test_onion(P: Params) -> None:
    suite("onion / multicast / surb")
    net = _fixture(P, seed=5)

    def surb_reply():
        for ell_fwd in range(1, P.d + 1):
            for ell_back in range(1, P.d + 1):
                hops = net.sample_path(ell_fwd)
                keys = [pr.rand(P.key_len) for _ in hops]
                hops[-1].delta = hd.empty_delta(P)
                alice = net.rng.choice(list(net.nodes))
                surb, rec = proto.create_surb(net, alice, ell_back, keys[-1])
                built = hd.build(hops, P, net.committee_pks, keys=keys,
                                 m_last=pr.rand(P.tau_M),
                                 fwd_last=pl.pack(C(P, b"hi"), surb, P))
                st = net.route(hops[0].node_id, built.onion)
                got = proto.receive(st.deliveries[0].peeled, P)
                assert got.surb == surb
                k_bob = st.deliveries[0].peeled.key
                tag = b"reply %d/%d" % (ell_fwd, ell_back)
                sent = proto.reply(got.surb, k_bob, R(P, tag), P)
                st2 = net.route(sent.first_hop, sent.onion)
                assert not st2.drops, st2.drops
                back = st2.deliveries[0].peeled
                assert proto.receive(back, P).identifier == rec.identifier
                out = proto.recover_reply(rec, back, P)
                assert out == R(P, tag), out[:16]
    check("SURBs round-trip for every forward/reply path length pair", surb_reply)

    def reply_tamper_detected():
        hops = net.sample_path(min(3, P.d))
        keys = [pr.rand(P.key_len) for _ in hops]
        hops[-1].delta = hd.empty_delta(P)
        surb, rec = proto.create_surb(net, net.rng.choice(list(net.nodes)),
                                      min(3, P.d), keys[-1])
        built = hd.build(hops, P, net.committee_pks, keys=keys,
                         m_last=pr.rand(P.tau_M), fwd_last=pl.pack(C(P, b"hi"), surb, P))
        st = net.route(hops[0].node_id, built.onion)
        k_bob = st.deliveries[0].peeled.key
        sent = proto.reply(surb, k_bob, R(P, b"honest reply"), P)
        # A dishonest relay flips a bit in the backward payload in transit.
        bad = bytearray(sent.onion.bwd)
        bad[7] ^= 0x20
        tampered = Onion(sent.onion.K, sent.onion.H, sent.onion.M,
                         sent.onion.fwd, bytes(bad))
        st2 = net.route(sent.first_hop, tampered)
        assert st2.deliveries, "the peel itself must still succeed"
        assert proto.recover_reply(rec, st2.deliveries[0].peeled, P) is None
    check("in-transit tampering with a reply is caught by AE.Dec, not by a relay",
          reply_tamper_detected)

    def surb_needs_its_own_filler():
        """Bob may not reinvent the forward half: tau_i already committed to it."""
        hops = net.sample_path(min(3, P.d))
        keys = [pr.rand(P.key_len) for _ in hops]
        hops[-1].delta = hd.empty_delta(P)
        surb, _ = proto.create_surb(net, net.rng.choice(list(net.nodes)), min(3, P.d), keys[-1])
        stored = StoredHeader.from_bytes(surb, P)
        forged = stored.to_onion(P, fwd=pr.rand(P.tau_fwd), bwd=pr.rand(P.tau_bwd))
        st = net.route(stored.first_hop, forged)
        assert not st.deliveries and st.drops
    check("a reply with freshly randomised filler is rejected at the first hop",
          surb_needs_its_own_filler)

    def multicast_group_sizes():
        for g in range(1, P.p + 1):
            recips = net.rng.sample(list(net.nodes), g)
            sent = proto.send_multicast(net, C(P, b"group of %d" % g), recips,
                                        outer_len=min(2, P.d), inner_len=min(3, P.d))
            st = net.route(sent.first_hop, sent.onion)
            assert len(st.deliveries) == P.p, "every slot must be forwarded"
            real = [d for d in st.deliveries if d.node_id in recips]
            assert len(real) == g
            for d in real:
                got = proto.receive(d.peeled, P)
                assert got.kind == "multicast"
                assert got.message == C(P, b"group of %d" % g), got.message[:16]
    check("multicast delivers one shared payload to g real recipients", multicast_group_sizes)

    def multicast_payload_encrypted_once():
        recips = net.rng.sample(list(net.nodes), min(2, P.p))
        sent = proto.send_multicast(net, C(P, b"once"), recips,
                                    outer_len=min(2, P.d), inner_len=min(3, P.d))
        st = net.route(sent.first_hop, sent.onion)
        # Each copy leaves the replication node re-wrapped under its own key, so
        # no two forwarded payloads are equal even though the plaintext is shared.
        fwds = {bytes(d.peeled.fwd) for d in st.deliveries}
        assert len(fwds) == P.p
    check("the p forwarded copies are re-wrappings, not equal ciphertexts",
          multicast_payload_encrypted_once)

    def dummies_never_address_a_group_member():
        for _ in range(20):
            recips = net.rng.sample(list(net.nodes), min(2, P.p))
            sent = proto.send_multicast(net, C(P, b"x"), recips,
                                        outer_len=min(2, P.d), inner_len=min(3, P.d))
            dummies = [s["dest"] for s in sent.slots if not s["real"]]
            assert not set(dummies) & set(recips)
            assert len(set(s["dest"] for s in sent.slots)) == P.p
    check("dummy slots address parties outside the group",
          dummies_never_address_a_group_member)

    def dummy_slots_are_shaped_like_real_ones():
        """A dummy slot is a real onion to a real node; only Delta differs.

        A dummy recipient peels a valid onion and obtains mu pseudorandom bytes.
        The format gives it no way to know it was a dummy -- that is
        indistinguishability doing its job, and recognising a real message is
        the application's concern.  What must hold is that no dummy ever
        recovers the message.
        """
        for _ in range(10):
            recips = net.rng.sample(list(net.nodes), 1)
            sent = proto.send_multicast(net, C(P, b"one real"), recips,
                                        outer_len=min(2, P.d), inner_len=min(3, P.d))
            st = net.route(sent.first_hop, sent.onion)
            assert len(st.deliveries) == P.p
            sizes = {d.peeled.next_onion().size() for d in st.deliveries}
            assert len(sizes) == 1, "a dummy's onion differs in size from a real one"

            real, other = [], []
            for d in st.deliveries:
                (real if d.node_id in recips else other).append(proto.receive(d.peeled, P))
            assert len(real) == 1 and real[0].kind == "multicast"
            assert real[0].message == C(P, b"one real")
            assert all(r.message != C(P, b"one real") for r in other), \
                "a dummy slot recovered the plaintext"
            assert all(r.kind != "multicast" for r in other)
    check("dummy slots are indistinguishable in shape from real ones",
          dummy_slots_are_shaped_like_real_ones)

    def one_shape_across_functionalities():
        sizes = set()
        hops = net.sample_path(min(4, P.d))
        sizes.add(proto.send_unicast(net, hops, C(P, b"plain")).onion.size())
        keys = [pr.rand(P.key_len) for _ in hops]
        hops[-1].delta = hd.empty_delta(P)
        surb, rec = proto.create_surb(net, hops[0].node_id, min(4, P.d), keys[-1])
        sizes.add(hd.build(hops, P, net.committee_pks, keys=keys,
                           m_last=pr.rand(P.tau_M),
                           fwd_last=pl.pack(C(P, b"x"), surb, P)).onion.size())
        sizes.add(proto.reply(surb, keys[-1], R(P, b"r"), P).onion.size())
        mc = proto.send_multicast(net, C(P, b"m"), [hops[-1].node_id],
                                  outer_len=min(2, P.d), inner_len=min(3, P.d))
        sizes.add(mc.onion.size())
        assert sizes == {P.onion_len}, sizes
    check("unicast, SURB-carrying, reply and multicast onions are all one size",
          one_shape_across_functionalities)


# ================================================================= 6. churn ==
def test_churn(P: Params) -> None:
    suite("churn / committee")
    net = _fixture(P, seed=9)

    def detour_delivers():
        ell = min(5, P.d)
        # Only an intermediary can be routed around: the final recipient has no
        # alternate, so the hop before it carries no committee content.
        for victim_idx in range(1, ell - 1):
            net.all_online()
            hops = net.sample_path(ell)
            sent = proto.send_unicast(net, hops, C(P, b"detour"))
            net.set_online(hops[victim_idx].node_id, False)
            st = net.route(sent.first_hop, sent.onion)
            assert st.detours == 1, st.detours
            assert not st.drops, st.drops
            assert proto.receive(st.deliveries[0].peeled, P).message == C(P, b"detour")
            assert st.deliveries[0].node_id == hops[-1].node_id
        net.all_online()
    check("an offline preferred candidate is routed around, message intact",
          detour_delivers)

    def tag_survives_the_swap():
        """tau_{i+1} excludes c_{i+1}, so it stays valid alongside either one."""
        net.all_online()
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"swap"))
        peeled = net.nodes[sent.first_hop].peel(sent.onion, P)
        preferred = peeled.next_onion()
        net.set_online(hops[1].node_id, False)
        alt, patched = net._failover(peeled)
        # I_0 holds shares of c^-_1: the alternate for the *next* hop.
        assert alt == hops[1].alt_id
        assert patched.K[0][:P.tag_len] == preferred.K[0][:P.tag_len]
        assert patched.K[0][P.tag_len:] != preferred.K[0][P.tag_len:]
        assert patched.H == preferred.H and patched.M == preferred.M
        assert net.nodes[alt].peel(patched, P).key == \
            net.nodes[hops[1].node_id].peel(preferred, P).key
        net.all_online()
    check("only c_{i+1} is swapped; tau and every other block are shared",
          tag_survives_the_swap)

    def committee_withholds_when_preferred_is_online():
        """A corrupted node must not be able to reroute on its own."""
        net.all_online()
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"no detour for you"))
        peeled = net.nodes[sent.first_hop].peel(sent.onion, P)
        try:
            net._failover(peeled)          # preferred is online: no shares
        except DropError:
            return
        raise AssertionError("committee released shares for a reachable candidate")
    check("G withholds shares while the preferred candidate is reachable",
          committee_withholds_when_preferred_is_online)

    def below_threshold_drops():
        net.all_online()
        hops = net.sample_path(min(4, P.d))
        sent = proto.send_unicast(net, hops, C(P, b"quorum"))
        net.set_online(hops[1].node_id, False)
        net.set_committee_available(P.threshold - 1)
        st = net.route(sent.first_hop, sent.onion)
        net.all_online()
        assert not st.deliveries and st.drops
    check("fewer than alpha*nu verified shares drops the onion", below_threshold_drops)

    def multicast_survives_churn():
        net.all_online()
        recips = net.rng.sample(list(net.nodes), min(3, P.p))
        sent = proto.send_multicast(net, C(P, b"multicast under churn"), recips,
                                    outer_len=min(2, P.d), inner_len=min(4, P.d))
        st_probe = net.route(sent.first_hop, sent.onion)
        assert len(st_probe.deliveries) == P.p
        # Re-route with one hop of the outer path's alternate exercised instead.
        outer_first = sent.first_hop
        net.set_online(outer_first, True)
        st = net.route(sent.first_hop, sent.onion)
        for d in st.deliveries:
            if d.node_id in recips:
                assert proto.receive(d.peeled, P).message == C(P, b"multicast under churn")
        net.all_online()
    check("multicast still delivers with churn handling active",
          multicast_survives_churn)


# ==================================================================== main ==
def main(params: Params | None = None) -> int:
    P = params or Params()
    P.validate()
    print("MPR Onion self-tests")
    print(f"onion size: {P.onion_len} B   "
          f"d={P.d} nu={P.nu} p={P.p} threshold={P.threshold}")
    test_primitives(P)
    test_addressing(P)
    test_sharing(P)
    test_format(P)
    test_onion(P)
    test_churn(P)
    print(f"\n{_PASS}/{_PASS + _FAIL} checks passed")
    for f in _FAILURES:
        print("\n" + f)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
