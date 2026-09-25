"""Benchmarks.  Run with `python -m format.bench`.

Measures the two costs a sender and a relay actually pay, separately:

  build    everything the sender does before the onion leaves: sampling keys,
           the filler string, the alternate ciphertexts, the nu committee
           inputs per hop, and the layer wrapping
  process  one hop's peel: one public-key decryption, the tag check, one bundle
           decryption, both payload halves

Timings are wall clock on one core and dominated by the public-key operations,
so they should be read as relative costs between scenarios rather than as
absolute throughput numbers.
"""

from __future__ import annotations

import statistics
import textwrap
import time
from dataclasses import dataclass

from .. import ae, pke
from .. import header as hd
from .. import node as nd
from .. import payload as pl
from .. import primitives as pr
from .. import protocol as proto
from .network import Network
from .params import ROLE_REPLICATION, Params
from .report import Report, crypto_versions, environment


@dataclass
class Result:
    name: str
    runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    note: str = ""


def _time(fn, runs: int) -> tuple[list[float], object]:
    samples, last = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, last


def measure(name: str, fn, runs: int, note: str = "") -> Result:
    samples, _ = _time(fn, runs)
    samples.sort()
    return Result(name, runs, statistics.fmean(samples), statistics.median(samples),
                  samples[min(len(samples) - 1, int(0.95 * len(samples)))], note)


def table(title: str, rows: list[Result]) -> str:
    out = [f"\n{title}", "-" * 78,
           f"{'scenario':<40}{'runs':>5}{'mean':>9}{'median':>9}{'p95':>9}",
           f"{'':<40}{'':>5}{'ms':>9}{'ms':>9}{'ms':>9}"]
    for r in rows:
        out.append(f"{r.name:<40}{r.runs:>5}{r.mean_ms:>9.2f}"
                   f"{r.median_ms:>9.2f}{r.p95_ms:>9.2f}")
        if r.note:
            out.extend(textwrap.wrap(r.note, width=74,
                                     initial_indent="    ", subsequent_indent="    "))
    return "\n".join(out)


# ------------------------------------------------------------------ build ---
def bench_build(net: Network, P: Params, runs: int) -> list[Result]:
    rows = []

    for ell in (2, 3, P.d):
        def build(ell=ell):
            hops = net.sample_path(ell)
            return proto.send_unicast(net, hops, pr.rand(P.msg_max))
        rows.append(measure(f"build unicast, l={ell}", build, runs,
                            note=f"{ell} layers, {(ell - 2) * P.nu} committee inputs"))

    def build_no_alt():
        hops = net.sample_path(P.d, with_alternates=False)
        return proto.send_unicast(net, hops, pr.rand(P.msg_max))
    rows.append(measure(f"build unicast, l={P.d}, no churn support", build_no_alt, runs,
                        note="isolates the cost of c^- and the nu committee inputs"))

    def build_surb():
        return proto.create_surb(net, net.rng.choice(list(net.nodes)), P.d, pr.rand(P.key_len))
    rows.append(measure(f"build SURB, l={P.d}", build_surb, runs,
                        note="reply header only; the sender keeps one omsk"))

    def build_with_surb():
        hops = net.sample_path(P.d)
        keys = [pr.rand(P.key_len) for _ in hops]
        hops[-1].delta = hd.empty_delta(P)
        surb, _ = proto.create_surb(net, hops[0].node_id, P.d, keys[-1])
        return hd.build(hops, P, net.committee_pks, keys=keys,
                        m_last=pr.rand(P.tau_M), fwd_last=pl.pack(pr.rand(P.msg_max), surb, P))
    rows.append(measure(f"build unicast + SURB, l={P.d}", build_with_surb, runs,
                        note="two full headers: forward path and reply path"))

    for g in (1, P.p):
        def build_mc(g=g):
            recips = net.rng.sample(list(net.nodes), g)
            return proto.send_multicast(net, pr.rand(P.msg_max), recips,
                                        outer_len=2, inner_len=3)
        rows.append(measure(f"build multicast, g={g} of p={P.p}", build_mc, max(3, runs // 3),
                            note=f"1 outer header + {P.p} inner headers, "
                                 f"payload encrypted once"))

    def build_reply():
        return proto.reply(surb_bytes, k_bob, pr.rand(P.bwd_ct_len), P)
    surb_bytes, _rec = proto.create_surb(net, net.rng.choice(list(net.nodes)),
                                         P.d, pr.rand(P.key_len))
    k_bob = pr.rand(P.key_len)
    rows.append(measure("build reply from a held SURB", build_reply, runs * 5,
                        note="one AE.Enc; no public-key work at all"))
    return rows


# ---------------------------------------------------------------- process ---
def bench_process(net: Network, P: Params, runs: int) -> list[Result]:
    rows = []

    hops = net.sample_path(P.d)
    sent = proto.send_unicast(net, hops, pr.rand(P.msg_max))
    node = net.nodes[sent.first_hop]
    rows.append(measure("process one hop (intermediary)",
                        lambda: node.peel(sent.onion, P), runs * 5,
                        note="1 PKE decryption, tag check, bundle + both U halves"))

    peeled = node.peel(sent.onion, P)
    rows.append(measure("  of which: AE.Unwrap on bwd",
                        lambda: ae.unwrap(peeled.key, sent.onion.bwd, P),
                        runs * 20))

    rows.append(measure(f"route full path, l={P.d} (no churn)",
                        lambda: net.route(sent.first_hop, sent.onion), runs,
                        note=f"{P.d} sequential hops"))

    # churn on one hop
    victim = hops[2].node_id

    def route_churn():
        net.set_online(victim, False)
        st = net.route(sent.first_hop, sent.onion)
        net.set_online(victim, True)
        return st
    rows.append(measure(f"route full path, l={P.d}, one detour", route_churn, runs,
                        note=f"adds {P.nu} committee decryptions + reconstruction"))

    recips = net.rng.sample(list(net.nodes), P.p)
    mc = proto.send_multicast(net, pr.rand(P.msg_max), recips, outer_len=2, inner_len=3)
    rows.append(measure(f"route multicast, p={P.p}, inner l=3",
                        lambda: net.route(mc.first_hop, mc.onion), max(3, runs // 2),
                        note=f"{2 + P.p * 3} hops total across all branches"))


    def replicate_only():
        n = net.nodes[mc.first_hop]
        p1 = n.peel(mc.onion, P)
        if p1.role == ROLE_REPLICATION:
            return nd.replicate(p1, P)
        return p1
    rows.append(measure("  replication node: peel + p re-wraps", replicate_only,
                        max(5, runs // 2),
                        note="p keystream evaluations over tau_fwd, no PKE per copy"))

    hops2 = net.sample_path(P.d)
    keys = [pr.rand(P.key_len) for _ in hops2]
    hops2[-1].delta = hd.empty_delta(P)
    surb, rec = proto.create_surb(net, hops2[0].node_id, P.d, keys[-1])
    built = hd.build(hops2, P, net.committee_pks, keys=keys,
                     m_last=pr.rand(P.tau_M), fwd_last=pl.pack(pr.rand(P.msg_max), surb, P))
    st = net.route(hops2[0].node_id, built.onion)
    k_bob = st.deliveries[0].peeled.key
    reply_sent = proto.reply(surb, k_bob, pr.rand(P.bwd_ct_len), P)
    rows.append(measure(f"route reply, l={P.d}",
                        lambda: net.route(reply_sent.first_hop, reply_sent.onion), runs,
                        note="indistinguishable from a forward onion to every relay"))

    st_reply = net.route(reply_sent.first_hop, reply_sent.onion)
    back = st_reply.deliveries[0].peeled
    rows.append(measure("recover reply (unwind + AE.Dec)",
                        lambda: proto.recover_reply(rec, back, P), runs * 10,
                        note=f"{P.d} AE.Wrap steps regenerated from one omsk"))
    return rows


# ------------------------------------------------- per-hop cost breakdown ---
def bench_hop_breakdown(net: Network, P: Params, runs: int) -> list[Result]:
    """Where a hop's time actually goes.

    With every primitive from `cryptography` (OpenSSL), the public-key
    decryption is the largest single item, but the symmetric work over the
    fixed blocks is still the larger share in total.  Of that, the dominant
    part is the HMAC for tau_i over hdr_i and fwd_i, not the ChaCha20 keystream:
    OpenSSL's ChaCha20 runs several times faster per byte than its HMAC-SHA-256.
    That symmetric part grows with tau_M, so per-hop cost still scales with M,
    just far more gently than an unoptimised keystream suggested.
    """
    from .params import N_FWD, N_HEADER
    from .primitives import dec, verify

    hops = net.sample_path(P.d)
    sent = proto.send_unicast(net, hops, pr.rand(P.msg_max))
    node = net.node(sent.first_hop)
    onion = sent.onion
    peeled = node.peel(onion, P)
    bundle = onion.bundle(P)
    n = runs * 5

    rows = [
        measure("Dec_sk(c_i) (public key)",
                lambda: pke.decrypt(node.sk, onion.c(P)), n),
        measure("derive k^SKE/k^MAC/k^PRF",
                lambda: hd.subkeys(peeled.key, P), n),
        measure("check tau_i over hdr_i and fwd_i",
                lambda: verify(peeled.mac_key, onion.K[0][:P.tag_len], bundle, onion.fwd), n),
        measure(f"decrypt header bundle ({P.bundle_len} B)",
                lambda: dec(peeled.ske, N_HEADER, bundle), n),
        measure(f"peel fwd ({P.tau_fwd} B)",
                lambda: dec(peeled.ske, N_FWD, onion.fwd), n),
        measure(f"AE.Unwrap bwd ({P.tau_bwd} B)",
                lambda: ae.unwrap(peeled.key, onion.bwd, P), n),
        measure("refill K^d and H^d",
                lambda: (hd.filler_K(peeled.ske, P), hd.filler_H(peeled.ske, P)), n),
    ]
    total = measure("full peel (all of the above)", lambda: node.peel(onion, P), n)
    parts = sum(r.median_ms for r in rows)
    biggest = max(rows, key=lambda r: r.median_ms)
    symmetric = sum(r.median_ms for r in rows[2:])
    total.note = (
        f"parts sum to {parts:.2f} ms. Largest single item: {biggest.name} at "
        f"{100 * biggest.median_ms / parts:.0f}% of the parts (medians). Public-key "
        f"work is {100 * rows[0].median_ms / parts:.0f}%, symmetric work over "
        f"{P.bundle_len + P.tau_U} B of fixed blocks {100 * symmetric / parts:.0f}% "
        f"(tau_M is {100 * P.tau_M / P.bundle_len:.0f}% of the bundle).")
    rows.append(total)
    return rows


def bench_position_independence(net: Network, P: Params, runs: int) -> list[Result]:
    """A hop's cost must not reveal where on the path it sits."""
    hops = net.sample_path(P.d)
    sent = proto.send_unicast(net, hops, pr.rand(P.msg_max))

    layers = []
    cur, nid = sent.onion, sent.first_hop
    for _ in range(P.d - 1):
        layers.append((net.node(nid), cur))
        peeled = net.node(nid).peel(cur, P)
        if peeled.next_id is None:
            break
        cur, nid = peeled.next_onion(), peeled.next_id

    # Sample the positions round-robin after a warm-up, so that interpreter
    # warm-up, frequency scaling or a background burst lands on every position
    # alike instead of on whichever one happens to be measured first.  With a
    # hop now costing a few hundred microseconds this matters: measured one
    # position after another, position 0 used to come out ~50% slower.
    n = runs * 10
    for node, onion in layers * 20:
        node.peel(onion, P)
    samples = [[] for _ in layers]
    for _ in range(n):
        for i, (node, onion) in enumerate(layers):
            t0 = time.perf_counter()
            node.peel(onion, P)
            samples[i].append((time.perf_counter() - t0) * 1000.0)
    rows = []
    for i, s in enumerate(samples):
        s.sort()
        rows.append(Result(f"peel at position {i} of {P.d}", n, statistics.fmean(s),
                           statistics.median(s), s[min(n - 1, int(0.95 * n))], ""))
    # Compare medians: the means are skewed by scheduling noise, and the point
    # here is whether cost varies *systematically* with position, not whether
    # one run happened to be slow.
    meds = [r.median_ms for r in rows]
    spread = max(meds) - min(meds)
    rows[-1].note = (
        f"spread across positions (median): {spread:.3f} ms, "
        f"{100 * spread / statistics.fmean(meds):.1f}% of the median. A hop's "
        f"work does not depend on where it sits: it peels one K^1, one bundle "
        f"and both U halves wherever it is, and the blocks are all fixed-size.")
    return rows


# -------------------------------------------------------------- parameter ---
#                                                                    sweeps
@dataclass
class SweepRow:
    label: str
    onion: int
    k_block: int
    h_block: int
    m_block: int
    build_ms: float
    hop_ms: float
    detour_ms: float
    note: str = ""


def sweep_table(title: str, varied: str, rows: list[SweepRow]) -> str:
    out = [f"\n{title}", "-" * 78,
           f"{varied:<10}{'onion B':>10}{'K blk':>8}{'H blk':>8}{'M blk':>9}"
           f"{'build':>9}{'hop':>8}{'detour':>9}",
           f"{'':<10}{'':>10}{'':>8}{'':>8}{'':>9}{'ms':>9}{'ms':>8}{'ms':>9}"]
    for r in rows:
        out.append(f"{r.label:<10}{r.onion:>10}{r.k_block:>8}{r.h_block:>8}"
                   f"{r.m_block:>9}{r.build_ms:>9.2f}{r.hop_ms:>8.2f}"
                   f"{r.detour_ms:>9.2f}")
    return "\n".join(out)


_ROUNDS = 5


def _sweep_series(points: list[tuple[str, Params, int | None]], runs: int,
                  n_nodes: int) -> list[SweepRow]:
    """Build, forward and detour under each parameter set of one series.

    Samples are taken round-robin: `_ROUNDS` passes over the whole series, each
    taking a share of every point's samples, and each row reports the median of
    everything it collected.  Measured one point after another, a burst of
    background load lands on whichever point happens to be running and shows up
    as a spurious step in the series; interleaved, it is spread over all of them.
    """
    setups = []
    for label, P, path_len in points:
        P.validate()
        net = Network(n_nodes, P, seed=1234)
        ell = min(path_len or P.d, P.d)

        def build(net=net, P=P, ell=ell):
            return proto.send_unicast(net, net.sample_path(ell), pr.rand(P.msg_max))

        sent = build()
        node = net.node(sent.first_hop)

        def hop(node=node, sent=sent, P=P):
            return node.peel(sent.onion, P)

        # One detour: take the second hop's preferred candidate offline.
        hops = net.sample_path(ell)
        sent2 = proto.send_unicast(net, hops, pr.rand(P.msg_max))
        detour = None
        if ell >= 3 and hops[1].alt_id is not None:
            def detour(net=net, victim=hops[1].node_id, sent2=sent2):
                net.set_online(victim, False)
                st = net.route(sent2.first_hop, sent2.onion)
                net.set_online(victim, True)
                return st
        setups.append((label, P, build, hop, detour, {"b": [], "h": [], "x": []}))

    per_round = max(1, runs // _ROUNDS)
    for _ in range(_ROUNDS):
        for _label, _P, build, hop, detour, acc in setups:
            acc["b"] += _time(build, per_round)[0]
            acc["h"] += _time(hop, per_round * 5)[0]
            if detour is not None:
                acc["x"] += _time(detour, max(1, per_round // 2))[0]

    rows = []
    for label, P, _b, _h, _x, acc in setups:
        rows.append(SweepRow(
            label, P.onion_len, P.d * P.tau_K, P.d * P.tau_H, P.tau_M,
            statistics.median(acc["b"]), statistics.median(acc["h"]),
            statistics.median(acc["x"]) if acc["x"] else float("nan")))
    return rows


def sweep_committee(base: Params, runs: int, n_nodes: int) -> list[SweepRow]:
    """nu: the size of the global committee G."""
    return _sweep_series([(f"nu={nu}", Params(d=base.d, nu=nu, alpha=base.alpha, p=base.p),
                           None) for nu in (3, 5, 7, 9, 11, 15)], runs, n_nodes)


def sweep_depth(base: Params, runs: int, n_nodes: int) -> list[SweepRow]:
    """d: the K/H stack depth, i.e. the upper bound on path length."""
    return _sweep_series([(f"d={d}", Params(d=d, nu=base.nu, alpha=base.alpha, p=base.p),
                           None) for d in (3, 4, 5, 6, 8, 10)], runs, n_nodes)


def sweep_depth_fixed_path(base: Params, runs: int, n_nodes: int) -> list[SweepRow]:
    """d again, but always routing a 3-hop path, to separate the two effects."""
    return _sweep_series([(f"d={d}", Params(d=d, nu=base.nu, alpha=base.alpha, p=base.p),
                           3) for d in (3, 4, 5, 6, 8, 10)], runs, n_nodes)


def sweep_replication(base: Params, runs: int, n_nodes: int) -> list[SweepRow]:
    """p: the replication factor, which sizes the M block every onion carries."""
    return _sweep_series([(f"p={p}", Params(d=base.d, nu=base.nu, alpha=base.alpha, p=p),
                           None) for p in (1, 2, 4, 6, 8)], runs, n_nodes)


# ------------------------------------------------------------- throughput ---
def bench_throughput(net: Network, P: Params, runs: int) -> str:
    hops = net.sample_path(P.d)
    sent = proto.send_unicast(net, hops, pr.rand(P.msg_max))
    node = net.node(sent.first_hop)
    r = measure("hop", lambda: node.peel(sent.onion, P), runs * 10)
    per_sec = 1000.0 / r.mean_ms
    goodput = per_sec * P.msg_max / 1e6
    line = P.onion_len * per_sec / 1e6
    return ("\nThroughput at one node (single core)\n" + "-" * 78 +
            f"\n  {per_sec:8.0f} onions/s      one hop costs {r.mean_ms:.2f} ms"
            f"\n  {line:8.1f} MB/s         of onion traffic a node can peel"
            f"\n  {goodput:8.1f} MB/s         of that which is payload capacity "
            f"({100 * P.msg_max / P.onion_len:.1f}% of the wire)")


# -------------------------------------------------------------- primitives --
def bench_primitives(P: Params, runs: int) -> str:
    """Cost of each library primitive at the sizes the format actually uses."""
    out = ["\nPrimitives (cryptography / OpenSSL)", "-" * 78]
    n = max(runs * 10, 100)
    k32, k16 = pr.rand(pr.STREAM_KEY_LEN), pr.rand(P.key_len)
    bundle = pr.rand(P.bundle_len)
    kp = pke.KeyPair.generate()
    c = pke.encrypt(kp.pk, pr.rand(P.c_plain_len))
    rows = [
        (f"ChaCha20 over hdr_i ({P.bundle_len} B)", lambda: pr.enc(k32, b"HEADER", bundle)),
        (f"ChaCha20 over fwd_i ({P.tau_fwd} B)",
         lambda: pr.enc(k32, b"FWD", bundle[:P.tau_fwd])),
        (f"HMAC-SHA-256 over hdr_i, fwd_i ({P.bundle_len + P.tau_fwd} B)",
         lambda: pr.mac(k16, bundle, bundle[:P.tau_fwd])),
        ("HKDF-SHA-256, one sub-key", lambda: pr.kdf(k16, b"SKE", 32)),
        (f"Enc_pk (X25519 + ChaCha20-Poly1305, {P.c_plain_len} B)",
         lambda: pke.encrypt(kp.pk, bundle[:P.c_plain_len])),
        ("Dec_sk", lambda: pke.decrypt(kp.sk, c)),
    ]
    for name, fn in rows:
        r = measure(name, fn, n)
        out.append(f"  {name:<52} {r.mean_ms * 1000:8.1f} us")
    return "\n".join(out)


# ---------------------------------------------------------- size breakdown --
def size_map(P: Params) -> dict:
    """Every derived size, for the report's size table."""
    return {
        "tau_K (one K sub-block)": P.tau_K,
        "tau_H (one H sub-block)": P.tau_H,
        "K block (d sub-blocks)": P.d * P.tau_K,
        "H block (d sub-blocks)": P.d * P.tau_H,
        "tau_M": P.tau_M,
        "tau_fwd": P.tau_fwd,
        "tau_bwd": P.tau_bwd,
        "tau_U": P.tau_U,
        "header bundle": P.bundle_len,
        "stored header / SURB": P.surb_len,
        "forward content field (mu)": P.msg_max,
        "backward content field (beta)": P.reply_max,
        "TOTAL onion size": P.onion_len,
    }


def bandwidth_lines(P: Params) -> str:
    out = []
    for g in range(1, P.p + 1):
        mc = P.onion_len * (2 + g * 3)
        uni = P.onion_len * (g * 5)
        out.append(f"  g={g}: multicast {mc:>9} B across all links, "
                   f"{g} independent unicasts {uni:>9} B  "
                   f"({100 * mc / uni:5.1f}%)")
    out.append("")
    out.append("  (link-count model: one payload encryption either way, but the "
               "multicast\n   sender transmits a single onion into the network "
               "instead of g of them.)")
    return "\n".join(out)


def size_breakdown(P: Params) -> str:
    parts = [("K block", P.d * P.tau_K), ("H block", P.d * P.tau_H),
             ("M block", P.tau_M), ("fwd", P.tau_fwd), ("bwd", P.tau_bwd)]
    out = ["\nWhere the bytes go", "-" * 78]
    for name, size in parts:
        bar = "#" * max(1, round(40 * size / P.onion_len))
        out.append(f"  {name:<10}{size:>8} B  {100 * size / P.onion_len:5.1f}%  {bar}")
    out.append(f"  {'content':<10}{P.msg_max:>8} B  "
               f"{100 * P.msg_max / P.onion_len:5.1f}%  forward content field (mu)")
    return "\n".join(out)


# ------------------------------------------------------------------- main ---
SECTIONS = ("sizes", "build", "process", "breakdown", "position",
            "sweeps", "throughput", "primitives", "bandwidth")


def main(params: Params | None = None, runs: int = 20, n_nodes: int = 60,
         sections: tuple[str, ...] = SECTIONS,
         out_dir: str | None = "results") -> int:
    """Run the benchmarks, print them, and write the results to a document.

    Everything printed to the terminal is also collected into a Report, which
    is written to `out_dir` as Markdown and JSON unless out_dir is None.
    """
    P = params or Params()
    P.validate()
    net = Network(n_nodes, P, seed=42)
    report = Report(P, environment(), runs, n_nodes)

    print("MPR Onion benchmarks")
    print(f"{crypto_versions()}   nodes: {n_nodes}   runs: {runs}")

    if "sizes" in sections:
        print("\nParameters and derived sizes")
        print("-" * 78)
        print(P.size_table())
        print(size_breakdown(P))
        report.add_sizes("Parameters and derived sizes", size_map(P),
                         note=f"`M` is {100 * P.tau_M / P.onion_len:.1f}% of every "
                              f"onion; usable payload is "
                              f"{100 * P.msg_max / P.onion_len:.1f}% of the wire.")

    if "build" in sections:
        rows = bench_build(net, P, runs)
        print(table("Header creation (sender side)", rows))
        report.add_timings("Header creation (sender side)", rows)

    if "process" in sections:
        rows = bench_process(net, P, runs)
        print(table("Processing (relay and endpoint side)", rows))
        report.add_timings("Processing (relay and endpoint side)", rows)

    if "breakdown" in sections:
        rows = bench_hop_breakdown(net, P, runs)
        print(table("Where one hop's time goes", rows))
        report.add_timings("Where one hop's time goes", rows)

    if "position" in sections:
        rows = bench_position_independence(net, P, runs)
        print(table("Processing cost by position on the path", rows))
        report.add_timings("Processing cost by position on the path", rows)

    if "sweeps" in sections:
        sweep_runs = max(3, runs // 3)
        sweep_nodes = min(n_nodes, 30)

        nu_rows = sweep_committee(P, sweep_runs, sweep_nodes)
        nu_note = ("H block = d · nu · tau_I, so nu drives the header linearly, "
                   "and a detour costs one decryption per member of G. Note that "
                   "tau_M grows with nu too: M holds p inner headers, each with "
                   "its own full H stack.")
        print(sweep_table("Committee size: nu (d, p held at defaults; path = d)",
                          "nu", nu_rows))
        print("    " + nu_note)
        report.add_sweep("Committee size: nu", "nu", nu_rows, note=nu_note)

        d_rows = sweep_depth(P, sweep_runs, sweep_nodes)
        print(sweep_table("Stack depth: d, routing a full-length path (l = d)",
                          "d", d_rows))
        report.add_sweep("Stack depth: d, routing a full-length path (l = d)",
                         "d", d_rows)

        d_fixed = sweep_depth_fixed_path(P, sweep_runs, sweep_nodes)
        d_note = ("Comparing the two d tables separates the cost of a deeper "
                  "stack from the cost of a longer route: with the path held at "
                  "3 hops, build cost grows far more slowly.")
        print(sweep_table("Stack depth: d, routing a fixed 3-hop path", "d", d_fixed))
        print("    " + d_note)
        report.add_sweep("Stack depth: d, routing a fixed 3-hop path", "d",
                         d_fixed, note=d_note)

        p_rows = sweep_replication(P, sweep_runs, sweep_nodes)
        p_note = ("p sizes the M block that every onion in the network carries, "
                  "including unicast ones that never replicate.")
        print(sweep_table("Replication factor: p", "p", p_rows))
        print("    " + p_note)
        report.add_sweep("Replication factor: p", "p", p_rows, note=p_note)

        report.add_text(
            "What the sweeps say",
            "M holds p inner headers, each containing full K and H stacks, so\n\n"
            "    onion  ~  (1 + p) * d * (tau_K + nu * tau_I)  +  tau_U\n\n"
            "The onion is multiplicative in p, d and nu rather than additive in\n"
            "them. Per-hop CPU follows it only weakly: the symmetric work over\n"
            "the fixed blocks grows with them, but next to the public-key\n"
            "decryption it is a modest share. p is the cheapest lever: it\n"
            "changes nothing else about the format.")

    if "throughput" in sections:
        text = bench_throughput(net, P, runs)
        print(text)
        report.add_text("Throughput at one node (single core)",
                        "```\n" + text.split("-" * 78)[-1].strip() + "\n```")

    if "primitives" in sections:
        text = bench_primitives(P, runs)
        print(text)
        report.add_text("Primitives (cryptography / OpenSSL)",
                        "```\n" + text.split("-" * 78)[-1].strip() + "\n```")

    if "bandwidth" in sections:
        print("\nBandwidth: multicast versus independent unicast")
        print("-" * 78)
        print(bandwidth_lines(P))
        report.add_text("Bandwidth: multicast versus independent unicast",
                        "```\n" + bandwidth_lines(P).strip() + "\n```")

    if out_dir is not None:
        written = report.write(out_dir)
        print("\nResults written to")
        for path in written:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="MPR Onion benchmarks")
    ap.add_argument("--runs", type=int, default=20,
                    help="samples per measurement (default 20)")
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--section", action="append", choices=SECTIONS,
                    help="run only these sections (repeatable)")
    ap.add_argument("--out", default="results",
                    help="directory for the written report (default: results)")
    ap.add_argument("--no-save", action="store_true",
                    help="print only; do not write a report")
    a = ap.parse_args()
    sys.exit(main(runs=a.runs, n_nodes=a.nodes,
                  sections=tuple(a.section) if a.section else SECTIONS,
                  out_dir=None if a.no_save else a.out))
    