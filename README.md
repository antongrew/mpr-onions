# MPR Onions — Python implementation

A implementation of the MPR Onion format: a churn-resilient,
multicast-capable, reply-supporting onion routing packet format, built as
described in the paper.

Every onion is `O_i = (K_i, H_i, M_i, U_i)` — two stacks of exactly `d`
sub-blocks, one fixed-size `M` field, and a `U` block split into a forward and a
backward half. The size is a protocol-wide constant: a plain forward message, a
message carrying a SURB, a reply, and a multicast message are all byte-identical
in shape at every hop of every path.

```
python -m format.selftest    # 39 checks across 6 suites
python -m format.bench       # sizes, build cost, processing cost
python demo.py               # unicast, churn, reply, multicast
pytest                       # the suite across three parameter sets
ruff check .                 # lint
```

Run all three from the `mpr_onion/` directory (the one holding `demo.py`), so
that `format` is importable as a package.

```
mpr_onion/
├── README.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── demo.py
├── format/              <- the package: import it as `format`
├── tests/               <- pytest wrapper over format/selftest.py
└── results/             <- benchmark reports
```

One dependency: `cryptography` (`pip install -e .`). Every cryptographic
primitive comes from it — nothing cryptographic is hand-rolled:

| paper | instantiation |
|---|---|
| `G_k`, `{·}ⁿₖ` | ChaCha20 (RFC 8439), 256-bit key; seek = start the block counter at `offset // 64` |
| `KDF` | HKDF-SHA-256 |
| `MAC`, `PRF` | HMAC-SHA-256, truncated to 16 B, constant-time verify |
| `Enc_pk` / `Dec_sk` | X25519 + HKDF-SHA-256 + ChaCha20-Poly1305 (48 B overhead) |
| `h_K` | SHA-256 |


## Modules

| module | contents |
|---|---|
| `params.py` | every parameter and every size derived from it |
| `address.py` | IPv4 addressing for the parties in the network |
| `primitives.py` | `{·}ⁿₖ` / `}·{ⁿₖ`, KDF, MAC |
| `pke.py` | `Enc_pk` / `Dec_sk` (X25519 hybrid, fixed-length ciphertexts) |
| `sharing.py` | `(α·ν, ν)` threshold secret sharing over GF(2⁸) |
| `ae.py` | `AE.Enc` / `AE.Dec` / `AE.Wrap` / `AE.Unwrap` |
| `onion.py` | the onion container, stored headers, wire format |
| `header.py` | header creation: filler string, committee inputs, layer wrapping |
| `node.py` | per-hop processing, committee members, replication |
| `network.py` | node registry, the fixed global committee `G`, the relay loop |
| `payload.py` | forward-payload layout, the multicast key tree `K[·]` |
| `protocol.py` | unicast, SURB creation and use, multicast, recipient recovery |
| `selftest.py`, `bench.py` | tests and benchmarks |
