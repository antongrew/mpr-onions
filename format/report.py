"""Collects a benchmark run into a document.

The point is that a number is only useful if you can tell later what produced
it.  Every report records the parameter set, the derived sizes, the machine, the
Python version, the cryptography/OpenSSL versions and the git commit, so two reports can be
compared without guessing which build they came from.

Two files are written per run:

  results/bench-<timestamp>.md     readable, for the paper and for review
  results/bench-<timestamp>.json   structured, for plotting or diffing

plus `results/latest.md`, overwritten each run, so a link to it never goes
stale.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .params import Params


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5,
                                   cwd=Path(__file__).resolve().parent)
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except Exception:
        pass
    return None


def crypto_versions() -> str:
    import cryptography
    from cryptography.hazmat.backends.openssl.backend import backend
    return f"cryptography {cryptography.__version__}, {backend.openssl_version_text()}"


def environment() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "crypto": crypto_versions(),
        "git_commit": _git_commit(),
    }


@dataclass
class Section:
    title: str
    kind: str                      # "timings" | "sweep" | "sizes" | "text"
    rows: list = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    body: str = ""
    note: str = ""


@dataclass
class Report:
    params: Params
    env: dict
    runs: int
    n_nodes: int
    sections: list[Section] = field(default_factory=list)

    # ------------------------------------------------------------- adding --
    def add_timings(self, title: str, results, note: str = "") -> None:
        self.sections.append(Section(
            title, "timings",
            rows=[{"scenario": r.name, "runs": r.runs, "mean_ms": round(r.mean_ms, 3),
                   "median_ms": round(r.median_ms, 3), "p95_ms": round(r.p95_ms, 3),
                   "note": r.note} for r in results],
            columns=["scenario", "runs", "mean_ms", "median_ms", "p95_ms"],
            note=note))

    def add_sweep(self, title: str, varied: str, rows, note: str = "") -> None:
        self.sections.append(Section(
            title, "sweep",
            rows=[{varied: r.label.split("=", 1)[-1], "onion_bytes": r.onion,
                   "k_block": r.k_block, "h_block": r.h_block, "m_block": r.m_block,
                   "build_ms": round(r.build_ms, 3), "hop_ms": round(r.hop_ms, 3),
                   "detour_ms": round(r.detour_ms, 3)} for r in rows],
            columns=[varied, "onion_bytes", "k_block", "h_block", "m_block",
                     "build_ms", "hop_ms", "detour_ms"],
            note=note))

    def add_sizes(self, title: str, sizes: dict, note: str = "") -> None:
        self.sections.append(Section(
            title, "sizes",
            rows=[{"quantity": k, "value": v} for k, v in sizes.items()],
            columns=["quantity", "value"], note=note))

    def add_text(self, title: str, body: str) -> None:
        self.sections.append(Section(title, "text", body=body.rstrip()))

    # ---------------------------------------------------------- rendering --
    def _table(self, section: Section) -> list[str]:
        cols = section.columns
        out = ["| " + " | ".join(cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
        for row in section.rows:
            out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        notes = [r["note"] for r in section.rows if r.get("note")]
        if notes:
            out.append("")
            out.extend(f"- {n}" for n in notes)
        return out

    def to_markdown(self) -> str:
        p = self.params
        out = [
            "# MPR Onion benchmark results",
            "",
            f"Generated {self.env['timestamp']} — "
            f"onion size **{p.onion_len} B**, "
            f"`d={p.d} nu={p.nu} p={p.p} alpha={p.alpha}`.",
            "",
            "Timings are wall clock on one core and should be read as relative",
            "costs between scenarios, not as absolute throughput.",
            "",
            "## Run",
            "",
            "| | |",
            "|---|---|",
            f"| timestamp | {self.env['timestamp']} |",
            f"| git commit | {self.env['git_commit'] or 'not a git checkout'} |",
            f"| python | {self.env['python']} |",
            f"| platform | {self.env['platform']} |",
            f"| processor | {self.env['processor']} |",
            f"| crypto library | {self.env.get('crypto', self.env.get('x25519_backend'))} |",
            f"| runs per measurement | {self.runs} |",
            f"| nodes in the network | {self.n_nodes} |",
            "",
            "## Parameters",
            "",
            "| | |",
            "|---|---|",
            f"| d (K/H stack depth) | {p.d} |",
            f"| nu (committee size) | {p.nu} |",
            f"| alpha | {p.alpha} |",
            f"| alpha·nu (threshold) | {p.threshold} |",
            f"| f (tolerated corruptions) | {p.f_max} |",
            f"| p (replication factor) | {p.p} |",
            f"| onion size | {p.onion_len} B |",
            "",
        ]
        for section in self.sections:
            out.append(f"## {section.title}")
            out.append("")
            if section.kind == "text":
                out.append(section.body)
            else:
                out.extend(self._table(section))
            if section.note:
                out.append("")
                out.append(section.note)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def to_json(self) -> str:
        return json.dumps({
            "environment": self.env,
            "runs": self.runs,
            "n_nodes": self.n_nodes,
            "parameters": {
                "d": self.params.d, "nu": self.params.nu,
                "alpha": self.params.alpha, "p": self.params.p,
                "threshold": self.params.threshold, "f_max": self.params.f_max,
                "key_len": self.params.key_len, "msg_max": self.params.msg_max,
            },
            "sizes": {
                "tau_K": self.params.tau_K, "tau_H": self.params.tau_H,
                "tau_M": self.params.tau_M, "tau_fwd": self.params.tau_fwd,
                "tau_bwd": self.params.tau_bwd, "tau_U": self.params.tau_U,
                "k_block": self.params.d * self.params.tau_K,
                "h_block": self.params.d * self.params.tau_H,
                "bundle_len": self.params.bundle_len,
                "surb_len": self.params.surb_len,
                "onion_len": self.params.onion_len,
            },
            "sections": [asdict(s) for s in self.sections],
        }, indent=2)

    # ------------------------------------------------------------ writing --
    def write(self, directory: str | Path = "results") -> list[Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stamp = (self.env["timestamp"].replace("-", "").replace(":", "")
                 .replace("+0000", "Z"))
        stem = f"bench-{stamp}"
        md, js, latest = d / f"{stem}.md", d / f"{stem}.json", d / "latest.md"
        md.write_text(self.to_markdown(), encoding="utf-8")
        js.write_text(self.to_json(), encoding="utf-8")
        latest.write_text(self.to_markdown(), encoding="utf-8")
        return [md, js, latest]
