#!/usr/bin/env python3
"""gen_walker_vectors.py — B1 stage-2 vectors: walker-vs-extracted-trace.

House pattern (verif/seq/gen_seq_vectors.py): an INDEPENDENT reference the TB
scoreboards against. Here the reference is not freshly invented — it is the L3
choreography itself, which docs/OPTIMIZATION.md:38 makes the spec ("the L3
choreography is the spec; walker-vs-script transaction-equivalence is
non-negotiable"). This script projects each generated case file to the ordered
score+pv drive subsequence via the stage-0 extractor and writes it as the
expected emission trace, together with everything the DUT needs to reproduce
it autonomously:

  - the compact layer descriptor (D, T, tier, the host-loaded requant pair,
    phase mask) — what a real host writes once through WALK_DPTR/WALK_DDATA;
  - s_q, as the tile publishes it on the ss_* tap at the end of q-inject;
  - the K/V fp16 records, replayed into the composite unit's SNOOP port so the
    store-time scale cache is populated the REAL way (§A-2) rather than
    back-doored. The cs/qs words the walker emits are therefore genuinely
    computed on-tile, not preloaded.

Only CQ-8 cases are emitted: walker v1 REFUSES grouped tiers (§A-1), and the
refusal path is tested separately by a directed vector, not by these.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parent / "top" / "l3"
sys.path.insert(0, str(L3))

from extract_trace import extract_case, load_manifest, decode_desc  # noqa: E402

# ops the walker must emit, in order, within the score+pv phases
DRIVE = {"ROUTE", "SJOB", "CS", "QJOB", "QS", "WJ", "FJOB", "KVW", "AJ", "DESC"}
PHASES = ("score", "pv")

# The suite's case sets. Chosen for SHAPE coverage, not count: the split path
# (ceil(T/8) > BPR) is exercised by only 3 of the 24 CQ-8 cases, and the m2
# composite mutant is observable ONLY at D=128 (D=64 cs and qs are exact at
# every width), so a set that omits either would report a false green.
SETS = {
    "small": ["adv_T1", "rand_d64_T5"],
    "split": ["calib_d64_T70", "calib_d64_T128"],
    "d128":  ["rand_d128_T5", "calib_d128_T100"],
}


def tap_payload(path: Path, kind: str) -> list[int]:
    """Pull a stateful tap stream ('<KIND> <count:hex>' then <count> bare
    hex lines) out of a case file."""
    out, want = [], 0
    for line in path.read_text().splitlines():
        if want:
            out.append(int(line.split()[0], 16))
            want -= 1
            continue
        f = line.split()
        if f and f[0] == kind:
            want = int(f[1], 16)
    return out


def ordered_drive(tr: dict) -> list[tuple[int, str, dict]]:
    """Merge the per-channel trace back into one line-ordered sequence,
    restricted to the walker's phases and opcodes."""
    rows = []
    for op, entries in tr["channels"].items():
        if op not in DRIVE:
            continue
        for e in entries:
            if e["phase"] in PHASES:
                rows.append((e["line"], op, e))
    rows.sort(key=lambda r: r[0])
    return rows


def emit_case(name: str, build: Path, man_by_name: dict) -> list[str]:
    c = man_by_name[name]
    path = build / "cases" / f"{name}.txt"
    tr = extract_case(path, params=c)
    D, T = tr["params"]["D"], tr["params"]["T"]
    BPR = D // 8

    # requant pair: from the pv descriptors (all BPR of them are identical —
    # assert it, because the walker carries ONE loaded pair for the phase)
    pv_desc = [e for e in tr["channels"]["DESC"] if e["phase"] == "pv"]
    assert pv_desc, f"{name}: no pv DESC"
    rq = {(e["rq_scale"], e["rq_shift"]) for e in pv_desc}
    assert len(rq) == 1, f"{name}: pv requant pair is not constant: {rq}"
    rq_scale, rq_shift = rq.pop()
    assert len(pv_desc) == BPR, f"{name}: expected {BPR} pv DESCs, got {len(pv_desc)}"

    # s_q: published on ss_* at the end of q-inject
    ess_q = [e for e in tr["channels"]["ESS"] if e["phase"] == "q_inject"]
    assert ess_q, f"{name}: no q-inject ESS (s_q)"
    s_q = ess_q[-1]["bits"]

    # K/V fp16 records: TAPF16 carries all K rows then all V rows, D each
    f16 = tap_payload(path, "TAPF16")
    assert len(f16) == 2 * T * D, \
        f"{name}: TAPF16 has {len(f16)}, expected 2*{T}*{D}={2*T*D}"

    L = [f"# case {name}  D={D} T={T} BPR={BPR} tier=CQ8"]
    geom = (D & 0xFF) | ((T & 0x1FF) << 8) | (0 << 16) | ((0 & 0xF) << 20)
    L.append(f"DESC_GEOM {geom:08x}")
    L.append(f"DESC_RQ {((rq_shift & 0x1F) << 16) | (rq_scale & 0xFFFF):08x}")
    L.append("DESC_MASK 00000003")               # en_score | en_pv
    L.append(f"SQ {s_q:04x}")
    # snoop replay: record r = K[r] for r<T, V[r-T] otherwise — the address
    # map the host programs in WRITE_ADDR (store_kv_phase:544,547)
    for r in range(2 * T):
        row = f16[r * D:(r + 1) * D]
        L.append(f"REC {r:04x} " + " ".join(f"{v:04x}" for v in row))
    for _ln, op, e in ordered_drive(tr):
        if op == "ROUTE":
            L.append(f"E ROUTE {e['r']:04x}")
        elif op == "SJOB":
            L.append(f"E SJOB {e['cols']:x}")
        elif op == "QJOB":
            L.append(f"E QJOB {e['mode']} {e['cols']:x}")
        elif op == "FJOB":
            L.append(f"E FJOB {e['rows']:x}")
        elif op in ("CS", "QS"):
            L.append(f"E {op} {e['bits']:08x}")
        elif op in ("AJ", "WJ"):
            L.append(f"E {op} {e['op']} {e['bank']} {e['pat']} "
                     f"{e['rows']:x} {e['nb']:x} {e['sel']:x}")
        elif op == "KVW":
            L.append(f"E KVW {e['data']:x}")
        elif op == "DESC":
            L.append(f"E DESC {e['opcode']:02x} {e['m']:x} {e['k']:x} "
                     f"{e['n']:x} {e['rq_en']} {e['rq_scale']:x} "
                     f"{e['rq_shift']:x} {e['mode_os']}")
    L.append("END")
    return L


def main(build_dir: str = str(L3 / "build")) -> int:
    build = Path(build_dir)
    man = load_manifest(build)
    if not man:
        print(f"no manifest under {build} — run `make -C verif/top/l3 vectors`")
        return 2
    by_name = {c["name"]: c for c in man}
    out_dir = HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for set_name, cases in SETS.items():
        lines: list[str] = []
        for nm in cases:
            if nm not in by_name:
                print(f"WARNING: case {nm} not in the manifest — skipped")
                continue
            lines += emit_case(nm, build, by_name)
        (out_dir / f"vectors_{set_name}.txt").write_text("\n".join(lines) + "\n")
        n_exp = sum(1 for x in lines if x.startswith("E "))
        total += n_exp
        print(f"vectors_{set_name}.txt: {len(cases)} case(s), "
              f"{n_exp} expected emissions")

    # directed refusal vectors. Case 1 (§A-1): a CQ-4 descriptor must be
    # REFUSED with WALK_ERR_TIER and emit NOTHING — grouped tiers never
    # produce a walker trace by construction. Case 2 (D-029 v1.1 hardening,
    # IB_WALK.md §2.1): a descriptor with an unknown fmt nibble
    # (GEOM[31:28] != 0) must be REFUSED with WALK_ERR_DESC — everything
    # else in it is a LEGAL CQ-8 geometry (D=64, T=8, tier=0), so the fmt
    # nibble is provably the only refusal cause. Before the hardening this
    # descriptor was silently mis-walked as v1; mutant m4_fmtskip re-creates
    # exactly that and must be caught by this vector.
    ref = ["# refusal: tier=CQ4 must raise WALK_ERR_TIER and emit nothing",
           "DESC_GEOM 00014001",          # D=1(bogus is fine), T=64, tier=1
           "DESC_RQ 00000000", "DESC_MASK 00000003", "SQ 3c00",
           "EXPECT_REFUSE 1", "END",
           "# refusal: fmt=1 (D-029 layer format) on a v1 walker -> DESC err",
           "DESC_GEOM 10000840",          # fmt=1, tier=0, T=8, D=64: legal but for fmt
           "DESC_RQ 00000000", "DESC_MASK 00000003", "SQ 3c00",
           "EXPECT_REFUSE 2", "END"]
    (out_dir / "vectors_refuse.txt").write_text("\n".join(ref) + "\n")
    print(f"vectors_refuse.txt: 2 directed refusal cases (tier, fmt)")
    print(f"TOTAL expected emissions across sets: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
