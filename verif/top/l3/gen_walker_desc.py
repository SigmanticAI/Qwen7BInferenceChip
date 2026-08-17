#!/usr/bin/env python3
"""gen_walker_desc.py — B1 stage-5 sidecar: the compact layer descriptor a
real host would load once through WALK_DPTR/WALK_DDATA before kicking a walk.

A NEW sibling file, not an edit to gen_l3_vectors.py: the spec generator stays
untouched (B1_WALKER.md §7, "the trace extractor is a *new* sibling file").
Everything here is derived from the already-generated case files via the
stage-0 extractor, so the descriptor cannot drift from the stream it must
reproduce.

Per case it emits build/walker/<name>.desc:
    WALKABLE <0|1>     1 = CQ-8, may walk; 0 = grouped tier, MUST be refused
    GEOM/RQ/MASK       the three descriptor words
    TROWS <n>          informational, for the run log

§A-1: grouped-tier cases are emitted with WALKABLE 0 and their REAL tier, so
walker-mode L3 exercises the refusal path on them rather than silently
skipping them. A case that is refused then falls back to host mode and must
still pass bit-exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_trace import extract_case, load_manifest  # noqa: E402

TIER_CODE = {"CQ-8": 0, "CQ-4": 1, "CQ-4+": 2}


def tier_of(case_path: Path) -> int:
    """The tier a case stores/reads through, taken from phase A's
    TIER_CTRL write (gen_l3_vectors.py:350) — the build truth, not the name."""
    for line in case_path.read_text().splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "CSRW" and int(f[1], 16) == 0x20:
            return int(f[2], 16) & 0x3
    return 0


def main(build_dir: str = "build") -> int:
    b = Path(build_dir)
    man = load_manifest(b)
    if not man:
        print(f"no manifest under {b} — run `make vectors` first")
        return 2
    out = b / "walker"
    out.mkdir(parents=True, exist_ok=True)

    n_walk = n_ref = 0
    for c in man:
        path = b / "cases" / f"{c['name']}.txt"
        tr = extract_case(path, params=c)
        D, T = tr["params"]["D"], tr["params"]["T"]
        tier = tier_of(path)

        pv = [e for e in tr["channels"]["DESC"] if e["phase"] == "pv"]
        if pv:
            rq_scale, rq_shift = pv[0]["rq_scale"], pv[0]["rq_shift"]
        else:
            rq_scale, rq_shift = 0, 0

        # tip_auto_mixed re-routes per chunk (blkmap) — walker v1 emits exactly
        # two route levels, so it cannot reproduce that stream even at CQ-8.
        multi_route = len([e for e in tr["channels"]["ROUTE"]
                           if e["phase"] in ("score", "pv")]) > 2
        walkable = (tier == 0) and not multi_route

        geom = (D & 0xFF) | ((T & 0x1FF) << 8) | ((tier & 0x3) << 16)
        rq = ((rq_shift & 0x1F) << 16) | (rq_scale & 0xFFFF)
        (out / f"{c['name']}.desc").write_text(
            f"WALKABLE {1 if walkable else 0}\n"
            f"GEOM {geom:08x}\n"
            f"RQ {rq:08x}\n"
            f"MASK 00000003\n"
            f"TROWS {T}\n")
        if walkable:
            n_walk += 1
        else:
            n_ref += 1

    print(f"walker descriptors: {len(man)} cases "
          f"({n_walk} walkable CQ-8, {n_ref} refusal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
