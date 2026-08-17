#!/usr/bin/env python3
# trace_to_fuel.py — IB-FUEL s3/s4: fuel-mode regops. Takes the committed
# per-job regops (trace_to_regops.py output) plus the DDR image manifest
# (make_ddr_image.py) and emits the SAME choreography with the weight
# stream coming FROM DDR instead of BAR0 pushes:
#
#   * every WB triple (w XW0, w XW1, pw XWP) is REMOVED — the beats now
#     arrive through decode->sh_ddr->reader->afifo->xw mux;
#   * a fuel prologue is PREPENDED per job:
#       poll FUEL_STAT.ddr_ready         (hardware truth: DDR calibrated —
#                                         uncounted, executor-parity with
#                                         the sim's per-file full reset)
#       w    FUEL_BASE  = region base_64B
#       w    FUEL_BEATS = region beats_64B
#       w    FUEL_CTRL  = {tag, GO, ar_owner=RUN, src=fuel}  (one write:
#                         mode change while idle + record push)
#   * NOTHING else changes — counted checks per job are byte-identical to
#     the host-mode file (§9.1 R7: the FUEL_ERR==0 audit is the EXECUTOR's
#     postlude, outside the counted stream — sim_main +fuel_audit).
#
# One record per job (the host-CSR producer); the walker's per-tensor
# records are IB-WALK's D-029 flow and do not pass through this tool.

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

XW0, XW1, XWP = 0x3040, 0x3044, 0x3048
FUEL_CTRL, FUEL_BASE, FUEL_BEATS, FUEL_STAT = 0x0020, 0x0024, 0x0028, 0x002C
STAT_DDR_READY = 0x8          # FUEL_STAT[3]
CTRL_SRC_FUEL, CTRL_AR_RUN, CTRL_GO = 0x1, 0x2, 0x100


def translate(job_path: Path, rec: dict, out_path: Path) -> tuple[int, int]:
    kept = removed = 0
    with job_path.open() as f, out_path.open("w") as o:
        def emit(op):
            o.write(json.dumps(op, separators=(",", ":")) + "\n")

        emit({"op": "note",
              "s": f"IB-FUEL: weights FROM DDR — region base_64B="
                   f"{rec['base_64B']} beats_64B={rec['beats_64B']} "
                   f"tag={rec['tag']} (WB triples removed)"})
        emit({"op": "poll", "a": FUEL_STAT,
              "m": STAT_DDR_READY, "e": STAT_DDR_READY})
        emit({"op": "w", "a": FUEL_BASE, "d": rec["base_64B"]})
        emit({"op": "w", "a": FUEL_BEATS, "d": rec["beats_64B"]})
        emit({"op": "w", "a": FUEL_CTRL,
              "d": (rec["tag"] << 16) | CTRL_GO | CTRL_AR_RUN | CTRL_SRC_FUEL})

        for ln in f:
            op = json.loads(ln)
            a = op.get("a")
            if (a == XW0 and op["op"] == "w") \
               or (a == XW1 and op["op"] == "w") \
               or (a == XWP and op["op"] == "pw"):
                removed += 1
                continue
            kept += 1
            o.write(ln)
    assert removed % 3 == 0, f"{job_path.name}: partial WB triple"
    assert removed // 3 == rec["beats_64B"] * 8, \
        f"{job_path.name}: WB count {removed//3} != region lane beats"
    return kept, removed // 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regops", default=str(REPO / "build/f2_regops"))
    ap.add_argument("--image", default=str(REPO / "build/ddr_image"))
    ap.add_argument("--out", default=str(REPO / "build/fuel_regops"))
    ap.add_argument("--jobs", type=int, default=0, help="limit (0 = all)")
    args = ap.parse_args()
    regions = {}
    with (Path(args.image) / "ddr_image.regions.jsonl").open() as f:
        for ln in f:
            r = json.loads(ln)
            regions[r["job"]] = r
    jobs = sorted(Path(args.regops).glob("job_*.regops.jsonl"))
    if args.jobs:
        jobs = jobs[:args.jobs]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for jp in jobs:
        name = jp.stem.replace(".regops", "")
        rec = regions[name]
        op = outdir / f"{name}.fuel.regops.jsonl"
        kept, beats = translate(jp, rec, op)
        print(f"[{name}] fuel regops: {kept} ops kept, "
              f"{beats} WB beats -> DDR record (tag={rec['tag']})")
    print(f"FUELREGOPS: {len(jobs)} jobs -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
