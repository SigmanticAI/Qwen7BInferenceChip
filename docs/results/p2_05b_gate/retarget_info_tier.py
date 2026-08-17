#!/usr/bin/env python3
"""retarget_info_tier.py — disclosed build-identity retarget for the b64 CL.

The L3 choreography (verif/top/l3/gen_l3_vectors.py:349) expects
INFO_TIER == 0x7 at D=64 — the l3 d64 REFERENCE build's tier population
(CQ-8/CQ-4/CQ-4+). cl_apex pins KVQ_OUTLIER_K=0, so the CL b64 has no CQ-4+
engine and INFO_TIER truthfully reads 0x3. This post-pass rewrites exactly
that ONE expectation per job (op=="r", a==0x1014, e==0x7 -> e==0x3) and
nothing else; the 0.5B jobs are CQ-8, so the absent engine is outside every
datapath check. Run after trace_to_regops.py:

    python3 docs/results/p2_05b_gate/retarget_info_tier.py \
        [src=build/p2_05b_regops] [dst=build/p2_05b_regops_b64cl]
"""
import json
import sys
from pathlib import Path

INFO_TIER_ADDR = 0x1014          # BAR0 0x1000 (CSR window) + INFO_TIER 0x14


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "build/p2_05b_regops")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else
               "build/p2_05b_regops_b64cl")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("job_*.regops.jsonl")):
        out = []
        for ln in f.read_text().splitlines():
            o = json.loads(ln)
            if (o.get("op") == "r" and o.get("a") == INFO_TIER_ADDR
                    and o.get("e") == 0x7):
                o["e"] = 0x3
                n += 1
                out.append(json.dumps(
                    {"op": "note",
                     "s": "INFO_TIER expect retargeted 0x7->0x3 (cl_apex "
                          "b64: KVQ_OUTLIER_K=0; jobs are CQ-8)"},
                    separators=(",", ":")))
            out.append(json.dumps(o, separators=(",", ":")))
        (dst / f.name).write_text("\n".join(out) + "\n")
    print(f"retargeted {n} INFO_TIER expectations -> {dst}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
