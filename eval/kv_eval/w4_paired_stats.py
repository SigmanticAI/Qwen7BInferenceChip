#!/usr/bin/env python3
# w4_paired_stats.py — paired per-document statistics for the B3 stage-6
# weight-tier matrix, vs the committed S5 base samples (the honest sigma for
# deltas — same design as the S5/S4 paired readings).
#
#   python w4_paired_stats.py --base <base.samples.json> <tier.samples.json>...
#
# Pairs on doc_id intersection; reports paired delta +/- SE, z, and the
# agree / right-only / wrong-only discordant counts on acc_norm.

import argparse
import json
import math
from pathlib import Path


def load(p):
    return {d["doc_id"]: d for d in json.loads(Path(p).read_text())}


def paired(base, tier):
    ids = sorted(set(base) & set(tier))
    d = [tier[i]["acc_norm"] - base[i]["acc_norm"] for i in ids]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    z = mean / se if se > 0 else float("nan")
    agree = sum(1 for x in d if x == 0)
    right_only = sum(1 for x in d if x > 0)
    wrong_only = sum(1 for x in d if x < 0)
    return {"n": n, "delta": mean, "se": se, "z": z, "agree": agree,
            "right_only": right_only, "wrong_only": wrong_only,
            "base_acc_norm": sum(base[i]["acc_norm"] for i in ids) / n,
            "tier_acc_norm": sum(tier[i]["acc_norm"] for i in ids) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("tiers", nargs="+")
    args = ap.parse_args()
    base = load(args.base)
    print(f"base: {args.base} ({len(base)} docs)")
    print(f"{'tier file':52s} {'n':>6s} {'base':>7s} {'tier':>7s} "
          f"{'paired Δ':>9s} {'SE':>7s} {'z':>7s}  agree/right/wrong")
    for t in args.tiers:
        r = paired(base, load(t))
        print(f"{Path(t).name:52s} {r['n']:6d} {r['base_acc_norm']:7.4f} "
              f"{r['tier_acc_norm']:7.4f} {r['delta']:+9.4f} {r['se']:7.4f} "
              f"{r['z']:+7.2f}  {r['agree']}/{r['right_only']}/{r['wrong_only']}")


if __name__ == "__main__":
    main()
