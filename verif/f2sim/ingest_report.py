#!/usr/bin/env python3
# ingest_report.py — perf/ingest-lane: reduce the f2sim ingest monitor's
# snapshot lines (sim_main.cpp, APEX_INGEST_MON build, +ingest_mon run) to
# the IB-FUEL ingest metrics: achieved bytes/cycle vs theoretical, busy-cycle
# decomposition, afifo occupancy distribution, burst/record gap stats.
#
# Every number this tool prints is SIM-measured on the behavioral twin
# (sh_ddr_beh.sv + verilated cl_apex). None is a silicon claim.
#
# Monitor build (no RTL change; default builds carry no monitor):
#   make -C verif/f2sim build D=64 DDR=1 OBJ=obj_ing_mon \
#     VFLAGS_EXTRA='+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 \
#       +define+APEX_CL_QSTAGE=14 +define+APEX_CL_DMODEL=64 \
#       --public-flat-rw -CFLAGS -DAPEX_INGEST_MON'
# Run with `+ingest_mon` added to the executor argv.
#
# Window: by default the E-6 walk window, the delta between the
# INGESTMON SNAP lines at CYCMARK E6R-PREGO and E6R-WALKDONE
# (walk_fuel_layer.py's own marks). --from/--to pick other marks;
# --whole-run uses first->last snapshot.
#
# Theoretical bounds printed alongside (derivation, IB_FUEL.md §2.2/§4):
#   * AXI R channel: 512 bit/beat = 64 B/shell-cycle.
#   * apex_afifo write side: W=64 bit = 8 B/shell-cycle — the architectural
#     ceiling of the current fuel line (2.000 GB/s at the 250 MHz shell).
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SNAP = re.compile(r"INGESTMON SNAP (\S+) (.*)")
REC = re.compile(r"INGESTMON REC n=(\d+) cyc=(\d+) arwords=(\d+) pushes=(\d+)")

FIELDS = ("cyc", "busy", "arhs", "arwords", "rbeats", "pushes", "stallfull",
          "pops", "tedges", "occ", "occmax", "occsum", "recs", "recgapsum",
          "recgapmax", "recgapn", "burstgapsum", "burstgapn",
          "reqtglsum", "reqtgln", "wfhs")
HIST_BUCKETS = ("0", "1-64", "65-128", "129-192", "193-256", "257-320",
                "321-384", "385-448", "449-511", "512")


def parse_snap(line: str):
    m = SNAP.search(line)
    if not m:
        return None
    label, rest = m.group(1), m.group(2)
    d = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)(?:\s|$)", rest)}
    hm = re.search(r"hist=([\d,]+)", rest)
    d["hist"] = [int(x) for x in hm.group(1).split(",")] if hm else [0] * 10
    return label, d


def load(log_path: Path):
    snaps, recs = [], []
    for line in log_path.read_text(errors="replace").splitlines():
        s = parse_snap(line)
        if s:
            snaps.append(s)
        r = REC.search(line)
        if r:
            recs.append(tuple(int(g) for g in r.groups()))
    return snaps, recs


def pick(snaps, label, which):
    hits = [d for (l, d) in snaps if l == label]
    if not hits:
        raise SystemExit(f"REFUSE: no INGESTMON SNAP labeled {label!r} "
                         f"({which}); labels present: "
                         f"{sorted(set(l for l, _ in snaps))}")
    if which == "from":
        return hits[0]
    return hits[-1]


def delta(a, b):
    # .get(k, 0): logs from the pre-split monitor lack the reqtgl/wfhs
    # fields; treat them as zero rather than refusing the whole log
    d = {k: b.get(k, 0) - a.get(k, 0) for k in FIELDS
         if k not in ("occ", "occmax", "recgapmax")}
    d["occ_end"] = b["occ"]
    d["occmax_end"] = b["occmax"]          # cumulative max, end-of-window
    d["recgapmax_end"] = b["recgapmax"]    # cumulative max, end-of-window
    d["hist"] = [y - x for x, y in zip(a["hist"], b["hist"])]
    return d


def fmt_bpc(nbytes, cycles):
    return nbytes / cycles if cycles else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--from", dest="mfrom", default="CYCMARK_E6R-PREGO")
    p.add_argument("--to", dest="mto", default="CYCMARK_E6R-WALKDONE")
    p.add_argument("--whole-run", action="store_true",
                   help="first->last snapshot instead of --from/--to marks")
    p.add_argument("--shell-mhz", type=float, default=250.0)
    p.add_argument("--tile-div", type=int, default=2)
    p.add_argument("--label", default="")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()

    snaps, recs = load(a.log)
    if not snaps:
        raise SystemExit(f"REFUSE: no INGESTMON SNAP lines in {a.log} — "
                         f"was the run made with +ingest_mon on a monitor "
                         f"build?")
    if a.whole_run:
        s0, s1 = snaps[0][1], snaps[-1][1]
        win = f"{snaps[0][0]} -> {snaps[-1][0]} (whole-run)"
    else:
        s0 = pick(snaps, a.mfrom, "from")
        s1 = pick(snaps, a.mto, "to")
        win = f"{a.mfrom} -> {a.mto}"
    d = delta(s0, s1)
    C = d["cyc"]
    if C <= 0:
        raise SystemExit("REFUSE: empty/negative window")

    ddr_bpc = fmt_bpc(64 * d["rbeats"], C)
    push_bpc = fmt_bpc(8 * d["pushes"], C)
    pop_bpc = fmt_bpc(8 * d["pops"], C)
    busy = d["busy"]
    supply_cyc = busy - d["stallfull"]
    supply_bpc = fmt_bpc(8 * d["pushes"], supply_cyc) if supply_cyc else 0.0
    bubbles = busy - d["pushes"] - d["stallfull"]
    ghz = a.shell_mhz * 1e6 / 1e9

    out = {
        "label": a.label, "window": win, "shell_cycles": C,
        "tile_cycles": d["tedges"],      # tedges counts clk_tile POSEDGES
        "theoretical_axi_Bpc": 64.0, "theoretical_afifo_Bpc": 8.0,
        "ddr_side_Bpc": ddr_bpc, "fifo_push_Bpc": push_bpc,
        "fifo_pop_Bpc": pop_bpc,
        "supply_Bpc_while_not_full": supply_bpc,
        "achieved_vs_afifo_bound": push_bpc / 8.0,
        "supply_vs_afifo_bound": supply_bpc / 8.0,
        "GBps_at_shell": {"ddr_side": ddr_bpc * ghz,
                          "fifo_push": push_bpc * ghz,
                          "supply_capability": supply_bpc * ghz,
                          "afifo_bound": 8.0 * ghz},
        "busy_cycles": busy,
        "busy_pct_of_window": 100.0 * busy / C,
        "busy_decomposition": {
            "push_cycles": d["pushes"], "stall_fifo_full": d["stallfull"],
            "other_bubbles_ar_lat_unpack": bubbles},
        "bursts": {"n": d["arhs"], "words": d["arwords"],
                   "avg_words": d["arwords"] / d["arhs"] if d["arhs"] else 0,
                   "interburst_gap_avg":
                       d["burstgapsum"] / d["burstgapn"]
                       if d["burstgapn"] else 0,
                   "interburst_gap_n": d["burstgapn"]},
        "records": {"n": d["recs"],
                    "framing_gap_avg":
                        d["recgapsum"] / d["recgapn"] if d["recgapn"] else 0,
                    "framing_gap_max_cum": d["recgapmax_end"],
                    "framing_gap_n": d["recgapn"],
                    "ack_to_reqtgl_avg":
                        d["reqtglsum"] / d["reqtgln"] if d["reqtgln"] else 0,
                    "walker_wf_handshakes": d["wfhs"]},
        "r_beats": d["rbeats"],
        "occupancy": {"mean": d["occsum"] / C, "max_cum": d["occmax_end"],
                      "end": d["occ_end"],
                      "hist_pct": {HIST_BUCKETS[i]:
                                   100.0 * d["hist"][i] / C
                                   for i in range(10)}},
        "consistency": {"pushes_eq_8x_rbeats": d["pushes"] == 8 * d["rbeats"],
                        "words_eq_rbeats": d["arwords"] == d["rbeats"]},
    }

    lab = f" [{a.label}]" if a.label else ""
    print(f"INGEST REPORT{lab}  ({win})  — SIM-measured, behavioral twin")
    print(f"  window: {C} shell cycles ({d['tedges']} tile cycles at "
          f"div={a.tile_div})")
    print(f"  DDR-side       : {ddr_bpc:6.3f} B/cyc  "
          f"({ddr_bpc*ghz:6.3f} GB/s @ {a.shell_mhz:.0f} MHz)   "
          f"[theoretical AXI 64 B/cyc]")
    print(f"  afifo push side: {push_bpc:6.3f} B/cyc  "
          f"({push_bpc*ghz:6.3f} GB/s)   [afifo bound 8 B/cyc = "
          f"{8*ghz:.3f} GB/s]")
    print(f"  afifo pop side : {pop_bpc:6.3f} B/cyc  ({pop_bpc*ghz:6.3f} "
          f"GB/s) — tile-demand-limited")
    print(f"  SUPPLY capability while fifo not full: {supply_bpc:6.3f} B/cyc "
          f"({supply_bpc*ghz:6.3f} GB/s) = {100*supply_bpc/8:5.1f}% of the "
          f"afifo bound")
    print(f"  reader busy: {busy} cyc ({100.0*busy/C:5.1f}% of window) = "
          f"{d['pushes']} push + {d['stallfull']} fifo-full stall + "
          f"{bubbles} bubble (AR/latency/unpack)")
    print(f"  bursts: {d['arhs']} (avg {out['bursts']['avg_words']:.2f} "
          f"words/burst), inter-burst gap avg "
          f"{out['bursts']['interburst_gap_avg']:.1f} cyc over "
          f"{d['burstgapn']} gaps")
    print(f"  records: {d['recs']}, framing gap (ack->next AR) avg "
          f"{out['records']['framing_gap_avg']:.1f} cyc over {d['recgapn']} "
          f"gaps, max(cum) {d['recgapmax_end']}")
    print(f"    of which ack->req-toggle (walker issue + ctl engage + CDC): "
          f"avg {out['records']['ack_to_reqtgl_avg']:.1f} cyc; walker wf "
          f"handshakes seen: {d['wfhs']}")
    print(f"  occupancy: mean {out['occupancy']['mean']:.1f} / 512, "
          f"max(cum) {d['occmax_end']}, end {d['occ_end']}")
    hp = out["occupancy"]["hist_pct"]
    print("  occupancy distribution (% of window cycles): "
          + "  ".join(f"{k}:{v:.1f}" for k, v in hp.items() if v >= 0.05))
    ck = out["consistency"]
    print(f"  consistency: pushes==8*rbeats {ck['pushes_eq_8x_rbeats']}, "
          f"arwords==rbeats {ck['words_eq_rbeats']}")
    if recs:
        print("  per-record (cumulative at ack):")
        prev_c, prev_w, prev_p = None, 0, 0
        for n, c, w, pu in recs:
            span = "" if prev_c is None else f" span={c-prev_c}"
            print(f"    rec {n}: cyc={c} words+={w-prev_w} "
                  f"pushes+={pu-prev_p}{span}")
            prev_c, prev_w, prev_p = c, w, pu
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
        print(f"  json -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
