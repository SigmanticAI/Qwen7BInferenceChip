#!/usr/bin/env python3
"""mutate_qpath.py — I-C IC-QPATH mutation gate: prove the q-path suite
CATCHES bugs in the NEW routes it introduces. The RTL is NEVER edited —
mutants are patched COPIES under build/ (the l3/mutate.py discipline).

A mutant is CAUGHT only on its OWN failure signature. "Some non-zero exit"
is not a catch: a mutant that hangs the tile would otherwise look identical
to a mutant that corrupts a value, and the gate would not be distinguishing
what it claims to.

  Q1 sink-bypass  : the q sink feeds the feeder from the PRE-rope seam
                    (sqf_data) instead of rope_row's output — the exact shape
                    of gap A as it stood. Caught by the post-RoPE tap.
  Q2 sink-leak    : kv_s_tvalid is NOT forced low in q_sink mode, so the q
                    row also reaches the KVQ write port and the store-time
                    scale snoop, committing a record that is not a KV record.
                    Caught ONLY by the post-staging occupancy poll (KVP 0x24
                    == 0) — this mutant survived the first cut of the case
                    and is why that check exists. Its signature is pinned to
                    that poll specifically: a tap/EFS failure would NOT be a
                    catch, because the leak's whole point is that the q path's
                    own values stay correct while the KV cache is corrupted.
  Q3 sink-narrow  : the widening of the seam beat into the feeder truncates
                    the fp16 mantissa's low bit before the C-1 quant — a
                    value-only corruption downstream of the tap, so it must
                    be caught by the FEEDER SCALE, not by the tap.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "rtl/top/apex_top.sv"   # default mutant source

MUTS = {
    "Q1": {
        "old": "  assign seam_data   = l_rope_en_q ? rr_m_data  : sqf_data;",
        "new": ("  assign seam_data   = q_sink ? sqf_data\n"
                "                     : (l_rope_en_q ? rr_m_data : sqf_data);"
                "  // MUTANT Q1"),
        "sig": ["tap f16"],
        "kind": "value",
    },
    "Q2": {
        "old": "  assign kv_s_tvalid = seam_valid && !q_sink;",
        "new": "  assign kv_s_tvalid = seam_valid;  // MUTANT Q2",
        "sig": ["KVP 24 timeout"],
        "kind": "value",
    },
    "Q4": {
        # the act-stage LOAD base row (I-C): per-head q staging puts head h's
        # codes in bank-1 row h. Forcing the base back to 0 is the pre-I-C
        # behaviour and makes every head read the LAST staged row — the exact
        # defect the per-head readback GEMM exists to catch.
        # Written as a pure PLACEMENT fault: the job still consumes exactly
        # its beats and completes, but every row lands at base 0 (the pre-I-C
        # behaviour). Resetting row_c instead would merely HANG the job, and a
        # hang would not demonstrate that the readback checks WHERE codes go.
        "old": "            mem[midx(bank_q, row_c, beat_c)] <= li_beat.data;",
        "new": ("            mem[midx(bank_q, row_c - sel_q, beat_c)]"
                " <= li_beat.data;  // MUTANT Q4"),
        "sig": ["ERO"],
        "kind": "value",
        "src": "rtl/top/glue/apex_stage_buf.sv",
    },
    "Q3": {
        "old": "        fq_in_data  = f16_to_f32_bits(seam_data);",
        "new": ("        fq_in_data  = f16_to_f32_bits(seam_data & 16'hFFFE);"
                "  // MUTANT Q3"),
        "sig": ["EFS"],
        "kind": "value",
    },
    # ── I-C IC-MHWALK: the three routes the MULTI-HEAD walk adds ────────────
    # Gated by `make mh_mutate_walk`, against the WALKED 4-head arm. Each
    # signature is MEASURED from the mutant's own run, never predicted, and
    # each is a DIFFERENT observer — so the three together show the walked
    # gate discriminates the head-boundary fence, the per-head activation row
    # and the per-head KV-group select from one another, not merely "red".
    "X1": {
        # THE DEFECT THIS LANE FIXED. Clearing pv_route unconditionally at the
        # kick flips rt_res_dst 0 -> 2 while the PREVIOUS head's last PV beat
        # is still in the MXE, so that beat lands in the score dequant.
        "old": ("              if (tile_idle) begin\n"
                "                pv_route <= 1'b0;\n"
                "                state    <= d.en_score ? W_S_FENCE"
                " : W_P_FENCE;\n"
                "              end else begin\n"
                "                state    <= W_DRAIN;\n"
                "              end"),
        "new": ("              pv_route <= 1'b0;             // MUTANT X1\n"
                "              state    <= d.en_score ? W_S_FENCE"
                " : W_P_FENCE;"),
        # MEASURED, and pinned to the VALUE, not merely to the observer: this
        # mutant reproduces the reported I-C defect character for character —
        # head 0 bit-exact, head 1 dead on its FIRST score beat.
        "sig": ["[tap score] idx 20: got 00000010 exp 00002b49"],
        "kind": "head-boundary route fence",
        "src": "rtl/seq/seq_layer_walker.sv",
    },
    "X2": {
        # the per-head q row select (F6(ii) staging): pinned to 0, every head
        # after the first re-emits HEAD 0's q8 against its own K.
        # written so Q_ROWS and h_idx both stay READ (a bare `= 5'd0` trips
        # UNUSEDPARAM and the mutant would not even compile — a mutant that
        # cannot be built proves nothing).
        "old": "  assign h_q_row = (Q_ROWS > 1) ? 5'(h_idx) : 5'd0;",
        "new": ("  assign h_q_row = (Q_ROWS > 1) ? 5'd0 : 5'(h_idx);"
                "  // MUTANT X2"),
        # MEASURED and value-pinned. X2 lands on the SAME observer as X1 (the
        # score tap, head 1 beat 0) but a DIFFERENT value — 0x141d is head 1's
        # K against head 0's q, where X1's 0x10 is a stray PV lane. Pinning the
        # value is what makes the two mutually exclusive: X1's signature does
        # not appear in X2's log and vice versa, so neither can pass on the
        # other's failure.
        "sig": ["[tap score] idx 20: got 0000141d exp 00002b49"],
        "kind": "per-head activation row",
        "src": "rtl/seq/seq_layer_walker2.sv",
    },
    "X3": {
        # the per-head KV-group select (F6(i) bank + R5 single ingress):
        # pinned to engine 0, every head reads KV GROUP 0's records and
        # composites while its own s_q replay still lands where it belongs.
        "old": "  assign kv_eng_sel = in_store ? sk_g : g_idx;",
        "new": "  assign kv_eng_sel = in_store ? sk_g : '0;  // MUTANT X3",
        # MEASURED: this one is caught EARLIER and by a DIFFERENT observer than
        # X1/X2 — the feeder scale of the very first K record head 1 reads is
        # group 0's, so the walked EFS stream (added by this lane) is what
        # bites, before any score beat exists to be wrong.
        "sig": ["[EFS] got 35ae/0 exp 2bdd/0"],
        "kind": "per-head KV-group select",
        "src": "rtl/seq/seq_layer_walker2.sv",
    },
}


def patch(m: str, outdir: str) -> int:
    mut = MUTS[m]
    srcp = REPO / mut.get("src", "rtl/top/apex_top.sv")
    src = srcp.read_text()
    n = src.count(mut["old"])
    assert n == 1, f"{m}: anchor found {n} times (expected exactly 1)"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / srcp.name).write_text(src.replace(mut["old"], mut["new"]))
    print(f"{m}: patched copy written to {out}/{srcp.name}")
    return 0


def check(m: str, runlog: str) -> int:
    mut = MUTS[m]
    txt = Path(runlog).read_text()
    if "L3 PASS" in txt:
        print(f"{m}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    fatal = ("%Fatal" in txt) or ("L3 FAIL" in txt) or ("%Error" in txt)
    if not fatal:
        print(f"{m}: run neither passed nor failed?! Gate FAIL.")
        return 1
    hit = [s for s in mut["sig"] if s in txt]
    if not hit:
        print(f"{m}: FAILED, but on NO expected signature {mut['sig']} — "
              f"not a catch. Gate FAIL.")
        return 1
    print(f"{m}: CAUGHT ({mut['kind']}, signature {hit})")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, m = sys.argv[1], sys.argv[2]
    if m not in MUTS:
        print(f"unknown mutant {m}")
        return 2
    if mode == "patch":
        return patch(m, sys.argv[3])
    if mode == "check":
        return check(m, sys.argv[3])
    print(f"unknown mode {mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
