#!/usr/bin/env python3
"""mutate.py — checker-credibility mutations for the B1 walker suite.

House pattern (verif/seq/mutate.py): mutate a SCRATCH COPY of the RTL, never
the tree; the suite must FAIL on every mutant or the checkers are decorative.

Each mutant is paired with the vector set that can see it — that pairing is
the point, not an accident:
  m1_kvskip   : any set. Record-walk order corruption.
  m2_narrow   : d128 ONLY. Dropping the RNE rounding is invisible at D=64,
                where the composite is exact at every width (B1_WALKER.md §2
                P3), so a suite without a D=128 set reports a FALSE GREEN.
  m3_pvorder  : split ONLY. pv's EMIT->DESC->EMIT becomes EMIT->EMIT->DESC —
                a real deadlock in the tile (stage_buf job_ready vs MXE
                S_INGEST), exercised by just 3 of the 24 CQ-8 cases.
  m4_fmtskip  : refuse ONLY. Disables the D-029 v1.1 fmt-nibble refusal
                (IB_WALK.md §2.1) — the pre-hardening behavior, where a
                future-format descriptor is silently mis-walked as v1. The
                condition keeps reading d.fmt (compares against an
                unreachable value) so the mutant build stays -Wall clean.

Stage-2 mutants (tb_walker2_sb, the fmt=1 sequencer):
  m5_gqamap   : l7b ONLY. Group advance '>=' -> '>' — heads at exact group
                boundaries stay on the previous engine. Only a case with
                H_kv > 1 can see it (the two tile-scale cases have one
                group), and 7B's group=7 is exactly the non-pow-2 geometry
                the amended R3 ruling is about.
  m6_accskip  : l7b ONLY. pj_first never clears — accumulate reads 0 on
                every k-split. Invisible at tile scale where K=D_model=128
                fits one job; only the 7B K=3584/18944 splits exercise the
                accumulate chain.
  m7_rqslot   : per-head RQ table index off by one — head h loads head
                h+1's PV requant pair.
  m8_fetchbase: TENS table index off by one — every fetch presents the NEXT
                tensor's DDR base.

Check-equivalence mutants (tb_check_sb, mutation gate 3):
  m9_resvskip : walk_desc2_check stops seeing the GEOM0 reserved bits —
                a fmt=1 descriptor with garbage in [27:24] would load
                cleanly and the wire format could silently grow. Killed by
                the directed "geom resv bit" row (and its random cousins).

Stage-5 mutants (tb_walker2_sb, the §3b full-layer steps):
  m10_jcorder : JOBC comp_b re-emits slot a — the §3b alternating protocol
                broken (swiglu would consume gate·gate). Killable only
                because the generator recipe differentiates s_wu from s_wg.
  m11_ropepos : the ROPE level arms one position late — LCTL word mismatch
                (the C-ROPE per-position phase row would be wrong).
  m12_waddr   : the store master writes READ_ADDR instead of WRITE_ADDR —
                a store that silently LAUNCHES READS; caught as a KVW-vs-
                KVWA scoreboard mismatch at the first store write.

D-029 erratum mutant (I-B, tb_walker2_sb + the MXE-legality check):
  m13_nmxe    : l7b. WALK2_N_MXE reverted to the descriptor FIELD width —
                the landed defect verbatim, where every 7B projection asks
                an 8-wide array for n=4095 and mxe_ctrl refuses the job.
                SIGNATURE-REQUIRED ("MXE-ILLEGAL DESC"): the scoreboard also
                trips here, but the scoreboard is the oracle that was BLIND
                to this class, so only the independent check counts as a
                kill. Its companion lives on the spec side — `make mutants4`
                moves the same bound in seq_walker_fmt and requires
                gen_layer_trace to hard-error, covering the case where BOTH
                sides are wrong together and the scoreboard sees nothing.

Usage:
  mutate.py <mutant_name> <src rtl/ dir> <dst dir>
      write the patched scratch RTL copy (unchanged house behavior)
  mutate.py check <suite> <mutant_name> <runlog>
      classify a mutant run (house shape: verif/top/l3/mutate.py check()).
      suite in {walk, walk2, comp, chk}. Exit 0 iff the run FAILED with the
      mutant's expected signature. A PASS banner, an INFRA failure (cannot
      open vectors / missing +vectors / watchdog), a run with neither
      banner (crash), or a failure WITHOUT the expected signature are all
      gate FAILURES — "the suite went red" is not evidence the checker
      that should catch this mutant exists.
"""
import shutil
import sys
from pathlib import Path

FILES = [
    "apex_pkg.sv",
    "kvq/cores/cq_fp_pkg.sv",
    "mxe/mxe_cfg_pkg.sv",
    "seq/seq_walker_pkg.sv",
    "seq/seq_walker_comp.sv",
    "seq/seq_layer_walker.sv",
    "seq/seq_layer_walker2.sv",
]

MUTS = {
    "m1_kvskip": (
        "seq/seq_layer_walker.sv",
        "              kv_addr  <= WALK_SC_AW'({1'b0, tk} + 9'd1);",
        "              kv_addr  <= WALK_SC_AW'({1'b0, tk} + 9'd2);",
    ),
    # P4 attack: use the f32-GRADE constant instead of the f64 significand.
    # This is the documented failure mode (86,592 of 419,629 reachable product
    # significands mismatch) and it is invisible at D=64, where the constant is
    # exactly 2^7 and the composite is exact at every width.
    "m2_narrow": (
        "seq/seq_walker_comp.sv",
        "  localparam logic [52:0] CM_D128 = 53'h16A09E667F3BCC;",
        "  localparam logic [52:0] CM_D128 = 53'h16A09E60000000;",
    ),
    "m3_pvorder": (
        "seq/seq_layer_walker.sv",
        "          W_P_AJHD: if (aj_ready) state <= W_P_DESC;",
        "          W_P_AJHD: if (aj_ready) state <= split ? W_P_AJTL : W_P_DESC;",
    ),
    "m4_fmtskip": (
        "seq/seq_walker_pkg.sv",
        "      if (d.fmt != WALK_FMT_V1)                    walk_desc_check = WALK_ERR_DESC;",
        "      if (d.fmt == 4'hF)                           walk_desc_check = WALK_ERR_DESC;",
    ),
    "m5_gqamap": (
        "seq/seq_layer_walker2.sv",
        "                    >= 16'({8'b0, d2.n_heads})) begin",
        "                    > 16'({8'b0, d2.n_heads})) begin",
    ),
    "m6_accskip": (
        "seq/seq_layer_walker2.sv",
        "              pj_first <= 1'b0;",
        "              pj_first <= 1'b1;",
    ),
    "m7_rqslot": (
        "seq/seq_layer_walker2.sv",
        "            h_desc[WALK_DW_RQ]   <= desc2_word[W2_RQ0 + {24'b0, h_idx}];",
        "            h_desc[WALK_DW_RQ]   <= desc2_word[W2_RQ0 + {24'b0, h_idx} + 32'd1];",
    ),
    "m8_fetchbase": (
        "seq/seq_layer_walker2.sv",
        "  assign wf_req.base64  = desc2_word[W2_TENS0 + {28'b0, pc_tens}]",
        "  assign wf_req.base64  = desc2_word[W2_TENS0 + {28'b0, pc_tens} + 32'd1]",
    ),
    "m9_resvskip": (
        "seq/seq_walker_pkg.sv",
        "      resv_nz = |{w_geom0[27:24], w_geom0[19:18], w_geom0[15:8],",
        "      resv_nz = |{4'b0, w_geom0[19:18], w_geom0[15:8],",
    ),
    "m10_jcorder": (
        "seq/seq_layer_walker2.sv",
        # anchor updated for the FFN interleave (2026-08-08): pc_ucols is
        # ffn_cw now and sits on its own line; the mutation is the same
        # jc-order corruption (UP composite loaded into the gate slot).
        "                      pc_jcb = 2'(WALK2_JC_UP);",
        "                      pc_jcb = 2'(WALK2_JC_GATE);",
    ),
    "m11_ropepos": (
        "seq/seq_layer_walker2.sv",
        "            lw_rope_pos  <= d2.pos_m[6:0];",
        "            lw_rope_pos  <= d2.pos_m[6:0] + 7'd1;",
    ),
    "m12_waddr": (
        "seq/seq_layer_walker2.sv",
        "  assign kvm_awaddr  = in_store ? KV_WADDR : h_aw_a;",
        "  assign kvm_awaddr  = in_store ? (KV_WADDR ^ 8'h04) : h_aw_a;",
    ),
    # E-3b mutant — the QSTAGE exit skips the fsrc_ext restore, leaving the
    # feeder input mux on the q sink: the next LCTL emission the spec
    # expects (the touch-one-field restore word) never appears, and the
    # first attention/store traffic would stream K-hat rows into a dead
    # seam on the tile. Caught by the qstage case's LCTL scoreboard.
    "m14_qsrestore": (
        "seq/seq_layer_walker2.sv",
        "              lw_fsrc_ext <= 3'd0;",
        "              lw_fsrc_ext <= 3'd3;",
    ),
    # E-4b mutant — the NFEED entry arms the owned l_nsrc 0 instead of 1:
    # the code-4 route flips but the norm x-mux never follows, so on the
    # tile the fed row would stream into the act/ws demux and S2_NFW would
    # wait on an nf_busy that never falls — the exact wedge the E-4b bit
    # exists to prevent, minus the loud refusal. Caught by the qstage-128
    # case's LCTL scoreboard: the NFEED level word must carry bit 19.
    "m15_nsrcarm": (
        "seq/seq_layer_walker2.sv",
        "            if (d2.en_mask[W2_EN_NSRC]) lw_nsrc <= 1'b1;",
        "            if (d2.en_mask[W2_EN_NSRC]) lw_nsrc <= 1'b0;",
    ),
    # D-029 ERRATUM (I-B) mutant — reverts the landed defect verbatim: the
    # MXE n-split re-sized by the 12-bit descriptor FIELD instead of the
    # implemented array width. SIGNATURE-REQUIRED: the Makefile rule demands
    # the "MXE-ILLEGAL DESC" line, because a plain scoreboard mismatch is NOT
    # evidence that the new check works — the scoreboard also fails here, and
    # it is the scoreboard that was blind to this class in the first place.
    "m13_nmxe": (
        "seq/seq_walker_pkg.sv",
        "  localparam int unsigned WALK2_N_MXE = MXE_N;            // 8     — mxe n_dim",
        "  localparam int unsigned WALK2_N_MXE = (1 << DIM_W) - 1; // 8     — mxe n_dim",
    ),
}


# ── mutant-run classification (mutation gates 1/2/3) ────────────────────────
# House shape: verif/top/l3/mutate.py check(). Each TB prints exactly one of
# "<SUITE> PASS" / "<SUITE> FAIL" iff its vector loop actually ran; the
# early-$finish paths ($fopen failure, missing +vectors) print neither.
SUITES = {
    "walk":  ("WALKER PASS",  "WALKER FAIL"),
    "walk2": ("WALKER2 PASS", "WALKER2 FAIL"),
    "comp":  ("COMP PASS",    "COMP FAIL"),
    "chk":   ("CHECKEQ PASS", "CHECKEQ FAIL"),
}

# Infra failures: the run went red WITHOUT evaluating the mutant (no vectors,
# no +vectors plusarg, or a hang). These are $finish-exit-0 paths, so the old
# "non-PASS == caught" logic counted them as kills. They are gate FAILURES.
# HARD infra: the harness never started the run properly, so NOTHING in the
# log is evidence about the mutant. These fail the gate unconditionally.
INFRA_HARD = [
    "FAIL: cannot open",
    "FAIL: +vectors=<file> required",
]

# SOFT infra: a hang. This is infra ONLY when the mutant's own signature is
# absent — a mutant can legitimately CAUSE a hang (l3's M3 is a hang kill),
# and discarding a run that carries the declared signature turns a genuine
# kill into a gate failure. Ordering matters: signature is checked first.
INFRA_SOFT = [
    "walk did not complete (watchdog)",
]

# Expected kill signature per (suite, mutant): substrings of the checker
# lines that CAN see this mutant (any hit counts, l3-style). The generic
# scoreboard-mismatch line is  "... check %0d: expected '%s' got '%s'"  —
# matched by ": expected '" (the refusal line uses "expected code=", the
# RESULT summary has no colon-expected, so the substring is unambiguous).
SIGS = {
    # gate 1: tb_walker_sb
    ("walk", "m1_kvskip"): [": expected '", "emission past end of reference",
                            "composite err_stale", "composite err_frame"],
    ("walk", "m3_pvorder"): [": expected '",
                             "emission past end of reference"],
    ("walk", "m2_narrow"): [": expected '"],
    ("walk", "m4_fmtskip"): ["refusal expected code=",
                             "emission past end of reference"],
    # gate 1: the exhaustive composite sweep (tb_comp_sb value checks)
    ("comp", "m2_narrow"): ["FAIL cs", "FAIL qs"],
    # gate 2: tb_walker2_sb
    ("walk2", "m5_gqamap"): ["on engine", ": expected '"],
    ("walk2", "m6_accskip"): [": expected '"],
    ("walk2", "m7_rqslot"): [": expected '"],
    ("walk2", "m8_fetchbase"): [": expected '"],
    ("walk2", "m10_jcorder"): [": expected '"],
    ("walk2", "m11_ropepos"): [": expected '"],
    ("walk2", "m14_qsrestore"): [": expected '",
                                 "reference emissions never produced"],
    ("walk2", "m15_nsrcarm"): [": expected '",
                               "reference emissions never produced"],
    ("walk2", "m12_waddr"): [": expected '",
                             "reference emissions never produced",
                             "READ_ADDR write with no preceding idle poll",
                             "AXIL write to"],
    # gate 3: tb_check_sb SV-vs-mirror rows
    ("chk", "m4_fmtskip"): ["FAIL C1", "FAIL C2"],
    ("chk", "m9_resvskip"): ["FAIL C1", "FAIL C2"],
}


def check(suite: str, name: str, runlog: str) -> int:
    if suite not in SUITES:
        print(f"{name}: unknown suite {suite!r}. Gate FAIL.")
        return 1
    if (suite, name) not in SIGS:
        print(f"{name}: no expected signature registered for suite "
              f"{suite!r} — add one to mutate.py SIGS. Gate FAIL.")
        return 1
    pass_banner, fail_banner = SUITES[suite]
    log = Path(runlog)
    if not log.is_file():
        print(f"{name}: run log {runlog} missing. Gate FAIL.")
        return 1
    txt = log.read_text(errors="replace")
    if pass_banner in txt:
        print(f"{name}: NOT CAUGHT — the mutant run PASSED. Gate FAIL.")
        return 1
    hard = [s for s in INFRA_HARD if s in txt]
    if hard:
        print(f"{name}: INFRA FAILURE {hard} — the mutant was never "
              f"evaluated; a red run proves nothing here. Gate FAIL.")
        return 1
    soft = [s for s in INFRA_SOFT if s in txt]
    sig_hits = [g for g in SIGS[(suite, name)] if g in txt]
    if soft and not sig_hits:
        print(f"{name}: INFRA FAILURE {soft} — hung WITHOUT the mutant's "
              f"signature; a red run proves nothing here. Gate FAIL.")
        return 1
    if fail_banner not in txt:
        print(f"{name}: run neither passed nor failed?! "
              f"(no '{pass_banner}' / '{fail_banner}' banner). Gate FAIL.")
        return 1
    sig = SIGS[(suite, name)]
    hit = [s for s in sig if s in txt]
    if not hit:
        print(f"{name}: FAILED but without the expected signature "
              f"{sig} — inspect. Gate FAIL.")
        return 1
    print(f"MUTANT {name} [{suite}] CAUGHT by the expected signature "
          f"{hit[0]!r}:")
    for line in txt.splitlines():
        if hit[0] in line:
            print(f"  {line.strip()}")
            break
    return 0


def main():
    if sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3], sys.argv[4]))
    name, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    mfile, old, new = MUTS[name]
    if dst.exists():
        shutil.rmtree(dst)
    for f in FILES:
        d = dst / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f, d)
    p = dst / mfile
    text = p.read_text()
    assert old in text, f"mutation anchor not found in {mfile}: {old!r}"
    p.write_text(text.replace(old, new, 1))
    print(f"MUTANT {name}: {mfile} patched")


if __name__ == "__main__":
    main()
