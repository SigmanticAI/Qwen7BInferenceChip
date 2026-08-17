#!/usr/bin/env python3
"""gen_status.py — mechanically generate STATUS.md from the ACTUAL suite outputs.

ARCHITECTURE.md §8 process rule: "a STATUS.md is updated only from fresh
full-suite runs (anti-fabrication: report generation is mechanically tied to
the last run's parsed results — never hand-written)."

This script is the only legal writer of STATUS.md. It:
  1. runs the golden gate LIVE (`make -C golden test`) and parses its output;
  2. parses the persisted run/coverage/mutation logs of every HDL suite
     (it never invents a verdict: missing logs => NO-RUN, failed pattern
     => FAIL, and the quoted evidence is the verbatim matched line);
  3. extracts the D-022 tier-quality table from the live golden run;
  4. counts LOC from the working tree;
  5. lifts open items mechanically from TRACEABILITY.md (PARTIAL/UNTESTED
     matrix rows + the known-limitations register IDs) and from any suite
     that is not PASS.

Usage:  python scripts/gen_status.py [--no-golden]
        (--no-golden skips the live golden run and marks it NO-RUN)
Exit:   0 always if STATUS.md was written; the VERDICT column carries truth.
"""

import argparse
import datetime
import glob as globmod
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(REPO, "STATUS.md")
PYTHON = sys.executable


# ──────────────────────────────────────────────────────────────────────────────
# suite spec: (name, description, base_dir, checks)
# check = (glob_rel_to_base, regex, all_files_must_match)
#   - glob matches no file            -> suite NO-RUN (that check "missing")
#   - all_files_must_match=True       -> every matched file must contain regex
#   - otherwise                       -> at least one matched file must
# evidence shown = first matching line found (verbatim, truncated)
# ──────────────────────────────────────────────────────────────────────────────
SUITES = [
    ("top/smoke", "apex_top tile smoke (D=64, D=128 AND CQ-4+ tier-bank full chains, T=8 + D-020 soft-reset rerun each; F-2 grouped keys + D-008 FLUSH + mask in the cq4p run)",
     "verif/top/smoke", [
        ("build/run_smoke.log", r"TOPSMOKE RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_smoke.log", r"TOPSMOKE PASS", False),
        ("build/run_smoke_d128.log", r"TOPSMOKE RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_smoke_d128.log", r"TOPSMOKE PASS", False),
        ("build/run_smoke_cq4p.log", r"TOPSMOKE RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_smoke_cq4p.log", r"TOPSMOKE PASS", False),
     ]),
    ("top/l2", "Layer-2 pairwise chains a/b/c (all tiers, D=64+128, FLUSH, resets, storms)",
     "verif/top/l2", [
        ("build/coverage.txt", r"L2 COVERAGE GATE: PASS", False),
        ("build/coverage.txt", r"total checks across required runs: \d+", False),
     ]),
    ("top/l3", "Layer-3 attention replays through the REAL apex_top (CQ-8/CQ-4+/TIP-auto mixed tiers, D=64+128, T<=128, incl. the D-027 CSR-mask d128 CQ-4+ case) + tile mutation gate + F-5 stage-buf unit regression",
     "verif/top/l3", [
        ("build/run_all.txt", r"L3 RUNS: all \d+ cases passed", False),
        ("build/run_unit_stagebuf.log", r"STAGEBUF PATD PASS", False),
        ("build/coverage.txt", r"total checks: \d+", False),
        ("build/coverage.txt", r"L3 COVERAGE GATE: PASS", False),
        ("build/mutation.txt", r"M1: CAUGHT", False),
        ("build/mutation.txt", r"M2: CAUGHT", False),
        ("build/mutation.txt", r"M3: CAUGHT", False),
     ]),
    ("l3/fail-first", "F-register regression-first evidence: durable pre-fix kept-failing logs (F-1/F-2/F-3/F-5a) + F-5b unit mutant kill (verif/top/l3/logs — survives make clean; the kept-PASSING flips run inside top/l3 'all')",
     "verif/top/l3/logs", [
        ("prefix_f1_keptfail_before_fix.log", r"drive_cs stall", False),
        ("prefix_f2_keptfail_before_fix.log", r"KVP 0c timeout", False),
        ("prefix_f3_keptfail_before_fix.log", r"\[ESTK\] got 00000001 exp 00000000", False),
        ("prefix_f5a_keptfail_before_fix.log", r"drive_g stall", False),
        ("run_unit_stagebuf_mutant.log", r"\[PAT_D sel=8 beat=0\]", False),
     ]),
    ("top/l4", "IB-LAYER L4 harness: level-propagation probe (every exported LAYER level observed at its consumer, hierarchically) + the segment-2 host-composition suite (RMSNorm-1 front-half through the REAL apex_top at CFG_D=128) + 3-mutant signature gate",
     "verif/top/l4", [
        ("build/run_levels.log", r"L4LEVELS RESULT: checks=\d+ errors=0 -> PASS", False),
        ("build/run_compose.log", r"L4COMPOSE RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_compose.log", r"L4COMPOSE PASS", False),
        # run_mut_*.log are INTENTIONAL-failure mutation runs (l3 discipline);
        # mutation.txt carries the signature-checked verdicts
        ("build/mutation.txt", r"mC1: CAUGHT", False),
        ("build/mutation.txt", r"mC2: CAUGHT", False),
        ("build/mutation.txt", r"mC3: CAUGHT", False),
     ]),
    ("top/wcomp", "F6(i) per-KV-head composite scale-cache BANK (apex_wcomp_bank: N_ENG=4 seq_walker_comp instances behind eng_sel) — routing + isolation proof, count-pinned 203 checks at D=64 AND D=128, 4 signature-required mutants (arithmetic itself is seq_walker's comp sweep; tile elaboration at GQANE=4 is kvq/gqa's lint_top_on)",
     "verif/top/wcomp", [
        ("build/run_d64.log", r"WCOMP BANK d64: checks=203 fails=0", False),
        ("build/run_d128.log", r"WCOMP BANK d128: checks=203 fails=0", False),
        ("build/mutation.txt", r"mW1: CAUGHT", False),
        ("build/mutation.txt", r"mW2: CAUGHT", False),
        ("build/mutation.txt", r"mW3: CAUGHT", False),
        ("build/mutation.txt", r"mW4: CAUGHT", False),
        ("build/gate.txt", r"WCOMP SUITE: ALL PASS \(d64 \+ d128, 203 checks each, 4/4 signature mutants caught\)", False),
     ]),
    ("top/bias", "IC-BIAS projection-bias seam (S-2 gap B): apex_proj_bias unit golden gate (element model, exact-or-refused, bias=+0 equivalence vs apex_scale_quant) + l4_bias q/K/V biased projections bit-exact vs decoder_layer_fx BUS_ON through the real tile + 4-mutant signature gate",
     "verif/top/bias", [
        ("build/run_unit.log", r"PBIAS RESULT: cycles=\d+ jobs=\d+ checks=\d+ errors=0", False),
        ("build/run_unit.log", r"PBIAS PASS", False),
        ("build/run_tile.log", r"BIASTILE RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_tile.log", r"BIASTILE PASS", False),
        ("build/mutation.txt", r"mB1: CAUGHT", False),
        ("build/mutation.txt", r"mB2: CAUGHT", False),
        ("build/mutation.txt", r"mB3: CAUGHT", False),
        ("build/mutation.txt", r"mB4: CAUGHT", False),
     ]),
    ("top/wg3", "W-G3 flagship (partial-layer walk + fuel-fed projection, real Qwen2.5-7B s019/L19): case renders (C/P/R/F self-checks), FENCED mask-0x03D H=28 walk at unit level, single-head hd128 tile replay host AND WALKED (fmt=1), GQA-4 K-rope with the D-030 arbiter (legacy-pinned arm kept RED), fuel tile leg + DDR leg (perturbed-DDR arm kept RED), topology A/B witness (control green, pre-F6(i) single-cache arm kept RED). Inputs are the gitignored walkgold stage-A/B artifacts (regenerate per the Makefile header)",
     "verif/top/wg3", [
        ("build/case.log", r"WG3 CASE RENDER: PASS", False),
        ("build/run_fenced.log", r"WG3-TOPO\[control-Neng-caches\] RESULT: cases=1 checks=\d+ errors=0", False),
        ("build/run_fenced.log", r"WG3-TOPO PASS", False),
        ("build/tilecase.log", r"WG3 TILE CASE: PASS", False),
        ("build/run_tilehost.log", r"L3 RESULT: cycles=\d+ checks=\d+ errors=0", False),
        ("build/run_tilehost.log", r"L3 PASS", False),
        ("build/run_tilewalk.log", r"WALKFMT2: fmt=1 WALKED clean", False),
        ("build/run_tilewalk.log", r"L3 PASS", False),
        ("build/kropecase.log", r"WG3 KROPE CASE: PASS", False),
        ("build/run_krope.log", r"L3 PASS", False),
        # F5 discrimination: the legacy-pinned arm must stay RED (kept-failing
        # evidence, the mxe/perf baseline idiom)
        ("build/run_krope_legacy.log", r"Assertion failed in tb_apex_l3: \[tap f16\]", False),
        ("build/fuelcase.log", r"WG3 FUEL CASE: PASS", False),
        ("build/run_fuel.log", r"L3 PASS", False),
        ("build/run_fuel_host.log", r"F2SIM RESULT: files=1 checks=30 fails=0 -> PASS", False),
        ("build/run_fuel_ddr.log", r"F2SIM RESULT: files=1 checks=32 fails=0 -> PASS", False),
        ("build/run_fuel_ddr.log", r"DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS", False),
        # the perturbed-DDR arm must stay RED — loads cleanly, then miscomputes
        ("build/run_fuel_bad.log", r"F2SIM RESULT: files=1 checks=32 fails=1 -> FAIL", False),
        ("build/run_fuel_bad.log", r"DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS", False),
        ("build/run_topoA.log", r"WG3-TOPO PASS", False),
        ("build/run_topoB.log", r"WG3-TOPO\[pre-F6i-single-cache\] RESULT: cases=1 checks=\d+ errors=[1-9]\d*", False),
     ]),
    ("mxe/sb", "MXE independent scoreboard suite (6 regimes, SVA pack, resets)",
     "verif/mxe/sb", [
        ("build/coverage.txt", r"COVERAGE: all required buckets hit", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("mxe/struct", "D-004 systolic structural discriminator (whitebox timing law, glitch hop locality, broadcast-wall mutation check)",
     "verif/mxe/struct", [
        ("build/run_real.log", r"STRUCT PASS", False),
        ("build/mutation.txt", r"MUTATION CHECK PASS", False),
     ]),
    ("mxe/perf", "D-005 load-under-compute perf evidence (overlap>0 + cycle win vs pinned sequential baseline; kept-failing overlap regression on the baseline)",
     "verif/mxe/perf", [
        ("build/run_new.log", r"PERF TB PASS", False),
        ("build/perf_gate.txt", r"PERF GATE: PASS", False),
        # kept-FAILING regression: the pinned sequential baseline must still
        # show zero overlap under +require_overlap=1 (D-005 evidence anchor)
        ("build/run_base_reqov.log", r"PERF OVERLAP MISSING", False),
     ]),
    ("mxe/w4", "B3 W4 nibble-packed weight feeder (wfeed W4 scoreboard: 6 run regimes incl. reset storms, coverage gate, 5-mutant gate)",
     "verif/mxe/w4", [
        ("build/coverage.txt", r"COVERAGE PASS: all required buckets hit", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
        ("build/mutation.txt", r"MUTATION GATE: 5/5 mutants killed", False),
     ]),
    ("mxe/w4b", "D-031 W4B given-scale feeder (pkg vector gate, 6 scoreboard regimes, EXHAUSTIVE 4,063,104-point operand sweep, 5-mutant gate; complete per-gate evidence committed at logs/run_all_full.log)",
     "verif/mxe/w4b", [
        ("build/pkg.log", r"W4B PKG GATE: \d+ vectors, ALL PASS", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
        ("build/sweep.log", r"W4B EXHAUSTIVE SWEEP: \d+ operand points, ALL PASS", False),
        ("build/mutants.txt", r"W4B MUTATION GATE: 5/5 mutants detected", False),
        ("logs/run_all_full.log", r"W4B SUITE: all gates green", False),
     ]),
    ("asu/sb", "ASU independent suite (softmax + rmsnorm, 13 configs, mutation gate)",
     "verif/asu/sb", [
        ("build/coverage.txt", r"COVERAGE: all required buckets hit", False),
        ("build/mutants.txt", r"MUTATION GATE: 4/4 mutants detected", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("asu/wide", "WIDE-D RMSNorm (C-RMSW): bit-exact vs golden rmsnorm_fx_wide at RMS_D_MAX=3584+8192 (MU/S k-sweep 2..28, sum2=2^27 corner, rejects, mid-op resets), params provenance gate, 4-mutant gate",
     "verif/asu/wide", [
        ("build/mutants.txt", r"WIDE MUTATION GATE: 4/4 mutants detected", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("asu/swiglu", "SwiGLU + layer-deq units (gate*silu composite and W4-grade dequant on real decoder-layer tensors: fast/bp/reset(+storm) regimes, 9-mutant gate)",
     "verif/asu/swiglu", [
        ("build/run_swg_*.log", r"SWIGLU RESULT: jobs=\d+ beats=\d+ .* mismatches=0 -> PASS", True),
        ("build/run_deq_*.log", r"DEQ RESULT: jobs=\d+ beats=\d+ .* mismatches=0 -> PASS", True),
        ("build/mutation.txt", r"SWIGLU/DEQ MUTATION GATE: 9/9 mutants detected", False),
     ]),
    ("tip/sb", "TIP independent suite (32 thresholds, N=64/4096, resets, framing)",
     "verif/tip/sb", [
        ("build/coverage.txt", r"COVERAGE: all required buckets hit", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("tip/smoke", "TIP smoke (D-011/D-017): frozen-143 replay, threshold sweeps T=1/5/31, clamp T=0, importance accumulators, V0.3-F-2 framing-guard regression (N=64 + N=4096 builds)",
     "verif/tip/smoke", [
        ("build/run_*.log", r"tiles=\d+ decisions=\d+ frame_err_pulses=\d+ fails=0", True),
        ("build/run_*.log", r"ALL TESTS PASSED", True),
     ]),
    ("kvq/smoke", "KVQ implementer smoke (V0-parity x8 + 4 behavioral regressions; baseline bug-repro halves retired with the third-party checkout — see smoke Makefile header)",
     "verif/kvq/smoke", [
        ("logs/run_p_d*.log", r"KVQ PARITY \[.*\]: PASS", True),
        ("logs/run_p_g128*.log", r"KVQ PARITY \[.*\]: PASS", True),
        ("logs/run_p_stall*.log", r"PASS", True),
        ("logs/run_r_*.log", r"RESULT \[.*\]: PASS", True),
     ]),
    ("kvq/sb", "KVQ independent scoreboard suite (records + readback, soft-reset, B-1..B-4 closed)",
     "verif/kvq/sb", [
        ("build/coverage.txt", r"COVERAGE GATE: PASS", False),
        # run_mut_*.log are INTENTIONAL-failure mutation runs — excluded here;
        # their teeth are asserted by the suite's own gate (build fails if a
        # mutant survives), so STATUS only requires the clean-config runs.
        ("build/run_cq*.log", r"checks=\d+ fails=0", True),
        ("build/run_d64*.log", r"checks=\d+ fails=0", True),
        ("build/run_framing.log", r"checks=\d+ fails=0", True),
        ("build/run_reset.log", r"checks=\d+ fails=0", True),
     ]),
    ("f2/firstlight", "AWS F2 first light: full tile + OCL bridge, Vivado zero-error P&R, timing MET at 250 MHz shell clock (VU47P), AGFI loaded on an f2.6xlarge, BAR0 probe of the verified CSRs ALL PASS (register first light only)",
     "docs/results/f2_firstlight", [
        ("smoke_session.log", r"PROBE: ALL PASS", False),
        ("smoke_session.log", r"agfi-\w+  loaded", False),
     ]),
    ("f2sim", "F2 stage-2 regops executor: the verilated cl_apex driven at its OCL AXI-Lite pins by the SAME regops files the F2 host runner uses — all 18 compiled S8 trace jobs bit-exact at tile_div=5 + behavioral-DDR AXI smoke (default and +ddr_stall). The IB-FUEL RED-mutant gate (control PASS + 3/3 fuel-path mutants RED) is rc-asserted by `make mutants` and not log-persisted. Logs are gitignored and removed by the lane's own clean -> NO-RUN then; rerun `make -C verif/f2sim run behsmoke`",
     "verif/f2sim", [
        ("run_d128_div5.log", r"F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS", False),
        ("beh_default.log", r"DDRBEH RESULT: checks=\d+ fails=0 -> PASS", False),
        ("beh_stall.log", r"DDRBEH RESULT: checks=\d+ fails=0 -> PASS", False),
     ]),
    ("kvq/replay", "S8 real-model RTL replay: Qwen2.5-7B 150-token artifact-trace fp16 K/V rows through the KVQ cores RTL, bit-exact vs the blobs the golden pipeline stored at generation time (durable logs — survive make clean)",
     "verif/kvq/replay", [
        ("logs/gen_artifact.log", r"SELF-CHECK OK: every stored blob field matches a fresh golden compression", False),
        ("logs/run_artifact.log", r"CONFIG artifact: checks=\d+ fails=0", False),
        ("logs/run_artifact.log", r"CORES RESULT \[artifact\]: PASS", False),
     ]),
    ("kvq/fparith", "KVQ synthesizable integer fp-arith (cq_fp_pkg real->int rewrite: exhaustive golden sweeps, seeded errors, sv2v+yosys)",
     "verif/kvq/fparith", [
        ("build/prove.log", r"DATAPATH INTEGER-MODEL PROOF: ALL PASS", False),
        ("build/run.log", r"FPARITH TB: ALL PASS", False),
        # run_mut_*.log are INTENTIONAL-failure seeded-error runs — their
        # teeth are asserted by `make mut` (fails if a mutant escapes)
        ("build/mutation.txt", r"MUTATION GATE: 2/2 seeded errors caught", False),
        ("build/synth.txt", r"SYNTH CHECK: PASS", False),
     ]),
    ("kvq/audit", "KVQ D-020 adversarial re-audit (260 soft resets, -0.0, fix-revert mutants)",
     "verif/kvq/audit", [
        ("build/coverage.txt", r"COVERAGE GATE: PASS", False),
        ("build/run_rst_*.log", r"TB PASS", True),
        ("build/run_negz_*.log", r"TB PASS", True),
     ]),
    ("kvq/mask", "S12/D-027 loadable-mask suite (ROM-vs-CSR record equivalence on golden-packer vectors, popcount/swap faults W1C, soft-reset persistence + ssid continuation, computed mask_valid + port mirror; 2 mutants)",
     "verif/kvq/mask", [
        ("build/run_rom.log", r"MASK GATE mask_rom: checks=\d+ fails=0", False),
        ("build/run_rom.log", r"MASK GATE mask_rom: ALL PASS", False),
        ("build/run_csr.log", r"MASK GATE mask_csr: checks=\d+ fails=0", False),
        ("build/run_csr.log", r"MASK GATE mask_csr: ALL PASS", False),
        # run_mut_*.log are INTENTIONAL-failure runs (teeth asserted by the
        # suite gate — a surviving mutant fails the build); STATUS pins the
        # gate summary, which counts them
        ("build/gate.txt", r"MASK SUITE: ALL PASS \(rom \d+ \+ csr \d+ checks, 2 mutants caught\)", False),
     ]),
    ("kvq/gqa", "S4b per-KV-head KVQ banks (apex_kvq_gqa_bank at KVQ_GQA_NENG=4: golden-vector bank scoreboard + apex_top elaboration proof at GQANE=4 + 3 signature-required mutants)",
     "verif/kvq/gqa", [
        ("build/run_gqa4.log", r"GQA BANK gqa4: checks=\d+ fails=0", False),
        ("build/run_gqa4.log", r"GQA BANK gqa4: ALL PASS", False),
        # run_mut_*.log are INTENTIONAL-failure runs; mutation.txt carries the
        # signature-checked verdicts and gate.txt the suite roll-up
        ("build/mutation.txt", r"mS1: CAUGHT", False),
        ("build/mutation.txt", r"mS2: CAUGHT", False),
        ("build/mutation.txt", r"mS3: CAUGHT", False),
        ("build/gate.txt", r"GQA SUITE: ALL PASS \(gqa4 \d+ checks, 3/3 signature mutants caught\)", False),
     ]),
    ("seam", "D-021 seam blocks (feeder-quant + score-dequant), 17 runs, mutation gate",
     "verif/seam", [
        ("build/coverage.txt", r"COVERAGE PASS: all required buckets hit", False),
        ("build/mutation.txt", r"MUTATION GATE: 4/4 mutants detected", False),
        ("build/run_*.log", r"TB PASS", True),
     ]),
    ("seq", "SEQ walker (D-006 serialization, aborts, rejects) vs MXE stub",
     "verif/seq", [
        ("build/coverage.txt", r"COVERAGE: all required buckets hit", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("seq_walker", "B1/D-029 layer-walker unit suite (walker1 score+pv vs the L3-extracted reference trace incl. refusal gate; walker2 real-7B H=28 layer walks at D=64/128; composite-unit sweeps + directed; check-equivalence sweep). 16 mutant kills asserted by `make all` (gates 1-4 echo-only: a surviving mutant fails the build — kvq/sb precedent)",
     "verif/seq_walker", [
        ("build/run_small.log", r"WALKER RESULT: cases=\d+ checks=\d+ errors=0", False),
        ("build/run_[sdr]*.log", r"WALKER PASS", True),
        ("build/run2_l7b.log", r"WALKER2 RESULT: cases=1 checks=\d+ errors=0 kvw=\d+ heads=28 mxelegal=\d+", False),
        ("build/run2_*.log", r"WALKER2 PASS", True),
        ("build/run_comp*.log", r"COMP PASS", True),
        ("build/run_check.log", r"CHECKEQ PASS", False),
     ]),
    ("csr", "CSR window (map, sticky/W1C, PERF counters at W=8/W=32)",
     "verif/csr", [
        ("build/coverage.txt", r"COVERAGE: all required buckets hit", False),
        ("build/run_*.log", r"TB PASS.*0 errors", True),
     ]),
    ("audit_seq", "SEQ adversarial attack audit", "verif/audit_seq", [
        ("build/run_attack.log", r"TB PASS", False),
     ]),
    ("audit_csr", "CSR adversarial attack audit (both PERF widths)", "verif/audit_csr", [
        ("build/run_w*.log", r"TB PASS", True),
     ]),
    ("rope/smoke", "RoPE swept-domain smoke (head_dim 64+128, pos 0..127, HF half-split; Verilator+Icarus)",
     "verif/rope/smoke", [
        ("build/run_*.log", r"12288/12288 channel-pair vectors bit-exact", True),
     ]),
    ("asu/silu", "SiLU EXHAUSTIVE full 16-bit domain vs golden silu_fx (Verilator+Icarus)",
     "verif/asu/silu", [
        ("build/run_*.log", r"65536/65536 input patterns bit-exact", True),
     ]),
    ("residual", "residual_add fp16 smoke (12 directed + 20000 random; Verilator+Icarus)",
     "verif/residual", [
        ("build/run_*.log", r"20012/20012 fp16 residual-add vectors bit-exact", True),
     ]),
    ("rope/row", "RoPE row unit (rope_row: full-row rotation with xbuf, fast/bp/reset/storm regimes at D=64 + bp at D=128, 6-mutant gate)",
     "verif/rope/row", [
        ("build/run_*.log", r"ROPEROW RESULT: rows=\d+ beats=\d+ .* mismatches=0 -> PASS", True),
        ("build/mutation.txt", r"ROPEROW MUTATION GATE: 6/6 mutants detected", False),
     ]),
    ("misc/resfx", "residual fixed-point unit + resid streaming unit (resfx: 2M-point golden file sweep; resid_unit: window/guard/corner cases under fast/stall/reset regimes; 3+4 mutant gates)",
     "verif/misc/resfx", [
        ("build/run_resfx.log", r"RESFX RESULT: file=\d+ insim=\d+ mismatches=0 -> PASS", False),
        ("build/mutation.txt", r"RESFX MUTATION GATE: 3/3 mutants detected", False),
        ("build/run_resid_*.log", r"RESID RESULT: cases=\d+ jobs=\d+ elems=\d+ .* mismatches=0 -> PASS", True),
        ("build/mutation_resid.txt", r"RESID MUTATION GATE: 4/4 mutants detected", False),
     ]),
    ("layer", "decoder-layer composition TB (RoPE+SiLU+residual on real decoder_layer_fx stage tensors; Verilator+Icarus)",
     "verif/layer", [
        ("build/run_*.log", r"LAYER COMPOSITION: ALL TESTS PASSED", True),
     ]),
    ("v0/kve", "V0.1/V0.2 per-channel INT4 codec KVQ re-verification (7 configs, tready bug->fix)",
     "verif/v0/kve", [
        ("logs/sha256.log", r"OK", False),
        ("logs/run_d*.log", r"PASS", True),
        ("logs/run_g128*.log", r"PASS", True),
        ("logs/run_p_*.log", r"PASS", True),
        ("logs/run_stall_bug.log", r"TREADY BUG REPRODUCED", False),
        ("logs/run_stall_fix.log", r"PASS", False),
     ]),
    ("v0/rsqrt", "V0.4 rsqrt_unit re-verification (1.067M-op sweep, latency pin)",
     "verif/v0/rsqrt", [
        ("logs/run_golden.log", r"PASS", False),
        ("logs/run_sweep.log", r"0 mismatches", False),
     ]),
    ("v0/skid", "V0.5 stream_skid standalone + run1 root cause", "verif/v0/skid", [
        ("logs/*.log", r"PASS", False),
     ]),
    ("tip", "V0.3 the ratio-test unit re-verification (W=32)", "verif/tip", [
        ("sim_w32_directed.log", r"ALL TESTS PASSED", False),
        ("sim_w32_n4096.log", r"ALL TESTS PASSED", False),
     ]),
]

FAIL_MARKERS = re.compile(r"TB FAIL|TOPSMOKE FAIL|GATE: FAIL|COVERAGE GATE: FAIL|MUTATION GATE: FAIL")

# Suites whose logs were deliberately removed with the third-party baseline
# checkout (clean-room rule — see verif/kvq/smoke/Makefile header). A missing
# log here is a documented retirement, not an open item; the covered behavior
# is re-verified by the in-tree clean-room suites named in the reason.
RETIRED = {
    "v0/kve":   "third-party checkout removed (clean-room); behavior covered by kvq/smoke V0-parity + kvq/sb",
    "v0/rsqrt": "third-party checkout removed; rsqrt covered inside asu/sb (D-018 latency pin)",
    "v0/skid":  "third-party checkout removed; skid covered by every suite's SVA pack (D-007)",
    "tip":      "V0.3-era standalone logs removed; TIP covered by tip/sb (67/67 buckets)",
}


def eval_suite(base, checks):
    """Return (verdict, evidence_lines, newest_mtime, notes)."""
    evidence, notes = [], []
    newest = None
    verdict = "PASS"
    for rel, pattern, all_must in checks:
        paths = sorted(globmod.glob(os.path.join(REPO, base, rel)))
        if not paths:
            notes.append(f"missing: {rel}")
            verdict = "NO-RUN"
            continue
        rx = re.compile(pattern)
        first_hit = None
        for p in paths:
            newest = max(newest or 0, os.path.getmtime(p))
            try:
                text = open(p, errors="replace").read()
            except OSError as e:
                notes.append(f"unreadable {rel}: {e}")
                verdict = "FAIL" if verdict != "NO-RUN" else verdict
                continue
            m = rx.search(text)
            if m:
                if first_hit is None:
                    line = next((l for l in text.splitlines() if rx.search(l)), m.group(0))
                    first_hit = line.strip()
            elif all_must:
                notes.append(f"{os.path.relpath(p, REPO)}: no match for /{pattern}/")
                verdict = "FAIL"
            if FAIL_MARKERS.search(text):
                notes.append(f"{os.path.relpath(p, REPO)}: explicit FAIL marker")
                verdict = "FAIL"
        if first_hit is None:
            notes.append(f"{rel}: no file matched /{pattern}/")
            verdict = "FAIL" if verdict == "PASS" else verdict
        else:
            evidence.append(first_hit)
    return verdict, evidence, newest, notes


def run_golden(skip):
    if skip:
        return "NO-RUN", [], "", ["skipped (--no-golden)"]
    proc = subprocess.run(["make", "-C", os.path.join(REPO, "golden"), "test"],
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    ok = (proc.returncode == 0
          and "GOLDEN SUITE: contract + compute + attention + transformer + plumbing7b + effbits + masksem ALL PASS" in out)
    ev = [l.strip() for l in out.splitlines()
          if re.search(r"GOLDEN SUITE|ALL \d+ CASES WITHIN DERIVED BUDGET|"
                       r"APEX COMPUTE GOLDEN", l)]
    notes = [] if ok else [f"exit={proc.returncode}"]
    return ("PASS" if ok else "FAIL"), ev, out, notes


def d022_table(golden_out):
    rows = []
    rx = re.compile(r"tier (CQ-\S+?)\s*:\s*cases=\s*(\d+)\s+worst e2e/\|y\|=(\S+)"
                    r"\s+worst e2e/\|V\|=(\S+)\s+worst P-lane/\|V\|=(\S+)")
    for l in golden_out.splitlines():
        m = rx.search(l)
        if m:
            rows.append(m.groups())
    docs = [l.strip() for l in golden_out.splitlines()
            if re.search(r"DOC\s+(D-022|D023-T1)", l)]
    return rows, docs


def loc_summary():
    groups = {}

    def count(path):
        try:
            with open(path, errors="replace") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    for root, dirs, files in os.walk(os.path.join(REPO, "rtl")):
        dirs[:] = [d for d in dirs if d != "obj_dir"]
        for f in files:
            if not f.endswith((".sv", ".svh")):
                continue
            rel = os.path.relpath(root, os.path.join(REPO, "rtl"))
            top = "apex_pkg" if rel == "." else rel.split(os.sep)[0]
            if "vendor" in rel.split(os.sep):
                top += " (vendored)"
            groups.setdefault(top, [0, 0])
            groups[top][0] += 1
            groups[top][1] += count(os.path.join(root, f))

    extra = {"golden (py)": ("golden", (".py",)),
             "verif SV/SVH": ("verif", (".sv", ".svh")),
             "verif py": ("verif", (".py",)),
             "scripts (py)": ("scripts", (".py",))}
    for label, (sub, exts) in extra.items():
        n = c = 0
        for root, dirs, files in os.walk(os.path.join(REPO, sub)):
            dirs[:] = [d for d in dirs if d not in
                       ("obj_dir",) and not d.startswith(("obj_", "mut_"))
                       and d != "build"]
            for f in files:
                if f.endswith(exts):
                    n += 1
                    c += count(os.path.join(root, f))
        groups[label] = [n, c]
    return groups


def traceability_open_items():
    path = os.path.join(REPO, "TRACEABILITY.md")
    partial, limits = [], []
    if not os.path.exists(path):
        return ["TRACEABILITY.md missing"], []
    for l in open(path, errors="replace"):
        m = re.match(r"\|\s*((?:D|C)-\d+)\s*\|\s*([^|]+)\|.*\*\*(PARTIAL|UNTESTED)\*\*", l)
        if m:
            partial.append(f"{m.group(1)} ({m.group(3)}): {m.group(2).strip()}")
        m2 = re.match(r"- \*\*(L-[A-Za-z0-9]+(?:\.\.[A-Za-z0-9]+)?)", l)
        if m2:
            limits.append(m2.group(1))
    return partial, limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-golden", action="store_true")
    args = ap.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    n_dirty = len([l for l in dirty.splitlines() if l.strip()])

    g_verdict, g_ev, g_out, g_notes = run_golden(args.no_golden)
    tier_rows, doc_lines = d022_table(g_out)

    results = []
    for name, desc, base, checks in SUITES:
        verdict, ev, mtime, notes = eval_suite(base, checks)
        if verdict == "NO-RUN" and name in RETIRED:
            verdict, notes = "RETIRED", [RETIRED[name]]
        results.append((name, desc, verdict, ev, mtime, notes))

    partial, limits = traceability_open_items()
    locs = loc_summary()

    L = []
    L.append("# APEX v0.2 — STATUS")
    L.append("")
    L.append("**GENERATED FILE — do not hand-edit.** Written by `scripts/gen_status.py`,")
    L.append("which parses the actual suite logs and runs the golden gate live")
    L.append("(ARCHITECTURE.md §8 anti-fabrication rule). Regenerate with:")
    L.append("`python scripts/gen_status.py`.")
    L.append("")
    L.append(f"- Generated: {now}")
    L.append(f"- HEAD: `{head}`" + (f" (+{n_dirty} working-tree entries)" if n_dirty else " (clean tree)"))
    L.append("")
    L.append("## 1. Gate roll-up")
    L.append("")
    L.append("| Suite | What | Verdict | Evidence (verbatim from parsed log / live run) | Log freshness |")
    L.append("|---|---|---|---|---|")
    g_ev_s = "<br>".join(f"`{e[:150]}`" for e in g_ev[:3]) or "—"
    L.append(f"| golden (LIVE) | golden contract + compute + attention gate | **{g_verdict}** | {g_ev_s} | ran now |")
    for name, desc, verdict, ev, mtime, notes in results:
        ev_s = "<br>".join(f"`{e[:150]}`" for e in ev[:3]) or "—"
        if notes:
            ev_s += "<br>" + "<br>".join(f"NOTE: {n[:120]}" for n in notes[:3])
        fresh = (datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                 if mtime else "—")
        L.append(f"| {name} | {desc} | **{verdict}** | {ev_s} | {fresh} |")
    L.append("")

    L.append("## 2. D-022 tier-quality table (parsed from the live golden run)")
    L.append("")
    if tier_rows:
        L.append("| Tier | Cases | worst e2e/\\|y\\| | worst e2e/\\|V\\| | worst P-lane/\\|V\\| |")
        L.append("|---|---|---|---|---|")
        for t, c, ey, ev_, pl in tier_rows:
            L.append(f"| {t} | {c} | {ey} | {ev_} | {pl} |")
        L.append("")
        L.append("Documented exceedance / floor register (gate fails on NEW or VANISHED entries):")
        L.append("")
        for d in doc_lines:
            L.append(f"- `{d}`")
    else:
        L.append("_No live golden run parsed (--no-golden or run failed) — table unavailable._")
    L.append("")

    L.append("## 3. LOC summary (working tree)")
    L.append("")
    L.append("| Group | Files | Lines |")
    L.append("|---|---|---|")
    total = 0
    for k in sorted(locs):
        n, c = locs[k]
        total += c
        L.append(f"| {k} | {n} | {c} |")
    L.append(f"| **total** | | **{total}** |")
    L.append("")

    L.append("## 4. Open items (mechanically derived)")
    L.append("")
    non_pass = [(n, v) for n, _, v, _, _, _ in results if v not in ("PASS", "RETIRED")]
    if g_verdict != "PASS":
        non_pass.insert(0, ("golden", g_verdict))
    if non_pass:
        L.append("Suites not PASS:")
        for n, v in non_pass:
            L.append(f"- {n}: **{v}**")
    else:
        L.append("- All parsed suites PASS (retired suites listed in the roll-up with reasons).")
    L.append("")
    if partial:
        L.append("TRACEABILITY.md matrix rows not fully PROVEN:")
        for p in partial:
            L.append(f"- {p}")
        L.append("")
    if limits:
        L.append(f"Known-limitations register (TRACEABILITY.md §3): {len(limits)} entries — "
                 + ", ".join(limits))
    L.append("")

    with open(STATUS_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {STATUS_PATH}")
    print(f"golden: {g_verdict}; suites: "
          + ", ".join(f"{n}={v}" for n, _, v, _, _, _ in results))


if __name__ == "__main__":
    main()
