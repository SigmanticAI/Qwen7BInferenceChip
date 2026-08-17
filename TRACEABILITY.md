# APEX v0.2 — Traceability & Known-Limitations Register

**Date:** 2026-07-09 (v0.2 closure) · **Repo state:** `ff8b467` + working tree (the
three v0.2 passes §0b/§0c/§0d + this closure — uncommitted, left for the orchestrator)
**Produced by:** independent closure passes. Every suite below was re-run **from clean on
this machine**; verbatim lines are quoted from those runs (persisted under each suite's
`build/` or `logs/`). Nothing in this file is copied from a claim without a fresh
reproduction or a named log file. §0 is the v0.1 closure record; §0b/§0c/§0d are the
three v0.2 work passes; **§0e is the final v0.2 closure reproduction (every hard gate
re-run from clean in one sitting, after all three passes landed).**

---

## 0. Fresh reproduction summary (this closure pass, 2026-07-09)

| Gate | Command | Exit | Verbatim key line |
|---|---|---|---|
| Golden (hard gate) | `make -C golden test` | 0 | `GOLDEN SUITE: contract + compute + attention ALL PASS` · `APEX ATTENTION GOLDEN: ALL 120 CASES WITHIN DERIVED BUDGET (hard + statistical), ALL SELF-CONSISTENCY CHECKS PASS` |
| Tile smoke | `make -C verif/top/smoke smoke` (after `make clean`) | 0 | `LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg.sv + vendored rsqrt/kvq files only)` · `TOPSMOKE RESULT: cycles=136951 checks=2173 errors=0` · `TOPSMOKE PASS` |
| Layer-2 pairwise | `make -C verif/top/l2 all` (from clean) | 0 | `total checks across required runs: 18863` · `L2 COVERAGE GATE: PASS` (5× `L2A PASS`, 10× `L2B PASS`, plus L2C runs, all `errors=0`) |
| Layer-3 e2e | `make -C verif/top/l3 all` (from clean) | 0 | `L3 RUNS: all 21 cases passed` · `total checks: 56406` · `L3 COVERAGE GATE: PASS` · `MUTATION GATE: PASS (3/3 tile-level mutants caught)` |
| Leaf spot-check | `make -C verif/mxe all` (from clean) | 0 | 6× `TB PASS … 0 errors` · `COVERAGE: all required buckets hit (38 required, 40 total)` |

The `ee6375c` commit message's honest-FAIL (`drive_qj stall @467354`, phase F wedge)
describes the **committed** `tb_apex_smoke.sv`; the working-tree fix
(`verif/top/smoke/{tb_apex_smoke.sv,gen_top_vectors.py}`, modified) is what passes.
The F-5 wedge signature survived as a kept-failing regression
(`[CONFIRMED-BUG] bug_d128_stagebuf_nb: still fails with 'drive_g stall'`) until the
F-1/F-5 closure pass later the same day (see §0b) flipped it to kept-PASSING:
the identical case now completes the full D=128 chain bit-exact
(`[PASS] bug_d128_stagebuf_nb: L3 RESULT: cycles=95815 checks=1143 errors=0`);
the pre-fix failure was re-reproduced fresh immediately before the fix
(`drive_g stall @400911`, `verif/top/l3/logs/prefix_f5a_keptfail_before_fix.log`).

## 0b. Envelope-extension pass (Task A1: F-5a/F-5b/F-1, 2026-07-09)

Regression-first, all fresh on this machine (logs persisted under the suites'
`build/`):

| Step | Command / case | Verbatim key line |
|---|---|---|
| F-5a fail-first (pre-fix) | l3 `bug_d128_stagebuf_nb` on the pre-fix build | `%Fatal: … drive_g stall @400911` (`logs/prefix_f5a_keptfail_before_fix.log`) |
| F-5b fail-first (unit vs reverted-index mutant) | `tb_stagebuf_patd` vs `sel_q[2:0]` mutant | `[PAT_D sel=8 beat=0] got e0c0a08060402000 exp 482808e8c8a88868` → FAIL (`logs/run_unit_stagebuf_mutant.log`) |
| F-1 fail-first (T_ROW_MAX=64 build) | l3 `calib_d64_T128` vs `-GT_ROW_MAX`-style pre-fix copy | `%Fatal: … drive_cs stall @928783` (`logs/prefix_f1_keptfail_before_fix.log`) |
| Fix + unit regression | `make -C verif/top/l3 all` (unit stage) | `STAGEBUF PATD RESULT: cycles=505 checks=184 errors=0` · `STAGEBUF PATD PASS` |
| L3 from clean | `make -C verif/top/l3 all` | `L3 RUNS: all 24 cases passed` · `total checks: 108483` · `L3 COVERAGE GATE: PASS` · `MUTATION GATE: PASS (3/3 tile-level mutants caught)` |
| L3 F-1/F-5 flagship cases | (within the run above) | `calib_d64_T128 … checks=17822 errors=0`, `adv_outlier1000 … checks=17822 errors=0`, `calib_d64_T70 … checks=9760 errors=0`, `calib_d128_T100 … checks=27538 errors=0`, `bug_d128_stagebuf_nb … checks=1143 errors=0`, `rand_d128_T5/T12 … errors=0` |
| Smoke from clean (both tiles) | `make -C verif/top/smoke smoke` | D=64: `TOPSMOKE RESULT: … checks=2173 errors=0`; D=128: `TOPSMOKE RESULT: … checks=4237 errors=0` (cycle counts vs the §0 run differ due to the concurrent D-005 mxe_ctrl load-under-compute change — checks identical/bit-exact) |
| Golden (hard gate) | `make -C golden test` | `GOLDEN SUITE: contract + compute + attention ALL PASS` |
| L2 from clean | `make -C verif/top/l2 all` | `L2 COVERAGE GATE: PASS` (unchanged suite, re-proven) |

RTL touched: `rtl/top/apex_top.sv` (STAGE_NB_W-wide aj_nb/wj_nb ports; new
`T_ROW_MAX=128` envelope parameter → `seam_score_dequant` N_MAX +
`apex_scale_quant` `SQ_COLS_MAX`), `rtl/top/glue/apex_stage_buf.sv` (NB_W
derived nb field; PAT_D full-`sel` indexing). `rtl/apex_pkg.sv` UNTOUCHED —
no contract field changed, APEX_VERSION stays `0x0001_0000`.

## 0c. D-004/D-005 closure pass (Task A2: rtl/mxe + verif/mxe, 2026-07-09)

Regression-first, all fresh on this machine (logs persisted under
`verif/mxe/{struct,perf,sb}/build/`):

| Step | Command / case | Verbatim key line |
|---|---|---|
| D-005 fail-first (pre-change ctrl) | `tb_mxe_perf +require_overlap=1` on the sequential mxe_ctrl | `PERF OVERLAP MISSING [K64_M16] … [K2048_M64] … [K64_M16_wgap3] … [K2048_M64_wgap3]` → `PERF SUITE FAIL: 4 error(s)` (kept permanently: `make -C verif/mxe/perf run_base_reqov` vs the pinned `perf/baseline/mxe_ctrl.sv`) |
| D-004 discriminator on real RTL | `make -C verif/mxe/struct run_real` | `STRUCT PASS: 20800 whitebox timing checks — (r+c)-offset act arrival, per-hop psum staggering, column-staggered acc writes, single-cycle glitch hop locality` |
| D-004 mutation gate | `make -C verif/mxe/struct mutation` | broadcast-wall mutant: `MXE SMOKE: ALL TESTS PASSED` (functional suite cannot discriminate) then `STRUCT FAIL: 2677/13120 checks failed` → `MUTATION CHECK PASS` |
| D-005 perf gate (post-change) | `make -C verif/mxe/perf all` | `PERF GATE: PASS`; K2048_M64 56901→53573 (1.062x, overlap=2048); K64_M16 645→541 (1.192x, overlap=64); weight-BW-limited: 60741→53573 (1.134x, load fully hidden) and 765→579 (1.321x) |
| Full MXE leaf suite from clean | `make -C verif/mxe all` (now smoke+sb+struct+perf) | exit 0; 6× `TB PASS … 0 errors` (836 legal / 65 illegal / 35 resets); `COVERAGE: all required buckets hit (38 required, 40 total)` incl. `rst_wload 5 HIT` |
| Golden (hard gate) | `make -C golden test` | 0 · `GOLDEN SUITE: contract + compute + attention ALL PASS` |
| Tile smoke from clean | `make -C verif/top/smoke smoke` | 0 · `TOPSMOKE RESULT: cycles=110211 checks=2173 errors=0` (D=64) · `TOPSMOKE RESULT: cycles=295789 checks=4237 errors=0` (D=128) — checks identical/bit-exact to the reference runs; D=64 tile smoke runs ~19% fewer cycles than §0's 136951, consistent with D-005 load-under-compute (Task A1's concurrent smoke-TB changes may also contribute to cycle deltas) |
| L2 from clean | `make -C verif/top/l2 all` | 0 · `L2 COVERAGE GATE: PASS` · `total checks across required runs: 18863` |
| L3 from clean | `make -C verif/top/l3 all` | 0 · `L3 RUNS: all 24 cases passed` · `total checks: 108483` · `MUTATION GATE: PASS (3/3 tile-level mutants caught)` |

RTL touched: `rtl/mxe/mxe_ctrl.sv` ONLY (weight loader decoupled from the
compute FSM; shadow bank fills under INGEST/COMPUTE/FLUSH; bank swap moved to
wavefront-clear COMPUTE entries; S_WLOAD = wait-for-shadow-bank, same enum
encoding; external stream/job/legality/reset contract UNCHANGED — inside the
D-006 envelope, which is why the unmodified top gates pass). `rtl/apex_pkg.sv`
UNTOUCHED — APEX_VERSION stays `0x0001_0000`. The pre-change sequential ctrl
is pinned verbatim at `verif/mxe/perf/baseline/mxe_ctrl.sv` as the perf
baseline + kept-failing overlap regression.

## 0d. F-2 / D-022-actuation / F-3 closure pass (Task B1: rtl/top + csr, 2026-07-09)

Regression-first, all fresh on this machine. RTL touched: NEW
`rtl/top/glue/apex_kvq_bank.sv` (D-024 tier bank: 3 unmodified kvq_engine
instances, tier-routed AXI-Lite/s_axis/m_axis/flush), `rtl/top/apex_top.sv`
(tier bank + auto_tier map + rt_kv_user + ERR_STICKY 0x58 W1C window +
KVQ_OUTLIER_K/KVQ_MASK_FILE params; KVQ_TIER param retired),
`rtl/csr/csr_regs.sv` (new TIERS parameter, default 3'b111 — INFO_TIER build
truth; standalone suites unchanged). `rtl/apex_pkg.sv` UNTOUCHED —
APEX_VERSION stays `0x0001_0000`. Golden extended (attention.py
`kvq_roundtrip_tiermap` / `attention_core(tier_map=)`, gated by new
test_attention section E).

| Step | Command / case | Verbatim key line |
|---|---|---|
| F-2/D-022 fail-first (pre-fix tile) | probe: TIER_CTRL=CQ4 then poll engine INFO_TIER | `%Fatal: … KVP 0c timeout (got 00000000) @1050033` (`verif/top/l3/logs/prefix_f2_keptfail_before_fix.log`) |
| F-3 fail-first (pre-fix tile) | probe: set sticky, W1C 0x58, verify | `%Error: … [ESTK] got 00000001 exp 00000000` (`verif/top/l3/logs/prefix_f3_keptfail_before_fix.log`) |
| Golden gate + section E | `make -C golden test` | `GOLDEN SUITE: contract + compute + attention ALL PASS` · `D-022 actuation: calib d64_T128 e2e/\|V\| uniform CQ-4=12.89% -> per-block map=4.03%` |
| Smoke from clean (3 tiles) | `make -C verif/top/smoke smoke` | `TOPSMOKE RESULT: cycles=110221 checks=2176 errors=0` (D=64) · `cycles=295799 checks=4240` (D=128) · `cycles=106603 checks=2176` (D=64 CQ-4+ grouped keys + FLUSH + mask) |
| L3 from clean | `make -C verif/top/l3 all` | `L3 RUNS: all 27 cases passed` · `total checks: 135241` · `L3 COVERAGE GATE: PASS` · `MUTATION GATE: PASS (3/3 tile-level mutants caught)` |
| L3 F-2 flagship cases | (within the run above) | `adv_T1_cq4p … checks=172 errors=0` (T=1 tile-level D-008 FLUSH) · `adv_outlier1000_cq4p … checks=17825 errors=0` (T=128, 8 full G=16 key groups) |
| D-022 actuation flagship | `tip_auto_mixed` | `[ETIPT] decision: blk=1 tier=2 fp16=1 (LOAD-BEARING: drives auto_tier)` (+blk 0/2→tier 1, blk 3→tier 2) · `L3 RESULT: cycles=290806 checks=8689 errors=0` — TIP-driven per-block CQ-4/CQ-4+ stores, IMPORTANCE_BASE map readback, bit-exact vs `attention_core(tier_map=…)` |
| L2 from clean | `make -C verif/top/l2 all` | `L2 COVERAGE GATE: PASS` · `total checks across required runs: 18863` (unchanged suite, re-proven) |
| csr leaf suites (TIERS param) | `make -C verif/csr all` + `make -C verif/audit_csr run mutant` | `COVERAGE: all required buckets hit (61 required, 70 total)` · `MUTATION GATE: 2/2 mutants caught` · `TB PASS: csr attack suite clean — 31359 checked responses (PERF_W=8 seed=902)` · `MUTANT perf_double CAUGHT` |

## 0e. v0.2 closure reproduction (independent, 2026-07-09, after §0b/§0c/§0d landed)

Every hard gate re-run **from clean** (`make clean` first where the suite has one) in a
single sitting on this machine; full transcripts persisted under each suite's `build/`
(and mirrored in the closure scratchpad). All five exits were 0; the kept-failing
regression failed exactly as registered.

| Gate | Command | Exit | Verbatim key line (this run) |
|---|---|---|---|
| Golden (hard gate) | `make -C golden test` | 0 | `APEX ATTENTION GOLDEN: ALL 120 CASES WITHIN DERIVED BUDGET (hard + statistical), ALL SELF-CONSISTENCY CHECKS PASS` · `GOLDEN SUITE: contract + compute + attention ALL PASS` · section E: `D-022 actuation: calib d64_T128 e2e/\|V\| uniform CQ-4=12.89% -> per-block map=4.03%` |
| Tile smoke (3 tiles) | `make -C verif/top/smoke smoke` (from clean) | 0 | `TOPSMOKE RESULT: cycles=110221 checks=2176 errors=0` (D=64) · `cycles=295799 checks=4240 errors=0` (D=128) · `cycles=106603 checks=2176 errors=0` (D=64 CQ-4+ grouped keys + FLUSH + mask) — counts byte-identical to §0d |
| Layer-2 pairwise | `make -C verif/top/l2 all` (from clean) | 0 | `total checks across required runs: 18863` · `L2 COVERAGE GATE: PASS` |
| Layer-3 e2e | `make -C verif/top/l3 all` (from clean) | 0 | `L3 RUNS: all 27 cases passed` · `total checks: 135241` · `STAGEBUF PATD PASS` · `L3 COVERAGE GATE: PASS` · `MUTATION GATE: PASS (3/3 tile-level mutants caught)` |
| MXE leaf (all four) | `make -C verif/mxe all` (from clean: smoke+sb+struct+perf) | 0 | 6× `TB PASS … 0 errors` (836 legal / 65 illegal / 35 resets) · `COVERAGE: all required buckets hit (38 required, 40 total)` · `STRUCT PASS: 20800 whitebox timing checks` · `MUTATION CHECK PASS: broadcast-wall mutant passes tb_mxe_smoke but is KILLED by tb_mxe_struct` · `PERF GATE: PASS` (K2048_M64 56901→53573 1.062x; wgap3 1.134x/1.321x) |
| D-005 kept-FAILING regression | `make -C verif/mxe/perf run_base_reqov` (inside `all`) | — | baseline still fails as required: 4× `PERF OVERLAP MISSING [K…] … overlap=0` → `[CONFIRMED-GAP] sequential baseline ctrl: PERF OVERLAP MISSING on every multi-chunk job (kept-failing, D-005 register)` |

F-register regressions, now **kept-PASSING inside the L3/smoke runs above** (same run,
verbatim): `bug_d128_stagebuf_nb … cycles=95825 checks=1146 errors=0` (F-5 wedge),
`calib_d64_T128 … checks=17825 errors=0` + `calib_d64_T70 … checks=9763` +
`calib_d128_T100 … checks=27541` (F-1 full-length), `adv_T1_cq4p … checks=172` (F-2
tile-level FLUSH) + `adv_outlier1000_cq4p … checks=17825` (F-2 grouped keys + mask),
`tip_auto_mixed … checks=8689 errors=0` (D-022 actuation), and every case's ERR_STICKY
set→W1C→verify tail (F-3 sticky). The durable pre-fix fail-first logs remain under
`verif/top/l3/logs/` (they survive `make clean`) and are now parsed by
`scripts/gen_status.py` as the `l3/fail-first` row. Check-count deltas vs §0/§0b
(e.g. `bug_d128_stagebuf_nb` 1143→1146) are the F-3 ERR_STICKY tail checks added in
§0d — cycle/check counts are otherwise deterministic and byte-identical to §0d.

**Status legend** — **PROVEN**: direct, reproduced test evidence at the decision's own
scope. **PARTIAL**: real evidence exists but a stated part of the decision has no
discriminating test (the missing part is named). **UNTESTED**: no direct evidence.

---

## 1. Decision matrix D-001..D-024

**v0.2 re-grade (2026-07-09, §0e reproduction):** the three v0.1 PARTIALs — D-004
(systolic structure), D-005 (load-under-compute), D-022 (TIP-driven tier selection
in-tile) — are all **PROVEN**; **no PARTIAL or UNTESTED rows remain** anywhere in the
matrix (D-001..D-024, C-1..C-5 = 29/29 PROVEN at their own scope). Rows whose evidence
was produced in §0b/§0c/§0d were additionally re-validated by the §0e from-clean
reproduction (the quoted suites re-ran green with identical counts). Named caveats
inside PROVEN rows are honest scope bounds, not missing evidence; they are mirrored in
§3.

| ID | Decision (short) | Test / suite / assertion | Verbatim evidence (fresh runs or named persisted log) | Status |
|---|---|---|---|---|
| D-001 | INT4 = per-channel INT4 codec [-8,7] RNE EPS=2⁻¹⁴; golden vectors arbiter | `golden/tests/test_contract.py` (9 frozen vectors); `verif/v0/kve` (7 configs through RTL top); `verif/kvq/{smoke,sb}` | `d64_T128_G64__CQ8 … tier=0 : PASS` (×9, golden); `KVQ V0.1 PARITY [d64_cq8]: PASS … checks=49923 fails=0` (×7, `verif/v0/kve/logs/`); kvq smoke re-run identical counts `49923/54020/54020/28130/77403/32665/54019` (`verif/kvq/smoke/logs/`) | **PROVEN** |
| D-002 | RNE rounding in quant + requant epilogue | golden C-2 property tests; MXE mutation 3 (RNE→half-up); seam mutant M1/M3; ASU mutant M3; KVQ sb mutation (b) | `PASS half-to-even on exact ties — got [0, 2, 0, -2, 2, -2]` (golden); `MUTANT M3_sd_rne_to_truncation: DETECTED` (`verif/seam/build/mutation.txt`); kvq `RNE tie → round-half-up … CAUGHT (10 fails … got 40400000 exp 40000000)` | **PROVEN** |
| D-003 | INT32 accumulators, K ≤ 2048/descriptor | golden C-3 tests; `verif/mxe/sb` directed K=2048 + illegal K>2048 | `PASS worst-case |acc| fits INT32` · `PASS K>2048 rejected` (golden); mxe coverage `k_eq_2048 20` hits + 65 illegal descriptors rejected, 0 errors (`verif/mxe/sb/build/`) | **PROVEN** |
| D-004 | True systolic MXE rewrite (PE-to-PE pipelining, not broadcast) | `verif/mxe` (836 legal jobs bit-exact, all shapes to 64×2048×8) + **`verif/mxe/struct`** structural discriminator (whitebox timing law P1..P5: (r+c)-offset act arrival per PE, per-hop psum staggering, column-staggered acc writes at T+MXE_N+c, 1-beat/cycle sustained wavefronts, single-cycle glitch hop locality) + broadcast-wall mutation check | `TB PASS: 520 legal jobs … 0 errors (bp=1 stall=1 seed=3)`; `STRUCT PASS: 20800 whitebox timing checks` (`verif/mxe/struct/build/run_real.log`); mutation gate: realistic broadcast-wall mutant (`struct/mutant/mxe_array.sv`) passes `tb_mxe_smoke` (`MXE SMOKE: ALL TESTS PASSED` — functional suite provably cannot discriminate) but is killed: `STRUCT FAIL: 2677/13120 checks failed` (`struct/build/mutation.txt`) | **PROVEN** (2026-07-09) — caveat: the mutation check uses the *realistic* run1/run2-style wall; an adversarial wall with per-column delay lines mimicking internal timing would be caught only by the P5 glitch test (needs a physical act path between adjacent columns). |
| D-005 | Double-buffered weight SRAM + wide (64 B/beat internal) load port, USED for load-under-compute | `verif/mxe` (load path under stall storms) + **`verif/mxe/perf`**: mxe_ctrl loads chunk k+1 into the shadow bank while chunk k computes (swap at wavefront-clear COMPUTE entry); tb_mxe_perf measures cycles + overlap on K=64/M=16 and K=2048/M=64 WS jobs vs the PINNED sequential ctrl (`perf/baseline/mxe_ctrl.sv`), bit-exact + SVA-bound in both builds; kept-failing regression `run_base_reqov` | `PERF GATE: PASS` (`verif/mxe/perf/build/perf_gate.txt`): K2048_M64 56901→53573 cyc (1.062x, overlap=2048); K64_M16 645→541 (1.192x, overlap=64); weight-BW-limited (1 beat/4 cyc): K2048 60741→53573 (1.134x — load fully hidden), K64 765→579 (1.321x); baseline overlap=0 asserted; `[CONFIRMED-GAP] sequential baseline ctrl: PERF OVERLAP MISSING` (kept-failing) | **PROVEN** (2026-07-09) — overlap demonstrated and gated (overlap_cycles > 0 + strict cycle win per job). Note: measured via whitebox TB counters, not the CSR `PERF_*` window (tile-level PERF-counter wiring for MXE phases remains v0.2 CSR work). |
| D-006 | `done` ⇒ post-skid acceptance; SEQ serializes jobs | `verif/common/apex_stream_sva.svh` (17 assertions, bound every MXE build); mxe mutation 2; `verif/seq` walker suite + `verif/audit_seq` | `%Error: apex_stream_sva.svh:169 … ap_done_means_idle: [SVA §5] busy still high at done` (mutant caught); seq `TB PASS: 56 dispatched (37 done, 19 rejected …) 0 errors` (`verif/seq/build/run_random.log`); `TB PASS: seq attack suite clean (seed=777)` (`verif/audit_seq/build/run_attack.log`) | **PROVEN** |
| D-007 | KVQ read path gets skid + honors tready | `verif/v0/kve` V0.2 stall test (bug→fix→re-prove); carried into `verif/kvq/smoke` | Unpatched: `beats_dropped=7171 … data_mismatches_on_accepted=5450 … TREADY BUG REPRODUCED`; patched: `beats_accepted=13824 beats_dropped=0 … PASS`; kvq smoke re-run: `KVQ STALL [stall_kvq]: PASS — all beats delivered bit-exact under backpressure` | **PROVEN** |
| D-008 | Partial-group flush via CSR `FLUSH` | `verif/kvq/{smoke,sb}` (flush shapes 1,2,G−1, mid-token, G-th token, no-op); `verif/csr` (FLUSH pulse); `verif/top/l2` chain (b) incl. D=128 flush; **tile level (2026-07-09):** smoke cq4p run (8<G=128 flush), l3 `adv_T1_cq4p` (T=1 flush), `tip_auto_mixed` (4 per-block flushes in AUTO mode) | kvq sb flush buckets closed (`COVERAGE GATE: PASS`); csr `181 checked reads, 2 soft_reset + 2 flush pulses, 0 errors`; l2 `[ok ] l2b_calib_d128.flush = 1`, `l2b_t1.flush = 3`; tile `adv_T1_cq4p … checks=172 errors=0` | **PROVEN** at block + L2 + **tile** (F-2 closed by D-024; former L-T2 gap retired) |
| D-009 | Byte-aligned 64b-padded KVQ record | `verif/v0/kve` + `verif/kvq/sb` field-by-field SRAM record checks vs golden §4 image | kve: per-record hierarchical `dut.u_sram.mem[addr]` field checks inside `checks=49923..77403 fails=0`; kvq sb `CONFIG d64: checks=3708 fails=0` (full record images) | **PROVEN** (§4's original 1312/2608 example totals were wrong; corrected in ARCHITECTURE.md — the 64b rule is normative) |
| D-010 | fp32 KVQ read bus; no narrowing inside parity boundary | same suites, fp32 readback bit-exact (`!==`); B-4 −0.0 fix + audit | `R addr=8 beat=4: got 00000000 exp 80000000` (pre-fix B-4) → post-fix `TB PASS [negz_cq4p] checks=1598 fails=0` and `[negz_d64p] checks=5122 fails=0` (`verif/kvq/audit`) | **PROVEN** |
| D-011 | TIP at SCORE_WIDTH=32, per-layer THRESHOLD_REG | `verif/tip` (W=32); `verif/tip` (all 32 threshold values, ties ±1 at each); `verif/csr` THRESHOLD_REG | v0.3 `tiles=143 pass=143 fail=0`; tip sb `COVERAGE: all required buckets hit (67 required, 67 total)` incl. `thr_t0..thr_t31`; smoke `143-TILE SELF-CHECK PASSED` | **PROVEN** |
| D-012 | Verilator-first TB discipline; SVA compiled as build gate | every Makefile: `--binary --timing --assert`, SVA in every build; mutation gates prove assertions live | `MUTATION GATE: 4/4 mutants detected` (asu, seam), `4/4 mutants caught` (tip), `3/3` (mxe-style sb checks, l3 tile mutants); SVA-caught mutants: `ap_done_means_idle`, `ap_out_only_in_job`, `ap_busy_dvalid` | **PROVEN** |
| D-013 | Vendored frozen vectors + SHA256SUMS | `verif/v0/kve` (`shasum -c`); `verif/tip/smoke` provenance | `90/90 OK` (`verif/v0/kve/logs/sha256.log`); `provenance scores.hex: sha256 191ecb48…`, `LABELS MATCH — regenerated vectors == upstream's 143 frozen cases` | **PROVEN** |
| D-014 | ASU exp LUT 256-entry, err ≤ 2⁻¹⁰ | `verif/asu/smoke` exhaustive Q6.10 domain; `verif/asu/sb` re-measurement; golden | `65536/65536 input patterns bit-exact vs apex_golden exp_fx`; `AUDIT exp LUT: max |exp_fx - exp| = 4.628203e-04 (budget 2^-10 …) -> OK`; golden `PASS exhaustive domain max|err|=4.63e-04 ≤ 2^-10` | **PROVEN** |
| D-015 | KVQ vendors the V0-patched top, never upstream verbatim | `rtl/kvq/kvq_engine.sv` lineage + `verif/kvq/smoke` parity counts identical to V0 | kvq smoke re-run counts verbatim equal to V0.1 (`49923/54020/54020/28130/77403/32665/54019`, all `fails=0`). *Amended by D-026 (2026-07-14): key-record layout superseded; new derived parity counts `49923/46084/46340/24162/77403/32665/46019` (CQ-8/no-group configs unchanged by construction; derivation in `verif/kvq/smoke/Makefile`)* | **PROVEN** |
| D-026 | A2 KV-REC-DEDUP: key record [tag {ssid,1}][k lanes][D codes][pad64], persistent scale_bank_store (SETS≥ceil(T_ROW_MAX/G), tile plumbs 8), SB_OVWR wrap-fault; golden packers are the arbiter of record+bank images | golden `effbits` gate (stored/codec/CSR pinned 3 ways) + `verif/kvq/smoke` (record+bank vs committed golden images) + `verif/kvq/sb` (persistent-bank storage model, staleness, k=5 config, +2 D-026 mutants) + `verif/kvq/audit` (bank-survives-soft-reset attack, SB_OVWR sticky/W1C, retargeted b4) + `verif/top/{smoke,l2,l3}` (byte-identical check counts — A2 confined to the engine) | `EFFECTIVE-BITS GATE: ALL PINNED ACCOUNTINGS MATCH` (3.16×/3.51× stored k≤2); smoke 7/7 parity derived==observed; sb `cq4 4324/0, k5 6112/0` + 5/5 mutants caught; audit `bank_peeks=93 bank_rst=6` clean, 3/3 mutants; l3 `all 27 cases passed` after the SETS sizing fix (set-exhaustion caught by adv_outlier1000_cq4p, SB_OVWR fired as designed) | **PROVEN** |
| D-016 | 3 latent V0 findings fixed (RWAIT hang, ST_IDLE beat drop, cnt==G alias) | `verif/kvq/smoke` reproduce-then-fix regressions | `all 'BUG REPRODUCED' on baseline, 'PASS' on kvq` (alias/rwait/collide/flush, `verif/kvq/smoke/logs/`) | **PROVEN** |
| D-017 | TIP programmable-threshold datapath (D-017) | `verif/tip` (F-1 finding: `UNUSEDPARAM 'THRESHOLD'`); `verif/tip` T=0..31 exhaustive + frozen-143 | `143-TILE SELF-CHECK PASSED: APEX golden (W=32, programmable T=10) == our golden expected`; sweep runs `sweep_t1/t5/t31, clamp_t0` all `ALL TESTS PASSED` | **PROVEN** (at N=64 exhaustive per-T; N=4096 spot T∈{1,10,31} — documented) |
| D-018 | rsqrt latency = 31 (not 30); wrapper serializes issue | `verif/v0/rsqrt` (1.067M ops); `verif/asu/sb` rmsnorm ISSUE/WAIT phases incl. reset mid-rsqrt | `[SWEEP] done: 1067137 ops, 0 mismatches, latency constant at 31 cycles`; `[BUSY] in_valid while busy is silently DROPPED`; asu `RESET [Z6 d=64 phase=4 …] (rsqrt mid-flight)` clean | **PROVEN** |
| D-019 | run1 root cause = TB drain bug + D-006 wart; skid clean, vendored | `verif/v0/skid` (100k+ txns, SVA P1–P9, waveform trace cyc 193–228); stale-beat SVA in mxe/asu | skid standalone `PASS — no RTL bug found`; run1 failure `Reproduced exactly: FAIL (new_errors=7)`; `ap_out_only_in_job: [SVA D-019] output beat with no job in flight` kills the ASU M4 mutant | **PROVEN** |
| D-020 | KVQ soft-reset semantics (B-1..B-4 closed) | `verif/kvq/sb` (found B-1..B-4); `verif/kvq/audit` (260 randomized resets, fix-revert mutants); tile smoke soft-reset round 2 | audit `TB PASS [rst_cq4] checks=17878 fails=0` (+4 more configs); fix-revert mutants B-1/B-2/B-4 all `CAUGHT`; smoke `D-020 tile soft reset: abort queued descs + KVQ reset + rerun` → round-2 bit-exact inside `checks=2173 errors=0` | **PROVEN** |
| D-021 | Attention seam: feeder-requant + score-dequant + INT8 P·V̂, golden-gated | `golden/tests/test_attention.py` (gate); `verif/seam` (both blocks, 17 logs, 53 buckets, 4/4 mutants); `verif/top/l2` chains a/b/c; tile smoke phases E–G; `verif/top/l3`; `verif/audit_d021` | golden `PASS feeder quant == C-1 compress_values codes (bit-exact)` · `PASS score-dequant |Δ| ≤ RNE + f32 bound (100k random)`; seam `COVERAGE PASS: all required buckets hit (53 buckets, 17 logs)`, `MUTATION GATE: 4/4`; l2 `run_l2a_random.log: checks=3247 errors=0`; l3 `all 21 cases passed` | **PROVEN** |
| D-022 | CQ-4 quality-fragile for K̂; TIP tier-select REQUIRED for quality | golden Layer-3 tier table + documented-exceedance register (gate fails on NEW or VANISHED exceedances); **actuation (2026-07-09, D-024):** golden section E (tier-map model + load-bearing calib measurement) + l3 `tip_auto_mixed` (TIP decisions write apex_top's auto_tier map; TIER_CTRL.tip_override routes every KVQ store/read per block; the mixed-tier decode replays bit-exact vs `attention_core(tier_map=…)`) | `tier CQ-4 : cases= 40 worst e2e/|V|=2.567e-01`; `DOC D-022: calib/d128_T100_G128__CQ4/CQ-4 e2e_abs=1.062e+00 = 25.7% of value scale`; golden E: `D-022 actuation: calib d64_T128 e2e/|V| uniform CQ-4=12.89% -> per-block map=4.03%`; l3: `[ETIPT] decision: blk=1 tier=2 fp16=1 (LOAD-BEARING: drives auto_tier)` · `tip_auto_mixed … checks=8689 errors=0` | **PROVEN** (2026-07-09) — measurement, policy register, regression gate AND the TIP-driven in-tile actuation (former L-T3). Honest bound: TIP decisions exist only for ≤8-score tiles (F-3 residual), so auto-mode profiling is per-block; tier granularity per D-024 (quasi-static per store run / read / block). |
| D-023 | Absolute Layer-3 gate: e2e error ≤ 5% of value scale (CQ-8/CQ-4+) | golden `derive_and_gate` d023_abs + T1 floor register + mutation gate | `d023_abs enforced on 80 CQ-8/CQ-4+ cases`; `PASS every D023-T1 floor register entry measured above 5%`; `PASS mutant 'value-lane-30pct-bias' caught by 'd023_abs' gate` | **PROVEN** (golden-level; the RTL tile is checked bit-exact against that gated golden in l3) |
| D-024 | In-tile runtime tiers = KVQ tier bank (3 verified engines + live mux; host/TIP-auto select; INFO_TIER build truth; documented granularity) | golden section E (uniform==single-tier bit-exact ×3 tiers, run composition, load-bearing calib map); smoke cq4p tile (grouped keys + FLUSH + mask, 2-round D-020 rerun); l3 `adv_T1_cq4p`/`adv_outlier1000_cq4p`/`tip_auto_mixed` (+ per-engine INFO_TIER routing checks in EVERY case's phase A); fail-first probe logs | golden `PASS tier_map uniform == single-tier (CQ-8/CQ-4/CQ-4+)`; smoke `TOPSMOKE RESULT: cycles=106603 checks=2176 errors=0`; l3 `adv_outlier1000_cq4p … checks=17825 errors=0`; auto-mode routing `KVP INFO_TIER` 2/1 per block inside `tip_auto_mixed … errors=0` | **PROVEN** (2026-07-09) — within the documented granularity (quasi-static per run/read/block; mid-stream switching is a host contract violation, listed infeasible in the l3 manifest) |

## 2. Contract clauses C-1..C-5

| ID | Clause | Test / suite / assertion | Verbatim evidence | Status |
|---|---|---|---|---|
| C-1 | INT4 quant: symmetric, s=max(amax/qmax,EPS), clamp [-8,7], RNE, fp16 scales, fp32 dequant | golden contract suite (9/9 vectors); kve/kvq parity; EPS-clamp + RNE-tie directed content; seam feeder mirrors C-1 | golden 9× `PASS`; kvq sb directed `EPS-clamp hits`, `116 RNE-tie hits` in coverage; seam mutant `M2_fq_eps_floor_off_by_one: DETECTED` | **PROVEN** — with a documented reachability bound: the −8 code (qmin) and the sat clamp are **unreachable through the engine top** (scale derives from the same data's amax; exhaustive fp16 sweep proof in `verif/kvq/sb`). The clamp is contract-conformant dead logic there; feeder-side C-1 (seam) exercises its own domain. |
| C-2 | One rounding rule per stage; RNE in quant + requant; ASU truncation-only inside LUT interp, final RNE to Q1.15 | golden C-2 properties; mxe mutation 3; seam M1/M3; asu M3 (rmsnorm RNE tie both parities) | `MUTANT M3_rms_rne_to_half_up: DETECTED … caught exactly at lane 9, the engineered EVEN-parity tie`; golden `PASS RNE sat scale=65535 shift=31` etc. | **PROVEN** — caveat: §6's "≤2 ULP" softmax acceptance holds at the shipping SCORE_FRAC=0 but measures up to 6 ULP at SCORE_FRAC=10 (golden-model property, ASU RESULT §7.1; see L-A1). |
| C-3 | INT32 accumulators; K capped 2048; legality reject; golden checks true bound | golden (`worst-case |acc| fits INT32`, `K>2048 rejected`); mxe K=2048 + all-(−128) extremes + illegal rejects | mxe directed: `all −128 at K=K_MAX (acc=+2²⁵)`; coverage `k_eq_2048 20` | **PROVEN** |
| C-4 | Array sees one dtype (INT8); dequant/requant at feeders | `verif/seam` feeder suite; `verif/top/l2` chain (b): real `kvq_engine` fp32 → `seam_feeder_quant` → real `mxe_top` INT8, all 3 tiers, D=64 + D=128 | l2 `run_l2b_t0/t1/t2 … errors=0` (CQ-8/CQ-4/CQ-4+), `run_l2b_calib_d128.log: checks=804 errors=0` | **PROVEN** (W4 weights clause is v1.1 scope — N/A in v0.1) |
| C-5 | Q·K̂ᵀ accumulates INT32; ASU consumes INT32 scores directly; TIP on \|INT32\| | l2 chain (a) mxe-OS→score-dequant→ASU; chain (c) fork→{ASU,TIP}; v0.3/tip at W=32 with INT32_MIN/MAX tiles | `run_l2a_storm.log: checks=3247 errors=0`; `run_l2c_storm.log: checks=364 errors=0`; tip directed `|INT32_MIN| = 2^31 exact ties at T∈{1,2,4,8,16}` | **PROVEN** |

---

## 2b. Feasibility register F-1..F-5 (tile-level) — v0.2 grades

Every F-item is fenced by BOTH a durable fail-first log (pre-fix failure, survives
`make clean`, parsed by gen_status.py) and a kept-passing (or, for open items, absent)
regression in the standing suites.

| F | Was (v0.1) | v0.2 grade | Evidence (fail-first → kept-passing, all re-run in §0e) | Residual |
|---|---|---|---|---|
| F-1 | T > 64 per job stalls (`apex_scale_quant`/`seam_score_dequant` buffers) | **CLOSED** | `prefix_f1_keptfail_before_fix.log` (`drive_cs stall @928783` on a T_ROW_MAX=64 build) → `calib_d64_T128`/`adv_outlier1000` (T=128), `calib_d64_T70`, `calib_d128_T100` all `errors=0` | T > 128 out of envelope (T_ROW_MAX=128, bounded by ASU SM_ROW_MAX=1024); l3 manifest `infeasible` entry |
| F-2 | CQ-4/CQ-4+ unreachable at tile level; INFO_TIER misleading; grouped keys/FLUSH/mask unplumbed | **CLOSED** (D-024 tier bank) | `prefix_f2_keptfail_before_fix.log` (`KVP 0c timeout` — tier select unwired) → smoke cq4p tile + l3 `adv_T1_cq4p` (tile-level D-008 FLUSH) + `adv_outlier1000_cq4p` (8 full G=16 key groups) `errors=0`; INFO_TIER = build truth in every case's phase A | one mask ROM per BUILD: b128 ships maskless → INFO_TIER=0x3, CQ-4+ degenerates to CQ-4 there (D=128 CQ-4+ covered at L2 chain b); tier switching quasi-static per D-024 granularity |
| F-3 | sd-frame/TIP-frame stickies unclearable; TIP silent for T>8 | **sticky half CLOSED** | `prefix_f3_keptfail_before_fix.log` (`[ESTK] got 00000001 exp 00000000`, 0x58 reserved no-op) → ERR_STICKY window 0x58 (RO + W1C, set-wins, bit 14 pulses TIP frame_err_clear); every l3 script + smoke ends set→W1C→verify-zero | **OPEN half:** TIP produces NO decision for score tiles >8 (structural BLOCK_N=8); TIP-auto profiling is per-block 8-score tiles |
| F-4 | no tile-level KVQ eviction/capacity test | **OPEN** | KVQ_DEPTH=256 in l3 builds holds the full 2·T=256 records — capacity never stressed at tile level; `sram_full` occupancy covered at block level only (`verif/kvq/sb`) | whole item — v0.3 candidate |
| F-5 | D=128 structurally broken (a: 4-bit nb truncation; b: PAT_D column aliasing 8..15→0..7) | **CLOSED** | `prefix_f5a_keptfail_before_fix.log` (`drive_g stall @400911`) + `run_unit_stagebuf_mutant.log` (`[PAT_D sel=8 beat=0] got e0c0…`) → `bug_d128_stagebuf_nb` PASS, `tb_stagebuf_patd` unit (184 checks, all 16 blocks) in every `make all`, smoke D=128, l3 D=128 cases | none (D=128 inside the envelope; b128 mask residual is F-2's) |

## 3. Known-limitations register (consolidated honest caveats)

Tile-level (the v0.2 envelope — from the feasibility register §2b / `verif/top/l3`
`gen_l3_vectors.py`, `apex_top.sv` scope boundary, and the closure passes):

- **L-T1 (F-1) — CLOSED 2026-07-09.** The per-job attention-row cap is now the
  `apex_top` envelope parameter **T_ROW_MAX = 128** (sizes `seam_score_dequant`
  N_MAX and the `apex_scale_quant` per-job buffers; the S-4 P-requant job is T
  columns). The named golden cases are replayed FULL LENGTH in l3 as kept-passing
  regressions: calib d64_T128 (T=128), adv/outlier1000 (T=128), calib d64_T70
  (T=70), calib d128_T100 (T=100, D=128) — all bit-exact (`errors=0`). Fail-first
  evidence: the T=128 case against a T_ROW_MAX=64 build stalls
  (`drive_cs stall @928783`, `verif/top/l3/logs/prefix_f1_keptfail_before_fix.log`).
  Residual: T > 128 is out of envelope (documented in the l3 manifest register).
- **L-T2 (F-2) — CLOSED 2026-07-09 (D-024).** The tile now instantiates the KVQ
  tier bank: grouped keys reachable via `rt_kv_user` (tuser=0) + WRITE_ADDR group
  sequencing, D-008 FLUSH reachable through CSR 0x28, OUTLIER_K/MASK_FILE plumbed
  as tile parameters (b64 verification builds carry a real {5,50} mask ROM), and
  CSR INFO_TIER reports the build TRUTH. Kept-passing regressions: smoke cq4p run,
  l3 `adv_T1_cq4p` (T=1 flush) / `adv_outlier1000_cq4p` (T=128, 8 full G=16
  groups). Fail-first: `verif/top/l3/logs/prefix_f2_keptfail_before_fix.log`.
  Residuals: one mask ROM per build (b128 ships maskless → INFO_TIER=0x3, CQ-4+
  degenerates to CQ-4 there; D=128 CQ-4+ covered at L2 chain b); tier switching is
  quasi-static per D-024's documented granularity.
- **L-T3 (D-022 gap) — CLOSED 2026-07-09 (D-024).** TIP now DRIVES the tier:
  accepted td_* decision beats write apex_top's 128-entry `auto_tier` map;
  TIER_CTRL.tip_override selects auto mode (per-block via rt_tip_blk). Demonstrated
  end-to-end in l3 `tip_auto_mixed` (outlier blocks→CQ-4+, benign→CQ-4, bit-exact
  vs `attention_core(tier_map=…)`; decisions checked bit-exact vs the verif/tip
  golden replica via ETIPT; map read back through IMPORTANCE_BASE). Residual: TIP
  produces decisions only for ≤8-score tiles (F-3 residual), so profiling is
  per-block 8-score tiles, and the auto map updates only through such tiles.
- **L-T4 (F-5) — CLOSED 2026-07-09.** F-5a: `aj_nb/wj_nb/job_nb` widened to the
  DERIVED width `NB_W = clog2(D/8)+1` end-to-end (apex_stage_buf, apex_top ports,
  both TBs) so nb = BPR = 16 is expressible. F-5b: `apex_stage_buf` PAT_D indexes
  the column block with the full legality-checked `sel` (was `sel_q[2:0]`, aliasing
  blocks 8..15 onto 0..7). Kept-passing regressions: `bug_d128_stagebuf_nb` (the
  exact former wedge, now bit-exact PASS), directed unit TB
  `verif/top/l3/tb_stagebuf_patd.sv` (all 16 PAT_D blocks byte-exact; kills the
  reverted-index mutant at `sel=8 beat=0`), smoke D=128 full chain, l3 named
  d128_T100 + 2 random D=128 full chains.
- **L-T5 (F-3) — sticky half CLOSED 2026-07-09.** T > 8 tile replays still raise
  the documented sd-frame + TIP-frame stickies by construction (chunked `res.last`
  + BLOCK_N=8 TIP framing), but every per-source sticky is now latched in the tile
  ERR_STICKY window (0x58: RO read + W1C, set-wins; bit 14 also pulses TIP's
  frame_err_clear so the one block-internal sticky with a clear pin stays
  coherent) — every l3 script and smoke ends with set→W1C→verify-zero. Fail-first:
  `verif/top/l3/logs/prefix_f3_keptfail_before_fix.log` (`[ESTK] got 0001 exp
  0000` on the pre-fix tile). RESIDUAL: TIP still gets no decision when the score
  tile exceeds 8 (structural, BLOCK_N=8); documented in the l3 manifest.
- **L-T6 (F-4)** KVQ eviction/capacity at tile level: KVQ_DEPTH raised to 256 in L3
  builds; §8's "eviction paths" have no tile-level test (kvq sb covers `sram_full`
  occupancy at block level only).
- **L-T7 (host-sequenced boundary)** v0.1 tile is host-driven phase-by-phase:
  descriptors/weights, glue job commands, seam scale composites (fp16 taps out, f32
  composites back — scale metadata crosses the host loop; tensor data never does),
  KVQ AXI-Lite control, epilogue calibration. The autonomous layer-walker is v0.2.
- **L-T8** apex_top smoke is **three directed micro-cases** (D=64 CQ-8, D=128
  CQ-8, D=64 CQ-4+ grouped-keys+FLUSH+mask; T=8) + soft-reset rerun each; breadth
  at tile level comes from l3's 27 cases (CQ-8 / CQ-4+ / TIP-auto mixed tiers,
  D∈{64,128}, T≤128 envelope).

Golden / numerics:

- **L-G1 (D-022)** CQ-4 is out-of-quality-budget on outlier-bearing data: documented
  exceedances 6.8%–25.7% of value scale (4-entry register, gate fails on NEW or
  VANISHED entries).
- **L-G2 (D-023-T1)** T=1 contexts at CQ-4+ hit the single-token INT4-V C-1 floor
  (7.1% > 5%); hard-gated at 8% with a floor register; use CQ-8 for T=1.
- **L-G3 (ASU §7.1)** §6's "≤2 ULP" softmax acceptance: 2 ULP at SCORE_FRAC=0 (at the
  budget edge), up to 6 ULP at SCORE_FRAC=10 — a golden fixed-point-scheme property.
  If integration ever feeds SCORE_FRAC>0, re-derive the §6 budget. (Note: `apex_top`
  instantiates the softmax **f10 config**.)
- **L-G4 (ASU §7.2)** RMSNorm EPS_INT=1.0 (raw-integer units) diverges from the
  arbiter's ε=2⁻¹⁴ value-units reference (≤~3% dev full-scale, up to 87% on tiny-norm
  rows). Deliberate + documented in the module header, but it is a numerics-contract
  decision living outside ARCHITECTURE.md — needs a D-number.

Per-block honest scope (from the RESULT.md files, deduplicated):

- **L-A1..A4 (ASU, RESULT §7/§8)** softmax is two-pass with a 4 KB row buffer,
  SM_ROW_MAX=1024 < K_MAX=2048 (a >1024-key row cannot pass as parameterized) and
  ~33 cycles/element emission (divider) — needs an explicit integration decision;
  gamma stream has no framing (a rejected row's gamma leaks into the next row —
  SEQ/XBR alignment rule needed); `y_sat` clamp structurally unreachable at D≤128;
  untested: SM_ROW_MAX≠1024, SCORE_FRAC∉{0,10}, sub-3-cycle reset glitches, Icarus.
- **L-M1..M4 (MXE, RESULT §7)** `desc` interface has no skid (documented reg-stage
  deviation from §5); `busy` covers output-pending only — input-skid beats survive
  into the NEXT job (SEQ must not overfeed); `desc.mode_os` ignored (opcode
  authoritative) — SEQ-level assert wanted; OS acc=1 chains onto ANY prior resident
  state (chain hygiene is SEQ/software responsibility).
- **L-M5 (D-004 mutation bound, RESULT §8.1, NEW v0.2)** the broadcast-wall mutation
  gate kills the *realistic* run1/run2-style wall (broadcast + combinational column
  adders); an ADVERSARIAL wall with per-column delay lines mimicking the internal
  timing law would be caught only by the P5 glitch-hop-locality test, which requires a
  physical act path between adjacent columns. The discriminator is strong evidence,
  not a structural proof.
- **L-M6 (D-005 measurement scope, RESULT §8.2, NEW v0.2)** load-under-compute overlap
  is measured via whitebox TB counters in `tb_mxe_perf`, NOT via the CSR `PERF_*`
  window (§7's "load/compute/drain per block" counters are not wired to MXE phases at
  tile level) — the run2 pathology is now *fixed and gated* at block level but not yet
  *measured through the tile CSR*; v0.3 CSR work.
- **L-K1..K5 (KVQ, RESULT/audit §5-6)** Inf/NaN fp16 in outlier lanes untested
  (contract vectors are finite-only — spec note wanted); `read_req` while busy outside
  ST_IDLE still dropped (upstream semantics, documented); soft-reset collision timing
  is deterministic for this RTL (drift is detected, not absorbed); late-`tlast` framing
  desync detectable only via STATUS; reset-crossed same-cycle full-token replay not
  exercised (beat-coincident tactic covers the accept cycle).
- **L-P1..P3 (TIP, RESULT §7)** short tiles use N (not actual length) in the ratio
  test — partial tiles skew FP16 at KV-length edges; importance update rule is
  implementer-defined (deserves a D-number); THRESHOLD/s_blk sampled quasi-statically
  (CSR writes must land between tiles). N=4096 exhaustive-per-T not swept (static
  bound proof + F-2 guard instead).
- **L-V1 (V0/kve)** upstream-top partial-group and D=128 grouped-key parity rests on
  upstream's `cq_key_path` TB + the C++ reference (top FSM couldn't reach it pre-D-008);
  closed for APEX by `rtl/kvq` + L2.
- **L-V2 (V0/skid)** skid verified standalone at WIDTH=65, single clock, 2-state, no
  formal proof; run1's GEMM core NOT re-verified beyond reproducing its own suite.
- **L-V3 (V0/rsqrt)** header latency doc off-by-one (31, not 30) — D-018 pins it;
  `in_valid` while busy silently dropped (wrapper must serialize — it does, verified).
- **L-S1 (cross-simulator)** Icarus cross-checks exist only at V0 (kve CQ-4+ config,
  rsqrt, skid, tip upstream TBs); all Layer-1/2/3 suites are Verilator-only (D-012
  names Verilator primary — recorded, not a violation).
- **L-S2 (sign-off scope)** simulation + assertion + coverage only; no synthesis,
  timing, CDC, X-prop, or formal anywhere (per §0 scope).
- **L-S3 (lint)** `-Wall` clean with waivers scoped to frozen `apex_pkg.sv` + vendored
  files only, on every new-RTL suite (verified in each fresh run's `LINT CLEAN` line);
  vendored-file warnings triaged individually in V0 RESULT §6 (two latent
  parameterization hazards documented: `residual_buffer` cnt==G alias — fixed by
  D-016(c) — and `sram_wr_addr` truncation if SRAM_DEPTH < KEY_GROUP, not exercised).

## 4. §8 layer scorecard

| Layer | Required by §8 | Delivered | Verdict |
|---|---|---|---|
| L0 foundation | V0.1..V0.5 before new RTL | all five PASS (`verif/v0/`), D-015..D-019 minted | **DONE** |
| L1 per-block | golden per block, directed+CR, SVA always, coverage buckets, mid-op reset per block | MXE/KVQ/ASU/TIP/CSR/SEQ/seam: independent suites, all coverage-gated, mutation-gated, mid-op reset per block (incl. the KVQ D-020 loop) | **DONE** |
| L2 pairwise | MXE↔ASU, KVQ↔MXE, TIP↔KVQ-control chains + stalling sweeps | chains a/b/c fresh-green: 18,863 checks, all tiers, D=64+128, FLUSH, resets, storms | **DONE** (TIP↔KVQ realized as score-fork→TIP decision chain; the tier-select *actuation* is L-T3) |
| L3 end-to-end | full decode layer vs float64 golden, all 3 tiers × {D=64,128} × full/partial-group/eviction | golden: FULL matrix (120 cases, D-023 gate + section E tier maps). RTL tile: 27 cases green — **CQ-8 / CQ-4+ / TIP-auto mixed CQ-4|CQ-4+, D∈{64,128}, T≤128** (F-1/F-5 closed, then F-2/D-022-actuation/F-3-sticky closed 2026-07-09; 135,241 checks incl. tile-level D-008 flush, full key groups, TIP-driven mixed tiers); remaining: no tile-level eviction path (F-4), D=128 CQ-4+ maskless (F-2 residual, covered at L2) | **PARTIAL** — golden side complete; tile side now fenced only by F-4 + L-T7 (+ the named F-2/F-3 residuals) |

**Bottom line (v0.2, sealed by the §0e from-clean reproduction):** every pinned
decision D-001..D-024 and contract clause C-1..C-5 is **PROVEN at its own scope**
(29/29); the matrix carries **zero PARTIAL / zero UNTESTED** rows. The three v0.1
PARTIALs were closed 2026-07-09: D-004 by the `verif/mxe/struct` discriminator +
broadcast-wall mutation kill (§0c), D-005 by the `mxe_ctrl` load-under-compute
change + `verif/mxe/perf` overlap gate with its kept-failing sequential-baseline
regression (§0c), and D-022's actuation half by the D-024 tier bank + `tip_auto_mixed`
(§0d). The **v0.2 tile envelope** is: an all-tier (CQ-8 / CQ-4 / CQ-4+ / TIP-auto
mixed per-block), D∈{64,128}, T≤128-per-job, host-sequenced attention tile — 135,241
L3 checks + 3 smoke tiles bit-exact vs the golden chain, with grouped keys, tile-level
D-008 flush, a real outlier-mask ROM (b64), truthful INFO_TIER, W1C per-source
stickies, a structurally-discriminated systolic MXE, and load-under-compute overlap
(up to 1.32x / full load hiding vs the pinned sequential baseline). `rtl/apex_pkg.sv`
untouched throughout — no contract field changed, APEX_VERSION stays `0x0001_0000`.
Remaining fences (the honest v0.3 backlog): F-4 (tile-level eviction/capacity), L-T7
(host-sequenced boundary — the autonomous layer-walker), F-3's open half (TIP silent
for >8-score tiles), F-2's mask residual (one ROM per build; b128 maskless), L-M6
(MXE-phase PERF counters through the tile CSR), plus the standing L-G/L-A/L-M/L-K/L-P/
L-S entries. See STATUS.md (generated) for the current-run roll-up.
