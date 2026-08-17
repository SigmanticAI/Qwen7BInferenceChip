# APEX v0.2 — STATUS

**GENERATED FILE — do not hand-edit.** Written by `scripts/gen_status.py`,
which parses the actual suite logs and runs the golden gate live
(ARCHITECTURE.md §8 anti-fabrication rule). Regenerate with:
`python scripts/gen_status.py`.

- Generated: 2026-07-26 23:58:31
- HEAD: `e134f29 Merge gate-integrity @ d70c4fd — 5-tool PATH sweep (52 files), tools.mk (SHELL + version pins, Icarus FLOOR), check_gate_hygiene.sh, mutation-macro rework` (+1 working-tree entries)

## 1. Gate roll-up

| Suite | What | Verdict | Evidence (verbatim from parsed log / live run) | Log freshness |
|---|---|---|---|---|
| golden (LIVE) | golden contract + compute + attention gate | **PASS** | `APEX COMPUTE GOLDEN: ALL PROPERTIES PASS`<br>`APEX ATTENTION GOLDEN: ALL 120 CASES WITHIN DERIVED BUDGET (hard + statistical), ALL SELF-CONSISTENCY CHECKS PASS`<br>`GOLDEN SUITE: contract + compute + attention + transformer + plumbing7b + effbits + masksem ALL PASS` | ran now |
| top/smoke | apex_top tile smoke (D=64, D=128 AND CQ-4+ tier-bank full chains, T=8 + D-020 soft-reset rerun each; F-2 grouped keys + D-008 FLUSH + mask in the cq4p run) | **PASS** | `TOPSMOKE RESULT: cycles=67507 checks=2176 errors=0`<br>`TOPSMOKE PASS`<br>`TOPSMOKE RESULT: cycles=209741 checks=4240 errors=0` | 2026-07-22 10:27 |
| top/l2 | Layer-2 pairwise chains a/b/c (all tiers, D=64+128, FLUSH, resets, storms) | **PASS** | `L2 COVERAGE GATE: PASS`<br>`total checks across required runs: 18863` | 2026-07-26 23:33 |
| top/l3 | Layer-3 attention replays through the REAL apex_top (CQ-8/CQ-4+/TIP-auto mixed tiers, D=64+128, T<=128, incl. the D-027 CSR-mask d128 CQ-4+ case) + tile mutation gate + F-5 stage-buf unit regression | **PASS** | `L3 RUNS: all 28 cases passed`<br>`STAGEBUF PATD PASS`<br>`total checks: 152883` | 2026-07-26 23:57 |
| l3/fail-first | F-register regression-first evidence: durable pre-fix kept-failing logs (F-1/F-2/F-3/F-5a) + F-5b unit mutant kill (verif/top/l3/logs — survives make clean; the kept-PASSING flips run inside top/l3 'all') | **PASS** | `[9287830000] %Fatal: tb_apex_l3.sv:540: Assertion failed in tb_apex_l3.drive_cs: drive_cs stall @928783`<br>`[10500335000] %Fatal: tb_apex_l3.sv:740: Assertion failed in tb_apex_l3.kv_poll: KVP 0c timeout (got 00000000) @1050033`<br>`[370000] %Error: tb_apex_l3.sv:750: Assertion failed in tb_apex_l3.chk32: [ESTK] got 00000001 exp 00000000` | 2026-07-21 12:56 |
| top/l4 | IB-LAYER L4 harness: level-propagation probe (every exported LAYER level observed at its consumer, hierarchically) + the segment-2 host-composition suite (RMSNorm-1 front-half through the REAL apex_top at CFG_D=128) + 3-mutant signature gate | **PASS** | `L4LEVELS RESULT: checks=27 errors=0 -> PASS`<br>`L4COMPOSE RESULT: cycles=37120 checks=67 errors=0`<br>`L4COMPOSE PASS (segment 2 RMSNorm-1 front-half; h8 deferred to segment 1, IB_LAYER sec 3c)` | 2026-07-26 23:24 |
| top/wcomp | F6(i) per-KV-head composite scale-cache BANK (apex_wcomp_bank: N_ENG=4 seq_walker_comp instances behind eng_sel) — routing + isolation proof, count-pinned 203 checks at D=64 AND D=128, 4 signature-required mutants (arithmetic itself is seq_walker's comp sweep; tile elaboration at GQANE=4 is kvq/gqa's lint_top_on) | **PASS** | `WCOMP BANK d64: checks=203 fails=0`<br>`WCOMP BANK d128: checks=203 fails=0`<br>`mW1: CAUGHT (routing; signature ['FAIL A cs g=1', 'FAIL A cs g=2'])` | 2026-07-26 23:33 |
| top/bias | IC-BIAS projection-bias seam (S-2 gap B): apex_proj_bias unit golden gate (element model, exact-or-refused, bias=+0 equivalence vs apex_scale_quant) + l4_bias q/K/V biased projections bit-exact vs decoder_layer_fx BUS_ON through the real tile + 4-mutant signature gate | **PASS** | `PBIAS RESULT: cycles=8590 jobs=13 checks=2037 errors=0`<br>`PBIAS PASS (golden element model + exact-or-refused + bias=+0 equivalence vs apex_scale_quant)`<br>`BIASTILE RESULT: cycles=115891 checks=2325 errors=0` | 2026-07-26 23:25 |
| top/wg3 | W-G3 flagship (partial-layer walk + fuel-fed projection, real Qwen2.5-7B s019/L19): case renders (C/P/R/F self-checks), FENCED mask-0x03D H=28 walk at unit level, single-head hd128 tile replay host AND WALKED (fmt=1), GQA-4 K-rope with the D-030 arbiter (legacy-pinned arm kept RED), fuel tile leg + DDR leg (perturbed-DDR arm kept RED), topology A/B witness (control green, pre-F6(i) single-cache arm kept RED). Inputs are the gitignored walkgold stage-A/B artifacts (regenerate per the Makefile header) | **PASS** | `WG3 CASE RENDER: PASS (C1 emissions byte-identical to stage B; C2 160 records from golden K_f16/V_f16; C3 fuel block byte-identical to the full image `<br>`WG3-TOPO[control-Neng-caches] RESULT: cases=1 checks=16139 errors=0 kvw=9528 heads=28`<br>`WG3-TOPO PASS` | 2026-07-26 23:35 |
| mxe/sb | MXE independent scoreboard suite (6 regimes, SVA pack, resets) | **PASS** | `COVERAGE: all required buckets hit (38 required, 40 total)`<br>`TB PASS: 48 legal jobs, 13 illegal descriptors, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=1)` | 2026-07-22 10:25 |
| mxe/struct | D-004 systolic structural discriminator (whitebox timing law, glitch hop locality, broadcast-wall mutation check) | **PASS** | `STRUCT PASS: 20800 whitebox timing checks — (r+c)-offset act arrival, per-hop psum staggering, column-staggered acc writes, single-cycle glitch hop lo`<br>`MUTATION CHECK PASS: broadcast-wall mutant passes tb_mxe_smoke but is KILLED by tb_mxe_struct (D-004 discriminates)` | 2026-07-22 10:25 |
| mxe/perf | D-005 load-under-compute perf evidence (overlap>0 + cycle win vs pinned sequential baseline; kept-failing overlap regression on the baseline) | **PASS** | `PERF TB PASS: 4 jobs bit-exact, overlap>0 on every multi-chunk job (D-005)`<br>`PERF GATE: PASS — overlap_cycles > 0 on every multi-chunk job and total cycles strictly below the sequential baseline (D-005)`<br>`PERF OVERLAP MISSING [K64_M16]: multi-chunk WS job ran with ZERO load-under-compute overlap (D-005 half (b) absent)` | 2026-07-22 10:25 |
| mxe/w4 | B3 W4 nibble-packed weight feeder (wfeed W4 scoreboard: 6 run regimes incl. reset storms, coverage gate, 5-mutant gate) | **PASS** | `COVERAGE PASS: all required buckets hit (26 buckets, 6 logs)`<br>`TB PASS: 17 legal jobs, 2 illegal jobs, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=201)`<br>`MUTATION GATE: 5/5 mutants killed — checkers proven live` | 2026-07-26 23:19 |
| mxe/w4b | D-031 W4B given-scale feeder (pkg vector gate, 6 scoreboard regimes, EXHAUSTIVE 4,063,104-point operand sweep, 5-mutant gate; complete per-gate evidence committed at logs/run_all_full.log) | **PASS** | `W4B PKG GATE: 31024 vectors, ALL PASS`<br>`TB PASS: 18 jobs, 3 illegal, 0 resets, 0 errors (bp=1 stall=1 gs=0 seed=51)`<br>`W4B EXHAUSTIVE SWEEP: 4063104 operand points, ALL PASS` | 2026-07-26 23:21 |
| asu/sb | ASU independent suite (softmax + rmsnorm, 13 configs, mutation gate) | **PASS** | `COVERAGE: all required buckets hit (60 required, 64 total)`<br>`MUTATION GATE: 4/4 mutants detected — checkers proven live`<br>`TB PASS: 20 rows, 5 rejects, 0 mid-op resets, 0 errors (bp=1 stall=1 g=0 seed=21)` | 2026-07-26 23:37 |
| asu/wide | WIDE-D RMSNorm (C-RMSW): bit-exact vs golden rmsnorm_fx_wide at RMS_D_MAX=3584+8192 (MU/S k-sweep 2..28, sum2=2^27 corner, rejects, mid-op resets), params provenance gate, 4-mutant gate | **PASS** | `WIDE MUTATION GATE: 4/4 mutants detected — checkers proven live`<br>`TB PASS: 4 rows, 1 rejects, 0 mid-op resets, 0 errors (D_MAX=8192 bp=1 stall=0 g=0 seed=37)` | 2026-07-26 23:41 |
| asu/swiglu | SwiGLU + layer-deq units (gate*silu composite and W4-grade dequant on real decoder-layer tensors: fast/bp/reset(+storm) regimes, 9-mutant gate) | **PASS** | `SWIGLU RESULT: jobs=23 beats=289 jerr=3 ferr=2 resets=0 mismatches=0 -> PASS`<br>`DEQ RESULT: jobs=26 beats=4138 jerr=4 xerr=3 ferr=2 resets=0 mismatches=0 -> PASS`<br>`SWIGLU/DEQ MUTATION GATE: 9/9 mutants detected — checkers proven live` | 2026-07-26 23:18 |
| tip/sb | TIP independent suite (32 thresholds, N=64/4096, resets, framing) | **PASS** | `COVERAGE: all required buckets hit (67 required, 67 total)`<br>`TB PASS: 1200 tiles, 3 overrun frames, 0 mid-op resets, 1 clears, 1200 decisions checked (+0 sacrificial discarded), 0 errors (N=64 bp=1 gap=1 seed=10` | 2026-07-22 10:26 |
| tip/smoke | TIP smoke (D-011/D-017): frozen-143 replay, threshold sweeps T=1/5/31, clamp T=0, importance accumulators, V0.3-F-2 framing-guard regression (N=64 + N=4096 builds) | **PASS** | `tiles=3024 decisions=3024 frame_err_pulses=0 fails=0`<br>`ALL TESTS PASSED` | 2026-07-26 23:30 |
| kvq/smoke | KVQ implementer smoke (V0-parity x8 + 4 behavioral regressions; baseline bug-repro halves retired with the third-party checkout — see smoke Makefile header) | **PASS** | `KVQ PARITY [d128_cq4]: PASS (bit-exact records + fp32 readback, irq=0)`<br>`KVQ PARITY [g128_cq4]: PASS (bit-exact records + fp32 readback, irq=0)`<br>`KVQ STALL [stall_kvq]: PASS — all beats delivered bit-exact under backpressure (irq=0)` | 2026-07-24 15:17 |
| kvq/sb | KVQ independent scoreboard suite (records + readback, soft-reset, B-1..B-4 closed) | **PASS** | `COVERAGE GATE: PASS — all required buckets closed`<br>`CONFIG cq4: checks=4324 fails=0`<br>`CONFIG d64: checks=3715 fails=0` | 2026-07-22 10:22 |
| f2/firstlight | AWS F2 first light: full tile + OCL bridge, Vivado zero-error P&R, timing MET at 250 MHz shell clock (VU47P), AGFI loaded on an f2.6xlarge, BAR0 probe of the verified CSRs ALL PASS (register first light only) | **PASS** | `PROBE: ALL PASS`<br>`AFI          0       agfi-0f7c93ffa798ecc3f  loaded            0        ok               0       0x10212415` | 2026-07-21 12:56 |
| f2sim | F2 stage-2 regops executor: the verilated cl_apex driven at its OCL AXI-Lite pins by the SAME regops files the F2 host runner uses — all 18 compiled S8 trace jobs bit-exact at tile_div=5 + behavioral-DDR AXI smoke (default and +ddr_stall). The IB-FUEL RED-mutant gate (control PASS + 3/3 fuel-path mutants RED) is rc-asserted by `make mutants` and not log-persisted. Logs are gitignored and removed by the lane's own clean -> NO-RUN then; rerun `make -C verif/f2sim run behsmoke` | **PASS** | `F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS`<br>`DDRBEH RESULT: checks=801 fails=0 -> PASS`<br>`DDRBEH RESULT: checks=801 fails=0 -> PASS` | 2026-07-26 23:58 |
| kvq/replay | S8 real-model RTL replay: Qwen2.5-7B 150-token artifact-trace fp16 K/V rows through the KVQ cores RTL, bit-exact vs the blobs the golden pipeline stored at generation time (durable logs — survive make clean) | **PASS** | `[replay-gen] SELF-CHECK OK: every stored blob field matches a fresh golden compression of the raw rows`<br>`CONFIG artifact: checks=382200 fails=0`<br>`CORES RESULT [artifact]: PASS` | 2026-07-22 10:27 |
| kvq/fparith | KVQ synthesizable integer fp-arith (cq_fp_pkg real->int rewrite: exhaustive golden sweeps, seeded errors, sv2v+yosys) | **PASS** | `DATAPATH INTEGER-MODEL PROOF: ALL PASS`<br>`FPARITH TB: ALL PASS (checks=8748634 fails=0)`<br>`MUTATION GATE: 2/2 seeded errors caught (mut_tie, mut_eps)` | 2026-07-22 10:19 |
| kvq/audit | KVQ D-020 adversarial re-audit (260 soft resets, -0.0, fix-revert mutants) | **PASS** | `COVERAGE GATE: PASS — all required buckets closed`<br>`TB PASS [rst_cq4]: resets=146 clean_v=140 clean_k=81 reads=514 peeks=514 bank_peeks=93 bank_rst=6`<br>`TB PASS [rst_cq4p]: resets=0 clean_v=12 clean_k=8 reads=45 peeks=45 bank_peeks=8 bank_rst=0` | 2026-07-22 10:17 |
| kvq/mask | S12/D-027 loadable-mask suite (ROM-vs-CSR record equivalence on golden-packer vectors, popcount/swap faults W1C, soft-reset persistence + ssid continuation, computed mask_valid + port mirror; 2 mutants) | **PASS** | `MASK GATE mask_rom: checks=800 fails=0`<br>`MASK GATE mask_rom: ALL PASS`<br>`MASK GATE mask_csr: checks=1887 fails=0` | 2026-07-26 15:33 |
| kvq/gqa | S4b per-KV-head KVQ banks (apex_kvq_gqa_bank at KVQ_GQA_NENG=4: golden-vector bank scoreboard + apex_top elaboration proof at GQANE=4 + 3 signature-required mutants) | **PASS** | `GQA BANK gqa4: checks=1859 fails=0`<br>`GQA BANK gqa4: ALL PASS`<br>`mS1: CAUGHT (hang; signature ['phase=info e1'])` | 2026-07-26 23:18 |
| seam | D-021 seam blocks (feeder-quant + score-dequant), 17 runs, mutation gate | **PASS** | `COVERAGE PASS: all required buckets hit (53 buckets, 17 logs)`<br>`MUTATION GATE: 4/4 mutants detected — checkers proven live`<br>`TB PASS: 10 legal jobs, 3 illegal jobs, 0 mid-op resets, 0 errors (D=128 bp=1 stall=1 seed=111)` | 2026-07-22 10:26 |
| seq | SEQ walker (D-006 serialization, aborts, rejects) vs MXE stub | **PASS** | `COVERAGE: all required buckets hit (39 required, 39 total)`<br>`TB PASS: 5 dispatched (5 done, 0 rejected, 84 discarded by abort/reset), 38 last-beats checked, 0 errors (lat=1 bp=1 gap=1 seed=401)` | 2026-07-26 23:14 |
| seq_walker | B1/D-029 layer-walker unit suite (walker1 score+pv vs the L3-extracted reference trace incl. refusal gate; walker2 real-7B H=28 layer walks at D=64/128; composite-unit sweeps + directed; check-equivalence sweep). 16 mutant kills asserted by `make all` (gates 1-4 echo-only: a surviving mutant fails the build — kvq/sb precedent) | **PASS** | `WALKER RESULT: cases=2 checks=166 errors=0 kvw=54 mxelegal=18`<br>`WALKER PASS`<br>`WALKER2 RESULT: cases=1 checks=22820 errors=0 kvw=3816 heads=28 mxelegal=16476` | 2026-07-26 23:29 |
| csr | CSR window (map, sticky/W1C, PERF counters at W=8/W=32) | **PASS** | `COVERAGE: all required buckets hit (61 required, 70 total)`<br>`TB PASS: 181 checked reads, 2 soft_reset + 2 flush pulses, 0 errors (PERF_W=32 seed=201)` | 2026-07-26 23:37 |
| audit_seq | SEQ adversarial attack audit | **NO-RUN** | —<br>NOTE: missing: build/run_attack.log | — |
| audit_csr | CSR adversarial attack audit (both PERF widths) | **NO-RUN** | —<br>NOTE: missing: build/run_w*.log | — |
| rope/smoke | RoPE swept-domain smoke (head_dim 64+128, pos 0..127, HF half-split; Verilator+Icarus) | **PASS** | `12288/12288 channel-pair vectors bit-exact vs apex_golden rope_fx` | 2026-07-22 10:25 |
| asu/silu | SiLU EXHAUSTIVE full 16-bit domain vs golden silu_fx (Verilator+Icarus) | **PASS** | `65536/65536 input patterns bit-exact vs apex_golden silu_fx` | 2026-07-22 10:12 |
| residual | residual_add fp16 smoke (12 directed + 20000 random; Verilator+Icarus) | **PASS** | `20012/20012 fp16 residual-add vectors bit-exact vs numpy fp16` | 2026-07-22 10:25 |
| rope/row | RoPE row unit (rope_row: full-row rotation with xbuf, fast/bp/reset/storm regimes at D=64 + bp at D=128, 6-mutant gate) | **PASS** | `ROPEROW RESULT: rows=364 beats=46208 errs=3 resets=0 mismatches=0 -> PASS`<br>`ROPEROW MUTATION GATE: 6/6 mutants detected — checkers proven live` | 2026-07-26 23:17 |
| misc/resfx | residual fixed-point unit + resid streaming unit (resfx: 2M-point golden file sweep; resid_unit: window/guard/corner cases under fast/stall/reset regimes; 3+4 mutant gates) | **PASS** | `RESFX RESULT: file=238091 insim=2000000 mismatches=0 -> PASS`<br>`RESFX MUTATION GATE: 3/3 mutants detected — checkers proven live`<br>`RESID RESULT: cases=8 jobs=18 elems=1551 ferr=3 werr=1 resets=0 mismatches=0 -> PASS` | 2026-07-26 23:15 |
| layer | decoder-layer composition TB (RoPE+SiLU+residual on real decoder_layer_fx stage tensors; Verilator+Icarus) | **PASS** | `LAYER COMPOSITION: ALL TESTS PASSED (t=2776)` | 2026-07-22 10:25 |
| v0/kve | V0.1/V0.2 per-channel INT4 codec KVQ re-verification (7 configs, tready bug->fix) | **RETIRED** | —<br>NOTE: third-party checkout removed (clean-room); behavior covered by kvq/smoke V0-parity + kvq/sb | — |
| v0/rsqrt | V0.4 rsqrt_unit re-verification (1.067M-op sweep, latency pin) | **RETIRED** | —<br>NOTE: third-party checkout removed; rsqrt covered inside asu/sb (D-018 latency pin) | — |
| v0/skid | V0.5 stream_skid standalone + run1 root cause | **RETIRED** | —<br>NOTE: third-party checkout removed; skid covered by every suite's SVA pack (D-007) | — |
| tip | V0.3 the ratio-test unit re-verification (W=32) | **RETIRED** | —<br>NOTE: V0.3-era standalone logs removed; TIP covered by tip/sb (67/67 buckets) | — |

## 2. D-022 tier-quality table (parsed from the live golden run)

| Tier | Cases | worst e2e/\|y\| | worst e2e/\|V\| | worst P-lane/\|V\| |
|---|---|---|---|---|
| CQ-8 | 40 | 3.503e-01 | 1.229e-02 | 1.218e-02 |
| CQ-4 | 40 | 3.867e-01 | 7.107e-02 | 1.193e-02 |
| CQ-4+ | 40 | 3.867e-01 | 7.107e-02 | 1.211e-02 |

Documented exceedance / floor register (gate fails on NEW or VANISHED entries):

- `DOC   D-022: adv/T1/CQ-4 e2e_abs=1.421e-01 = 7.1% of value scale (> 5%) — CQ-4 out-of-quality-budget on outlier-bearing data; TIP tier-select REQUIRED (D-022)`
- `DOC   D023-T1: adv/T1/CQ-4+ e2e_abs=1.421e-01 = 7.1% of value scale — single-token INT4-V C-1 floor (0.51·s_v ≈ 7.3%); 5% unattainable by contract, hard-gated at 8%; use CQ-8 for T=1 contexts`

## 3. LOC summary (working tree)

| Group | Files | Lines |
|---|---|---|
| apex_pkg | 1 | 94 |
| asu | 9 | 1586 |
| csr | 1 | 264 |
| golden (py) | 20 | 5884 |
| kvq | 14 | 2530 |
| misc | 3 | 393 |
| mxe | 10 | 1884 |
| rope | 4 | 647 |
| scripts (py) | 6 | 1386 |
| seam | 2 | 731 |
| seq | 5 | 2440 |
| tip | 3 | 536 |
| top | 12 | 4945 |
| verif SV/SVH | 73 | 27760 |
| verif py | 98 | 21043 |
| xbr | 1 | 152 |
| **total** | | **72275** |

## 4. Open items (mechanically derived)

Suites not PASS:
- audit_seq: **NO-RUN**
- audit_csr: **NO-RUN**

Known-limitations register (TRACEABILITY.md §3): 24 entries — L-T1, L-T2, L-T3, L-T4, L-T5, L-T6, L-T7, L-T8, L-G1, L-G2, L-G3, L-G4, L-A1..A4, L-M1..M4, L-M5, L-M6, L-K1..K5, L-P1..P3, L-V1, L-V2, L-V3, L-S1, L-S2, L-S3

