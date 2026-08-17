# WIDE-D RMSNorm verification (C-RMSW) — RESULT

**Date:** 2026-07-20 · **DUT:** `rtl/asu/asu_rmsnorm.sv` wide elaborations
(`RMS_D_MAX=3584` and `RMS_D_MAX=8192`) + `rsqrt.sv` + `stream_skid.sv`
**Contract:** `docs/design/WIDE_RMSNORM.md` (stage 4) · **Branch:**
`comp/wide-rmsnorm`
**Tools:** Verilator 5.044 `--binary --timing --assert -Wall` (waivers scoped
to frozen `apex_pkg.sv` only) · Python 3 + NumPy
**Golden arbiter:** `golden/apex_golden/attention.py` `rmsnorm_fx_wide` /
`_wide_mu` (C-RMSW, landed S6, gated by `make -C golden plumbing7b` §B).
Never edited to match RTL.

## Verdict: PASS — bit-exact vs the golden arbiter at every wide length

Full log: [`logs/run_all_2026-07-20.log`](logs/run_all_2026-07-20.log)
(the tee'd `make -C verif/asu/wide all`). Headline evidence, verbatim:

```
PARAMS CHECK: rtl/asu/asu_wide_rms_params.svh matches the golden arbiter (_wide_mu)
wide_random : 60 rows, 3 rejects, 16 narrow rows == smoke oracle, k coverage 2..28 complete
LINT CLEAN (-Wall, wide elaborations 3584 + 8192; waivers scoped to frozen apex_pkg.sv only)
TB PASS: 17 rows, 5 rejects, 0 mid-op resets, 0 errors (D_MAX=3584 bp=1 stall=1 g=0 seed=31)
TB PASS: 17 rows, 5 rejects, 0 mid-op resets, 0 errors (D_MAX=3584 bp=0 stall=0 g=0 seed=32)
TB PASS: 60 rows, 3 rejects, 0 mid-op resets, 0 errors (D_MAX=3584 bp=1 stall=1 g=1 seed=33)
TB PASS: 17 rows, 5 rejects, 0 mid-op resets, 0 errors (D_MAX=3584 bp=2 stall=2 g=2 seed=34)
TB PASS: 60 rows, 3 rejects, 0 mid-op resets, 0 errors (D_MAX=3584 bp=1 stall=0 g=2 seed=35)
TB PASS: 14 rows, 0 rejects, 14 mid-op resets, 0 errors (D_MAX=3584 bp=1 stall=0 g=0 seed=36)
TB PASS: 4 rows, 1 rejects, 0 mid-op resets, 0 errors (D_MAX=8192 bp=1 stall=0 g=0 seed=37)
MUTANT MW1_rms_rne_to_half_up: DETECTED (rc=1) — FAIL [N2 d=3584] cyc=36130: y[10] got 0xfcb1 exp 0xfcb0
MUTANT MW2_mu_off_by_one: DETECTED (rc=1) — FAIL [N7 d=3584] cyc=101325: y[1] got 0xf564 exp 0xeac8
MUTANT MW3_mu_shift_off_by_one: DETECTED (rc=1) — FAIL [N1 d=3584] cyc=14377: y[0] got 0xffde exp 0xffd0
MUTANT MW4_sumw_trunc_27: DETECTED (rc=1) — FAIL [N1 d=8192] cyc=33022: y[0] got 0x8000 exp 0xff00
WIDE MUTATION GATE: 4/4 mutants detected — checkers proven live
ASU WIDE: full suite passed (params provenance, bit-exact vs rmsnorm_fx_wide, rejects, resets, mutation gate)
```

## What was proven

- **Bit-exact vs `rmsnorm_fx_wide` `(y, r, nrm)`** per output beat (Q7.8
  exact), framing exact, `dbg_norm == nrm` — 189 legal rows across the two
  elaborations and 7 adversary configurations (backpressure none/75%/storm ×
  x-feed stalls × gamma lockstep/late/eager), all timing-invariant.
- **Row lengths covered:** D=3584 (random, extremes, all-zero, engineered
  ties, engineered μ-boundary); a **deterministic k sweep — EVERY (MU, S)
  table entry k = 2..28 gets at least one row** (generator-asserted:
  `k coverage 2..28 complete`), so a wrong table entry at any k cannot
  escape; the full narrow pow-2 sweep {1..128}: **19 narrow rows asserted
  equal to the smoke oracle `rmsnorm_fx` at generation time** (the
  regression link — `wide_directed: … 3 narrow rows == smoke oracle`,
  `wide_random: … 16 narrow rows == smoke oracle`).
- **The D=8192 all-(−128) corner** (`sum2 == 2^27` EXACTLY, the golden
  accumulator bound): runs in the `RMS_D_MAX=8192` elaboration,
  `(r,nrm) = (64,128) == the D=128 all-(−128) corner` (golden plumbing7b
  B-6 mirrored and asserted at generation).
- **Legality rejects, same observable as the frozen unit** (pulse + sticky,
  row consumed, no output/done): B 129 / B 200 / B 3520 (multiple of 64,
  not 128) rejected at `last`; O 3585 (last on the overflow beat) and
  O 3648 (FLUSH arm); O 8193 in the 8k build; 9 distinct reject records,
  22 reject executions across the run configs.
- **Mid-op resets:** 14 resets targeting EVERY reachable FSM phase
  (COLLECT/FLUSH/ISSUE/WAIT/EMIT/DRAIN via `dut.st`) + cycle-count aborts,
  clean-row recovery after each.
- **Provenance gate (`params-check`):** `asu_wide_rms_params.svh` regenerated
  from golden `_wide_mu` and diffed byte-identical — the durable home of the
  stage-1 gate (tables-check house pattern).
- **Checker credibility — 4/4 mutants detected:**
  - MW1 RNE→half-up on the SHARED emission line (M3 reuse): killed by a
    ±1-LSB tie divergence (engineered odd/even tie lanes guarantee this even
    when random rows lack ties).
  - MW2 `MU[k]+1`: killed by the engineered μ-divisor-boundary row
    (`sum2=10752` → den 1 (MU) vs 2 (MU+1); generator asserts the flip).
  - MW3 `>>(S+16)` → `>>(S+15)`: mean2 doubles — killed on the first row.
  - MW4 `SUM_W` formula → 27: killed ONLY by the `sum2 == 2^27` corner
    (wraps to 0 → `y[0] 0x8000 vs 0xff00`), proving the width formula is
    load-bearing at D=8192.

## Scope notes

- `r` (inv_norm) is not observable on the DUT interface; it is verified
  indirectly (every nonzero y lane is a bit-exact function of r) and the
  floor-sqrt directly via `dbg_norm` — same scoreboard shape as
  `verif/asu/sb`.
- The frozen D≤128 anchor (byte-identical logs, zero test edits) is stage 3
  of the contract, proven separately — see the stages-2+3 commit message
  (`verif/asu/smoke` + `verif/asu/sb` A/B vs `e578872`, all TB-emitted bytes
  identical, sb `mutants.txt`/`coverage.txt` byte-identical unfiltered).
- The frozen D≤128 elaboration's synthesis is unchanged and covered by the
  existing tile flows (S10a ECP5 ship config; F2/VU47P CL build).

## Synthesis probe (2026-07-20, follow-on flag resolved)

The contract §5 note flagged the comb-read `x_buf` as a possible BRAM
blocker (registered-read variant as follow-on). Probed standalone with the
house sv2v→yosys flow (yosys 0.66, wrapper `asu_rmsnorm #(.RMS_D_MAX(3584))`
+ rsqrt + skids): **both targets infer block RAM from the RTL as-is** —
yosys forms the synchronous read port by absorbing the registered read
address (`emit_idx`), a semantics-preserving mem transform. Verbatim:

```
mapping memory rms_wide_synth.u.x_buf via $__DP16KD_                (synth_ecp5 -abc9)
mapping memory $paramod\asu_rmsnorm\RMS_D_MAX=...E00.x_buf via $__XILINX_BLOCKRAM_TDP_   (synth_xilinx -family xcup)
```

| target | wide 3584 | narrow 128 (same flow, baseline) |
|---|---|---|
| ECP5 (`synth_ecp5 -abc9`) | **2 DP16KD**, 7 MULT18X18D, 2506 LUT4, 365 FF | 16 DPR16X4 (distributed), 4 MULT18X18D, 527 LUT4, 343 FF |
| US+ (`synth_xilinx -family xcup`) | **1 RAMB36E2**, 5 DSP48E2, ~1164 LUT1-6 + 36 MUXF9, 1638 cells, CHECK 0 problems | — |

Wide-vs-narrow ECP5 delta ≈ +2 BRAM, +3 mult, +~2k LUT4 (the 43-bit
variable shifter, the 64-entry μ ROM, 12-bit counters/legality) — ~10% of
the S10a ship tile's 19,368 LUT4 and 2 of 208 DP16KD. **Conclusion: the
registered-read variant is NOT needed for either target; retired.** The
sum2-export/external-r variant survives only as an area option. Scope:
yosys synthesis mapping evidence only — no P&R/timing, and the post-synth
netlist was not re-simulated (bit-exactness is proven on the RTL by this
suite; yosys's transform correctness is relied on as with every other
block in the flow).
