# E-6 REPLICATION — an independent driver reproduces the walked epilogue, bit for bit

**Date:** 2026-08-05 · **Subject tree:** `comp/prompt-b-c` @ 59a5ff7
("E-6 COMPLETE") · **Twin:** `verif/f2sim/obj_e6r_b64_ddr1` — a FRESH
`--Mdir` verilated from that tree (D=64 DMODEL=64 DM=896 GQA=2 QSTAGE=14,
DDR=1) · **Driver:** `scripts/fpga/f2/walk_fuel_layer.py run` — **PASS** ·
**Record:** `build/walk_fuel_layer/walk_fuel_layer_result.json` · sim only.

## Why this file exists

Two sessions implemented E-6 CONCURRENTLY against the same measured
evidence (STEP_MATRIX + the dead-relaunch stub). One landed
(`walk_fuel_proj.py rune6`, increments 7ed40ee..59a5ff7); this driver is
the OTHER one — an independently written emitter, golden derivation,
predicate set and discriminator search — re-pointed, after the landing, at
a fresh twin built from the LANDED tree. Same claims, different program
text, different capture path (produce-mode `erd` caps + a strict
zero-ops-of-any-kind window predicate checked on the artefact), different
discriminator third leg (a golden-searched RQ-pair delta instead of the
V-flip). It agrees to the cycle:

| walk | this driver (tile_div=2) | landed gate (tile_div=5) |
|---|---|---|
| {OPROJ, RES1, FPROJ} | 979,596 shell = **244,899 tile** | 2,448,960 shell = **244,896 tile** |
| {QKV, OPROJ, RES1, FPROJ} | 2,238,788 shell = **559,697 tile** | 5,596,970 shell = **559,697 tile** |

(tile = shell / (2·tile_div): the sim divider TOGGLES clk_tile every
tile_div shell posedges — `cl_apex.sv:82` "divide clk_main_a0 by
2*tile_div", freq = 250/(2N) MHz. Two implementations, two clock ratios,
the same tile-cycle count to within 3 cycles: the walk is tile-clock-bound
and the number is real.)

## What this run proved on the landed tree

* **Chain A** mask {FPROJ, OPROJ, RES1}: fuel-fed OPROJ, requant epilogue
  on-tile (RQ[14] slot), o8 → serializer → deq (JC_OPROJ) → residual —
  **r1 896/896 BIT-EXACT** vs golden's own composition
  (`gemm_i8_ksplit` → `calib_requant` → `requant_i32_to_i8` →
  `f16_grade(s_wo·2^shift/scale)` → `f16(X+o8·comp)`) on the same
  operands, computed AFTER the run. Walk window: **ZERO host operations**
  (not even a read), predicate-checked on the emitted artefact.
* **Chain B** mask {FPROJ, QKV, OPROJ, RES1}: ONE descriptor, 4
  walker-issued fetch records, 256 MXE jobs — 144/144 QKV raw-INT32
  bit-exact AND r1 896/896 bit-exact. This driver stages TWO REAL operand
  families (the 'h' row at bank-1 rows 0..13 for QKV, the real attention
  row at rows 14..27 per the landed `fp_oproj_base` rule) — so both
  projection classes contract their real 0.5B operands in one kick.
* **Discriminators, all RED by predicted deltas:** walk-off
  (W2_MASK[14] cleared → the measured legacy park/starvation class); ONE
  poisoned resident-Wo byte, searched on golden first (the argmax-|x|
  flip is ABSORBED by the requant: the o8 step is ~amax/126 ≈ 130k counts
  at the 896-deep contraction — independently rediscovered here, the same
  finding the landed increment 5 recorded); an RQ-pair delta (+377,
  searched) in the descriptor slot moves r1 by exactly golden's requant
  prediction on 20 elements — **the epilogue calibration slots are live
  on-tile**, a leg the landed disc set did not cover.

## One receipts correction to the landed result doc

`WALKED_EPILOGUE_E6.md` converted its walk windows at shell/TILE_DIV; the
divider is shell/(2·TILE_DIV) (`cl_apex.sv:82`, and this file's
cross-ratio table is the measured proof). Its tile-cycle column and the
figures derived from it were 2× off; corrected in the same commit as this
file (the 2347ab6 receipts-pass precedent). Corrected headline numbers:
~2,187 tile cycles per fuel-fed k=896 job, ≈3.3 weight bytes/tile-cycle
sustained; the labelled full-layer prediction becomes ≈4.4M tile cycles
≈ 70 ms/layer at a 62.5 MHz tile ⇒ **≈1.7 s/token over 24 layers at the
measured ingest** (≈0.35 s/token at the 8 B/cyc peak-ingest floor; at the
15.625 MHz clock the flown A2 images verified, 4× those walls). The
conclusion direction is unchanged — and 2× stronger.
