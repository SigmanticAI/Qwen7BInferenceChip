# E-6 ON THE FPGA — THE TILE CONSUMES ITS OWN PROJECTION OUTPUT

**Date:** 2026-08-05 · **Branch:** `comp/prompt-b-c` @ `a85ea8e` ·
**Image:** `afi-056c6bc6ab8709a0b` / **`agfi-0bc20880b50f5faba`**
(apex-convergence2-20260805: E-6 walked o8 epilogue (S2_PSJ/RT_FPRQ) +
D-aware `k_job` (Wd legal at D=64) + E-5 fuel projections + E-3b/E-4b chains
+ DDR=1 + real CDC constraints; D=64 DMODEL=64 GQA=2 DM=896 QSTAGE=14,
clkgen A2, **Slack MET +0.297 ns**, PRV-GREEN, ingested first try — no APEX
submission has failed ingestion since 2026-07-28; the account AFI listing
(audited 2026-08-05) shows 8 consecutive first-try APEX images since then) ·
**Instance:** f2.6xlarge us-west-2
(terminated) · **Driver:** `scripts/fpga/f2/walk_fuel_layer.py` ·
**Captures:** `walk_oproj.hw.cap.jsonl`, `walk_qkv_oproj.hw.cap.jsonl`.

## The claim (exactly)

**On an FPGA, under ONE fmt=1 descriptor, APEX's sequencer fetched real
Qwen2.5-0.5B weights from the card's own DRAM, computed the Q/K/V
projections AND the output projection, applied the requant epilogue ON THE
TILE, and fed the o8 product through the dequantizer into the residual unit
— and the resulting 896-element r1 activation row is BIT-EXACT against
golden. The tile consumed its own output; the host wrote nothing inside the
walk window.**

```
chain A  walk_oproj      (OPROJ+RES1 walked)          36 checks 0 fails,  896 caps
         -> r1 BIT-EXACT 896/896 vs golden_epilogue
chain B  walk_qkv_oproj  (QKV+OPROJ+RES1, ONE desc)   50 checks 0 fails, 2192 caps
         -> r1 BIT-EXACT 896/896 vs golden_epilogue
DISC 1   walk_oproj_off  (mask bit cleared)           23 checks 1 FAIL,    0 caps
DISC 2   walk_oproj_rq   (perturbed RQ calibration)   ran, r1 differs 801/896
```

Both discriminators fired as designed: removing the walker's step makes the
identical program stall with zero captures, and perturbing the descriptor's
**on-tile requant calibration slots** moves 801 of 896 output words — proving
the epilogue's calibration is genuinely applied by the tile, not by the host.

## Why this is the era-2 hinge

E-5 proved the walker could FETCH and COMPUTE (raw INT32 accumulators).
E-6 closes the loop: the walked projection now produces the **activation the
next operation consumes**, in the tile's own arithmetic, on-chip. That is the
difference between "the tile computed a GEMM for us" and "the tile is running
the layer".

## Honest scope

- **8 of 13** step families walk in a proven mode; 3 under one descriptor at
  real 0.5B geometry ({QKV, OPROJ, RES1}), 5+NSRC in the toy one-kick chain.
- Loudly fenced with measured reasons (S2_CHECK `fp_mask_ok`): NORM1/2 gamma
  (no xw->xg route), FFN/SWIGLU (phase-alternation vs template order), DOWN
  (mask-tied to FFN), QSTAGE-under-FPROJ, QKV+attention together.
- Host still does: DDR weight-image load, pre-walk staging + calibration
  slots + descriptor + fuel arm, GO, one poll, post-walk reads.
- **Prediction, labelled:** ~2.19k tile cycles/job, ~4.4M tile cycles for a
  fully-walked layer ⇒ **~70 ms/layer at 62.5 MHz ⇒ ~1.7 s/token** (4x that
  at today's flown 15.625 MHz). NOT measured; the measured item here is
  bit-exactness, not throughput.

## Verification provenance worth noting

E-6 was implemented TWICE by accident (a duplicate task in a shared
worktree). The two independent implementations converged on the same design
and agreed **to the cycle**: chain B = 559,697 tile cycles from both drivers
at two different clock ratios. The replication also caught a 2x error in the
first implementation's tile-clock conversion (shell/(2*TILE_DIV), not
shell/TILE_DIV — `cl_apex.sv:82`), which is why the prediction above is the
corrected one.

## Cost

~35 min f2.6xlarge + ~70 min m6a.4xlarge build ≈ **$1.80**.
