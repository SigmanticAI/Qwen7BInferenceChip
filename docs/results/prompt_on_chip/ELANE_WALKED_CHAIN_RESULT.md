# E-LANE ON THE FPGA — THE SEQUENCER DRIVES AN IN-TILE CHAIN

**Date:** 2026-08-02 · **Branch:** `comp/prompt-b-c` @ `47a5a3a` ·
**Image:** `afi-05505bed29348569e` / **`agfi-006b1314fcbbb3505`**
(apex-elane-walker-20260802; D=128 GQA=1 DM=128 DDR=0, clkgen A2, kit
v2.3.3, shell 0x10212415) · **Instance:** `i-0c9dd94acc784a18b` f2.6xlarge
us-west-2b (terminated + verified) · **Driver:**
`scripts/fpga/f2/elane_walk_norm.py` · **Log:** `build/elane_hw.log`.

## The claim (exactly)

**On an FPGA, APEX's own layer sequencer armed and completed a compute chain
whose intermediate value NEVER LEFT THE TILE: the residual's row RAM streamed
out through the C-1 quantizer into RMSNorm, bit-exact against golden — and
the host did nothing during the walk.**

This is the first time the sequencer block has driven real hardware. Every
prior FPGA result in this project (including 6/6 op types and the C2 prompt
run) was HOST-driven: software pushed each job over the bus.

```
[elane] remote hw shim attached (clock gate ON)
  [walk_nfeed] rc=0 ok=True caps=274 4.3s grade=True
  disc walk_off     caught=True
  disc x_ulp        caught=True
  disc gamma        caught=True
  disc cap_code     caught=True
  disc cap_scale    caught=True
ELANE-WALK SMOKE: PASS
```

**`walk_off` is the load-bearing discriminator.** It runs the IDENTICAL
program with only the walker's step-enable mask bit cleared. If anything
host-side were performing the chain, the run would still pass. It does not —
the route-arm poll is never satisfied and the run fails. That is the
evidence the SEQUENCER did it.

## What changed in RTL to make this possible

- **E-1** (`apex_residual.sv`): an egress job port + fp16 stream so the row
  RAM has an INTERNAL reader (before: the host register path only).
- **E-2** (`apex_top.sv` glue + new `apex_lane8_unpack.sv`): `l_nsrc` muxes
  asu_rmsnorm's x port onto the feeder's codes (the norm module itself
  UNTOUCHED); LAYER_JOB unit 3 is the start verb; a push with the route
  unarmed refuses loudly (err_code 9).
- **E-3a** (`seq_walker_pkg.sv`, `seq_layer_walker2.sv`, `apex_top.sv:1217`):
  the walker's `fsrc_ext` level was 2 bits and apex_top ZERO-EXTENDED it
  (`{1'b0, wk_lw_fsrc_ext}`), so a walked program physically could not name
  feeder source code 4. Widened end-to-end; new walker step `PC_NFEED`
  (gated by the previously-reserved mask bit) issues arm -> feeder job ->
  unit-3 job -> wait on `nf_busy`.

RED on the pre-change twin: `poll stall ... last 00080000 want 00000080` —
`l_nsrc` set, bit 7 unreachable. GREEN after: 7,697 ops, 26 checks, 0 fails.
Walker suites all green with new coverage (l128 nfeed 657 checks); the three
frozen descriptor images are byte-identical.

## What the host still does (precisely)

- **Before the walk:** load X and gamma, produce r1 via the host CSR path
  (RES1 is outside this fence), arm `l_nsrc` (a build-shaped selector
  apex_top deliberately holds across a walk), set the drain routes, load the
  64-word descriptor, write WALK_CTRL.
- **During the walk: NOTHING.** `walk_en` holds every host job/level/route
  port not-ready; all three arming verbs come from the sequencer.
- **After:** drain the norm's output (the y -> projections seam belongs to
  the next lane) and read back r1.

## Scope fence — read with the claim

This is the **128-wide toy configuration**, where `head_dim == D_model ==
128` is a consistent single-head layer and the C-1 quantizer is natively
full-row (one row, one scale — the same form golden uses). It is an
**ARCHITECTURE** claim. It says NOTHING about Qwen2.5-7B, and must never be
blurred with the N-lane/C2 results, which are about the real model but are
host-staged between ops.

Still open in this lane: **E-3b** (the walker staging its own rotated-q rows
— today they are host-staged before WALK_GO) and **E-4** (a synthetic full
layer as the subject). The chain proven here is residual -> norm, not the
whole layer.
**[FOLLOW-ON, closed after this doc: E-3b + E-4 + E-4b landed 2026-08-04
(sim, `STEP_MATRIX.md`); E-5/E-6 fuel-fed walks landed 2026-08-05 and flew
on `agfi-0183a4b88c8d21163` / `agfi-0bc20880b50f5faba`
(`SELF_RUNNING_CARD_RESULT.md`, `E6_ON_SILICON.md`).]**

## Cost

f2.6xlarge ~15 min + m6a.4xlarge build ~55 min ≈ **$2**. Terminated and
verified; account F2 sweep shows only the unrelated `apex-f2-fpga` box.
