# ALL SIX OP TYPES OF A REAL Qwen2.5-7B DECODER LAYER, ON AN FPGA

**Date:** 2026-08-01 · **Branch:** `comp/prompt-b-c` @ `a5dcfbc` (RTL
`d444b6e`) · **Image:** `afi-093093be6d909aedc` / **`agfi-09e947873048a6877`**
(apex-narrow-r4-20260731; D=128 GQA=1 **DM=128** DDR=0, clkgen A2, kit
v2.3.3, shell 0x10212415) · **Instance:** `i-076df27e28eccddd7` f2.6xlarge
us-west-2b (terminated + verified) · **Driver:**
`scripts/fpga/f2/narrow_flight.py` · **Record:**
`flight_result_hw_r4.json`.

## The claim (exactly)

**Every one of the six compute op types of a real Qwen2.5-7B decoder layer
(layer 0, step 10 of the committed S8 run) has now been computed by APEX's
own blocks on an FPGA, bit-exact against the golden fixed-point arbiter, in
produce-mode (the tile returns values the host did not pre-compute).**

| op type | coverage on the FPGA | where |
|---|---|---|
| attention (score+PV) | **28/28 heads** | C1, fa4707c |
| projections Wq/Wk/Wv/Wo | **1024/1024 blocks**, full K=3584 | S3 sweep |
| RoPE | **28/28 heads**, head_dim=128 | N-lane, 2026-07-31 |
| SwiGLU | **18,944/18,944 columns** (full d_ffn) | N-lane, 2026-07-31 |
| **residual r1 + r2** | **3,584/3,584 elements each**, reassembled == golden | **this session** |
| **RMSNorm-2** | **3,584/3,584 codes + 28/28 scales**, full row | **this session** |

**None of it needed the wide image.** The two "wide-only" ops were dissolved:

- **Residual is ELEMENTWISE** (`row[i] <- f16(row[i] + b[i])`, no
  cross-element state), so the full row is 28 aligned 128-element slices,
  each bit-identical to what one wide pass would produce. **No RTL change.**
- **RMSNorm-2** uses golden's OWN definition of the wide norm
  (`rmsnorm_fx_wide`): per-chunk sums -> accumulate -> mu-scalar mean ->
  the unchanged per-element datapath with ONE broadcast r. **R4** made the
  tile export each chunk's sum2 and accept an external SUM2+k, so mu, the
  rsqrt and **all 3,584 per-element operations run on the tile**.

## Verdict (pasted)

```
N-FLIGHT (hw): 93/93 stages green, full-row residual reassembly 2/2 rows == golden,
               wall 66.4 s -> PASS
  [residfull r1] slices 28/28 fully captured, 3584/3584 elements covered,
                 reassembled row == golden r1: True (n_diff=0)
  [residfull r2] slices 28/28 fully captured, 3584/3584 elements covered,
                 reassembled row == golden r2: True (n_diff=0)
  nf_norm2c  caps=4088   nf_norm2x  caps=4060   nf_swiglu  caps=21460
CANONICAL 18-JOB REGRESSION on the R4 image: 18/18, batch ok=True
```

**41,237 captures**, produce-mode, **zero baked output expectations**
(audit enforced at generation: 0 violations across all 93 programs). The
identical program set graded **93/93 in simulation** on the parity twin of
this exact commit BEFORE the card was launched. The full-row residual is
graded twice — per slice, and by REASSEMBLY from the tile's **captured**
words (never the expectation), with per-element coverage tracking; the
reassembly grader has a host-side selftest proving it goes red four ways
(flipped bit, missing slice, short slice, duplicated slice).

## ADDENDUM 2026-08-01 — the 27 host adds are GONE (verified on the FPGA)

`norm2m` closes the last arithmetic gap and was flown on
`agfi-09e947873048a6877`: **94/94 stages green, wall 1.2 min**.

The MXE now ACCUMULATES the K-split itself (29 x `OP_GEMM_OS` with
`accumulate=1` on all but the first — golden compute.py's own on-tile arm of
C-KSPLIT), so the last readback beat's lane 0 IS the whole row's sum2. The
host's ENTIRE role is moving one u32 from a capture register into CSR
RMS_SUM2 — and that is re-derived FROM THE EMITTED PROGRAM by
`narrow_flight.assert_move_only` (exactly one write to RMS_SUM2,
bit-identical to the captured accumulator, zero read-backs), with four red
tests (transformed / missing / duplicated / read-back).

```
[nf_norm2m]      armed=120498 (delta +0)    grade=PASS caps=4060 -> as expected
[nf_norm2m_disc] armed=125441 (delta +4943) grade=FAIL caps=4060 -> as expected
[nf_norm2m_ctl]  armed=120499 (delta +1)    grade=PASS caps=4060 -> as expected
N-FLIGHT (hw): 94/94 stages green, residual reassembly 2/2 == golden -> PASS
NORM2M (0 host adds): sum2=120498 computed and ACCUMULATED on the tile,
  moved verbatim into RMS_SUM2; full-row norm PASS, discriminator CAUGHT
```

**On the discriminator (why +1 is a CONTROL, not a failure):** sum2 reaches
the graded row only through `r = 2^13 // isqrt(((sum2*mu)>>(s+16))+1)`, so
golden itself returns a bit-identical row for every sum2 inside one r bucket
— 4,942 wide here. A +1 perturbation therefore CANNOT go red unless the tile
were MORE sensitive than the arbiter. The red test uses the bisected first
delta that leaves the bucket (**4,943**, r 1638 -> 1365, predicted to move
872/3584 codes and 28/28 scales — matched exactly), and the +1 control must
stay GREEN. Both are gated: the tile must be exactly as sensitive as golden,
no more and no less.

## What is HONESTLY still on the host

1. ~~27 INT32 adds in the chunked RMSNorm~~ — **CLOSED 2026-08-01, see the
   addendum above.** Zero host arithmetic in the full-row RMSNorm-2 path;
   the host moves one u32 and nothing else. (`norm2c`, the 27-add variant,
   is retained and still green as the self-contained form.)
2. **Staging between ops.** Each op type is proven INDIVIDUALLY; the host
   still carries activations from one op to the next. A layer running END
   TO END on-chip is a different claim and a different lane
   (`docs/design/E2E_TOY_LANE.md`), blocked by two seams: the residual row
   has no internal egress, and asu_rmsnorm's input is a top-level port.
   **[The two seams were closed in RTL 2026-07-31 (E-1/E-2, 7c0d4a5) and the
   E-lane has since walked in-tile chains on silicon — toy/0.5B geometry,
   never this 7B image; see `ELANE_WALKED_CHAIN_RESULT.md`, `STEP_MATRIX.md`,
   `E6_ON_SILICON.md`. The 7B claim in THIS file remains host-staged.]**
3. Softmax + requant scales are host-applied in the attention path
   (the standing fence), and weights/KV live in host memory.

## What this says about aws-fpga#799

Two images in two days, each carrying substantial NEW logic at DM=128
(#1: rope/swiglu/deq/residual; #2: + R4 norm ports/CSRs), both **PRV-GREEN
with 0x HDPRVerify-41** and both ingested first try. Added CL area at narrow
width does not trip the defect. It remains bound to the WIDE configuration,
exactly as filed. **#799 is now an OPTIMIZATION** (one-pass norm, unsliced
residual), not a gate on any op-type claim.

## Cost

f2.6xlarge ~25 min + m6a.4xlarge build ~55 min ≈ **$2**. Terminated and
verified; account F2 sweep shows only the unrelated `apex-f2-fpga` box,
untouched.
