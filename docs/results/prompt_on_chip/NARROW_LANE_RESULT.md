# N-LANE — RoPE AND SwiGLU OF A REAL Qwen2.5-7B STEP, ON OUR SILICON

**Date:** 2026-07-31 · **Branch:** `comp/prompt-b-c` · **Image:**
`afi-0408bd9c4fbf4a45c` / **`agfi-0cc7aa798fe3abce2`**
(apex-narrow-layerwindow-20260731, built from `9b993bf`, D=128 GQA=1 DM=128
DDR=0, clkgen recipe A2, stock kit v2.3.3, shell 0x10212415) ·
**Instance:** `i-0634af8f958021dc9` f2.6xlarge us-west-2b (terminated +
verified) · **Records:** `flight_result_hw.json`, `flight_result_sim.json`,
`flight_plan.json` (build/n_flight/), driver
`scripts/fpga/f2/narrow_flight.py`.

## The claim (exactly)

**Two more op families of a REAL Qwen2.5-7B decode step now run on our
silicon, at full model fidelity:**

- **RoPE — ALL 28 query heads**, head_dim=128, of the committed S8 step.
  128 rotated fp16 codes + the row scale per head, **bit-exact vs the
  arbiter's `q_rope`**, produce-mode.
- **SwiGLU — ALL 296 chunks = the FULL d_ffn = 18,944 columns** of the same
  step, from the real gate/up INT32 accumulators, recovered through the C-1
  view (fp16 row scale x INT8 codes), **bit-exact vs golden's `swiglu`**.
  21,460 captures in this stage alone.

**These are NOT scaled-down.** RoPE is width-local (a head IS 128 wide) and
SwiGLU is chunk-local (the unit consumes 64 columns per job at any model
width), so the narrow image runs them at true 7B geometry. The claim is the
real model's ops, not a toy.

**Also demonstrated, honestly scoped as UNIT demos (not the model's op):**

- **residual r1 and r2** over a 128-element window of the real row
  (128/128 fp16 codes bit-exact, both). The model's residual spans 3,584 —
  the full-row form needs the wide image (aws-fpga#799).
- **RMSNorm-2 at dm=128** — graded against the arbiter's OWN `rmsnorm_fx_wide`
  applied at 128 width, NOT against the full-row h2. A 128-wide norm has a
  different denominator, so grading it against the 3,584-row result would be
  dishonest; the codes alone would have passed vacuously because row-quant
  codes are scale-invariant — the row SCALE is what carries the denominator.
- **R3 geometry-refusal guard on silicon**: a residual job whose window
  footprint exceeds the row RAM is REFUSED loudly (sticky + err_code 3) and
  the W1C clears it — the RTL fix written this morning, behaving on the FPGA
  exactly as in simulation.

## Verdict

```
N-FLIGHT (hw): 34/34 stages green, wall 49.5 s -> PASS
CANONICAL 18-JOB REGRESSION on the NEW image: 18/18 complete+attributed -> PASS
```

| stage | captures | grade |
|---|---|---|
| rope h00..h27 (28 stages) | 145 each | bit-exact vs `q_rope` |
| swiglu (296 chunks, full d_ffn) | 21,460 | bit-exact vs `swiglu` |
| resid_r1 / resid_r2 | 128 / 128 | bit-exact vs r1 / r2 (128-window) |
| norm2 (dm=128 unit demo) | 145 | bit-exact vs golden's norm at 128 |
| resid_probe / swiglu_probe | 0 (checked guards) | refusal + no-error asserted |

~25,900 real captures, produce-mode, **zero baked output expectations**
(audit enforced at generation: 0 violations across all 34 programs).
The identical program set graded **34/34 in simulation** on the parity twin
of this exact commit BEFORE the card was launched.

## Op-type coverage after this session

```
  in SIMULATION  ████████████████████  6/6
  on SILICON     █████████████░░░░░░░  4/6 at FULL model fidelity
                 attention (28/28 heads) · projections (1024/1024 blocks)
                 · RoPE (28/28 heads) · SwiGLU (18,944/18,944 cols)
                 + residual & RMSNorm-2 demonstrated as UNITS at 128 width
```

The remaining 2/6 are the **full-row** forms of residual and RMSNorm-2,
which need the wide image and are gated on aws-fpga#799.
**[SUPERSEDED 2026-08-01: both were dissolved WITHOUT the wide image the
next day — elementwise-sliced residual + R4 chunked RMSNorm-2, 6/6 on the
FPGA. See `SIX_OF_SIX_RESULT.md`; #799 became an optimization, not a gate.]**

## A second, independent data point for #799

This image carries four blocks the flying image never had (rope_row,
asu_swiglu, apex_layer_deq, apex_residual) and **passed `pr_verify` clean**
(0x HDPRVerify-41, 1x HDPRVerify-42 locked-static) and passed AWS ingestion
first try. Added CL area at DM=128 does NOT trip the defect: it is bound to
the WIDE configuration, exactly as the escalation states.

## Incidents (all caught, none affecting integrity)

1. **Two hw attempts returned zero captures.** Cause: `narrow_flight.py`
   never called `remote_hw_exec.attach()`, so the bridge tried to drive BAR0
   from the laptop. The refuse-loudly capture gate caught it both times —
   nothing was ever credited. Fixed (9727066); the sweep driver had it right
   at `proj_sweep_batched.py:126`.
2. **80 MB single-batch upload never landed** (the full-depth swiglu stage
   alone is 74 MB / ~2.8M regops). Fixed with size-bounded batches
   (`--max-batch-mb`, default 8; an oversized stage becomes its own batch).
3. **Pre-flight sim gate caught two real defects before any spend**:
   B-STAGE-ROWS (act-stage rows ALIASED past the 16-row bank and still
   graded GREEN below the wrap — a silent wrong answer) and a
   dimension-dishonest norm2 oracle. Both fixed and committed (eb25dad).

## Cost / teardown

f2.6xlarge ~40 min + m6a.4xlarge build 54 min ≈ **$2**. Instance
`terminate-instances` → `wait instance-terminated` → describe reads
**terminated**; account F2 sweep afterwards shows only the unrelated
`apex-f2-fpga` box, untouched all session.
