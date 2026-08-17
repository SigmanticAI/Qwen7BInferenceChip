# FFN/SWIGLU WALK — the interleave redesign (fence 2 closure plan)

**Status: DESIGN ONLY (2026-08-07). Nothing here is implemented.** This is
the phase-2 long pole on the tokens/second ladder (MASTER_TABLE §perf):
with attention, OPROJ, RES1, NORM2, NFEED silicon-proven walkable and the
walked-attention defect instrumented, FFN gate/up + SwiGLU + DOWN are the
remaining un-walked MAC families.

## The fence, restated from source (STEP_MATRIX.md FENCE 2)

`asu_swiglu` consumes ONE gate frame THEN one up frame per job
(`asu_swiglu.sv:106` ST_GATE/ST_UP, ≤64 cols each), and `apex_top`'s
`swg_up_q` phase tracker flips on every accepted `last`
(`apex_top.sv:1547-1555`). The walker's fmt=1 template is linear:
PC_USWI(21) pushes ALL chunked USWI jobs, then PC_WGJ(23) runs ALL gate
n-jobs, then PC_WUJ(25) ALL up n-jobs. Two structural faults:

1. The second USWI push deadlocks — one pending LAYER job slot; its
   frames need pcs the walker has not reached.
2. A walked all-gates stream is eaten as up data once `swg_up_q` flips.

## The fix shape (per the fence's own prescription)

Replace the linear PC_USWI→PC_WGJ→PC_WUJ segment with an inner chunk loop,
`NCHUNK = d_ffn/64` iterations of:

```
{ PC_USWI   1 swiglu job (this 64-col chunk)
  PC_WGF/J  1 sj frame + 8 gate n-jobs   (fuel: Wg[:, c*64:(c+1)*64])
  PC_WUF/J  1 sj frame + 8 up n-jobs     (fuel: Wu[:, c*64:(c+1)*64]) }
```

with `rdst=1` routing raw INT32 to the serializer, exactly as the fence
prescribes. Two coupled deliverables:

### A. Walker RTL (`seq_layer_walker2.sv`)

- New loop counter `ffn_chunk_q` (16-bit, compare against
  `d2.d_ffn >> 6`); the FFN pc-range becomes re-entrant: after PC_WUJ's
  last accept, if `ffn_chunk_q+1 < NCHUNK`, jump back to PC_USWI with
  `ffn_chunk_q+1`, else fall through to PC_LVLD (DOWN) as today.
- The per-chunk fuel record: today's PC_WGF/PC_WUF fetch the WHOLE Wg/Wu
  region as one record each. The loop needs per-chunk records — see (B);
  the walker computes the chunk's record base as
  `base + ffn_chunk_q * CHUNK_BEATS` (pure address arithmetic, same
  `walk2_k_job` row-cap machinery as QKV/OPROJ).
- The existing k-loop precedent: DOWN's D-aware `k_job` split and the
  WALK2_STAGE_R_CAP row banking show the walker already carries
  loop-with-address-arithmetic idioms; this adds one more counter, not a
  new mechanism class.
- Reserved-bit discipline: no new descriptor fields needed — `d_ffn` is
  already in the descriptor and NCHUNK derives from it. Legacy images
  (FFN mask bits 0) walk byte-identically; the S2_CHECK fence
  `fp_bad_steps(FFN)` flips from refuse to accept only when the new
  sequencing is in.

### B. DDR image re-lay (`make_weight_image.py`)

One fuel stream must deliver beats in the NEW consumption order:
interleave Wg/Wu in 64-column blocks —
`[Wg[:,0:64] | Wu[:,0:64] | Wg[:,64:128] | Wu[:,64:128] | …]` as ONE
`L{n}_Wgu` tensor. Loader/regions/sha machinery is layout-agnostic
(tensors are opaque byte ranges), so the change is confined to the image
writer + the golden staging that mirrors it. Keep the old separate
Wg/Wu tensors emitted alongside during bring-up so host-mode FFN (the
proven path) still runs from the same image — space is not a concern
(0.5B: ~30 MB total today).

### C. Verification ladder (the project's standard shape)

1. Unit: walker-pkg python twin + RTL `walk_desc2_check` unchanged
   (no new fields); new twin build; frozen descriptor images byte-diff.
2. Sim RED first: the CURRENT template on the interleaved image must
   refuse/park exactly as today (fence receipts kept).
3. Sim GREEN: one-chunk toy (d_ffn=64: NCHUNK=1 — degenerate, must equal
   today's single-pass behavior bit-for-bit), then d_ffn=128 (NCHUNK=2,
   the first true loop), then the 0.5B d_ffn=4864 (NCHUNK=76).
4. Discriminators: walk-off (FFN mask bit cleared → the E-6 refusal),
   a gate/up SWAP poison (interleave order flipped in the image → SwiGLU
   consumes gate-as-up; product must move by golden-predicted values),
   and the walk-window host-silence check as in E-6/E-7.
5. Silicon: only after the walked-attention defect is closed (score+pv
   sit between attention and FFN in every full-layer flight) — or fly an
   FFN-only chain (mask {NFEED, NSRC, FFN…} without score) the way E-7ng
   flew without attention.

## Interaction with DOWN (fence 3) and RES2

DOWN's mask un-tie (`W2_EN_DOWN`, E-7b) is already landed; its blocker at
0.5B is the 76-row stage bank vs WALK2_STAGE_R_CAP=31 — a separate,
smaller redesign (k-chunk act re-staging carry). Sequence AFTER the FFN
interleave: the interleave's chunk loop is the natural carrier for DOWN's
re-staging (the swiglu product of chunk c is DOWN's k-slice c — the loop
alignment is free).

## Size estimate (honest)

RTL: ~1 counter + pc-graph edit + record-address arithmetic (~100-150
lines with comments in walker2) + image writer (~60 lines) + golden
staging mirror + the 5-rung ladder. The ladder, not the RTL, is the bulk:
expect 1-2 focused sessions to sim-green with discriminators, then a
build cycle. No silicon dependency until the flight rung.
