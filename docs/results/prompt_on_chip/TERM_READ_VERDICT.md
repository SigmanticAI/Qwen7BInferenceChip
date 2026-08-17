# THE TERM-READ VERDICT — the walked-attention stale is the sc_val READ, definitively

**Date:** 2026-08-08 · **Image:** `agfi-080a4f90f14a51bca` (apex-dbg5-
20260808, @d1794be — the sc_val DEFENSIVE REWRITE + WALK_DBG v5 term split;
WNS +0.445) · **Instance:** `i-024453cac82a41b15` (terminated+verified) ·
**Probe:** `walk_e6_dbg2`.

## The word (decisive)

```
THE TERM-READ: 0x02000542
  snp=2      two snoop commits happened (sc_mem/sc_val writes)
  ca=1       committed at addresses 0 AND 1 (OR = 1)
  acc_idx=0  the walked request asked for sc_val[0]
  acc_eng=0  on engine 0
  seng=0b01  engine 0 is the one that pulsed stale (self-consistent)
  TERM_SC=1  !sc_val[req_idx] fired -> the RECORD-cache valid bit read 0
  TERM_SQ=0  s_q_val was PRESENT -> the q-scale tap is NOT the fault
  stale=1 frame=0
```

## What this settles

The four-flight instrument campaign (DBG v2-v5) converges to a single,
unambiguous statement: **on silicon, `sc_val[0]` reads 0 in ST_IDLE even
though the snoop committed a write to it (snp=2, addresses 0 and 1). The
companion term `s_q_val` is fine.** The fault is exclusively the
record-scale-cache valid-bit READ.

Two hypotheses are now KILLED, not just unlikely:
- **s_q / the q-scale tap** — TERM_SQ=0 on silicon. Retired.
- **the defensive rewrite fixes it** — this image CARRIES the rewrite
  (@50088f4: sc_val split into a per-bit-decoded register vector in its
  own always_ff, the RAM write and beat-tracking in separate blocks) and
  the read STILL returns 0. The structural rewrite did not change silicon
  behavior. The scfix flight (`agfi-0ecb3a62da3fda9bd`) already showed
  walk_e6 still red; the term-read now says exactly why.

## Honest status: this is a hard synthesis-vs-simulation hardware bug

Every gate is green in Verilator on the identical RTL. Timing is MET
(+0.445), CDC audit clean, the write side proven, the request well-formed.
A resettable per-bit register vector, written by a decoded index and read
by another index, behaves differently in the synthesized VU47P netlist
than in simulation — and a defensive rewrite of that exact structure did
not move it. This is no longer a "spin another instrument image" problem;
it needs a different class of work.

## The next class of work (NOT quick, stated honestly)

1. **Netlist forensics** — open the synthesized/routed checkpoint
   (`write_checkpoint` is already in the flow), locate how `sc_val` was
   mapped (flops vs distributed-RAM re-inference, any X-optimization or
   don't-care on the reset/decode), and diff the read cone against sim
   semantics. This is the direct path but it is Vivado-checkpoint
   spelunking, hours of uncertain effort.
2. **Design-around** — replace the indexed-valid-array read with a form
   synthesis cannot re-infer as RAM: e.g. a small explicit register file
   with a one-hot read mux, or carry the "written" state as a companion
   to the record rather than a separately-indexed vector. Lower-risk than
   forensics but a real redesign of the block's read path, re-verified
   from the unit suite up.

## What is NOT blocked by this

The walked back-half (FFN rungs 1+2, DOWN gate rund7) is silicon-
independent progress and all green in sim; the token-loop harness measures
tok/s the moment a walked layer composes. The A0 clock campaign (3 cones
dead) is orthogonal. This defect gates ONLY walked ATTENTION on silicon
(E-7/E-8) — not the rest of the tokens/second ladder.

## Cost

f2.6xlarge ~30 min ≈ $1. Terminated + verified. 17 images, 17 first-try
ingestions.
