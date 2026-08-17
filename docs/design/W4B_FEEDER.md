# W4-B FEEDER — the adopted G=32/(B) weight path in RTL (lane contract)

**Branch:** `comp/w4b-feeder` (worktree `../apex-w4b`, cut from
`comp/b3-weight-path` @ 0109ed7). **Spec:** B3_WEIGHT_PATH.md §8 (stage-6
fallout notes) + the adoption verdict in §5 row 6 / RESULTS.md — the
recipe this lane implements is **W4 G=32 through the (B) chain with
DIRECT-from-source host prep** (−1.0 pt vs shipped weights, BEATS the
current INT8 feed; measured at n=10,042). **Decision ID: D-031** (renumbered
2026-07-23: this lane briefly claimed D-030, but integration owner-SIGNED
a different D-030 — the grade-definition/choreography canon — first;
D-029 is IB-WALK's fmt=1 full-layer descriptor and D-032 is B2/A3's
(tree-wide canon). D-031 verified free across every live branch).

**Golden arbiter (exists, never edited):**
`weight_codec.wfeed_w4_to_i8(blob, realization="B")` — 54M golden stripe
checks on real model bytes behind it. This lane ADDS one thin oracle
(stage 0): `wfeed_w4b_to_i8(blob, s8_bits)` — the same chain with the
requant scale as an INPUT, because the RTL consumes a **host-computed
stripe-global sideband scale** (§8.1: multi-job K-split stripes share one
epilogue factor; only the host sees the whole stripe — the D-021 delta,
disclosed in RESULTS.md). Reduction identity (gated): with s8 = the
tile-amax scale, `wfeed_w4b_to_i8 == wfeed_w4_to_i8("B")` bit-exactly.

## Interface (additive sibling module — the landed (A) feeder is untouched)

`mxe_wfeed_w4b.sv`, sibling of `mxe_wfeed_w4.sv` (whose unpack front half
is reused verbatim per §8.1; the (A) module stays verified/inert,
quality-quarantined per §8.6):

- packed-W4 stream in: identical S1–S6 framing to (A) (`pack_stream`,
  element e at bits [4e +: 4], job_beats = emitted INT8 beats = KB·N,
  `emitted == 2·consumed − (job_beats & 1)`).
- **`gs_*` group-scale sideband IN** (fp16, skid-buffered — the
  `seam_feeder_quant` scl channel pattern mirrored in reverse direction):
  one fp16 scale per G-group, order = ascending `group_ids` gid
  (= c·ceil(K/G) + floor(k/G): column-major groups). Count/job =
  N·ceil(K/G) (≤ 512 at K=2048, N=8, G=32).
- **`s8_job` requant scale as a JOB PARAMETER** (fp16 bits, sampled with
  the job pulse like job_beats — one value per job, not a stream).
- lane8 INT8 beats out (unchanged downstream contract — MXE sees INT8,
  ARCHITECTURE C-4).
- G is an elaboration parameter; **SHIP VALUE FROZEN: G=32**
  (2026-07-22, full matrix complete — RESULTS.md group-size call):
  direct@G32 = −1.0 pt at 4.5 b/w; direct@G16 = −0.63 pt but the gain is
  +0.41 pt at z=+2.05 (the weakest resolved effect in the table) for
  +11% weight traffic (5.0 b/w, ≈−4–5% decode vs the ≥10 tok/s floors).
  G=16 remains BUILDABLE and golden-gated (stage-0 tests run both) — a
  validated quality option for bandwidth-rich configurations, never the
  default.

## Arithmetic (normative, from the golden)

Per element: `real = dequant_f32(code4, s_group)` (exact, cq_codec) →
`q8 = clamp(RNE(real / f16(s8_job)), −128, 127)` (the D-021
quant_rows_i8 requant rule with the scale given). The composite the
epilogue receives is host-side `f16_grade(s8_job × s_w)` (§8.3 — host
obligation, carried in the contract, not this module).

## Staged plan

| stage | what | state |
|---|---|---|
| 0 | this contract + golden `wfeed_w4b_to_i8` + `golden/tests/test_w4b_feeder.py` (reduction identity, stripe-shared-scale semantics, given-scale corners) + Makefile target | THIS COMMIT |
| 1 | RTL `mxe_wfeed_w4b.sv`: (A) unpack front + per-group fp16 dequant + given-scale INT8 requant (cq unit family), gs skid, job legality incl. scale-count mismatch reject | |
| 2 | TB `verif/mxe/w4b/` (sb pattern): scoreboard vs the stage-0 oracle, gs late/eager/storm adversaries, mid-op resets, coverage | |
| 3 | **REQUIRED exhaustive operand sweep** (§8.4, fparith pattern): 16 codes × every positive finite fp16 group scale × representative s8 set — full dequant×requant domain; + mutation gate (≥4: RNE→trunc, clamp bound, gid-order, scale-sample-timing) | |
| 4 | integration notes — WRITTEN (§ Integration notes below) | THIS COMMIT |

## Integration notes (stage 4 — for the combine session)

1. **Placement:** `mxe_wfeed_w4b` sits ON the xw path (unconditional, like
   the (A) module's contract): passthrough at w4b_en=0 means no datapath
   mux is needed at apex_top — only the route/CSR bit. Resolves the
   §5-row-5 "on the path vs at the mux" ambiguity in favor of ON-PATH;
   the `rt_wgt_src` mux in apex_top (locate by signal name — absolute
   line refs shift under the IB-LAYER glue merge) stays untouched upstream.
2. **Walker descriptor fields per W4B weight job:** (pw base, job_beats,
   K, N, s8_f16, gs base, gs_count = N*ceil(K/32)). gs rides its own
   sideband channel; the DDR image carries scales contiguous in
   ascending-gid order immediately after the packed beats (the "image IS
   the wire format" rule extends: packed || scales per job).
3. **Host prep:** `run_tinynpu.py prepare --w4-direct` quantizes DIRECT
   from the mlx-dequantized source (never via per-tensor INT8 —
   deprecated, −1.7 pt), G=32, computes per-stripe s8 by the tile-amax
   rule over the whole stripe, and grades the epilogue composite
   f16_grade(s8 × s_w) (§8.3 obligation).
4. **Enable gate:** W4B stays CSR-disabled until integration reruns the
   full matrix with it instantiated; quality evidence for the enabled
   config = docs/results/b3_w4_adoption/RESULTS.md (−1.0 pt, n=10,042).
5. Area follow-on (measure-first): the 4-pipe/16-entry-LUT variant
   halves dividers at G=32; do a synth probe before building it.

## Fences

Owns `rtl/mxe/mxe_wfeed_w4b.sv` + `verif/mxe/w4b/` + golden ADDITIONS
only. Does NOT touch: the (A) feeder or `verif/mxe/w4` (green anchor),
csr_regs/apex_top (integration's), apex_pkg (frozen). Machine: Verilator
on the c6a box (standing rule); golden tests are numpy-light. Anti-
fabrication: no PASS without pasted output.
