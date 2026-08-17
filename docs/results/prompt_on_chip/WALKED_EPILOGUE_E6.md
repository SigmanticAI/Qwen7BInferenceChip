# E-6 — THE WALKED EPILOGUE: a fuel-fed projection whose product STAYS IN THE TILE

**Date:** 2026-08-05 · **Branch:** `comp/e6-walked-epilogue` (from
`comp/prompt-b-c` @ `ee393be`, the W1 third-erratum base) ·
**Twin:** `verif/f2sim/obj_e6_b64_ddr1` (DDR=1, D=64 DMODEL=64 DM=896
GQA=2 QSTAGE=14 — the b64 0.5B geometry, fresh `--Mdir`) ·
**SIM-PROVEN; no hardware flight in this session** (the owner flies; the
E-5 hw recipe applies unchanged — same descriptors, same golden, plus a
new AFI from this tree).
**[UPDATE 2026-08-05: FLOWN.** The hardware replay landed the same day on
`agfi-0bc20880b50f5faba` — r1 896/896 bit-exact in both chains, walk-off
RED, RQ-perturbation moves 801/896 — see `E6_ON_SILICON.md`.]

## The claim (exactly)

**ONE fmt=1 descriptor, mask {OPROJ, RES1, FPROJ}: APEX's layer sequencer
fetched the real 896x896 Qwen2.5-0.5B `L00_Wo` from (behavioral) card DRAM
by its own record, ran the o-projection on the real C-1-framed
attention-output row with the requant epilogue ON-TILE (`requant_en=1`,
RQ[14] — the encoding that had never executed), and the o8 product flowed
through the tile's own serializer → `apex_layer_deq` (JC composite) →
`apex_residual` chain into the resident row: r1 = f16(X + o8'·comp),
896/896 BIT-EXACT against golden's own composition on the same operands —
and the walk window contains ZERO host operations. E-5's disclosed RO
drain is GONE, because the projection's product never leaves the tile.**

```
[CLAIM A]  walk_fuel_oproj: mask {OPROJ, RES1, FPROJ}
           112 fuel-fed MXE jobs, requant epilogue on-tile,
           r1 896/896 BIT-EXACT vs golden f16(X + o8*comp)
           host writes in the walk window: 0 (STRICT — no drain exists)
           walk window: 2,448,960 sim cycles
[CLAIM B]  walk_fuel_qkv_oproj: mask {QKV, OPROJ, RES1, FPROJ}
           TWO fuel-fed projections, ONE kick, 256 MXE jobs:
           144/144 QKV blocks bit-exact (raw INT32, RO drain disclosed)
           AND r1 896/896 bit-exact — 5,596,970 sim cycles
[CHAIN]    walk_e6 (toy 64, H=1, T=1): ONE kick, mask {SCORE, PV, OPROJ,
           RES1, NFEED, NSRC, FPROJ} — walked attention (o8 requant
           on-tile) -> fuel-fed OPROJ + epilogue -> r1 in the residual row
           -> NFEED feeds THE EPILOGUE'S OWN PRODUCT to the RMSNorm.
           Host between WALK_GO and completion: NOTHING.
[DISC]     walk-off (FPROJ cleared): RED (the measured p_oproj class)
           one poisoned resident-Wo byte (golden-SEARCHED — requant
           absorbed the first candidate, measured): r1[0] moves by the
           exactly-predicted f16 delta
           V-record flip: the walked o8 moves, r1 does NOT — the o8 ->
           OPROJ-act seam is a host-staged copy, MEASURED (see fences)
[FENCES]   4 pkg-legal images refused at S2_CHECK on the tile: OPROJ
           without RES1; FFN under FPROJ; QKV+attention under FPROJ;
           QSTAGE under FPROJ (fuel_src=1 disconnects the mailbox xw
           stream its k2 injection rides — the wedge became a refusal)
```

## What the RTL grew (all `pc_hasrq`/`EN_FPROJ`-gated; legacy byte-identical)

`rtl/seq/seq_layer_walker2.sv`:
1. **S2_PSJ** — the walker pushes the serializer job framing the whole o8
   stream (n_splits beats × 8 = the deq job's cols) before the first act
   EMIT. With PC_URES1's residual job and PC_UOPRJ's JOBC+deq job already
   pushed in template order, the entire consumer chain is armed before
   any result beat exists.
2. **RT_FPRQ = 8'h94** — RT_FPROJ with exactly ONE field moved: rdst
   0 → 1, the measured host-mode o8-leg destination
   (`gen_layer_ops.py:795`). Selected by `fp_rq_q` (the registered
   pc_hasrq of the dispatched projection) and held through S2_PJW.
3. **fp_oproj_base** — the OPROJ act family emits from its own bank-1 row
   window, above the staged-q rows / the QKV family.
4. **S2_PDW, the pre-epilogue drain fence** — measured on the chain: the
   PV walk retires with its o8 tail still in the result pipeline and the
   epilogue's rdst flip is ~10 cycles behind it, so the tail misrouted
   into the serializer. S2_PDW holds the window entry (no valid
   presented — D-020-clean) until `tile_idle`.
5. The widened S2_CHECK fence (`fp_mask_ok`): OPROJ↔RES1 paired;
   NORM1/NORM2/FFN/RES2 refused (armed-fuel gamma poisoning; the
   all-gate-then-all-up template order starves `asu_swiglu`'s per-frame
   phase alternation); QSTAGE refused (dead mailbox xw under fuel_src);
   QKV excludes SCORE/PV (act-bank collision); ONE k-split
   (`d_model <= walk2_k_job(FEED_DM)` — the W1 third-erratum bound);
   family footprint ≤ STAGE_ROWS; o8 frame ≤ 255 serializer beats.

`verif/f2sim/sim_main.cpp`: `+poll_limit=` (default 2M unchanged) — the
host-silent window puts the whole fuel-fed projection under ONE done-poll
(~21.7k cyc/job at tile_div=5 ⇒ 112 jobs > 2M); claim runs pass 16M,
every RED/refusal run keeps the 2M stall detection. `CYCMARK` notes print
the sim cycle counter for the walk-window measurement.

## The honest step count

Of the layer's 13 template steps (NORM1, QKV, ROPE, STOREKV, SCORE, PV,
OPROJ, RES1, NORM2, FFN-gate/up, SWIGLU, DOWN, RES2):

* **Walked AND fuel-fed under ONE descriptor: 3** — {QKV, OPROJ, RES1}
  (CLAIM B, the real 0.5B geometry, 256 jobs, one kick).
* **Composed under ONE kick with attention: 5 (+NSRC)** — {SCORE, PV,
  OPROJ, RES1, NFEED} (the toy chain), host writing NOTHING inside.
* **The activation never leaving the tile: the OPROJ→RES1→NORM2-input
  path** — o8' → deq → residual → (NFEED) → norm x, end to end in-tile.
  QKV's raw INT32 still leaves via the RO lanes (disclosed); the
  attention-o8 → OPROJ-act seam is a host-staged copy (the V-flip
  discriminator measures exactly that seam).
* **Walked in some proven mode: 8 of 13** — QKV(fueled), ROPE(arm),
  STOREKV(addressing), SCORE, PV, OPROJ(fueled + epilogue), RES1(fed by
  that epilogue), NORM2's front half (NFEED + in-tile arithmetic; gamma
  host-loaded).
* **Not walkable today, each a LOUD S2_CHECK refusal, never a wedge:**
  NORM1 & NORM2's gamma fetch (no xw→gamma route; with fuel armed the
  fetch would poison the weight stream — `seq_layer_walker2.sv`
  fp_mask_ok block), FFN gate/up + SWIGLU (template order PC_WGJ < PC_WUJ
  vs `asu_swiglu`'s per-64-frame gate/up alternation, `apex_top.sv:1548`),
  DOWN (mask-inseparable from FFN, `PC_WDF/WDJ` gate on `W2_EN_FFN`; its
  k = 4864 decomposition is now LEGAL per `walk2_k_job()` (ee393be) but
  the walked act re-staging per k-chunk is not carried — S2_PAD note),
  RES2 (starved without DOWN).

**What the host still does:** load the DDR weight image (pre-flight,
once, sha-gated); pre-walk per step: the X row, the C-1 activation
families, KV records, q pre-staging (for attention kicks), gamma, the
calibration slots (RQ/QC/JC — the descriptor's documented host-loaded
fields), the 64-word descriptor, fuel-mode arm; then WALK_GO, ONE
completion poll, and the post-walk drains/readbacks.

## Measured cycles, and the labelled PREDICTION

Measured on the DDR=1 twin at tile_div=5 (shell-cycle counter).
[RECEIPTS CORRECTION, same day — independent re-verification: the sim
divider TOGGLES clk_tile every tile_div shell posedges, i.e. tile period
= 2·tile_div shell cycles (`cl_apex.sv:82` "divide clk_main_a0 by
2*tile_div", freq = 250/(2N) MHz), so the original tile-cycle column here
(shell/5) was 2× HIGH. Cross-proof: the independent replication driver
measured the SAME walks at tile_div=2 — 979,596 / 2,238,788 shell — and
shell/(2·div) agrees across both ratios to within 3 tile cycles
(WALKED_EPILOGUE_E6_REPLICATION.md). Figures below are corrected.]

| walk | jobs | window (sim cyc) | tile cyc (= shell/10) |
|---|---|---|---|
| CLAIM A {OPROJ, RES1} | 112 | 2,448,960 | ~245k |
| CLAIM B {QKV, OPROJ, RES1} | 256 | 5,596,970 | ~560k |

⇒ ~21.9k shell ≈ ~2.19k tile cycles per fuel-fed job = **~3.3 weight
bytes per tile cycle** sustained ingest (7168 B/job), CDC + behavioral
DDR included.

**PREDICTION (labelled, sim-derived, hardware-unvalidated):** a FULLY
walked 0.5B layer streams ~14.25 MB of weights (the image's own
per-layer footprint). At the measured 3.3 B/cyc and a 62.5 MHz
tile clock: ~4.4M tile cycles ≈ **70 ms/layer ⇒ ~1.7 s/token over 24
layers**; at the tile's 8 B/cyc peak ingest the floor is **~14.3
ms/layer ⇒ ~0.35 s/token**. (At the 15.625 MHz clock the flown A2 images
verified, 4× those walls.) Host steady-state control cost at 3 MMIO/step
(~1k MMIO/token) adds ~milliseconds. Against today's measured ~4.5
min/token at 8% offload (per-op dispatch dominated), that is a
**~160–760× per-token improvement IF the whole layer walks** — the number
depends on silicon ingest rate, and the un-walked five steps (NORM1,
FFN/SWIGLU/DOWN/RES2) must land first; with only today's walked coverage
the token wall stays dominated by the host-side FFN.

## Gate roster on this tree (all green, this session)

`verif/seq_walker make all` (incl. every mutant gate + W1's mutants6) ·
l3 host 28/28 + WALKER-MODE all-passed · f2sim 18-job 27,996 checks/0
fails + behsmoke ×2 + capgate 505/505 + fuel mutants 3/3 · elane
norm_feed/walk_norm/walk_qstage/walk_steps 10/10 · host-mode fuel gate
PASS · the full E-5 gate PASS re-run on this twin (144/144, silence,
walk-off RED, poison RED by the predicted +381) · `rune6` PASS · the toy
chain PASS (this file's header claims).

## Cost of the found-the-hard-way items (kept, in the tree)

* The S2_CHECK `wf_ready` refusal fired on the first emitted E-6 program
  (fuel un-armed) — the fence works, loudly.
* The first poison byte was ABSORBED by requant (o8 7→7) — poisons are
  now golden-searched (`poison_search`).
* A refused walk has zero activity for `final_phase` to assert — refusal
  probes end at DONE (the e4 idiom).
* The toy o-act row's amax is 126, not 127 — the identity-scale staging
  shortcut only holds for `c1_frames` subjects; the staged codes are
  `quant_rows_i8(o8_vals)` and the ess check carries the row's real
  scale.
* The attention-tail/route-flip hazard (S2_PDW above).
