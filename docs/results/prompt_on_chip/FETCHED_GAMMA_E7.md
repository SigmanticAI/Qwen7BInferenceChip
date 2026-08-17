# E-7 / E-7b / E-8 — THE CONVERGENCE SESSION: three walk fences close

**Date:** 2026-08-05 · **Branch:** `comp/prompt-b-c` ·
**Twin:** `verif/f2sim/obj_e7_b64_ddr1` (fresh --Mdir, D=64 DM=896 GQA=2
QSTAGE=14 DDR=1, tile_div=5) · **Sim-proven; no hardware flown this
session.** · **Drivers:** `scripts/fpga/f2/elane_walk_qstage.py`
(--chain7/--chain8), `scripts/fpga/f2/walk_fuel_proj.py` (rund7 + the
updated rune6).

## What closed (each fence reason re-verified by execution FIRST — five
## refusal probes on the pre-change twin, all still biting as documented)

**1. NORM gamma (E-7, mask bit W2_EN_FGAM = W2_MASK[15]).** The xw -> xg
route that never existed is built: `rtl/top/glue/apex_gam_unpack.sv` (NEW)
+ the `gam_win_q` window in apex_top steer fuel beats into the norm's g
port for exactly the fetch's d_model int16 gammas; the walker opens the
window at the G1/G2 fetch dispatch and — the measured necessity — arms the
norm's OUTPUT drain itself (seq_layer_walker2 S2_GDL/GDA/GDF/GDW +
RT_YDRAIN 8'h80, the host's own drain verbs as sequencer states), because
a fetched-gamma norm emits mid-walk and dn_rms requires every y beat
accepted. NORMx-under-FPROJ-without-FGAM keeps the E-6 poisoning refusal
verbatim.

**2. QKV + attention (E-8, no new bit).** The E-6 exclusion was a row
assignment: `fp_base` stacks every act family in template order (q rows,
QKV, OPROJ, DOWN, y-drain window); legality is the fp_top capacity bound.
Masks legal before keep their exact bases — byte-identical, proven by the
E-5/E-6 gates re-passing with identical cycle counts.

**3. DOWN (E-7b, mask bit W2_EN_DOWN = W2_MASK[16]).** DOWN's pcs un-tied
from W2_EN_FFN, paired with RES2; the DOWN requant epilogue (E-6's
pc_hasrq class) walks at k = d_ffn <= walk2_k_job(D).

## The proof runs (golden the only arbiter; zero baked expectations)

```
walk_e7   ONE kick {SCORE,PV,OPROJ,RES1,NFEED,NSRC,NORM2,FGAM,FPROJ}
          attention -> fuel-fed epilogue -> r1 -> NFEED -> NORM2 with the
          REAL L00_g2 FETCHED FROM CARD DRAM -> walker-armed y-drain
          ALL grades bit-exact (o8, fs ladder incl. r1+h2 scales, h2
          codes vs golden-on-resident-gamma, r1 row); host-SILENT window
          34,080 cycles
walk_e8   + {QKV}: SIX template steps, ONE descriptor — the 24 raw QKV
          projections (real Wq/Wk/Wv prefixes, walker-fetched) bit-exact
          alongside everything above; RO drain disclosed. 72,410 cycles
rund7     {DOWN, RES2, FPROJ} at k=1984 x n=896 over the REAL resident
          Wd bytes: r2 = f16(r1 + down8'*comp) 896/896 BIT-EXACT, STRICT
          silence, 5,419,200 cycles
```

Discriminators, all fired as designed: FGAM-cleared / FGAM-without-FPROJ /
FGAM-without-NORM refusals loud; ONE flipped resident g2 byte moves h2 by
the golden-predicted codes AND r1 holds (localizes to the fetched-gamma
norm — the weight really came out of card memory through the walker's
record); walk_e7 on the PRE-change twin is RED (reserved-bit refusal); Wd
poison moves r2[0] by the predicted f16 delta; d7_fence_05b proves ON THE
TWIN that the full 0.5B d_ffn=4864 DOWN stays refused (76 stage rows vs
the 31-row bank — apex_stage_buf.sv:103-104 — and S2_PAE has no in-tile
re-staging source).

## What stays fenced, precisely

* **FFN gate/up + SWIGLU** — an ORDER redesign, not a route fix: asu_swiglu
  alternates gate/up per 64-col frame (asu_swiglu.sv:106; apex_top.sv:1547-
  1555 swg_up_q) vs the template's all-gate-then-all-up (PC_WGJ=23 <
  PC_WUJ=25), and the chunked USWI pushes deadlock at the second job.
  Closure shape: a per-64-frame interleaved sub-sequence + a re-laid
  interleaved Wg/Wu DDR image.
* **QSTAGE under FPROJ** — STRUCTURAL: the k2-injection beats are
  data-dependent (gen_l3_vectors.py:495-505), produced mid-walk, and ride
  the mailbox xw stream fuel_src=1 disconnects (cl_apex.sv:1074); no
  second xw ingress exists. Needs a fuel-delivered staging path or an
  in-tile decompose unit.
* **DOWN at 0.5B d_ffn=4864** — per-k-chunk act re-staging with no
  in-tile replay source (real datapath work).
* **NORM1** — the E-7 machinery accepts it (same pcs as NORM2) but it is
  UNFLOWN: its x feed is a host xa stream (no NFEED equivalent at the
  layer entry). Stated, not claimed.

## The honest count

**10 of 13 step families walk in a proven mode** (QKV, ROPE, STOREKV,
SCORE, PV, OPROJ+epilogue, RES1, NORM2 complete — in-tile x, FETCHED
gamma, in-tile output drain — DOWN+epilogue at k-legal geometry, RES2);
**SIX template steps compose under ONE descriptor, ONE kick** (walk_e8),
with the attention -> OPROJ-epilogue -> r1 -> NORM2 -> act-bank path
keeping the activation inside the tile (the o8 -> OPROJ-act seam stays a
host-staged copy, disclosed). Deepest measured chains: walk_e8 72,410
cycles (toy fence); rund7 5,419,200 cycles (0.5B width, k=1984).

## Green roster on this tree (all re-run this session)

seq_walker `make all` incl. mutants 1-6 + 7 NEW refuse2 cases (j-p) ·
l3 host 28/28 + walker 24/28 + tile mutants · f2sim 18-job (507 checks) +
behsmoke + capgate + CL mutants on fresh obj_d128_ddr1/0 · elane x4
(step matrix 10/10 on the new twin) · FULL E-5 gate (144/144 + poison
+381) · E-6 rune6 (CLAIM A/B cycle counts BYTE-IDENTICAL: 2,448,960 /
5,596,970; its qkvattn refusal probe updated to the capacity shape —
that refusal's disappearance IS fence 5 closing, caught by the gate
exactly as designed).
