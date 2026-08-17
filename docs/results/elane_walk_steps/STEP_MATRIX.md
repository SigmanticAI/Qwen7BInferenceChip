# The walked-step matrix — measured by execution, 2026-08-04

**Question (E2E_TOY_LANE.md, the E-3b/E-4 lane):** which of the fmt=1 layer
walker's steps can run WALKED against the REAL datapath today?
**Method:** one probe program per step (`scripts/fpga/f2/elane_walk_steps.py`),
each a fmt=1 descriptor enabling ONLY that step, run on the verilated b64
twin (`verif/f2sim/obj_b64_05b*`: CFG_D=CFG_DM=64, GQA_NENG=2,
QSTAGE_H_MAX=14, DM_MAX=896) at the E-lane toy fence (head_dim == d_model
== 64, H=1). Every expected outcome is a CHECKED read; 10/10 probes behave
exactly as stated (`build/f2_elane_steps/step_matrix.json`). Golden is the
only arbiter wherever data is produced.

## The matrix

| step (fmt=1 pc) | verdict | receipt |
|---|---|---|
| NORM1 (PC_G1F+PC_NORM1) | **parks at the gamma FETCH** | class of p_norm2: fuel-only wf_ready; ALSO no xw->xg route exists, so a fetched gamma is a dead end even with fuel — gamma stays host-fed (xg) |
| QKV (PC_WQF..WVJ) | **parks at S2_FETCH** (Wq fetch, first enabled pc) — *unless W2_MASK[14] (EN_FPROJ) is set, in which case it is now **WALKED, FUEL-FED, BIT-EXACT** (E-5 below)* | p_qkv: WALK busy stuck, no err sticky; no abort path out of S2_FETCH (seq_layer_walker2 abort set excludes it) |
| — q half via **QSTAGE (E-3b, new)** | **WALKED, bit-exact** | walk_qstage: 64/64 codes + scale == golden; **H=14 @ 0.5B geometry: 896/896 + 14/14 scales**; walk-off RED; pre-E-3b twin RED |
| ROPE (PC_ROPE) | **WALKED** (level arm on the real regs) | p_rope: walk retires clean; LAYER_CTRL readback == the walker's arm |
| STOREKV (PC_STORE) | **WALKED** (addressing) | p_storekv: two real AXI WRITE_ADDR transactions, engine idle after, walk clean. Record DATA still streams via the host squant path (pre-walk) |
| SCORE+PV (PC_ATTN) | **WALKED, bit-exact** | l3 walker modes re-run green (24/28 walked fmt=0 + fmt=1 walkfmt2); **walk_e4: o8 64/64 on the b64 twin, ON the walker-staged q row, requant on-tile** |
| OPROJ (PC_LVLO..WOJ) | ser_dst arm + JOBC + real deq-job push land (receipts green), then **parks at the Wo FETCH**; even with fuel: rt_wgt_src pinned 1 in walk mode + rt_res_dst never 1 -> operands unreachable | p_oproj; facts F2/F3 in the probe header |
| RES1 (PC_URES1) | **control-WALKED / data-starved** | p_res1: real apex_residual accepts the push, walk retires; unit busy STUCK (no walked step can route the deq stream) |
| NFEED (PC_NFEED, E-3a) | **WALKED, bit-exact** at d=64 | p_nfeed: r1 + h2 codes + both C-1 scales == golden; walk-off RED at the route-arm poll |
| NORM2 (PC_G2F+PC_NORM2) | NFEED half WALKED (receipts green), then **parks at the G2 fetch**; norm arithmetic itself is stream-triggered and completes in-tile once gamma arrives (host xg) | p_norm2; p_nfeed proves the arithmetic path |
| FFN gate/up + SWIGLU (PC_LVLG..WUJ) | ser_dst=2 arm + swiglu JOBC+job accepted (receipts green), then **parks at the Wg FETCH** | p_ffn |
| DOWN / RES2 | same classes as OPROJ / RES1 (fetch park / data starvation) | by construction — identical pc kinds, same consumers |

**New measured findings along the way**
* A zero descriptor-JC composite is REFUSED by the real deq unit (LAYER
  sticky + code 2, push consumed, walker proceeds to the fetch park) — p_jc0.
* MODE_F16 emits no s-tap (ss) beat; the walked S-4 does (1 beat).
* The walked score's TIP decision beat lands in the td FIFO and must be
  drained (ETIP).
* `l_nsrc` WAS a host-held level: one walk could not both stage q
  (codes->act) and nfeed (codes->norm) — the measured two-kick shape.
  **RETIRED by E-4b (2026-08-04):** the previously-reserved mask bit
  `W2_EN_NSRC` (W2_MASK[13]) makes the walker OWN `l_nsrc` for that walk
  (E2E_TOY_LANE.md §4 E-4b) — `walk_e4` is ONE kick on the rebuilt twin
  (`obj_b64_05b_e4b`), all grades preserved, host writing nothing between
  walker start and completion; a {QSTAGE, NFEED} image WITHOUT the bit is
  now loudly REFUSED (e4_fence_nonsrc), as is NSRC-without-NFEED
  (e4_fence_nsrconly); the single-kick image is RED on the pre-E-4b twin
  (reserved-bit refusal, `--redproof`).

## The honest claim after E-3b/E-4

Of the layer's steps, **six now run WALKED on the real datapath** — QSTAGE
(the q half of QKV), ROPE, STOREKV(addressing), SCORE, PV, NFEED — and
**four of those keep the activation entirely in-tile** (QSTAGE's
squant->rope->feeder->act-bank chain; SCORE/PV's KVQ->feeder->MXE->ASU
chain up to the o8 egress; NFEED's residual->feeder->norm chain). The
walked E-4 chain (staged q -> attention -> o8, then X -> norm) is
bit-exact vs golden with the host doing only: table/record/X loads,
calibration slots (RQ/QC — the descriptor's documented host-loaded
fields), the paced xw data stream, and the post-walk drains — **ONE kick
since E-4b (2026-08-04): the between-kicks level flip is the walker's own
NFEED arm now** (mask bit NSRC; the post-walk LAYER_CTRL word is CHECKED
with bit 19 set by the sequencer).

**What still blocks a fully-walked FULL layer** (each measured, not
assumed): ~~(1) K_FETCH requires fuel mode and has no abort path — and fuel
delivers to xw only, dead-ended by (2) the walk-mode route pins
(rt_wgt_src=1, rt_res_dst never 1)~~ — the QSTAGE route-override (`rt_f1_q`)
was indeed the template, and **E-5 lifted (1), (2) and (3) together for the
QKV step**; ~~(3) the projection steps emit no act/wgt staging jobs (S2_PJOB
is ds-only)~~ RESOLVED for QKV by E-5's S2_PAE; (4) ~~`l_nsrc` is host-held~~
RESOLVED by E-4b (walker-owned via W2_MASK[13]);
(5) at DM > CFG_DM the C-1 feeder chunks the row family with per-chunk
scales (B-FEED-WIDTH) — the wide-feeder item, out of this lane.
**Still open after E-5:** OPROJ/FFN/DOWN remain fetch-parked — their pc
programs also push LAYER-unit jobs that no walked step can feed (the RES1
data-starvation class), so E-5 FENCES an FPROJ mask to QKV-only rather
than degrading. And the o8 requant epilogue is still unproven walked: it
lives on PC_WOJ/PC_WDJ (`pc_hasrq`), which the fence excludes.

## E-5 — THE CONVERGENCE: a WALKED, FUEL-FED projection (2026-08-04)

`scripts/fpga/f2/walk_fuel_proj.py run` — **PASS**, on the clean DDR=1 b64
twin (`verif/f2sim/obj_e5_b64_ddr1`, D=64 GQA=2 DM=896 QSTAGE=14 DMODEL=64).
ONE fmt=1 descriptor, mask exactly {EN_QKV, EN_FPROJ}:

* **144 MXE jobs graded BIT-EXACT vs golden** — Wq (112 n-jobs) + Wk (16) +
  Wv (16), 1296 RO captures, `bad=0`, golden =
  `apex_golden.compute.gemm_i8_ksplit` computed AFTER the run on the same
  INT8 operands. Zero baked expectations: every RO site is a `cap` (pure
  egress, no `e`), enforced by an audit in the emitter.
* The **SEQUENCER** issued every fuel record from its own descriptor tensor
  table (W2_TENS0) — the host arms fuel MODE only (src=1, ar_owner=1) and
  writes no base/beats/GO.
* **host-write silence**: 0 control writes between WALK_GO and the
  walk-done poll (1586 ops in the window; the 144 writes are all RO FIFO
  advances — a data-plane drain, disclosed, because that FIFO is 16 deep
  and one job is 9 entries). Checked on the emitted artefact.
* **walk-off discriminator**: the identical descriptor with W2_MASK[14]
  cleared is RED — the legacy path takes the measured S2_FETCH/starve
  behaviour and produces nothing.
* **DDR poison discriminator**: one flipped byte at
  `L00_Wq + tensor byte 448` (weight W[k=56][lane 0], activation -127)
  moves lane 0 by **exactly the predicted 381** — the weights really came
  out of card memory.

Three RTL deltas made it, all gated on the reserved bit W2_MASK[14] so every
legacy image is byte-identical (`verif/seq_walker make all` green including
the check-equivalence mutants; elane 10/10; l3; f2sim 18-job + capgate +
mutants; the host-mode fuel gate unchanged):
1. `RT_FPROJ = 8'h84` (seq_layer_walker2) — the route override, copied bit
   for bit from the host-mode proof's own projection route word
   (`gemm_job.py:355`), so `rt_wgt_src=0` puts the MXE weight port on the
   external xw stream that cl_apex muxes to the fuel FIFO, and `rt_res_dst=0`
   sends the accumulators to the RO lanes. Held through a drain state
   (S2_PJW) so a mid-egress revert cannot misroute the tail.
2. `S2_PAE` — the walker emits the MXE's activation family itself
   (PAT_ROW EMITs from act bank 1, replayed per n-job), in the PROVEN
   EMIT(row0)/DESCRIPTOR/EMIT(rest) order; emitting the whole family first
   deadlocks on `apex_stage_buf`'s D-006 job_ready.
3. **The provisional projection opcode was WRONG and nothing had caught it**,
   because no walked projection had ever executed: `pj_desc` emitted
   `OP_GEMM_OS/mode_os=1`, but the resident DDR image is laid out
   WEIGHT-STATIONARY (`make_weight_image.block_bytes`, cross-checked against
   the frozen `gen_l3_vectors.wgt_beats_ws`). FPROJ emits `OP_GEMM_WS`; the
   legacy encoding is left untouched rather than silently redefined.

The S2_FETCH wedge is **fenced, not fixed**: D-020 forbids aborting by
retracting a presented `wf_valid`, so the escape is a PRE-ENTRY refusal at
S2_CHECK (`wf_ready` read as a level — fuel armed?), which turns "hangs the
tile forever" into a loud WALK_ERR_DESC. A fuel-mode drop MID-walk still
wedges; that needs a non-retracting abort or a timeout, and neither is here.

**Gate-integrity finding, 2026-08-04 — discriminator verdicts on runs that
never ran.** The E-5 session's `build/f2_elane_walk/report.json` was produced
by pointing the D=128 walk suite at the D=64 fuel twin
(`obj_e5_b64_ddr1`): phase A's INFO_D check refused every program (`FAIL L6:
[100c] got 00000040 want 00000080`), zero captures. The smoke's TOP verdict
failed correctly — but the discriminator table LIED in both directions: the
old empty-capture fallbacks reported `walk_off`/`x_ulp`/`cap_code`/
`cap_scale` as `caught=True` (vacuous RED — nothing executed) and `gamma` as
`caught=False` ("not biting"). Root cause was the fallbacks, not the gamma
perturbation: on a clean fresh-Mdir D=128 twin the gamma disc bites for real
(`ran=true, r1_equal=true, codes_equal=false`; last provable silicon RED
2026-08-02, agfi-006b1314fcbbb3505). Fixed in all four elane emitters
(`elane_norm_feed.py` hygiene helpers + gated verdicts, imported by
`elane_walk_norm.py` / `elane_walk_qstage.py` / `elane_walk_steps.py`): a
perturbation verdict now requires (a) the UNPERTURBED control green on the
SAME binary, (b) the perturbed run executed with gradeable captures, and
(c) for stall-shaped discs, the abort AT the discriminated seam
(`walk_off`/`p_nfeed_off`: `poll stall … [1070]` route-arm; `walk_qstage_off`:
`pw stall` xw pacing) — otherwise the verdict is WITHHELD with the reason
and the gate fails loudly. Residual, stated honestly: the EXPECT-RED wedge
probes above (p_oproj/p_qkv/p_ffn/p_jc0/p_norm2) still count ANY failure as
as-expected at row level; on a wrong twin the matrix fails via its GREEN
probes, but those rows read misleadingly — tighten with per-probe failure
signatures if they ever anchor a claim alone.

**Build-hygiene finding worth keeping** (cost an hour of false regression):
re-verilating IN PLACE over an obj dir built with a DIFFERENT define set
leaves stale objects behind. The reused `obj_b64_05b_ddr1` advertised
INFO_TIER=3 where a clean build of the same RTL and same defines advertises
1, and 4 elane probes went red on it while the identical RTL at DDR=0 and
the pre-change RTL at DDR=1 were both 10/10. Always verilate a NEW `--Mdir`
for a new configuration.

## E-6 — THE WALKED EPILOGUE (2026-08-05, sim-proven on the DDR=1 b64 twin;
## FLOWN the same day on `agfi-0bc20880b50f5faba` — `E6_ON_SILICON.md`)

E-5's fence excluded the o8 requant epilogue (`pc_hasrq`). E-6 un-fences it
for OPROJ (`rtl/seq/seq_layer_walker2.sv`: S2_PSJ + RT_FPRQ 8'h94 (rdst
0->1, the measured host o8-leg destination) + the fp_oproj_base act-row
window + the widened fp_mask_ok fence; every delta pc_hasrq/FPROJ-gated so
E-5 and legacy walks are byte-identical — the whole green roster re-ran to
prove it). `walk_fuel_proj.py rune6` on obj_e6_b64_ddr1:

* **CLAIM A, mask {OPROJ, RES1, FPROJ}**: the walker fetches the REAL
  896x896 L00_Wo by its own record, runs the projection with requant_en=1
  (RQ[14]) and its o8' lands in the residual row THROUGH the tile's own
  serializer -> apex_layer_deq (JC composite) -> apex_residual chain:
  **r1 = f16(X + o8'*comp) 896/896 BIT-EXACT** vs golden's own composition
  on the same operands. **The walk window contains ZERO host operations**
  — E-5's RO drain is gone because the epilogue's product never leaves the
  tile. Walk window 2,448,960 sim cycles.
* **CLAIM B, mask {QKV, OPROJ, RES1, FPROJ}**: TWO fuel-fed projections,
  ONE descriptor, ONE kick — 256 MXE jobs: 144/144 QKV blocks bit-exact
  (RO drain disclosed) AND r1 896/896 bit-exact. 5,596,970 sim cycles.
* Chain (toy fence, `elane_walk_qstage.py --chain` walk_e6): ONE kick,
  {SCORE, PV, OPROJ, RES1, NFEED, NSRC, FPROJ} — walked attention, then
  the fuel-fed epilogue, then NFEED feeds THE EPILOGUE'S OWN r1 to the
  norm. q host-PRE-staged: **QSTAGE+FPROJ is REFUSED at S2_CHECK**
  (fuel_src=1 disconnects the mailbox xw stream its k2 injection rides,
  cl_apex.sv:1074 — the wedge became a loud fence, proven on the tile).
  The V-flip discriminator moves the walked o8 and NOT r1 — the o8 ->
  OPROJ-act seam is a host-staged copy, MEASURED and disclosed.

**Matrix updates** (OPROJ row): *walked, fuel-fed, epilogue ON-TILE,
bit-exact* under EN_FPROJ with the OPROJ<->RES1 pair; RES1 row: *walked
AND data-fed* (the deq stream is the walked epilogue's own o8'). Still
fetch-parked/refused: NORM1/NORM2 (gamma fetches would poison the armed
fuel stream — no xw->xg route), FFN gate/up + SWIGLU (the
all-gate-then-all-up template order vs asu_swiglu's per-64-frame gate/up
alternation — starvation), DOWN (mask-inseparable from FFN; its
k = d_ffn = 4864 decomposition is LEGAL since the W1 third erratum
(walk2_k_job), but a walked DOWN needs per-k-chunk act re-staging S2_PAE
does not carry, plus a walkable swiglu producer), RES2 (starved without
DOWN).

**The honest claim after E-6:** of the layer's 13 template steps, EIGHT
have now run WALKED against the real datapath in some proven mode — QKV
(fuel-fed), ROPE (arm), STOREKV (addressing), SCORE, PV, OPROJ (fuel-fed
WITH its requant epilogue), RES1 (fed by that epilogue), NORM2's front
half (NFEED + in-tile arithmetic, gamma host-loaded) — and up to SIX
compose under ONE kick ({SCORE, PV, OPROJ, RES1, NFEED} + the NSRC arm;
or {QKV, OPROJ, RES1}). The OPROJ -> RES1 -> NORM2-input path keeps the
activation inside the tile end to end. NOT walkable today: NORM1, FFN
gate/up, SWIGLU, DOWN, RES2 (five steps, reasons above, each fenced with
a loud S2_CHECK refusal rather than a wedge).

## E-7 / E-7b / E-8 — THE CONVERGENCE SESSION (2026-08-05, sim-proven)

**Assignment: close the remaining walk fences.** Every fence reason was
RE-VERIFIED BY EXECUTION first (5 refusal probes on obj_e6_b64_ddr1, all
still biting as documented) — none had shifted. Verdicts:

**FENCE 1 — NORM gamma (CLOSED, E-7).** The measured reason ("no xw->xg
route"; apex_top's xw port had exactly one consumer, the MXE weight mux
apex_top.sv:2317-2322, and asu_rmsnorm's g port exactly one producer, the
host xg stream) is closed the house way — reserved mask bit W2_EN_FGAM
(W2_MASK[15]) arms a walker-owned GAMMA WINDOW level (`lw_gsrc`):
* apex_top steers xw beats into the NEW `rtl/top/glue/apex_gam_unpack.sv`
  (4 x int16 LE per beat — make_weight_image.gamma_payload's exact
  layout) feeding the norm's g port; `gu_busy` joins blk_busy[2] so
  tile_idle covers the window tail. gam_win_q = 0 on every legacy walk —
  all four xw/g nets reduce to their exact old forms.
* A fetched-gamma norm EMITS mid-walk, so the walker performs the host
  drain choreography ITSELF (measured wedge otherwise — dn_rms requires
  every y beat ACCEPTED): S2_GDL (levels + RT_YDRAIN = the host's own
  measured 0x80 drain route) / S2_GDA (act-LOAD, n_heads rows at the
  fp_top window) / S2_GDF (C-1 feeder job) / S2_GDW (route hold until
  tile_idle — the S2_PJW tail rule).
* Fences kept/added at S2_CHECK: NORMx-under-FPROJ-without-FGAM keeps the
  E-6 poisoning refusal VERBATIM; FGAM requires FPROJ + exactly one NORM
  step; the y-drain envelope (NF_ROWS_EXACT, n_heads <= FEED_ROWS,
  fp_top + n_heads <= STAGE_ROWS).

**FENCE 5 — QKV+attention (CLOSED, E-7 fp_base).** The collision was a ROW
ASSIGNMENT, not structure: the QKV act family emitted from rows
0..fp_rows-1 — the staged-q rows. `fp_base` now stacks every family in
template order (q rows, QKV, OPROJ, DOWN, then the y-drain window);
legality is the fp_top capacity bound. At 0.5B H=14: {QKV, SCORE, PV,
FPROJ} = 28 rows FITS (legal now); adding OPROJ = 42 > 31 is REFUSED with
the measured reason (rune6's e6_fence_qkvattn_cap probe; unit refuse2
case (p)). Masks legal before keep their exact bases — byte-identical.

**FENCE 3 — DOWN (PARTIALLY CLOSED, E-7b).** W2_EN_DOWN (W2_MASK[16])
un-ties DOWN's pcs from W2_EN_FFN, paired with RES2 (the OPROJ<->RES1
rule). The DOWN requant epilogue — E-6's pc_hasrq class, RQ[H+1] +
JC_DOWN + the S2_PSJ frame + RT_FPRQ — is PROVEN WALKED at the k-legal
geometry (`walk_fuel_proj.py rund7`, obj_e7_b64_ddr1): k = d_ffn = 1984
(= walk2_k_job(64), 31 whole stage rows), n = the REAL d_model = 896, the
REAL resident Wd region's first 1984x896 bytes fetched by the walker's own
record, **r2 = f16(r1 + down8'*comp) 896/896 BIT-EXACT**, STRICT
host-silence (zero window ops), walk window 5,419,200 sim cycles; Wd
poison moves r2[0] by the predicted f16 delta; 4 fence refusals proven on
the twin. **The 0.5B d_ffn = 4864 walk stays REFUSED with a measured
reason, ON THE TWIN (d7_fence_05b):** 4864/64 = 76 stage rows cannot be
resident in the 31-row bank (apex_stage_buf.sv:103-104 R_MAX cap) and
S2_PAE has no in-tile re-staging source (it EMITs a HOST-staged resident
family; nothing re-fills act bank 1 mid-walk) — per-k-chunk act
re-staging is REAL DATAPATH WORK (a mid-walk family producer), left
fenced at seq_layer_walker2's one-k-split clause, not half-open. The act
family is host-staged (the disclosed E-6 o8->act seam class); the in-tile
producer is fence 2's interleave.

**FENCE 2 — FFN/SWIGLU (STAYS FENCED — an order redesign, not a route
fix).** Re-verified: asu_swiglu consumes ONE gate frame THEN one up frame
per job (asu_swiglu.sv:106 ST_GATE/ST_UP, <=64 cols) and apex_top's
swg_up_q phase tracker flips per accepted `last` (apex_top.sv:1547-1555),
while the walker template runs ALL gate jobs then ALL up jobs (PC_WGJ=23
< PC_WUJ=25, seq_layer_walker2.sv) and pushes ALL chunked USWI jobs
before either (PC_USWI=21) — the second USWI push DEADLOCKS (one pending
LAYER job; its frames need pcs the walker has not reached), and a walked
gate stream would be eaten as up data. Closing it needs: a per-64-frame
interleaved sub-sequence ({USWI job, sj frame, 8 gate n-jobs, sj frame,
8 up n-jobs} x d_ffn/64, with rdst=1 routing raw INT32 to the serializer)
PLUS a re-laid interleaved Wg/Wu DDR image so ONE fuel stream delivers
beats in the new consumption order. Named follow-on with this shape;
refused at fp_bad_steps (FFN) meanwhile.

**FENCE 4 — QSTAGE-under-FPROJ (STRUCTURALLY ILLEGAL today — documented,
kept fenced).** The k2-injection weight beats are DATA-dependent: w0/w1 =
the decomposition of the very f16 values being staged, produced MID-walk
(gen_l3_vectors.py:495-505 inject_jobs) and streamed by the host on the
MAILBOX xw path — which fuel_src=1 disconnects (cl_apex.sv:1074-1077,
the ONE xw mux). They cannot be pre-baked into a DDR image (not known
until the walk runs) and no second xw ingress exists. Closing it needs a
fuel-delivered staging path or an in-tile decompose unit — real datapath
work. The refusal stays (re-proven on the twin this session).

**The chains (all on obj_e7_b64_ddr1, fresh --Mdir, tile_div=5):**
* `elane_walk_qstage.py --chain7` **walk_e7 PASS** — ONE kick, mask
  {SCORE, PV, OPROJ, RES1, NFEED, NSRC, NORM2, FGAM, FPROJ}: walked
  attention -> fuel-fed epilogue -> r1 -> NFEED -> **NORM2 with the
  gamma FETCHED from card DRAM** -> walker-armed y-drain -> h2 codes in
  act bank row 2. ALL grades bit-exact (o8, the fs ladder incl. the r1
  AND h2 scales, h2 codes vs golden-on-the-resident-gamma, r1 row);
  host-SILENT window, 34,080 cycles. Discriminators: FGAM-cleared
  refusal, FGAM-without-FPROJ, FGAM-without-NORM (all loud); ONE
  resident g2 byte flipped -> h2 moves by the golden-predicted codes AND
  r1 HOLDS (localizes to the fetched-gamma norm); the identical image on
  the PRE-change twin (obj_e6_b64_ddr1) is RED (reserved-bit refusal).
* `--chain8` **walk_e8 PASS** — fence 5's positive arm: + {QKV} in the
  SAME kick (rows q=0/QKV=1/o-act=2/y-drain=3, the walker's own stack):
  the 24 raw QKV projections (REAL Wq/Wk/Wv region prefixes,
  fuel-fetched) graded bit-exact alongside everything walk_e7 grades —
  SIX template steps under one descriptor, 72,410 cycles. RO drain
  disclosed (216 entries vs the 16-deep FIFO — the E-5 data plane).
* Green roster on this tree: seq_walker `make all` (incl. mutants 1-6 +
  7 NEW refuse2 cases j-p), l3 host 28/28 + walker 24/28 + tile mutants,
  f2sim 18-job + behsmoke + capgate + CL mutants (fresh obj_d128_ddr1/0),
  elane x4 (step matrix 10/10 on the new twin), FULL E-5 gate re-run
  (144/144 + poison +381) and E-6 rune6 re-run on the new twin (CLAIM
  A/B cycle counts BYTE-IDENTICAL: 2,448,960 / 5,596,970; its qkvattn
  probe updated to the capacity shape — the old refusal is the closed
  fence).

**Matrix updates:** NORM2 row: **fully walked** — NFEED + in-tile
arithmetic + FETCHED gamma + walker-armed output drain. DOWN row:
**walked, fuel-fed, epilogue ON-TILE, bit-exact** under EN_DOWN at
k <= walk2_k_job(D) (0.5B k=4864 refused, measured reason above). RES2
row: **walked AND data-fed** (by the DOWN epilogue's own stream). QKV
row: **composes with attention** under one descriptor (fp_base). NORM1
row: the E-7 machinery ACCEPTS it (same pcs/kinds as NORM2; S2_CHECK
legal with FGAM) but it has NOT been flown — its x feed is a host xa
data stream (no NFEED equivalent exists at the layer entry), stated
rather than claimed.

**The honest claim after E-7/E-7b/E-8:** of the layer's 13 template
steps, **TEN have now run WALKED against the real datapath in a proven
mode** — QKV, ROPE, STOREKV, SCORE, PV, OPROJ (+epilogue), RES1, NORM2
(complete: in-tile x feed AND fetched gamma AND in-tile output drain),
DOWN (+epilogue, at k <= walk2_k_job(D)), RES2 — and **SIX compose under
ONE descriptor, ONE kick** ({QKV, SCORE, PV, OPROJ, RES1, NORM2} +
the NFEED/NSRC arms, walk_e8), with the OPROJ -> RES1 -> NORM2 ->
act-bank path keeping the activation inside the tile end to end (the
o8 -> OPROJ-act seam stays a host-staged copy, disclosed). NOT walkable
today: NORM1 (unflown, machinery present), FFN gate/up + SWIGLU (the
interleave redesign, fence 2), and DOWN at the full 0.5B d_ffn (the
re-staging datapath, fence 3's remnant) — each fenced loudly at
S2_CHECK, with its closure shape named above.

## A card run of this needs — **[SATISFIED 2026-08-05]**

*(Historical, written 2026-08-04.)* The then-current **b64v4 silicon image
(agfi-0ecab46b8a8376b21) has E-1/E-2/E-3a but NOT E-3b/E-4b** — the
descriptor's W2_MASK[12] (and, post-E-4b, W2_MASK[13]) would be REFUSED
(WALK_ERR_DESC, the old reserved-bit clause; loud, safe). Flying walk_qstage
/ the one-kick walk_e4 therefore required a **new AFI** built from a tree at
or after the E-4b commit (same CL knobs as b64_05b: D=64, DM=896, GQA=2,
QSTAGE=14). The step-matrix probes (p_nfeed and the park/refusal probes)
fly on that image as-is. Note 25ddb66's warning: the afi named
apex-b64-05b-20260803 is MISLABELED (actually D=128) — build fresh, and
verify INFO_D on-card before trusting any run.

**UPDATE 2026-08-05: those AFIs now exist and have flown.**
`agfi-0183a4b88c8d21163` (apex-convergence-20260805: E-3b/E-4b/E-5 + DDR=1)
carried the self-running-card claim (`SELF_RUNNING_CARD_RESULT.md`), and
`agfi-0bc20880b50f5faba` (apex-convergence2-20260805: + E-6) carried the
walked-epilogue claim on silicon (`E6_ON_SILICON.md`).
