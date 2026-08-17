# E-LANE — a COMPLETE decoder layer, end to end, on our own silicon (toy width)

**Status:** E-1/E-2 LANDED IN RTL 2026-07-31, E-3a 2026-08-01 (see §4 update
notes; sim-proven bit-exact on the toy width, red/green vs the pre-change
twin — `scripts/fpga/f2/elane_norm_feed.py` host-driven,
`scripts/fpga/f2/elane_walk_norm.py` WALKER-driven). E-3a FLOWN 2026-08-02
(`agfi-006b1314fcbbb3505`, `ELANE_WALKED_CHAIN_RESULT.md`). E-3b CLOSED and
the first E-4 chain landed 2026-08-04; E-4b (walker-drivable `l_nsrc`, the
one-kick walk_e4) LANDED 2026-08-04 — see §4. E-5 (walked fuel-fed QKV)
2026-08-04 and E-6 (walked epilogue) 2026-08-05, both sim-proven then FLOWN
2026-08-05 (`agfi-0183a4b88c8d21163` / `agfi-0bc20880b50f5faba` —
`SELF_RUNNING_CARD_RESULT.md`, `E6_ON_SILICON.md`). ·
**Depends on:** nothing external — no wide image, no vendor
**Companion:** `MASTER_TABLE.md` (N-lane = the 7B op-type work; this is a
different, parallel claim)

---

## 1. What this lane claims, and what it does NOT

**Claim if it lands:** *"A complete transformer decoder layer — RMSNorm,
Q/K/V projections, RoPE, attention, output projection, residual, RMSNorm-2,
gate/up projections, SwiGLU, down projection, residual-2 — ran end to end on
our FPGA, every operation computed by our blocks, each op's output feeding
the next INSIDE the chip, bit-exact against the golden model. Host role:
load weights, start it, read the answer."*

**It does NOT claim** anything about Qwen2.5-7B. The subject is a
**synthetic 128-dimensional, single-head layer**, because that is the shape
today's RTL can elaborate (§2). This is an **architecture** claim, not a
model claim, and the two must never be blurred:

| | N-lane (7B) | E-lane (this) |
|---|---|---|
| subject | a REAL Qwen2.5-7B decode step | a SYNTHETIC 128-dim 1-head layer |
| coverage | op types proven individually | ALL ops, CHAINED |
| host between ops | yes — stages every activation | **no** |
| gated on vendor | yes (2 full-row ops, #799) | **no** |

Both are honest; neither substitutes for the other. The eventual "real model,
end to end" claim needs BOTH lanes plus the wide feeder (§5).

## 2. Why 128 and not 3584 (the elaboration wall)

`rtl/seam/seam_feeder_quant.sv:100-101` hard-`$error`s unless `D == 64 ||
D == 128`. That module is the C-1 activation quantizer every op hands off
through. Consequences, measured (I-B GAP D, re-verified 2026-07-31):

- At **D=128** the tile quantizes a whole row with ONE amax and ONE scale —
  *identical in form* to golden's `quant_rows_i8` over that row. Format
  parity is exact.
- At **D=3584** the same values must be framed as 28x128 rows with 28
  independent scales — NOT what golden does (B-FEED-WIDTH). Closing that is
  the "wide feeder" (IB_LAYER stage 6), weeks of work, and then it still
  needs a wide image, which is blocked on aws-fpga#799.

At 128, `head_dim == D_model == 128` is a *consistent* single-head layer, so
the historical GAP D overload ("one build implies head_dim == D_model") is
satisfied by construction rather than violated.

## 3. What is ALREADY DONE (audited 2026-07-31, current tree 9b993bf)

- **The walker already sequences a COMPLETE layer.** `verif/seq_walker`
  walks all 11 steps — NORM1, QKV, ROPE, STOREKV, per-head SCORE+PV, OPROJ,
  RES1, NORM2, gate/up, SWIGLU, DOWN, RES2 — with `mask = EN_ALL`, at three
  geometries including true 7B (H=28, hd=128, Hkv=4, d_ffn=18944), checked
  bit-exact in-order against the golden-gated spec, with mutation gates.
  **The hard conceptual work exists.** What it does NOT do: every consumer
  is stubbed (`tb_walker2_sb.sv`) — no datapath computes. This lane makes
  those emissions drive real hardware.
- **F1 CLOSED** — walker n-splits now sized by the array's own `MXE_N`
  (`seq_walker_pkg.sv:343-346`, `seq_layer_walker2.sv:457-465`), mutation-
  guarded (`Makefile` m13 signature-required).
- **GAP A CLOSED** — rotated-q sink exists (`apex_top.sv:1210` q_sink,
  feeder mux `:1642-1649`), multi-head s_q capture landed.
- **GAP B CLOSED as RTL** — `apex_proj_bias.sv` exists; NOTE
  `PROJ_BIAS_EN = 1'b0` default (`apex_top.sv:175`), so a bias-carrying
  build is a parameter flip. A synthetic layer may simply be defined
  bias-free; if it models Qwen-style biased q/k/v, flip the parameter.
- **GAP C CLOSED for the q path** — MODE_F16 narrow -> rope -> C-1 quant is
  exactly golden's order (`apex_top.sv:1191-1204`); other activation paths
  audited and consistent.
- **R1/R2/R3** all landed and silicon-bound in the narrow image built today.

## 4. What is MISSING — the whole lane, four items

The layer's chain is already tile-internal everywhere EXCEPT one seam, which
it crosses twice (layer entry `X -> NORM1`, and mid-layer `r1 -> NORM2`).

### E-1 — residual row has no internal exit — **CLOSED 2026-07-31**
`apex_residual`'s row RAM holds the running activation (r1 then r2). Its
only reader WAS the host register path: `rd_data -> layer_rdata_q`.
**Landed:** an egress job port (`ej_*`, R3 window-base addressing, geometry
refused at accept in the R3 frame class) + fp16 stream (`ev/er/edata`)
on `apex_residual.sv`, driving the C-1 feeder as `l_fsrc_ext` code 4
(`l_fsrc_ext[2]` = LAYER_CTRL[7], reserved-0 before). Same exact-widen
argument as the q sink: the feeder's C-1 over the resident row ==
golden `quant_rows_i8`.

### E-2 — RMSNorm cannot be fed from inside, and has no job port —
**CLOSED 2026-07-31**
`asu_rmsnorm`'s input WAS a top-level port of apex_top (`xa_valid`/`xa_x`),
structurally host-fed. **Landed:** (a) `l_nsrc` (LAYER_CTRL[19]) muxes the
norm's x port onto the feeder's codes, unpacked lane8 -> serial by the new
`rtl/top/glue/apex_lane8_unpack.sv` (asu_rmsnorm itself UNTOUCHED — the mux
is apex_top glue on the instance ports); (b) LAYER_JOB **unit 3** is the
NORM/EGRESS job (cols = whole C-1 rows, window base at [15:14]) — pushed by
the host CSR path or the walker's LU channel through the same single
ingress; completion observables are LAYER_STATUS[5] (`nf_busy`, the whole
feed path) then `dn_rms` (the walker's existing lj_* tie works unchanged).
A push with the route unarmed refuses loudly (NEW err_code 9, NFEED_ROUTE).
Demonstrator + red/green + discriminators:
`scripts/fpga/f2/elane_norm_feed.py` (nfeed bit-exact vs golden at
tile_div 5 AND 2; all three stages FAIL on the pre-change twin).
NOTE the walker's own level net is still 2-bit — a WALKED program cannot
arm code 4 yet; that is E-3a. **(RETIRED 2026-08-01 by E-3a below.)**

### E-3a — the chain could only be armed BY THE HOST — **CLOSED 2026-08-01**
E-1/E-2 built the in-tile route but left it host-CSR-only, so **the layer
sequencer had never driven this datapath**. Two independent causes, both in
the walk-mode path: `seq_layer_walker2`'s `lw_fsrc_ext` was **2 bits** and
`apex_top` ZERO-EXTENDED it (`{1'b0, wk_lw_fsrc_ext}`), so a walked program
could not name source code 4 at all; and no walker step pushed the unit-3
job. **Landed:**
- the level is 3 bits end to end — `seq_walker_pkg::walk2_lctl` carries
  `fsrc_ext[2]` at LAYER_CTRL's own bit 7 (the slot the old `1'b0` filler
  occupied, so every legacy level word is byte-identical),
  `seq_layer_walker2.lw_fsrc_ext` is `[2:0]`, and the apex_top mux copies it
  whole;
- **PC_NFEED**, one new pc at the r1 -> NORM2 seam (before the gamma-2
  fetch), gated by the previously reserved mask bit `W2_EN_NFEED` (bit 11 —
  `EN_ALL` is unchanged, so legacy images walk byte-identically). It issues
  the host demonstrator's exact verbs in the same order: arm code 4 (that
  field ALONE — the surrounding LVL steps are not re-armed), push the C-1
  feeder job (`d_model / FEED_DM` rows, division-free because
  `walk_desc2_check` already pins `d_model == n_heads * head_dim` and
  `head_dim == CFG_D`), push LAYER_JOB unit 3 UNCHUNKED, then hold on
  `nf_busy` — the SAME LAYER_STATUS[5] term the host polls, fed back to the
  walker as a new input.
- **New loud failure mode:** the NFEED envelope fence (`S2_CHECK`,
  WALK_ERR_DESC). Downstream refusals of the feeder job / unit-3 push land
  AFTER the walker has taken the push handshake and committed to waiting on
  `nf_busy` — i.e. a silent wedge — so the descriptor is refused before any
  state changes. Reachable at the unit level (`refuse2.ops` case (d), H=17
  vs the TB's `FEED_ROWS`=16); NOT reachable on the F2 CL, where
  `FEED_ROWS_MAX`=31 exceeds `WALK2_H_MAX`=30, and that is stated rather
  than probed.
Demonstrator + red/green + discriminators:
`scripts/fpga/f2/elane_walk_norm.py` (walked, `walk_en=1`; bit-exact vs
golden at tile_div 5 AND 2; the SAME final regops ABORT on the pre-change
twin at the route-arm poll). Host role while the walker drives: load X and
gamma, produce r1 (RES1 is not in this fence), arm `l_nsrc`, load the
descriptor, `WALK_GO`, and drain the norm's OUTPUT.

### E-3b — q rows are host-staged before the walk starts — **CLOSED 2026-08-04**
The fmt=1 walker never drove `fsrc_ext=3`; the H rotated-q rows were staged
by the host **before** `WALK_GO`. **Landed:** mask bit `W2_EN_QSTAGE`
(W2_MASK[12], reserved-0 before — the E-3a discipline one bit out; the
mask field grows 12 -> 13 bits, pkg + Python mirror in one commit) arms
**PC_QSTAGE**, which performs the HOST's exact staging choreography per
head (`build_rope_stage` + `inject_jobs` order): arm {rope-q, q_sink} +
the production route override {rdst=1, wsrc=0, asrc=0, fdst=0, qsrc=0}
(`rt_f1_q` — tracks the wrapped engine outside QSTAGE, so every legacy
walk is byte-identical), then FJOB, MODE_F16 QJOB, the ONE per-step S-2
composite (`W2_QC`, word 58 — repeated D times; per-tensor-scale model,
the W2_WSCALE0 shape) , the SERIALIZER job (new walker `sj_*` channel;
apex_top's u_ser job port joins the walk-mode mux), D/8 x {loader-row act
EMIT, k2 WS DESC}, the act-bank-1 LOAD at row h, a drain wait, and a
touch-one-field `fsrc_ext` restore. Fences at S2_CHECK (§A-1): QSTAGE+QKV
(double production), n_heads > Q_ROWS (the tile's QSTAGE_H_MAX), split-
feeder builds. Suite: qstage cases at both geometries + 2 refusals +
mutant m14_qsrestore. Tile demonstrator
`scripts/fpga/f2/elane_walk_qstage.py`: H=1 64/64 codes + scale bit-exact
vs golden (rope_phase_q/rope_fx/quant_rows_i8), **H=14 at the 0.5B
geometry (d_model=896, GQA 14:2 image): 896/896 codes + 14/14 scales**;
walk-off RED; pre-E-3b twin RED (reserved-bit refusal). Host role during
the walk: streaming the k2 xw DATA beats (paced by the FIFO-free poll
against the walker's own consumption) — data movement, disclosed.

### E-4 — the subject and the arbiter — **FIRST CHAIN LANDED 2026-08-04**
`walk_e4` (same demonstrator, b64 twin, toy 64/H=1/T=1): **one fmt=1
image, two kicks** — kick 1 mask {QSTAGE, SCORE, PV}: the sequencer stages
q and walks the whole attention ON the staged row (composites from the
store-snooped caches, hd interlock armed by the WALKER's own q_sink edge,
PV o8 requantised on-tile from RQ[0]); kick 2, after the ONE host verb
between kicks (arm `l_nsrc` + patch the MASK word), mask {NFEED}: X ->
C-1 -> RMSNorm in-tile. Graded vs golden only: **o8 64/64 BIT-EXACT**
(`attention_core`), h2 codes 64/64, the full 11-beat fs scale ladder +
s_c. Discriminators: walk-off RED, a golden-searched V-record flip RED at
o8, pre-E-3b twin RED.
**MEASURED FINDING (why two kicks):** `l_nsrc` is a HOST-held level that
steers the one feeder output for a whole walk — QSTAGE needs codes -> act
stage, NFEED needs codes -> norm. With both in one kick the staged row
detoured into the norm and the walk wedged at the staging drain wait
(observed: LAYER_STATUS[5] stuck, attention already complete). A
walker-drivable `l_nsrc` is the named follow-on delta that fuses the
kicks. Also measured: MODE_F16 emits no s-tap beat; the walked score's
TIP decision beat must be drained (ETIP).

### E-4b — walker-drivable `l_nsrc` — **LANDED 2026-08-04, ONE KICK**
The follow-on delta above, done the E-3a/E-3b house way. The
previously-reserved mask bit `W2_EN_NSRC` (W2_MASK[13]) makes the WALKER
the owner of `l_nsrc` for that walk: `lw_nsrc` (value) + `lw_nsrc_own`
(walk-scoped ownership, high from the descriptor-check pass to walk end)
are new walker levels, apex_top's l_nsrc register TRACKS `lw_nsrc` only
while ownership is high (else the landed host-held HOLD, byte-identical),
the walker level word grows 15 -> 20 bits with nsrc at the REGISTER's own
position (LAYER_CTRL[19]; `seq_walker_pkg::walk2_lctl` /
`seq_walker_fmt.lctl`, single-source pair), S2_NFL arms nsrc=1 WITH the
code-4 flip, and S2_QSL + every S2_LVL case re-arm it 0 absolutely.
S2_CHECK fences (§A-1): NSRC requires NFEED; QSTAGE+NFEED requires NSRC
(the measured wedge shape is now a loud WALK_ERR_DESC refusal, proven on
the tile by `e4_fence_nonsrc` / `e4_fence_nsrconly`).
`walk_e4` is therefore **ONE kick** — mask {QSTAGE, SCORE, PV, NFEED,
NSRC}, the host writing NOTHING between walker start and completion (the
paced xw DATA stream disclosed; program-level predicate in the selftest) —
and every prior grade holds on the rebuilt twin (`obj_b64_05b_e4b`):
o8 64/64 BIT-EXACT, h2 64/64, the 11-beat fs ladder + s_c, the post-walk
LAYER_CTRL word now CHECKED with bit 19 set by the SEQUENCER; walk-off
RED, V-flip RED at o8, and the single-kick image RED on the pre-E-4b twin
(`--redproof` on obj_b64_05b_e3b: reserved-bit refusal, zero caps).
E-3b's H=14 0.5B-geometry staging grade replayed green on the new twin
(896/896 codes + 14/14 scales).
The full-layer walk remains bounded by the step matrix
(`scripts/fpga/f2/elane_walk_steps.py`, measured 2026-08-04): every
weight-consuming step (QKV/OPROJ/FFN/DOWN) parks at S2_FETCH (fuel mode
only; no abort path out), and even with fuel the walk-mode routes pin
rt_wgt_src=1 / never rdst=1, so walked GEMMs cannot reach xw weights nor
the serializer-fed units without the route-override treatment PC_QSTAGE
now demonstrates.

### E-6 — THE WALKED EPILOGUE — **LANDED 2026-08-05 (sim-proven)**
E-5 fenced OUT the o8 requant epilogue (`pc_hasrq` — OPROJ/DOWN): its
fuel-fed walks egressed RAW INT32 on the RO lanes and the mask was pinned
to exactly {FPROJ, QKV}. E-6 un-fences it for OPROJ, the E-5 house way —
three deltas in `seq_layer_walker2`, all gated on the dispatched pc's own
`pc_hasrq` so every E-5/legacy walk is byte-identical:
  (i)   **S2_PSJ** — the walker pushes the serializer job framing the
        projection's whole o8 stream (n_splits beats x 8 = the deq job's
        cols) BEFORE the first act EMIT; with PC_URES1's residual job and
        PC_UOPRJ's JOBC+deq job already pushed in template order, the
        tile's ser -> apex_layer_deq -> apex_residual chain is armed
        before any result beat exists — consumer-before-producer, one
        level deeper.
  (ii)  **RT_FPRQ (8'h94)** — RT_FPROJ with EXACTLY ONE field moved:
        rdst 0 -> 1, the measured host-mode o8 leg's own destination
        (gen_layer_ops.py:795). The requantised o8 posts to the
        serializer, never the RO lanes — so the claim window needs NO
        host drain at all.
  (iii) the OPROJ act family emits from its own bank-1 row window
        (`fp_oproj_base` — above the staged-q rows / the QKV family).
Fences (S2_CHECK, §A-1 — each measured): OPROJ<->RES1 paired (either
alone is the starvation class); NORM1/NORM2/FFN/RES2 refused (gamma
fetches would poison the ARMED fuel stream — no xw consumer; the
all-gate-then-all-up template order starves asu_swiglu's per-64-frame
phase alternation, which keeps DOWN out with it); **QSTAGE refused under
FPROJ outright** (fuel_src=1 disconnects the mailbox xw stream its k2
injection rides, cl_apex.sv:1074 — the wedge is now a loud refusal);
QKV excludes SCORE/PV (act-bank row collision); one k-split only
(d_model <= walk2_k_job(FEED_DM), the W1 third erratum's bound); o8
frame <= 255 serializer beats.
**Proven, golden the only arbiter** (obj_e6_b64_ddr1, DDR=1 b64 twin,
real 0.5B tensors from the resident image):
  * `walk_fuel_proj.py rune6` CLAIM A — mask {OPROJ, RES1, FPROJ}: the
    REAL 896x896 L00_Wo fetched by the walker's own record, requant
    epilogue ON-TILE (RQ[14]), o8' -> deq (JC comp) -> residual + the
    real X row: **r1 896/896 BIT-EXACT** vs golden's
    f16(X + o8*comp) composition; **the walk window contains ZERO host
    operations** (not even a data-plane drain — checked on the emitted
    artefact); walk window 2,448,960 sim cycles.
  * CLAIM B — mask {QKV, OPROJ, RES1, FPROJ}: **TWO fuel-fed projections
    under ONE descriptor, 256 MXE jobs** — all 144 QKV blocks bit-exact
    (RO-drained, the E-5 disclosed data plane) AND r1 896/896 bit-exact;
    5,596,970 sim cycles.
  * `elane_walk_qstage.py --chain` walk_e6 — ONE kick, mask {SCORE, PV,
    OPROJ, RES1, NFEED, NSRC, FPROJ} at the toy fence: walked attention
    (o8 on-tile requant) -> fuel-fed OPROJ + epilogue -> r1 in-tile ->
    NFEED feeds THE EPILOGUE'S OWN PRODUCT to the norm. q host-PRE-staged
    (the l3 walker-mode idiom; QSTAGE cannot share a fuel kick — fenced).
  * Discriminators: walk-off (FPROJ cleared) RED; one poisoned resident
    Wo byte moves r1[0] by the exactly-predicted f16 delta; the V-record
    flip moves the walked o8 and NOT r1 — the o8 -> OPROJ-act seam is a
    host-staged copy and the discriminator MEASURES that (disclosed
    honestly: the in-tile o8->act re-staging is follow-on work);
    3 + 1 fence refusals proven on the tile.
DOWN's epilogue is the same pc class and stays refused with FFN; its
k = d_ffn decomposition is now LEGAL per the W1 third erratum
(walk2_k_job), but a walked DOWN needs the per-k-chunk act re-staging
S2_PAE does not carry (S2_PAD note) AND a walkable swiglu producer —
named follow-on, not faked.

### E-7 — THE FETCHED GAMMA — **LANDED 2026-08-05 (sim-proven)**
The biggest remaining fence falls: NORM1/NORM2 were refused under FPROJ
because NO xw -> xg ROUTE EXISTED (a fetched gamma dead-ended in the fuel
FIFO — stream poisoning). Closed the house way, mask bit `W2_EN_FGAM`
(W2_MASK[15], reserved-0 before):
  (i)   `rtl/top/glue/apex_gam_unpack.sv` (NEW) + the gam_win_q window in
        apex_top: the walker's `lw_gsrc` level steers xw beats into a
        4-per-beat int16 LE unpacker feeding asu_rmsnorm's g port; every
        legacy walk reduces all four xw/g nets to their exact old forms.
  (ii)  the walker opens the window at a G1/G2 K_FETCH dispatch and — the
        measured necessity — arms the norm's OUTPUT drain ITSELF
        (S2_GDL/GDA/GDF: levels + RT_YDRAIN 0x80 (the host's own measured
        drain route) + act-LOAD at the fp_top row window + the C-1 feeder
        job): a fetched-gamma norm EMITS mid-walk, and dn_rms requires
        every y beat ACCEPTED — without the walker-armed drain the
        S2_NORM wait is a wedge. S2_GDW holds the route until tile_idle
        (the S2_PJW tail rule) and closes the window.
  (iii) fences (S2_CHECK): FGAM requires FPROJ + exactly ONE named NORM
        step; the y-drain envelope (n_heads <= FEED_ROWS, fp_top +
        n_heads <= STAGE_ROWS, NF_ROWS_EXACT); NORMx under FPROJ without
        FGAM keeps the E-6 poisoning refusal VERBATIM.
Proven: `elane_walk_qstage.py --chain7` walk_e7 — ONE kick, {SCORE, PV,
OPROJ, RES1, NFEED, NSRC, NORM2, FGAM, FPROJ}: the epilogue's own r1 is
NFEED-fed to the norm, the REAL resident L00_g2 is fetched from card DRAM
by the walker's record, and the norm's output lands in act bank row 2
through the walker-armed drain — every grade bit-exact, host-SILENT
window, 34,080 cycles. Gamma poison: one resident g2 byte -> h2 moves by
the golden-predicted codes AND r1 holds. Pre-change twin RED
(reserved-bit refusal). NORM1 rides the same machinery but is UNFLOWN
(its x is a host xa stream; no NFEED equivalent at the layer entry) —
stated, not claimed.

### E-7b — THE SEPARATED DOWN — **LANDED 2026-08-05 (sim-proven)**
`W2_EN_DOWN` (W2_MASK[16]) un-ties the DOWN pcs from W2_EN_FFN, paired
with RES2. `walk_fuel_proj.py rund7`: the DOWN requant epilogue (E-6's
pc_hasrq class — RQ[H+1], JC_DOWN, S2_PSJ frame, RT_FPRQ) WALKED at
k = d_ffn = 1984 (= walk2_k_job(64)) x n = the real d_model = 896 over
the REAL resident Wd region's leading bytes: **r2 = f16(r1 + down8'*comp)
896/896 BIT-EXACT**, STRICT silence, 5,419,200 sim cycles, Wd poison RED
by the predicted delta, 4 fences proven on the twin — including
**d7_fence_05b: the full 0.5B d_ffn = 4864 stays REFUSED with its
measured reason** (76 stage rows vs the 31-row bank; S2_PAE has no
in-tile re-staging source — per-k-chunk re-staging is real datapath
work, fenced not faked). The act family is host-staged (the E-6 o8->act
seam class); its in-tile producer is fence 2's FFN interleave.

### E-8 — QKV JOINS THE ONE KICK (fence 5 closed) — **LANDED 2026-08-05**
The E-6 "QKV excludes SCORE/PV" fence was a row ASSIGNMENT: fp_base now
stacks every act family in template order (q rows, QKV, OPROJ, DOWN,
y-drain), so composition legality is just the fp_top capacity bound (at
0.5B H=14: QKV+attention = 28 rows LEGAL; +OPROJ = 42 > 31 REFUSED —
rune6's e6_fence_qkvattn_cap + unit refuse2 (p) hold the negative).
`--chain8` walk_e8 — ONE kick, **SIX template steps** {QKV, SCORE, PV,
OPROJ, RES1, NORM2(+NFEED/NSRC/FGAM)}: the 24 raw QKV projections
(real Wq/Wk/Wv prefixes, fuel-fetched, RO-drained — disclosed) graded
bit-exact alongside the whole walk_e7 chain, 72,410 cycles.

**Still fenced, with measured shapes** (each re-verified by execution
2026-08-05 before this session changed anything): FFN gate/up + SWIGLU
(the per-64-frame gate/up interleave + a re-laid Wg/Wu image — an ORDER
redesign, asu_swiglu.sv:106 / apex_top.sv:1547-1555 vs PC_WGJ < PC_WUJ);
QSTAGE-under-FPROJ (STRUCTURAL: the k2 beats are data-dependent,
gen_l3_vectors.py:495-505, and ride the mailbox xw stream fuel_src=1
disconnects, cl_apex.sv:1074 — no second xw ingress exists); DOWN at the
full 0.5B d_ffn (the re-staging datapath above).

## 5. Order, and honest cost

E-1, E-2 and E-3a are the lane (E-3b is small at H=1; E-4 is host-side).
All three are localized: one new stream port, one input mux, one job port,
one widened level field, one walker ROM step. Every step must clear this
repo's bar — red/green evidence, mutation gates, bit-exact vs golden —
which is most of the time, not the RTL itself.

Then: one narrow image (today's build proves narrow ingests, ~1 h build +
ingestion), and a card session.

**Explicitly NOT in this lane:** the wide feeder, DDR-resident weights,
multi-layer, multi-token throughput. Those are the "real model, fast"
track and remain gated on aws-fpga#799 plus IB_LAYER stage 6.
**[Update 2026-08-05: DDR-resident weights are no longer a fence — IB-FUEL
s4b/s5 + E-5/E-6 landed and flew them (0.5B, D=64 geometry). The wide
feeder (B-FEED-WIDTH at D_model=3584) and the wide image (#799) remain the
open items for the real-7B fast track.]**
