# E7_TOKEN_LOOP — the E-7 composition inside the token loop, per (step, layer)

**Status:** design (this doc) → implemented as `--mask e7` in
`scripts/fpga/f2/token_loop.py`. **Branch:** `feat/e7-token-loop` off
`comp/prompt-b-c`. **Sim only** (`verif/f2sim` twin); no AWS, no devbox.
The `--mask b` default (wfl.MASK_B, the flown 2026-08-11 path) is
UNTOUCHED — everything here is flag-gated.

## 1. What this extends, and from what lineage

`token_loop.py --engine sim-walked` today walks, per (decode step, layer),
the E-6 chain `wfl.MASK_B = {FPROJ, QKV, OPROJ, RES1}` and grades every
walked value bit-exact against golden. Attention (score/softmax/PV) and
both norms are HOST steps.

The E-7 chain — walked attention (SCORE+PV) + the fuel-fed OPROJ epilogue
+ RES1 + NFEED + NORM2 with the gamma FETCHED from card DRAM — is
silicon-proven STANDALONE: `build/e7_flight/walk_e7.regops.jsonl` flew
27/27 GREEN on both images (2026-08-11, D-033 closed the snoop-replica
defect). Its builder lineage is `elane_walk_qstage.py` (`e7_descriptor`,
`E7C_MASK`, the `g2_base` gamma record, `g3.store_kv_phase` KV staging)
+ `fly_e7ng.py` / `fly_e7_hw.py`. **That proof is at the toy fence:
d = 64, H = 1, T = 1, one static prefill step.** This design maps which
parts of the composition carry to the token loop's real geometry
(d_model = 896, H = 14, GQA n_kv = 2, live T per step) and which are
fenced, with receipts.

## 2. The geometry verdict at 0.5B — why `--mask e7` is TWO kicks

Three measured facts about today's RTL (all on this branch, ancestor
f967659):

**(a) The act-bank capacity bound.** `seq_layer_walker2.sv` stacks the
staged act families in template order (q rows, QKV, OPROJ, DOWN, y-drain
window) and refuses at S2_CHECK when `fp_top > STAGE_ROWS = 31`
(rtl/seq/seq_layer_walker2.sv:746-756, 1540). At H=14 every family is 14
rows, so:

| mask shape | rows | verdict |
|---|---|---|
| {QKV, SCORE, PV, FPROJ} | q 14 + x 14 = 28 | **LEGAL** (fence 5's own worked example, STEP_MATRIX.md) |
| + OPROJ (the one-kick E-8 shape) | 28 + 14 = 42 | **REFUSED** (e6_fence_qkvattn_cap, refuse2 case (p)) |
| {SCORE, PV, OPROJ, RES1, NFEED, NSRC, NORM2, FGAM, FPROJ} (walk_e7's mask) | q 14 + o-act 14 + y 14 = 42 | **REFUSED** (same bound) |
| {FPROJ, OPROJ, RES1} (E-6 chain A) | o-act 14 | LEGAL, **FLOWN** |

The one-kick E-7/E-8 composition does not fit the 31-row bank at H=14.

**(a2) NEW MEASURED FINDING (2026-08-11, this bring-up, obj_tokenloop
twin):** a `{FPROJ, QKV}` image with `W_STEP.t_rows > 1` WEDGES at QKV
job 123 of 144 (mid-Wk), busy stuck, no error sticky. Bisect: the same
program with `pack_step(1, 0)` runs all 144 jobs; zeroing RQ[0..13]
changes nothing. The fueled projection template is T-aware (the K/V
projections of a T-row context are T-row jobs) and starves on an act
family no 31-row bank can hold — S2_CHECK accepts the image and the walk
wedges rather than refusing. Fence-5's "28 rows FITS" is a t_rows=1
statement. Consequence: **the live T can never ride the fueled projection
image**; attention walks in its OWN kick at the live T, and the
projection kick keeps the FLOWN T-free MASK_B image verbatim.

The split therefore is:

```
kick ATT  MASK_E7A = {SCORE, PV} @ W_STEP=(T, step)   q rows 14   (the B1/qmh
          shape at the live T — no fuel, nothing fetches)
kick B    wfl.MASK_B {FPROJ, QKV, OPROJ, RES1}        28 rows     (the FLOWN
          per-layer program, BYTE-PATH-IDENTICAL to --mask b: same builder,
          same T-free descriptor shape, same grades)
```

**(b) The fetched-gamma norm at 896 is value-blocked (B-FEED-WIDTH).**
The E-7 machinery (FGAM route, walker y-drain) is real and S2_CHECK-legal
at this geometry (`NF_ROWS_EXACT` holds — FEED_DM == CFG_D == 64 in every
buildable image; 14 y rows, fp_top 14+14 = 28 ≤ 31 on a {FPROJ, OPROJ, RES1} kick). But
the VALUE would not be golden's: NFEED runs r1 through the C-1 feeder,
which frames the 896-wide row as 14 × 64 frames each with its OWN scale
(seq_layer_walker2.sv:984-993; STEP_MATRIX.md blocker 5;
walk_fuel_layer.py header), while golden's norm2 quantizes the whole row
with ONE scale (`quant_rows_i8(r1)` → `rmsnorm_fx`,
transformer.py:568-570). RMS-norm is invariant to a UNIFORM scale — the
per-frame scales are exactly what breaks the equality. The toy proof
carried because at d = 64 there is one frame. A walked NORM2 at 896 would
compute a value no golden function vouches for, so under this repo's
bit-exact discipline it is **fenced, not degraded**: `--mask e7` keeps
NORM2 (and the FGAM fetch, which S2_CHECK ties to a NORM consumer) OUT of
the walked set, loudly. Closure shape (named, not started): the
wide-feeder split — a single-scale 896-wide C-1 framing (or a scale-aware
norm input) — after which FGAM+NORM2 ride a {FPROJ, OPROJ, RES1,
NFEED, NSRC, NORM2, FGAM} kick within capacity (o-act 14 + y 14 = 28).

**(c) NORM1 / FFN / SWIGLU / 0.5B-DOWN / RES2** keep their documented
fences (STEP_MATRIX.md); none is touched here.

**The honest claim of `--mask e7`:** per (step, layer), SIX template step
families run WALKED and bit-exact-graded — QKV, SCORE, PV, OPROJ
(+epilogue), RES1 under two fmt=1 kicks — extending the flown MASK_B set
by walked attention at the LIVE context length, with the projection kick
byte-path-identical to the flown program. The fetched-gamma norm is the
named follow-on, blocked by (b); attention-in-the-fueled-image is blocked
by (a2) — both fenced, neither degraded.

## 3. Descriptor re-point map, per (step, layer)

Both kicks use `fmt` (seq_walker_fmt.py) word packing at the 0.5B
geometry: `pack_geom0(64)`, `pack_model0(896, 0)`,
`pack_model1(14, 2, kv_map=1)`.

**ATT kick (`MASK_E7A = 0x30` = {SCORE, PV}):**

| word(s) | content | re-points per |
|---|---|---|
| W_GEOM0/W_MODEL0/W_MODEL1/W_MASK | geometry + mask | never |
| W_STEP (21) | `pack_step(T, q_pos)` = `(step+1, step)` — the S8 self-inclusive composition (Session.step: T = t+1 rows including the dup, m_q = t) | **per STEP** |
| W_TENS0+{WQ,WK,WV,WO} (11..14) | layer li's resident region bases — written for image uniformity; a no-FPROJ mask never reads them | per LAYER |
| W_RQ0+h, h=0..13 (22..35) | `pack_rq(*core_walk[h].rq)` — the per-head PV requant pair, golden `calib_requant` of the very acc_o it configures, from the **D-030 recomposition** (§6). HOST-LOADED (the B1/D-028 fenced scope, disclosed: causally circular on-tile) | per (STEP, LAYER) |

**Projection kick (`wfl.MASK_B = 0x40C2`):** the flown per-layer image,
BYTE-PATH-IDENTICAL to `--mask b` — `wfl.walk_descriptor(bases,
mask=MASK_B, rq=(ge.scale, ge.shift), comp=ge.comp)`, built by the same
`_emit_program`, drained and graded by the same code. `W_STEP` stays
`pack_step(1, 0)` — the (a2) finding makes this a REQUIREMENT, not a
convenience. Re-point set per (step, layer) = `wl.REPOINT_WORDS`
verbatim: the 4 tensor bases + RQ[14] (o-proj slot) + JC_OPROJ.

**Drift fences (selftest-checked, engine-enforced):** consecutive ATT
descriptors within one step may move only
{TENS WQ/WK/WV/WO} ∪ {RQ0..RQ13}; across a step boundary additionally
W_STEP. Consecutive MASK_B descriptors may move only `wl.REPOINT_WORDS`.
Anything else REFUSES the run (the walk_layers_05b claim, extended).

## 4. KV staging per step within the CQ-8 window (the hard part)

The E-7 flight staged KV statically: one `g3.store_kv_phase(s, K16, V16)`
call, T = 1, one engine. The token loop must stage the LIVE history at
every (step, layer). The mechanics, from the l3/qmh lineage
(gen_l3_vectors.py:509-553, gen_qmh_case.py:181-200):

* **Operands are golden's own.** At (step, layer li), for KV head
  g ∈ {0, 1}: `K16_g = fx.heads[7g].K_f16` (= f16 of the ROTATED K rows,
  [T, 64]) and `V16_g = fx.heads[7g].V_f16` (unrotated V, [T, 64]) — the
  exact arrays golden's `decoder_layer_fx` handed to `attention_core`
  (transformer.py:535-541; query head h reads KV group h//7). The engine
  asserts all 7 heads of a group carry identical K_f16/V_f16 before
  staging.
* **Addressing is T-relative.** Per engine: K row t → KVQ record addr t,
  V row t → addr **T + t**, occupancy check 2T
  (gen_l3_vectors.py:544-553). When T grows by one, every V record's
  address moves — so KV is **fully re-staged per (step, layer) program**.
  This is not an economy loss: each program runs from TILE_RST (per-file
  reset is the executor contract on sim AND hw), so nothing persists to
  reuse anyway.
* **Engine select.** `l_kv_map` = LAYER_CTRL[17:15], host-writable while
  the KVQ subsystem is idle (apex_top.sv:869-877): write kv_map = g, then
  `store_kv_phase` for engine g (its own per-record idle polls satisfy the
  quasi-static rule). During the walk the WALKER owns the select
  (wk_kv_eng_sel), mapping head h → engine floor(h·2/14) = h//7.
* **Scales ride the store.** apex_top snoops the KVQ WRITE_ADDR stream
  into the per-engine composite caches (seq_walker_comp); at CQ-8 the
  stored record scale is bit-identical to the feeder's read-time recompute
  — so re-staging refreshes the walker's CS/QS inputs by construction, and
  a stale s_q is a LOUD err_stale, not a wrong answer.
* **The CQ-8 window.** T ≤ 8 is the B1 fenced scope and exactly
  `token_loop.T_SIM_MAX = 8` (prompt + generated ≤ 8) — KVQ_DEPTH holds
  2T with room; TIP at T ≤ 8 is a single importance block, so each head's
  decision beat is structurally block 0 (drained/checked as the toy did).
  The builder REFUSES T > 8.
* **The 2-record delta is disclosed, not walked.** EN_STOREKV (walked
  WRITE_ADDR issue for the decode token's own K/V records) is proven
  standalone but NOT composed here — the host stages all 2T records.
  Composing it saves 2 record-stores/step/engine and is follow-on work.

**q rows.** Per head h: the PRE-rope staged slice `_f16(fx.q_real[sl_h])`
(ONE NP-r narrowing — see the D-030 note in §6) is injected through the
gap-A sink with **ONE arming for all 14 rows** (the arming edge resets the
glue's s_q file pointer — gen_qmh_case.py:216-230): route rdst=1/asrc=0,
`lctl(rope_en=1, rope_bank=1, rope_pos=step, fsrc_ext=3)` once, then per
row: feeder job, MODE_F16 inject, fs cap (= the staged row's C-1 scale =
`core_walk[h].s_q`), act-LOAD → **bank 1 row h**. The phase-q table
(bank 1) is loaded ONCE per program at `rope_phase_q(step, theta)` — all
heads share the position. The tile rotates during staging, so the staged
codes are `quant(rope_fx(_f16(q_real_h), step))` — proven at exactly this
geometry by E-3b ("H=14 @ 0.5B: 896/896 + 14/14 scales", STEP_MATRIX.md)
and by the qmh per-head readback. Post-staging: per-engine occupancy
re-check (the sink must not leak into KVQ), then LAYER_CTRL = 0 — the
walker owns every level inside the walk. QSTAGE-under-FPROJ stays
structurally refused, so this staging is host verbs PRE-fuel-arm, as in
every flown E-6/E-7 program.

**Act rows.** The ATT kick stages ONLY the 14 q rows (bank-1 rows 0..13 —
fp stack with no other family). The MASK_B kick stages its two families
exactly as today (h8 at 0..13, attn8 at 14..27 — its own file, after
TILE_RST). No family ever shares a bank with another kick's.

## 5. Program shape and egress (per layer: two regops files, one executor invocation)

```
walk_t{s}_L{li}_att:  prologue → phase-q table(pos=step) → per-engine D-020
                      resets + LIVE K/V stores → 14 q rows through ONE sink
                      arming @ rows 0..13 (fs cap per row = s_q) → occupancy
                      re-check → levels 0 → FMT_SUP check → 64-word E7A image
                      → CYCMARK E7A-PREGO → WALK_GO
                      → in-window drain, production order, per head h = 0..13:
                            T × efs            (s_k ladder)
                            1 × ess            (s_c)
                            8 × [T × efs, 1 × ero]   (s_v ladder + o8 block)
                            1 × ETIP           (block 0 — single block at T ≤ 8)
                      → done-poll → CYCMARK E7A-WALKDONE → status/hygiene → final
walk_t{s}_L{li}_qkv:  _emit_program(sub, descB) — the FLOWN MASK_B program
                      VERBATIM (fuel-fed QKV+OPROJ+RES1; its own silence
                      census; 144-job RO drain; r1 erd 896)
```

Both files run in ONE `bridge.run_job([att, qkv])` invocation; per-file
TILE_RST gives each its reset semantics (the existing hw/flush_step
contract). Captures come back concatenated and are split by each file's
own baked cap count (the flush_step splitter idiom). The ATT file never
arms fuel (nothing fetches — mailbox staging needs no arm dance); the
MASK_B file splices its arm exactly as today. Measured on L0
(obj_tokenloop, tile_div=2): ATT window 97,690 tile cycles; MASK_B window
559,697 tile cycles — BYTE-IDENTICAL to the flown E-6 chain-B constant,
the "unchanged" claim as a number.

## 6. The grade (golden is the ONLY arbiter; the D-030 recomposition)

**The F5/D-030 erratum applies** (measured during bring-up, exactly as
`gen_qmh_case.py:98-146` documents): the session's golden is a BUS-OFF
composition — `fx.q_real` is NOT on the fp16 grid, and `fx.heads`' cores
rotated that UN-narrowed row. The tile can only rotate what it stages
(f16(q_real), MODE_F16's one RNE), so pinning the grade to `fx.heads`
would fail a CORRECT tile. The walk's arbiter is therefore
`core_walk[h] = attention_core(rope_fx(_f16(q_real_h), step), K16_g,
V16_g, TIER_CQ8, G)` — golden PUBLIC functions composed in the tile's own
order on the STAGED operands, the qmh D-030 idiom verbatim. K16/V16 need
no recomposition: `fx.heads`' K_f16/V_f16 ARE the staged record bits
(already narrowed once in golden; MODE_F16 is idempotent on grid values —
the qmh M2 property, asserted per row).

| walked stream | caps | graded against |
|---|---|---|
| QKV raw accs | 144 × 8 lanes | `wfp.grade` vs `gemm_i8_ksplit(h8, blocks)` — UNCHANGED from MASK_B |
| q staging scale | 14 fs (pre-walk) | `core_walk[h].s_q` |
| score key ladder | T fs per head | `core_walk[h].s_k` |
| softmax row scale | 1 ss per head | `core_walk[h].s_c` |
| PV value ladder | 8 × T fs per head | `core_walk[h].s_v` (each block re-reads the T records) |
| **attention o8** | 8 ero per head (64 codes) | **`core_walk[h].o8`** — attention_core on the same operands |
| TIP decision | 1 beat per head | block 0 (structural at T ≤ 8) |
| epilogue r1 | 896 lrd | `wfl.grade_r1` vs `golden_epilogue` — UNCHANGED |

Said plainly: the walked attention is a bit-exact PROBE on the staged
operands (the same claim class as the QKV probes on their c1-framed
rows), while the layer output the loop consumes remains golden's OWN
composition — the layer-output gate (`check_engine_bits`) still requires
the engine's r2 == golden's r2 bits before any token is sampled. Any
mismatch REFUSES the run (SystemExit), exactly as today.

Disclosures carried verbatim from the lineage: the o8 → OPROJ-act seam
remains a host-staged copy (`attn8 = c1_frames(fx.attn)` — golden's
composition of the very o8 values kick 1 graded; the in-tile link is the
one-kick shape capacity excludes); per-head RQ is host-loaded calibration
(descriptor slots, B1 §2); the QKV RO drain and the attention fs/ss/RO/TIP
drains are data-plane pops, zero CONTROL writes in the window
(silence_predicate, extended to the ATT drain set).

## 7. The walked-MAC share

Per layer, MACs the tile produced AND that graded bit-exact (shapes of
the graded tensors, the token_loop rule):

```
QKV      896 × (896+128+128)      = 1,032,192      (unchanged)
OPROJ    896 × 896               =   802,816      (unchanged)
score    14 heads × T × 64        =       896·T ┐  o8 grades the composition
PV       14 heads × 64 × T        =       896·T ┘  (acc_s → softmax → acc_o → requant)
────────────────────────────────────────────────
total    1,835,008 + 1,792·T      (T = step+1)
```

At the committed prompt (5 ids) + 1 generated token: T = 6 →
1,845,760 walked MACs/layer. The attention share is small in absolute
MACs — its value is the STEP-FAMILY coverage (6 of 13 walk vs 4) and the
live-T composition; the report says so and carries both census
denominators as today. The score/PV MACs are graded THROUGH the o8
composition (every acc_s/acc_o element feeds the graded o8 through
softmax + requant), the same transitive rule the B1 walker claim used.

## 8. Harness changes (`token_loop.py`), all flag-gated

* `run --mask {b,e7}` (default `b`). `--mask b` is byte-path-identical to
  today: same builders, same descriptors, same grades, same JSON.
* `SimWalkedEngine._subject` (e7): adds `heads` (golden AttnCore refs),
  `q16` (pre-rope q row bits), `K16`/`V16` per engine, `T`, `pos`. No new
  refusal class (MODE_F16 staging has no amax-127 constraint; c1 refusals
  unchanged).
* `_prep_one` (e7): builds BOTH descriptors; two drift fences as §3.
* `_emit_program_e7` (module-level, pool-safe like `_emit_program`):
  emits the ATT file (new builder, §5) and calls `_emit_program` for the
  MASK_B file (VERBATIM — same checks, same splice, same census); the ATT
  window gets its own census (advances on the RO/FS/SS/TD FIFOs are the
  disclosed data plane) and its cap count is pinned to the drain ledger.
* `_execute` (e7): `run_job([att, epi], …)` one invocation; `_grade_one`
  (e7): splits caps per file, runs the §6 grade table, then the unchanged
  QKV/r1 grades. New CYCMARKs (`E7A-…`) parsed alongside the E6R pair;
  the per-token prediction covers the walked steps only, labelled as
  today.
* `HwWalkedEngine` inherits the seams (batched flush splits by per-entry
  cap counts, already the contract). UNFLOWN for e7 — a card run
  additionally needs an AFI carrying the E-7 RTL; a pre-E-7 image refuses
  loudly at the program's own FMT_SUP/reserved-bit checks. `--ddr-attested`
  gating unchanged.
* selftest: mask arithmetic (§2 table — the split's row sums vs the
  31-row bound), descriptor slot/drift-set checks on synthetic layers and
  steps, drain-count arithmetic (fs/ss/ro/tip per head at T), the
  1,792·T MAC formula, and registry/default checks (`--mask b` untouched).

## 9. Done criteria (the gate)

`sim-walked --mask e7 --tokens 1`: every layer bit-exact INCLUDING
attention (o8 + full scale ladder vs fx.heads), token identity PASS vs
host-golden, MASK_B regression green (`--mask b` run + selftest), the
JSON record carrying the §7 split and the §6 disclosures.
