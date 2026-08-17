# B1 stage-1 design notes (working reference, not a contract)

Generated 2026-07-21 by the stage-1 design panel. The **contract** is
`B1_WALKER.md`; this file is the implementation-level research behind it —
kept so the RTL can be written without re-deriving it.

Decisions already promoted into the contract: §A-1 (CQ-8-only tier scope),
§A-2 (scale source = Option B snoop + `cq_fp_pkg::scale_from_amax`), §B-1
(tap mirror budgeted). Everything below is *input to the RTL*, and like any
research note it is a hypothesis until the RTL and its TB agree with the
golden — treat line cites as needing a re-check before you rely on one.

---


## 1. The FSM ROM — exact score/pv op sequence at CQ-8

# Walker v1 ROM — exact op sequence for `score_phase` + `pv_phase` at CQ-8

All of this is **verified by re-derivation, not by reading**: I wrote an independent emitter from the table below and diffed it line-for-line against the 24 CQ-8 case files in `verif/top/l3/build/cases/`. **24/24 exact match, 0 mismatches**, covering D∈{64,128}, T∈{1,2,3,4,5,6,8,9,10,12,16,17,20,24,28,30,70,100,128} — including the `nbT>BPR` split cases. The cross-check script was scratch-only (not a repo artifact); the result it produced is the 24/24 match recorded here.

Derived quantities (all pure in `(D,T)`; source `verif/top/l3/gen_l3_vectors.py:199,185-191,412`):
```
BPR   = D/8                       # beats per stage-buffer row  (:199)
nbT   = ceil(T/8)                 # c8 row length in lane8 beats (:412)
NCH   = ceil(T/8)                 # number of column chunks      (:185-191)
chunk ci: c0(ci) = 8*ci,  nc(ci) = min(8, T - 8*ci)     # last chunk short
split = (nbT > BPR)               # i.e. T > D  →  c8 spans 2 stage rows
```

---

## 0. Precondition the walker inherits (not emitted by these two phases)

`gen_l3_vectors.py:575` — `s.aj(0, 1, 0, 1, BPR, 0)`, the LOAD of the q8 row into **act-stage bank1 row0**, is the last op of q-inject. `score_phase` re-EMITs that exact row every chunk and never reloads it. If the walker's scope starts at score (B1_WALKER.md §5 stage 1: "Scope = score+pv only"), bank1/row0 = q8 is an entry invariant.

---

## 1. SCORE PHASE ROM (`gen_l3_vectors.py:360-401`)

| # | op | args (formula) | notes / ordering |
|---|---|---|---|
| S0 | `CSRP` | `addr=0x04, mask=0x1, exp=0x1` | **APEX CSR** `STATUS.idle`. Emitted by `route(poll=True)` (`:233-234`). Gate for the level change in S1. |
| S1 | `ROUTE` | `0x00EF` (constant) | `fsrc=1,fdst=1,asrc=1,wsrc=1,rdst=2,qsrc=1,kvu=1,blk=0` per `:235-237,371`. Bit map: `[0]feeder_src [1]feeder_dst [2]act_src [3]wgt_src [5:4]res_dst [6]squant_src [7]kv_user [14:8]tip_blk`. |
| S2 | `SJOB` | `cols = T` | `:374`. Arms score-dequant for the full T-column row. **Must precede the CS words.** |
| S3 | `CS` ×T | `t=0..T-1`: `cs[t] = f32bits( f64(s_q) · f64(s_k[t]) · (2^SCORE_FRAC / sqrt(D)) )`, **one** f32 narrowing (`:375-380`) | **All T words, contiguously, before anything else.** This is the composite unit's first output stream. |
| S4 | `QJOB` | `mode=1, cols = T` | `:381`. S-4 P-requant fold job. |
| S5 | `QS` ×T | `t=0..T-1`: `qs[t] = f32bits_exact( f64(s_v[t]) · 2^-15 )` (`:382-384`, `f32_bits_exact` `:133-140`) | **All T words, contiguously, immediately after S4 and before the first chunk.** |
| — | *(per chunk `ci = 0 .. NCH-1`)* | | |
| S6 | `WJ` | `op=0(LOAD), bank=0, pat=0, rows=nc(ci), nb=BPR, sel=0` | `:389`. Loads the k8 chunk into wgt-stage bank0 rows `0..nc-1`, `BPR` beats each. |
| S7 | `FJOB` | `rows = nc(ci)` | `:390`. Arms the feeder for `nc` K records. |
| S8 | `KVP` | `addr=0x04, mask=0x1, exp=0x1` | **KVQ AXI-Lite** `STATUS[0]=idle` (`rtl/kvq/kvq_engine.sv:950-953`). One per token. |
| S9 | `KVW` | `addr=0x2C (RADDR), data = t`, for `t = c0(ci) .. c0(ci)+nc-1` | `:393`. **K record index = t.** Writing `REG_READ_ADDR` *launches* the read (`kvq_engine.sv:920-923`) — the walker needs a real AXI-Lite master here. |
| — | *(S8,S9 repeat per token; the TB's `EFS` check sits between them — walker does **not** emit it)* | | |
| S10 | `WJ` | `op=1(EMIT), bank=0, pat=1(PAT_T), rows=nc(ci), nb=BPR, sel=0` | `:395`. Emits `nc*BPR = nc*D/8` beats = the transposed k8 chunk (one MXE WLOAD stream, K=D). |
| S11 | `AJ` | `op=1(EMIT), bank=1, pat=0(PAT_ROW), rows=1, nb=BPR, sel=0` | `:396`. Re-emits the q8 row; `BPR` beats = MXE `total_a = M·ceil(K/8) = 1·D/8` (`rtl/mxe/mxe_ctrl.sv:169-175`). Identical every chunk. |
| S12 | `DESC` | `desc_words(OP_GEMM_OS=0x02, m=1, k=D, n=nc(ci), mode_os=1)` → 4 words emitted **hi→lo**: `[v>>96, v>>64, v>>32, v]`, `v = 0x02 | 1<<8 | D<<20 | nc<<32 | 1<<67` (`:143-148, :397`) | e.g. D=64,nc=8 → `DESC 00000000 00000008 00000008 04000102`. **Verified exactly for all 24 cases.** |
| — | *(end chunk loop)* | | |
| S13 | *(TB-only)* `ETIP 00` iff `T ≤ 8` | `:398-399` | Not a walker emission. Flagged only because it is the one `T`-conditional in the phase. |
| S14 | *(TB-only)* `ESS s_c 1` | `:400` | Not a walker emission. |
| S15 | `AJ` | `op=0(LOAD), bank=1, pat=0, rows=1, nb=min(nbT,BPR), sel=0` | `stage_c8_load` `:311`. Overwrites bank1 row0 (q8 is dead). |
| S16 | `AJ` **iff `split`** | `op=0, bank=0, pat=0, rows=1, nb=nbT-BPR, sel=0` | `:312-314`. Tail into **bank0 row0**. Generator asserts `nbT-BPR ≤ BPR`. |

**Score-phase DESC ordering is `WJ-EMIT → AJ-EMIT → DESC` (weights first).** This is safe *only* because `WJ` and `AJ` target two different `apex_stage_buf` instances, so neither blocks the other's `job_ready`. Do not generalise it to the pv phase.

---

## 2. PV PHASE ROM (`gen_l3_vectors.py:404-431`)

| # | op | args (formula) | notes / ordering |
|---|---|---|---|
| P0 | `CSRP` | `addr=0x04, mask=0x1, exp=0x1` | APEX CSR idle poll (`route(poll=True)`, `:415`). |
| P1 | `ROUTE` | `0x00CF` (constant) | Same as score but `rdst=0` (MXE result → `ro` out) (`:415`). |
| — | *(per output block `j = 0 .. BPR-1`)* | | |
| P2 | `AJ` | `op=1(EMIT), bank=1, pat=0(PAT_ROW), rows=1, nb=min(nbT,BPR), sel=0` | `stage_c8_emit` `:325`. |
| P3 | `DESC` | `desc_words(0x02, m=1, k=T, n=8, mode_os=1, rq_en=1, rq_scale=RQ_S, rq_shift=RQ_SH)`; `v = 0x02 \| 1<<8 \| T<<20 \| 8<<32 \| RQ_S<<44 \| RQ_SH<<60 \| 1<<65 \| 1<<67`, words hi→lo (`:417-419`) | **`RQ_S`/`RQ_SH` are LOADED descriptor fields** (causally circular, B1_WALKER.md §2 corr. 1). Constant across all `BPR` blocks — verified: exactly `BPR` distinct-position DESCs, all identical, in every case. Decoded values e.g. `calib_d64_T128 → rq_scale=0xF387, rq_shift=22`; `calib_d128_T100 → 0xA5F3, 20`; the trivial-V cases → `0xFFFC, 23`. |
| P4 | `AJ` **iff `split`** | `op=1(EMIT), bank=0, pat=0, rows=1, nb=nbT-BPR, sel=0` | `:327-328`. **MUST come after P3 — see §4.** |
| — | *(per chunk `ci = 0 .. NCH-1`, nested inside j)* | | |
| P5 | `WJ` | `op=0(LOAD), bank=0, pat=0, rows=nc(ci), nb=BPR, sel=0` | `:424`. Identical to S6. |
| P6 | `FJOB` | `rows = nc(ci)` | `:425`. |
| P7 | `KVP` | `addr=0x04, mask=0x1, exp=0x1` | KVQ idle, one per token. |
| P8 | `KVW` | `addr=0x2C (RADDR), data = T + t`, for `t = c0(ci) .. c0(ci)+nc-1` | `:428`. **V record index = T + t** (K occupies records `0..T-1`, V `T..2T-1`; `store_kv_phase:546`). |
| P9 | `WJ` | `op=1(EMIT), bank=0, pat=2(PAT_D), rows=nc(ci), nb=1, sel=j` | `:430`. `nb=1` and `sel=j` — **not** `BPR`/`0`. PAT_D emits exactly `MXE_N=8` beats regardless of `rows`/`nb` (`rtl/top/glue/apex_stage_buf.sv:187`), and legality pins `sel < BPR` (`:173`), which is exactly the `j` range. |
| — | *(end chunk loop)* | | |
| P10 | *(TB-only)* `ERO o8[8j..8j+7] 1` | `:431` | Not a walker emission. |
| — | *(end j loop)* | | |

**`pv` re-reads all T records `BPR` times** — the `chunks(T)` loop is fully nested inside the `j` loop, so the KVQ read sequence `T..2T-1` is replayed verbatim for every one of the `BPR` output blocks, each preceded by its own idle poll. Total pv KVQ traffic = `BPR·T` `KVW` + `BPR·T` `KVP`. Only `WJ`-EMIT's `sel` changes between blocks; `WJ`-LOAD, `FJOB`, `KVP`, `KVW`, `AJ`, `DESC` are byte-identical across `j`.

---

## 3. Where CS / QS(S-4) sit — precisely

They form **one contiguous run each, at the very front of the score phase**, entirely before the first K-record read:

```
CSRP → ROUTE(0x00EF) → SJOB(T) → CS[0..T-1] → QJOB(1,T) → QS[0..T-1] → [chunk 0: WJ,FJOB,KVP,KVW,…]
```

The last `QS` word precedes the **first** `KVW RADDR` (which is what surfaces `s_k[0]` on `fs_*`) by `NCH·5 + …` ops. Concretely at `calib_d64_T128` the 128 `CS` + 128 `QS` words are ops 6..133 and 135..262 of the phase, and the first `KVW 2c 00000000` is op 265. There are **zero** `CS`/`QS` words anywhere in `pv_phase`.

This is exactly the ordering that forces the **store-time fp16 scale cache** (B1_WALKER.md §4): the composite unit must have `s_k[t]` and `s_v[t]` for all `t` *before* any read is issued, so it cannot source them from `fs_*`. At CQ-8 the KVQ's stored per-record scale is bit-identical to the feeder's read-time scale, so the cache (2·T·16 b) closes it with zero added transactions.

Note also: `s_q` (the other `cs` input) is produced in q-inject and checked at `ESS` (`:571`), i.e. it is available before the score phase starts. `SCORE_FRAC` and `1/sqrt(D)` are constants.

---

## 4. Load-bearing ordering constraints (deadlock-relevant)

**(a) `stage_c8_emit` = EMIT → DESC → EMIT, and it is a real deadlock, not a style rule.**
Mechanism, both halves verified in RTL:
- `apex_stage_buf.sv:216`: `assign job_ready = (state == ST_IDLE) && !done && !job_error;` — the **second** `AJ` on the *same* instance cannot be accepted until the first job has fully drained (post-skid `done`).
- `mxe_ctrl.sv:189`: `assign a_ready = (state == S_INGEST);` — MXE accepts activation beats **only** in `S_INGEST`, which is entered on descriptor acceptance (`:298`).

So the emission order `AJ, AJ, DESC` wedges: job 1 cannot drain (MXE idle), therefore job 2's `job_ready` never rises, therefore the walker blocks *before* it ever emits the `DESC` that would unblock it. `AJ, DESC, AJ` is the only working order. The two jobs together deliver exactly `nbT = ceil(T/8)` beats, matching MXE's `total_a = m_dim · ceil(k_dim/8) = 1 · ceil(T/8)` (`mxe_ctrl.sv:169-175`).

**Exercised only when `split` (`nbT > BPR` ⟺ `T > D`).** Among the 24 CQ-8 cases that is exactly **3**: `calib_d64_T70` (nbT=9,BPR=8), `calib_d64_T128` and `adv_outlier1000` (nbT=16,BPR=8). (B1_WALKER.md:503 says 4 of 27 — the fourth, `adv_outlier1000_cq4p`, is grouped-tier and therefore **refused** by walker v1, so the walker's own coverage of this path is 3 cases. A naive `AJ,AJ,DESC` walker passes 21/24 and looks green.)

**(b) Score-phase asymmetry.** Score is `WJ-EMIT, AJ-EMIT, DESC`; pv is `AJ-EMIT, DESC, [AJ-EMIT]`. Both are correct for the reason in §1 (different `apex_stage_buf` instances vs. the same one). A walker that "normalises" these into one template breaks pv.

**(c) `FJOB` must follow its `WJ`-LOAD and precede the reads.** The LOAD job must be armed before the feeder starts pushing (route has `fdst=1`, feeder → wgt-stage).

**(d) Poll-before-level-change.** Both `ROUTE` ops are preceded by a `CSRP STATUS.idle` (`route(poll=True)`, `:233-234`). This is not decorative — it is the fix for the phase-F wedge of 2026-07-09 (comment at `:224-226`).

**(e) No mid-phase re-routes in walker v1.** The poll-free re-route branches at `:386-388` (score, predicate `bm[ci] != bm[max(ci-1,0)]`) and `:421-423` (pv, predicate `ci == 0 or bm[ci] != bm[ci-1]`) are guarded by `blkmap is not None`. Only `tip_auto_mixed` (`:817-819`) passes a blkmap, and it is grouped-tier ⇒ refused. **The walker v1 ROM emits exactly two `ROUTE` ops across both phases.** Do not implement the two divergent predicates in v1.

**(f) `fq_out_ready` footgun** (`rtl/top/apex_top.sv:771-772`): `assign fq_out_ready = rt_feeder_dst ? ws_ld_ready : (rt_act_src ? 1'b0 : as_ld_ready);` — with `rt_feeder_dst=0 && rt_act_src=1` the feeder is hard-stalled with no error. Both ROM routes have `fdst=1`, so any walker route word that drops bit 1 while bit 2 is set silently wedges.

---

## 5. Poll points — exhaustive

Two distinct buses, both using numeral `0x04`. Do not conflate them.

| poll | bus | address | mask/exp | count in score | count in pv |
|---|---|---|---|---|---|
| `CSRP` — APEX `STATUS.idle` | APEX CSR (`csr_regs`) | `0x04` bit 0 | `0x1 / 0x1` | 1 (before `ROUTE`) | 1 (before `ROUTE`) |
| `KVP` — KVQ `STATUS[0]=idle` | KVQ AXI-Lite (`kvq_engine.sv:950-953`) | `0x04` bit 0 | `0x1 / 0x1` | `T` (one per `KVW RADDR`) | `BPR · T` |

There are **no other polls** in either phase — no `OCC` read, no `KVW FLUSH`, no `CSRW`. Every `KVP` is immediately followed by its `KVW RADDR`; the pairing is 1:1 with no batching.

---

## 6. Op-count formulas (for FSM sizing and the equivalence gate)

Walker-**emitted** drive ops (`ROUTE|SJOB|CS|QJOB|QS|WJ|FJOB|KVW|AJ|DESC`):
```
score_drive = 1 + 1 + T + 1 + T + 5·NCH + T + (2 if split else 1)
pv_drive    = 1 + BPR · [ (2 if split else 1) + 1 + 3·NCH + T ]
```
Polls: `score_poll = 1 + T`, `pv_poll = 1 + BPR·T`.

Sanity-check against the doc's own figures: at `calib_d64_T128` this gives **drive = 1,902**, **drive+poll = 3,056** — exactly the two numbers in `docs/design/B1_WALKER.md:474` ("3,056 ops … walker-emitted drive subset is 1,902"). Full table:

| case | D | T | BPR | nbT | split | drive | poll | total |
|---|---|---|---|---|---|---|---|---|
| adv_T1 / rand_d64_T1 | 64 | 1 | 8 | 1 | – | 61 | 11 | 72 |
| bug_d128_stagebuf_nb | 128 | 4 | 16 | 1 | – | 166 | 70 | 236 |
| rand_d128_T5 | 128 | 5 | 16 | 1 | – | 185 | 87 | 272 |
| rand_d64_T30 | 64 | 30 | 8 | 4 | – | 467 | 272 | 739 |
| calib_d64_T70 | 64 | 70 | 8 | 9 | **yes** | 1,061 | 632 | 1,693 |
| calib_d64_T128 / adv_outlier1000 | 64 | 128 | 8 | 16 | **yes** | 1,902 | 1,154 | 3,056 |
| calib_d128_T100 | 128 | 100 | 16 | 13 | – | 2,626 | 1,702 | 4,328 |

---

## 7. Explicitly NOT walker emissions

`TAPSC` / `TAPPR` (`:372-373`), `EFS` (`:394,429`), `ESS` (`:400`), `ERO` (`:431`), `ETIP` (`:399`). These are TB scoreboard expectations parsed at `verif/top/l3/tb_apex_l3.sv:940-975`. They interleave with the drive stream in the case file and must be filtered out of the extracted-trace comparison. The `EFS` `last` flag (`t == c0+nc-1`) is per-*chunk*, not per-phase — relevant only if the mirror-based scoreboard of §B-1 re-derives it.

## 8. Constant reference

`OP_GEMM_OS = 0x02` (`:108`); `KV["STATUS"]=0x04`, `KV["RADDR"]=0x2C` (`:110-111`); `CSR["STATUS"]=0x04` (`:112-113`); `MXE_N = 8`; `PAT_ROW=0, PAT_T=1, PAT_D=2` (`rtl/top/glue/apex_stage_buf.sv:107-109`); stage-buf legality `rows∈[1,R_MAX]`, `nb∈[1,BPR]`, `PAT_ROW: sel<R_MAX`, `PAT_T: rows≤MXE_N`, `PAT_D: sel<BPR ∧ rows≤MXE_N` (`:36-38,167-173`).

---


## 2. Composite arithmetic in hardware (S-3 / S-4)

All numeric claims below were verified by exhaustive/structured Python runs (scratch-only scripts: composite_check.py, const_width.py — not repo artifacts) against the spec emitter formulas at verif/top/l3/gen_l3_vectors.py:375-384,:133-140 and the oracle verif/top/l3/walker_composite_golden.py:89-129. SCORE_FRAC=10 pinned (golden/apex_golden/attention.py:75, walker_composite_golden.py:65-69); OUT_FRAC=15 (attention.py:76).

== 1. INPUT DOMAIN (the load-bearing precondition) ==
All three scale inputs are POSITIVE NORMAL fp16 by construction — every scale producer floors at EPS = 2^-14 = 0x0400 (smallest normal fp16) and can otherwise emit only a normal value or +inf 0x7C00 on out-of-contract overflow:
- read-time s_k/s_v: seam_feeder_quant.sv:161-202 `scale_from_amax` — `:175` (amax zero/subnormal -> 16'h0400), `:192` (`es < -14` -> 16'h0400), `:199` (overflow -> 16'h7C00, SVA-guarded P6);
- store-time record scale (the walker's actual CQ-8 source via the scale cache): cq_scale_unit/cq_scale_pipe use the same functions; EPS_F16 = 16'h0400 at rtl/kvq/cores/cq_fp_pkg.sv:56-57; cqv_scale wire kvq_engine.sv:255/:279/:290;
- s_q (ss_* tap, apex_top.sv:245-247, from apex_scale_quant): rtl/top/glue/apex_scale_quant.sv:298 (`a_pn==0 -> 16'h0400`) and :313 (`es < -14 -> 16'h0400`).
Precondition sweep of f32_bits_exact_checked over ALL 65,536 fp16 patterns: violations are EXACTLY {negative: 31,744, +0: 1, inf/nan: 2,048}; all 1,023 positive denormals PASS the asserts (they normalize into a positive-normal, fp16-grade f32). So the golden's asserts alone prove "only s_v <= 0 (or inf/nan) violates" — they do NOT exclude denormal inputs. The exclusion of denormal inputs comes from the EPS floor above, not from the asserts. Consequence for RTL: the trivial qs/cs unpack ({1'b1, m} hidden bit, exponent arithmetic) is legal, but it MUST carry an SVA `assert (!s[15] && s[14:10] != 0 && s[14:10] != 31)` on every consumed scale, because a denormal would silently break the rebias while still passing the software oracle's own preconditions.

== 2. qs (S-4 fold) — pure exponent rebias, zero arithmetic ==
Exhaustively verified over all 30,720 positive-normal fp16 patterns, 0 mismatches:
  fp16 {s=0, e[4:0], m[9:0]}  ->  f32 bits = ((e + 97) << 23) | (m << 13)
Derivation: f32 biased exp = (e-15) - 15 + 127 = e + 97. RTL:
  wire [7:0] e32 = {3'b0, sv[14:10]} + 8'd97;         // e in [1,30] -> e32 in [98,127]; never overflows, result always positive normal
  assign qs_bits = {1'b0, e32, sv[9:0], 13'b0};
No rounding, no normalization, no special cases inside the contract domain. The fp16-grade assert (bits & 0x1FFF == 0, gen_l3_vectors.py:138) is satisfied structurally (low 13 bits are wired 0).

== 3. cs — the per-D constant (computed, exact) ==
constant = float(1<<10)/sqrt(D), kept in f64 by the emitter (gen_l3_vectors.py:377-378):
- D=64: 2^10/8 = 128.0 = 2^7 — EXACT POWER OF TWO. f64 bits 0x4060000000000000, f32 bits 0x43000000.
- D=128: 2^10/(8*sqrt(2)) = 64*sqrt(2) = 90.50966799187808 — NOT a power of two. f64 bits 0x4056A09E667F3BCC; 53-bit significand Cm = 0x16A09E667F3BCC, unbiased exp 6 (value = Cm * 2^-46). Its f32 narrowing (0x42B504F3) is a DIFFERENT number — the hardware constant must be the f64-grade significand, not an f32 one: using the 24-bit f32-rounded constant mismatches the golden on 86,592 of 419,629 reachable product significands (measured).

== 4. cs @ D=64 — fully exact, no rounding hardware at all ==
s_q*s_k product P = sig1*sig2 with sig = {1'b1, m[9:0]} (11 bits) -> P in [2^20, 2^22), 21-22 significant bits, exactly representable in f32; *2^7 is an exponent add. Verified: integer model vs numpy golden, 20,064 pairs (corners incl. 0x0400/0x0401/0x07FF/0x3C00/0x7BFF + 20k random normals), 0 mismatches. Range check: E32 = e1 + e2 + 104 + nrm with nrm = P[21]; e1,e2 in [1,30] -> E32 in [106,165] — always normal f32, no overflow/underflow/clamp logic needed. RTL:
  P[21:0] = sig1 * sig2;                      // one 11x11 unsigned multiplier
  nrm     = P[21];                            // 1-bit normalize
  sig24   = nrm ? {P, 2'b00} : {P[20:0], 3'b000};   // left-align to 24 bits (exact)
  frac    = sig24[22:0];
  E32     = e1 + e2 + 8'd104 + nrm;
  cs_bits = {1'b0, E32, frac};
Sanity: 1.0*1.0 -> P=2^20, E32=134, value 128.0 ✓.

== 5. cs @ D=128 — one real multiply, one RNE rounding, and a proven simplification ==
Golden semantics: comp64 = RN53((s_q*s_k) * C) in f64 (the 22-bit product is f64-exact; ONE f64 rounding on the constant multiply), then comp32 = RN24(comp64) — nominally a double rounding. KEY RESULT (exhaustive): over ALL 419,629 distinct reachable product significands P = m1*m2 (m1,m2 in [1024,2047]), RN24(RN53(P*Cm)) == RN24(P*Cm) — double rounding is innocuous on the entire reachable set, with ZERO exact ties at the f32 boundary (guard=1 & sticky=0 occurs 0 times; 4 ties exist at the 53-bit intermediate but never perturb the final f32 — that is precisely why the two-step and one-step agree). Exponents cannot affect rounding because the result range is [~2^-21.5, ~2^38.5] (E32 in [105,166]) — no f32 subnormal or overflow is reachable, so the significand-space proof is fully exhaustive. Therefore the RTL implements ONE RNE rounding of the exact 75-bit product:
  localparam [52:0] CM = 53'h16A09E667F3BCC;   // f64 significand of 64*sqrt(2)
  P[21:0]  = sig1 * sig2;                      // 11x11
  M[74:0]  = P * CM;                           // 22x53 -> M in [2^72, 2^75)
  nb       = M[74] ? 75 : (M[73] ? 74 : 73);   // 2-bit normalize
  sig24    = M >> (nb-24);
  guard    = M[nb-25];  sticky = |M[nb-26:0];
  rnd      = guard & (sticky | sig24[0]);      // RNE
  {carry, sig} = sig24 + rnd;                  // carry: 24-bit overflow -> exp+1
  frac     = carry ? sig[23:1] : sig[22:0];
  E32      = e1 + e2 + nb + 8'd30 + carry;     // in [105,166]: no clamps needed
  cs_bits  = {1'b0, E32, frac};
Verified: this integer model vs numpy golden, 20,064 pairs, 0 mismatches (composite_check.py check F; exponent identity E32 = (e1+e2-50) + (cexp-52) + (nb-1) + 127 + carry with cexp=6). Sanity: 1.0*1.0 -> 0x42B504F3 ✓ (matches f32(90.50966799187808)).
Constant-width sweep (const_width.py): an RNE-truncated constant works down to W=42 bits (W=41 fails on exactly 1 significand; W=24, i.e. the f32 constant, fails on 86,592). RECOMMENDATION: keep the full 53-bit CM — on DSP48-class fabric 22x42 and 22x53 both cost 3 DSPs (18-bit slices), so truncation buys nothing and adds a proof dependency.

== 6. Multiplier hardware needed & rounding mode ==
Both builds exist (-GCFG_D=64/128, verif/top/l3/Makefile:85-91; F2 AFI needs +define+APEX_CL_D=128 per commit 2964ea8), and D is a BUILD parameter, so use generate: shared 11x11 P multiplier + generate-if (CFG_D==64) exponent-add path / (CFG_D==128) the 22x53 constant multiply + RNE round. Runtime descriptor D != CFG_D is a WALK_ERR_DESC refusal (B1_WALKER.md §3 table). Throughput need is one composite per token per phase (score: T cs then T qs, gen_l3_vectors.py:379-384) — 2-3 pipeline stages at will; even a sequential multiplier meets timing, but the DSP cascade is simpler. Rounding mode MUST be RNE (round-to-nearest, ties-to-even): numpy's .astype(np.float32) is the IEEE-754 default-mode C double->float cast; the RNE integer model matched the golden on every tested vector, and the round-up fraction on reachable D=128 products is 0.496, so any truncation (RTZ) implementation diverges on ~half of all vectors immediately. Note on ties: since guard&~sticky is unreachable at D=128 (0 cases) and D=64/qs never round at all, ties-to-even vs ties-away is formally indistinguishable on the reachable domain — implement RNE anyway (it is the stated contract), but know that no vector can kill a ties-direction mutant; do not burn stage-3 time hunting one. This sharpens walker_composite_golden.py:225-227's "directed tie-case vector is stage-3 work": at the f32 boundary no reachable tie exists, so that note applies only to discriminating f64-level reassociation in SOFTWARE replicas — the hardware, computing the exact product with a single RNE24, has no reassociation hazard by construction (proven equal to the golden's two-step rounding, §5).

== 7. Unit-test plan (stage 3, B1_WALKER.md §5 row 3) ==
Oracle = walker_composite_golden.py (score_composite/p_requant_composite — inputs are fp16 bit patterns, outputs uint32 bit patterns, exactly the RTL port semantics). Vector generation (no Verilator needed to build them): (a) qs: ALL 30,720 positive-normal fp16 — exhaustive in simulation, seconds; (b) cs: exhaustive mantissa sweep m1 x m2 = 1,048,576 pairs at fixed exponents (covers every reachable rounding decision, per §5's exponent-independence) + an exponent sweep e1 x e2 at a few fixed mantissas (covers E32 arithmetic incl. min/max 105/166 and the carry path) + the 677 distinct (s_q,s_k,D) triples / 610 distinct s_v the L3 corpus exercises (walker_composite_golden.py selftest prints these; corners adv_outlier1000, calib_d64_T128, d128_T100 per §5) — run both CFG_D builds; (c) directed round-up-vs-truncate vectors: any of the ~50% round-up P's kills an RTZ mutant; the W=24-constant mutant (f32-narrowed constant) is killed by 86,592 enumerable pairs — include a handful; (d) mutant m2 ("drop the S-3 narrowing / keep f64 precision") is ONLY observable at D=128 (D=64 cs and qs are exact at every width) — the mutation-kill gate for m2 must run the D=128 build or it will falsely report survival; (e) SVA checks: input-scale positive-normal assertion (§1) fires on a driven denormal/zero/negative/0x7C00. Pass criterion: bit-equality of all emitted uint32 words vs the oracle file, plus the oracle's own selftest staying green (make -C verif/top/l3 vectors && python3 verif/top/l3/walker_composite_golden.py verif/top/l3/build).

Files cited: verif/top/l3/gen_l3_vectors.py, verif/top/l3/walker_composite_golden.py, golden/apex_golden/attention.py, rtl/seam/seam_feeder_quant.sv, rtl/kvq/cores/cq_fp_pkg.sv, rtl/kvq/kvq_engine.sv, rtl/top/glue/apex_scale_quant.sv, docs/design/B1_WALKER.md. Verification scripts were scratch-only and are not repo artifacts; their results are recorded inline above.

---


## 3. House RTL contract a new module must follow

All paths repo-relative (surveyed on branch comp/b1-walker). Modules read end-to-end: rtl/seq/seq_walker.sv (216 l), rtl/xbr/stream_skid.sv (152 l), rtl/top/glue/apex_stage_buf.sv (317 l), rtl/seam/seam_feeder_quant.sv (386 l), plus rtl/apex_pkg.sv, verif/top/l3/Makefile, verif/top/l3/lint_waivers.vlt, verif/seq/*, verif/common/apex_stream1_sva.svh.

1. HEADER COMMENT STRUCTURE (mandatory, load-bearing — it is the per-file contract)
Every house module opens with a `//` block (no /* */) in this order:
 a. `// <filename>.sv — <one-line role>.` (seq_walker.sv:1, apex_stage_buf.sv:1, seam_feeder_quant.sv:1)
 b. "Implements:" line citing ARCHITECTURE.md sections AND decision IDs — e.g. seq_walker.sv:3-8 cites "ARCHITECTURE.md §1 SEQ row … §5 stream/job semantics, D-006 … §7 CTRL.soft_reset"; seam_feeder_quant.sv:3-6 cites "ARCHITECTURE.md D-021 … C-1 quant rule, C-2 (RNE), D-010 … §5 stream contract, D-006" and names the golden arbiter file+function ("golden/apex_golden/attention.py quant_rows_i8").
 c. FUNCTION / Structure / FSM description with state list (seq_walker.sv:15-24, seam_feeder_quant.sv:10-18).
 d. Numeric blocks carry BIT-EXACTNESS PROOFS P1..Pn vs the float64 golden in the header (seam_feeder_quant.sv:19-48, apex_scale_quant.sv:40+).
 e. "Job legality (§3 style)" paragraph: error pulse + sticky, NO state change (apex_stage_buf.sv:36-39, seam_feeder_quant.sv:54-56).
 f. "D-006:" paragraph defining done (post-skid acceptance) and busy (seam_feeder_quant.sv:57-59, apex_stage_buf.sv:53-58).
 g. Closed bug IDs are recorded in the header when relevant (F-5a/F-5b closure with date, apex_stage_buf.sv:41-51).
Section separators inside the body use box comments: `// ── <name> ──────…` (6 occurrences in seam_feeder_quant.sv).
B1's walker module header must therefore cite B1_WALKER.md §3/§4 (A-1, B-1), D-006, §5, and the composite golden (verif/top/l3/walker_composite_golden.py).

2. RESET STYLE — synchronous, active-low, uniformly
All new/house RTL: `always_ff @(posedge clk) begin if (!rst_n) … end` with the port commented `// synchronous, active low` (seq_walker.sv:61, stream_skid.sv:91, apex_stage_buf.sv:70, seam_feeder_quant.sv:71). NO async resets: `@(posedge clk or negedge rst_n)` exists ONLY in the vendored kvq tree (rtl/kvq/* — 10 files) and vendor/rsqrt_unit.sv, and both are explicitly waived SYNCASYNCNET as "vendored-lineage files only" (verif/top/l3/lint_waivers.vlt:22-26). A new B1 module with async reset would need a new waiver = contract violation. Reset clears ALL FSM/counter state to '0 / S_IDLE (seq_walker.sv:153-161, seam_feeder_quant.sv:282-298). Data memories (`mem`, `rbuf`) are deliberately NOT reset — "stored bytes are don't-care after reset since any consumer re-loads first" (apex_stage_buf.sv:56-58).

3. always_ff vs always_comb
- One monolithic `always_ff @(posedge clk)` per module holding the FSM + all counters (seq_walker.sv:152-203, apex_stage_buf.sv:219-311, seam_feeder_quant.sv:281-384). Pattern inside: default-pulse clears first (`done <= 1'b0; job_error <= 1'b0;`), then post-skid acceptance counters (`if (out_valid && out_ready) out_acc <= out_acc + CNT_W'(1);` seam:302-303), then `unique case (state) … default: state <= ST_IDLE; endcase`.
- `always_comb` only for genuinely multi-branch combinational logic: legality check and beat-count derivation (apex_stage_buf.sv:166-190), emit-word mux (apex_stage_buf.sv:194-209). Simple combinational = `assign` (all handshake outputs: seq_walker.sv:206-214, seam:268-276).
- Pure functions are `function automatic` declared in-module, fully combinational, with locals at top (seam_feeder_quant.sv:161-237, apex_stage_buf.sv:114-118 `midx`).
- No `always @`, no `always_latch`, no inferred latches anywhere in house RTL (grep clean outside kvq/vendor; single historic `lint_off DECLFILENAME` pragma pair in rtl/asu/rsqrt.sv:66,201 — vendored wrapper; new code uses NO inline pragmas).

4. NAMING CONVENTIONS
- State enum: `typedef enum logic [N-1:0] { … } state_e;` with variable `state_e state;`. Newer modules use `ST_*` prefixes (ST_IDLE/ST_LOAD/ST_EMIT/ST_WAIT apex_stage_buf.sv:154-155; ST_IDLE/ST_INGEST/ST_SCALE/ST_SPUSH/ST_DRAIN/ST_WAIT seam:240-247); seq_walker (older) uses `S_*` (seq_walker.sv:119). Use ST_* for B1. Enum values get explicit encodings (`2'd0` etc.).
- localparams: SCREAMING_SNAKE, `int unsigned` (BPR, CNT_W, EW, NB_W, PTR_W, DW; seq_walker.sv:90-92, seam:107-109). Pattern constants as sized localparams (`localparam logic [1:0] PAT_ROW = 2'd0;` apex_stage_buf.sv:107-109).
- Handshakes: `<ch>_valid` / `<ch>_ready` / payload (`<ch>_data`, `<ch>_desc`, `<ch>_beat`); channel prefixes are 2-3 letters (ds_, md_, ld_, em_, in_, out_, scl_, sc_, o_, i_, eo_, li_). Pre-skid internal side of an output uses a distinct short prefix (o_/eo_/sc_), post-skid keeps the port name.
- Registered copies of job fields get `_q` suffix (bank_q, pat_q, rows_q, sel_q, nb_q, head_q); counters `_c`/`_cnt`/`_idx`; post-skid accept counters `_acc` (em_acc, out_acc, scl_acc).
- Instances: `u_<name>` (u_in_skid, u_ld_skid, u_em_skid, u_scl_skid, u_seq at apex_top.sv:296). Named generate blocks for elab checks: `g_chk_*` or `gen_*_err` (apex_stage_buf.sv:100-105, seq_walker.sv:94).
- Job/status port set (the D-006 idiom every job-driven block repeats): `job_valid, job_ready, job_<fields>, job_error (1-cycle pulse), job_error_sticky, busy, done` (apex_stage_buf.sv:73-84, seam:74-80). `job_ready = (state == ST_IDLE) && !done && !job_error;` (stage_buf:216, seam:275) — ready gated strictly after done. `busy = (state != ST_IDLE) || <post-skid output valids>` (stage_buf:217, seam:276, seq:213-214).

5. PARAMETERS + ELABORATION VALIDATION
`parameter int unsigned NAME = default` with trailing `// comment`; derived widths as `localparam` INSIDE the #() list when they size ports (F-5a pattern: `localparam int unsigned NB_W = $clog2(D / 8) + 1` apex_stage_buf.sv:64-67, "derived, never a magic number"). Validation is a bare generate-if + `$error` at elaboration, in a named block, immediately after the port list:
  `if (!(D == 64 || D == 128)) begin : g_chk_d  $error("seam_feeder_quant: D must be 64 or 128 (D-021)"); end` (seam:100-105; same at stage_buf:100-105, seq_walker:94-96). Message = "<module>: <constraint>" + decision ID.

6. SVA LOCATION — verif/, NEVER in-module
grep for `assert property` over rtl/ returns nothing. All assertions live in verif/ as bindable checker modules in `.svh` files with include guards: reusable §5 stream checker verif/common/apex_stream1_sva.svh (module apex_stream1_sva, params WIDTH/NAME; the one property: `(valid && !ready) |=> (valid && $stable(data))`, :34-38), block-contract packs like verif/seq/seq_sb_sva.svh (models in_flight itself, checks D-006/busy/queue/abort). They are `bind`-ed by the TB (tb_apex_l3.sv:297-327 binds into apex_scale_quant/apex_stage_buf/etc.; verif/seq binds into seq_walker). Written to the "Verilator 5.x SVA subset: simple concurrent assertions with |-> / |=> / $past / $stable / $rose only" (apex_stream1_sva.svh:8-9, seq_sb_sva.svh:4-5), compiled under `--assert` as a build gate (D-012). B1: put walker-contract SVA in verif/seq_walker/<name>_sva.svh + reuse apex_stream1_sva; do NOT put asserts in the RTL file.

7. LINT CONTRACT (Verilator 5.044, /opt/homebrew/bin/verilator)
VOPTS = `-Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. lint_waivers.vlt` (verif/top/l3/Makefile:68-69); lint target is `--lint-only -Wall` on the full apex_top and must print "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg + vendored files only)" (:75-77). Waiver inventory (verif/top/l3/lint_waivers.vlt, 33 lines): UNUSEDPARAM on *apex_pkg.sv only (frozen contract, :10); WIDTHEXPAND/WIDTHTRUNC per-line + SYNCASYNCNET on vendor/rsqrt_unit.sv (:13-18); SYNCASYNCNET on *kvq/* (vendored async-reset lineage, :26); one CMPCONST per-line on apex_stage_buf.sv:167 (config-dependent constant fold at R_MAX=31, :27-33). The stated rule: "NONE of the new top-level RTL (rtl/top/*, rtl/top/glue/*), the SVA packs, or the smoke TB is waived — those files must be -Wall clean on their own" (:4-5). verif/seq/lint_waivers.vlt repeats it: "NO waivers on seq_walker.sv, stream_skid.sv, the SVA checkers, or the TB".
-Wall pitfalls and the house counter-idioms (B1 RTL must ship with ZERO new waivers):
 - WIDTHEXPAND/WIDTHTRUNC: every arithmetic literal and cross-width move is explicitly size-cast with the tick-cast: `count + CNT_W'(1)`, `wr_ptr + PTR_W'(1)` (seq_walker.sv:171,188-189), `5'(R_MAX)`, `NB_W'(BPR)`, `CNT_W'(job_rows) * CNT_W'(job_nb)` (stage_buf:167-186), `EW'(D - 1)`, `DIM_W'(ROWS_MAX)` (seam:265,329), struct→vector `DW'(ds_desc)` (seq:107), int-widening `32'(bank)` inside functions (stage_buf:117), comparisons against enums of int e.g. `r < int'({27'b0, rows_q})` (stage_buf:201). Never rely on implicit extension.
 - UNUSEDSIGNAL: consumed-but-unused fields are tied off explicitly: `logic unused_ok; assign unused_ok = li_beat.last;` with a WHY comment (stage_buf:313-315, apex_lane32_ser.sv:188-189, apex_q78_to_fp32.sv:56-57); multi-signal form `assign unused_ok = &{1'b0, wdata[31:9], …};` (csr_regs.sv:261-262, apex_top.sv:936-937). Inside functions, unused hidden bits become `unused_hid` locals (seam:171,195).
 - CASEINCOMPLETE / latches: every case is `unique case` WITH a `default:` arm (`default: state <= ST_IDLE;` seq:183, stage_buf:308, seam:381; `default: ;` for no-ops seq:190, stage_buf:295); always_comb blocks assign a default first (`em_word = '0;` stage_buf:195, `legal = …` before the case, :166-178).
 - UNUSEDPARAM: don't add parameters you don't consume (waiver is apex_pkg-only).
 - CMPCONST: avoid tautological compares on config-max fields (the one stage_buf waiver exists because R_MAX=31 saturates the 5-bit field — keep B1 field widths derived so legality compares stay non-degenerate, or expect to justify a per-line waiver).
 - SYNCASYNCNET: use sync reset (item 2) — mixing styles on rst_n is what forced the kvq waiver.

8. stream_skid INSTANTIATION IDIOM (§5/D-019: a skid on EVERY external stream boundary)
stream_skid (rtl/xbr/stream_skid.sv:87-102) is generic — `parameter int WIDTH`, does NOT import apex_pkg, flat `logic [WIDTH-1:0]` payload. House idiom for struct payloads:
  - declare pre-skid pair + vector: `logic o_valid, o_ready; lane8_beat_t o_beat; logic [$bits(lane8_beat_t)-1:0] out_vec;`
  - instantiate `stream_skid #(.WIDTH($bits(lane8_beat_t))) u_out_skid ( .clk(clk), .rst_n(rst_n), .s_valid(o_valid), .s_ready(o_ready), .s_data({o_beat.data, o_beat.last}), .m_valid(out_valid), .m_ready(out_ready), .m_data(out_vec));` — struct fields concatenated explicitly on s_data, and the m-side unpacked with a tick-cast: `assign out_beat = lane8_beat_t'(out_vec);` (seam:127-141; stage_buf:120-151; sideband example WIDTH=17 packing {last, data} seam:143-158; seq_walker packs the whole descriptor `DW'(ds_desc)` / `mxe_desc_t'(head_q)` :102-111,207). All ports named-association, one per line, aligned. Skid latency is 1 cycle, capacity 2 — the D-006 consequence: `done` counts POST-skid accepted beats via `*_acc` counters and an ST_WAIT state (`if (out_acc == total_beats) done <= 1'b1;` seam:374-379, stage_buf:301-306); busy includes skid occupancy (seq_walker `in_m_valid` :213, and the seq abort/skid-drop bookkeeping :129-150 shows the level of care expected when gating around the 1-cycle skid latency).

9. apex_pkg IMPORT + PACKAGE DISCIPLINE
House form: `module <name>\n  import apex_pkg::*;\n#(\n  parameter …\n)(\n  ports\n);` — import clause between module name and #() (seq_walker.sv:55-59, apex_stage_buf.sv:60-68, seam:64-69). Generic infrastructure (stream_skid) does not import it. apex_pkg.sv is include-guarded (`ifndef APEX_PKG_SV) and FROZEN: "Any change here is a contract change: bump APEX_VERSION and update the golden mirrors in the same commit" (apex_pkg.sv:5-7). B1_WALKER.md §3 (:162-166, :195-205) therefore mandates the walker's descriptor struct/enums go in a NEW `rtl/seq/seq_walker_pkg.sv` — same guard+package idiom as apex_pkg — NOT into apex_pkg. Types available from apex_pkg that B1 will use: mxe_desc_t (:38-52, layout FROZEN), lane8_beat_t (:57-60), DIM_W=12, MXE_N=8, kvq_tier_e {KVQ_CQ8, KVQ_CQ4, KVQ_CQ4P} (:86-90).

10. BUILD/TB INTEGRATION EXPECTATIONS for the new module
- Add the file to RTL_CORE in verif/top/l3/Makefile (:24-64, one path per line) — it compiles under the same VOPTS in lint, both -GCFG_D=64/128 builds, and the mutation builds (RTL_CORE is reused at :134).
- Unit suite: copy the verif/seq/ pattern (B1_WALKER.md:344-352 names it the template): own Makefile with `-Wall --timing --assert`, own minimal lint_waivers.vlt (apex_pkg UNUSEDPARAM only), independent Python golden vector gen, D-006 stub, apex_stream1_sva binds, mutation-kill gate, pipefail on every gate line ("the tee lesson", l3 Makefile:14,17-18).

11. CONCRETE SKELETON for a new B1 module (all conventions above applied):

```systemverilog
// seq_layer_walker.sv — B1: autonomous layer-walker, score+pv @ CQ-8.
//
// Implements: docs/design/B1_WALKER.md §3 (WALK CSR window, refusal gate
//             A-1: tier != TIER_CQ8 => WALK_ERR_TIER, NO walk), §4
//             (Acceptance A strict-order equivalence), ARCHITECTURE.md §5
//             stream contract, D-006 job semantics. Golden arbiter:
//             verif/top/l3/walker_composite_golden.py + extract_trace.py.
//
// FUNCTION (one job = one layer step, D-006): …FSM/state list…
//
// Job legality (§3 style): tier == KVQ_CQ8, D in {64,128}, 1 <= T <= T_MAX;
// violation => walk_err pulse + sticky + err_code, NO state change.
//
// D-006: done <=> every emitted control word ACCEPTED downstream
// (post-skid); busy = (FSM != ST_IDLE) || beats pending in output skids.

module seq_layer_walker
  import apex_pkg::*;
  import seq_walker_pkg::*;
#(
  parameter int unsigned T_MAX = 128,               // scale-cache depth
  localparam int unsigned TW   = $clog2(T_MAX + 1)  // derived, sizes ports
)(
  input  logic        clk,
  input  logic        rst_n,          // synchronous, active low

  // job interface (D-006)
  input  logic        job_valid,
  output logic        job_ready,
  input  walk_desc_t  job_desc,
  output logic        job_error,
  output logic        job_error_sticky,
  output logic [2:0]  job_err_code,   // WALK_ERR_* (B1_WALKER.md §3)
  output logic        busy,
  output logic        done,

  // emitted descriptor stream out (drives seq_walker ds_* in walk mode)
  output logic        wd_valid,
  input  logic        wd_ready,
  output mxe_desc_t   wd_desc
);

  // ── parameter legality (elaboration-time) ────────────────────────────
  if (T_MAX < 1 || T_MAX > 4095) begin : g_chk_t
    $error("seq_layer_walker: T_MAX must be in [1, 4095]");
  end

  // ── output skid (§5/D-019: every boundary crossing) ──────────────────
  localparam int unsigned DW = $bits(mxe_desc_t);
  logic          d_valid, d_ready;
  mxe_desc_t     d_desc;
  logic [DW-1:0] wd_vec;

  stream_skid #(.WIDTH(DW)) u_wd_skid (
    .clk     (clk),
    .rst_n   (rst_n),
    .s_valid (d_valid),
    .s_ready (d_ready),
    .s_data  (DW'(d_desc)),
    .m_valid (wd_valid),
    .m_ready (wd_ready),
    .m_data  (wd_vec)
  );
  assign wd_desc = mxe_desc_t'(wd_vec);

  // ── FSM / state ──────────────────────────────────────────────────────
  typedef enum logic [2:0] { ST_IDLE = 3'd0, ST_EMIT = 3'd1,
                             ST_WAIT = 3'd2 } state_e;
  state_e           state;
  logic [TW-1:0]    tok_c;            // counters: _c/_cnt
  logic [15:0]      cnt_acc;          // POST-skid accepted words

  logic legal;
  assign legal = (job_desc.tier == KVQ_CQ8)          // A-1 refusal gate
              && (job_desc.t != '0) && (job_desc.t <= TW'(T_MAX));

  assign d_valid   = (state == ST_EMIT);
  assign job_ready = (state == ST_IDLE) && !done && !job_error;
  assign busy      = (state != ST_IDLE) || wd_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= ST_IDLE;
      done             <= 1'b0;
      job_error        <= 1'b0;
      job_error_sticky <= 1'b0;
      job_err_code     <= '0;
      tok_c            <= '0;
      cnt_acc          <= '0;
    end else begin
      done      <= 1'b0;
      job_error <= 1'b0;
      if (wd_valid && wd_ready) cnt_acc <= cnt_acc + 16'(1);

      unique case (state)
        ST_IDLE: begin
          if (job_valid && job_ready) begin
            if (legal) begin
              state <= ST_EMIT;  tok_c <= '0;  cnt_acc <= '0;
            end else begin                 // §3: pulse + sticky, no effects
              job_error        <= 1'b1;
              job_error_sticky <= 1'b1;
              job_err_code     <= (job_desc.tier != KVQ_CQ8)
                                ? WALK_ERR_TIER : WALK_ERR_DESC;
            end
          end
        end
        ST_EMIT: if (d_ready) begin /* advance; -> ST_WAIT on last */ end
        ST_WAIT: begin                     // D-006: post-skid acceptance
          if (cnt_acc == 16'(/* total */0)) begin
            done  <= 1'b1;
            state <= ST_IDLE;
          end
        end
        default: state <= ST_IDLE;
      endcase
    end
  end

  // (any unused input field:)
  // logic unused_ok;  assign unused_ok = <field>;

endmodule
```

Companion obligations: seq_walker_pkg.sv (walk_desc_t incl. LOADED rq_scale[15:0]/rq_shift[4:0], WALK_ERR_* enum, include-guarded package); SVA in verif/seq_walker/*.svh bound by the TB (never in-module); zero new lint waivers; add to l3 Makefile RTL_CORE; header cites B1_WALKER.md + D-006/§5 + the golden.

---


## 4. KVQ AXI-Lite master requirements

## KVQ AXI-Lite master — what the walker must implement

### 1. Register map (kvq_engine.sv:184-202) and the walker's needed subset

Full engine map (8-bit AXIL window, word-aligned exact-match decode; misaligned/unknown → write no-op + still acked, read 0xDEADBEEF — kvq_engine.sv:928,972):

| addr | reg | R/W | walker v1 relevance |
|---|---|---|---|
| 0x00 | CTRL | W-only decode (:915-918); readback :949 | bit0 = 1-cycle soft-reset pulse (D-020 abort), bit1 = ctrl_enable. **Side-effectful — walker must NOT touch** (enable is host phase_a, `gen_l3_vectors.py:352` `kvw(KV["CTRL"],0x2)`). |
| 0x04 | STATUS | RO (:953-954) | `[0] idle && !cqv_busy && !kp_busy`, `[1] RD_ERR` (IRQ_STATUS[0] mirror), `[2] rsvd0`, `[3] sram_full`. **The walker's poll target.** |
| 0x24 | OCCUPANCY | RO (:962) | host-only (store_kv close-out `:549`). Not in walker scope. |
| 0x28 | WRITE_ADDR | RW (:919, readback :963) | latch only, no side effect. **Not needed in v1** (store_kv/q-inject are out of scope per B1_WALKER §8 row 3). |
| 0x2C | READ_ADDR | RW (:920-923, readback :964) | **THE walker write target.** Write latches `read_addr <= wdata[ADDR_WIDTH-1:0]` AND pulses `read_req` for 1 cycle — "writing READ_ADDR launches a read". CONFIRMED at kvq_engine.sv:920-923. |
| 0x30 | KV_SELECT | RW (:924) | latched bit with **no consumer** anywhere in the engine (only readback :965) — dead config, ignore. |
| 0x34/0x38 | IRQ_MASK / IRQ_STATUS | RW / W1C (:925-927) | host-only. IRQ_STATUS write is W1C with same-cycle-set-wins. |
| 0x08-0x20, 0x3C-0x48 | INFO_* | RO | INFO_TIER 0x0C used by host phase_a check (`:353`); walker doesn't need INFO. |

**READ_ADDR side-effect semantics (the critical one):** the `read_req` pulse (:922, auto-cleared every cycle at :907) is consumed ONLY in `ST_IDLE` (:676-682). If it fires while the FSM is in any other state it is **silently dropped — no error, no sticky**. The only in-IDLE protection is D-016(b): a coincident accepted s_axis beat defers it via `read_pending` (:666). This is why the host's idle-poll-before-RADDR is functionally mandatory, not politeness: an RADDR write issued while the engine is mid-burst or mid-store loses the read and wedges the downstream `fjob` waiting for D beats that never come. RD_ERR (STATUS[1]) only fires for an *unwritten address* timeout (RD_WAIT_MAX=8, :818-822), NOT for a dropped request.

Address width: `wdata[ADDR_WIDTH-1:0]` where ADDR_WIDTH=$clog2(SRAM_DEPTH); the L3 TB builds KVQ_DEPTH=256 (tb_apex_l3.sv:52,157) → 8 bits. At CQ-8, K record t is at address t, V record t at T+t (score_phase `kvw(KV["RADDR"], t)` gen_l3_vectors.py:393; pv_phase `kvw(KV["RADDR"], T+t)` :428). KV addr dict: gen_l3_vectors.py:110-111.

### 2. AXI-Lite handshake the slave expects (kvq_engine.sv:889-976)

**Write channel — AW and W must be presented TOGETHER.** `wr_accept = awvalid && wvalid && (!bvalid || bready)` (:889); `awready = wvalid && (!bvalid||bready)` (:890), `wready = awvalid && (!bvalid||bready)` (:891). Each ready is combinationally gated on the *other* channel's valid, so a master that raises AW alone will spin forever. Single-flight: the next write is not accepted until the previous BRESP is retired. `bvalid` rises the cycle after accept (:911-913); `bresp` hardwired OKAY (:892). With `bready` held 1 (the TB does, tb_apex_l3.sv:804), one write = accept cycle + 1 bresp cycle.

**Read channel:** `arready = !rvalid || rready` (:937); `rvalid`+registered `rdata` the cycle after AR accept (:946-948); `rresp` OKAY. Single-flight. With `rready` held 1, one read per 2 cycles.

**Write and read CAN overlap** — independent always blocks (:894 vs :940), no shared state. But the walker's poll→write sequence is inherently serial (data dependency), so overlap buys nothing in v1; keep the master single-transaction for simplicity. Zero wait states beyond the single-flight gating: no other stall source exists in the slave.

**Bank-level caveat:** the tier mux fans valid/ready per engine combinationally (apex_kvq_bank.sv:135-163); non-selected engines see valid=0, so a transaction issued while `tier_sel` moves would be torn. The walker must only master the bus while the tier select is static (it is, in v1 — see §3).

### 3. How apex_top routes kv_* to engine[tier]

`kv_*` top ports (apex_top.sv:146-161) go straight into `apex_kvq_bank u_kvq` (apex_top.sv:427-468, file rtl/top/glue/apex_kvq_bank.sv). The bank instantiates three UNMODIFIED kvq_engine's — e0 TIER=0/CQ-8, e1 CQ-4, e2 CQ-4+ (:168-215) — behind a combinational mux on `tier_sel`:
- `idx = (tier_sel==3) ? CQ4 : tier_sel` (:132-133, defensive clamp);
- per-engine input gating `e_awvalid[g] = axil_awvalid && sel[g]` etc. (:136-147); output muxes `axil_awready = e_awready[idx]` etc. (:150-163). Non-selected engines see valid=0 / ready=0.
- `tier_sel` comes from apex_top: `live_tier = csr_tip_override ? auto_tier[rt_tip_blk] : csr_tier_sel` (apex_top.sv:412); auto_tier is written by accepted TIP decision beats (:404-410).

Walker v1 consequence: a walk runs only with descriptor tier==TIER_CQ8 (A-1 refusal gate, B1_WALKER.md §A-1), and the host sets `TIER_CTRL=0` in phase_a, so during a walk the mux statically routes the walker's master to **engine 0**. The blkmap/tip_override mid-phase route flips (score_phase gen_l3_vectors.py:386-388) occur only in `tip_auto_mixed`, which is a refused grouped-tier case — the walker never masters the bus under a moving tier select. Each engine has its own SRAM/CTRL, so the walker's RADDR writes land in e0's private address space (the same one store_kv filled).

Also note the read-data drain path the walk depends on: score/pv routes set `fsrc=1` → `fq_in_* = kv_m_t*` and `kv_m_tready = rt_feeder_src && fq_in_ready` (apex_top.sv:500-503) — the feeder is the burst sink; the engine's idle can't return until the feeder accepts all D beats + the ST_OFLUSH tlast beat (D-007, kvq_engine.sv:856-864).

### 4. The host poll idiom, exactly, and the hardware guarantee

Host (score_phase, gen_l3_vectors.py:391-394; identically pv_phase :426-429):
```
for t in chunk:
    KVP STATUS mask=0x1 exp=0x1     # poll engine idle (kvp -> tb kv_poll, tb_apex_l3.sv:763-774:
                                    #   AXIL read of 0x04, compare, 20-cycle backoff, retry)
    KVW RADDR t (score) / T+t (pv)  # AXIL write of 0x2C -> launches the record read
```
1,409 KVW + 1,412 KVP at calib_d64_T128 (B1_WALKER.md §8 "Unscoped work"). The poll is also used as a phase fence (final_phase :439, store_kv :548).

**What the hardware equivalent must guarantee (the invariant, not the idiom):**
1. **Never issue an RADDR write unless a STATUS read has returned bit0==1 with no intervening engine activity.** Because `read_req` is silently dropped outside ST_IDLE (§1), this is a correctness requirement, not throughput hygiene. Since the walker is the only agent touching the bank during a walk, "poll returned 1, then write" is race-free by construction (no s_axis traffic in score/pv, so the D-016(b) coincidence branch can't fire either).
2. The poll ordering also guarantees **burst serialization**: STATUS[0] = `idle && !cqv_busy && !kp_busy` (:954), and idle stays 0 until the previous record's final fp32 beat is accepted downstream — so consecutive RADDR writes can never overlap bursts, and the per-token `efs` (s_k[t]) tap beats stay in record order.
3. **Do not "optimize" the poll into snooping `kv_m_tlast`** acceptance + fixed delay: that bakes FSM latency (ST_OFLUSH→ST_IDLE→idle-reg update = 2 cycles after tlast accept) into the walker and breaks if the engine changes. Implement a real AXIL read poll; back-to-back polling (no 20-cycle backoff) is fine — the bus is otherwise idle, each poll costs 2 cycles.
4. Optionally sample STATUS[1] (RD_ERR) in the same poll read and trap to `WALK_ERR_SEQ` — a template-generated address can't legally be unwritten, so RD_ERR during a walk means a walker bug; failing loudly beats the host-visible symptom (fjob hang).
5. KVP lines are NOT in the Acceptance-A drive subset (B1_WALKER.md §4 lists KVQ-addr, not polls), so the walker may poll more or fewer times than the host without breaking trace equivalence — only the KVW RADDR sequence must be bit-exact and in-order.

### 5. Bus mastering / muxing in apex_top

The walker's AXIL master is internal; in walker mode the `walk_en` mode mux (apex_top edit (b), B1_WALKER.md §B-1/§7) must select {walker master → u_kvq.axil_*} and hold off the external `kv_*` ports (park external awready/wready/arready at 0 — the host contract already forbids touching the bank mid-walk, and the temporal split is clean: host does phase_a/loader/store_kv/q-inject, WALK_GO covers score+pv, host resumes for final_phase). Do NOT edit kvq_engine.sv or apex_kvq_bank.sv — the mux lives entirely in apex_top glue (kvq_engine.sv is B2's collision surface).

### 6. State-machine sketch (walker KVQ master, per record read)

Inputs from walk template: `addr` (= t for K in score, T+t for V in pv). Master holds `bready=1`, `rready=1` permanently.

```
KV_IDLE      : wait for walker core "read record <addr>" request
KV_POLL_AR   : arvalid=1, araddr=0x04            ; on arready -> KV_POLL_R
               (arready = !rvalid||rready, so 1 cycle when quiet)
KV_POLL_R    : on rvalid: if (rdata[1]) -> WALK_ERR_SEQ trap        # RD_ERR
                          else if (rdata[0]) -> KV_WR               # idle
                          else -> KV_POLL_AR                        # busy, re-poll
KV_WR        : awvalid=1, wvalid=1 (SAME cycle — slave requires both),
               awaddr=0x2C, wdata=addr           ; on (awready&&wready) -> KV_BRESP
KV_BRESP     : on bvalid (bready held 1) -> KV_DONE
KV_DONE      : notify walker core (RADDR issued; engine now walking
               ST_RLOAD->ST_RWAIT->ST_OUTPUT(D beats into feeder)->ST_OFLUSH);
               -> KV_IDLE
```
Cost: poll 2 cycles/iteration + write 2 cycles; the engine needs ~D+5 cycles per record before the next poll succeeds, so the master is never the bottleneck. Soft-reset (`CTRL.soft_reset` walk abort, WALK_ERR_ABORT) must return the master to KV_IDLE only at a transaction boundary — never deassert awvalid/wvalid/arvalid before the handshake completes (AXI), and never retract bready/rready.

v1 scope check: the master needs exactly TWO transactions types — READ 0x04 and WRITE 0x2C. WADDR/CTRL/FLUSH/OCC stay host-driven (store_kv/q-inject/phase_a are out of walker scope; CSR FLUSH 0x28 is a tile-CSR, not this bus, and grouped-only anyway).

Key files: rtl/kvq/kvq_engine.sv (:184-202 map, :889-932 write ch + :920-923 RADDR side effect, :937-976 read ch, :658-683 ST_IDLE read_req consumption, :953-954 STATUS); rtl/top/glue/apex_kvq_bank.sv (:132-163 mux, :168-215 engines); rtl/top/apex_top.sv (:146-161 ports, :412 live_tier, :427-468 bank inst, :500-503 kv_m→feeder); verif/top/l3/gen_l3_vectors.py (:110-111 KV dict, :391-394/:426-429 poll idiom); verif/top/l3/tb_apex_l3.sv (:435-465 kv_wr/kv_rd, :763-774 kv_poll, :804 bready/rready=1).

---
