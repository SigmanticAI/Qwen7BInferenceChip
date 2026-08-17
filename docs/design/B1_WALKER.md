# B1 — Hardware Layer-Walker (L-T7) + On-Tile Scale Composition

**Status:** ✅ **COMPLETE — stages 0-6 landed 2026-07-21 (ARCHITECTURE D-028).**
L3 bit-exact in BOTH host and walker mode. Gates: `make -C verif/top/l3 all`
(host byte-identical + walker-mode + 3/3 tile mutants) and
`make -C verif/seq_walker all` (unit suites + 4/4 mutants).
Follow-on: **B1b** — the commit-time amax pass that lets grouped tiers walk.
**Read §8 first if you have read v1 of this doc** — three of its load-bearing
claims were wrong, and one of the stage-0 corrections was itself wrong and has
been reverted (§2 timing / §4 Acceptance A).
**Backlog:** `docs/OPTIMIZATION.md:38` (Tier B B1, "large") and `:63` (rank-1
overall). **Goal metric (`:63`):** MMIO transactions per decode step ~300 → ≤3,
"L3 bit-exact in walker AND host mode."
**Contract discipline:** `apex_pkg.sv` stays FROZEN — `APEX_VERSION 0x0001_0000`
(`rtl/apex_pkg.sv:14`), `mxe_desc_t` layout unchanged (`rtl/apex_pkg.sv:38-52`).
B1 adds a *separate* module + a tile-window CSR extension in glue; it edits
neither `apex_pkg` nor the verified `csr_regs`/`seq_walker` internals-as-frozen
(see §3).

---

## 1. What the block does & where it sits

Today `apex_top` is **host-sequenced** (`rtl/top/apex_top.sv:16-39`): the host
drives every MXE descriptor (`ds_*`), every glue job command
(`fj/qj/dj/lj/aj/wj_*`, ports `apex_top.sv:195-225`), every route level
(`rt_*`, `:227-238`), all KVQ AXI-Lite addressing (`kv_*`), and — the expensive
part — the **per-token seam scale composites** (`qs_*/cs_*`, `:187-193`), which
it computes by reading the fp16 scale taps (`fs_*/ss_*`, `:240-248`) and folding
them on the host. That is the O(T) round-trip term `docs/OPTIMIZATION.md:86`
names ("only on-tile scale composition (inside B1) removes the term").

B1 makes the tile's sequencer **walk the decode step autonomously** from a
compact **layer descriptor** loaded once, replacing the token-by-token host
control stream. Two coupled pieces:

1. **Walker FSM** — a **NEW sibling module** alongside `rtl/seq/seq_walker.sv`
   (today a *descriptor-queue* walker: host pushes `ds_desc`, FSM serializes to
   MXE, `seq_walker.sv:55-216`). *v1 said "extends `seq_walker.sv`"; stage 0
   revised this.* The emit-mode module drives `ds_*` exactly as the host does,
   so `seq_walker.sv` stays **bit-identical**, its D-006 FSM keeps enforcing job
   serialization for the walker too, and host-mode regression risk is zero by
   construction. From a layer descriptor it emits the
   same ordered `md_*` (MXE descriptor) stream **plus** the glue-job / route /
   KVQ-address control the host emits today — the phase templates
   `phase_a → loader → store_kv → q-inject → score → pv → final`
   (`verif/top/l3/gen_l3_vectors.py:331,461,505,552-576,360,404,434`).
2. **On-tile fp16 composite unit** — reads the tile's own `fs_*/ss_*` scale taps
   and produces `qs_*/cs_*` internally, reproducing the golden's single-f32
   narrowing (S-3) bit-exactly (arithmetic in §2). This removes the per-token
   host round-trip that dominates decode MMIO.

**Data is NOT walked.** Tensor streams (`xw_*` weights, `xa_*`/`xg_*`
activation+gamma, and the `XR/GR/WB` payloads) still arrive from their memory
sources — B1 removes *control* round-trips, not the (DMA/bandwidth-bound) data
movement. The v0.1 boundary note "v0.1 has no on-tile weight memory"
(`apex_top.sv:24`) is unchanged by B1.

**Scope reality (read before estimating).** The full decoder layer
(`golden/apex_golden/transformer.py:12-28`: RMSNorm→QKV→RoPE→KV-compress→attn→
o-proj→SwiGLU→residual ×28) is the north star, but **RoPE / SwiGLU / o-proj /
residual are not yet integrated into `apex_top`** — they are standalone,
composition-verified blocks (`rtl/rope/rope.sv`, `rtl/asu/asu_silu.sv`,
`rtl/misc/residual_add.sv`; PASS in `verif/layer/RESULT.md`). The op stream that
`apex_top` realizes today — and therefore the buildable, non-negotiable B1
target — is the **attention decode step** emitted by `gen_l3_vectors.py`.
Walking the *full* layer end-to-end is gated on those blocks landing in
`apex_top` first (a separate work item, e.g. B5/GQA plumbing); B1 must be
structured so that walk is a descriptor extension, not a rewrite.

---

## 2. The golden reference (already exists — do not add a new one)

Per `docs/OPTIMIZATION.md:38`, **"the L3 choreography is the spec."** The golden
op stream is exactly what `verif/top/l3/gen_l3_vectors.py` emits — the same
script the host runs today. The walker must **autonomously reproduce that exact
ordered op/descriptor stream.** No new golden model; the arbiter is the existing
`golden/apex_golden/attention.py` (via the taps the L3 cases already check).

**Control ops the walker must emit** (the drive-subset of the `Script` emitters,
`gen_l3_vectors.py:194-298`): `DESC` (`:246`), `ROUTE` (`:222-237`),
`AJ`/`WJ` act/wgt-stage jobs (`:263-267`), `FJOB`/`QJOB`/`SJOB`/`LJOB`
(`:258-261`), and KVQ `WADDR`/`RADDR`/`FLUSH` writes (`kvw`, `:217`; used by
`store_kv_phase:505`, `score_phase:391-395`, `pv_phase:424-430`). The per-token
KVQ addressing and chunk counts are pure functions of `(D, T, tier)` —
`chunks(T)` (`:185-191`), `BPR = D//8` — so they are template state, not data.

> **CORRECTION (stage 0, 2026-07-20 — verified, see §8).** The sentence above
> is true for *addressing, counts and geometry* and **false for three control
> payloads.** Purity was assumed, never checked. What is pure (by inspection of the generator — no stage-0 artifact *tests* purity; only the ordering claim below is measured):
> `ROUTE` (all 8 fields), `AJ`/`WJ` (all six), `FJOB`/`QJOB`/`SJOB`/`LJOB`,
> KVQ `WADDR`/`RADDR`/`FLUSH`, the score-phase `DESC`, and the poll addresses.
> What is **not**:
> 1. **`DESC` `rq_scale`/`rq_shift` in `pv_phase` — causally circular.**
>    `gen_l3_vectors.py:411,417-419` takes them from `r.rq`, produced by
>    `calib_requant(amax|acc_o|)` where `acc_o` is the output of *the very
>    P·V GEMM this descriptor configures* (`attention.py:395-398`). No on-tile
>    amax probe exists. **Resolution (D-028 proposal): `rq_scale`/`rq_shift`
>    are LOADED layer-descriptor fields**, host-written per step. Bit-exactness
>    is preserved (same value the script emits); autonomy is partial and named.
>    A truly autonomous PV needs two-pass amax + an on-tile `calib_requant`
>    (`floor(log2)` + RNE multiply) — its own contract, not smuggled in here.
> 2. **`CS`/`QS(S-4)` are data-dependent** on `s_q`, `s_k[t]`, `s_v[t]` — the
>    composite unit's whole reason to exist (§1 item 2). They are *not*
>    derivable from `(D,T,tier)`.
>
> **Timing — measured, then corrected.** `extract_trace.py` over all 27 cases:
> of 1,696 composite words, **0** are emitted after their input appears on the
> `fs_*`/`ss_*` taps (1,664 INVERTED, 32 tap-UNRESOLVED). The host gets away
> with it by precomputing from the golden. **But the taps are not the only
> on-tile source, and an earlier draft of this doc wrongly concluded they
> were.** The KVQ stores a per-record fp16 scale at store time
> (`rtl/kvq/kvq_engine.sv:290`; `scale_bank_store` `:431` for grouped keys),
> and **at CQ-8 that stored scale is bit-identical to the feeder's read-time
> `s_k[t]`/`s_v[t]`** — the row's max |code| is always exactly 127, so
> `RNE_f16(amax/127) == s_record`. Verified: zero mismatches over the suite
> tensors (D=64/128, several T) and 13k random tokens.
>
> **Consequence: a store-time scale cache lets the walker emit every composite
> in the host's exact order.** No reorder allowance is required at CQ-8.
> Grouped tiers (CQ-4/CQ-4+) recompute a *different* read-time scale from the
> dequantized record — store-time determined, but not the stored per-record
> scale — so covering them needs a feeder-equivalent commit-time amax pass.
> That is a hardware-cost decision for stage 1, **not** an impossibility.

**`QS` is two different populations.** `QS` in `store_kv`/`q-inject` is the
per-element `decompose_f16` injection composite (`:489-491`) — testbench
data-path scaffolding, and in real decode these come from the projection
GEMMs. Only the score-phase S-4 folds are walker-owned. In `calib_d64_T128`
that is **128 of 16,576** `QS` words. `extract_trace.py` separates them by
`role` (`inject` vs `s4_fold`).

**On-tile scale composition — the bit-exact numeric contract (S-3/S-4).** The
composite unit must reproduce, exactly:

- **Score composite (cs)** `gen_l3_vectors.py:375-381`:
  `comp64 = s_q · s_k[t] · (float(1<<SCORE_FRAC) / sqrt(D))`, then
  `comp32 = comp64.astype(np.float32)` — **one** f32 narrowing — emitted as
  `cs(int(comp32[t]))`. `SCORE_FRAC` imported from `attention.py`
  (`gen_l3_vectors.py:90`).
- **P-requant composite (qs)** `:382-384`:
  `qs = f32_bits_exact(s_v[t] · 2^-15)` — the S-4 fold, `f32_bits_exact` at
  `:133-140` asserts fp32-exact **and** fp16-grade (`bits & 0x1FFF == 0`) **and**
  positive-normal.
- Inputs `s_q`, `s_k[t]`, `s_v[t]` are fp16 scales the tile already exposes on
  `fs_*`/`ss_*` (feeder / scale_quant taps) and cross-checked at
  `EFS`/`ESS` (`efs:272`, `ess:276`). The composite unit consumes those same
  fp16 values on-tile.

The single-f32-narrowing (S-3) is load-bearing (`docs/OPTIMIZATION.md:38`,
ARCHITECTURE D-021 §9b:243-250). The KV write-port narrowing it must not
perturb is `decompose_f16` (`gen_l3_vectors.py:164-182`, self-checked against
`f64_to_f16_bits`).

**Golden trace for equivalence:** the emitted `build/cases/<name>.txt` files
*are* the ordered transaction trace. A verif-side extractor (NOT a golden edit)
projects each case file to its canonical control-drive subsequence; the walker's
captured emissions diff against it (§4).

---

## 3. Interface contract (what B1 adds/changes)

**Frozen (must not change):** `apex_pkg.sv` — `APEX_VERSION` (`:14`),
`mxe_desc_t` (`:38-52`); the walker still emits `mxe_desc_t` on `md_*`
(`seq_walker.sv:73-75,206-207`). `csr_regs.sv` is instantiated UNMODIFIED
(`apex_top.sv:10`) — do **not** widen its decoded map.

**New — layer descriptor & control, as a tile-window CSR extension in
`apex_top` glue.** Reuse the exact pattern already used for `ERR_STICKY 0x58`
(`apex_top.sv:76-87,864-918` — **not** `:864-916`; the `csr_rdata` override mux
at `:917-918` is half the mechanism and the old cite truncated it):
`csr_regs` acks the address as reserved (returns
`0xDEADBEEF`, no side effects, `csr_regs.sv:178`), and `apex_top` glue overrides
read data and captures the write. `csr_regs` decodes only up to `PERF_BUSY_i`
`0x38+4i` (`csr_regs.sv:149,195-197`; `N_BLOCKS≤8` ⇒ max `0x54`) and `0x58`
(ERR_STICKY, glue). **Free base for the walker window: `0x5C+`.** Proposed:

| Addr | Name | Dir | Meaning |
|---|---|---|---|
| `0x5C` | `WALK_CTRL` | RW | `[0]` walk_en (autonomous mode), `[1]` walk_go (kick, self-clearing) |
| `0x60` | `WALK_DPTR` | RW | descriptor-SRAM write pointer (auto-inc on DDATA write) |
| `0x64` | `WALK_DDATA` | RW | descriptor-SRAM data word (load the compact layer descriptor once) |
| `0x6C` | `WALK_RQ` | RW | PV requant pair, direct write — the only per-STEP descriptor field |
| `0x68` | `WALK_STATUS` | RO/W1C | `[0]` walk_busy, `[7:1]` phase/step, `[8]` walk_err sticky (W1C), `[11:9]` walk_err_code |

`walk_err_code` (`WALK_STATUS[11:9]`) — a refusal must say *why*, so an
unsupported configuration is never confused with a walk that went wrong:

| code | meaning |
|---|---|
| `0` | no error |
| `1` | **`WALK_ERR_TIER`** — descriptor `tier != TIER_CQ8`; walk refused, host mode retained (§A-1 gate) |
| `2` | `WALK_ERR_DESC` — descriptor field out of range (`D`, `T`, geometry) |
| `3` | `WALK_ERR_SEQ` — internal sequencing fault (should be unreachable; SVA-guarded) |
| `4` | `WALK_ERR_ABORT` — walk terminated by `CTRL.soft_reset` mid-flight |

Descriptor fields (walker-local struct in a NEW `rtl/seq/seq_walker_pkg.sv` —
**not** `apex_pkg`, so no version bump): `D`, `T`, `tier` (`kvq_tier_e`),
`outlier_k`, chunk/group geometry, phase-enable mask, **plus `rq_scale[15:0]`
and `rq_shift[4:0]`** — the PV requant pair, host-loaded because it is not
computable on-tile (§2 correction 1). Naming them as descriptor state is the
honest encoding of that limit: the walker is autonomous in *sequencing*, and
carries exactly one per-step calibration input it cannot derive.
This mirrors the D-024/B3
principle "route-level + CSR selection keeps `apex_pkg` frozen"
(`docs/OPTIMIZATION.md:40`) and S12's "runtime-selectable, structure frozen"
(`docs/design/S12_LOADABLE_MASK.md:13-22`).

**Steady-state MMIO/step (the `:63` metric) — DERIVED, not measured across
steps.** The geometry words (`D`, `T`, `tier`, mask) are per-CONFIGURATION;
only the PV requant pair changes per step (§2 correction 1). So a steady-state
decode step costs exactly **3**: `WALK_RQ` write + `WALK_CTRL.go` write +
`WALK_STATUS` done-poll. Cold start (a new `(D,T,tier)`) additionally costs the
one-time descriptor load: `WALK_DPTR` + 3×`WALK_DDATA` = 4 more.

> **Honesty note.** The L3 harness cannot MEASURE the steady-state figure: each
> case is a single decode step, so every walker-mode run is a cold start (6
> WALK-window transactions, counted and printed per case). The **3** above is
> read off the register interface, not observed across consecutive steps.
> `WALK_RQ` (0x6C) exists precisely because without it the per-step cost is
> **4** — routing the one per-step field through the `DPTR`/`DDATA` pointer
> window costs an extra transaction, and missing the `:63` target by one while
> claiming to have hit it is exactly the kind of number this project does not
> ship.

**Mode mux (fallback is non-negotiable, `:38`).** `WALK_CTRL.walk_en=0` ⇒ host
mode: `ds_*` / glue-job / `rt_*` / `qs_*`/`cs_*` behave exactly as today (the
verified path). `walk_en=1` ⇒ the walker owns those internal control fanouts;
external host-driven `ds_*`/job/route/seam inputs are held off. The mux lives in
`apex_top` glue; `seq_walker`'s host `ds_*` port and D-006 FSM (`seq_walker.sv:
118-203`) are preserved as the fallback engine.

**Preserved SEQ contracts (must still hold in walker mode):** D-006 job
serialization (`ARCHITECTURE.md:216`, `seq_walker.sv:15-25`), `done`⇒post-skid
acceptance (`§5:131-143`), soft-reset abort/drain (`seq_walker.sv:26-52`,
D-020 `ARCHITECTURE.md:230`), busy honesty (`seq_walker.sv:213-214`). CSR
`CTRL.soft_reset` must abort a walk mid-flight and drain the in-flight MXE job
(never violate §5).

---

## 4. Verification target

**Acceptance A — transaction equivalence (the non-negotiable, `:38`).** For each
L3 manifest case, the walker in autonomous mode (loaded with that case's compact
descriptor) emits a control trace whose canonical drive-subsequence
(`DESC`/`ROUTE`/`AJ`/`WJ`/`FJOB`/`QJOB`/`SJOB`/`LJOB`/KVQ-addr/`FLUSH` + the
`qs`/`cs` composite words) is **bit-exact vs the extracted drive-subset** of
`build/cases/<name>.txt`, **bit-exact and in-order**, globally.

**Keep the strong relation.** An earlier stage-0 draft weakened this to
per-channel in-order after measuring that no composite is emitted after its
input appears on a tap. That measurement is real but its scope is narrower
than the conclusion drawn from it: the KVQ's store-time record scale is
bit-identical to the feeder's read-time scale at CQ-8 (§2), so the design that
preserves strict order is available and is the one to build:

- **On-tile fp16 scale cache**, written when a record is stored, read when a
  composite is due. Sizing: 2·T·16 b = **4 Kbit at T=128** — negligible. This
  is also the structurally right shape, since in real decode the K/V for
  earlier tokens were stored on earlier steps.
- At **CQ-8** this reproduces the host's emission order exactly, with zero
  added transactions.
- At **CQ-4/CQ-4+** the read-time scale is recomputed from the dequantized
  record and differs from the stored per-record scale.

> Rationale for holding the line: a weakened relation is a permanent loss of
> gate strength, and it was proposed on a false premise. Weaken only against a
> demonstrated hardware cost, tier by tier, never as a convenience.

### A-1. Tier scope of Acceptance A — DECIDED (2026-07-21, owner call)

**Walker v1 is bit-exact and strictly in-order at CQ-8 only.** CQ-8 is the tier
the real-model S8 artifact actually ran, so the walker milestone stays on the
verified path. The commit-time amax pass that would extend strict order to the
grouped tiers is **deferred to a named follow-on stage (B1b, §5)** — bounded
and honest, not abandoned. KVQ-4 quality has its own track through S12/D-027.

**The relaxation is a GATE, not a waiver.** It is scoped, enumerated, and must
fail loudly if it ever widens:

| tier | walker v1 obligation | enforcement |
|---|---|---|
| **CQ-8** (`TIER_CQ8`) | full Acceptance A: bit-exact **and strictly in-order** vs the extracted trace, zero added transactions | walker-mode L3 over every CQ-8 case |
| **CQ-4 / CQ-4+** (`TIER_CQ4`, `TIER_CQ4P`) | **walker mode is REFUSED, not degraded** | see the hard gate below |

**Hard gate (stage 1 RTL obligation).** `walk_en=1` with a descriptor whose
`tier != TIER_CQ8` must **refuse to start**: no walk, `WALK_STATUS.walk_err`
set, and the tile falls back to host mode. It must NOT silently run a relaxed
walk. Rationale: an unsupported tier that *quietly* produces a differently
ordered stream is exactly the silent narrowing this decision forbids — the
failure has to be visible at the CSR, in the TB, and in the case log.

**Verification obligation.** The walker-mode L3 run must:
1. cover **every CQ-8 case** in the manifest at full strict-order equivalence;
2. assert the refusal path on the grouped-tier cases (`adv_T1_cq4p`,
   `adv_outlier1000_cq4p`, `tip_auto_mixed`) — walk refused, `walk_err` set,
   host-mode fallback still bit-exact;
3. **print the tier split explicitly** in the run log (`walker: N/27 cases
   walked (CQ-8), M/27 refused (grouped tier, B1b)`), so the coverage limit is
   legible in every artifact rather than inferable from a doc.

A future change that lets a grouped tier walk **must** update this table, the
refusal gate, and the printed split together. If the printed split ever shows a
grouped-tier case as "walked", the gate has been breached.

**Acceptance B — L3 bit-exact in BOTH modes (`:63`).** The full L3 suite passes
byte-identically in host mode (regression: prove B1 changed nothing) AND in
walker mode. Pass criterion is mechanical: `run_cases.py` requires `"L3 PASS"`
in stdout and `rc==0` per case (`verif/top/l3/run_cases.py:27`). All **27 cases
pass**, `bug_d128_stagebuf_nb` included — despite its name it is
`kind: f5-fix-regression`, `expect: pass`, `fail_sig: null`: it pins a *fixed*
wedge, and a regression **to** the old fail signature means the fix broke
(`gen_l3_vectors.py:60-62`). `run_cases.py`'s `expect: bug` machinery
(`:29-41`) exists but **no current case uses it** — do not assume an expected
failure in the suite.

Two check counts exist and they are **not** interchangeable — quote whichever
you mean:
- **135,133 scripted** — `sum(manifest.json cases[].checks)`, printed by
  `gen_l3_vectors.py` as "scripted checks". Regenerate to reproduce.
- **135,241 simulated** — the simulator-emitted total (`coverage_report.py`
  sums the per-case `L3 RESULT: checks=` lines; recorded in `l3_rerun.log`).
  This is the figure `docs/OPTIMIZATION.md:25` pins. `coverage_report.py`
  enforces `simulated >= scripted`, not equality. The 108-check delta is
  **exactly 4 per case × 27 cases** — the four leftover-queue checks in the
  TB's `DONE` handler (`tb_apex_l3.sv:1006-1009`: leftover `fs`/`ss`/`tip`/`ro`),
  which have no scripted counterpart. It is *not* the tap counters: `tap()`
  already adds `len(vals)` to `n_expect`, so those net to zero. Verified: the
  per-case delta is 4 for all 27 cases.

Envelope: `ARCHITECTURE.md` §11 (`:276-372`, not `:276-324`).

### A-2. Scale source — DECIDED (2026-07-21, stage-1 design panel)

Strict order (§A-1) needs `s_k[t]`/`s_v[t]` *before* the score phase reads
them. `cqv_scale` — the KVQ's stored per-record fp16 scale — is an **internal
wire** (`rtl/kvq/kvq_engine.sv:255` decl, `:279` produced, `:290` consumed into
the SRAM record image). It is on **no module port**. Four options were built
out and judged on correctness, blast radius and goal-metric preservation:

| option | mechanism | verdict |
|---|---|---|
| A | additive record-commit observation port on `kvq_engine` | feasible, **rejected on blast radius** |
| **B** | **walker snoops the existing `kv_s_*` fp16 store stream and recomputes the scale with `cq_fp_pkg::scale_from_amax`** | **CHOSEN** |
| C | walker-issued KVQ scale-harvest read pass | **rejected** — 320 added control ops, fails Acceptance A by construction |
| D | host-loaded scale vectors in the descriptor RAM | **rejected** — 259 MMIO/step at T=128 vs the ≤3 goal; the exact anti-pattern `OPTIMIZATION.md:86` names |

**Chosen: Option B.** The walker taps `kv_s_tvalid/tready/tdata/tlast`
(`apex_top.sv:389-390` — the same nets behind the existing passive
`dbg_f16_*` mirror, `:921-923`), accumulates a 15-bit masked `amax` over each
record's D fp16 beats, and derives the scale with a serialized instantiation of
**`cq_fp_pkg::scale_from_amax`** (`rtl/kvq/cores/cq_fp_pkg.sv`) — the *same
package function the KVQ engine itself uses*. Result is cached in a B1-owned
2·T×16 b scale RAM (**4 Kbit at T=128**).

*Why not A, which scored higher on pure correctness (it fans out already-stored
bits and adds no arithmetic):* A touches `kvq_engine.sv` — the B2 lane's file
and one of the "three verified, UNMODIFIED engine instances" D-024 rests on —
plus `apex_kvq_bank.sv`, a **fourth** `apex_top` region, and PINMISSING fixes in
six B2-lane TBs under fatal `-Wall`. That is a structural cross-lane change
requiring B2 sign-off, against a coordination baseline of *trivial adjacent-line
merges*. Option B needs **zero edits outside B1's three budgeted `apex_top`
regions** and cannot regress a verified block.

*Correctness basis:* function identity by **shared construction**, not
re-derivation — the walker imports the same `cq_fp_pkg` function over
bit-identical operands (`dbg_f16_*` mirrors accepted beats only, and the engine
assembles exactly those beats, `kvq_engine.sv:669-690`). Stage 3 proves it
bit-exact against `walker_composite_golden.py`. **If stage 3 shows any numeric
gap, Option A is the named fallback** and the cross-lane cost gets paid then.

*Residual risks to handle in RTL:* record-index association is positional
(K at `t`, V at `T+t`) and assumes the L3 store order — trap a `tlast` at
beat ≠ D−1 into `walk_err` rather than mis-associating; mirror D-020 soft-reset
token abort (clear `amax`/beat count, do not commit); and a `WALK_GO` before the
cache is fully populated must raise `WALK_ERR_SEQ`, never silently use stale
scales (per-entry valid bits).

### B-1. Tap observability — DECIDED (2026-07-21, owner call): build the mirror

`tb_apex_l3.sv:342-343` hardwires `assign fs_ready = 1'b1; assign ss_ready =
1'b1;` — the TB is the tap sink, and every `EFS`/`ESS` check depends on it
(**1,153 `EFS` + 2 `ESS`** in `calib_d64_T128` alone). In walker mode the
composite unit becomes that sink, which would delete those checks.

**A reduced check count at stage 5 is NOT acceptable.** Those checks sit
exactly where the walker changes behaviour — composite emission — so dropping
them would remove observability precisely where the new logic is least proven.
"L3 bit-exact in walker mode with an unchanged check count" is the acceptance.

> **RESOLVED AT STAGE 4 — the mirror proved UNNECESSARY, and the budget is
> returned, not quietly spent.** This section was written against a
> composite unit FED BY THE TAPS. §A-2 then chose the store-snoop scale
> source, so the built unit never *consumes* `fs_*`/`ss_*` as a stream: it
> takes `s_q` by observing `ss_valid && ss_ready` **passively, without taking
> ready**, from inside the glue. The TB therefore remains the tap sink and
> **every `EFS`/`ESS` check survives walker mode unchanged** — the outcome
> this section demanded, reached without the port. `apex_top` gets **two**
> fenced edits, not three, and no new ports. Verified at stage 4: host-mode
> L3 is byte-identical on cycles, checks and errors across all 27 cases.
>
> Two decisions interacted: had §B-1 been implemented before §A-2 was
> settled, the mirror would have been built and then carried forever as dead
> observability. The original reasoning is kept below because the *principle*
> — a reduced check count at stage 5 is not acceptable — still governs.

**Budgeted now (superseded, see above):** a **passive accepted-beat mirror** for `fs_*`/`ss_*` in the
`apex_top` glue, copying the existing pattern for
`dbg_f16_*`/`dbg_sc_*`/`dbg_pr_*` (`apex_top.sv:277-285,921-929`) — new output
ports `dbg_fs_v/data/last` and `dbg_ss_v/data/last`, no flow control, no effect
on the tapped streams. The TB sinks the mirror instead of the tap when
`+walker` is set, so `expect_fs`/`expect_ss` and the leftover-queue checks are
unchanged in both modes.

This makes **three** fenced `apex_top` edits, not the two §7 promised: WALK CSR
window, walk_en mode mux, and this mirror. All three are additive; the mirror
is pure observation (no `ready`, no backpressure), so it cannot perturb
host-mode behaviour — which the stage-4 byte-identical host-mode L3 run proves.

**TB pattern to follow — copy `verif/seq/` verbatim in structure** (the
closest, purpose-built template): vector-driven scoreboard against an
**independent golden queue/trace model** (`gen_seq_vectors.py:4-30`), a
**D-006-contract MXE/glue stub** with randomized latency/backpressure
(`mxe_stub.sv`), house §5 stream SVA (`apex_stream1_sva.svh`) + SEQ-contract SVA
(`seq_sb_sva.svh`), phase-targeted mid-op resets, manual coverage buckets, and a
**mutation-kill gate** (`verif/seq/Makefile` `mutants:`, mutate.py). For the
tile-level mode, follow `verif/top/l3/` (`tb_apex_l3.sv` op parser
`:840-1010`; build matrix `-GCFG_D=64/128` `Makefile:85-91`).

**Mutation-gate expectations.** New unit-TB mutants that MUST be caught: (m1)
walker emits `DESC` for job N+1 before job N `done` (D-006 violation); (m2)
composite unit drops the S-3 f32 narrowing (uses f64) — must break the `cs`/
downstream `ERO`/`ESS` checks; (m3) a route/glue-job emitted out of phase order.
Tile-level: the existing 3/3 gate (`verif/top/l3/Makefile:129-136`) must stay
green with the walker path compiled in.

---

## 5. Staged landing plan (golden-first, machine-aware)

| stage | what | machine need |
|---|---|---|
| **0 ✅ DONE** | This doc + `verif/top/l3/extract_trace.py` (case `.txt` → canonical per-channel control trace **with composite provenance** + per-file line accounting; no golden edit) + `verif/top/l3/walker_composite_golden.py` (§2 arithmetic replica + self-test **with a coverage gate**). **Result: replica PASS — 1,664 composite words bit-exact vs the L3 op stream across all 27 cases (677 distinct `(s_q,s_k,D)` triples, 610 distinct `s_v`), 5/5 arithmetic mutants killed, coverage gate proven to fire.** Outputs: the §2 scale-observability analysis and the §8 correction table. | none |
| **1** | `rtl/seq/seq_walker_pkg.sv` (descriptor struct, incl. the loaded `rq` pair) + emit-mode walker + fp16 composite unit **with a store-time scale cache** (2·T·16 b, §4) — the cache is what buys strict-order equivalence, so it is stage-1 core, not an optimisation. Keep host-mode fallback intact. **Structure:** a NEW sibling module driving `ds_*` exactly as the host does, *not* a rewrite of `seq_walker.sv` — that leaves the D-006 FSM bit-identical and makes host-mode regression risk zero by construction. Scope = **score+pv only** (3,056 of 24,678 transactions at `calib_d64_T128`; store_kv/q-inject are injection scaffolding, §2). Decide and record the grouped-tier scale question (§4) here. | edit only |
| **2 ✅ DONE** | **Unit TB** `verif/seq_walker/` (copy `verif/seq/` pattern): walker-vs-extracted-trace equivalence over the manifest cases, glue/MXE stub, §5+SEQ SVA, ≥2 mutants (m1/m2) | Verilator (serialize) |
| **3 ✅ DONE** | Composite-unit unit TB: on-tile `cs`/`qs` bit-exact vs the §2 golden replica across the L3 scale corners (`adv_outlier1000`, `calib_d64_T128`, `d128_T100`); +mutant m2 | Verilator (serialize) |
| **4 ✅ DONE** | `apex_top` glue, **three fenced additive edits**: (a) `WALK_*` window (0x5C, ERR_STICKY pattern `:864-918`), (b) `walk_en` mode mux, (c) **passive `fs_*`/`ss_*` accepted-beat mirror** (`dbg_fs_*`/`dbg_ss_*`, §B-1 — budgeted, not optional). Gate: **host-mode L3 byte-identical** (compat proof, S10-style) — the mirror has no `ready` so it cannot perturb the tapped streams. | Verilator (serialize) |
| **5 ✅ DONE** | Walker-mode `tb_apex_l3` (`+walker` plusarg: control from walker, data still from file; TB sinks the §B-1 mirror so `EFS`/`ESS` counts are unchanged). Gates: full L3 bit-exact in walker mode at **CQ-8, unchanged check count**; grouped-tier cases prove the **refusal path** (§A-1); run log prints the tier split; tile mutation gate 3/3. Adds the `WALKER=1` Makefile target (does not exist today, §8 row 5). | Verilator (serialize) |
| **6 ✅ DONE** | ARCHITECTURE §9 decision entry (propose **D-028**, incl. the §A-1 tier scope and the §B-1 mirror) + §11/STATUS rows + task-board close | none |
| **B1b** *(named follow-on, not v1)* | Commit-time feeder-equivalent amax pass so **CQ-4/CQ-4+** also walk in strict order; removes the §A-1 refusal gate. Bounded: one amax accumulator + `calib_requant`-grade fp16 RNE at record commit. Pairs with S12/D-027 KVQ-4 work. | Verilator (serialize) |

Stages 2–5 need the Verilator machine — **serialize** them behind the shared
build queue. Stages 0/1/6 are edit-only and can proceed in parallel with the
other two blocks.

---

## 6. Files to read / touch & repro

**Read (do not modify):**
`docs/OPTIMIZATION.md` (38,63,25,86) · `ARCHITECTURE.md` §1(40)/§5(131-143)/§9
(216,229,230)/§11(276-324) · `rtl/seq/seq_walker.sv` · `verif/top/l3/
gen_l3_vectors.py` · `golden/apex_golden/attention.py` + `transformer.py` ·
`rtl/csr/csr_regs.sv` · `rtl/apex_pkg.sv` · `rtl/top/apex_top.sv` (76-80,
166-286, 864-916) · `verif/seq/` (whole dir — the TB template) ·
`verif/top/l3/{tb_apex_l3.sv,run_cases.py,Makefile,mutate.py}`.

**Touch (new, own paths — see §7):** `rtl/seq/seq_walker_pkg.sv` (new),
`rtl/seq/seq_walker.sv` (emit-mode extension), one new composite-unit RTL file
under `rtl/seam/` or `rtl/seq/`, `apex_top.sv` glue (WALK window + mux only),
`verif/seq_walker/` (new TB dir), `verif/top/l3/extract_trace.py` (new),
`+walker` path in `verif/top/l3/tb_apex_l3.sv`.

**Repro (from repo root):**
```
# golden gate first (arbiter untouched):
make -C golden test
# stage 0 (WORKS TODAY, ~2 s, no Verilator):
make -C verif/top/l3 vectors                       # 27 cases, deterministic
python3 verif/top/l3/extract_trace.py verif/top/l3/build
python3 verif/top/l3/walker_composite_golden.py verif/top/l3/build
# walker unit TB (stage 2/3):
make -C verif/seq_walker
# host-mode L3 compat + walker-mode L3 (stage 4/5):
make -C verif/top/l3          # host mode, must stay byte-identical
```
```
# stage 5 (LANDED): walker-mode L3
make -C verif/top/l3 run_walker      # 24 walked (CQ-8) + 3 refused, no checks lost
```
> v1 cited `make -C verif/top/l3 run WALKER=1` as the walker-mode gate.
> At the time it **did not exist** — the Makefile's targets are exactly `all lint vectors
> build unit run coverage mutate waves clean` and there is no `WALKER`
> variable anywhere in it or in `run_cases.py`. Stage 5 must *add* that path
> (a `+walker` plusarg in `tb_apex_l3.sv` plus a Makefile target); it is work,
> not an existing command.

---

## 7. Independence notes (parallel build with 2 other blocks)

**Own everything you can.** New module (`seq_walker_pkg.sv`, composite unit) and
a **new TB dir `verif/seq_walker/`** — zero collision.

**Shared-file collision risks & the rule:**
- `rtl/apex_pkg.sv` — **frozen; never edit** (all blocks depend on it; a version
  bump breaks everyone). B1 needs nothing from it.
- `rtl/top/apex_top.sv` — the one true shared file. B1's edits are confined to
  **three additive, clearly fenced regions**: (a) the WALK CSR window,
  (b) the `walk_en` mode mux, (c) the passive `fs_*`/`ss_*` accepted-beat
  mirror (§B-1). Mirror the `ERR_STICKY` glue region `:864-918`. Coordinate the
  merge order with B2 (TIP-EVICT autonomous-retire pairs with B1 per
  `docs/OPTIMIZATION.md:39` — its `kvq_engine.sv:294` stub is a *different*
  file, but it also touches `apex_top` glue) and B3 (W4 route/CSR). Land B1's
  `apex_top` glue in **stage 4** only, as one small reviewable diff; stages 0-3
  are entirely in B1-owned files.
- `verif/top/l3/gen_l3_vectors.py` / `tb_apex_l3.sv` — shared L3 harness. Do
  **not** rewrite; the trace extractor is a *new* sibling file, and the walker
  path in `tb_apex_l3` is a `+walker`-gated addition that leaves the default
  (host) path byte-identical (the S10 compat discipline, `S12:70`).
- CSR address space — B3 (native W4) also wants CSR/route selection. **Reserve
  `0x5C-0x6C` for WALK now** in this doc so B3 picks a disjoint range; both stay
  above the `csr_regs` map (≤`0x54`) and clear of `ERR_STICKY 0x58`.

**Cross-lane shared-file state (2026-07-21).** The B3 lane has landed additive
edits to `scripts/gen_status.py`, `ARCHITECTURE.md` §6, `docs/OPTIMIZATION.md`
(B5 row) and the Level-C hub status. B1 touches several of the same files at
stage 6 (ARCHITECTURE §9/§11, OPTIMIZATION, STATUS rows). Expect **trivial
adjacent-line merges at integration — nothing structural**; both lanes are
appending rows, not restructuring. Resolve by keeping both additions; if a
conflict looks structural, stop and reconcile rather than picking a side.

**Machine contention.** B1's Verilator stages (2-5, and B1b) serialize against
the other lanes on the one 18 GB box — check
`ps aux | grep -cE "[v]erilator|[o]bj_dir/V"` before a heavy build. The
`c6a.xlarge` unlock (`LEVEL_C_PARALLEL.md` §bottlenecks) exists if contended;
stage 0/1/6 are edit-only and never need it.

**Golden is the arbiter, one-directional:** if the walker's emitted stream
disagrees with `gen_l3_vectors`, the walker is wrong — never adjust the golden
op stream to match RTL.

---

## 8. Amendments (stage 0, 2026-07-20) — what this doc got wrong

Every line cite in v1 was re-checked against the source. Corrections, with the
evidence that produced them. **Design changes** (need review): 1, 2, 3.

| # | v1 claim | status | correction |
|---|---|---|---|
| 1 | §2 "pure functions of `(D,T,tier)` … template state, not data" | **FALSE** | `pv_phase` `DESC.rq_scale/rq_shift` is *causally circular* (`gen_l3_vectors.py:411,417-419` ← `attention.py:395-398`); `CS`/`QS(S-4)` need scales the tile has not yet computed. → §2 correction; `rq` becomes a loaded descriptor field. |
| 2 | §4 Acceptance A "bit-exact **and in-order**" | **UPHELD** (a stage-0 draft wrongly proposed weakening it) | 0 / 1,696 composites are emitted after their input reaches a *tap* — but the KVQ's store-time record scale is bit-identical to the feeder's read-time scale at CQ-8 (`kvq_engine.sv:290`; verified, 0 mismatches over suite tensors + 13k random tokens). A store-time scale cache preserves strict order. Grouped tiers need a commit-time amax pass — a cost decision, not an impossibility. |
| 3 | §1/§5 walk `store_kv → q-inject` | **OUT OF SCOPE** | `QS` injection words alone are **66.7 %** of the drive+sync stream (16,448 / 24,678); the whole `store_kv`+`q_inject` phase pair is **87.5 %** (21,593 / 24,678). Buildable target = score+pv: **3,056** ops at `calib_d64_T128`, of which the walker-*emitted* drive subset is **1,902**. |
| 4 | `:169` "pinned figure is 135,241 … reproduce the live manifest total" | **SELF-CONTRADICTORY** | scripted 135,133 ≠ simulated 135,241; both real, different metrics. → §4. |
| 5 | `:234` `make -C verif/top/l3 run WALKER=1` | **DOES NOT EXIST** | no such target or variable. → §6. |
| 6 | `:116` `apex_top.sv:76-80,864-916` | **TRUNCATED** | mechanism runs `:864-918`; header entry `:76-87`. |
| 7 | `:96` `ARCHITECTURE D-021 §9b:243-250` | **DRIFTED** | `:243-250` is the S-4 fold prose; the D-021 row is `:260`; normative S-3 text is `golden/apex_golden/attention.py:34-37,222-240`. |
| 8 | `:170` envelope `ARCHITECTURE.md:276-324` | **DRIFTED** | §11 runs `:276-372`. |
| 9 | `:251` B2's "`kvq_engine.sv:294` stub" | **WRONG LINE** | `:294` is a comment (`reg sram_wr_en;` is `:296`). The eviction stub is `:317-318`. (`docs/OPTIMIZATION.md:39` carries the same bad line.) |
| 10 | §4 "copy `verif/seq/`… `apex_stream1_sva.svh`" | **PATH** | it lives in `verif/common/`, not `verif/seq/`. Both `apex_stream_sva.svh` and `apex_stream1_sva.svh` exist — bind whichever `verif/seq/` binds. |
| 11 | §3 "free base `0x5C+`" | **CONFIRMED** | `csr_regs` decodes ≤ `0x54` (`A_PERF_EVT0=6'h0E`, `N_BLOCKS=8` at `apex_top.sv:833`), `0x58` is glue-only, `0x5C-0x6C` fall to the reserved path (`csr_regs.sv:178` read `0xDEADBEEF`, `:239` write no-op, `:220` still acks). *Footgun:* S12/D-027 uses the same numerals `0x5C`/`0x60` in the **KVQ engine AXI-Lite** window — a physically separate bus. No conflict; document it. |
| 12 | §5 stage 6 "propose D-028" | **CONFIRMED** | highest in `ARCHITECTURE.md` is D-026 (D-025 reserved); D-027 is claimed by `docs/design/S12_LOADABLE_MASK.md:1`. D-028 is correct. |
| 13 | `:5` "MMIO per decode step ~300" | **UNSOURCED** | traces to `docs/OPTIMIZATION.md:3` citing "ARCHITECTURE §11 + measured", but ARCHITECTURE contains **zero** MMIO figures; `perf/apex_perf_model.py:46 mmio_per_l3_step=300` is dead code never read. Measured (drive+sync, see counting rule below): **356** at `adv_T1`, **360** at `rand_d64_T1`, **24,678** at `calib_d64_T128` (D=64), **37,326** at `calib_d128_T100` (D=128, suite max). The ≤3 target is unaffected; the baseline is not "~300". |

**Counting rule** (every "ops"/"transactions" figure above): *drive* lines
(`CSRW`/`KVW`/`ROUTE`/`IMP`/`DESC`/`AJ`/`WJ`/`FJOB`/`QJOB`/`SJOB`/`LJOB`/`QS`/
`CS`) **plus** status polls and reads (`KVP`/`CSRP`/`CSRR`/`CSRN`). It excludes
TB checks (`EFS`/`ESS`/`ERO`/`ESTK`), tensor payload (`XR`/`GR`/`WB`) and tap
header+payload lines. Note these are *host control ops*, a mix of bus
transactions and direct pin drives — only the `CSR*`/`KV*` subset (2,847 of the
24,678 at `calib_d64_T128`) is MMIO in the strict sense. Quote the composition
when comparing against the old "~300 MMIO" figure.

**Unscoped work surfaced (not yet in §5):** a KVQ **AXI-Lite master FSM** — the
walker's inner loop is `{poll STATUS.idle, write READ_ADDR}` per token per
phase (1,409 `KVW` + 1,412 `KVP` in `calib_d64_T128`), and `REG_READ_ADDR`
writes have a side effect (`kvq_engine.sv:920-923`). This is the walker's
dominant sequencing cost and v1 never mentions bus mastering.

**Landmines for stage 1** (each breaks a minority of cases, so a naive walker
looks green): `stage_c8_emit`'s EMIT→DESC→EMIT order is deadlock-load-bearing
and only exercised at `ceil(T/8) > BPR`, i.e. 4 of 27 cases (`adv_outlier1000`, `adv_outlier1000_cq4p`, `calib_d64_T128`, `calib_d64_T70`)
(`gen_l3_vectors.py:325-328`); `score_phase` and `pv_phase` use *different*
poll-free re-route predicates (`:386` vs `:421`), exercised only by
`tip_auto_mixed`; `fq_out_ready` is forced 0 when `rt_feeder_dst=0 &&
rt_act_src=1` (`apex_top.sv:771-772`) and wedges the feeder with no error;
`err_sticky[15]` is reserved-0 and pinned by every case's `ESTK`
(`gen_l3_vectors.py:447-455`) — a `walk_err` bit must not land there.
