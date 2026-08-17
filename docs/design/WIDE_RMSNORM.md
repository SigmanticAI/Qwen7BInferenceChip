I have everything needed. All files are read and cross-referenced. Here is the complete design contract.

---

```markdown
# WIDE-D RMSNORM — Golden-First Design Contract & Take-Over Brief

**Block:** `asu_rmsnorm` extended to hidden D = 3584 (7B) via C-RMSW.
**Status:** golden LANDED (S6, `plumbing7b` §B); RTL/TB not started.
**One-line:** widen the sum-of-squares path + replace the pow-2 mean shift with
the μ_D scalar multiply, keeping the rsqrt input range and the per-element
emission datapath **bit-identical** to today's D≤128 unit.
**apex_pkg.sv stays FROZEN** — `APEX_VERSION = 32'h0001_0000`
(`rtl/apex_pkg.sv:14`). D is a module parameter, not a descriptor field, so no
`§5`/route/DIM_W (`rtl/apex_pkg.sv:19`) change is required or permitted here.

---

## 1. What it does & where it sits

`x → [RMSNorm] → [MXE Q/K/V proj] → …` — RMSNorm is the first op of every
decoder layer (`ARCHITECTURE.md:17`). The tile's RMSNorm today caps at
**D≤128 pow-2** (`docs/HANDOFF.md:94`; `rtl/asu/asu_rmsnorm.sv:8-11`). 7B hidden
= **3584 = 28·128**, so the tile cannot run RMSNorm-1/2 for 7B without this
block. Golden `decoder_layer_fx` already calls the wide reference at both norm
sites (`golden/apex_golden/transformer.py:405-410, 466-469`).

**The one architectural invariant that makes this "the smallest of the three":**
`mean2 = mean(x²) ≤ max(xᵢ²) = 16384 = 2¹⁴` **independent of D**. So `n2 =
mean2+1 ≤ 2¹⁴+1` lands in the exact same 32-bit `norm2` operand the 31-cycle
`rsqrt_unit` already takes (`rtl/asu/rsqrt.sv:11-20, 74-77`). **The rsqrt unit,
its D-018 wrapper, and the per-element `x·γ·r >>18 RNE sat16` datapath are
reused verbatim** (`rtl/asu/asu_rmsnorm.sv:201-217`). Only the *sum → mean*
front-end widens.

---

## 2. Golden reference (the arbiter — never edit golden to match RTL)

- **RTL bit-exact arbiter:** `rmsnorm_fx_wide(x, g, chunk=128)` →
  `(y, r, nrm)` — `golden/apex_golden/attention.py:124-175`. Composition
  (hardware-honest, per its own docstring): chunk into ≤128 pow-2 blocks
  (`:144-150`); accumulate per-chunk `sum2`, each obeying the SAME 22-bit
  proof as the D≤128 unit, host accumulator bounded `sum2 ≤ 2²⁷` for D≤8192
  (`:153-158`); `mean2 = (sum2·μ_D) >> (s+16)` (`:162`); then `n2 = mean2+1`,
  `den = isqrt(n2)`, `r = 8192//den`, `nrm = min(den, 2¹⁴−1)` — **identical
  tail** to `rmsnorm_fx` (`:163-166` vs `attention.py:92-95`); per-element loop
  (`:168-175`) is byte-for-byte the D≤128 loop.
- **The μ_D constant:** `_wide_mu(d)` → `(mu, s)`,
  `golden/apex_golden/attention.py:109-121`: `s = (d−1).bit_length()`
  (`:119`), `mu = RNE(2¹⁶·2ˢ/d)` (`:120`, `rne` from `apex_golden/fp.py:40`).
  **For pow-2 d, mu = 2¹⁶ exactly and `(sum2·mu)>>(s+16)` reduces bit-exactly
  to `sum2>>log2(d)`** — this is why D≤128 stays identical.
- **No new golden ref is needed.** The RTL matches `rmsnorm_fx_wide`
  bit-for-bit, exactly as the D≤128 RTL matches `rmsnorm_fx`
  (`golden/tests/test_7b_plumbing.py:174`, and the verifier oracle
  `verif/asu/smoke/gen_asu_vectors.py:43`). The wide ref's *quality* vs float
  `rmsnorm_ref` (`golden/apex_golden/compute.py:101`) is already proven within
  the `[RMS]` analytic bound at D=3584 (`test_7b_plumbing.py:186-203`); the
  RTL does **not** re-litigate float accuracy — it only reproduces the integer
  `(y,r,nrm)` of `rmsnorm_fx_wide`.

---

## 3. Interface contract (cited against real RTL)

Keep the existing port list of `asu_rmsnorm` (`rtl/asu/asu_rmsnorm.sv:71-104`)
**unchanged** (x/γ/y streams, `busy/done/len_error/len_error_sticky/dbg_norm`).
The extension is a **parameter widening + μ path**, additive:

| item | today | wide change |
|---|---|---|
| `RMS_D_MAX` param | 128, guarded `[1,128]` (`:74, 106-111`) | keep **128 as default**; add a wide elaboration `RMS_D_MAX=3584` (relax the `[1,128]` guard's upper bound to the width-proven max; sum2 proof re-derived below) |
| `SUM_W` | 22 (`:113`, proof `:40-43`) | **28** (`sum2 ≤ 2²⁷` at D=8192, golden `attention.py:158`); re-state the proof in the header |
| `DCNT_W`/`IDX_W`/`cnt`/`n_len`/`emit_idx`/`out_cnt` | 8b / `$clog2` (`:114-115,158-161`) | width for `0..RMS_D_MAX` (12b at 3584) |
| `x_buf [RMS_D_MAX]` | 128×8b (`:165`) | 3584×8b (BRAM) — **or** the external-r variant, see §5 note |
| mean shift | `(sum2 >> f_log2(n_len)) + 1` (`:174-179, 189`) | `((sum2·MU)>>(S+16)) + 1`, MU/S synthesis constants |
| legality | reject non-`$onehot` or `>128` (`:281-294`) | accept `d≤128 pow-2` **OR** `d = k·CHUNK, k≥1, d≤RMS_D_MAX`; reject else — same reject *observable* (len_error pulse + sticky, row consumed, no output/done) |

`MU`/`S` are generated from `_wide_mu(D)` by a new `gen_*` script and
**provenance-gated** (regenerate → `diff` the checked-in `.svh`), mirroring the
LUT house pattern (`verif/asu/smoke/Makefile` `tables-check`; `golden/Makefile`
`lut-tables`). No CSR added for the self-contained variant → apex_pkg frozen.
The rsqrt operand `rs_norm2` (`:183, 189`) still feeds a 32-bit port
(`rtl/asu/rsqrt.sv:74`); `mean2 ≤ 2¹⁴` so the wider `sum2·MU` product is shifted
back into range before it reaches the unit.

---

## 4. Verification target

**Acceptance = bit-exact vs `rmsnorm_fx_wide` `(y, r, nrm)`** — per output beat
`m_y` (Q7.8) exact, framing (`m_last`) exact, `dbg_norm == nrm`
(floor-sqrt cross-check, as `tb_asu_rmsnorm.sv:check_row`). This is the identical
scoreboard shape used today; only the oracle call site changes
(`rmsnorm_fx` → `rmsnorm_fx_wide`).

**TB template to follow:** `verif/asu/smoke/tb_asu_rmsnorm.sv` (T1 bit-exact
row, T2/T3 rejects, T4 D=1, T5 replay; §5 SVA bound on all three boundaries via
`apex_stream1_sva`; D-006 `done⇒drained` monitor; 100k-cycle `$fatal`
watchdog). For adversarial/mutation depth follow `verif/asu/sb/
tb_asu_rmsnorm_sb.sv` (vector records `N/B/O/Z`, `+bp_mode/+stall_mode/+g_mode`
adversaries, and the **mandatory mid-op reset** `Z <phase>` targeting each FSM
state `rtl/asu/asu_rmsnorm.sv:153-155`).

**New wide vectors:** D=3584 random rows, D∈{256,512,1024}, the D=128 pow-2
regression rows (must equal the smoke oracle), plus the D=8192 all-(−128)
corner (`test_7b_plumbing.py:225-230`) and non-multiple-of-128 rejects
(D=129, D=3585 — `test_7b_plumbing.py:218-223`).

**Mutation gate** (extend `verif/asu/sb/mutation_check.py`, house pattern):
- reuse **M3** RNE→half-up on the shared emission line (`mutation_check.py`
  `M3_rms_rne_to_half_up`) — proves the reused datapath is still tie-checked;
- **new μ-path mutant:** `MU` off-by-one (or `>>(S+16)` → `>>(S+15)`) — must be
  caught by a D=3584 row where `mean2` sits on a divisor boundary
  (the B-2 `|mean2_μ − floor(sum2/D)| ≤ 1` envelope, `test_7b_plumbing.py:
  177-184`);
- **new SUM_W truncation mutant:** `SUM_W=27` — must be caught by the
  all-(−128) D=8192 corner (`sum2 = 2²⁷` overflows a 27-bit reg).

---

## 5. Staged landing plan (golden-first, machine-aware)

| stage | what | machine need |
|---|---|---|
| ✅ 0 | golden gates — `rmsnorm_fx_wide` + `_wide_mu`, `plumbing7b` §B (`test_7b_plumbing.py:163-230`) + **this doc** | none (LANDED S6) |
| 1 | `gen_wide_rms_params.py` → `MU`/`S`/`CHUNK` `.svh` for D=3584, provenance-diff vs `_wide_mu(3584)` | edit only |
| 2 | RTL: widen `SUM_W`→28 + counters; add μ-mean path; extend legality; **guard the D≤128 path so `RMS_D_MAX=128` elaboration is unchanged** | edit only |
| 3 | **Regression anchor** — re-run `verif/asu/smoke` + `verif/asu/sb` at `RMS_D_MAX=128`, **byte-identical logs**, zero test edits (S10-style compat proof) | Verilator (serialize) |
| 4 | New `verif/asu/wide/` suite: wide vectors from `rmsnorm_fx_wide`, D=3584 bit-exact + rejects + D=8192 corner + mid-op reset; ≥3 mutants (M3 + μ + SUM_W) | Verilator (serialize) |
| 5 | ARCHITECTURE.md §6 (`:145-161`) note + STATUS row + retire the "asu_rmsnorm sum2-export/external-r" backlog delta (`docs/OPTIMIZATION.md:42`; `docs/HANDOFF.md:201`) | none |

Stages 3–4 need Verilator → **serialize on the one build machine**; 1/2/5 are
edit-only and can proceed in parallel with the other two blocks.

**Microarchitecture note (the one open choice):** widening `x_buf` to
3584×8b buffers the whole row (BRAM, simplest, keeps today's COLLECT→EMIT
structure). The resource-saving alternative the backlog actually named —
**export `sum2` / accept an external `r`** (`docs/OPTIMIZATION.md:42`,
`docs/HANDOFF.md:201`) — lets the host do the μ multiply and stream x once per
emit, avoiding the buffer. Either is bit-identical to `rmsnorm_fx_wide`; pick
the buffer variant first (self-contained, matches the existing FSM), flag the
external-r variant as a follow-on. **Do not** change the D≤128 build's behavior
in either case.

---

## 6. Files to read / touch, and repro

**Read:** `rtl/asu/asu_rmsnorm.sv`, `rtl/asu/rsqrt.sv`,
`golden/apex_golden/attention.py:82-175`,
`golden/tests/test_7b_plumbing.py:163-230`, `ARCHITECTURE.md:145-161`,
`verif/asu/smoke/tb_asu_rmsnorm.sv`, `verif/asu/sb/tb_asu_rmsnorm_sb.sv`,
`verif/asu/smoke/gen_asu_vectors.py`, `docs/design/S12_LOADABLE_MASK.md` (stage
table style).

**Touch (this block only):** `rtl/asu/asu_rmsnorm.sv` (widen + μ path);
NEW `verif/asu/wide/` (TB + `gen_wide_rms_vectors.py` + `Makefile`);
NEW gen script for `MU`/`S` `.svh`. **Do NOT edit** `verif/asu/smoke/*` or
`verif/asu/sb/*` — they are the byte-identical regression anchor (stage 3).

**Repro:**
```
# golden arbiter (must already pass):
make -C golden plumbing7b            # test_7b_plumbing.py §B C-RMSW

# stage 3 regression anchor (byte-identical):
make -C verif/asu/smoke run_rmsnorm
make -C verif/asu/sb all             # incl. mutants gate

# stage 4 (new wide suite, once created):
make -C verif/asu/wide               # D=3584 bit-exact vs rmsnorm_fx_wide
```
Verilator is `/opt/homebrew/bin/verilator`; RTL file list mirrors
`RTL_RMSNORM` (`verif/asu/smoke/Makefile`): `apex_pkg.sv`, `stream_skid.sv`,
`rsqrt.sv`, `asu_rmsnorm.sv`.

---

## 7. Independence notes (collision surface)

- **apex_pkg.sv:** MUST stay frozen (`APEX_VERSION 0x0001_0000`). This block
  needs no pkg edit — the only cross-block collision risk with the walker/
  weight-path sessions is apex_pkg, and this block does not touch it. If any
  session needs a pkg field, it is a D-numbered contract change coordinated
  separately, **not** in this block.
- **RTL file:** only `rtl/asu/asu_rmsnorm.sv` is edited — an ASU leaf the
  walker (kvq/route) and weight-path (mxe/W4) blocks do not compile.
  Collision with those two: **none** (confirmed against `RTL_RMSNORM`).
- **TB dirs:** new work lives in a **new** `verif/asu/wide/` dir; the shared
  `verif/asu/smoke` + `verif/asu/sb` suites are read-only regression anchors.
  No shared TB, no shared vector `.svh`.
- **Golden:** read-only. `rmsnorm_fx_wide`/`_wide_mu` are the arbiter; do not
  modify them to match RTL.
```

---

## 8. Chunk composition on a NARROW build (R4, LANDED 2026-07-31)

The §5 microarchitecture note flagged an "export `sum2` / accept an external
`r`" variant as a follow-on. It landed as **R4** — with one deliberate
difference: the unit accepts an external **`sum2` + `k`** (not an external
`r`), so the μ multiply, EPS, rsqrt AND the whole per-element datapath stay
on the tile; the host's only arithmetic is the chunk-sum accumulation that
`rmsnorm_fx_wide` step 2 already assigns to "the host (or a scalar adder)".
This is what makes the FULL 3584-wide RMSNorm-2 computable on the
`RMS_D_MAX=128` image that the vendor bug (aws-fpga #799) forces us to fly.

- **RTL:** `rtl/asu/asu_rmsnorm.sv` R4 header + ports (`ext_sum_en`,
  `ext_r_en`, `ext_sum2[27:0]`, `ext_k[6:0]`, `s2_push`, `s2_val[27:0]`);
  `rtl/top/apex_top.sv` R4 CSR pair `0x90 RMS_SUM2` / `0x94 RMS_EXT`.
  Illegal arms are REFUSED loudly (LAYER_STATUS sticky + err_code 8:
  k outside [2,64], sum2 > k·2²¹, both modes at once, any write while the
  unit is busy, nonzero sum2[31:28]). Both levels 0 ⇒ frozen behavior,
  byte-identical (walker l64/l128/l7b/refuse2 + capgate replays unchanged).
- **Sum sources, both proven on the narrow twin:** pass-1 chunk-sum export
  (this block; graded bit-exact per chunk) and the MXE x·x dot product
  (`scripts/fpga/f2/sum2_mxe.py`, same INT32: 120498 on the real L00/s010
  row). Either feeds the same `0x90` arm.
- **Proof:** `scripts/fpga/f2/gen_layer_ops.py` stages `norm2c` (two-pass,
  self-contained, host = 27 INT32 adds), `norm2x` (ext-only; the MXE-fed
  shape), `norm2_probe` (refusal guard). All bit-exact vs the arbiter
  `rmsnorm_fx_wide` (via the step npz's own `h2`) on the NARROW twin, and
  unchanged-green beside the wide-elaboration `norm2` stage on the wide twin.
