# W-G3 `layer_walk_golden` — the flagship replay harness

**Gate definition:** `docs/design/IB_WALK.md` §4 (stage 6) — "one committed
decode step of one real Qwen2.5-7B layer traced from `run_tinynpu.py`,
pre-swizzled into a DDR image, replayed loader→DDR→reader→walker→tile in the
extended f2sim — weights genuinely from (modeled, then real) DDR."
Owned by the COMBINE per `LEVEL_C_INTEGRATION.md` §9.1 and the IB-WALK
stage-6 row ("W-G3 replay = COMBINE-owned, do not attempt in-lane").

**Discipline:** golden is the arbiter, one-directional. Every artifact here is
generated FROM golden and self-checked BEFORE any RTL runs. `golden/`,
`run_tinynpu.py` and `rtl/apex_pkg.sv` are NOT edited by this harness; the two
hooks it needed in the IB-WALK generator are additive with defaults that keep
every existing vector byte-identical (proven, see §3).

---

## 1. The three stages

| stage | tool | output (under `build/walkgold/`) |
|---|---|---|
| A — tracer | `trace_layer_golden.py` | `layer_s<step>_L<layer>.npz` + `.json` — one REAL decoder-layer step of the committed S8 run |
| B — compiler | `gen_walk_golden.py` | the fmt=1 descriptor image + expected walker emissions, the pre-swizzled DDR weight image, the per-step MMIO stream, the fenced-walk case, the golden-gated expectations |
| C — replay | (executor case) | the two halves below |

### The deliverable is TWO HALVES — not a walked full layer

Integration-lead rulings, 2026-07-26. **The harness banner and any RESULT
must say "partial-layer walk + fuel-fed projection".**

- **Walked half** — step-enable mask `0x03D` =
  `NORM1 | ROPE | STOREKV | SCORE | PV` at real 7B geometry (H=28, H_kv=4,
  head_dim=128, T=20) with GQA-4 banking. QKV / o-proj / FFN / RES1 / RES2 /
  NORM2 are fenced OUT: their projection descriptors are refused by the
  implemented MXE (F1 — a tracked D-029 erratum owned by the walker lane, NOT
  fixed here), and RES/NORM2 lose their producer with o-proj disabled. K is
  staged PRE-RoPE so the tile's `rope_row` rotates it on the S-2 store path
  (F5 gives the arbiter). Consumes no DDR tensor (F4).
- **Fuel half** — a HOST-driven projection GEMM at the legal `n=8`, reading
  REAL Qwen weights out of the DDR image through
  loader→DDR→reader→afifo→xw→MXE, gated against golden.

Together these cover walker→tile and loader→DDR→reader→tile on one real
traced layer step. They are NOT one unified walked-full-layer flow, and no
such claim may be built on them while F1 is open.

### Stage A — what makes the trace trustworthy

The S8 tracer samples per-HEAD attention jobs; W-G3 needs the whole
`decoder_layer_fx` step. `trace_layer_golden.py` re-runs the **committed** S8
token sequence (`docs/results/s8_7b_token/artifact_trace/run.json`:
prompt_ids + generated_ids) through the S8 `GoldenModel`/`decoder_layer_fx`
and captures the target layer.

Its **provenance gate** is the reason the result is not merely "a 7B layer":
the committed artifact holds the SAME head of the SAME (step, layer) of the
SAME run, so the re-derived per-head fields are asserted BIT-EXACT against it
(22 arrays: `q_f16 K_f16 V_f16 k8 v8 q8 s_k s_v s_q acc_s score_fx p_q115 c8
s_c acc_o rq_scale rq_shift o8 s_out out_hat sm_m sm_l`). A replay off by one
token or one layer cannot pass.

Default target = **step 19, layer 19, T=20** — a committed traced job
(`job_s019_L19_h03.npz`) whose T is inside both the walker fence (T≤128) and
the tile's M limit (§2 F2), with heads unchunked.

### Stage B — the artifacts

- **(a) descriptor + emissions** are built by the EXISTING IB-WALK compiler
  (`verif/seq_walker/gen_layer_trace.py::build_case`) through its additive
  `inject=(w, X, fx)` hook, so the fmt=1 wire format keeps ONE source.
- **(b) the DDR image** follows IB_WALK §2.6 ("the image IS the wire format"):
  per-tensor regions at the `seq_walker_fmt.image_bases` offsets, each tensor
  laid out as its per-job weight blocks concatenated in decomposition order
  (n-split-major, k-split-minor). The per-job byte order is the VERIFIED WS
  load order of `verif/top/l3/gen_l3_vectors.py::wgt_beats_ws`, vectorized and
  proven identical to that scalar reference by `_selftest_swizzle`.
- **(c) the per-step stream** is IB_WALK §2.5's H+11 exactly: DPTR→W21, STEP,
  H+2 RQ words, DPTR→W54, 4 JC words, GO, STATUS poll = **39 MMIO at H=28**.

### Self-checks (all pure Python, all hard asserts)

| id | what |
|---|---|
| S1 | descriptor passes the `check2` mirror at CFG_D |
| S2 | epilogue replicas reproduce `fx.attn_proj`/`fx.ffn_out` bit-exact; the RQ table equals the golden per-head `rq` pairs |
| S3 | JC composites are D-030 CANONICAL-graded (idempotent, `frac[12:0]==0`, positive-normal) |
| S4 | **DDR image round-trip**: every tensor region re-read from the emitted image and un-swizzled reproduces the source matrix bit-exactly (7/7) |
| S5 | the decomposition tiles each matrix exactly (full coverage, no overlap) |
| S6 | IB_FUEL §2.3 divisibility: every region's lane-beat count ≡ 0 (mod 8) |
| S7 | every emitted MXE descriptor is legal for the IMPLEMENTED tile (§2 F1) |
| S8 | `fuel_req` records fit the frozen field widths and address the image |
| S9 | the per-step MMIO stream is exactly H+11 |

---

## 2. FINDINGS — escalated, not worked around

### F1 — the fmt=1 N-split is derived from the descriptor FIELD width, not the implemented ARRAY width

`seq_walker_pkg.sv:302` (and its Python mirror `seq_walker_fmt.N_JOB`) set

```
WALK2_N_JOB = (1 << DIM_W) - 1;   // 4095 (= DIM_MAX)
```

so the walker splits projections at 4095 columns and emits
`pj_desc.n_dim = n_cur` up to 4095 (`seq_layer_walker2.sv:437,469`). At the
7B geometry that is n ∈ {512, 2564, 3584, 4095}.

The **implemented** MXE limit is the array width:

```
rtl/mxe/mxe_cfg_pkg.sv:10-11   M <= M_TILE_MAX (64)
                               N <= MXE_N      (8) — one array width per
                                                 descriptor (no N tiling v0.1)
rtl/mxe/mxe_ctrl.sv:164        && (desc.n_dim != '0) && (desc.n_dim <= DIM_W'(MXE_N))
```

`legal` gates descriptor acceptance; an illegal descriptor raises the
`desc_error` pulse (`mxe_ctrl.sv:81`) and the job is NOT accepted. So **every
fmt=1 projection descriptor at 7B would be rejected by the real tile.**

Why it was not caught earlier (not a lane failure — a seam):
- the walker unit TB (`tb_walker2_sb.sv`) is a scoreboard, not a real MXE: it
  captures DESC fields and compares them to this same spec, so spec and RTL
  are self-consistently wrong together;
- the W-G2 tile case (`run_walkfmt2`) walked the L3-derived **attention**
  descriptors, which are n=8 by construction from the D-028 emitters.

W-G3 is the first gate that drives PROJECTION descriptors into a real
`mxe_ctrl`, which is exactly where it surfaces.

**Note the constant is overloaded.** `WALK2_N_JOB` is used for BOTH the MXE
projection n-split (must be ≤ `MXE_N`) and the `LAYER_JOB` cols chunking
(`seq_layer_walker2.sv:444,706`), where 4095 is genuinely correct — the §3b
`LAYER_JOB[11:0] cols` field. A one-constant edit would therefore be WRONG;
the fix needs two distinct constants. That is why this is escalated as a
design change rather than fixed mechanically here.

**Options for the lead**

1. **Split the constant** (recommended): keep `WALK2_N_JOB = 4095` for the
   `LAYER_JOB` cols chunking, add `WALK2_N_MXE = MXE_N` for the projection
   n-split; update the Python mirror, regenerate the IB-WALK vectors, re-gate.
   Cost: the walker's projection job count rises (7B: 16,000 weight jobs/layer
   vs 52), and `verif/seq_walker`'s stage-2/5 counts change by design — so it
   breaks the "byte-identical" expectation for that suite ONLY, deliberately
   and with a documented reason. No descriptor-format change, no
   `apex_pkg.sv` change, no image-format change (the image bytes are the same
   set; only the block ORDER within a tensor changes).
2. **Widen the tile** (N tiling in `mxe_ctrl`) — a real datapath project,
   out of I-B scope, and it would invalidate the "no N tiling v0.1" note.
3. **Leave the walker as-is and drive projections host-side** — W-G3's
   host-driven half still lands (this harness supports it), but the WALKED
   full-layer claim cannot be made.

This harness defaults to `--n-job 8` so its artifacts are correct against the
real tile today; `--n-job 4095` reproduces the current spec's ordering for
comparison and is caught by S7.

### F2 — the fmt=1 T envelope exceeds the tile's M limit for the K/V projections

`proj_desc_lines(..., m_dim=T)` for Wk/Wv (`gen_layer_trace.py`) puts the
context length in `m_dim`. The fmt=1 envelope allows `t_rows ≤ 128`
(`seq_walker_fmt.T_MAX`, checked by `check2`), but `M_TILE_MAX = 64`. So any
fmt=1 descriptor with **T > 64** is `check2`-legal yet produces MXE-illegal
K/V projection descriptors. T=20 (this harness's default) and T≤64 are clear.

Recommendation: fold `m ≤ M_TILE_MAX` into the fmt=1 legality story — either
tighten `check2`'s `t_rows` bound to 64 (envelope change, needs the lead's
ruling since T≤128 is a published fence) or M-split the K/V projections in
the walker templates (keeps T≤128, adds a split loop).

### F3 — the step-enable fence is UNPOLICED (erratum vs IB_WALK §2.2)

IB_WALK §2.2 specifies dependency checking: "en_qkv without en_storekv (and
similar dependency violations) ⇒ `WALK_ERR_DESC`". **No such clause exists.**
`seq_walker_pkg.sv::walk_desc2_check` (and its `seq_walker_fmt.check2` mirror)
refuse only `en_mask == '0'`; every other subset — including incoherent ones
like SCORE without any K/V producer, or RES1 whose producing o-proj is
disabled — is ADMITTED and walked.

**Host responsibility, explicitly.** Choosing a coherent step subset, and
staging the inputs that the fenced-out steps would otherwise have produced,
is the HOST's job; the descriptor check will not catch a mistake. Every use
of the fence in this harness therefore stages and golden-gates the inputs of
each enabled step (`build_fenced_walk`). Recorded per the integration lead's
ruling 4 (2026-07-26).

Either implement the §2.2 clauses or strike them from the contract; today the
doc overstates the RTL.

### F4 — the fuel line cannot feed the norm gammas

The fuel reader's only sink is the `xw` lane8 weight stream (`cl_apex.sv` xw
mux); RMSNorm gammas enter the tile through the SEPARATE `xg_valid` /
`xg_gamma` port (`apex_top.sv:198-200`). So the walker's `PC_G1F`/`PC_G2F`
gamma FETCH records (`TENS_G1`/`TENS_G2`, which the fmt=1 image lays out and
this harness emits) have no path from DDR into the norm unit — gammas must be
host-pushed today. Consequence for W-G3: with the projections fenced out, the
WALKED half consumes NO DDR-sourced tensor, which is why the fuel exercise is
a separate host-driven projection half (lead ruling 3).

### F5 — the K-rope expectation must use the D-030 `rope_in_f16` arbiter, not the legacy trace field

Measured on the traced layer (step 19 / L19, T=20, 4 KV groups, 80 K rows):

| expectation source | rows matching |
|---|---|
| legacy `decoder_layer_fx` rotating the UNNARROWED `K_real` | 80/80 (`K_rope`) |
| the TILE's path — rotate the f16-narrowed S-2 bus value | **4/80** vs legacy (only the t=0 identity rows), max abs delta 1.95e-3 ≈ 1 fp16 ulp |
| golden `BusMode(rope_in_f16=True)` | **80/80** vs the tile's path |

`K_real` is NOT on the fp16 grid (`K_rope` is), so rotating it is not what the
hardware does. This is exactly the §2b-3 composition point D-030 exists for —
not a bug — but it means a K-rope gate written against the legacy `K_f16`
field would FAIL on a correct tile in 76/80 rows. This harness therefore
stages `f16(K_real)` and gates against golden `rope_fx` applied to that same
f16 bus value, which is verified equal to the signed `rope_in_f16`
composition. Relevant to the segment-1 lane's K-path work.

---

## 3. Byte-identity of the two additive hooks

`seq_walker_fmt.jobs()` gained an optional `n_job`; `gen_layer_trace.build_case()`
gained optional `inject=` and `n_job=`. Both default to the previous behaviour.
Proven on the pristine tree (`diff -r` over all seven generated files AND the
selftest stdout — rc=0, byte-identical), with both selftests re-run green.

## 4. Repro

```
python3 verif/walkgold/trace_layer_golden.py --step 19 --layer 19   # ~5 min
python3 verif/walkgold/gen_walk_golden.py                          # ~1 min
```

The tracer needs the S8 prepared weight cache (gitignored, ~7.6 GB — the
`run_tinynpu.py --prepare` output the committed S8 run used); its path is
recorded in the bundle's `weights_dir` and every weight file is sha256-pinned.
