# W4 DATAPATH — the tile's W4 weight-ingest lane (D-031 combine)

**Status:** 🟢 lane COMPLETE IN SIM 2026-08-10 — tile-level W4B GEMMs
bit-exact vs golden (host mode), full gate ladder below.
**Branch:** `feat/w4-datapath` (worktree `../apex-w4`, cut from
`comp/prompt-b-c` @ 05ea8c5).
**Recipe (FROZEN, not revisited here):** W4 **G=32 through the (B) chain
with DIRECT host prep** — D-031, adopted at −1.0 pt vs shipped weights
(n=10,042, `docs/results/b3_w4_adoption/RESULTS.md`); accuracy campaign
CLOSED 2026-07-22 (`B3_WEIGHT_PATH.md` §5 row 6). This lane is the
INTEGRATION the W4B_FEEDER.md staged plan left for "the combine session":
the verified feeder wired into the tile, the host prep, and the
end-to-end golden.
**Golden arbiters:** `weight_codec.wfeed_w4b_to_i8` (per element, landed,
54M stripe checks + the 4.06M-point exhaustive sweep behind it) +
**NEW** `golden/apex_golden/w4_lane.py` (DIRECT prep, wire framing,
job-level GEMM composition — every element deferred to landed arbiters).
**Fences kept:** `seq_walker_comp.sv` / `apex_wcomp_bank.sv` untouched
(under active fix elsewhere); `apex_pkg.sv` frozen (no descriptor field —
`w4b_en` stays route/CSR-level exactly as the feeder header contracts);
`mxe_wfeed_w4b.sv` / `verif/mxe/w4b` byte-untouched (the unit anchor);
no AWS, sim only.

---

## 0. What was built

```
                     (E-7 gamma window — unchanged, walker-only)
                          │
xw ──►(gam_win_q demux)───┼──► w4_lane_act=0 ────────────────► rt_wgt_src
                          │                                      mux ─► MXE wgt
                          └──► w4_lane_act=1 ──► apex_w4_ingest ─┘
                                                │
                                     ┌──────────┴─────────┐
                                     │ PH_GS: ceil(ng/4)  │  4 fp16/beat
                                     │   beats → gs_*     │  ascending gid
                                     │ PH_PW: ceil(b/2)   │
                                     │   beats → pw_*     │
                                     │  mxe_wfeed_w4b     │  (verified unit,
                                     │  (G=32, D-031)     │   instantiated
                                     └────────────────────┘   verbatim)
```

New files: `rtl/top/glue/apex_w4_ingest.sv` (the demux/framing glue that
owns one `mxe_wfeed_w4b` instance), `golden/apex_golden/w4_lane.py`,
`golden/tests/test_w4_lane.py`, `scripts/prep_w4_weights.py`,
`verif/top/w4/` (tile suite + mutants). Edited: `rtl/top/apex_top.sv`
(parameter-gated region + CSR window + the mux splice),
`verif/top/l3/tb_apex_l3.sv` (additive `W4EN` param, default 0),
`golden/Makefile` (additive `w4lane` gate, existing banners byte-identical).

### Elaboration + runtime gating (two independent OFF-identities)

* **`W4_LANE` apex_top parameter, default 0** — the PROJ_BIAS_EN /
  KVQ_GQA_NENG idiom: no glue, no divider pipes, no CSR window
  elaborates; 0x9C–0xAC keep reading 0xDEADBEEF (the absent-feature
  probe); every existing build is byte-identical BY CONSTRUCTION (the
  generate's off-branch ties `w4_lane_act = 0` and the weight-mux
  expressions constant-fold to their exact legacy forms).
* **`W4_CTRL.lane_en` route level, reset 0** — with the lane elaborated
  but off, the muxes reduce to the legacy nets at runtime (quasi-static
  rt_* rule: change only while `W4_STAT.busy=0`). Gate j8 below proves
  the off-path in-situ in the W4 build.

### Placement — ON the xw leg, and WHY it is a mux (one deliberate
### deviation from W4B_FEEDER.md note 1)

The integration note said "passthrough at w4b_en=0 means no datapath mux
is needed." That premise does not survive contact with the feeder's own
FSM: passthrough is still JOB-FRAMED (`pw_ready` only in ST_RUN), so an
unconditionally-in-line feeder would require a job push for every legacy
weight stream — every existing host test would wedge on a 2-deep skid.
The lane therefore splices behind a route mux whose off-form is the
legacy net exactly. The `rt_wgt_src` mux stays untouched downstream; the
gamma window keeps first claim on the stream (walker-only, exclusive by
its window contract).

### Wire order — SCALES FIRST (normative; the second deliberate
### deviation, from note 2's image layout)

`W4B_FEEDER.md` note 2 puts scales "immediately after the packed beats"
in the DDR image. As a WIRE order on one in-order pipe that deadlocks by
construction: the feeder stalls beat 0's emission until gid 0's scale is
resident and holds only a 2-deep pw skid, so packed beats would jam the
pipe with every scale behind them. Scales-first is legal by the feeder's
own contract ("early (eager) scales park in the skid + RAM" — all
`ngtot ≤ 512` of them), costs nothing, and keeps the image layout free:
a fuel producer stores scales wherever it likes and fetches the scale
record FIRST (fetch order is the producer's choice — two records per
job, exactly the walker-descriptor plan already contracted there).

**GS beat format:** one 64-bit beat carries FOUR fp16 scales
little-endian — slot i at bits [16i +: 16] = gid 4b+i, ascending-gid,
zero-padded tail (padding is COUNTED OFF by the glue, never pushed: the
feeder's gs_ready backpressures beyond ngtot by design, so pushing pad
would wedge). Byte-identical to an x86 memcpy of the fp16 array — the
"image IS the wire format" byte-pipe rule extends unchanged.

**Packed format:** unchanged weight_codec S4/S5 (element e at packed-blob
bits [4e +: 4]; `ceil(job_beats/2)` beats; odd tail's upper half is
dropped padding).

### W4 CSR window (0x9C–0xAC; the ERR_STICKY/WALK seam idiom — csr_regs
### acks the range as reserved, apex_top owns the data + read override)

| addr | reg | bits |
|---|---|---|
| 0x9C | W4_CTRL | [0] lane_en (route level) · [1] mode_pass (INT8 passthrough framing debug) · [8] GO (W1: kick the job in W4_JOB) · [9] W1C err_sticky · [10] W1C done_seen |
| 0xA0 | W4_JOB | {s8_f16[31:16], k[15:4], n[3:0]} — `beats = ceil(k/8)·n` and `ngtot = n·ceil(k/G)` are DERIVED in the glue, so the feeder's beats-match legality holds by construction; the prep manifest emits this exact word per job |
| 0xA4 | W4_STAT | RO {present=1[31], phase[7:6], done_seen[2], err_sticky[1], busy[0]} |
| 0xA8 | W4_CNTI | RO {gs_beats[31:16], pw_beats[15:0]} — cumulative xw beats consumed per phase (reset-only clear) |
| 0xAC | W4_CNTO | RO cumulative INT8 beats emitted toward the MXE |

Error model (§3 idiom): illegal {K,N} on GO, or GO anywhere but a truly
idle lane → 1-cycle pulse + sticky, ZERO other state change (the refuse
is explicit for every non-idle phase — never a silent drop). W4 errors
live HERE, not in `err_sticky[15:0]` — that bundle's bits are pinned by
every L3 case's ESTK expectation (the B1 lesson, kept).

### Job flow (host mode, per job)

1. `W4_JOB ← {s8, k, n}`; `W4_CTRL ← GO|lane_en`
2. glue prechecks legality, pushes the feeder job (beats derived),
   opens PH_GS: routes `ceil(ngtot/4)` xw beats through the 4-slot
   serializer into `gs_*`
3. PH_PW: routes `ceil(beats/2)` xw beats into `pw_*`; the feeder
   dequant/requants at II=1 against MXE backpressure (OF_DEPTH=32
   credit); PH_DRAIN until the feeder's D-006 done
4. `W4_STAT.done_seen` latches; counters advance; lane idles for the
   next GO. The MXE descriptor rides the normal `ds_*` path unchanged —
   the tile still sees exactly one weight dtype (C-4).

K-split stripes (K > K_MAX): one GO per segment, SAME `s8` in every
segment's W4_JOB (the D-021 shared-stripe-factor semantics, B3 §8.1);
OS-accumulate composes the partials on-tile — gate j6 proves the full
chain including the C-2 requant of the accumulated result.

## 1. Host prep — `scripts/prep_w4_weights.py` (the §8.3 obligations)

DIRECT from source reals (fp16-grid, where mlx dequant lives), never via
the deprecated INT8 hop (−1.7 pt — not even offered as a flag). Per
8-column job tile: G=32 group scales by the C-1 INT4 chain
(`scale_from_amax`/`quant_codes`, the cq_codec certified primitives);
per stripe: `s8` by the tile-amax rule taken FROM the landed (B) arbiter
itself (so the reduction identity holds by construction), and the graded
epilogue composite `f16_grade(s8 × s_w)`. Emits the wire-order beat
stream (`.bin`/`.memh`), and a manifest with per-job `w4_job_word`
(the ready-to-write W4_JOB value), per-stripe s8/composite, sha256, and
the ingest accounting. `--check` golden-replays every stripe.

```
$ python3 scripts/prep_w4_weights.py --in W.npy --out build/w4 --k-split 64 --memh --check
```

`run_tinynpu.py --prepare` growing a `--w4-direct` mode remains the
stage-5 follow-on it always was (`_proj_epilogue` carries one scalar s_w
per tensor — the documented golden-pipeline blocker, B3 §5); this script
is the tile-feed producer that does not need that refactor.

## 2. The ingest headline, honestly stated

The packed phase carries exactly **2 INT8 beats per xw beat** (S6:
`emitted == 2·consumed − (beats & 1)`); G=32 scales add 0.5 b/w, so the
whole-job wire ratio is **8/4.5 ≈ 1.78×** at real shapes (measured 4.5
b/w exactly in the prep manifest above, converging with the adopted perf
model's figure). The tile gate's small jobs land at 1.476× whole-case
(tails and tiny K inflate the scale share — printed by the generator,
never rounded up to "2×"). The 2× claim is the packed-phase density and
is measured in-RTL: gate j2 (K=128, N=8) consumes exactly 64 packed
beats for 128 emitted beats, checked by the W4_CNTI/W4_CNTO CSRs.

## 3. Gate ladder (all local, pinned Verilator 5.044; no PASS without
## pasted output)

**G1 — lint, both elaborations** (`make -C verif/top/w4 lint`): apex_top
`-Wall` clean (l3 waiver scope) at `W4_LANE=0` AND `=1`:

```
W4OFF_EXIT=0
W4ON_EXIT=0
```

**G2 — golden lane gates** (`make -C golden w4lane`; also runs inside
`make -C golden test`):

```
W4LANE RESULT: checks=87 fails=0 -> PASS
```

[A] DIRECT prep + reduction identity (7 shapes × G∈{16,32} × 3 seeds),
[B] G-aligned stripe segmentation reassembles the whole-stripe W8 and
partial-sums equal the full-K GEMM (incl. K=2080 via `gemm_i8_ksplit`),
[C] wire framing round-trip + beat counts + the S6 density identity,
[D] composite grade (idempotent f16_grade), [E] requant composition,
[F] determinism.

**G3 — the tile gate** (`make -C verif/top/w4 vectors build run`): the
UNMODIFIED L3 harness (b64 point, `-GW4EN=1`) replaying
`w4_gemm_b64.txt` — 9 sub-jobs: e0 illegal-GO refuse+W1C; j1 WS K=64
raw; j2 WS K=128 two-bank act + GO-while-busy refuse; j3 WS K=41 (G-tail
AND lane-tail); j4 WS K=8 N=5 (odd beats → packed-tail padding drop); j5
WS requant; j6 OS K=96=64+32 stripe, ONE shared s8, partial raw drain +
accumulated requant; j7 passthrough framing; j8 INT8 baseline with j1's
EXPECTED W8 image through the LEGACY path (lane off) — same ERO lanes as
j1, which is the lane==INT8-image equivalence AND the identity-mux proof
in the W4 build; plus the W4_CNTI/W4_CNTO cumulative counter checks
(341 int8-equivalent beats from 203 packed + 28 scale beats across the
W4B jobs) and W4_JOB readback:

```
L3 RESULT: cycles=2532 checks=31 errors=0
L3 PASS
```

Every ERO expectation is `w4_lane.gemm_w4b` — i.e.
`wfeed_w4b_to_i8` → `gemm_i8` → (`requant_i32_to_i8`) — so the check is
end-to-end against the golden, not a hardware self-consistency. j8
additionally cross-checks the SAME expected values through the second,
independent hardware path (the F1 self-consistent-oracle lesson).

**G4 — tile mutation gate** (`make -C verif/top/w4 mutants`; f2sim
mutants.sh discipline — clean-binary control, one mutation at a time,
EXIT-trap revert, verdict = the sim's own exit code under pipefail):

```
GATE[control]: exit=0 want=PASS got=PASS  OK
GATE[M1]: exit!=0 want=FAIL got=FAIL  OK (gs slot tap)
GATE[M2]: exit!=0 want=FAIL got=FAIL  OK (pw ceil drop)
GATE[M3]: exit!=0 want=FAIL got=FAIL  OK (cnt_gs phase)
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS
```

Three distinct failure classes: scale/gid value corruption (M1: wrong
serializer slot), stream starvation (M2: odd-tail ceil dropped → the
K=8/N=5 job times out), and the proof-counter channel (M3: W4_CNTI goes
red — the beat-count evidence is live, not decorative).

**G5 — the full verif battery** (`bash scripts/run_full_matrix.sh` on
this tree: golden + perf-model check + all 21 suites + the S8 kvq
replay). Result:

```
== FULL MATRIX DONE fail=0 Mon Aug 10 14:26:09 PDT 2026
```

Zero `FAILED` markers in the run log; the l3 suite inside it (the one my
apex_top/tb_apex_l3 edits rebuild) reports its full profile unchanged:

```
WALKER SPLIT: 24/28 cases walked (CQ-8), 4/28 refused (grouped tier, B1b)
WALKER-MODE L3: all cases passed
MUTATION GATE: PASS (3/3 tile-level mutants caught)
```

gen_status.py ran as the matrix's last step and rewrote STATUS.md; that
regeneration is integration-owned and was DISCARDED from this branch
(`git checkout -- STATUS.md`) — the run log is the evidence. The feeder
unit anchor (`verif/mxe/w4b`, not in the matrix) re-ran green first, on
this tree:

```
W4B SUITE: all gates green (pkg, 6 regimes, exhaustive sweep, mutants)
```

(sweep = 4,063,104 expectations, 8 s8 x 16 codes x 31743 scales.)

## 4. NOT in scope / follow-ons

* **Walker-mode W4** — the walked flow needs the descriptor fields
  W4B_FEEDER.md note 2 already sketches (pw base, gs base, s8, counts)
  and a walker-side W4_JOB push; the walker files are under active fix
  elsewhere and are fenced out of this lane. The CSR window + glue were
  shaped so that hook is additive (a second job-source mux, the
  established wk_/host pattern).
* **DDR image codec** — `make_ddr_image.py`/`make_weight_image.py` grow
  the gs+packed records ("the image IS the wire format" §2.2 hook:
  reader unchanged); the wire framing here is the byte-identical
  contract they emit into, with the scale record fetched first.
* **Enable-gate policy** — W4B stays elaboration- and CSR-disabled by
  default (`W4_LANE=0`); flipping a build point (e.g. the F2 CL) is that
  build owner's call, per W4B_FEEDER.md note 4's full-matrix rule.
* **G=16** — buildable and golden-gated (`W4_G` elaboration guard, unit
  suite runs both), never the default (D-031 freeze).
* **run_tinynpu `--w4-direct`** — stage-5 (golden-pipeline s_w
  refactor), unchanged.
