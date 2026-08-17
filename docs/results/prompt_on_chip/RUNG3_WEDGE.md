# RUNG-3 WEDGE — the walked FFN→DOWN seam, cornered in waves and closed

**Date:** 2026-08-10 · **Status: CLOSED — walked FFN→DOWN bit-exact in sim,
3/3 discriminators RED, rung-1/2 regressions green.** · Branch
`fix/rung3-wedge` (from `comp/prompt-b-c` @ 05ea8c5) · Sim only — no AWS,
no silicon claim.

## The symptom (as inherited)

`scripts/fpga/f2/walk_ffn_toy.py --down` (the rung-3 composed back-half:
`{FPROJ, FFN, DOWN, RES2}`, NCHUNK=1) wedges: the walker parks at **PC_WDJ
with nrem=24** (job 6 of 8 of the DOWN projection), `W_STATUS` polls
**busy-clean 0x3001** forever (busy + fmt-sup, no error bits), and the fuel
FIFO holds undelivered Wd beats (`FUELAUDIT stat=00000608`, `fifo_empty=0`).
DOWN-only (`walk_fuel_proj.py rund7`) passes, so DOWN itself is healthy —
the FFN→DOWN **seam** is the fault. CSR polling cannot fingerprint it: a
stalled poll aborts the executor.

## Method: the Verilator twin, cycle-stamped and windowed

All on a fresh worktree + fresh `--Mdir` twins verilated from HEAD (the
walk_fuel_proj build-hygiene rule):

```
make -C verif/f2sim build D=64 DDR=1 OBJ=obj_ffn_b64_ddr1 \
  VFLAGS_EXTRA="+define+APEX_CL_DM=896 +define+APEX_CL_GQA=2 \
                +define+APEX_CL_QSTAGE=14 +define+APEX_CL_DMODEL=64"
python3 scripts/fpga/f2/walk_ffn_toy.py --down       # reproduces: rc=1,
#   poll stall L614: [1068] last 00003001 want 00000000/00000001
#   FUELAUDIT err=00000000 stat=00000608 -> FAIL
```

Two sim-side debug affordances were added for the hunt (both inert in
shipped twins):

* `sim_main.cpp`: FST support (`--trace-fst` → `wave.fst`) and a
  **windowed dump** — `+trace_from_cyc=N +trace_to_cyc=N` (host-clock
  cycles, the CYCMARK counter) so a wedge run's 16M-cycle poll spin does
  not drown the seam. VCD path (`--trace`) untouched; both are `VM_TRACE`-
  guarded.
* `seq_layer_walker2.sv` `APEX_E6_DBG` block (debug-only, never in a
  shipped twin): a `dbg_cyc` tile-cycle counter appended to the state-
  transition displays, to correlate `[W2DBG]` lines with the trace window.

A debug twin (`+define+APEX_E6_DBG`) put the seam at tile cycles
125712–126218 (tile_div=5, kick at host cyc 1,230,229): the DOWN chain
dispatches **cleanly** — PC_LVLD level-arm, RES2 push, JC_DOWN + deq push,
Wd fetch, S2_PDW passes tile_idle in ONE cycle, S2_PSJ accepted — then five
`DS … rq=1` descriptors retire at the fuel-streaming cadence (~158 tile
cyc/job) and the walker parks in **S2_PAE** (act-emit wait) with nrem=24.
An FST/VCD window (host cyc 1,256,900–1,264,500) over the seam pinned every
handshake.

## The stall, read off the wave (times are VCD vtime)

```
t2515456  S2_LVL (PC_LVLD): ser_dst 2→1, resid_arm=1, fsrc_ext→SWGP
t2515516  RES2 job accepted: u_resid.st=1, cols_q=0x40      ← consumer armed
t2515536  S2_JCA writes JC_DOWN: jc_data=0x309fb95a
t2515556  u_ldeq.jb_comp=0x309fb95a → comp_legal=0          ← low 13 bits ≠ 0
t2515576  deq push: jb_valid=1 (jb_cols=0x40, jb_ready=1)
t2515596  jb accepted → err_jb=1, stk_jb=1 — job CONSUMED, unit stays
          ST_IDLE (busy=0, cols_q=0)                        ← NO deq consumer
t2518937  serializer egress starts beat 0: ldq_iv=1, lane_no 0→4…
t2518977  ldq_ir=0 (deq input skid full) — egress frozen at lane_no=4
t2525277  ser.in_r=0 — ingress skid full after 3 accepted o8 beats
t2528476  4th o8 beat parks at ser.in_v=1 — steady state: MXE result path
          jammed, descriptor 6's act EMIT starves, S2_PAE forever
```

`apex_layer_deq.sv` ST_IDLE (the reject-and-consume arm):

```systemverilog
wire comp_legal = (jb_comp[31] == 1'b0)                // positive
              && (c_e8 != 8'h00) && (c_e8 != 8'hFF)    // normal
              && (c_m[12:0] == 13'h0);                 // fp16-grade (C2)
...
if (!comp_legal || (jb_cols == '0)) begin
  err_jb <= 1'b1;  stk_jb <= 1'b1;   // reject: consumed, no state
end
```

The backpressure ladder then accounts for every inherited symptom: no deq
consumer → serializer egress freezes (skid depth ≈ 4 words) → ingress
refuses after ~3 beats → MXE requant/result skids fill → descriptor 6
cannot retire → `apex_stage_buf` D-006 holds `aj_ready` → walker parks in
S2_PAE at **nrem=24** → `W_STATUS` busy-clean (the walker's error surface
was never involved) → the un-consumed Wd records sit in the **fuel FIFO**
(`stat=0x608`). The `nrem=24` fingerprint is just "5 jobs' beats fit in the
skid chain, the 6th doesn't."

## Root cause — one ungraded composite (host-side, rung-3 toy only)

`walk_ffn_toy.py` computed the DOWN epilogue composite RAW:

```python
comp_d = np.float32(s_p * s_wd * float(1 << shift) / float(scale))   # BUG
```

The C2 contract (and the deq unit's accept fence) requires JOBC composites
to be **fp16-graded fp32** — the grade is what makes the tile's
`o8 × comp` product EXACT in fp32 (8-bit code × 11-bit graded significand
≤ 24-bit significand; an ungraded 24-bit significand would be inexact —
"exact or refused", so it is refused). Both proven epilogue paths grade:
`walk_fuel_proj.py:749` (E-6 OPROJ, `fmt.grade_f32`) and `:1242` (rund7
DOWN). The rung-3 toy alone missed the grade — its subject's word
`0x309fb95a` carries mantissa low bits `0x195a`.

The tile behaved CORRECTLY throughout: it refused the illegal job loudly at
the unit (`job_error` pulse + sticky on the LAYER surface). What made the
refusal present as a silent wedge is a **walker observability gap**, named
below as a follow-on.

**Fix (one line + golden composed with the graded value it already
flows through):**

```python
comp_d = np.float32(tf.f16_grade(
    s_p * s_wd * float(1 << shift) / float(scale)))
```

## Result

```
$ python3 scripts/fpga/f2/walk_ffn_toy.py --down --discriminate
[rung3] subject fits at x_shift=12
[walk_ffn1 @sim] rc=0 ok=True caps=137 grade=True 7.9s
  disc walk_off  : REFUSED=True (DESC sticky)
  disc jc_swap   : predicted-move=True landed-on-prediction=True -> RED=True
  disc d_comp    : factor=2.0 landed-on-poisoned-r2=True ffn-half-clean=True -> RED=True
FFN-WALK RUNG 1 @sim: PASS
```

`grade=True` covers the full composed back-half: the per-chunk product
scales (`fs`), the walker-staged product codes (`p_codes`), and the **r2
row** — `f16(r1 + deq(Wd @ product))`, the DOWN+RES2 epilogue's final
product, against golden's own composition. (The graded composite also
changes the deq magnitudes slightly — the x_shift search now lands at 12,
not 19.)

Discriminators (all REQUIRED red, extended for the rung-3 composition):

* **walk_off** — for `--down` builds the poison clears `EN_DOWN` but keeps
  `RES2`: the E-7b pairing fence (`fp_dwn == RES2` in `fp_mask_ok`) must
  refuse at the kick — busy never rises, ESTK carries DESC. It does.
  (Rung-1 builds keep the original EN_FFN-clear arm byte-identically.)
* **jc_swap** — descriptor comp_g/comp_u exchanged; the golden now extends
  THROUGH the epilogue: the tile requants `Wd @ p_codes_swapped` under the
  host-loaded `rq_d` and deqs under the descriptor's `comp_d`, so r2 is
  predicted exactly. The run must land on the swapped prediction (fs,
  p_codes AND r2). It does. (If a swapped subject violates the residual
  window, the arm instead requires the tile's loud refusal — both shapes
  are red; this subject fits.)
* **d_comp** (new, rung-3-specific) — `JC_DOWN` poisoned **graded**
  (×2; ÷2 fallback; first factor whose prediction fits and moves): r2 must
  land EXACTLY on the poisoned prediction while fs/p_codes stay on the
  healthy golden — the C2 composite seam localized to the descriptor word.
  It does. (The UNGRADED word is the wedge class this rung closed; it
  cannot fly as a graded arm — it is refused at the unit, which is the
  finding.)

Regressions: rung 1 (`--discriminate`, NCHUNK=1) and rung 2 (`--dffn 128`,
NCHUNK=2) both PASS unchanged.

## Verif battery

The full evidence battery (the `run_full_matrix.sh` suite list + the
walker lanes `verif/seq_walker all`, `verif/top/l4 levels|compose`, + the
f2sim executor gates `run behsmoke mutants capgate`, + the S8 KVQ replay)
was re-run serially on this tree, 2026-08-10: **`BATTERY DONE fail=0`**,
zero FAILED lines across 41 suite invocations. Key verdicts, verbatim:

```
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS
CAPGATE: PASS (job=job_s019_L19_h03 caps=505 values_matched=505/505 tile_div=5 executor=sim:obj_d128_ddr1)
S8 RTL REPLAY [artifact]: real-model KVQ rows bit-exact in RTL
== BATTERY DONE fail=0 Mon Aug 10 14:21:16 PDT 2026
```

(The f2sim gates rebuild and run the executor this session touched —
`sim_main.cpp` compiles and gates identically with tracing off.)

## Named follow-on (fenced, not skirted): the silent-refusal window

A unit-level job REJECT during a walked FPROJ window is invisible to the
walker: the push handshake succeeds (reject-and-consume), the LAYER_STATUS
sticky IS set, but nothing samples that surface mid-walk and the wedged
walk never reaches the host's post-walk poll. §A-1 calls this the worst
failure shape, and the walker already refuses every SHAPE-illegal mask at
S2_CHECK — but composite VALUE legality is data-dependent and only the
unit sees it. The design fix is walker-side: sample the unit job-error
stickies at the projection window's push edges and abort to S2_ERR with a
new code (busy falls, the poll fails loudly with ESTK/ECODE). NOT done
here: it extends the frozen walker error enum and every byte-identity gate
around it, for a defect that was host-side. The host-side rule that stands
today: **every JOBC composite that flies must be `grade_f32`/`f16_grade`
output** — the two proven epilogue paths and (now) the rung-3 toy all
comply.

## Repro / artifacts

* Wedge (pre-fix): `git stash` the toy fix, run `--down` — poll stall at
  L614, `[1068] last 00003001`, FUELAUDIT `stat=00000608`.
* Waves: `make -C verif/f2sim build D=64 DDR=1 OBJ=obj_ffn_tr
  VFLAGS_EXTRA="<IMG_05B defines> +define+APEX_E6_DBG --trace-fst
  --trace-depth 10"`, then run the wedge regops with
  `+trace_from_cyc=1256900 +trace_to_cyc=1264500` → `wave.fst` (~210 KB);
  `--trace` for the VCD twin if a text parse is wanted.
* Green: `python3 scripts/fpga/f2/walk_ffn_toy.py --down --discriminate`
  (7.9 s main walk + 3 arms, sim, tile_div=5).
