# F2 stage-2 sim — findings

> ## ✅ CLOSED 2026-07-20 (commit 811aa11) — read `docs/results/f2_stage2_sim/RESULT.md`
> All 18 jobs now pass (`checks=27996 fails=0`). **The §2 conclusion below —
> "the K̂ the score path reads back is wrong … the fault is in the K
> store-via-squant-MODE_F16 or the K̂ readback" — was WRONG.**
>
> The real fault was `apex_f2_mailbox.sv:200`: `lane8_beat_t` packs
> `{data[63:0], last}` but the mailbox stored `{last, data}` and *cast* it
> rather than re-ordering, shifting every external **weight** beat right by
> one bit. The phase-F feeder scales were merely the first failure anyone
> looked at; joining failing regop lines to op-script lines showed the first
> wrong value was `s_q`, at the end of the **q** injection, before phase F.
> q, K and V all inject through the weight path — only the loader phase does
> not, which is why the loader beat was the single passing data check. That
> asymmetry was visible in the evidence the whole time and was read as
> "K-specific" instead of "weight-path-specific".
>
> The ruled-out list below stayed valid (nothing in it was the cause). The
> analysis kept below is preserved as the record of the search.

## (historical, session B, 2026-07-18)

Diagnostic pass on `verif/f2sim` while session A owns the executor. No
source of `cl_apex`/`f2sim`/`trace_to_regops` was changed by this pass — the
working tree is clean vs HEAD; these are observations + a repro.

## 1. The descriptor path WORKS from clean source

The prior STATUS "one open item" (descriptor pushed via the mailbox but the
tile stays `idle`, STATUS=0x1) does **not** reproduce from a clean rebuild.
After `make -C verif/f2sim clean build`, `job_s019` dispatches: an illegal
descriptor (opcode 0x101) correctly sets `desc_error` → STATUS bit1
(observed STATUS=0x3, bridge ERRSTK=0x1). Likely cause of the old symptom: a
**stale `obj_d128/`** built before the committed `rst_main_n_sync` reset
wiring. Recommendation: `make clean` is load-bearing here; add it to the run
recipe.

Minimal repro of the reject path (a probe, not a trace job):
```
# build/f2_regops/probe_ds2.regops.jsonl — push opcode-257 desc, expect STATUS bit1
verif/f2sim/obj_d128/f2sim build/f2_regops/probe_ds2.regops.jsonl
# → STATUS (0x1004) = 0x3, ERRSTK (0x1058) = 0x1  ✓ dispatch + reject work
```

## 2. The real remaining blocker: ONE phase-F feeder-scale divergence

`job_s019_L19_h03` (T=20, D=128, CQ-8, G=16) now runs ~25k ops deep and
fails in phase F (scores+softmax+P-requant):
```
FAIL L25387: [3254] got 00013bb3 want 00012c4b (mask 0001ffff)   # FS_W0
```
- `0x3254` = `FS_W0` (feeder-scale capture, `{last, data16}`). got/want
  differ in `data16`: **0x3bb3 vs 0x2c4b** (last=0 both).
- This is **one** bad beat: the host pops **341** fs beats across the job
  (0x3248 advances) but value-checks only **2** of them (EFS); the other
  passes, this one fails. So it is NOT a gross capture-FIFO overflow — the
  host drains interleaved and the tile backpressures via `fs_ready=!fs_full`.

**Ruled out this pass:** reset wiring (fixed in source), KVQ_G mismatch
(CL builds G=16 = trace INFO_GROUP, phase-A CSR sanity passes), gross
capture-FIFO overflow (interleaved drain, 341 pops complete).

### Root cause NARROWED to the K-store/read path (session B, follow-up)

Full fs-emit-vs-golden diff (tile `u_mb.fs_*` beats from the committed
`TRACE=1` VCD vs the 341 golden `EFS` values from `g3.core_case`):

- **341 emitted, 341 golden — identical count.** NOT an off-by-one/capture
  bug (an insert/drop would change the count). Hypothesis 1 REFUTED.
- **Only beat #0 matches; beats #1..340 ALL diverge.** Emitted is **not a
  permutation** of golden (sorted lists differ; only 1 of 33 unique emitted
  values appears anywhere in golden). So it is a genuine **compute/data**
  divergence, not reordering.
- Mapping beats to the op-script: **beat #0 = the loader ACTIVATION feeder
  scale** (op-line 25, `EFS 2800 1`, after the `XR`/`GR` loader token) — it
  is correct. **Beat #1 = the first K̂ SCORE-feeder scale** (2nd `EFS`,
  op-line ~17218, golden `0x35ae`; tile emits `0x43ad`). Everything from the
  first K̂ requant on is wrong.
- Therefore the K̂ the score path reads back is wrong in the CL/mailbox
  build. The K/V were injected via **"squant MODE_F16 → KVQ" (CQ-8 store)**
  right after beat #0; the score path then re-quantizes K̂ read from the KVQ
  engine. The activation feeder (beat #0) is unrelated to K and is fine — so
  the fault is in the **K store-via-squant-MODE_F16 or the K̂ readback under
  mailbox drive**, not the feeder itself.
- **REFUTED param hypothesis:** setting `FEED_ROWS_MAX=STAGE_R_MAX=31` in the
  CL `apex_top` (to match the L3 reference build, which does pass 31 vs the
  apex_top default 16) does **not** change the emitted values — same 341
  beats, same `0x43ad`. It is a correctness-alignment worth doing anyway (the
  choreography is generated for a 31-row build) but it is not this bug.

**Next probe for session A** (KVQ-store/read side, your lane): compare the
KVQ engine SRAM record contents (or the `kv_rdata` K̂ readback stream)
after the mailbox store against `cq_codec` golden records for K row 0 — that
splits "store wrote wrong records" from "readback/feed sequencing wrong".
Prime suspects in the store path: the `squant MODE_F16` sideband (`QS`
word) / `QJOB` mode packing in `trace_to_regops.py` vs `tb_apex_l3.sv`
`drive_qj`, and the `KVW`/`kv_*` AXI-Lite forwarding order.

Repro:
```
make -C verif/f2sim clean build
make -C verif/f2sim run   # first FAIL at the 2nd fs beat = first K̂ requant
make -C verif/f2sim build TRACE=1 && verif/f2sim/obj_d128/f2sim <regops>
# then diff u_mb.fs_* emit stream vs g3.core_case EFS list (session-B method)
```

## 3. Host executor landed (session B lane, does not touch the above)

`scripts/fpga/f2/f2_host_run.py` — the real-silicon counterpart to
`sim_main.cpp`: executes the SAME regops.jsonl over BAR0 via `fpga_pci`.
Covers every op in the trace (note/w/pw/jf/poll/r/rn — verified against
job_s019: 506 value-checks, 0 unknown ops). Ready to replay on the loaded
AGFI the moment the sim goes green — the last mile to attention jobs on real
F2 silicon.
