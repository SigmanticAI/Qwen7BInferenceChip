# F2 stage-2 simulation — all 18 real-model attention jobs replay bit-exact

**Date:** 2026-07-20 · **Scope:** simulation only (Verilator). No hardware
claim is made here — see "What is NOT claimed" at the bottom.

The stage-2 mailbox CL (`scripts/fpga/f2/cl_apex/`) driven at its OCL
AXI-Lite pins by the compiled register-op stream now replays **every one of
the 18 committed S8 Qwen2.5-7B trace jobs** with zero mismatches against the
golden model.

```
F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS
```

Verbatim run: [`f2sim_all18.log`](f2sim_all18.log) (`make -C verif/f2sim run`,
exit 0). 1,862,470 register ops, 16,188,576 simulated cycles.

## What runs

`docs/results/s8_7b_token/artifact_trace/job_*.npz` (real Qwen2.5-7B
attention jobs, T=20..128, D=128, CQ-8) → `verif/top/l3` `core_case()` (the
**verified** L3 choreography, imported and called verbatim) → op-script →
`scripts/fpga/f2/trace_to_regops.py` → regops JSONL → executed at the OCL
pins by `verif/f2sim/sim_main.cpp`. The identical regops file is what
`scripts/fpga/f2/f2_host_run.py` replays over BAR0 on a live F2 instance —
one artifact, two executors.

Checks are the binding result checks (KVQ record scale taps, feeder scales,
softmax scales, INT8 result beats, CSR/KVQ registers, sticky errors), plus
the polls, which must all succeed for a job to complete. The passive debug
taps (TAPF16/TAPSC/TAPPR) the L3 TB additionally checks are not observable
through the mailbox and are skipped-and-counted in the manifest (disclosed
carve-out, `trace_to_regops.py` header).

## The bug that was blocking it

**`apex_f2_mailbox.sv:200` mis-ordered the fields of `lane8_beat_t` when
driving the external weight stream**, so every weight beat reached the MXE
shifted right by one bit.

`lane8_beat_t` (`rtl/apex_pkg.sv:57-60`) is `{logic [63:0] data; logic
last;}` — as a packed vector, `data` is bits [64:1] and `last` is bit [0].
The mailbox stores its FIFO word the other way round, `{last, data}`
(`apex_f2_mailbox.sv:176,319`), and then *cast* it instead of re-ordering it:

```systemverilog
assign xw_beat = lane8_beat_t'({xw_head[64], xw_head[63:0]});   // WRONG
```

The right-hand side is a no-op re-assembly of the stored word, so the tile
received `data = {last, data[63:1]}` and `last = data[0]`. The golden L3
driver builds the same port correctly, data-first
(`verif/top/l3/tb_apex_l3.sv:195`: `lane8_beat_t'({xw_data, 1'b0})`).

Why it presented the way it did: `last` on a lane8 beat is informational
(`mxe_ctrl` sinks it in `unused_fields`), so no FSM saw the bogus flag — the
handshakes, beat counts and every control/CSR/poll check stayed correct while
the payload was numerically wrong. The loader phase is the one phase that
uses no weight beats (`XR`/`GR` + `FJOB` only), which is why its feeder-scale
check was the single data check that passed.

Fix: re-order rather than cast (`apex_f2_mailbox.sv:200`).

## Also fixed in this pass

- **`cl_apex.sv` tile config did not match the verified L3 reference build.**
  `KVQ_DEPTH` was 128 where `tb_apex_l3.sv` builds 256, so every job with
  T>64 stalled polling KVQ `OCC` (it needs 2·T records; F-4 in
  `gen_l3_vectors.py`). `FEED_ROWS_MAX`/`STAGE_R_MAX` were left at the
  `apex_top` defaults of 16 where the L3 reference passes 31, though the
  choreography is generated for a 31-row build. All three now derive from
  named localparams so they cannot drift from the reference again. **This
  changes the required AFI:** the first-light AGFI was built D=64/G=128/
  DEPTH=128; stage 2 needs a D=128/G=16/DEPTH=256 build.
- **The compiler never audited `MB_STATUS` (0x2F0).** The mailbox sets a
  sticky bit on each of its twelve silent-drop paths (6 stream FIFOs, 6 job
  registers) and the file header documents the register as the audit hook,
  but nothing read it — so a dropped beat and a miscomputed value were
  indistinguishable in the log. `trace_to_regops.py` now emits the check in
  every job's `DONE` block; all 18 jobs read 0.
- **A job fire arriving in the same cycle its predecessor is consumed was
  dropped** (`!*_valid` was evaluated against the pre-clear value). Now
  accepted when `!valid || ready`; only a write onto a pending-and-unconsumed
  pulse is flagged. Not observed in these runs — a hardware-timing hardening.
- **`rt_imp_hi` reset default** was `0x0000` where the golden driver powers up
  at `0xFFFF` (`tb_apex_l3.sv:819`). Inert for these CQ-8 jobs (no
  `tip_override`), but a real divergence from the reference driver.
- **`sim_main.cpp` aborted on teardown** (SIGABRT, exit 134) because the
  file-scope `unique_ptr`s destructed after the Verilated context was gone —
  the harness printed PASS but exited nonzero, so it could never gate.
- **`make -C verif/f2sim run` now runs all 18 jobs**, not just the first.

## Cross-checks

- The golden model is independently confirmed: `trace_to_regops.compile_job`
  asserts `r.o8 == trace.o8` against the stored S8 trace for every job.
- The same op-script for `job_s019_L19_h03`, run through the **verified L3
  testbench** rather than the mailbox, passes: `L3 RESULT: cycles=146100
  checks=5541 errors=0 / L3 PASS`. That A/B is what proved the script and the
  tile RTL were both correct and localised the fault to the F2 path.
- `verilator --lint-only -Wall` on the CL: zero errors, zero width warnings
  in our sources (remaining warnings are kit-internal shell templates).

## What is NOT claimed

Nothing here ran on an FPGA. This is a Verilator simulation of the CL. The
stage-2 hardware claim requires a D=128/G=16/DEPTH=256 AFI build plus a
committed `f2_host_run.py` session log from a live F2 instance, under the
same anti-fabrication rule as everything else: no PASS without pasted output.
