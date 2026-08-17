# B3 stages 1–3 — native-W4 weight feeder RTL: unit verification GREEN

**Date:** 2026-07-20 · **Branch:** `comp/b3-weight-path` · **Verilator:** 5.044
**Command:** `make -C verif/mxe/w4 all` → **exit 0**
Full verbatim output: [`make_all.log`](make_all.log)

Golden arbiter: `apex_golden.weight_codec` (`wfeed_w4_to_i8`, realization "A"),
landed in stage 0 — see [`../b3_w4_golden/RESULT.md`](../b3_w4_golden/RESULT.md).
Every expected byte in the vectors comes from that model; the TB only drives,
collects and compares.

## What was built

`rtl/mxe/mxe_wfeed_w4.sv` — realization (A), a **pure unpack**: output is the
INT4 code sign-extended to INT8, the group's fp16 scale rides downstream. No
fp16 arithmetic and no scale sideband in the feeder. Add-only:
`mxe_top`/`mxe_ctrl`/`mxe_requant` and the descriptor are untouched, and
`apex_pkg.sv` stays frozen (APEX_VERSION `0x0001_0000`).

`w4_en` is a route/CSR-level input **sampled at job accept and held for the
job**, so a mid-job route change cannot tear a stream (SVA-checked:
`ap_mode_stable`). `w4_en = 0` is a transparent 1:1 passthrough, so the module
can sit unconditionally on the weight path — integration needs only the route
bit, not a datapath mux.

## Simulation — 6 regimes, 0 errors, all bit-exact

```
TB PASS: 17 legal jobs, 2 illegal jobs, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=201)
TB PASS: 17 legal jobs, 2 illegal jobs, 0 mid-op resets, 0 errors (bp=0 stall=0 seed=202)
TB PASS: 220 legal jobs, 0 illegal jobs, 0 mid-op resets, 0 errors (bp=1 stall=1 seed=203)
TB PASS: 40 legal jobs, 0 illegal jobs, 0 mid-op resets, 0 errors (bp=2 stall=2 seed=204)
TB PASS: 40 legal jobs, 0 illegal jobs, 0 mid-op resets, 0 errors (bp=0 stall=2 seed=205)
TB PASS: 0 legal jobs, 0 illegal jobs, 15 mid-op resets, 0 errors (bp=1 stall=0 seed=206)
```

334 legal + 4 illegal jobs + 15 mid-op resets. Lint `-Wall` clean (exit 0,
waivers scoped to the frozen `apex_pkg.sv` only). §5/D-006 checked every cycle
by the bound `w4_job_sva` (15 assertions) and `apex_stream1_sva` on all three
streams.

## PERF — the headline, parsed from TB output

```
PERF (parsed from 'PERF [...]: consumed/emitted' — ARCHITECTURE.md:203-205)
  w4 jobs measured    : 255 (192 even, 63 odd-tail)
  emitted/consumed min: 1.0000x  [F5 beats=1 w4=1: consumed=1 emitted=1]
  emitted/consumed max: 2.0000x  [F99 beats=16 w4=1: consumed=8 emitted=16]
  aggregate           : 31595 emitted / 15829 consumed = 1.9960x
  S6 identity emitted == 2*consumed - (emitted & 1): 255/255 jobs hold
```

**`xw` beats per job are halved** at every even-`KB·N` config, including the
shipped D=64/N=8 (64 → 32). The ratio is reported as a *measured range*, never
as a hand-written "2×", because the unqualified claim is false: for legal odd
`KB·N` (N is legal 1..8) the relation is `emitted = 2·consumed − 1`. 63 of the
255 W4 jobs are odd-tail, and the identity holds on all 255.

Two qualifiers that belong next to this number and are easy to lose:
- The perf model books the *byte* saving at 4.125 b/w = **48.4%**, not 50%
  (`perf/apex_perf_model.py:101–109`) — it counts the fp16 group scales.
  "Halved" is the beat count; 48.4% is the byte count.
- This is a **TB-measured** number. There is still no `xw` beat counter in the
  design (the only `xw_cnt` is an F2 mailbox FIFO free-space register), so the
  on-silicon PERF counter remains to be built at integration.

## Coverage — 26/26 required buckets closed

`COVERAGE PASS: all required buckets hit (26 buckets, 6 logs)`. Highlights:
both modes (255 W4 / 79 passthrough), both tails (63 odd / 271 even), INT4 code
extremes (`code_min` 34, `code_max` 294, `code_neg` 328), all three
backpressure and stall regimes, and `code_zero` 19.

`w4_code_zero` gets a note: it is the all-zero (EPS-scale) tile, the one case
where a dropped sign-extension is **invisible**. It is closed by *designed*
directed vectors, not by incidental all-zero beats in the random set — an
earlier run closed it only by luck (13 incidental hits) and that was fixed.

`w4_rst_idle` (0) is the sole non-required bucket, with a reachability note: an
abort landing in `ST_IDLE` means the job had already completed, so it is not a
mid-op case.

## Mutation gate — 5/5 killed, after relocating one mutant

```
M1_nibble_order_swap      KILLED   1  FAIL [F1 beats=64 w4=1] cyc=112: beat 0
M2_sign_ext_dropped       KILLED   1  FAIL [F1 beats=64 w4=1] cyc=112: beat 0
M3_stuck_w4_en            KILLED   1  FAIL [F11 beats=1 w4=0] cyc=761: beat 0
M4_lane_off_by_one        KILLED   1  FAIL [F1 beats=64 w4=1] cyc=113: beat 0
M5_odd_tail_floor_not_ceil KILLED  1  [4041600000] %Fatal: tb_wfeed_w4_sb.sv:287: Assertion fa
MUTATION GATE: 5/5 mutants killed — checkers proven live
```

**M5 was relocated against measurement, not waived.** The contract named
"`hold_retire` drops `|| last_beat`" as the odd-tail mutant. That mutant was
built and run and **survives every regime** (300 legal jobs, 61 odd-tail). It
is an *equivalent mutant*, not a checker hole: on the cycle the odd tail's
final beat is emitted, `ST_RUN` takes `if (last_beat) state <= ST_WAIT` and
`o_valid` is gated on `ST_RUN`, so the stale padding half cannot be emitted
either way; the next job accept re-clears the flops regardless. The term only
clears two flops one cycle earlier — **no port-level TB can kill it**.

Resolution follows the `seam_feeder_quant` P5 precedent (never weaken a check;
document the reachability): the term is **kept** in the RTL with a reachability
comment, and the gate now targets `need_new`'s ceil
(`(job_beats+1)>>1` → `job_beats>>1`), which IS load-bearing for odd tails and
is killed on the first odd job via the TB's `JOB_TIMEOUT` hang path. That also
proves the TB is *not* blind to the odd tail, and that the hang→kill path works
end to end.

## A note on the stage-3 "exhaustive operand sweep"

The contract asked for a `verif/kvq/fparith`-style exhaustive sweep. For
realization (A) that is **structurally unnecessary**: the feeder is a pure
unpack, so its entire operand domain is the 16 nibble codes — exhausted many
times over by the vectors, with the extremes tracked as coverage buckets. There
is no fp16 scale arithmetic in the datapath to sweep. The fparith pattern
becomes required only if realization (B) is ever adopted.

## Repro

```
git worktree add ../apex-weightpath comp/b3-weight-path
make -C verif/mxe/w4 all      # lint, vectors, build, 6 runs, coverage, mutation
```
