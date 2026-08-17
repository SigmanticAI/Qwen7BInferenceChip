# KVQ block — INDEPENDENT verification (verif/kvq/sb) + smoke reproduction

> **RE-AUDIT ADDENDUM (2026-07-08, adversarial re-audit of the D-020 fix):
> B-1..B-4 below are FIXED and the fixes HOLD — verdict upgraded to PASS,
> KVQ earns vendored-verified status.** Evidence: `verif/kvq/audit/RESULT.md`
> (both suites clean-rebuilt: smoke exit 0 with V0-parity counts verbatim
> 49923/54020/54020/28130/77403/32665/54019, sb exit 0 incl. `cq4p
> checks=1497 fails=0` and reset `checks=979 fails=0`; NEW attack suite:
> 260 randomized soft resets across every FSM state × pending-beat ×
> mid-token × mid-group × collision conditions, each followed by a bit-exact
> clean job; −0.0 outlier/non-outlier directed at D=16 AND D=64; fix-revert
> mutations of B-1/B-2/B-4 all CAUGHT; `make all` in verif/kvq/audit → exit
> 0). The original FAIL report below is preserved as the evidence trail.

**Role:** independent check of the implementer's `rtl/kvq/` + `verif/kvq/smoke/`
claims. Nothing from their suite was trusted; the arbiter for every numeric
expectation is `golden/apex_golden` (cq_codec.py, D-001/D-013 lineage) via
a fresh generator, TB, SVA pack and coverage gate in `verif/kvq/sb/`.

**Verdict: FAIL** — the datapath is bit-exact everywhere I could push it
(9,163 scoreboard checks across 4 configurations under backpressure storms and
stalled feeds, 765 randomized/directed transactions, all mutations caught),
**but mid-operation SOFT reset is broken four ways** (B-1..B-3) and the CQ-4+
outlier lane drops the sign of −0.0 (B-4). Their own smoke — which they
honestly caveated as having "no mid-op reset test" — reproduces 100%.

**Date:** 2026-07-08 · Verilator 5.044 (`--binary --timing --assert`, `-Wall`)
· single-command gate: `make all` in `verif/kvq/sb/` (exits non-zero; the only
fail markers are the two genuine DUT findings: `fail_cq4p`, `fail_reset`).

---

## 1. Their smoke: reproduced from clean — all claims verified

`make clean && make smoke` in `verif/kvq/smoke/` → **exit 0**
(console: `verif/kvq/smoke/logs_smoke_repro.txt`, logs under `smoke/logs/`).

| claim | reproduced |
|---|---|
| lint `-Wall` clean, kvq_engine.sv zero waivers | ✔ (`logs/lint.log`; waiver file touches only `rtl/kvq/vendor/*`) |
| 7 parity configs, check counts IDENTICAL to V0 | ✔ `49923 / 54020 / 54020 / 28130 / 77403 / 32665 / 54019`, all `fails=0` |
| V0.2 stall flow | ✔ `KVQ STALL [stall_kvq]: PASS` |
| 4 regressions reproduce-then-fix (alias/rwait/collide/flush) | ✔ all `BUG REPRODUCED` on baseline, `PASS` on kvq |
| flush oracle self-check vs frozen T=70 set | ✔ `GEN SELF-CHECK` bit-identical |

## 2. Independent suite (verif/kvq/sb) — what it is

House pattern of `verif/mxe/sb`: `gen_kvq_sb_vectors.py` simulates the
engine's storage state with the Python golden and emits stimulus + the FULL
expected §4 record image per address + expected fp32 readback; `tb_kvq_sb.sv`
only drives/collects/compares (script-driven; negedge stimulus, posedge
observation, `$fatal` watchdog, per-cycle FAIL tags). `kvq_sb_sva.sv` is a
KVQ-shaped §5 pack bound into `kvq_engine` in every build (the MXE
`verif/common/apex_stream_sva.svh` is descriptor/job-shaped — binding it to
KVQ's AXI surface would be vacuous; I verified that judgement and lifted the
same rules instead: m/s stream stability, tlast exactly on beat D−1, plus
"m_valid only in ST_OUTPUT/ST_OFLUSH" bound to the internal FSM state).

Configurations (all `-Wall`, vendor-scoped waivers only):
`cq4` D=16/CQ-4/G=4 · `cq8` D=16/CQ-8 · `cq4p` D=16/CQ-4+/K=2 (mask ROM)
· `d64` D=64/CQ-4/G=8. 765 randomized/directed transactions
(V=224, K=222, R=319; `build/gen.log`), each config re-run under
backpressure ~75%/storm ~6% duty and short/bursty feed-gap stalls.

**Functional matrix (from `make all`, `build/run_*.log`):**

```
CONFIG cq4 : checks=4309 fails=0   (bp=1 stall=1, bp=0 stall=0, bp=2 stall=2)
CONFIG cq8 : checks=4737 fails=0   (bp=1 stall=1)
CONFIG d64 : checks=3708 fails=0   (bp=1 stall=2)
CONFIG cq4p: checks=1497 fails=1   (bp=1 stall=1)  <- B-4 below
framing    : checks=148  fails=0
```

Directed content: zero / subnormal / max-normal / negative-amax tokens,
guaranteed RNE-tie tokens (amax=qmax → scale=1.0 exactly), EPS-clamp hits,
partial-group flushes of size 1, 2, G−1, mid-token-latched flush, flush on the
G-th token (pending-flush discard), flush no-op in IDLE and during KEMIT,
overwrites (incl. key-record→value-record tag flip), full-SRAM occupancy +
`sram_full`, RD_ERR/IRQ/W1C with mask 1 and 0.

## 3. Bugs found (do-not-fix; traces below) — all in the SOFT-reset/outlier area

### B-1 — CTRL.soft_reset is LOST when it lands in a single-cycle FSM state
`kvq_engine.sv:482-489` assigns `state <= ST_IDLE` on `ctrl_reset`, but the
`case (state)` that follows in the same always block executes unconditionally;
any branch that assigns `state` overrides the reset (last NBA wins).
Deterministically reproduced (final key beat / READ_ADDR write and the CTRL
write accepted on back-to-back posedges):

```
FAIL cyc=8542: X k=2 p=8: soft reset LOST in collision — FSM state=9 5 cycles later (case-override defeats ctrl_reset)
FAIL cyc=8545: X k=2 p=8: STATUS.idle=1 while FSM state=9 after lost soft reset
FAIL cyc=8975: X k=3 p=4: soft reset LOST in collision — FSM state=6 5 cycles later (case-override defeats ctrl_reset)
FAIL cyc=8997: X k=3 p=4: 16 m_axis beat(s) delivered after colliding soft reset
```

ST_KFEED case: the engine silently continues the key group (ST_KACCEPT) while
`STATUS.idle` reads 1 — a host that trusts STATUS then streams a "new" token
which is swallowed into the aborted group. ST_RLOAD case: the full 16-beat
read burst streams out AFTER the acknowledged soft reset. The same override
exists for ST_COLLECT/ST_KCOLLECT beat-coincident cases and ST_COMPRESS→STORE
(`state <= ST_STORE` at `cqv_out_valid`) — a soft reset there still writes the
record.

### B-2 — soft reset does not reset the datapaths, and no status bit covers them
`ctrl_reset` clears only top-FSM state; `cq_value_path` / `cq_key_path` /
`amax_unit` keep walking, while `STATUS.idle` immediately reads 1.
Two consequences, both reproduced:

```
FAIL cyc=4438: P addr=32 ridx=186: record mismatch          (value path)
NOTE [X k=1 p=2] cyc=4467: IMMEDIATE reuse after soft reset failed (datapath not reset); retrying after quiesce
FAIL cyc=10096: P addr=50 ridx=194: record mismatch         (key path, X k=4 p=10 + Y probe)
FAIL cyc=10125: P addr=51 ridx=195: record mismatch
```

(a) reset during ST_COMPRESS, then a new token: the still-running
`cq_value_path` finishes the ABORTED token and its codes/scale are stored at
the NEW token's address (bit-exact wrong record + wrong readback). After a
quiesce delay the same round-trip passes — proving the root cause is the
un-reset datapath, not the data. (b) reset during ST_KEMIT, then a new key
group: the still-walking `cq_key_path` ignores the new group's first
`in_valid` (and its `group_start` buffer clear), corrupting every record of
the group. Nothing host-visible reports datapath business — `STATUS.idle` is
top-FSM-only.

### B-3 — soft reset mid-read-burst retracts a pending m_axis beat (§5 violation)
Soft reset parked in ST_OUTPUT / ST_OFLUSH (beat pending, `tready=0`):
`state→ST_IDLE` then IDLE clears `m_axis_kv_tvalid` without the beat ever
being accepted — valid retraction + burst never terminates with `tlast`
(downstream skid/framing corruption):

```
SVA-VIOL [ap_m_only_output_phase] m_valid while FSM state=0 (not OUTPUT/OFLUSH)
SVA-VIOL [ap_m_stable] m_axis valid retracted or data/last unstable under backpressure
STABILITY-VIOL cyc=7488: m_axis retracted/changed under backpressure (valid=0 ...)   (p6)
STABILITY-VIOL cyc=8356: ... last=0/1                                                (p11: the pending TLAST beat is killed)
```

Whether a soft reset may abort an in-flight burst is unspecified in
ARCHITECTURE.md §7 — but §5 has no reset carve-out, and XBR skids assume no
retraction. Needs a spec decision + implementation (see design notes).

### B-4 — CQ-4+ outlier identity lane loses the sign of −0.0
Directed −0.0 (fp16 `0x8000`) in an outlier channel; record stores the raw
fp16 correctly (P check passes for the field), but readback reconstructs the
outlier via `cq_dequant_unit_syn` (code=+1 × scale=raw), whose zero-flush
returns +0 (`cq_units_syn.sv:78`, `xhat_f32 = zero ? 32'h0000_0000 : ...`):

```
FAIL cyc=2094: R addr=8 beat=4: got 00000000 exp 80000000 (hidx=3)   (run_cq4p.log)
```

Golden (`decompress_keys`: identity fp16→fp32 widen) and IEEE-754
(1 × −0.0 = −0.0) both say `0x80000000`. Negative subnormal `0x8001` reads
back correctly — ONLY exact −0.0 is affected. Tiny numerically, but a real
deviation from the §4 outlier identity contract / D-010, and invisible to the
frozen vectors (no −0.0 sidecar entries — V0 parity could not catch it).

### Non-bug observations
- **Hard reset (rst_n): clean in ALL 11 FSM phases** (targeted hierarchically;
  full recovery + occupancy=0 + bit-exact round-trip each time).
- Soft reset in parked ST_COLLECT/ST_KCOLLECT/ST_KACCEPT (no coincident beat):
  honored correctly.
- Malformed framing (no hangs, always recoverable): early tlast → garbage
  record, engine self-returns to IDLE, next op clean; missing tlast →
  **bit-exact record** (beat count alone completes the token — tlast is
  informational, matching the ISA); LATE tlast → the stray beat starts a
  phantom token and parks the engine (`STATUS.idle=0`, observable), soft reset
  recovers. Note `tlast` on beat 0 is silently ignored in ST_IDLE — framing
  desync is only detectable via STATUS, worth a spec note.

## 4. Coverage (gated in `make all`) + mutation checks

`COVERAGE GATE: PASS — all required buckets closed` (60 required buckets:
volume, numeric conditions incl. 116 RNE-tie hits, structural flush shapes,
error paths, bp/stall modes, 11 hard-reset + 7 soft-reset phases + 2
deterministic collisions; `build/coverage.txt`). Reachability notes proven,
not hand-waved: exhaustive sweep over ALL finite fp16 amax magnitudes shows
`rne(x/s) ≤ qmax` for both tiers, so the quant **saturation clamp and
qmin (−8/−128) are unreachable through the engine top** (scale is derived from
the same data's amax); soft reset cannot be *parked* in the 1-cycle states
STORE/RLOAD/RWAIT/KFEED — covered by the collision tests instead.

Mutation checks (each: copy RTL → sed one bug → rebuild → run MUST fail):

| mutation | result |
|---|---|
| (a) data corruption — value-record payload bit 24 XOR | **CAUGHT** (61 fails, first at cyc=105 `P addr=0: record mismatch`) |
| (b) KVQ contract — ROUNDING DRIFT: RNE tie → round-half-up in `cq_quant_unit_syn` | **CAUGHT** (10 fails, exactly the tie beats: `got 40400000 exp 40000000` = 2.5→3 vs 2) |
| (c) framing — tlast also on beat 1 | **CAUGHT** by SVA `ap_m_last_pos` + TB (118 fails) |

## 5. Honest caveats
- The reset-run FAIL counts (55) are dominated by per-beat readback prints of
  the two corrupted post-soft-reset records; the independent findings are the
  four B-1..B-3 mechanisms listed above.
- `s_axis` feed-gap adversary initially produced 1148 false fails from MY
  driver holding stale `tvalid` through gaps (duplicate-beat acceptance —
  legal DUT behavior); fixed (valid deasserted post-accept during gaps) and
  the matrix re-run from clean. Lesson recorded so nobody mistakes the interim
  logs for DUT bugs.
- Soft-reset collision tests are timing-deterministic for THIS RTL (AXI write
  accepted back-to-back); the TB prints the observed phase and skips gracefully
  if the collision is missed.
- `STATUS.idle`-lie evidence is whitebox (hierarchical `dut.state`); a purely
  black-box host would see it as silent data corruption.
- No Icarus cross-check; no synthesis/timing claims (§0: simulation sign-off).
- Their smoke was reproduced ONCE from clean (deterministic vectors); I did
  not re-run their 7 parity configs under different seeds — their TB is
  fixed-stimulus by design.

## 6. Repro
```
cd verif/kvq/smoke && make clean && make smoke        # their gate: exit 0
cd verif/kvq/sb    && make clean && make all          # exit != 0
#   -> build/fail_cq4p (B-4), build/fail_reset (B-1..B-3), everything else green
#   waves: run any config with +dump
```

## 7. F1.5 area rework (2026-07-12) — serialized codec, all suites re-run green

The KVQ codec was reworked for area with NUMERICS FROZEN (every result still
bit-exact vs golden/apex_golden):

- `cq_fp_pkg`: the general 64/64 `$div`+`$mul` RNE divide (`rne_div_u`) was
  replaced by `rne_div_bounded`, a 12-step restoring compare-subtract array
  (zero `$div`/`$mul` cells). Exact where callers need exactness (scale
  significand in [1024,2048]; quant clamps to |code| <= 128); quotients
  >= 2^12 saturate and clamp identically. Re-proven exhaustively: prove.py
  (Python mirror vs golden) + tb_fparith (8,748,634 HW checks) + both seeded
  errors still caught (mut_tie retargeted to the new single tie line; the
  sb `mut_round` sed and this file's §6 exit-code note now refer to the
  fixed-and-green suite state).
- `cq_value_path` / `cq_key_path`: the D-parallel `cq_quant_unit` /
  `cq_scale_unit` lanes (the dominant F1 area item: 320 quant + 131 scale
  instances tile-wide) are now ONE unit each per path, walked one channel
  per cycle. Latency changes (value compress ~2 -> 2D+1 cycles; key group
  emit 1 -> D+2 cycles/token; amax sweep 1 -> 2 cycles/token; scale phase
  1 -> D cycles); all consumers are handshake-driven, contracts unchanged.
- `residual_buffer`: read port registered -> BRAM-inferable (was 131,072
  discrete FFs + a 128:1x1024b mux per key engine at ship config).
  tb_residual_alias updated to clocked sampling.
- `kvq_engine`: read-side dequant operand capture registers replaced by
  combinational views of the record SRAM's registered rd_data (stable for
  the whole burst); `amax_unit.sv` deleted (dead after serialization).
- Makefile hygiene: pipefail added to every suite Makefile with piped
  recipes (a masked TIER-0 build failure was running a stale binary);
  the audit Makefile's clobbered `KVQ_CORES :=` opener repaired.

Re-run from clean after the rework: smoke (8 parity configs, exact recorded
check counts, + 4 regressions) PASS; sb `make all` GATE PASS (coverage + 3/3
mutants); fparith gate PASS; cores 6-config unit suite PASS; top smoke/l2/l3
PASS. Full-tile coarse synth (scripts/synth_f1.sh): 47,537 -> 14,064 cells,
`$div` 906 -> 4, `$mul` 994 -> 92 (remaining divides are the seam/glue quant
blocks, out of KVQ).
