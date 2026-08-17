# KVQ D-020 fix — ADVERSARIAL RE-AUDIT (verif/kvq/audit)

**Role:** independent re-audit of the just-fixed `rtl/kvq/` (D-020: soft-reset
semantics B-1..B-3 + CQ-4+ outlier −0.0 sign B-4, closing the
`verif/kvq/RESULT.md` FAIL). Nothing from the fixer was trusted: both existing
suites were clean-rebuilt from scratch, the fixer's edits to the independent
TB were diff-audited for weakened checks, and a NEW attack suite was written
against the fixes specifically. Golden arbiter: `golden/apex_golden`
(D-001/D-013).

**Verdict: PASS — KVQ earns vendored-verified status** (§5 below for the
caveats that bound that claim).

**Date:** 2026-07-08 · Verilator 5.044 (`--binary --timing --assert`, `-Wall`)
· single-command gate: `make all` in `verif/kvq/audit/` → **exit 0**
(console: scratchpad `audit_final.log`; logs under `build/`).

---

## 1. Both existing suites clean-rebuilt (this session, this machine)

`verif/kvq/smoke`: `make clean && make smoke` → **exit 0**. V0-parity check
counts IDENTICAL to the recorded ones, verbatim:

```
CONFIG d64_cq8:    checks=49923 fails=0    CONFIG d128_cq8: checks=77403 fails=0
CONFIG d64_cq4:    checks=54020 fails=0    CONFIG d128_cq4: checks=32665 fails=0
CONFIG d64_cq4p:   checks=54020 fails=0    CONFIG g128_cq4: checks=54019 fails=0
CONFIG d64_t70_cq4: checks=28130 fails=0
KVQ STALL [stall_kvq]: PASS — all beats delivered bit-exact under backpressure
```
All 4 reproduce-then-fix regressions (alias/rwait/collide/flush) still
BUG-REPRODUCED on baseline and PASS on `rtl/kvq/`.

`verif/kvq/sb`: `make clean && make all` → **exit 0** (was exit≠0 pre-fix).
The two pre-fix fail markers are gone: `CONFIG cq4p: checks=1497 fails=0`
(B-4), reset run `CONFIG cq4: checks=979 fails=0` with
`COV rst_collide_kfeed 1` / `COV rst_collide_rload 1` (B-1..B-3),
`COVERAGE GATE: PASS`, all 3 sb mutations still CAUGHT. This matches the
fixer's claimed AFTER strings verbatim.

**Fixer's TB edits audited (git diff `verif/kvq/sb/tb_kvq_sb.sv`): NOT a
weakening.** The `soft_reset_check` split follows D-020 exactly and the
outburst branch ADDS checks (burst must complete with exactly D beats,
bit-exact, tlast on D−1, STATUS.idle=0 while draining, no retraction). The
`rload_collide` sample point moved one cycle EARLIER with a correct
justification (on fixed RTL the FSM is already back in IDLE one cycle after
the collision — the old sample point only "hit" because the bug parked the
FSM). `lint_waivers.vlt` changes are pure line-number rebasing for the D-020
edits; same rules on the same constructs, nothing new waived (verified
hunk-by-hunk).

## 2. RTL fix code-review findings (read before attacking)

- Fix architecture is sound: `ctrl_reset` priority skips the FSM case except
  in ST_OUTPUT/ST_OFLUSH (burst completes per §5 no-retraction); `dp_clear`
  is a same-cycle sync abort into `cq_value_path`/`cq_key_path`/the shared
  `cq_quant_unit_syn`; the clear branches take priority over same-cycle
  `in_valid`/`q_done`; STATUS.idle ANDs in `cqv_busy`/`kp_busy`.
- A reset-crossed burst cannot be corrupted by `dp_clear`: the dequant units
  are combinational off engine-held registers (`dec_*`/`kp_dec_*`), not off
  the cleared walk FSMs. Confirmed by the drain-exact runs below.
- B-4 sign-force is the identity for every nonzero raw fp16 (code=+1 ⇒
  dequant sign == raw sign) and only flips exact −0.0; scoped to the outlier
  lane. Confirmed bit-exact vs golden at D=16 and D=64 below.
- Reset landing in the FIRST ST_OUTPUT cycle (beat 0 not yet loaded) lets the
  burst run to completion rather than dropping it — compliant with D-020's
  "in flight in ST_OUTPUT completes" reading; noted as a spec nuance, not a
  bug (completing is always §5-safe, and STATUS.idle stays 0 throughout).

## 3. New attack suite (this directory) — all PASS on the fixed RTL

Fresh generator (`gen_audit_vectors.py`, golden pool of fully-precomputed §4
records + fp32 readbacks), fresh TB (`tb_kvq_audit.sv`), §5 SVA pack bound in
EVERY run (including the reset storms — stricter than sb, which exempted its
reset run), `-Wall`, watchdogs, cycle-stamped fails.

**Randomized soft-reset injection — 260 resets** (≥200 required), 5 configs
(D=16 CQ-4/CQ-8/CQ-4+, D=64 CQ-4, D=64 CQ-4+), 15 tactics: parked mid-token
(random beat 1..D−1), mid-group (random group progress), COMPRESS/KEMIT
random-delay, pending-beat parks in OUTPUT (random drained-beat count) and
OFLUSH (tlast beat pending), free-flowing-read resets, beat-coincident resets
(accept in the ctrl_reset cycle), and deterministic 1-cycle-state collisions
(ST_STORE via `cqv_out_valid` timing, ST_RLOAD/ST_RWAIT via back-to-back AXI
writes, ST_KFEED via final-beat+CTRL coincidence — including the
group-completing KFEED). The landing FSM state is binned hierarchically from
the actual `ctrl_reset` cycle, and **every reset is followed IMMEDIATELY by a
full clean job checked bit-exact vs golden** (value job every event, full
key-group job on key-ish/3rd events — 260 value + 144 key-group clean jobs).

Post-reset D-020 contract, checked per event:
- landing outside OUTPUT/OFLUSH → FSM in ST_IDLE ≤6 cycles, ZERO leaked
  m_axis beats (read victims: 0 beats or the full pre-completed D);
- landing in OUTPUT/OFLUSH → burst completes EXACTLY (D beats, bit-exact,
  tlast on D−1; 41 such drains), STATUS.idle=0 while parked;
- datapath cores quiesced same-cycle (whitebox `cqv_busy`/`kp_busy`) and
  STATUS.idle==1 **immediately** (a demand, not a poll — see §4);
- continuous IDLE-LIE monitor: STATUS-visible idle while FSM≠IDLE or a beat
  live ⇒ fail (0 hits);
- epilogue: a record written BEFORE all 260 resets is still bit-exact after
  (SRAM/occupancy preservation).

```
TB PASS [rst_cq4]  checks=17878 fails=0  (140 resets, D=16 CQ-4  G=4)
TB PASS [rst_cq4p] checks=5768  fails=0  ( 40 resets, D=16 CQ-4+ k=2)
TB PASS [rst_d64]  checks=32873 fails=0  ( 40 resets, D=64 CQ-4  G=8)
TB PASS [rst_cq8]  checks=1073  fails=0  ( 20 resets, D=16 CQ-8)
TB PASS [rst_d64p] checks=14927 fails=0  ( 20 resets, D=64 CQ-4+ k=4, bp storm)
```

Landing coverage (gated): p1..p11 ALL hit — COLLECT 32 · COMPRESS 13 ·
STORE 20 · RLOAD 17 · RWAIT 19 · OUTPUT 26 · KCOLLECT 32 · KFEED 20 ·
KACCEPT 14 · KEMIT 18 · OFLUSH 15 (+34 idle-landings); pending-beat 28,
mid-token 64, mid-group 63, beat-coincident 30, drain-exact 41, 0 timing
misses.

**−0.0 directed (B-4 + the non-outlier INT4 path), D=16 AND D=64:**
`TB PASS [negz_cq4p] checks=1598 fails=0` · `TB PASS [negz_d64p] checks=5122
fails=0`. Content (independently re-derived from `apex_golden`, generator
self-consistent): 0x8000 in OUTLIER lanes → record stores raw 0x8000, fp32
readback **0x8000_0000** (identity widen); 0x8000 in NON-outlier channels →
quantizes through INT4 (code 0) and reads back **0x0000_0000** = golden
`code*scale`; whole ±0 channel columns (amax=0 → EPS scale); all-−0.0 value
tokens; +0.0 / 0x8001 / −2048.0 outlier neighbors. All bit-exact vs golden,
records AND readback. (These also ride the reset-attack pools as clean jobs.)

## 4. Fix-revert mutation checks — the audit's teeth (each: scratch copy of
rtl/kvq → revert ONE fix → rebuild → rerun → MUST fail; real RTL untouched)

| revert | result |
|---|---|
| **B-1** (drop the `ctrl_reset` case-priority guard) | **CAUGHT** — 13 fails: `soft reset NOT honored — FSM state=6 6 cycles after ctrl_reset (landed in 4)` (RLOAD collision → burst plays after reset), `landed in 8` (KFEED collision → group continues), each with `STATUS.idle=1 ... idle flag LIES` + IDLE-LIE monitor storms |
| **B-2** (tie off `dp_clear`) | **CAUGHT** — 10 fails: `datapath still walking after soft reset (cqv_busy=1/kp_busy=1) — D-020 same-cycle abort violated` + `STATUS.idle=0 immediately after the reset settled` |
| **B-4** (drop the outlier sign force) | **CAUGHT** — 8 fails: `R addr=0 beat=4: got 00000000 exp 80000000` (the original finding's exact signature) |

Honest note on B-2: my first suite version MISSED this mutant — the post-reset
`wait_idle` POLL let the un-cleared datapath quiesce behind the (unreverted)
STATUS busy-AND before the clean job streamed. The check was strengthened
(same-cycle whitebox quiesce + immediate STATUS.idle demand), the full suite
re-run from clean, and only then did all three mutants die. Recorded so the
poll-vs-demand distinction isn't lost on the next block.

## 5. Coverage gates (extended)

`audit_coverage.py` gates the build on: rst_total ≥200, every landing state
p1..p11, all four 1-cycle-state collisions ≥2, pending-beat ≥15, mid-token
≥25, mid-group ≥15, beat-coincident ≥3, drain-exact ≥15, clean jobs ≥200
value + ≥50 key-group, ≥300 peeks + ≥300 reads, and every −0.0 bucket at BOTH
D=16 and D=64 (outlier, non-outlier, all-zero column, value path, +0.0,
negative-subnormal). `COVERAGE GATE: PASS — all required buckets closed`.
These sit ON TOP of the sb gate's 60 buckets, which was re-run and still
passes; the sb gate itself was not edited (it is the verifier's artifact —
the new obligations live here).

## 6. Verdict + honest caveats

**KVQ earns vendored-verified status.** The four independent findings are
fixed and hardened: the fixes survive 260 randomized/targeted mid-operation
soft resets with bit-exact recovery, §5 holds under reset (zero SVA/stability
violations across every run including reset storms), STATUS.idle provably
cannot lie under this suite, −0.0 is bit-exact through both the identity and
quant paths at both head dims, and every checker is mutation-proven against
reverts of the exact fixes.

Caveats (none blocking, all recorded):
- Simulation sign-off only (§0); no Icarus cross-check of this suite.
- Non-finite fp16 (Inf/NaN) in outlier lanes untested — the contract's
  vectors are finite-only and `rand_tok` mirrors that; the outlier identity
  lane for Inf/NaN is unpinned spec. Flag for a contract note, not RTL work.
- Soft-reset collision timing is deterministic for THIS RTL's AXI write
  latency; the TB bins the OBSERVED landing state, so timing drift is
  detected (0 misses this run), not silently absorbed.
- Clean jobs stream ~10 cycles post-reset gated on tready per beat (plus the
  immediate STATUS demand); a same-cycle tready-only host beat after reset is
  covered by the beat-coincident tactic, not by a full same-cycle token.
- The B-1 mutant is not detectable via a STORE-cycle collision (mutant's
  override writes the same `state <= ST_IDLE`); it is caught via the
  RLOAD/KFEED collisions — inherent to the mutation, not a hole in the RTL.
- Fixer's smoke logs under `verif/kvq/smoke/logs/` were regenerated by this
  session's clean rebuild (same PASS content, fresh timestamps).

## 7. Repro
```
cd verif/kvq/smoke && make clean && make smoke   # exit 0, V0-parity counts verbatim
cd verif/kvq/sb    && make clean && make all     # exit 0 (was FAIL pre-fix)
cd verif/kvq/audit && make clean && make all     # exit 0 — this audit
```
