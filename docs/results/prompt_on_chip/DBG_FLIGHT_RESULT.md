# THE WALK_DBG FLIGHT — THE WALKED-ATTENTION FAULT IS err_stale, ON THE RECORD

**Date:** 2026-08-07 · **Image:** `afi-0c045962a00fdc0b0` /
**`agfi-07245eb04d8bede9e`** (apex-a2dbg-20260807, @8ebdf52: the e7e8 RTL +
WALK_DBG 0x98; WNS +0.453 TNS 0 at A2; 11th image, 11th first-try
ingestion) · **Instance:** `i-04cb43f01c9948385` f2.6xlarge us-west-2
(terminated + verified) · **Driver:** `scripts/fpga/f2/fly_dbg_session.py` ·
**Records:** `build/dbg_session/dbg_session_verdict.json`, capture streams
alongside.

## The claim (exactly)

**On silicon, the walked-attention fault is `err_stale` — the FIRST walked
score request found the composite scale-cache entry unwritten — and every
observable of the fault now matches the sim empty-cache repro exactly.**

```
arm 1  walk_e7ng_dbg0      PASS  138 caps, WALK_DBG quiet   (instrument clean)
arm 2  walk_e6_silicon_dbg FAIL  WALK_DBG = 2 = err_stale   (THE answer)
arm 3  hostattn_fuelarm    PASS                             (fuel-arm EXONERATED)
arm 4  walk_e6 plain       FAIL  0x3703, 1 cap              (signature stable, 3rd image)

fingerprint (post-fault, same card):
  walk_dbg=2  walk_status=0x3703  err_sticky=0x1 (mxe_desc_error)
  layer_status=0  l_ctrl=0  fuel_stat=0x9 (EMPTY|DDR_READY)  fuel_err=0

attribution (W1C-cleared + verified 0 at program start):
  passing e7ng leaves ERR_STICKY = 0x0
  the failing walk itself latches BOTH 0x1 and stale
  the SIM empty-cache repro latches the IDENTICAL pair -> the MXE desc
  error is mechanical fallout of the error-stop, not a primary symptom
```

## What this pins down

1. **The stale class, not frame** — record framing is fine; the request
   found nothing at the slot.
2. **The s_q leg is healthy** — the walker's head handshake (`hd_ready`)
   requires s_q present; a missing s_q parks the walker BEFORE any
   request, with no error. We got the error — so requests flowed, and the
   failing leg is **`sc_val` (the KV record scales)**.
3. **Fuel-arm exonerated** — host-mode attention with the fuel system
   armed passes on the same silicon.
4. **The desc error is secondary** — identical in the sim repro.
5. **Signature parity with the sim repro is now total**: stall word,
   capture count, WALK_DBG class, ERR_STICKY pair, fuel/layer state.
   On silicon the cache behaves as if the KV store's snoop never wrote it
   — while host-mode programs using the SAME staging idiom read it fine.

## What is still open — and the instrument that decides it

Three mechanisms survive, and WALK_DBG v2 (committed @8c1c8db,
sim-calibrated: repro reads snp=0/idx=0/eng=0, a passing run reads snp=2)
distinguishes them in ONE flight on the next image:

| next silicon reading | the seam it names |
|---|---|
| `snp=0` | the snoop WRITE path (staging's stream never reached the cache) |
| `eng=1` in the error ctx | the walk-mode engine SELECT |
| `snp=2, eng=0, idx=0` | the bank's INTERNAL read/valid addressing |

An engine-mirror probe (store into both engines) was attempted and is
infeasible cheaply: `l_kv_map` steers the KVQ STORE engine too, so the
second store pass wedges legitimately (`pw stall`, sim-confirmed).

## Cost

f2.6xlarge ~35 min ≈ **$1.20**; terminated + verified. Account sweep:
only the unrelated `apex-f2-fpga` box remains.

## SECOND FLIGHT (same day) — THE SEAM, NAMED

**Image:** `agfi-03064a1c113e19c0b` (apex-dbg2-20260807, @8c1c8db tree +
squant pipeline; WNS +0.238; 12th image, 12th first-try ingestion) ·
**Instance:** `i-08b76f116735cdb58` (terminated+verified) · **Probe:**
`walk_e6_dbg2` (W1C both banks + verified-zero start + ESTK-gated
full-word capture) · **Capture:** `walk_e6_dbg2.hw.cap.jsonl`.

```
clean control (e7ng): PASS — instrument quiet
SEAM PROBE: word=0x02000002
  snp=2   the snoop WROTE both KV rows (the exact passing-run count)
  idx=0   the walked request asked for the right slot
  qs=0    a record-scale (cs) request
  eng=0   ...on the right engine
  stale=1 ...and the cache still answered "unwritten"
  stk=0x1 (the known mechanical desc error)
```

**Verdict: outcome 3 of 3.** The write side is live (snp=2), the read
request is well-formed (eng=0, idx=0) — the fault is INSIDE the composite
bank's write->valid->read path (`apex_wcomp_bank.sv` +
`seq_walker_comp.sv`). Sim-green, silicon-red, MET timing, clean CDC
audit: the suspect class is now synthesis-vs-simulation semantics in a
few hundred lines — X-propagation, width truncations, don't-care
optimizations, or the bank's snoop engine-routing at store time (the ctx
captures the READ engine; the WRITE engine at snoop time is not yet
instrumented — DBG bits [7:2] remain free for a v3 if the code audit
does not fall out first).

Cost: ~25 min ≈ $1.
