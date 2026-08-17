# KVQ subsystem smoke — kvq_engine (APEX-owned, fixed, wrapped per-channel INT4 codec)

**Scope:** ARCHITECTURE.md §1 KVQ row, §4; D-007 (carried from V0), D-008
(partial-group flush), D-009 (byte-aligned record), D-015 (vendor the V0-patched
top), D-016(a/b/c) (three latent-bug fixes, each with a reproduce-then-fix
regression).
**Verdict: PASS** — all 8 parity/stall configs bit-exact with check counts
IDENTICAL to the V0 runs, and all 4 regressions first REPRODUCED on the
vendored/baseline code, then proven fixed on `rtl/kvq/`.
**Date:** 2026-07-07 · Verilator 5.044 (`--binary --timing --assert`, `-Wall`)
**Everything below is reproduced by `make smoke` in this directory** (logs
under `logs/`; the Makefile GATES on the exact strings quoted here).

## 1. What was built (rtl/kvq/)

- `kvq_engine.sv` — starts from the V0.2-verified `verif/v0/kve/patched/`
  top (D-015; upstream verbatim is banned — it drops 52% of read beats under
  backpressure). APEX changes, each detailed in the file header:
  - **D-016(a)** bounded `ST_RWAIT` (RD_WAIT_MAX=8) + error response: read of
    an unwritten address is dropped cleanly (0 beats, engine returns idle),
    `IRQ_STATUS[0]` RD_ERR sticky (W1C), `STATUS[1]` mirror, new `irq` pin
    (`IRQ_STATUS & IRQ_MASK`).
  - **D-016(b)** `ST_IDLE` collision: an ACCEPTED `s_axis` beat always wins
    over a same-cycle `read_req`; the read is remembered (`read_pending`) and
    serviced at the next `ST_IDLE`; `s_axis_kv_tready` drops for the whole read
    walk so no beat can be handshaken while unserviceable.
  - **D-008** new `flush_req` pulse input (future CSR FLUSH, §7 0x28): freezes
    a partial key group (g<G) at its current size and pushes it through the
    EXISTING scale/quant/emit datapath, unwedging `ST_KACCEPT`. Latched
    (`flush_pending`) if it arrives mid-token; no-op with no open group.
  - **D-009/§4** byte-aligned records, LSB-first, padded to the next 64-bit
    boundary: key `[tag=01:8b][field D×16b][codes D×4b][pad]`, value
    `[tag=00:8b][scale 16b][payload][pad]`. D=64 key record: 1288 → 1344b.
    ⚠ §4's example totals (1312b/2608b) do NOT lie on the 64b boundary its own
    rule states (1288+24=1312 is not 64-aligned); the RULE was taken as
    normative over the examples. Flagged for the next ARCHITECTURE.md edit.
  - AXI-Lite responses now honor `bready`/`rready` (hold until accepted).
  - `-Wall` clean with ZERO waivers on this file (checked at D∈{16,64,128},
    TIER∈{0,1,2}, G∈{2,64,128} — `make lint` covers the shipped set).
- `vendor/*.sv` — 7 upstream leaves at commit `b3a5e807`, provenance headers.
  5 verbatim; 2 modified and documented in their headers:
  - `residual_buffer.sv` — **D-016(c)**: write guarded by `cnt != G` (upstream
    aliased mem[0] on a (G+1)-th write via index truncation); cnt saturates.
  - `cq_key_path.sv` — **D-008**: new `flush` input; in `S_COLLECT` with
    icnt>0 it freezes `g_cnt <= icnt` and enters the normal `S_SCALE` walk.
    The amax freeze reuses the existing `group_done` semantics (amax_unit
    unchanged: with `in_valid` low it latches the accumulated max).
  - Vendor lint waivers are per-line in `lint_waivers.vlt`, each mapped to the
    V0 §6 triage.

## 2. Parity re-run — no functional regression (gated on EXACT V0 counts)

Same golden vectors, same TB structure (adapted only for the module name, the
§4 record offsets, and `flush_req=0`); `checks` totals match V0 exactly:

| config | V0 count | kvq_engine (`logs/run_p_*.log`) |
|---|---|---|
| d64_cq8   | 49923 | `checks=49923 fails=0` PASS |
| d64_cq4   | 54020 | `checks=54020 fails=0` PASS |
| d64_cq4p  | 54020 | `checks=54020 fails=0` PASS |
| d64_t70   | 28130 | `checks=28130 fails=0` PASS (incl. the wedge demo with flush_req tied 0) |
| d128_cq8  | 77403 | `checks=77403 fails=0` PASS |
| d128_cq4  | 32665 | `checks=32665 fails=0` PASS |
| g128_cq4  | 54019 | `checks=54019 fails=0` PASS |
| stall (V0.2 flow) | — | `beats_accepted=13824/13824 dropped=0 mismatches=0 stability proc=0 sva=0` PASS |

The §5 stream SVA pack rides in every kvq build via `bind`
(`kvq_axis_sva.sv`; the MXE `apex_stream_sva.svh` is descriptor/job-shaped —
binding it to KVQ's AXI surface would be vacuous, so its §5 stream properties
were lifted into a KVQ-shaped checker + a burst-framing rule: tlast exactly on
beat D-1). Zero `%Error` in any run is a Makefile gate (D-012).

## 3. Regressions — written first, REPRODUCED on the vendored code, then fixed

| ID | baseline (bug) run | kvq (fix) run |
|---|---|---|
| D-016(a) `r_rwait` | `STATUS.idle=0 2000 cycles after the unwritten-address read`, follow-up valid read dropped, only soft reset recovers → **BUG REPRODUCED** | idle again ≤100 polls, 0 beats, `IRQ_STATUS[0]`/`STATUS[1]`/`irq` set, W1C clears, next read bit-exact — `checks=137 fails=0` |
| D-016(b) `r_collide` | read burst plays DURING the stream (59 beats), all 128 write beats handshaken, OCCUPANCY stuck at 1, token EVAPORATED → **BUG REPRODUCED** | beat wins; token 1's §4 record bit-exact; deferred read burst appears only after the token completes, bit-exact; OCC=2 — `checks=166 fails=0` |
| D-016(c) `r_alias` | UNIT level (unreachable through the top by construction — both FSMs cap groups at G): upstream `residual_buffer` (G+1)-th write ALIASES mem[0], fill=65 → **BUG REPRODUCED** | vendored copy: overflow write dropped, mem[0]/mem[G-1]/fill intact — `checks=4 fails=0` |
| D-008 `r_flush` | V0-baseline still wedged in ST_KACCEPT 20000 cycles after the (unconnected) flush pulse, occupancy frozen, soft reset loses the tail → **WEDGE REPRODUCED** | T=70 G=64: full group + g=6 tail flushed; OCCUPANCY 64→70; ALL 70 tokens checked as §4 records AND fp32 readback **bit-exact vs the PYTHON golden** — `checks=17996 fails=0` |

**Flush oracle discipline:** `gen_kvq_vectors.py` runs
`apex_golden.cq_codec` (partial groups per §3.1) on the frozen T=70 inputs
and SELF-CHECKS its output bit-identical against the frozen golden set (which
covers the partial tail from upstream's datapath-level run) before the TB
consumes the *generated* files — the regression's oracle is the Python golden,
validated against the frozen arbiter (D-013 lineage).

## 4. Honest caveats

- **Icarus cross-check not re-run** for the kvq TB (V0 did one config; the
  adapted TB uses the same constructs and should port, but it was not run).
- **No constrained-random/illegal-framing stress, no mid-op reset test, no
  coverage metrics** — Layer-1 territory per ARCHITECTURE.md §8; this suite is
  the Layer-0-style parity + directed regressions for the D-016/D-008 deltas.
- The engine still DROPS `read_req` while busy outside `ST_IDLE` (upstream
  semantics, documented); only the ST_IDLE collision case queues. A read
  colliding with the first beat of a key group is deferred until the group
  completes or is flushed.
- Flush quantizes complete TOKENS: a flush during a partially-streamed token
  waits for that token's remaining beats (documented in the header); a host
  that stops mid-token must soft-reset, as before.
- The `sram_wr_addr` group-write truncation hazard is now a hard elaboration
  error (`SRAM_DEPTH < KEY_GROUP` → `$fatal` at time 0), not silently wrong.
- Baseline (bug-reproduction) builds use `-Wno-fatal` (V0 discipline — those
  warnings are triaged in verif/v0/kve/RESULT.md §6); all kvq builds are
  `-Wall` with vendor-scoped per-line waivers only.
- No synthesis/timing claim; simulation sign-off only (§0).
