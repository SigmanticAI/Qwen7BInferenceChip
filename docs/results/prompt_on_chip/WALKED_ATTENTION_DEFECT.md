# WALKED ATTENTION DOES NOT SURVIVE SILICON — E-7/E-8 FLIGHT, AND THE DEFECT IT FOUND

**Date:** 2026-08-06 · **Branch:** `decoder-layer-and-fpga-fit` @ `a772713` ·
**Image:** `afi-0520bda18ad0d38f1` / **`agfi-04cfe164ba90b8ab0`**
(apex-e7e8-20260806; D=64 DMODEL=64 GQA=2 DM=896 QSTAGE=14 DDR=1, clkgen A2,
Slack MET +0.319 ns, PRV-GREEN, ingested first try) · **Instance:**
`i-0029f4cb32ec22847` f2.6xlarge us-west-2 (terminated + verified) ·
**Machine record:** `e7_hw_differential.json` (this directory) · **Logs:**
`build/e7_hw_flight2.log`, `build/e6c_bisect.log`, `build/latsweep.log`.

## The result (exactly)

**The E-7 and E-8 chains PASS in simulation and FAIL on the FPGA. The failure
is a walker-latched `WALK_ERR_SEQ`, and a five-program differential on the
same card, the same image and the same resident weights localizes it to one
thing: WALKED ATTENTION (`SCORE`+`PV`) on the GQA scale-cache bank.**

Every walked chain WITHOUT score/pv passes. Every walked chain WITH score/pv
faults. Nothing else in the image is implicated.

| program | walked steps | silicon | captures | checks |
|---|---|---|---|---|
| `walk_fuel_oproj` | OPROJ, RES1 | **PASS** | 896 | 36, 0 fails |
| `walk_fuel_qkv_oproj` | QKV, OPROJ, RES1 | **PASS** | 2192 | 50, 0 fails |
| `walk_e6` | **+SCORE, PV**, NFEED, NSRC, FPROJ | **FAIL** | 1 | 11, 1 fail |
| `walk_e7` | + NORM2, FGAM (fetched gamma) | **FAIL** | 1 | 11, 1 fail |
| `walk_e8` | + QKV | **FAIL** | 217 | 12, 1 fail |

**The sharpest fact — the fault seam is bit-exact on both sides.** `walk_e8`
drained 217 captures (the walked QKV y-drain) before its done-poll stalled,
and those 217 hardware words are **bit-identical to the first 217 of the sim
run's 437** (`build/e8_prefix_diff.log`: `compared=217 mismatches=0`).
Within ONE one-kick walk, on silicon: QKV walked perfectly, then SCORE
faulted. The defect is not corruption creeping in — it is a clean stop at
the QKV→SCORE step boundary, exactly where the walker first engages the
composite scale-cache.

All three failures stall at the same place with the same word:

```
poll stall @0x1068 (WALK_STATUS): last 0x00003703 want 0x0/0x1
  BUSY[0]=1  ESTK[8]=1  ECODE[11:9]=3 (WALK_ERR_SEQ)  FMT_SUP[15:12]=3
```

The walker latched an internal sequencing fault and never cleared busy, so
the host's done-poll never completed.

## How the fault is localized to the composite unit

`WALK_ERR_SEQ` has exactly two sources in this RTL:

1. `rtl/seq/seq_layer_walker2.sv:1858` — the STOREKV path (`S2_SKPR`).
   **Not reachable here:** `EN_STOREKV` (bit 3) is clear in all three
   failing masks (`0x68f0`, `0xe9f0`, `0xe9f2`).
2. `rtl/top/apex_top.sv:552` — `wc_err_frame || wc_err_stale`, the
   **composite scale-cache unit**.

So the fault is a composite-unit fault, and the composite unit is precisely
what `SCORE`/`PV` consume. Its two error meanings:

- `err_frame` — "tlast at the wrong beat" (record framing) —
  `seq_walker_comp.sv:97`
- `err_stale` — "requested an unwritten entry" (scale-cache miss) —
  `seq_walker_comp.sv:98`

`apex_top` routes both to the same code, so the sticky alone cannot say
which; distinguishing them needs either a debug CSR or a sim reproduction.

## What is and is NOT established about the mechanism

This image is **GQA=2**, so `apex_top` instantiates the **bank**
(`rtl/top/glue/apex_wcomp_bank.sv`, the F6 per-KV-head cache) and not the
single `seq_walker_comp` (`apex_top.sv:701` generate).

**A refuted first hypothesis, on the record.** The obvious suspect was the
walker's dynamic engine select (`kv_eng_sel = walk_en_q ? wk_kv_eng_sel :
l_kv_map_q`, `apex_top.sv:742` — the bank's comments assume a
"quasi-static" select, and the walker bumps `g_idx` per head at
`seq_layer_walker2.sv:1911`). **The failing programs' own descriptors
refute it:** decoding `W_MODEL1` from the flown `walk_e7.regops.jsonl`
gives `0x10101` = `pack_model1(1, 1, 1)` — **H=1, n_kv=1**. In these toy
chains the select sits at engine 0 for the entire walk and never moves.
Whatever the mechanism is, it is not select-crossing.

**The control caveat.** Host-driven score+PV at this D=64 geometry is
silicon-proven bit-exact 1792/1792 (`p05b_hw_check2.log:78`) — but on a
**different image** (`agfi-0ecab46b8a8376b21`, pre-E-7 RTL). It separates
walker-vs-host only if the composite subsystem is otherwise identical
across the two images, and the E-7 glue (`gam_win_q` muxes, walker
y-drain) touches `apex_top` near it. **Host-mode composite on TODAY'S
image is untested** — that is the sharpest open split:

- host jobs pass on this image AND walked score faults → the defect is
  exclusively the walk-mode composite path on this silicon;
- host jobs ALSO fail on this image → the E-7 RTL broke the composite
  subsystem generally (and sim still doesn't show it).

**[RESOLVED, same day — the split ran on a second card session
(`i-0ec2487d966f900f0`, this image, weights full-verified):** all 7
committed host-mode attention jobs **PASS** (both kv engines, 139–199 baked
checks each; `build/hostctl_result.json`) and `walk_e6` fails identically
(same `0x3703`, same stall line). The composite subsystem is healthy on
this image in host mode; **the defect is exclusively the walker driving the
composite scale-cache.** The same session also proved the fetched-gamma
machinery silicon-correct in isolation with its physical gamma-poison
discriminator RED — see `E7NG_ON_SILICON.md`.]**

**The xdc exception audit (follow-up 2) also ran:** every
`set_max_delay`/`set_false_path` in `cl_synth_user.xdc` is scoped to true
CDC structures (`u_ocl_cdc`, `u_fuel_fifo`, `u_fuel_rdr`, `u_fuel_ctl`,
`arown_*`, the reset synchronizer). Nothing touches `apex_wcomp_bank`, the
walker, or any composite path — those are ordinary intra-tile synchronous
paths, fully covered by the MET +0.319 STA. The swallowed-synchronous-path
mechanism is eliminated.

**[2026-08-07 — the signature REPRODUCES in sim, and it is stale-shaped.]**
A `walk_e7` variant with the KV store removed (the scale cache never
populated) fails in sim with the EXACT silicon fingerprint: done-poll
stall at `WALK_STATUS=0x3703`, exactly 1 capture — the same failure shape,
to the register. A new debug CSR (`WALK_DBG` 0x98, splits the two
composite faults; validated in both arms — clean walk reads 0, this run
reads 2) identifies it: **err_stale** (request against an unwritten entry
/ missing s_q). So the silicon fault behaves precisely as if the composite
cache holds nothing when the first walked score request arrives. Repro
recipe + probe committed (`walk_e7_nokv_dbg.regops.jsonl`). NOTE the
epistemics: signature match is necessary, not sufficient — the silicon
root cause may differ; the ESTK-gated `WALK_DBG` probe pair
(`walk_e6_silicon_dbg` + `walk_e7ng_dbg0` control, both sim-validated) is
built to read frame-vs-stale on the REAL fault the next time an image
with 0x98 flies.

**Staging op-level audit (host-passing vs walked-failing programs):** the
KV store idiom is identical between the two families (same KVQ CSR block,
same record framing); the ONLY staging difference is that walked programs
write LAYER_CTRL (rope/staging levels) — and their `l_kv_map` bits [17:15]
are 0 in both families, so both store into engine 0. The divergence is not
in what the host stages; it is in what the walk-mode request path finds.

The plumbing consistent with the evidence (fault at the FIRST walked score,
one clean stop, no corruption): the composite cache is populated by
host-mode snooping during the program's KV store, and the first walked
request either finds `sc_val` unset or `s_q_val` missing (`err_stale`), or
the record framing misaligns (`err_frame`) — `apex_top` folds both into
`WALK_ERR_SEQ`, so the sticky alone cannot say which.

## What the simulator says, and why it did not catch this

The twin was built from the SAME commit as the image
(`verif/f2sim/obj_e7fly_b64_ddr1`), and every chain passes on it:

- `walk_e7` — 221 captures, 27 checks 0 fails, **all four discriminators
  fire** (walk-off, no-FPROJ, no-NORM, and the gamma-byte poison RED by the
  exact predicted delta)
- `walk_e8` — 437 captures, all six sub-grades bit-exact
- `walk_e6` — the full E-6 chain gate green

**The simulator's clock divider is not the divergence.** `walk_e7` passes at
`+tile_div=5`, `2` and `1` (undivided). A DDR latency/backpressure sweep
(`+ddr_lat` 25→600, `+ddr_stall` LFSR) is recorded in `build/latsweep.log`.

Also swept: `+tile_div=16` (the REAL shell:tile ratio on the card, 250/15.6)
and `32` — both pass. And the twin verifiably builds the SAME bank: its
`INFO_TIER` reads `0x1` (bit1 clear = GQA build), confirmed by a checked
read, so this is not a sim-compiled-the-wrong-generate story.

This is the honest shape of it: a cycle-accurate RTL twin of the same code
passing while silicon faults leaves only mechanisms the twin cannot express —
a timing violation on a path STA does not cover (e.g. a synchronous path
swallowed by a CDC exception in `cl_synth_user.xdc`), a power-on state a
2-state simulator zero-fills, or a CDC race. Until the fault reproduces in
sim, the defect is characterized but not diagnosed.

One more scope note: walked score+pv had NEVER flown before today on any
image. The B1 "I-A silicon replay" (2026-07-22) was HOST-mode attention
jobs; the E-lane walked chain (D=128) walked residual→norm only. So there
is no GQA=1 walked-attention silicon datapoint to compare against — the
next card session should fly one (see follow-ups).

## What this does NOT retract

- **`E6_ON_SILICON.md` stands.** Its two chains were re-flown on THIS new
  image today and both pass (896 and 2192 captures, 86 checks, 0 fails).
- **The 6/6 op-type and C2 prompt results stand** — they are host-driven.
- **`p05b` host-driven attention on silicon stands** — and it is the control
  that makes this localization possible.

What cannot be claimed today: **E-7 (fetched-gamma one-kick chain) and E-8
(QKV composed in) on silicon.** Both are sim-only, and now known to be more
than sim-only-by-omission — they are sim-only *because they fail on the
card*. The `ELANE_WALKED_CHAIN_RESULT.md` E-lane claim (residual→norm, D=128
GQA=1, single composite unit) is unaffected: it never walked score/pv on a
bank.

## Card-session integrity

- DDR loaded and **full-verified** before the runs (20 tensors, 466,288
  words read back, 0 fails) and **re-verified after** them — the same
  resident weights served every program in the differential.
- The three E-7 descriptor refusals (`walk_e7_off`, `e7_fence_nofproj`,
  `e7_fence_nonorm`) behave on silicon exactly as in sim: all three refuse
  with 0 captures. The walker's descriptor checker is healthy on this image.
- The gamma-poison arm did **not** run: `f2_ddr_load`'s whole-image sha256
  gate refused the mutated image (the poison re-hashes the per-region
  manifest but not `ddr_image.json`). The card's DRAM was never poisoned —
  the loader aborts before the first burst. Fixing the poison path to
  re-hash both is tracked below.
- The clock gate did its job twice: it refused every run until
  `agfi-04cfe164ba90b8ab0` was registered in `clock_key.IMAGE_RECIPE`, and
  the recipe registered (A2) was read from the build's **own manifest**
  (`s3://apex-f2-dcp-099597653601/dcp/e7_2026_08_06-071435.Developer_CL.tar
  :: to_aws/2026_08_06-071435.manifest.txt`), not from the AFI description
  I wrote myself at submission time.

## Follow-ups

1. **The decisive next-card experiment (15 min of card time):** fly the 7
   COMMITTED host-mode attention jobs (`build/p2_05b_regops_b64_05b/`, D=64,
   kv_head 0 AND 1, baked expectations) on THIS image, alongside a re-run
   of `walk_e6`. Host-pass + walk-fail isolates the walk-mode composite
   path on this silicon; host-fail means the E-7 glue broke the composite
   subsystem across modes. Either way the search space halves. (The
   originally-planned select-crossing A/B is moot — the failing walks are
   already H=1.)
2. **Reproduce in sim.** Make the testbench's KV/mailbox handshake latency
   configurable the way the DDR model already is, and sweep it. Audit
   `cl_synth_user.xdc` for any exception group that touches
   `apex_wcomp_bank`/`g_idx` fan-in — a synchronous path swallowed by a CDC
   `set_max_delay` would keep STA green while silicon glitches. Without a
   repro or a proven mechanism the fix cannot be verified.
3. **Split the sticky.** `err_frame` and `err_stale` both land on
   `WALK_ERR_SEQ`; a debug CSR that distinguishes them would have cut this
   investigation roughly in half.
4. **`fly_e7_hw.py` poison path** — re-hash `ddr_image.json` as well as the
   region manifest, so the poison arm can arm the integrity gate rather than
   trip it.
5. Both `fly_e7_hw.py` guards added this session earned their keep: it now
   aborts before touching card DRAM when the claim run did not execute.

## Cost

f2.6xlarge ~1.5 h ≈ **$2.50**. Terminated and verified; the account F2 sweep
afterwards shows only the unrelated `apex-f2-fpga` box (still running, ~6+
days — not this project's, and not mine to stop).
