# CACHE_SWEEP_FIX — walked attention GREEN ON SILICON (D-033 closed)

**Date:** 2026-08-11 · **Images:** `agfi-0e103916142fa216e` (apex-snpflop-
20260811, fix-only) and `agfi-0500f4afe435b5e71` (apex-full-20260811,
fix + rung-3 FFN→DOWN + W4 ingest lane + fuel prefetch), both @A2,
WNS +0.356 · **Card:** `i-0f7284d7425bb5b35` (terminated+verified).

## The verdict

| program | fix image | merged image |
|---|---|---|
| walk_e6 (walked ATTENTION, score+pv) | **27/27** | **27/27** |
| walk_e7 (composed walked front) | **27/27** | **27/27** |
| walk_e7ng (gamma control) | 26/26 | 26/26 |
| hostattn_fuelarm (host control) | 113/113 | 113/113 |

walk_e6 — the chain that latched `WALK_ERR_SEQ 0x3703` on SIX consecutive
images over four days — runs clean, streaming 222 walked captures. The
dbg2 instrument, built to wait for the stale error, timed out waiting:
**the defect no longer exists to observe.**

## Root cause (the full chain, for the record)

1. On silicon, `sc_val[req_idx]` read 0 after commits (TERM_SC=1,
   word 0x02000542, bit-identical across five images).
2. Netlist forensics: Vivado had SWEPT the composite cache (~10 cells) —
   but forcing the storage to exist (dont_touch, kill-shot, sledgehammer)
   left the word bit-identical → the sweep was a SYMPTOM.
3. Forensics 9 found the disease: the bank's `snp_valid` input was fed by
   a REPLICA of the snoop conjunction (`wc_snp_valid_k_inferred_i_1`,
   LUT5) re-derived INSIDE g_eng[1] with the kvq bank's
   `e_s_tready[idx]` mux FOLDED to engine 1 — no idx term, no engine-0
   ready. The walk stores via engine 0, so the strobe never fired.
   Meanwhile the walk_dbg counters sampled the ORIGINAL (live-mux) cone:
   every instrument said "healthy" because every instrument watched the
   correct copy. Sim can never see it: the fold exists only in the
   synthesized netlist.
4. **Fix (D-033):** register the snoop bundle at the source —
   `wc_snp_{valid,data,last,addr}_q` (apex_top.sv, dont_touch flop).
   Replication copies a flop exactly (same D-net); it can never re-derive
   its function. Forensics 10 on the shipped DCP: the bank's snp_valid is
   driven by `wc_snp_valid_q_reg` (FDRE) — the failure mode is
   structurally impossible.

## Also proven tonight (the merged image)

Rung-3 walked FFN→DOWN composition, the W4 ingest lane glue, and the fuel
reader prefetch are all IN the flying netlist (sim-verified; their
silicon-level walk programs come with the token-loop work). Timing MET
with margin on both images.

## Ops notes

- Presigned S3 PUT/GET from the devbox/card beats the home uplink by ~100×
  (tarballs: seconds vs 13 min). Pattern: boto3 generate_presigned_url on
  the Mac, curl -T / curl -o on the EC2 side.
- Both builds: 75 min each on m6a.4xlarge (b64_05b config), first-try MET.
- 22 images, 22 first-try AFI ingestions.

## What this opens

The ~100× walked-mode gate on the 4 tok/s ladder. Next: token-loop
composition on the merged image (measure real tok/s), A0 clock campaign
(4×), W4 weight streaming end-to-end (2×). Demo:
`bash scripts/fpga/f2/run_walked_demo.sh agfi-0500f4afe435b5e71`
