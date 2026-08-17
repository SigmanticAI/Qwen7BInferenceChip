# I-C DCP feasibility probe — GQA-4 / wide-D CL: places and meets timing, routing OOM'd the build host

**Date:** 2026-07-27 · **Config:** `cl_apex` at `APEX_CL_D=128`, **`APEX_CL_GQA=4`**,
**`APEX_CL_DM=3584`**, clock recipe **A2** (tile on `clk_extra_a1`, 64 ns) ·
**Box:** apex-f2-devbox, c6a.4xlarge (30 GiB usable) · **Flow:** Vivado 2025.2,
F2 Small Shell · **Evidence:** [`apex_dcp_ic.log`](apex_dcp_ic.log),
[`cl_apex.2026_07_27-160421.post_phy_opt_timing.rpt`](cl_apex.2026_07_27-160421.post_phy_opt_timing.rpt)

## Scope — read this before quoting anything

This was an owner-approved **feasibility probe**, explicitly *not* the artifact
intended to carry a claim. Its job was to retire the risk in a configuration
that had **never been built**: four CQ-8 KVQ engines + per-KV-head composite
caches + the wide-D elaboration, at the A2 bring-up clock.

**DISCLOSED EXCLUSION: `APEX_CL_DDR=0` in this build.** The DDR fuel line —
the leg W-G3 proved in simulation with real Qwen `Wq` weights — is compiled
out. This DCP proves nothing about the DDR path. (Caught in pre-flight by a
sibling session; the `APEX_CL_DDR` knob is now plumbed for the follow-up.)

## Result: NO TARBALL. Not a design failure — the build host ran out of memory.

```
Phase 22.1 Optimize Skews
Phase 22.1.1 Leaf ClockOpt Init
/opt/Xilinx/2025.2/Vivado/bin/rdiArgs.sh: line 57: 3843 Killed  "$RDI_PROG" "$@"
ERROR: Did not find the post-route DCP file
```
Kernel confirms the mechanism:
```
vivado invoked oom-killer
Out of memory: Killed process 3843 (vivado)
  total-vm:44180820kB  anon-rss:31466252kB
```
**31.4 GiB resident on a 30 GiB box, during routing.** The 4× engine
replication plus wide-D is materially larger than any previously built CL.

## What the probe DID establish (the stages that completed)

| stage | outcome |
|---|---|
| synthesis | completed, defines confirmed in the synth command line: `APEX_CL_D=128 APEX_CL_GQA=4 APEX_CL_DM=3584` |
| opt_design | completed |
| **place_design** | **completed — the GQA-4/wide-D design FITS the VU47P device** |
| phys_opt | completed, **timing MET, worst slack +0.177 ns**, tile path group `clk_out1_clk_mmcm_a` at **period = 64.000 ns** (recipe A2 applied) |
| route_design | **OOM-killed by the host**, no post-route DCP |

Clock-recipe application is correct: the only "Clock recipe … not applied"
CRITICAL WARNINGs are for the B/C/HBM MMCMs, which are disabled groups with no
cells — group A applied cleanly, exactly as in the I-A build.

**So the honest reading is: this configuration places on the device and meets
timing at the A2 clock through phys_opt.** Routing may still fail on its own
merits — that is untested — but nothing observed so far indicates a design
problem, and the failure that did occur is a property of the build machine.

## What this does NOT establish

- No routed design, **no timing signoff**, no tarball, no AFI. Post-route
  timing is the number that counts and it does not exist for this config.
- Nothing about the DDR path (excluded, above).
- Nothing about silicon. 15.625 MHz is a **correctness clock**, never a
  throughput number.

## Next step

Re-run routing on a larger build host (≥64 GiB; c6a.8xlarge is the natural
step and exceeds the standing ≤~$0.70/hr rule, so it needs an explicit
owner go). Everything up to `place_design` is reproducible from this commit;
the flow, the clock recipe and the build knobs are all now proven correct.

**Spend:** devbox ~2.6 h ≈ $1.60. No F2 instance was launched.
