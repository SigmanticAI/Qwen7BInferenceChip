# I-C DCP — GQA-4 / wide-D CL ROUTES and MEETS TIMING (tarball produced)

**Date:** 2026-07-27 · **Build:** `2026_07_27-195354` · **Config:**
`APEX_CL_D=128`, **`APEX_CL_GQA=4`**, **`APEX_CL_DM=3584`**, **`APEX_CL_DDR=0`**,
clock recipe **A2** · **Source:** `comp/level-c-integration` @ **3173fd5** ·
**Box:** m6a.4xlarge (61 GiB usable, 16 vCPU, ~$0.69/hr — inside the standing
spend rule) · **Build time:** 1 h 38 m 53 s · **Evidence:**
[`cl_apex.2026_07_27-195354.post_route_timing.rpt`](cl_apex.2026_07_27-195354.post_route_timing.rpt)

## Result: ROUTED, TIMING MET

```
Slack (MET) :             0.711ns
Slack (VIOLATED) count:   0
tile path group clk_out1_clk_mmcm_a  {period=64.000ns}
AWS FPGA: SUCCESS: Design has no negative slack path
tarball: 2026_07_27-195354.Developer_CL.tar   (191,272,960 bytes)
```
Clock recipe A2 applied to the A-group MMCM: **zero** `CLK_GRP_A … not
applied` warnings (the B/C/HBM warnings are disabled groups with no cells, the
same benign pattern as I-A). Tile at 64 ns = **15.625 MHz — a CORRECTNESS
clock, never a throughput number.**

## What this build contains that no previous DCP did

Built from tip 3173fd5, so it carries the whole I-B/I-C stack:
- **GQA-4 KVQ banking** (`apex_kvq_gqa_bank`, 4× CQ-8 engines) and the
  **per-KV-head composite caches** (`apex_wcomp_bank`, F6(i))
- **`CFG_DM` = 3584** — the gap-D width split (model-wide family separated
  from the per-head family)
- the **multi-head walk fix** (`seq_layer_walker` W_DRAIN, 34d2077) — verified
  present in the synthesized source; without it an AFI would be one commit
  short of the thing worth demoing
- the fmt=1 walker (D-029), projection-bias seam and q-sink (both parameter-
  gated off), S12 mask, W4B feeder (CSR-disabled)

## The memory story (why the first attempt died)

The identical configuration OOM-killed Vivado on a **30 GiB** c6a.4xlarge at
`Phase 22.1.1 Leaf ClockOpt Init` during routing (`anon-rss 31,466,252 kB`).
On 61 GiB it routed to completion with no OOM. **Diagnostic caveat worth
keeping:** Vivado's self-reported `peak = 12,477 MB` in that failed run looks
like ample headroom — the real signal was system-wide free bottoming at
5,953 MB. Reading Vivado's own memory line alone leads to the wrong
conclusion.

## Scope — what is NOT established

- **`APEX_CL_DDR=0`: the DDR fuel line is compiled out of this build.** The
  leg W-G3 proved in simulation with real Qwen `Wq` weights is absent here;
  this DCP proves nothing about it. A DDR=1 build is the natural next one.
- No AFI yet, and **nothing on silicon**. This is a routed checkpoint.
- Routing succeeded at this clock; that is not a claim about any other clock.

## Provenance note (process)

Two DCP builds were briefly running concurrently on this box (integration and
the W4B session both launched one). The duplicate was identified by walking
each engine's process ancestry — a launcher-name check could not see it,
because the integration launcher `exec`s itself away, and the orphaned build
had been reparented to init. The duplicate was killed **during synthesis, before
any checkpoint was written**, and the surviving build's CL sources were diffed
byte-for-byte against a fresh extract of 3173fd5 (identical) before it
continued. The reliable check is `pgrep -f aws_build_dcp_from_cl.py` —
system-wide and ppid-agnostic. A `~/BUILD_OWNER` claim file is now the
protocol on this box.

**Spend:** m6a.4xlarge ~1.7 h ≈ $1.20 (plus the earlier failed attempt ≈ $1.60).
No F2 instance launched.
