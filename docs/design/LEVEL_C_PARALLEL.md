# Level C — "full 7B model runs autonomously on the FPGA": parallel build plan

The goal beyond stage-2 (one attention job on silicon): the whole model
computes token-by-token on the F2 FPGA, host only sends the prompt / reads
logits. That needs several new **verified** RTL blocks. This file is the
coordination hub so multiple sessions build them in parallel without
colliding.

## Why this parallelizes well

Every block has a **golden reference** and its own Verilator testbench —
the block's contract is "match the golden bit-exact," which needs no other
block to exist. So the three cleanly-independent blocks below can be built
and unit-verified in separate sessions, then integrated.

## The three parallel lanes (each its own branch + contract + session)

| block | branch | contract | golden ref | collision surface |
|---|---|---|---|---|
| **B1 layer-walker** (long pole) | `comp/b1-walker` | `docs/design/B1_WALKER.md` | the `gen_l3_vectors` op schedule (spec, §OPTIMIZATION 38) | `rtl/seq/seq_walker.sv`, `rtl/csr/csr_regs.sv` — own to this lane |
| **B3 weight path** | `comp/b3-weight-path` | `docs/design/B3_WEIGHT_PATH.md` | extended golden W4 dequant vs C-2 sweep | `rtl/mxe/*`, `rtl/seam/seam_feeder_quant.sv` |
| **wide-D RMSNorm** | `comp/wide-rmsnorm` | `docs/design/WIDE_RMSNORM.md` | C-RMSW (golden/tests/test_7b_plumbing.py) | `rtl/asu/asu_rmsnorm.sv` only |

The three lanes touch **disjoint** RTL modules, so no file contention. The
integration (composing the full layer + the F2 CL DRAM path) is the
serial "combine" step that follows.

## The two real bottlenecks (and their unlocks)

1. **One local machine — Verilator builds swap-kill each other.** Parallel
   authoring is fine; parallel *verification* serializes on the 18 GB box.
   **Unlock:** each lane runs its Verilator unit-verify on its own cheap AWS
   box (`c6a.xlarge`, ~$0.60/hr) instead of queuing locally — the same AWS
   account already used for F2. A lane's unit test only needs Python venv +
   Verilator + the repo checkout (no F2 kit), so any box works.
2. **Shared git index across sessions.** **Unlock:** each lane works in its
   own `git worktree` on its branch:
   ```
   git worktree add ../apex-walker comp/b1-walker
   git worktree add ../apex-weightpath comp/b3-weight-path
   git worktree add ../apex-rmsnorm comp/wide-rmsnorm
   ```
   Separate dirs, separate branches, merge at integration. No index fights.

## Honest calendar with full parallelism

- Authoring + unit-verify of the three blocks: **overlaps**, ~days each.
- **Integration + a new AFI build + on-silicon debug: serial, irreducible**
  (multi-hour hardware loop). This tail is why the whole thing is ~2 weeks
  with good parallelism, not days. Parallelism removes the authoring
  serialization, not the integration serialization.

## Rules (all lanes)

- Golden is the arbiter — never edit golden to match RTL.
- No PASS/counts without pasted output; bit-exact vs golden before anything
  claims done; keep `apex_pkg.sv` frozen (APEX_VERSION 0x0001_0000) unless a
  contract explicitly justifies a version bump.
- Commit atomically with explicit paths on your own branch.
- Stage-2 (attention on silicon) proceeds independently on
  `decoder-layer-and-fpga-fit` — see `docs/results/f2_stage2_hw/RESULT.md`.

## Status

- 2026-07-20: branches created; design contracts generated (see the three
  `docs/design/*.md`); RTL not yet started in any lane. Machine free.
- 2026-07-20 (later): **wide-rmsnorm lane COMPLETE** on `comp/wide-rmsnorm`
  (stages 1–5 of WIDE_RMSNORM.md): MU/S params provenance-gated from golden;
  `asu_rmsnorm` wide mean path landed with the D≤128 anchor byte-identical
  (smoke + sb A/B vs e578872, zero test edits); NEW `verif/asu/wide` GREEN —
  bit-exact vs `rmsnorm_fx_wide` at RMS_D_MAX=3584+8192, k-sweep 2..28,
  sum2=2²⁷ corner, 4/4 mutants (evidence: verif/asu/wide/RESULT.md + log).
  Ready for integration; no apex_pkg change, no shared-file collision.
  Integration budget (yosys probe, verif/asu/wide/RESULT.md): wide unit =
  2 DP16KD + 7 MULT18X18D + ~2.5k LUT4 on ECP5; 1 RAMB36E2 + 5 DSP48E2 +
  ~1.2k LUT on US+ — x_buf infers BRAM as-is (comb-read flag retired).
- 2026-07-21: **B1 layer-walker COMPLETE** (`comp/b1-walker`, ARCHITECTURE
  D-028). L3 bit-exact in both host and walker mode; host mode byte-identical
  so the verified path is unchanged. The long pole is done — integration is no
  longer blocked on B1 RTL.
  - `apex_top.sv` gained **two** fenced additive regions (WALK CSR window
    0x5C-0x6C + walk_en mode mux). Merging lanes: expect adjacent-line only.
  - `verif/top/l3/mutate.py` M1/M2 anchors moved to the muxed net names
    (`w_rt_act_src` / `w_rt_squant_src`) — same mutants, 3/3 still caught.
  - Known follow-on **B1b**: grouped tiers (CQ-4/CQ-4+) are REFUSED by walker
    v1, not degraded; they need a commit-time feeder-equivalent amax pass.
    `tip_auto_mixed` needs per-chunk re-routing too, so it is refused for two
    independent reasons.
