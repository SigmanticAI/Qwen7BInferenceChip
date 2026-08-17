# ESCALATION PACKAGE — F2 AFI ingestion HDPRVerify-41 at DM=3584 scale

**Date assembled:** 2026-07-30 · **Assembled in:** `comp/prompt-b-c` (this
tree) · **Companion / source of record:** `docs/results/f2_afi_ingest/ANALYSIS.md`
@ commit `e22c310` ("F2 ingest: E1 RED — GQA=1/DM=3584 fails same slices",
branch `comp/s14-sky130`; the full experiment history including the E1
addendum). This file exists so that filing the vendor case is a copy-paste,
not a project. Every claim below is either sourced to an artifact or flagged
as an inference.

---

## (a) 30-second summary (vendor-triage paragraph)

Building a partial-reconfiguration CL (`cl_apex`, AWS F2 small shell, HDK kit
v2.3.3, shell 0x10212415, Vivado 2025.2) at ~1.3M-cell / `DM=3584`-class
scale, `place_design` deterministically alters the site programming of two
LOCKED STATIC-REGION slices — SLICE_X117Y357 and SLICE_X117Y358, which hold
the static shell's clock-A MMCM DRP state machine — by dropping a
**physical-only VCC tie** (site pin `D6` VCC physical-only net + arc
`D6 -> D6LUT[A6]`) that the shell reference checkpoint
(`cl_bb_routed.small_shell.dcp`, sha256-matching the kit pin) carries.
`pr_verify -full_check` therefore fails with HDPRVerify-41 on those two
sites, and `create-fpga-image` rejects the AFI at ingestion. The failure is
scale-triggered and deterministic: 4/4 builds at DM=3584 fail identically
(including a minimal one-engine, place-only config), 3/3 builds at
DM=128-class scale pass and ingested. Bisection: `post_link` verifies clean,
`post_place` is damaged. Four candidate mechanisms were each exonerated by
direct A/B experiment (kit clock-recipe TCL, placement adjacency, placer
directive `AltSpreadLogic_high`, engine count 4→1). Vivado version was never
varied (all runs 2025.2). All queryable programming at the two sites (INIT,
BEL bindings, LOCK_PINS, site PIPs, pin→BEL maps — 139/139 entries) matches
the reference; the delta is visible only to pr_verify's byte-level site
comparison. We need a supported workaround or an ingestion-side tolerance;
no wide image can ship until then.

## (b) Exoneration matrix — one row per variable tested

| # | Variable | Experiment | Result | Artifact |
|---|----------|-----------|--------|----------|
| 0 | Kit drift (BB checkpoint) | sha256 of the kit's `cl_bb_routed.small_shell.dcp` vs the kit pin; upstream `f2` tip = June 22, predates our clone | **No drift** — hash matches; reference is authoritative | ANALYSIS.md "Established facts" §1; devbox `~/prv.log` |
| 1 | Clock-recipe TCL (`aws_clock_properties.tcl` writing `set_property` on the static shell MMCM) | **ic3**: guarded out the three `proc_set_mmcm $RL_A0_MMCM` calls (patch marker `APEX-SKIP`, `.orig` preserved), rebuilt the SAME design at the SAME recipe | **PRV-RED — exonerated.** Patched build fails identically. Independent corroboration: the July-22 I-A A2 build ran the same recipe on an UNPATCHED kit and ingested fine (`afi-036d83cafa00d26ea`) | devbox `~/rebuild_chain.sh`, `~/apex_dcp_ic3.log`; ANALYSIS.md "RESOLVED MECHANISM" |
| 2 | Placement adjacency (CL cells packed at SLICE_X116Y358, sharing the interconnect column with damaged static X117Y358) | **ic4**: `APEX-KEEPOUT` site PROHIBITs on the CL-facing apron (SLICE_X116Y355..360 + X117Y359..360) appended to `small_shell_cl_pnr_user.xdc`; prohibits verified to HOLD (X116Y357-359 EMPTY at post_place) | **PRV-RED — exonerated.** Same two sites, same missing tie, with the apron empty | devbox kit-side `small_shell_cl_pnr_user.xdc` (+`.orig`); ANALYSIS.md "ic4 result" |
| 3 | Placer directive | place-only run under `AltSpreadLogic_high` + in-memory pr_verify | **RED — exonerated.** Failed identically | ANALYSIS.md:112 ("The last local attempt"). *Note: this run's log file is not individually named in the record; it is preserved with the other experiment artifacts on the apex-f2-devbox EBS* |
| 4 | GQA engine count (4 → 1) | **E1** (2026-07-30): place-only + in-memory pr_verify at GQA=1 / DM=3584 / DDR=0 — the minimal config any wide image needs | **RED — exonerated.** Same two slices, same missing D6 VCC tie, at ONE engine | devbox `~/e1_prv.log`, `~/e1_build.log`; ANALYSIS.md "E1 result" |
| 5 | Design scale (DM=128-class vs DM=3584-class) | Controls: first-light DCP `2026_07_17-202346` (no clkgen recipe) and the July-22 I-A A2 DCP (clkgen recipe, unpatched kit) both pass the identical local check and both ingested; every DM=3584 build fails | **CORRELATED — the trigger.** DM=128-class: 3/3 green. DM=3584-class: 4/4 red. Threshold sits between DM=128 and DM=3584 (not further localized) | devbox `~/prv_ctrl.log` (control passes, `HDPRVerify-42 ... locked static`); ANALYSIS.md facts §2, E1 addendum |
| 6 | Vivado version | **NOT varied.** Every build and every pr_verify in this record ran Vivado 2025.2 | **Open variable.** A 2024.x rebuild was queued, then deliberately deferred (version was never the tested variable; an older toolchain could ADD static deltas vs the BB rather than remove them) | `docs/design/MASTER_TABLE.md` "✂ T3" (this tree) |

Checker validation: the local `pr_verify` procedure agrees with AWS's
ingestion verdict **4-for-4** across the checkpoints where both exist
(ANALYSIS.md "RESOLVED MECHANISM").

## (c) The byte-level delta

`pr_verify`'s own site-diff dumps (devbox
`from_aws/SLICE_X117Y358_{first,second}dcp.txt`; the two failing sites are
SLICE_X117Y357 and SLICE_X117Y358) show exactly one difference per site.
The shell reference (BB) checkpoint has, and our post-place checkpoints
LACK:

```
Site pin 'D6'   VCC   Physical-only net
arc  D6 -> D6LUT[A6]
```

(Excerpt as recorded in ANALYSIS.md "RESOLVED MECHANISM"; the verbatim dump
files are preserved on the apex-f2-devbox EBS and can be attached to the
case on request.)

This is a fractured-LUT unused-input VCC tie, **physical-only** — invisible
to logical Tcl queries, which is why every queryable comparison matched
first: INIT/INIT_A–D, per-pin inversions, BEL bindings, LOCK_PINS, used
site-PIPs, pin→BEL maps are identical 139/139 between our DCP and the BB at
both sites (devbox `~/init_dump.txt`, `~/pip_dump.txt`, `~/site_dump.log`).
The two slices hold the static shell's
`WRAPPER/IO_SHIM/RL_CLKS/MMCM_CLK_MAIN_A/.../mmcm_drp_inst` state machine
and its LUTRAM recipe ROM (`ram_reg[24][35]`, `ram_do_reg[32]`, …), in both
checkpoints, identically placed.

The AWS-side symptom (verbatim, from the failed ingestion of
`2026_07_27-195354.Developer_CL.tar`):

```
ERROR: [Constraints 18-13341] HDPRVerify-41: Static site SLICE_X117Y358 has different
site programming in design checkpoint in-memory-design and ../checkpoints/SH_CL_BB_routed.dcp.
ERROR: [Constraints 18-13341] HDPRVerify-41: Static site SLICE_X117Y357 ... (same)
ERROR: [Common 17-39] 'pr_verify' failed due to earlier errors.   (ingest.tcl line 82)
```

AWS ingestion logs:
`s3://apex-f2-dcp-099597653601/afi-logs/ic_gqa4/afi-04b0d37f1789bb2f6/`.

## (d) Stage bisection — and the later probe, stated honestly

**Bisection** (local pr_verify on the ic3 build's saved checkpoints,
ANALYSIS.md "RESOLVED MECHANISM"):

- `post_link` → **CLEAN**
- `post_place` → **DAMAGED** (both sites) — everything after inherits.

**Later probe** (2026-07-29, recorded in the project log; probe artifacts on
the devbox EBS, not committed to a tree): tracking the affected site pin's
`IS_USED` property through the flow shows **1 → 1 through `post_opt`** (the
property never changes), and **pr_verify on the `post_opt` checkpoint is
clean**. This narrows the damaging stage to `place_design` proper —
`opt_design` is exonerated.

**The tension, honestly:** during our investigation we found an
AMD-documented family of Vivado 2025.2-era defects around dropped
unused-LUT-input tie programming (knowledge-base articles 000040413,
000040409, 000035406 — identified in our notes as the closest documented
siblings; *the association is our reading of those articles, not a vendor
confirmation*). That documented family fingerprints in **`opt_design`**,
with `IS_USED` visibly changing. Our failure preserves everything through
opt — clean pr_verify at post_opt, `IS_USED` untouched — and dies inside
**`place_design`**, where the resulting delta is *below* the
property-queryable layer entirely (the post-place checkpoint still looks
correct to every Tcl query we ran; only pr_verify's byte-level site
comparison sees the missing tie). What this implies (inference, flagged as
such):

1. If this is the same underlying defect class, it has a placer-phase
   manifestation the published KB articles do not describe — so their
   listed workarounds (which target opt_design behavior) cannot be assumed
   to apply, and we did not find a documented placer-side switch to try
   beyond the directive already exonerated in row 3.
2. Because the damage is invisible to logical queries, no user-level Tcl
   probe can detect or repair it post-hoc; only pr_verify (ours or AWS's)
   can attest a checkpoint. A fix or workaround has to come from the tool
   or the ingestion policy — which is why this escalation exists.

## (e) Exact reproduction steps (runnable by AWS/AMD)

1. **Host:** launch the FPGA Developer AMI (Ubuntu) 1.19.2
   (`ami-07a164f1a402ab274`, product `prod-rhng4b6alkhdq`), instance
   `m6a.4xlarge` or larger (a 30 GiB `c6a.4xlarge` OOMs during routing of
   this design — `vivado invoked oom-killer`, 31.4 GiB resident; see
   `docs/results/ic_dcp_probe/RESULT.md`). Vivado 2025.2 as shipped on the
   AMI.
2. **Kit:** `git clone` aws-fpga branch `f2` at **v2.3.3**; `source
   hdk_setup.sh`. Confirm the BB pin:
   `hdk/common/shell_stable/.../from_aws/cl_bb_routed.small_shell.dcp`
   sha256 matches the kit's pin file (it did for us — no drift).
3. **CL:** `cl_apex` from this repo, `scripts/fpga/f2/cl_apex/` (design +
   constraints + build scripts), copied over a CL_TEMPLATE skeleton per
   `scripts/fpga/f2/BRINGUP.md`. Failing configuration defines:
   `APEX_CL_D=128  APEX_CL_GQA=4  APEX_CL_DM=3584  APEX_CL_DDR=0`.
   (Minimal trigger, per E1: `APEX_CL_GQA=1  APEX_CL_DM=3584  APEX_CL_DDR=0`
   — place-only suffices, no route needed.)
4. **Build** (per `scripts/fpga/f2/build_dcp.sh`):
   ```
   cd $CL_DIR/build/scripts
   python3 aws_build_dcp_from_cl.py -c cl_apex --aws_clk_gen --clock_recipe_a A2
   ```
   The reference failing build is `2026_07_27-195354` (routed, timing MET,
   worst slack +0.711 ns, tarball `2026_07_27-195354.Developer_CL.tar`,
   191,272,960 bytes — `docs/results/ic_dcp_probe/RESULT_ROUTED.md`).
5. **Local check (no AWS submission needed):** open the post-place (or any
   later) checkpoint in Vivado 2025.2 and run
   ```
   pr_verify -full_check -in_memory -additional <kit>/…/from_aws/cl_bb_routed.small_shell.dcp
   ```
   Expected: `HDPRVerify-41` on SLICE_X117Y357 and SLICE_X117Y358.
   On the same design's `post_link` (and `post_opt`) checkpoint the same
   command reports clean (`HDPRVerify-42 … locked static`).
6. **End-to-end check (optional):** submit the tarball via
   `aws ec2 create-fpga-image … --logs-storage-location <s3>` — ingestion
   fails with the identical two errors (our failed submission:
   `afi-04b0d37f1789bb2f6`, logs at the S3 path in (c)).
7. **Negative control:** the same flow at `APEX_CL_DM=128`-class scale
   passes pr_verify and ingested twice for us
   (`afi-03232169ea962036d` first-light, `afi-036d83cafa00d26ea` I-A A2 —
   the latter currently running production jobs, see
   `docs/results/prompt_on_chip/RESULT.md`).

## (f) What we are asking for

1. **A supported workaround** — a placer parameter/directive/patch for
   Vivado 2025.2 that preserves static-region physical tie programming at
   this design scale (the one directive we tried, `AltSpreadLogic_high`, is
   exonerated in row 3 above).
2. **An ingestion-policy answer** — can `create-fpga-image`'s pr_verify
   tolerate deltas that are *provably physical-only VCC ties on static
   sites* (functionally a constant; every logical/queryable property
   matches)? If yes, under what attestation?
3. **A tracking id** — for the tool defect (AMD) and/or the ingestion case
   (AWS), so future kit/tool updates can be tested against a named issue.

Attachments available on request (all preserved on the apex-f2-devbox EBS):
build logs, pr_verify logs for every stage checkpoint, the two site-diff
dumps, manifests, and every checkpoint (post_link / post_opt / post_place /
routed) of the failing and control builds.

## (g) Environment

| Item | Value |
|---|---|
| AWS account / region | 099597653601 / us-west-2 |
| HDK kit | aws-fpga branch `f2`, **v2.3.3** (BB checkpoint sha256 matches kit pin) |
| Shell | small shell, AFI manifest shell version **0x10212415** |
| Vivado | **2025.2** (only version ever used — see matrix row 6) |
| Build AMI | FPGA Developer AMI (Ubuntu) 1.19.2, `ami-07a164f1a402ab274` (`prod-rhng4b6alkhdq`) |
| Build instances | c6a.4xlarge (30 GiB — OOM during route at this scale), **m6a.4xlarge** (61 GiB — completes) |
| Runtime instance | f2.6xlarge (VU47P), us-west-2 |
| Failing AFI (this session) | `afi-04b0d37f1789bb2f6` — logs: `s3://apex-f2-dcp-099597653601/afi-logs/ic_gqa4/afi-04b0d37f1789bb2f6/` |
| Failing AFI (second, independent) | submitted by a second session against the same design lineage; **its AFI id was not recorded in our docs** — recoverable from the account's `describe-fpga-images`/`create-fpga-image` history. *Flagged: id missing from the written record.* |
| Available (passing) AFI | `afi-036d83cafa00d26ea` / `agfi-0ae06ea568e5667ba` (I-A A2, D=128, GQA_NENG=1) — plus the first-light control `afi-03232169ea962036d` / `agfi-0f7c93ffa798ecc3f` (D=64, no clkgen recipe) |
| Failing tarball | `2026_07_27-195354.Developer_CL.tar` (191,272,960 bytes; routed, slack +0.711 ns) |
| Design scale | ~1.3M CL cells at `APEX_CL_DM=3584`; failing sites SLICE_X117Y357/358 (static shell `MMCM_CLK_MAIN_A` DRP block) |

## Ready-to-send case text

The subject/body/ask drafted for the AWS support case is in ANALYSIS.md
("AWS SUPPORT CASE — ready to file", @ `e22c310`), and should be filed with
**the E1 addendum appended** (it preempts a "reduce your utilization" first
response: one engine at DM=3584 still fails, so utilization is not the
lever). Filing is an owner action (account 099597653601).

---
*Provenance note: this package was assembled read-only from
`docs/results/f2_afi_ingest/ANALYSIS.md` @ `e22c310` (branch
`comp/s14-sky130`), `docs/results/ic_dcp_probe/RESULT.md` +
`RESULT_ROUTED.md`, `docs/design/MASTER_TABLE.md`,
`docs/results/prompt_on_chip/RESULT.md` (this tree), and the project log of
2026-07-29 for the IS_USED probe. Devbox artifact paths (`~/…`,
`from_aws/…`) live on the stopped apex-f2-devbox's EBS volume, preserved
per ANALYSIS.md.*
