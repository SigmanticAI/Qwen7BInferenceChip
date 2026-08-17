#!/bin/bash
# build_a0_62p5.sh — the 62.5 MHz (recipe A0) tile image: build recipe +
# numeric validation ladder. docs/design/CLOCK_LADDER.md §7-§8 is the design
# doc; THIS file is the runnable form.
#
# ⏸ NEVER AUTO-RUN. Every rung is a separate, explicit subcommand; `plan`
# (the default) only prints. Build rungs run ON THE DEVBOX (FPGA Developer
# AMI, aws-fpga f2 @ v2.3.3, hdk_setup.sh sourced) — the same box/flow as
# build_dcp.sh. The card rungs are PRINTED, not executed: they need a live
# F2 instance and the owner runs them.
#
# THE ONE-LINE SUMMARY OF THE MECHANISM (verified against the kit, see
# CLOCK_LADDER.md §1.2): the tile frequency is constrained AUTOMATICALLY
# from --clock_recipe_a (the A-group MMCM's derived clock; A0 branch =
# mult 15 / div0 24 -> 62.5 MHz, aws_clock_properties.tcl:90-111). No RTL
# edit, no xdc edit, no tcl edit. build_dcp.sh appends "$@" after its pinned
# A2 and the kit's optparse store semantics make the LAST flag win.
#
# GEOMETRY: defaults below are the E-6 convergence2 image
# (agfi-0bc20880b50f5faba, docs/results/prompt_on_chip/E6_ON_SILICON.md) —
# the newest netlist, the one the walked path (E-5/E-6) actually exercises,
# and the one whose A2 sign-off carries the walked epilogue + fuel + DDR
# logic. ALL SIX knobs must be exported or synth_cl_apex.tcl silently
# reverts to the D=128 defaults (the mislabeled-image trap of 25ddb66,
# already paid for once).
#
# Usage:
#   bash build_a0_62p5.sh                 # print the full ladder (no action)
#   bash build_a0_62p5.sh build           # devbox: DCP build at A0 + gate
#   bash build_a0_62p5.sh check-timing    # gate an existing $CL_DIR build
#   bash build_a0_62p5.sh prv             # devbox: pr_verify the routed DCP
#   bash build_a0_62p5.sh ingest <tar>    # create-fpga-image via create_afi.sh
#   bash build_a0_62p5.sh card            # PRINT the on-card A/B ladder
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── the A0 geometry (override via env only if you know why) ─────────────────
: "${APEX_CL_D:=64}"
: "${APEX_CL_DMODEL:=64}"
: "${APEX_CL_GQA:=2}"
: "${APEX_CL_DM:=896}"
: "${APEX_CL_QSTAGE:=14}"
: "${APEX_CL_DDR:=1}"
export APEX_CL_D APEX_CL_DMODEL APEX_CL_GQA APEX_CL_DM APEX_CL_QSTAGE APEX_CL_DDR

GEOM="D=$APEX_CL_D DMODEL=$APEX_CL_DMODEL GQA=$APEX_CL_GQA DM=$APEX_CL_DM QSTAGE=$APEX_CL_QSTAGE DDR=$APEX_CL_DDR"

say()  { echo "[a0] $*"; }
die()  { echo "[a0] FAIL: $*" >&2; exit 1; }

plan() {
  cat <<'EOF'
════════════════════════════════════════════════════════════════════════════
 THE A0 (62.5 MHz) LADDER — run each rung EXPLICITLY, read each gate
 (design doc: docs/design/CLOCK_LADDER.md §7-§8; nothing here auto-runs)
════════════════════════════════════════════════════════════════════════════
 rung 0  LOCAL, FREE, FIRST — preserve the div-2 (=62.5 MHz-ratio) records
         the docs currently only claim (CLOCK_LADDER.md §3.2):
           python3 scripts/fpga/f2/elane_walk_norm.py --tile-div 2 \
               --out build/f2_elane_walk_div2
           (verif/f2sim) make run D=64 ... TILE_DIV=2 on the 7-job 0.5B set
         PASS = bit-exact, same check counts as the div-5 records.

 rung 1  DEVBOX BUILD (~70 min m6a.4xlarge + gate):
           export CL_DIR=<assembled cl_apex dir>   # BRINGUP.md §3 step 3
           source <aws-fpga>/hdk_setup.sh
           bash scripts/fpga/f2/build_a0_62p5.sh build
         The build is exactly build_dcp.sh --clock_recipe_a A0 with the
         six geometry knobs exported; `check-timing` then gates:
           G1 the kit echoed  clock_recipe_a : A0   (never assume)
           G2 the repo xdc was read AND "APEX CDC gate OK" printed
              (the constraints-never-read trap of 2026-08-05)
           G3 no A-group "Clock recipe ... not applied" warning
              (B/C/HBM warnings are benign — those groups are disabled)
           G4 tile path group clk_out1_clk_mmcm_a shows period=16.000ns
              (64.000 = A2 leaked through; 8.000 = XCI default, recipe
               silently skipped — the attempt-4 failure class)
           G5 "All user specified timing constraints are met", no
              *VIOLATED* checkpoint, tarball exists
         READ (not just grep) the top clk_out1_clk_mmcm_a paths: the one
         cone never timed at 16 ns is the wide-norm chained DSP multiply
         (rtl/asu/asu_rmsnorm.sv:342-343) — the standing instruction to
         READ the mu-multiply timing from the report applies (§6).
         NOTE the overall WNS is expected to sit on the AWS DDR4 core's
         own intra-domain path (~+0.3/+0.4 as at A2) — that number is
         recipe-independent and is NOT the tile verdict. Read the GROUP.

 rung 2  PRV (devbox, ~90 s):  bash scripts/fpga/f2/build_a0_62p5.sh prv
         Expected clean at DM=896 scale (HDPRVerify-41 triggers at
         DM=3584-class; DM=896 has ingested 8x first-try, account-verified).

 rung 3  INGEST:  bash scripts/fpga/f2/build_a0_62p5.sh ingest <tarball>
         (create_afi.sh; free; record the AGFI id.)

 rung 4  REGISTER THE IMAGE — BEFORE it is ever flown:
           scripts/fpga/f2/clock_key.py IMAGE_RECIPE +=
             "agfi-XXXX...": (0, "A0 62.5 MHz image apex-b64-a0-<date> — "
                                 "TIMING MET <WNS> at A0 (<report path>)"),
         One line, one commit. An unregistered AGFI REFUSES at bring-up
         and at every job — that is the gate working, not an obstacle.

 rung 5  ON CARD (owner runs; `card` prints the exact commands):
         load -> program -a 0 -> NUMERIC a1-column check = 62.50 ->
         identity reads -> the A/B: the ENTIRE proven suite first at
         --run-recipe 2 (15.625, setup-safe underclock of the A0 image),
         then at A0's own 62.5 — byte-identical captures both arms.
════════════════════════════════════════════════════════════════════════════
EOF
  say "geometry for 'build': $GEOM"
  say "nothing was executed. Next: 'build' on the devbox."
}

check_timing() {
  : "${CL_DIR:?export CL_DIR=<the cl_apex CL dir that was built>}"
  local log="$CL_DIR/build/apex_dcp_build.log"
  [ -f "$log" ] || die "no build log at $log (build_dcp.sh tees it there)"
  local rpt
  rpt="$(ls -t "$CL_DIR"/build/reports/*.post_route_timing.rpt 2>/dev/null | head -1 || true)"
  [ -n "$rpt" ] || die "no post_route_timing.rpt under $CL_DIR/build/reports"
  say "log: $log"
  say "rpt: $rpt"

  # G1 — the kit echoed the EFFECTIVE recipe (optparse banner; never assume)
  grep -Eq 'clock_recipe_a[[:space:]]*:[[:space:]]*A0' "$log" \
    || die "G1: the kit did not echo 'clock_recipe_a : A0' — the A2 default won. Check the build_dcp.sh invocation order."
  say "G1 ok: kit echoed clock_recipe_a : A0"

  # G2 — the repo constraints were read and MATCHED (2026-08-05 trap)
  grep -q "APEX: user constraints from APEX_REPO_DIR" "$log" \
    || die "G2: cl_synth_user.xdc was not read from APEX_REPO_DIR — the constraints-never-read trap"
  grep -q "APEX CDC gate OK" "$log" \
    || die "G2: 'APEX CDC gate OK' absent — the xdc did not match the netlist"
  say "G2 ok: repo xdc read + CDC gate OK"

  # G3 — recipe application: warnings naming the DISABLED groups are benign,
  # anything else mentioning an unapplied clock recipe is fatal
  local badrec
  badrec="$(grep -i "clock recipe" "$log" | grep -i "not applied" \
            | grep -Eiv 'grp_b|grp_c|group b|group c|hbm|_b_|_c_' || true)"
  [ -z "$badrec" ] || die "G3: an A-group clock-recipe warning: $badrec"
  say "G3 ok: no A-group recipe warning (B/C/HBM ones are the known benign set)"

  # G4 — the tile group is REALLY timed at 16 ns
  grep -q "clk_out1_clk_mmcm_a" "$rpt" \
    || die "G4: no clk_out1_clk_mmcm_a in the report — wrong report or no A-group MMCM"
  grep "clk_out1_clk_mmcm_a" "$rpt" | grep -q "period=16.000ns" \
    || die "G4: tile group is not at period=16.000ns — recipe not applied to timing"
  if grep "clk_out1_clk_mmcm_a" "$rpt" | grep -q "period=64.000ns"; then
    die "G4: tile group still shows period=64.000ns (A2 leaked through)"
  fi
  if grep "clk_out1_clk_mmcm_a" "$rpt" | grep -q "period=8.000ns"; then
    die "G4: tile group at the 8 ns XCI default — aws_clock_properties.tcl was skipped (instance name? attempt-4 class)"
  fi
  say "G4 ok: clk_out1_clk_mmcm_a timed at period=16.000ns"

  # G5 — MET, no VIOLATED checkpoint, tarball present
  grep -q "All user specified timing constraints are met" "$rpt" \
    || die "G5: the report does not say all constraints are met"
  if ls "$CL_DIR"/build/checkpoints/*VIOLATED* >/dev/null 2>&1; then
    die "G5: a *VIOLATED* checkpoint exists — timing did NOT close"
  fi
  ls "$CL_DIR"/build/checkpoints/to_aws/*.Developer_CL.tar >/dev/null 2>&1 \
    || die "G5: no Developer_CL.tar — the build did not complete"
  say "G5 ok: MET, no VIOLATED checkpoint, tarball present"

  echo
  say "── READ THIS, do not just trust the greps (CLOCK_LADDER.md §6) ──"
  say "worst tile-group paths (first 3 'Slack' entries in the group):"
  grep -n -B2 -A8 "Path Group:             clk_out1_clk_mmcm_a" "$rpt" \
    | head -60 || true
  echo
  say "the never-timed-at-16ns cone to look for by name: u_norm/asu_rmsnorm"
  grep -n "rmsnorm" "$rpt" | head -10 || say "  (no rmsnorm cell named in the top paths — good sign, still read the group)"
  echo
  say "CHECK-TIMING: ALL GATES PASSED — rung 2 (prv) is next"
}

cmd_build() {
  : "${AWS_FPGA_REPO_DIR:?source aws-fpga/hdk_setup.sh first (devbox only)}"
  : "${CL_DIR:?export CL_DIR=<path to assembled cl_apex dir> (BRINGUP.md §3)}"
  say "geometry: $GEOM  (all six exported — the D=128 silent-revert trap)"
  say "invoking build_dcp.sh --clock_recipe_a A0 (last flag wins in the kit)"
  bash "$HERE/build_dcp.sh" --clock_recipe_a A0
  say "build done — running the timing gate"
  check_timing
}

cmd_prv() {
  : "${AWS_FPGA_REPO_DIR:?source aws-fpga/hdk_setup.sh first (devbox only)}"
  : "${CL_DIR:?export CL_DIR=<the built cl_apex dir>}"
  local dcp shell_dcp tcl
  dcp="$(ls -t "$CL_DIR"/build/checkpoints/*.post_route.dcp 2>/dev/null | head -1 || true)"
  [ -n "$dcp" ] || die "no post_route.dcp under $CL_DIR/build/checkpoints"
  shell_dcp="$(ls "$AWS_FPGA_REPO_DIR"/hdk/common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp 2>/dev/null || true)"
  [ -n "$shell_dcp" ] || die "shell checkpoint cl_bb_routed.small_shell.dcp not found under the kit"
  tcl="$(mktemp /tmp/a0_prv_XXXX.tcl)"
  {
    echo "open_checkpoint $dcp"
    echo "pr_verify -full_check -in_memory -additional $shell_dcp"
    echo "puts {A0_PRV_DONE}"
  } > "$tcl"
  say "pr_verify on $dcp (procedure: docs/results/f2_afi_ingest/ESCALATION.md:158-165)"
  vivado -mode batch -source "$tcl" -nolog -nojournal | tee "$CL_DIR/build/a0_prv.log"
  grep -q "A0_PRV_DONE" "$CL_DIR/build/a0_prv.log" || die "pr_verify did not complete"
  if grep -q "HDPRVerify-41" "$CL_DIR/build/a0_prv.log"; then
    die "HDPRVerify-41 present — do NOT submit; see f2_afi_ingest/ESCALATION.md"
  fi
  say "PRV: no HDPRVerify-41 — rung 3 (ingest) is next"
}

cmd_ingest() {
  local tar="${1:?usage: build_a0_62p5.sh ingest <path/to/*.Developer_CL.tar>}"
  say "submitting via create_afi.sh as apex-b64-a0-$(date +%Y%m%d)"
  bash "$HERE/create_afi.sh" "$tar" "apex-b64-a0-$(date +%Y%m%d)"
  say "RECORD the AGFI, then rung 4: register it in clock_key.py IMAGE_RECIPE"
  say "  as (0, \"...receipt...\") BEFORE any card session."
}

cmd_card() {
  cat <<'EOF'
════════════════════════════════════════════════════════════════════════════
 ON-CARD LADDER for the A0 image (PRINTED ONLY — the owner runs this)
 Precondition: the AGFI is REGISTERED in clock_key.py IMAGE_RECIPE as
 recipe 0 (rung 4). Unregistered = every tool refuses, by design.
════════════════════════════════════════════════════════════════════════════
 1  load + program + verify (an AFI load resets the MMCMs to A1=125 MHz;
    for an A0 image that is 2x over — the recipe step stays mandatory):
      sudo fpga-load-local-image -S 0 -I <AGFI>
      sudo fpga-load-clkgen-recipe -S 0 -a 0
      python3 scripts/fpga/f2/remote_hw_exec.py --check-clock \
          --host <ubuntu@ip> --key <pem> --agfi <AGFI>
    PASS = a1 column NUMERICALLY 62.50 (the a3-also-prints-62.50-under-A2
    trap is defeated by the header-indexed parser; never grep a bare 62.5).

 2  identity reads (cheap go/no-go): BAR0 0x0000 == "A9EX"; VERSION;
    KVQ INFO_DIM == 0x40; INFO_TIER == 0x1 (GQA=2, CQ-8-only); INFO_GROUP.
    FUEL_STAT ddr_ready=1 after the DDR image load (DDR_BRINGUP_RESULT.md).

 3  ARM A — the SAME image at 15.625 (setup-safe underclock, attributes any
    later divergence to FREQUENCY, not to the new netlist):
      sudo fpga-load-clkgen-recipe -S 0 -a 2
      APEX_F2_AGFI=<AGFI> APEX_F2_RUN_RECIPE=2  <the proven suite>
    Suite = the 7-job real-0.5B trace replay (1,052 checks) via
    remote_hw_exec; prompt05b --layers 0,1 --executor hw (geometry audit +
    INFO_D per program + token verdict — the 2026-08-03 flight, 4,380
    programs); the E-5 gate (walk_fuel_qkv 144/144 + walk-off RED + DDR
    poison RED) and the E-6 gate (walk_oproj / walk_qkv_oproj r1 896/896 +
    both discriminators) exactly as in their result docs.
    (The D=128 18-job set does NOT apply — INFO_D refuses it by design.)

 4  ARM B — flip ONLY the clock and rerun everything:
      sudo fpga-load-clkgen-recipe -S 0 -a 0
      APEX_F2_AGFI=<AGFI>  <the same suite>          # no RUN_RECIPE
    PASS = byte-identical captures and identical verdicts vs ARM A.
    ANY divergence at 62.5 that ARM A did not show = a real >15.625 MHz
    timing escape: STOP, keep A2, file the failing path from the report.

 5  DISCLOSE (CLOCK_LADDER.md §9): the walked-path prediction is ~70 ms/layer
    at 62.5 vs ~280 ms at 15.625 (E6_ON_SILICON.md, labelled prediction);
    per-op host-dispatched jobs barely move. Never book "4x" as an
    end-to-end demo number without the walked-path caveat.
════════════════════════════════════════════════════════════════════════════
EOF
}

case "${1:-plan}" in
  plan)          plan ;;
  build)         cmd_build ;;
  check-timing)  check_timing ;;
  prv)           cmd_prv ;;
  ingest)        shift; cmd_ingest "$@" ;;
  card)          cmd_card ;;
  *)             die "unknown subcommand '${1}' (plan|build|check-timing|prv|ingest|card)" ;;
esac
