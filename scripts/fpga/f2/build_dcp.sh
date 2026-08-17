#!/bin/bash
# build_dcp.sh — APEX F2 CL → post-route DCP tarball (S15 prep).
# ⏸ VALIDATION-DEFERRED: never executed. Runs ON AN EC2 DEV BOX (FPGA
# Developer AMI ≥1.19.x, aws-fpga branch f2 @ v2.3.3, hdk_setup.sh sourced),
# NOT on the 18 GB local machine. See BRINGUP.md §3 step 4.
set -euo pipefail

: "${AWS_FPGA_REPO_DIR:?source aws-fpga/hdk_setup.sh first (sets AWS_FPGA_REPO_DIR)}"
: "${CL_DIR:?export CL_DIR=<path to cl_apex (CL_TEMPLATE copy with APEX rtl/)>}"
[ -f "$CL_DIR/build/scripts/aws_build_dcp_from_cl.py" ] || {
  echo "ABORT: $CL_DIR does not look like an F2 CL dir (copy CL_TEMPLATE first)"; exit 1; }

# Kit-pinned build entry point (Python on F2, replaces F1's .sh):
#   30-90 min for kit examples on a c6a.8xlarge-class box; APEX likely similar.
#
# DECISION-LC-1 (RESOLVED 2026-07-21): the CL is two-clock — tile on
# AWS_CLK_GEN o_clk_extra_a1 under RECIPE A2 = 15.625 MHz. --aws_clk_gen is
# a HARD GATE for any recipe flag (the kit script aborts otherwise). The
# D=128 trace config additionally needs +define+APEX_CL_D=128 in the synth
# tcl (2964eb8 lesson: without it the AFI silently builds D=64).
#
# RECIPE OVERRIDE (CLOCK_LADDER.md §7.1): "$@" is appended AFTER the pinned
# A2 below, and aws_build_dcp_from_cl.py parses with optparse default-store
# semantics — the LAST occurrence wins. So a 62.5 MHz tile build is
#     bash build_dcp.sh --clock_recipe_a A0
# with NO edit here; use build_a0_62p5.sh, which also runs the timing gate
# (the kit echoes the effective value as `clock_recipe_a : A0` in the
# option banner — verify it in the log, never assume it).
cd "$CL_DIR/build/scripts"
python3 aws_build_dcp_from_cl.py -c cl_apex \
  --aws_clk_gen --clock_recipe_a A2 \
  "$@" 2>&1 | tee "$CL_DIR/build/apex_dcp_build.log"

echo "== tarball(s):"
# the kit writes the tar to checkpoints/ and only sometimes mirrors it to
# to_aws/ (a prior failed build leaves to_aws/ moved aside as
# to_aws_backup_*). Accept ONLY the tar whose timestamp TAG matches THIS
# build's newest post_route dcp — a stale tar from an earlier build must
# never mask a failed one (2026-08-07: a timing-MET build was misreported
# "no tarball" by the to_aws-only check; the naive both-dirs fallback
# would have the opposite failure mode).
newest_dcp="$(ls -t "$CL_DIR"/build/checkpoints/*.post_route.dcp 2>/dev/null | head -1)"
tag="$(basename "${newest_dcp:-}" | sed -E 's/^cl_[^.]*\.//; s/\.post_route\.dcp$//')"
tarhit="$(ls "$CL_DIR"/build/checkpoints/to_aws/${tag}.Developer_CL.tar \
             "$CL_DIR"/build/checkpoints/${tag}.Developer_CL.tar 2>/dev/null | head -1)"
[ -n "$tarhit" ] && ls -la "$tarhit" || {
  echo "no tarball for build tag '${tag}' — check apex_dcp_build.log (timing? DRC?)"; exit 1; }
echo "== next: bash create_afi.sh <tarball> (BRINGUP.md step 5)"
