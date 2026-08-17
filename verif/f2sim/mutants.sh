#!/usr/bin/env bash
# mutants.sh — IB-FUEL s4 RED-mutant gate (docs/design/IB_FUEL.md §1.5).
#
# Applies each fuel-path mutation ALONE to the clean tree, builds it into a
# throwaway obj dir (the real gate binary is never touched), runs the
# single-job fuel smoke, and REQUIRES the sim's own exit code to be nonzero
# — pipefail is ON so `| tail` cannot mask a surviving mutant (the exact
# failure class an audit on 2026-07-23 flagged). A clean-binary CONTROL must
# exit 0, so a harness that failed everything could not launder a survivor
# into a pass.
#
# SAFETY: an EXIT trap reverts all three RTL files unconditionally, so an
# aborted run can never strand a mutation on disk (an earlier version did,
# via a path bug — that is why the trap exists).
#
# Exit 0 iff: control PASS(exit 0) AND all three mutants FAIL(exit nonzero).
set -u -o pipefail

FS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$FS/../.." && pwd)"
KIT="${AWS_FPGA:-$REPO/tools/aws-fpga}"
JOB="$REPO/build/fuel_regops/job_s019_L19_h03.fuel.regops.jsonl"
IMG="+ddr_image=$REPO/build/ddr_image/ddr_image.bin +ddr_regions=$REPO/build/ddr_image/ddr_image.regions.jsonl +fuel_audit"

# repo-relative (so `git -C "$REPO" checkout "$X"` resolves correctly)
R=scripts/fpga/f2/cl_apex/design/apex_fuel_reader.sv
C=scripts/fpga/f2/cl_apex/design/apex_fuel_ctl.sv
A=scripts/fpga/f2/cl_apex/design/apex_afifo.sv
rc=0

cleanup() { git -C "$REPO" checkout -- "$R" "$C" "$A" >/dev/null 2>&1; rm -rf "$FS/obj_mut"; }
trap cleanup EXIT

cd "$FS"
wait_vl() { until ! pgrep -f verilator_bin >/dev/null; do sleep 5; done; }

apply() { python3 - "$REPO/$1" "$2" "$3" <<'PY'
import sys
p,old,new=sys.argv[1:4]
s=open(p).read(); assert s.count(old)==1, f"{p}: {s.count(old)} matches"
open(p,"w").write(s.replace(old,new))
PY
}

# $1 label  $2 objdir  $3 want(PASS|FAIL); captures the SIM's exit via
# PIPESTATUS[0] under pipefail so `| tail` cannot mask it
gate() {
  ./"$2"/f2sim +tile_div=8 $IMG "$JOB" 2>&1 | tail -2; local e=${PIPESTATUS[0]}
  local got=PASS; [ "$e" -ne 0 ] && got=FAIL
  if [ "$got" = "$3" ]; then echo "GATE[$1]: exit=$e want=$3 got=$got  OK"
  else echo "GATE[$1]: exit=$e want=$3 got=$got  *** WRONG ***"; rc=1; fi
}

mut() { # $1 label  $2 repo-rel file  $3 old  $4 new
  apply "$2" "$3" "$4"; wait_vl
  make build AWS_FPGA="$KIT" D=128 DDR=1 OBJ=obj_mut >/dev/null 2>&1 \
    || { echo "GATE[$1]: BUILD FAILED"; rc=1; git -C "$REPO" checkout -- "$2"; return; }
  gate "$1" obj_mut FAIL
  git -C "$REPO" checkout -- "$2"
}

echo "== CONTROL (clean gate binary must PASS) =="
gate control obj_d128_ddr1 PASS
echo "== M1 reader lane-slice swap =="
mut M1 "$R" "assign fw_data  = unpk_q[63:0];" "assign fw_data  = unpk_q[511:448];"
echo "== M2 record beats off-by-one =="
mut M2 "$C" "assign fr_beats_64B = eng_rec_q[55:30];" "assign fr_beats_64B = eng_rec_q[55:30] + 26'd1;"
echo "== M3 afifo read-pointer wrap bit lost =="
mut M3 "$A" "wire  [LOG2D:0] rptr_bin_n = rptr_bin_q + {{LOG2D{1'b0}}, 1'b1};" "wire  [LOG2D:0] rptr_bin_n = {1'b0, rptr_bin_q[LOG2D-1:0]} + {{LOG2D{1'b0}}, 1'b1};"

echo "MUTANTS RESULT: $([ $rc -eq 0 ] && echo 'control PASS + 3/3 mutants RED -> PASS' || echo 'FAIL')"
exit $rc
