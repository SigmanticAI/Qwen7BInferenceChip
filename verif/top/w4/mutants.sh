#!/bin/bash
# mutants.sh — W4 INGEST LANE tile mutation gate.
#
# The f2sim mutants.sh discipline, kept verbatim: a clean-binary CONTROL
# must PASS first (so a fail-everything harness cannot launder a survivor);
# each mutation is applied ALONE to the real tree and reverted by an EXIT
# trap (no mutant can strand on disk); the verdict is the sim's OWN exit
# code captured via PIPESTATUS under pipefail — `| tee` cannot mask it.
#
# Three mutants, three distinct failure classes of the NEW glue:
#   M1  gs serializer taps the WRONG slot (gh_q[63:48]) — scale/gid
#       corruption -> wrong dequant -> ERO value fails
#   M2  pw_need loses the odd-tail ceil — the K=8/N=5 job starves the
#       feeder -> ERO timeout ($fatal) — starvation class
#   M3  cnt_gs counts in the wrong phase — the W4_CNTI ingest-proof
#       CSRR goes red — the beat-count evidence channel is live, not
#       decorative
set -u -o pipefail
cd "$(dirname "$0")"

GLUE="../../../rtl/top/glue/apex_w4_ingest.sv"
VEC="build/cases/w4_gemm_b64.txt"

mkdir -p build
cp "$GLUE" build/.glue_pristine.sv
trap 'cp build/.glue_pristine.sv "$GLUE"; rm -f "$GLUE.mutbak"' EXIT

run_bin () {
  "$1" +vectors="$VEC" +watchdog=8000000 2>&1 | tee build/mut_run.log \
      | tail -3
  return "${PIPESTATUS[0]}"
}

fail=0

if run_bin build/obj_w4/Vtb_apex_l3 >/dev/null; then
  echo "GATE[control]: exit=0 want=PASS got=PASS  OK"
else
  echo "GATE[control]: clean binary FAILED — gate is broken, not the RTL"
  exit 1
fi

mutate () {
  local name="$1" expr="$2" why="$3"
  cp build/.glue_pristine.sv "$GLUE"
  sed -i.mutbak -e "$expr" "$GLUE"
  if cmp -s build/.glue_pristine.sv "$GLUE"; then
    echo "GATE[$name]: mutation DID NOT APPLY ($why)"
    fail=1
    return
  fi
  if ! make build_mut >build/mut_build.log 2>&1; then
    echo "GATE[$name]: build refused want=FAIL got=FAIL  OK ($why)"
    return
  fi
  if run_bin build/obj_mut/Vtb_apex_l3 >/dev/null; then
    echo "GATE[$name]: exit=0 want=FAIL got=PASS  SURVIVOR ($why)"
    fail=1
  else
    echo "GATE[$name]: exit!=0 want=FAIL got=FAIL  OK ($why)"
  fi
}

mutate M1 "s/(gh_q\[15:0\])/(gh_q[63:48])/" "gs slot tap"
mutate M2 "s/(beats_c + 16'd1) >> 1/(beats_c) >> 1/" "pw ceil drop"
mutate M3 "s/(ph == PH_GS)) cnt_gs/(ph == PH_DRAIN)) cnt_gs/" "cnt_gs phase"

cp build/.glue_pristine.sv "$GLUE"

if [ "$fail" -eq 0 ]; then
  echo "MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS"
else
  echo "MUTANTS RESULT: FAIL"
  exit 1
fi
