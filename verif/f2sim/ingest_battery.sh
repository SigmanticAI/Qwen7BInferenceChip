#!/bin/bash
# perf/ingest-lane battery — final tree. Sequential, pipefail-safe, one
# verdict line per gate + BATTERY RESULT at the end. Verilator builds wait
# for the machine to be free (one-build-at-a-time rule).
set -u -o pipefail
cd /Users/nabilabdelazizferhattaleb/Desktop/apex-ingest/verif/f2sim
LOGD=/Users/nabilabdelazizferhattaleb/Desktop/apex-ingest/build/battery
mkdir -p "$LOGD"
rc_all=0
say() { echo "BATTERY[$1]: $2"; }
wait_vl() { until ! pgrep -f verilator_bin >/dev/null; do sleep 10; done; }

# 1. behsmoke — model directed AXI smoke, both knob sets (801 checks each)
wait_vl
if make behsmoke > "$LOGD/behsmoke.log" 2>&1 \
   && [ "$(grep -c 'DDRBEH RESULT: checks=801 fails=0 -> PASS' "$LOGD/behsmoke.log")" -eq 2 ]; then
  say behsmoke "PASS (2x 801/0)"
else say behsmoke "FAIL"; rc_all=1; fi

# 2. D=128 DDR=1 twin (fresh Mdir)
wait_vl
rm -rf obj_d128_ddr1
if make build D=128 DDR=1 > "$LOGD/build_d128_ddr1.log" 2>&1 && [ -x obj_d128_ddr1/f2sim ]; then
  say build-ddr1 "OK"
else say build-ddr1 "FAIL"; rc_all=1; fi

# 3. 18-job HOST-mode replay div8 — counts must hold; cycle-invisibility is
# adjudicated against the SAME-branch pre-change reference run (apex-ref),
# not a cross-branch doc constant (the July-22 432469308 predates this
# branch's E-7/B1 RTL evolution)
./obj_d128_ddr1/f2sim +tile_div=8 ../../build/f2_regops/job_*.regops.jsonl \
  > "$LOGD/host_d128_div8.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q 'F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS' "$LOGD/host_d128_div8.log"; then
  cycl=$(grep -o 'cyc [0-9]*' "$LOGD/host_d128_div8.log" | tail -1)
  say host-div8 "PASS counts (27996/0, final $cycl; cycle parity vs apex-ref adjudicated separately)"
else say host-div8 "FAIL rc=$rc"; rc_all=1; fi

# 4. 18-job FUEL-mode replay at the three CDC ratios (afifo write-side
# timing changed by F1/F2 — the three-ratio discipline applies)
for DIV in 8 7 2; do
  ./obj_d128_ddr1/f2sim +tile_div=$DIV \
    +ddr_image=../../build/ddr_image/ddr_image.bin \
    +ddr_regions=../../build/ddr_image/ddr_image.regions.jsonl +fuel_audit \
    ../../build/fuel_regops/job_*.fuel.regops.jsonl \
    > "$LOGD/fuel_d128_div$DIV.log" 2>&1
  rc=$?
  audits=$(grep -c 'FUELAUDIT .* -> ok' "$LOGD/fuel_d128_div$DIV.log" || true)
  if [ $rc -eq 0 ] && grep -q 'F2SIM RESULT: files=18 checks=27996 fails=0 -> PASS' "$LOGD/fuel_d128_div$DIV.log" \
     && [ "$audits" -eq 18 ]; then
    cycl=$(grep -o 'cyc [0-9]*' "$LOGD/fuel_d128_div$DIV.log" | tail -1)
    say fuel-div$DIV "PASS (27996/0, 18/18 audits, final $cycl)"
  else say fuel-div$DIV "FAIL rc=$rc audits=$audits"; rc_all=1; fi
done

# 5. mutants — control PASS + 3/3 RED (rebuilds obj_mut per mutant)
wait_vl
make mutants AWS_FPGA=../../tools/aws-fpga > "$LOGD/mutants.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q 'MUTANTS RESULT: control PASS + 3/3 mutants RED -> PASS' "$LOGD/mutants.log"; then
  say mutants "PASS (control + 3/3 RED)"
else say mutants "FAIL rc=$rc"; rc_all=1; fi

wait_vl
rm -rf obj_d128_ddr0
if make build D=128 DDR=0 > "$LOGD/build_d128_ddr0.log" 2>&1 && [ -x obj_d128_ddr0/f2sim ]; then
  say build-ddr0 "OK"
else say build-ddr0 "FAIL"; rc_all=1; fi
make capgate > "$LOGD/capgate.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say capgate "PASS"; else say capgate "FAIL rc=$rc"; rc_all=1; fi

( cd ../.. && python3 scripts/fpga/f2/walk_fuel_layer.py selftest > "$LOGD/walk_selftest.log" 2>&1 )
rc=$?
if [ $rc -eq 0 ] && grep -q 'ALL PASS' "$LOGD/walk_selftest.log"; then
  say walk-selftest "PASS"
else say walk-selftest "FAIL rc=$rc"; rc_all=1; fi

echo "BATTERY RESULT: $([ $rc_all -eq 0 ] && echo PASS || echo FAIL)"
exit $rc_all
