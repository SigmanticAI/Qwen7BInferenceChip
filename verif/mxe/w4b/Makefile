# =============================================================================
# verif/mxe/w4b — W4-B feeder suite (D-031, docs/design/W4B_FEEDER.md):
# golden-arbiter vectors (wfeed_w4b_to_i8) + scoreboard TB (6 regimes) +
# package gate + the REQUIRED exhaustive operand sweep (§8.4) + mutation gate.
#
#   make all      - vectors, pkg gate, build, 6 runs, sweep, mutants
#   make sweep    - 4.06M-point exhaustive fp16 operand sweep
#   make mutants  - 5-mutant checker-credibility gate (scratch copies only)
# =============================================================================
SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL = $(REPO)/rtl/apex_pkg.sv $(REPO)/rtl/xbr/stream_skid.sv \
      $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
      $(REPO)/rtl/kvq/cores/cq_rne_div_pipe.sv \
      $(REPO)/rtl/mxe/w4b_fp_pkg.sv $(REPO)/rtl/mxe/mxe_wfeed_w4b.sv

VOPTS = -Wno-fatal --timing --timescale 1ns/1ps -CFLAGS -std=gnu++17

RUN = set -o pipefail; $(BUILD)/obj_sb/Vtb_w4b_sb

.PHONY: all logged vectors pkg build sweep mutants \
        run_directed run_fast run_random run_storm run_gslate run_reset clean

all: vectors pkg build run_directed run_fast run_random run_storm \
     run_gslate run_reset sweep mutants
	@echo "W4B SUITE: all gates green (pkg, 6 regimes, exhaustive sweep, mutants)"

# COMPLETE committed-evidence log. `make all` only tails each gate to console;
# build/*.log hold the full output. `logged` runs everything then assembles the
# untruncated per-gate logs into logs/run_all_full.log, so RESULT.md's pasted
# banners are genuinely verbatim from a committed log (audit batch #2 lesson:
# the old logs/run_all_2026-07-24.log was tail-stripped to 4 of 10 lines).
logged:
	rm -rf $(BUILD)
	set -o pipefail; $(MAKE) all 2>&1 | tee $(BUILD)/console.log
	mkdir -p logs
	@{ echo "# W4B full suite log — complete per-gate output (NOT tail-truncated)."; \
	   echo "# Reproduce: make -C verif/mxe/w4b logged"; echo ; \
	   for f in gen pkg run_directed run_fast run_random run_storm run_gslate run_reset sweep ; do \
	     echo "===== build/$$f.log ====="; cat $(BUILD)/$$f.log 2>/dev/null; echo ; done ; \
	   echo "===== build/mutants.txt ====="; cat $(BUILD)/mutants.txt 2>/dev/null; echo ; \
	   grep -E "W4B SUITE:" $(BUILD)/console.log ; } > logs/run_all_full.log
	@echo "assembled logs/run_all_full.log"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_w4b_vectors.py | tee $(BUILD)/gen.log
	$(PYTHON) gen_pkg_vectors.py
	$(PYTHON) gen_sweep.py

pkg:
	$(VERILATOR) --binary $(VOPTS) $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
	    $(REPO)/rtl/mxe/w4b_fp_pkg.sv tb_w4b_pkg.sv \
	    --top-module tb_w4b_pkg -Mdir $(BUILD)/obj_pkg > /dev/null
	set -o pipefail; $(BUILD)/obj_pkg/Vtb_w4b_pkg | tee $(BUILD)/pkg.log | tail -1

build:
	$(VERILATOR) --binary $(VOPTS) $(RTL) tb_w4b_sb.sv \
	    --top-module tb_w4b_sb -Mdir $(BUILD)/obj_sb > /dev/null

run_directed:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_directed.txt +bp_mode=1 \
	    +stall_mode=1 +gs_mode=0 +seed=51 | tee $(BUILD)/run_directed.log | tail -2
run_fast:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_directed.txt +bp_mode=0 \
	    +stall_mode=0 +gs_mode=0 +seed=52 | tee $(BUILD)/run_fast.log | tail -2
run_random:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_random.txt +bp_mode=1 \
	    +stall_mode=1 +gs_mode=1 +seed=53 | tee $(BUILD)/run_random.log | tail -2
run_storm:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_random.txt +bp_mode=2 \
	    +stall_mode=2 +gs_mode=2 +seed=54 | tee $(BUILD)/run_storm.log | tail -2
run_gslate:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_random.txt +bp_mode=1 \
	    +stall_mode=0 +gs_mode=1 +seed=55 | tee $(BUILD)/run_gslate.log | tail -2
run_reset:
	$(RUN) +vectors=$(BUILD)/vectors_w4b_reset.txt +bp_mode=1 \
	    +stall_mode=1 +gs_mode=0 +seed=56 | tee $(BUILD)/run_reset.log | tail -2

sweep:
	$(VERILATOR) --binary $(VOPTS) $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
	    $(REPO)/rtl/mxe/w4b_fp_pkg.sv tb_w4b_sweep.sv \
	    --top-module tb_w4b_sweep -Mdir $(BUILD)/obj_sweep > /dev/null
	set -o pipefail; $(BUILD)/obj_sweep/Vtb_w4b_sweep | tee $(BUILD)/sweep.log | tail -1

mutants:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutants.txt

clean:
	rm -rf $(BUILD)
