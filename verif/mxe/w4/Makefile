# =============================================================================
# verif/mxe/w4/Makefile — B3 native-W4 weight feeder verification suite:
#   vector-driven scoreboard TB (V0/seam pattern, golden arbiter
#   apex_golden.weight_codec: wfeed_w4_to_i8 realization "A") + bound
#   w4_job_sva / apex_stream1_sva §5/D-006 property packs + adversarial
#   protocol runs + mid-operation reset runs + manual coverage buckets +
#   mutation gate + the PERF beats-halved assertion.
#
# Targets:
#   make all       - lint, generate vectors, build, run all regimes,
#                    aggregate coverage (gates on required-bucket closure),
#                    run the mutation gate
#   make lint      - verilator --lint-only -Wall on the feeder
#   make vectors   - regenerate stimulus/expected from the golden model
#   make mutate    - mutation gate only (requires vectors + build)
#   make waves     - rerun directed with +dump (FST trace)
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on rtl/mxe/mxe_wfeed_w4.sv, the SVA pack, or the TB.
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_wfeed_w4.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/verif/common -I. lint_waivers.vlt

RUN = set -o pipefail; $(BUILD)/obj_w4/Vtb_wfeed_w4_sb

.PHONY: all lint vectors build run coverage mutate waves clean

all: lint vectors build run coverage mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_wfeed_w4 $(RTL)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_w4_vectors.py $(BUILD) | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_w4
	$(VERILATOR) --binary $(VOPTS) --trace-fst \
	    --top-module tb_wfeed_w4_sb -Mdir $(BUILD)/obj_w4 \
	    $(RTL) tb_wfeed_w4_sb.sv

# directed under random bp+gaps, directed full-rate, the big random set,
# backpressure storm, stalled feeds, mid-op resets
run:
	$(RUN) +vectors=$(BUILD)/vectors_w4_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=201 | tee $(BUILD)/run_directed.log
	$(RUN) +vectors=$(BUILD)/vectors_w4_directed.txt +bp_mode=0 \
	    +stall_mode=0 +seed=202 | tee $(BUILD)/run_fast.log
	$(RUN) +vectors=$(BUILD)/vectors_w4_random.txt +bp_mode=1 \
	    +stall_mode=1 +seed=203 +watchdog=30000000 \
	    | tee $(BUILD)/run_random.log
	$(RUN) +vectors=$(BUILD)/vectors_w4_storm.txt +bp_mode=2 \
	    +stall_mode=2 +seed=204 | tee $(BUILD)/run_storm.log
	$(RUN) +vectors=$(BUILD)/vectors_w4_storm.txt +bp_mode=0 \
	    +stall_mode=2 +seed=205 | tee $(BUILD)/run_stall.log
	$(RUN) +vectors=$(BUILD)/vectors_w4_reset.txt +bp_mode=1 \
	    +stall_mode=0 +seed=206 | tee $(BUILD)/run_reset.log

# NOTE: pipefail is load-bearing on every gate line — without it `tee`
# masks a failing gate's exit status and `make all` lies green.
coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log \
	    | tee $(BUILD)/coverage.txt

mutate:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutation.txt

waves:
	$(RUN) +vectors=$(BUILD)/vectors_w4_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=201 +dump | tee $(BUILD)/run_waves.log

clean:
	rm -rf $(BUILD) dump.fst
