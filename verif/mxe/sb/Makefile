# =============================================================================
# verif/mxe/sb/Makefile — MXE independent verification suite:
#   vector-driven scoreboard TB (V0 pattern, golden arbiter apex_golden) +
#   bound apex_stream_sva §5/D-006 property pack + adversarial protocol runs +
#   mid-operation reset runs + manual coverage buckets.
#
# Targets:
#   make all       - lint, generate vectors, build, run all 6 configurations,
#                    aggregate coverage (gates on required-bucket closure)
#   make lint      - verilator --lint-only -Wall on the MXE RTL (mxe_top)
#   make vectors   - regenerate stimulus/expected from the golden model
#   make waves     - rerun the directed config with +dump (FST trace)
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any new MXE file, the SVA pack, or the TB.
# =============================================================================

# GATE INTEGRITY: stock macOS make is GNU 3.81, which PREDATES .SHELLFLAGS
# (added in 3.82) and silently ignores it — so .SHELLFLAGS alone does NOT
# make 'cmd | tee log' gate on cmd. Every gating recipe that pipes through
# tee therefore ALSO sets pipefail inline (see RUN and the coverage recipe).
# .SHELLFLAGS is kept for makes that honor it.
.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build
OBJ       := $(BUILD)/obj_dir

# The six run logs the coverage gate aggregates — the EXPLICIT list, never a
# run_*.log glob: a glob also sweeps in foreign logs (e.g. run_waves.log from
# 'make waves') whose stale COV lines could fill a required bucket and make a
# failed closure look green.
RUN_LOGS = $(BUILD)/run_directed.log $(BUILD)/run_directed_fast.log \
           $(BUILD)/run_random.log $(BUILD)/run_storm.log \
           $(BUILD)/run_stall.log $(BUILD)/run_reset.log

RTL_SRCS = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv

TB_SRCS = tb_mxe_sb.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/verif/common lint_waivers.vlt

RUN = set -o pipefail; $(OBJ)/Vtb_mxe_sb

.PHONY: all lint vectors build \
        run_directed run_directed_fast run_random run_storm run_stall \
        run_reset coverage waves clean

all: lint vectors build run_directed run_directed_fast run_random \
     run_storm run_stall run_reset coverage

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_top $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_mxe_vectors.py $(BUILD)

build:
	mkdir -p $(OBJ)
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_mxe_sb \
	    -Mdir $(OBJ) $(RTL_SRCS) $(TB_SRCS)

# directed cases under random backpressure + random feed gaps
run_directed:
	$(RUN) +vectors=$(BUILD)/vectors_directed.txt +bp_mode=1 +stall_mode=1 \
	    +seed=1 | tee $(BUILD)/run_directed.log

# same directed cases at full rate (no backpressure, no gaps)
run_directed_fast:
	$(RUN) +vectors=$(BUILD)/vectors_directed.txt +bp_mode=0 +stall_mode=0 \
	    +seed=2 | tee $(BUILD)/run_directed_fast.log

# >=500 seeded random jobs + interleaved illegal descriptors
run_random:
	$(RUN) +vectors=$(BUILD)/vectors_random.txt +bp_mode=1 +stall_mode=1 \
	    +seed=3 | tee $(BUILD)/run_random.log

# backpressure STORM on the result stream (the V0.2 lesson) + bursty feeds
run_storm:
	$(RUN) +vectors=$(BUILD)/vectors_storm.txt +bp_mode=2 +stall_mode=2 \
	    +seed=4 | tee $(BUILD)/run_storm.log

# stalled weight/activation feeds with NO result backpressure
run_stall:
	$(RUN) +vectors=$(BUILD)/vectors_storm.txt +bp_mode=0 +stall_mode=2 \
	    +seed=5 | tee $(BUILD)/run_stall.log

# mid-operation reset (the run2 hole): full-rate feeds so the generator's
# abort-cycle phase targeting lands; TB records the actual phase hit
run_reset:
	$(RUN) +vectors=$(BUILD)/vectors_reset.txt +bp_mode=1 +stall_mode=0 \
	    +seed=6 | tee $(BUILD)/run_reset.log

# This suite has NO mutation target — coverage closure is its ONLY aggregate
# gate, so it must actually bite:
#   * the logs are prerequisites: 'make coverage' refuses to run before all
#     six configurations have produced their logs (no rule builds them here;
#     a missing log is a hard make error, not a silently thinner aggregate);
#   * inline 'set -o pipefail' because GNU make 3.81 (stock macOS) ignores
#     .SHELLFLAGS — without it the recipe's status is tee's, and a failing
#     coverage_report.py (exit 1 on any required hole) would still pass.
coverage: $(RUN_LOGS)
	set -o pipefail; \
	$(PYTHON) coverage_report.py $(RUN_LOGS) | tee $(BUILD)/coverage.txt

waves:
	$(RUN) +vectors=$(BUILD)/vectors_directed.txt +bp_mode=1 +stall_mode=1 \
	    +seed=1 +dump | tee $(BUILD)/run_waves.log

clean:
	rm -rf $(BUILD) dump.fst
