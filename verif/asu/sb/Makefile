# =============================================================================
# verif/asu/sb/Makefile — ASU independent verification suite:
#   vector-driven scoreboard TBs (house pattern, verif/mxe/sb lineage; golden
#   arbiter apex_golden) + bound apex_stream1_sva §5 pack + job-level D-006/§3
#   assertions + adversarial protocol runs (backpressure storms, stalled
#   feeds, gamma-delivery adversaries, malformed framing) + phase-targeted
#   mid-operation resets + manual coverage buckets + checker mutation gate.
#
# Targets:
#   make all       - lint, vectors, build (3 configs), 13 run configurations,
#                    coverage aggregation gate, mutation gate
#   make lint      - verilator --lint-only -Wall on the ASU RTL
#   make vectors   - regenerate stimulus/expected from the golden arbiter
#   make mutants   - prove the checkers catch injected RTL bugs (scratch
#                    copies only; rtl/ is never modified)
#   make waves_sm / waves_rms - rerun directed with +dump (FST)
#   make clean
#
# Waivers: lint_waivers.vlt scopes waivers to the FROZEN rtl/apex_pkg.sv only.
# No waivers on ASU RTL (incl. clean-room rtl/asu/rsqrt.sv), SVA, or TBs.
# =============================================================================

.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_SM = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv

RTL_RMS = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/rsqrt.sv \
    $(REPO)/rtl/asu/asu_rmsnorm.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps lint_waivers.vlt \
        -I$(REPO)/verif/common +incdir+$(REPO)/rtl/asu

RUN_SM     = set -o pipefail; $(BUILD)/obj_sm/Vtb_asu_softmax_sb
RUN_SM_F10 = set -o pipefail; $(BUILD)/obj_sm_f10/Vtb_asu_softmax_sb
RUN_RMS    = set -o pipefail; $(BUILD)/obj_rms/Vtb_asu_rmsnorm_sb

.PHONY: all lint vectors build_sm build_sm_f10 build_rms \
        run_sm_directed run_sm_fast run_sm_random run_sm_storm run_sm_stall \
        run_sm_f10 run_sm_reset \
        run_rms_directed run_rms_fast run_rms_random run_rms_storm \
        run_rms_geager run_rms_reset \
        coverage mutants waves_sm waves_rms clean

all: lint vectors build_sm build_sm_f10 build_rms \
     run_sm_directed run_sm_fast run_sm_random run_sm_storm run_sm_stall \
     run_sm_f10 run_sm_reset \
     run_rms_directed run_rms_fast run_rms_random run_rms_storm \
     run_rms_geager run_rms_reset \
     coverage mutants
	@echo "ASU SB: full independent suite passed (scoreboards, SVA, resets, coverage gate, mutation gate)"

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_softmax $(RTL_SM)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_rmsnorm $(RTL_RMS)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_exp_lut \
	    $(REPO)/rtl/apex_pkg.sv $(REPO)/rtl/asu/asu_exp_lut.sv
	@echo "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_asu_sb_vectors.py --outdir $(BUILD) | tee $(BUILD)/gen.log

build_sm:
	mkdir -p $(BUILD)/obj_sm
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_asu_softmax_sb \
	    -Mdir $(BUILD)/obj_sm $(RTL_SM) tb_asu_softmax_sb.sv

build_sm_f10:
	mkdir -p $(BUILD)/obj_sm_f10
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_asu_softmax_sb \
	    -GTB_SCORE_FRAC=10 \
	    -Mdir $(BUILD)/obj_sm_f10 $(RTL_SM) tb_asu_softmax_sb.sv

build_rms:
	mkdir -p $(BUILD)/obj_rms
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_asu_rmsnorm_sb \
	    -Mdir $(BUILD)/obj_rms $(RTL_RMS) tb_asu_rmsnorm_sb.sv

# ── softmax runs ─────────────────────────────────────────────────────────────
run_sm_directed:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=11 | tee $(BUILD)/run_sm_directed.log

run_sm_fast:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_directed.txt +bp_mode=0 \
	    +stall_mode=0 +seed=12 | tee $(BUILD)/run_sm_fast.log

run_sm_random:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_random.txt +bp_mode=1 \
	    +stall_mode=1 +seed=13 | tee $(BUILD)/run_sm_random.log

# backpressure STORM on the probability stream + bursty feed
run_sm_storm:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_directed.txt +bp_mode=2 \
	    +stall_mode=2 +seed=14 | tee $(BUILD)/run_sm_storm.log

# stalled feed with NO output backpressure
run_sm_stall:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_random.txt +bp_mode=0 \
	    +stall_mode=2 +seed=15 | tee $(BUILD)/run_sm_stall.log

# SCORE_FRAC=10 configuration: LUT interpolation inside the softmax datapath
run_sm_f10:
	$(RUN_SM_F10) +vectors=$(BUILD)/vectors_sm_f10.txt +bp_mode=1 \
	    +stall_mode=1 +seed=17 | tee $(BUILD)/run_sm_f10.log

# phase-targeted mid-operation resets (full-rate feed for phase landing)
run_sm_reset:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_reset.txt +bp_mode=1 \
	    +stall_mode=0 +seed=16 | tee $(BUILD)/run_sm_reset.log

# ── rmsnorm runs ─────────────────────────────────────────────────────────────
run_rms_directed:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_directed.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=0 +seed=21 | tee $(BUILD)/run_rms_directed.log

run_rms_fast:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_directed.txt +bp_mode=0 \
	    +stall_mode=0 +g_mode=0 +seed=22 | tee $(BUILD)/run_rms_fast.log

# gamma LATE adversary: emission repeatedly starved on the gamma stream
run_rms_random:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_random.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=1 +seed=23 | tee $(BUILD)/run_rms_random.log

# backpressure storm + bursty x feed + eager gamma
run_rms_storm:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_directed.txt +bp_mode=2 \
	    +stall_mode=2 +g_mode=2 +seed=24 | tee $(BUILD)/run_rms_storm.log

# gamma EAGER adversary on the random set (g-skid parked through rsqrt wait)
run_rms_geager:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_random.txt +bp_mode=1 \
	    +stall_mode=0 +g_mode=2 +seed=25 | tee $(BUILD)/run_rms_geager.log

run_rms_reset:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_reset.txt +bp_mode=1 \
	    +stall_mode=0 +g_mode=0 +seed=26 | tee $(BUILD)/run_rms_reset.log

# ── gates ────────────────────────────────────────────────────────────────────
coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log | tee $(BUILD)/coverage.txt

mutants:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutants.txt

waves_sm:
	$(RUN_SM) +vectors=$(BUILD)/vectors_sm_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=11 +dump | tee $(BUILD)/run_sm_waves.log

waves_rms:
	$(RUN_RMS) +vectors=$(BUILD)/vectors_rms_directed.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=0 +seed=21 +dump | tee $(BUILD)/run_rms_waves.log

clean:
	rm -rf $(BUILD) dump_sm.fst dump_rms.fst
