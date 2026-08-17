# =============================================================================
# verif/seam/Makefile — D-021 seam independent verification suite:
#   vector-driven scoreboard TBs (V0 pattern, golden arbiter apex_golden:
#   quant_rows_i8 / score_dequant_fx of attention.py) + bound seam_job_sva /
#   apex_stream1_sva §5/D-006 property packs + adversarial protocol runs +
#   mid-operation reset runs + manual coverage buckets + mutation gate.
#
# Targets:
#   make all       - lint, generate vectors (incl. the >=1M-operand S-3
#                    equivalence set), build 3 configs (feeder D=64/D=128,
#                    score dequant), run all regimes, aggregate coverage
#                    (gates on required-bucket closure), run the mutation gate
#   make lint      - verilator --lint-only -Wall on both seam blocks
#   make vectors   - regenerate stimulus/expected from the golden model
#   make mutate    - mutation gate only (requires vectors + builds)
#   make waves_fq / waves_sd - rerun directed with +dump (FST trace)
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any new seam file, the SVA packs, or the TBs.
# =============================================================================

SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FQ = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/seam/seam_feeder_quant.sv

RTL_SD = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/seam/seam_score_dequant.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/verif/common -I. lint_waivers.vlt

RUN_FQ64  = set -o pipefail; $(BUILD)/obj_fq64/Vtb_seam_feeder_sb
RUN_FQ128 = set -o pipefail; $(BUILD)/obj_fq128/Vtb_seam_feeder_sb
RUN_SD    = set -o pipefail; $(BUILD)/obj_sd/Vtb_seam_score_sb

.PHONY: all lint vectors build \
        run_fq64 run_fq128 run_sd coverage mutate waves_fq waves_sd clean

all: lint vectors build run_fq64 run_fq128 run_sd coverage mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module seam_feeder_quant $(RTL_FQ)
	$(VERILATOR) --lint-only $(VOPTS) --top-module seam_score_dequant $(RTL_SD)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_seam_vectors.py $(BUILD) | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_fq64 $(BUILD)/obj_fq128 $(BUILD)/obj_sd
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GD_CFG=64 \
	    --top-module tb_seam_feeder_sb -Mdir $(BUILD)/obj_fq64 \
	    $(RTL_FQ) tb_seam_feeder_sb.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GD_CFG=128 \
	    --top-module tb_seam_feeder_sb -Mdir $(BUILD)/obj_fq128 \
	    $(RTL_FQ) tb_seam_feeder_sb.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst \
	    --top-module tb_seam_score_sb -Mdir $(BUILD)/obj_sd \
	    $(RTL_SD) tb_seam_score_sb.sv

# feeder D=64: directed under random bp+gaps, directed full-rate, the big
# random set, backpressure storm, stalled feeds, mid-op resets
run_fq64:
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_directed_d64.txt +bp_mode=1 \
	    +stall_mode=1 +seed=101 | tee $(BUILD)/run_fq64_directed.log
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_directed_d64.txt +bp_mode=0 \
	    +stall_mode=0 +seed=102 | tee $(BUILD)/run_fq64_fast.log
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_random_d64.txt +bp_mode=1 \
	    +stall_mode=1 +seed=103 +watchdog=30000000 \
	    | tee $(BUILD)/run_fq64_random.log
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_storm_d64.txt +bp_mode=2 \
	    +stall_mode=2 +seed=104 | tee $(BUILD)/run_fq64_storm.log
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_storm_d64.txt +bp_mode=0 \
	    +stall_mode=2 +seed=105 | tee $(BUILD)/run_fq64_stall.log
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_reset_d64.txt +bp_mode=1 \
	    +stall_mode=0 +seed=106 | tee $(BUILD)/run_fq64_reset.log

run_fq128:
	$(RUN_FQ128) +vectors=$(BUILD)/vectors_fq_directed_d128.txt +bp_mode=1 \
	    +stall_mode=1 +seed=111 | tee $(BUILD)/run_fq128_directed.log
	$(RUN_FQ128) +vectors=$(BUILD)/vectors_fq_directed_d128.txt +bp_mode=0 \
	    +stall_mode=0 +seed=112 | tee $(BUILD)/run_fq128_fast.log
	$(RUN_FQ128) +vectors=$(BUILD)/vectors_fq_random_d128.txt +bp_mode=1 \
	    +stall_mode=1 +seed=113 +watchdog=30000000 \
	    | tee $(BUILD)/run_fq128_random.log
	$(RUN_FQ128) +vectors=$(BUILD)/vectors_fq_storm_d128.txt +bp_mode=2 \
	    +stall_mode=2 +seed=114 | tee $(BUILD)/run_fq128_storm.log
	$(RUN_FQ128) +vectors=$(BUILD)/vectors_fq_reset_d128.txt +bp_mode=1 \
	    +stall_mode=0 +seed=115 | tee $(BUILD)/run_fq128_reset.log

# score dequant: the random run IS the >=1M-operand S-3 equivalence proof
run_sd:
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=121 | tee $(BUILD)/run_sd_directed.log
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_directed.txt +bp_mode=0 \
	    +stall_mode=0 +seed=122 | tee $(BUILD)/run_sd_fast.log
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_random.txt +bp_mode=1 \
	    +stall_mode=1 +seed=123 +watchdog=60000000 \
	    | tee $(BUILD)/run_sd_random.log
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_storm.txt +bp_mode=2 \
	    +stall_mode=2 +seed=124 | tee $(BUILD)/run_sd_storm.log
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_storm.txt +bp_mode=0 \
	    +stall_mode=2 +seed=125 | tee $(BUILD)/run_sd_stall.log
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_reset.txt +bp_mode=1 \
	    +stall_mode=0 +seed=126 | tee $(BUILD)/run_sd_reset.log

# NOTE: pipefail is load-bearing on every gate line — without it `tee`
# masks a failing gate's exit status and `make all` lies green.
coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log \
	    | tee $(BUILD)/coverage.txt

mutate:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutation.txt

waves_fq:
	$(RUN_FQ64) +vectors=$(BUILD)/vectors_fq_directed_d64.txt +bp_mode=1 \
	    +stall_mode=1 +seed=101 +dump | tee $(BUILD)/run_fq64_waves.log

waves_sd:
	$(RUN_SD) +vectors=$(BUILD)/vectors_sd_directed.txt +bp_mode=1 \
	    +stall_mode=1 +seed=121 +dump | tee $(BUILD)/run_sd_waves.log

clean:
	rm -rf $(BUILD) dump.fst
