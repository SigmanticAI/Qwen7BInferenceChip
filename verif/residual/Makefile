# =============================================================================
# verif/residual/Makefile — residual_add smoke: the C-6 fp16 residual primitive
# y = f16(a+b), bit-exact vs the golden numpy fp16 add, Verilator PRIMARY and
# Icarus SECONDARY. residual_add is standalone (no apex_pkg), so -Wall clean
# with NO waivers.
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      = ../..
NEED_IVERILOG := 1   # this suite also drives Icarus
include $(REPO)/verif/tools.mk
BUILD     = build

RTL_RES = $(REPO)/rtl/misc/residual_add.sv
VOPTS   = -Wall --timing --assert --timescale 1ns/1ps +incdir+$(BUILD)

.PHONY: residual vectors lint run_vl run_iv clean

residual: vectors lint run_vl run_iv
	@echo "RESIDUAL_ADD SMOKE: passed on Verilator AND Icarus"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_residual_vectors.py --outdir $(BUILD)

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module residual_add $(RTL_RES)
	@echo "LINT CLEAN (-Wall, no waivers)"

compile_vl: vectors
	mkdir -p $(BUILD)/obj_res
	$(VERILATOR) --binary $(VOPTS) --top-module tb_residual_add \
	    -Mdir $(BUILD)/obj_res $(RTL_RES) tb_residual_add.sv

run_vl: compile_vl
	set -o pipefail; $(BUILD)/obj_res/Vtb_residual_add 2>&1 | tee $(BUILD)/run_vl.log

run_iv: vectors
	mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -Wall -o $(BUILD)/tb_res.iv.vvp \
	    -I $(BUILD) -s tb_residual_add $(RTL_RES) tb_residual_add.sv
	set -o pipefail; $(VVP) $(BUILD)/tb_res.iv.vvp 2>&1 | tee $(BUILD)/run_iv.log

clean:
	rm -rf $(BUILD)
