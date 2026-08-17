# =============================================================================
# verif/layer/Makefile — FULL decoder-layer COMPOSITION TB. Drives the three
# new blocks (rope, asu_silu, residual_add) with the bit-exact stage tensors of
# a real apex_golden.transformer.decoder_layer_fx decode step, in layer order,
# Verilator PRIMARY and Icarus SECONDARY. Covers the attention half (RoPE) and
# the SwiGLU FFN half (SiLU), plus both C-6 residual adds.
#
#   make layer         - everything (default): golden gate, vectors, lint,
#                        build+run under Verilator AND Icarus
#   make golden-gate   - run the golden transformer suite (the arbiter's own
#                        lemmas + 9 layer cases) as the upstream provenance gate
#   make clean
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      = ../..
NEED_IVERILOG := 1   # this suite also drives Icarus
include $(REPO)/verif/tools.mk
BUILD     = build

RTL = $(REPO)/rtl/rope/rope.sv \
      $(REPO)/rtl/asu/asu_silu.sv \
      $(REPO)/rtl/misc/residual_add.sv

# rope includes rope_lut_tables.svh (rtl/rope); asu_silu includes
# silu_lut_tables.svh (rtl/asu); the layer nvec include lives in build/.
INCS  = +incdir+$(REPO)/rtl/rope +incdir+$(REPO)/rtl/asu +incdir+$(BUILD)
VOPTS = -Wall --timing --assert --timescale 1ns/1ps $(INCS)

.PHONY: layer golden-gate vectors lint compile_vl run_vl run_iv clean

layer: golden-gate vectors lint run_vl run_iv
	@echo "LAYER COMPOSITION: passed on Verilator AND Icarus"

# ── upstream provenance: the golden arbiter's own suite must pass ────────────
golden-gate:
	$(MAKE) -C $(REPO)/golden test

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_layer_vectors.py --outdir $(BUILD)

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module rope $(REPO)/rtl/rope/rope.sv
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_silu $(REPO)/rtl/asu/asu_silu.sv
	$(VERILATOR) --lint-only $(VOPTS) --top-module residual_add $(REPO)/rtl/misc/residual_add.sv
	@echo "LINT CLEAN (-Wall, no waivers)"

compile_vl: vectors
	mkdir -p $(BUILD)/obj_layer
	$(VERILATOR) --binary $(VOPTS) --top-module tb_layer \
	    -Mdir $(BUILD)/obj_layer $(RTL) tb_layer.sv

run_vl: compile_vl
	set -o pipefail; $(BUILD)/obj_layer/Vtb_layer 2>&1 | tee $(BUILD)/run_vl.log

run_iv: vectors
	mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -Wall -o $(BUILD)/tb_layer.iv.vvp \
	    -I $(REPO)/rtl/rope -I $(REPO)/rtl/asu -I $(BUILD) \
	    -s tb_layer $(RTL) tb_layer.sv
	set -o pipefail; $(VVP) $(BUILD)/tb_layer.iv.vvp 2>&1 | tee $(BUILD)/run_iv.log

clean:
	rm -rf $(BUILD)
