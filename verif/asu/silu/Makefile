# =============================================================================
# verif/asu/silu/Makefile — asu_silu smoke: tables-check + vectors + lint +
# build + run, Verilator PRIMARY and Icarus SECONDARY, both exhaustive over the
# full 16-bit input domain vs the golden arbiter apex_golden.transformer.silu_fx.
#
# Targets:
#   make silu         - everything (default): tables-check, vectors, lint,
#                       then build+run under Verilator AND Icarus
#   make tables-check - regenerate silu_lut_tables.svh from the golden arbiter
#                       and diff against the checked-in rtl/asu copy (D-013)
#   make vectors      - generate the exhaustive silu oracle into build/
#   make lint         - verilator --lint-only -Wall on asu_silu
#   make run_vl       - Verilator primary sim
#   make run_iv       - Icarus secondary sim
#   make clean
#
# asu_silu is standalone (no apex_pkg dependency), so there are NO lint waivers:
# the RTL and TB must be -Wall clean on their own.
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      = ../../..
NEED_IVERILOG := 1   # this suite also drives Icarus
include $(REPO)/verif/tools.mk
BUILD     = build

RTL_SILU = $(REPO)/rtl/asu/asu_silu.sv
INCS     = +incdir+$(REPO)/rtl/asu +incdir+$(BUILD)
VOPTS    = -Wall --timing --assert --timescale 1ns/1ps $(INCS)

.PHONY: silu tables-check vectors lint compile_vl run_vl run_iv clean

silu: tables-check vectors lint run_vl run_iv
	@echo "ASU SILU: exhaustive-domain check passed on Verilator AND Icarus"

# ── provenance gate: checked-in LUT tables == regenerated-from-golden ────────
tables-check:
	mkdir -p $(BUILD)
	$(PYTHON) $(REPO)/golden/gen_silu_lut_tables.py --out $(BUILD)/silu_lut_tables.regen.svh
	diff -q $(BUILD)/silu_lut_tables.regen.svh $(REPO)/rtl/asu/silu_lut_tables.svh
	@echo "TABLES CHECK: rtl/asu/silu_lut_tables.svh matches the golden arbiter"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_silu_vectors.py --outdir $(BUILD)

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_silu $(RTL_SILU)
	@echo "LINT CLEAN (-Wall, no waivers)"

# ── Verilator primary: exhaustive sweep ──────────────────────────────────────
compile_vl: vectors
	mkdir -p $(BUILD)/obj_silu
	$(VERILATOR) --binary $(VOPTS) --top-module tb_asu_silu \
	    -Mdir $(BUILD)/obj_silu $(RTL_SILU) tb_asu_silu.sv

run_vl: compile_vl
	set -o pipefail; $(BUILD)/obj_silu/Vtb_asu_silu 2>&1 | tee $(BUILD)/run_vl.log

# ── Icarus secondary: same TB + oracle, independent simulator ────────────────
run_iv: vectors
	mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -Wall -o $(BUILD)/tb_asu_silu.iv.vvp \
	    -I $(REPO)/rtl/asu -I $(BUILD) \
	    -s tb_asu_silu $(RTL_SILU) tb_asu_silu.sv
	set -o pipefail; $(VVP) $(BUILD)/tb_asu_silu.iv.vvp 2>&1 | tee $(BUILD)/run_iv.log

clean:
	rm -rf $(BUILD)
