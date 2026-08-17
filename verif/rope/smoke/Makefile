# =============================================================================
# verif/rope/smoke/Makefile — rope smoke: tables-check + vectors + lint +
# build + run, Verilator PRIMARY and Icarus SECONDARY, both bit-exact over
# swept positions 0..127 and head_dim in {64,128} vs the golden arbiter
# apex_golden.transformer.rope_fx.
#
# Targets:
#   make rope         - everything (default): tables-check, vectors, lint,
#                       then build+run under Verilator AND Icarus
#   make tables-check - regenerate rope_lut_tables.svh from the golden arbiter
#                       and diff against the checked-in rtl/rope copy (D-013)
#   make vectors      - generate the rope oracle + Icarus ROM hex into build/
#   make lint         - verilator --lint-only -Wall on rope
#   make run_vl       - Verilator primary sim
#   make run_iv       - Icarus secondary sim
#   make clean
#
# rope is standalone (no apex_pkg dependency; its own localparams + the
# generated cos/sin table include), so there are NO lint waivers: the RTL and
# TB must be -Wall clean on their own.
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      = ../../..
NEED_IVERILOG := 1   # this suite also drives Icarus
include $(REPO)/verif/tools.mk
BUILD     = build

RTL_ROPE = $(REPO)/rtl/rope/rope.sv
INCS     = +incdir+$(REPO)/rtl/rope +incdir+$(BUILD)
VOPTS    = -Wall --timing --assert --timescale 1ns/1ps $(INCS)

.PHONY: rope tables-check vectors lint compile_vl run_vl run_iv clean

rope: tables-check vectors lint run_vl run_iv
	@echo "ROPE SMOKE: swept-domain check passed on Verilator AND Icarus"

# ── provenance gate: checked-in LUT tables == regenerated-from-golden ────────
tables-check:
	mkdir -p $(BUILD)
	$(PYTHON) $(REPO)/golden/gen_rope_lut_tables.py --out $(BUILD)/rope_lut_tables.regen.svh
	diff -q $(BUILD)/rope_lut_tables.regen.svh $(REPO)/rtl/rope/rope_lut_tables.svh
	@echo "TABLES CHECK: rtl/rope/rope_lut_tables.svh matches the golden arbiter"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_rope_vectors.py --outdir $(BUILD)

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module rope $(RTL_ROPE)
	@echo "LINT CLEAN (-Wall, no waivers)"

# ── Verilator primary: swept sweep ───────────────────────────────────────────
compile_vl: vectors
	mkdir -p $(BUILD)/obj_rope
	$(VERILATOR) --binary $(VOPTS) --top-module tb_rope \
	    -Mdir $(BUILD)/obj_rope $(RTL_ROPE) tb_rope.sv

run_vl: compile_vl
	set -o pipefail; $(BUILD)/obj_rope/Vtb_rope 2>&1 | tee $(BUILD)/run_vl.log

# ── Icarus secondary: same TB + oracle, independent simulator ────────────────
run_iv: vectors
	mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -Wall -o $(BUILD)/tb_rope.iv.vvp \
	    -I $(REPO)/rtl/rope -I $(BUILD) \
	    -s tb_rope $(RTL_ROPE) tb_rope.sv
	set -o pipefail; $(VVP) $(BUILD)/tb_rope.iv.vvp 2>&1 | tee $(BUILD)/run_iv.log

clean:
	rm -rf $(BUILD)
