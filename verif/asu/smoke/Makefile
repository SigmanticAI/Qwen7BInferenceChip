# =============================================================================
# verif/asu/smoke/Makefile — ASU smoke: vectors + lint + build + run.
#
# Targets:
#   make smoke        - everything (default): tables-check, vectors, lint,
#                       then build+run all three smoke TBs
#   make tables-check - regenerate the exp LUT .svh from the golden arbiter
#                       and diff against the checked-in rtl/asu copy
#   make vectors      - generate golden smoke vectors into build/
#   make lint         - verilator --lint-only -Wall on the ASU RTL
#   make run_exp / run_softmax / run_rmsnorm - individual sims
#   make clean
#
# Waivers: lint_waivers.vlt scopes waivers to the FROZEN rtl/apex_pkg.sv only.
# No waivers on any ASU RTL file (incl. the clean-room rtl/asu/rsqrt.sv) or
# any TB — they must be -Wall clean on their own.
# =============================================================================

PYTHON ?= python3
REPO      = ../../..
include $(REPO)/verif/tools.mk
BUILD     = build

RTL_SOFTMAX = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv

RTL_RMSNORM = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/rsqrt.sv \
    $(REPO)/rtl/asu/asu_rmsnorm.sv

RTL_EXP = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv

INCS  = +incdir+$(REPO)/rtl/asu +incdir+$(REPO)/verif/common +incdir+$(BUILD)
VOPTS = -Wall --timing --assert --timescale 1ns/1ps lint_waivers.vlt $(INCS)

.PHONY: smoke tables-check vectors lint \
        compile_exp run_exp compile_softmax run_softmax \
        compile_rmsnorm run_rmsnorm clean

smoke: tables-check vectors lint run_exp run_softmax run_rmsnorm
	@echo "ASU SMOKE: all three TBs passed (see logs above)"

# ── provenance gate: checked-in LUT tables == regenerated-from-golden ────────
tables-check:
	mkdir -p $(BUILD)
	$(PYTHON) $(REPO)/rtl/asu/gen_exp_lut_tables.py --out $(BUILD)/asu_exp_lut_tables.regen.svh
	diff -q $(BUILD)/asu_exp_lut_tables.regen.svh $(REPO)/rtl/asu/asu_exp_lut_tables.svh
	@echo "TABLES CHECK: rtl/asu/asu_exp_lut_tables.svh matches the golden arbiter"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_asu_vectors.py --outdir $(BUILD)

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_softmax $(RTL_SOFTMAX)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_rmsnorm $(RTL_RMSNORM)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_exp_lut $(RTL_EXP)
	@echo "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg.sv only)"

# ── exp LUT: exhaustive-domain check ─────────────────────────────────────────
compile_exp: vectors
	mkdir -p $(BUILD)/obj_exp
	$(VERILATOR) --binary $(VOPTS) --top-module tb_asu_exp_lut \
	    -Mdir $(BUILD)/obj_exp $(RTL_EXP) tb_asu_exp_lut.sv

run_exp: compile_exp
	$(BUILD)/obj_exp/Vtb_asu_exp_lut

# ── softmax ──────────────────────────────────────────────────────────────────
compile_softmax: vectors
	mkdir -p $(BUILD)/obj_softmax
	$(VERILATOR) --binary $(VOPTS) --top-module tb_asu_softmax \
	    -Mdir $(BUILD)/obj_softmax $(RTL_SOFTMAX) tb_asu_softmax.sv

run_softmax: compile_softmax
	$(BUILD)/obj_softmax/Vtb_asu_softmax

# ── rmsnorm ──────────────────────────────────────────────────────────────────
compile_rmsnorm: vectors
	mkdir -p $(BUILD)/obj_rmsnorm
	$(VERILATOR) --binary $(VOPTS) --top-module tb_asu_rmsnorm \
	    -Mdir $(BUILD)/obj_rmsnorm $(RTL_RMSNORM) tb_asu_rmsnorm.sv

run_rmsnorm: compile_rmsnorm
	$(BUILD)/obj_rmsnorm/Vtb_asu_rmsnorm

clean:
	rm -rf $(BUILD)
