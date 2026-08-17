# =============================================================================
# verif/mxe/struct/Makefile — D-004 structural discriminator suite.
#
# The functional suites (smoke + sb) prove C = A·B bit-exact; they CANNOT
# distinguish a true systolic array from a broadcast MAC wall (run1/run2,
# critic 1.15). This suite closes that gap:
#
#   make run_real  - tb_mxe_struct (whitebox timing law P1..P5, incl. the
#                    single-cycle glitch hop-locality test) on the REAL
#                    rtl/mxe/mxe_array.sv + mxe_pe.sv  -> must PASS
#   make mutation  - discrimination proof on the scratch broadcast wall
#                    (mutant/mxe_array_bcast.sv):
#                      (a) tb_mxe_smoke on the mutant  -> ALL TESTS PASSED
#                          (functional suite cannot discriminate), and
#                      (b) tb_mxe_struct on the mutant -> must FAIL.
#                    Both are asserted; result recorded in build/mutation.txt.
#   make all       - lint + run_real + mutation
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any MXE file, the mutant, or the TBs.
# =============================================================================

SHELL     := /bin/bash
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

PKG_SRCS = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv

REAL_ARR = \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv

MUT_ARR  = mutant/mxe_array.sv

# full MXE RTL around the array (for the mutant functional-equivalence run)
MXE_REST = \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps lint_waivers.vlt

.PHONY: all lint build_real run_real build_mut_struct build_mut_smoke mutation clean

all: lint run_real mutation

# lint at mxe_top scope (as the sb suite does — mxe_cfg_pkg params are all
# used there), once with the real array and once with the mutant substituted
lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_top \
	    $(PKG_SRCS) $(REAL_ARR) $(MXE_REST)
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_top \
	    $(PKG_SRCS) $(MUT_ARR) $(MXE_REST)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

build_real:
	mkdir -p $(BUILD)/obj_real
	$(VERILATOR) --binary $(VOPTS) +define+GLITCH_TEST \
	    --top-module tb_mxe_struct -Mdir $(BUILD)/obj_real \
	    $(PKG_SRCS) $(REAL_ARR) tb_mxe_struct.sv

run_real: build_real
	set -o pipefail; $(BUILD)/obj_real/Vtb_mxe_struct \
	    | tee $(BUILD)/run_real.log
	grep -q "STRUCT PASS" $(BUILD)/run_real.log

build_mut_struct:
	mkdir -p $(BUILD)/obj_mut_struct
	$(VERILATOR) --binary $(VOPTS) \
	    --top-module tb_mxe_struct -Mdir $(BUILD)/obj_mut_struct \
	    $(PKG_SRCS) $(MUT_ARR) tb_mxe_struct.sv

build_mut_smoke:
	mkdir -p $(BUILD)/obj_mut_smoke
	$(VERILATOR) --binary $(VOPTS) \
	    --top-module tb_mxe_smoke -Mdir $(BUILD)/obj_mut_smoke \
	    $(PKG_SRCS) $(MUT_ARR) $(MXE_REST) ../smoke/tb_mxe_smoke.sv

# discrimination proof: the broadcast wall PASSES the functional smoke but
# FAILS the structural TB — i.e. THIS suite (and only this suite) tests D-004
mutation: build_mut_struct build_mut_smoke
	@rm -f $(BUILD)/mutation.txt
	set -o pipefail; $(BUILD)/obj_mut_smoke/Vtb_mxe_smoke \
	    | tee $(BUILD)/run_mut_smoke.log
	grep -q "ALL TESTS PASSED" $(BUILD)/run_mut_smoke.log
	@echo "mutant(bcast wall) vs functional smoke: PASSED (functional suite cannot discriminate)" >> $(BUILD)/mutation.txt
	@set -o pipefail; \
	if $(BUILD)/obj_mut_struct/Vtb_mxe_struct > $(BUILD)/run_mut_struct.log 2>&1; then \
	    echo "MUTATION ESCAPE: broadcast wall PASSED the structural TB" | tee -a $(BUILD)/mutation.txt; \
	    exit 1; \
	fi
	grep -q "STRUCT FAIL" $(BUILD)/run_mut_struct.log
	@grep -m1 "checks failed" $(BUILD)/run_mut_struct.log >> $(BUILD)/mutation.txt || \
	    tail -1 $(BUILD)/run_mut_struct.log >> $(BUILD)/mutation.txt
	@echo "MUTATION CHECK PASS: broadcast-wall mutant passes tb_mxe_smoke but is KILLED by tb_mxe_struct (D-004 discriminates)" | tee -a $(BUILD)/mutation.txt

clean:
	rm -rf $(BUILD)
