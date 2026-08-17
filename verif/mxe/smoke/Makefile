# =============================================================================
# verif/mxe/smoke/Makefile — MXE smoke: lint + build + run.
#
# Targets:
#   make lint    - verilator --lint-only -Wall on the MXE RTL (mxe_top)
#   make smoke   - lint, then build the smoke TB and run it (default)
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only (contract package declares constants for all blocks). No waivers on
# any new MXE file or the TB.
# =============================================================================

REPO      = ../../..
include $(REPO)/verif/tools.mk
BUILD_DIR = build/obj_dir

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

TB_SRCS = tb_mxe_smoke.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps lint_waivers.vlt

.PHONY: smoke lint compile run clean

smoke: lint compile run

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_top $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

compile:
	mkdir -p $(BUILD_DIR)
	$(VERILATOR) --binary $(VOPTS) --top-module tb_mxe_smoke \
	    -Mdir $(BUILD_DIR) $(RTL_SRCS) $(TB_SRCS)

run:
	$(BUILD_DIR)/Vtb_mxe_smoke

clean:
	rm -rf build
