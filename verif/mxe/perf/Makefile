# =============================================================================
# verif/mxe/perf/Makefile — D-005 load-under-compute perf evidence.
#
#   make all - lint; build tb_mxe_perf twice:
#                obj_new  = rtl/mxe/mxe_ctrl.sv (load-under-compute ctrl)
#                obj_base = baseline/mxe_ctrl.sv (PINNED copy of the
#                           sequential load->compute->flush controller —
#                           the pre-D-005-closure scheduler, kept verbatim)
#              then:
#                run_new         (+require_overlap=1)  -> must PASS
#                run_base                              -> measures baseline
#                run_base_reqov  (+require_overlap=1)  -> must FAIL: the
#                    permanent kept-failing demonstration that the
#                    sequential scheduler has zero overlap (regression-first
#                    evidence for the D-005 gap)
#                compare: perf_compare.py gates overlap>0 + cycle win and
#                    prints the speedup table (build/perf_gate.txt)
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any MXE file or the TB.
# =============================================================================

.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_COMMON = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv

CTRL_NEW  = $(REPO)/rtl/mxe/mxe_ctrl.sv
CTRL_BASE = baseline/mxe_ctrl.sv
TOP_SRC   = $(REPO)/rtl/mxe/mxe_top.sv

TB_SRCS = tb_mxe_perf.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/verif/common lint_waivers.vlt

.PHONY: all lint build_new run_new build_base run_base run_base_reqov \
        compare clean

all: lint run_new run_base run_base_reqov compare

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module mxe_top \
	    $(RTL_COMMON) $(CTRL_NEW) $(TOP_SRC)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

build_new:
	mkdir -p $(BUILD)/obj_new
	$(VERILATOR) --binary $(VOPTS) --top-module tb_mxe_perf \
	    -Mdir $(BUILD)/obj_new $(RTL_COMMON) $(CTRL_NEW) $(TOP_SRC) $(TB_SRCS)

run_new: build_new
	set -o pipefail; $(BUILD)/obj_new/Vtb_mxe_perf +require_overlap=1 \
	    | tee $(BUILD)/run_new.log
	grep -q "PERF TB PASS" $(BUILD)/run_new.log

build_base:
	mkdir -p $(BUILD)/obj_base
	$(VERILATOR) --binary $(VOPTS) --top-module tb_mxe_perf \
	    -Mdir $(BUILD)/obj_base $(RTL_COMMON) $(CTRL_BASE) $(TOP_SRC) $(TB_SRCS)

run_base: build_base
	set -o pipefail; $(BUILD)/obj_base/Vtb_mxe_perf \
	    | tee $(BUILD)/run_base.log
	grep -q "PERF TB PASS" $(BUILD)/run_base.log

# permanent kept-failing regression: the SEQUENTIAL scheduler must keep
# failing the overlap requirement (if this ever "passes", the baseline pin
# is broken and the D-005 evidence is void)
run_base_reqov: build_base
	@set -o pipefail; \
	if $(BUILD)/obj_base/Vtb_mxe_perf +require_overlap=1 \
	    > $(BUILD)/run_base_reqov.log 2>&1; then \
	    echo "D-005 REGRESSION BROKEN: sequential baseline satisfied +require_overlap"; \
	    exit 1; \
	fi
	grep -q "PERF OVERLAP MISSING" $(BUILD)/run_base_reqov.log
	@echo "[CONFIRMED-GAP] sequential baseline ctrl: PERF OVERLAP MISSING on every multi-chunk job (kept-failing, D-005 register)"

compare:
	set -o pipefail; $(PYTHON) perf_compare.py \
	    $(BUILD)/run_new.log $(BUILD)/run_base.log | tee $(BUILD)/perf_gate.txt

clean:
	rm -rf $(BUILD)
