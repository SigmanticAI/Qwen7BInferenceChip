# =============================================================================
# verif/top/l2/Makefile — Layer-2 pairwise integration chains (§8 Layer 2):
#   chain a: mxe_top -> seam_score_dequant -> asu_softmax   (tb_l2a_chain)
#   chain b: kvq_engine -> seam_feeder_quant -> mxe_top     (tb_l2b_chain,
#            ALL 3 tiers incl. CQ-4+ outlier masks + D-008 partial FLUSH;
#            the named golden cases run at CQ-4+ here because the committed
#            apex_top is CQ-8-only — see ../l3)
#   chain c: seam_score_dequant -> apex_score_fork -> {asu_softmax, tip_top}
#
# Every chain: golden-generated vectors, SVA compiled (D-012), backpressure
# + stall storms, mid-operation reset, coverage-gated. pipefail is
# load-bearing on every gate line (the tee lesson).
#
#   make all       - lint + vectors + build + every run regime + coverage
#   make clean
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_A = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv \
    $(REPO)/rtl/seam/seam_score_dequant.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv

RTL_B = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
    $(REPO)/rtl/kvq/cores/cq_scale_unit.sv \
    $(REPO)/rtl/kvq/cores/cq_quant_unit.sv \
    $(REPO)/rtl/kvq/cores/cq_dequant_unit.sv \
    $(REPO)/rtl/kvq/cores/scale_bank.sv \
    $(REPO)/rtl/kvq/cores/scale_bank_store.sv \
    $(REPO)/rtl/kvq/cores/cq_rne_div_pipe.sv \
    $(REPO)/rtl/kvq/cores/cq_quant_pipe.sv \
    $(REPO)/rtl/kvq/cores/cq_scale_pipe.sv \
    $(REPO)/rtl/kvq/cores/residual_buffer.sv \
    $(REPO)/rtl/kvq/cores/sram_controller.sv \
    $(REPO)/rtl/kvq/cores/cq_value_path.sv \
    $(REPO)/rtl/kvq/cores/cq_key_path.sv \
    $(REPO)/rtl/kvq/kvq_engine.sv \
    $(REPO)/rtl/seam/seam_feeder_quant.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv

RTL_C = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/seam/seam_score_dequant.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv \
    $(REPO)/rtl/tip/tip_decide.sv \
    $(REPO)/rtl/tip/tip_importance.sv \
    $(REPO)/rtl/tip/tip_top.sv \
    $(REPO)/rtl/top/glue/apex_score_fork.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. lint_waivers.vlt

# kvq_engine builds: one per synthesis config (TIER/G/OUTLIER_K/MASK_FILE
# are parameters — the same fact that pins apex_top v0.1 to CQ-8)
B_CFGS = t0 t1 t2 adv_T1 adv_out1000 calib_T128 calib_T70 calib_d128
GB_t0          = -GTIER_CFG=0 -GG_CFG=16  -GOUTK_CFG=0 -GDEPTH_CFG=512
GB_t1          = -GTIER_CFG=1 -GG_CFG=16  -GOUTK_CFG=0 -GDEPTH_CFG=512
GB_t2          = -GTIER_CFG=2 -GG_CFG=16  -GOUTK_CFG=2 -GDEPTH_CFG=512 \
                 -GMASKF='"$(BUILD)/mask_t2.hex"'
GB_adv_T1      = -GTIER_CFG=2 -GG_CFG=128 -GOUTK_CFG=2 -GDEPTH_CFG=512 \
                 -GMASKF='"$(BUILD)/mask_adv_T1.hex"'
GB_adv_out1000 = -GTIER_CFG=2 -GG_CFG=128 -GOUTK_CFG=2 -GDEPTH_CFG=512 \
                 -GMASKF='"$(BUILD)/mask_adv_out1000.hex"'
GB_calib_T128  = -GTIER_CFG=2 -GG_CFG=64  -GOUTK_CFG=2 -GDEPTH_CFG=512 \
                 -GMASKF='"$(BUILD)/mask_calib_T128.hex"'
GB_calib_T70   = -GTIER_CFG=2 -GG_CFG=64  -GOUTK_CFG=2 -GDEPTH_CFG=512 \
                 -GMASKF='"$(BUILD)/mask_calib_T70.hex"'
GB_calib_d128  = -GD_CFG=128 -GTIER_CFG=2 -GG_CFG=128 -GOUTK_CFG=2 \
                 -GDEPTH_CFG=512 -GMASKF='"$(BUILD)/mask_calib_d128.hex"'

RUN_A = set -o pipefail; $(BUILD)/obj_l2a/Vtb_l2a_chain
RUN_C = set -o pipefail; $(BUILD)/obj_l2c/Vtb_l2c_chain

.PHONY: all lint vectors build run_a run_b run_c coverage clean

all: lint vectors build run_a run_b run_c coverage

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module tb_l2a_chain $(RTL_A) tb_l2a_chain.sv
	$(VERILATOR) --lint-only $(VOPTS) --top-module tb_l2b_chain $(RTL_B) tb_l2b_chain.sv
	$(VERILATOR) --lint-only $(VOPTS) --top-module tb_l2c_chain $(RTL_C) tb_l2c_chain.sv
	@echo "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg + clean-room kvq cores only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_l2a_vectors.py $(BUILD) | tee $(BUILD)/gen_l2a.log
	set -o pipefail; $(PYTHON) gen_l2b_vectors.py $(BUILD) | tee $(BUILD)/gen_l2b.log
	set -o pipefail; $(PYTHON) gen_l2c_vectors.py $(BUILD) | tee $(BUILD)/gen_l2c.log

# NOTE: a shell `for cfg` loop with $(GB_$$cfg) silently expands the per-
# config parameter list to EMPTY (make cannot see the shell variable) — all
# engines were built at default TIER=0. $(call) expands at make time.
define BUILD_B
	mkdir -p $(BUILD)/obj_l2b_$(1)
	$(VERILATOR) --binary $(VOPTS) --trace-fst $(GB_$(1)) \
	    --top-module tb_l2b_chain -Mdir $(BUILD)/obj_l2b_$(1) \
	    $(RTL_B) tb_l2b_chain.sv

endef

build:
	mkdir -p $(BUILD)/obj_l2a $(BUILD)/obj_l2c
	$(VERILATOR) --binary $(VOPTS) --trace-fst \
	    --top-module tb_l2a_chain -Mdir $(BUILD)/obj_l2a $(RTL_A) tb_l2a_chain.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst \
	    --top-module tb_l2c_chain -Mdir $(BUILD)/obj_l2c $(RTL_C) tb_l2c_chain.sv
	$(foreach cfg,$(B_CFGS),$(call BUILD_B,$(cfg)))

run_a:
	$(RUN_A) +vectors=$(BUILD)/vectors_l2a_directed.txt +bp_mode=1 +stall_mode=1 \
	    +seed=201 | tee $(BUILD)/run_l2a_directed.log
	$(RUN_A) +vectors=$(BUILD)/vectors_l2a_directed.txt +bp_mode=0 +stall_mode=0 \
	    +seed=202 | tee $(BUILD)/run_l2a_fast.log
	$(RUN_A) +vectors=$(BUILD)/vectors_l2a_random.txt +bp_mode=1 +stall_mode=1 \
	    +seed=203 | tee $(BUILD)/run_l2a_random.log
	$(RUN_A) +vectors=$(BUILD)/vectors_l2a_random.txt +bp_mode=2 +stall_mode=2 \
	    +seed=204 +watchdog=20000000 | tee $(BUILD)/run_l2a_storm.log
	$(RUN_A) +vectors=$(BUILD)/vectors_l2a_reset.txt +bp_mode=1 +stall_mode=0 \
	    +seed=205 | tee $(BUILD)/run_l2a_reset.log

run_b:
	set -o pipefail; $(BUILD)/obj_l2b_t0/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t0.txt +bp_mode=1 +stall_mode=1 \
	    +seed=211 | tee $(BUILD)/run_l2b_t0.log
	set -o pipefail; $(BUILD)/obj_l2b_t0/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t0.txt +bp_mode=2 +stall_mode=2 \
	    +seed=212 +watchdog=60000000 | tee $(BUILD)/run_l2b_t0_storm.log
	set -o pipefail; $(BUILD)/obj_l2b_t1/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t1.txt +bp_mode=1 +stall_mode=1 \
	    +seed=213 | tee $(BUILD)/run_l2b_t1.log
	set -o pipefail; $(BUILD)/obj_l2b_t1/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t1.txt +bp_mode=2 +stall_mode=2 \
	    +seed=214 +watchdog=60000000 | tee $(BUILD)/run_l2b_t1_storm.log
	set -o pipefail; $(BUILD)/obj_l2b_t1/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t1_reset.txt +bp_mode=1 +stall_mode=0 \
	    +seed=215 | tee $(BUILD)/run_l2b_t1_reset.log
	set -o pipefail; $(BUILD)/obj_l2b_t2/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_t2.txt +bp_mode=1 +stall_mode=1 \
	    +seed=216 | tee $(BUILD)/run_l2b_t2.log
	set -o pipefail; $(BUILD)/obj_l2b_adv_T1/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_adv_T1.txt +bp_mode=1 +stall_mode=1 \
	    +seed=217 | tee $(BUILD)/run_l2b_adv_T1.log
	set -o pipefail; $(BUILD)/obj_l2b_adv_out1000/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_adv_out1000.txt +bp_mode=1 +stall_mode=0 \
	    +seed=218 +watchdog=120000000 | tee $(BUILD)/run_l2b_adv_out1000.log
	set -o pipefail; $(BUILD)/obj_l2b_calib_T128/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_calib_T128.txt +bp_mode=1 +stall_mode=0 \
	    +seed=219 +watchdog=120000000 | tee $(BUILD)/run_l2b_calib_T128.log
	set -o pipefail; $(BUILD)/obj_l2b_calib_T70/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_calib_T70.txt +bp_mode=1 +stall_mode=0 \
	    +seed=220 +watchdog=120000000 | tee $(BUILD)/run_l2b_calib_T70.log
	set -o pipefail; $(BUILD)/obj_l2b_calib_d128/Vtb_l2b_chain \
	    +vectors=$(BUILD)/vectors_l2b_calib_d128.txt +bp_mode=1 +stall_mode=0 \
	    +seed=221 +watchdog=200000000 | tee $(BUILD)/run_l2b_calib_d128.log

run_c:
	$(RUN_C) +vectors=$(BUILD)/vectors_l2c_directed.txt +bp_mode=1 +stall_mode=1 \
	    +seed=231 | tee $(BUILD)/run_l2c_directed.log
	$(RUN_C) +vectors=$(BUILD)/vectors_l2c_directed.txt +bp_mode=0 +stall_mode=0 \
	    +seed=232 | tee $(BUILD)/run_l2c_fast.log
	$(RUN_C) +vectors=$(BUILD)/vectors_l2c_random.txt +bp_mode=2 +stall_mode=2 \
	    +seed=233 +watchdog=20000000 | tee $(BUILD)/run_l2c_storm.log
	$(RUN_C) +vectors=$(BUILD)/vectors_l2c_random.txt +bp_mode=1 +stall_mode=1 \
	    +seed=234 | tee $(BUILD)/run_l2c_random.log
	$(RUN_C) +vectors=$(BUILD)/vectors_l2c_reset.txt +bp_mode=1 +stall_mode=0 \
	    +seed=235 | tee $(BUILD)/run_l2c_reset.log

coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD) | tee $(BUILD)/coverage.txt

clean:
	rm -rf $(BUILD) dump.fst
