# =============================================================================
# verif/top/smoke/Makefile — apex_top v0.1 tile smoke suite:
#   1. lint  : verilator --lint-only -Wall on the FULL tile (waivers scoped to
#              frozen apex_pkg + vendored rsqrt/kvq-vendor files only)
#   2. vectors: golden-driven command script from gen_top_vectors.py
#              (golden arbiter: golden/apex_golden/attention.py attention_fx)
#   3. build : verilator --binary --timing --assert (SVA compiled = D-012 gate)
#              THREE tiles: D=64 (default), D=128 (-GCFG_D=128, F-5 closure)
#              and D=64 CQ-4+ (-GOUTK=2 + mask ROM, F-2/D-024 closure)
#   4. run   : one full directed attention micro-case PER TILE (T=8),
#              phase-by-phase through the real tile streams, every mid-chain
#              tensor + the final o8 bit-exact vs the golden run. The D=128
#              run is a kept-passing F-5 regression (nb=BPR=16 stage jobs +
#              PAT_D blocks 8..15); the cq4p run is a kept-passing F-2
#              regression (grouped keys tuser=0 + WRITE_ADDR group base +
#              D-008 CSR FLUSH + fp16 outlier-mask lane, all through the
#              D-024 tier bank).
#
#   make smoke  - the whole gate (lint + vectors + build + run, both tiles)
#   make waves  - rerun D=64 with +dump (FST trace)
#   make clean
#
# NOTE: pipefail is load-bearing on every gate line — without it `tee` masks
# a failing gate's exit status and `make smoke` lies green (the tee lesson).
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv \
    $(REPO)/rtl/asu/rsqrt.sv \
    $(REPO)/rtl/asu/asu_rmsnorm.sv \
    $(REPO)/rtl/rope/rope_pair_fx.sv \
    $(REPO)/rtl/rope/rope_row.sv \
    $(REPO)/rtl/top/glue/apex_layer_deq.sv \
    $(REPO)/rtl/top/glue/apex_residual.sv \
    $(REPO)/rtl/asu/asu_swiglu.sv \
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
    $(REPO)/rtl/top/glue/apex_kvq_bank.sv \
    $(REPO)/rtl/top/glue/apex_kvq_gqa_bank.sv \
    $(REPO)/rtl/top/glue/apex_wcomp_bank.sv \
    $(REPO)/rtl/tip/tip_decide.sv \
    $(REPO)/rtl/tip/tip_importance.sv \
    $(REPO)/rtl/tip/tip_top.sv \
    $(REPO)/rtl/seq/seq_walker_pkg.sv \
    $(REPO)/rtl/seq/seq_walker.sv \
    $(REPO)/rtl/seq/seq_walker_comp.sv \
    $(REPO)/rtl/seq/seq_layer_walker.sv \
    $(REPO)/rtl/seq/seq_layer_walker2.sv \
    $(REPO)/rtl/csr/csr_regs.sv \
    $(REPO)/rtl/seam/seam_feeder_quant.sv \
    $(REPO)/rtl/seam/seam_score_dequant.sv \
    $(REPO)/rtl/top/glue/apex_q78_to_fp32.sv \
    $(REPO)/rtl/top/glue/apex_lane32_ser.sv \
    $(REPO)/rtl/top/glue/apex_lane8_unpack.sv \
    $(REPO)/rtl/top/glue/apex_gam_unpack.sv \
    $(REPO)/rtl/top/glue/apex_stage_buf.sv \
    $(REPO)/rtl/top/glue/apex_score_fork.sv \
    $(REPO)/rtl/top/glue/apex_proj_bias.sv \
    $(REPO)/rtl/top/glue/apex_scale_quant.sv \
    $(REPO)/rtl/top/apex_top.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/rope \
        -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. lint_waivers.vlt

.PHONY: smoke lint vectors build run waves clean

smoke: lint vectors build run

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module apex_top $(RTL)
	@echo "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg.sv + clean-room kvq cores only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_top_vectors.py $(BUILD) | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_top $(BUILD)/obj_d128 $(BUILD)/obj_cq4p
	$(VERILATOR) --binary $(VOPTS) --trace-fst \
	    --top-module tb_apex_smoke -Mdir $(BUILD)/obj_top \
	    $(RTL) tb_apex_smoke.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GCFG_D=128 \
	    --top-module tb_apex_smoke -Mdir $(BUILD)/obj_d128 \
	    $(RTL) tb_apex_smoke.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GOUTK=2 \
	    -GMASKF='"$(BUILD)/mask_smoke_cq4p.hex"' \
	    --top-module tb_apex_smoke -Mdir $(BUILD)/obj_cq4p \
	    $(RTL) tb_apex_smoke.sv

run:
	set -o pipefail; $(BUILD)/obj_top/Vtb_apex_smoke \
	    +vectors=$(BUILD)/vectors_smoke.txt | tee $(BUILD)/run_smoke.log
	set -o pipefail; $(BUILD)/obj_d128/Vtb_apex_smoke \
	    +vectors=$(BUILD)/vectors_smoke_d128.txt \
	    | tee $(BUILD)/run_smoke_d128.log
	set -o pipefail; $(BUILD)/obj_cq4p/Vtb_apex_smoke \
	    +vectors=$(BUILD)/vectors_smoke_cq4p.txt \
	    | tee $(BUILD)/run_smoke_cq4p.log

waves:
	set -o pipefail; $(BUILD)/obj_top/Vtb_apex_smoke \
	    +vectors=$(BUILD)/vectors_smoke.txt +dump | tee $(BUILD)/run_waves.log

clean:
	rm -rf $(BUILD) dump.fst
