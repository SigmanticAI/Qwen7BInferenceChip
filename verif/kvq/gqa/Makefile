# =============================================================================
# verif/kvq/gqa/Makefile — IB-LAYER S4b per-KV-head GQA bank suite
# (apex_kvq_gqa_bank: N_ENG=4 CQ-8 kvq_engine instances behind eng_sel).
#
#   make all   - bank lint + apex_top KVQ_GQA_NENG=4 elaboration lint (the
#                parameter-ON proof, warnings FATAL) + golden vectors +
#                build + count-pinned run (SVA grep-gated) + 3 signature-
#                required mutants (l3/mutate.py discipline — never
#                any-nonzero-exit) + final gate (exit != 0 on ANY failure)
#   make clean
#
# Golden arbiter: gen_gqa_vectors.py imports golden/apex_golden/cq_codec —
# CQ-8 value records from the golden primitives, never re-derived. The §5
# stream SVA pack (verif/kvq/smoke/kvq_axis_sva.sv, `bind kvq_engine`) rides
# into all N_ENG engine instances; run logs are grep-gated for "[SVA".
# Waivers: verif/kvq/smoke/lint_waivers.vlt (vendored engine internals ONLY
# — apex_kvq_gqa_bank.sv itself is -Wall clean, no waivers).
#
# MACHINE RULE (LEVEL_C §9.1, all lanes): at most ONE Verilator build at a
# time across lanes — every build target waits on the hardened pgrep.
#
# CHECK-COUNT PIN (derived from the TB structure, then confirmed by run):
#   B1a 4 eng x 4 INFO            =   16
#   B1b 4 CTRL readbacks          =    4
#   B1c 4 OCC                     =    4
#   B2b 22 rec x (64 data + last) = 1430
#   B3  rewrite 65 + OCC 1 + 3 holds x 65 = 261
#   B4  8 directed + recover 65   =   75
#   B5  2 directed + stalled read 65 =  67
#   B6  2                         =    2
#                            total = 1859
# =============================================================================
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

KVQ_CORES := \
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
$(REPO)/rtl/kvq/cores/cq_key_path.sv
ENG_SRCS  := $(REPO)/rtl/kvq/kvq_engine.sv $(KVQ_CORES)
BANK      := $(REPO)/rtl/top/glue/apex_kvq_gqa_bank.sv
TB_SRCS   := tb_kvq_gqa.sv $(REPO)/verif/kvq/smoke/kvq_axis_sva.sv
WAIVERS   := $(REPO)/verif/kvq/smoke/lint_waivers.vlt

# full tile list for the KVQ_GQA_NENG=4 elaboration lint (l3 list + the bank)
TILE_RTL = \
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
    $(KVQ_CORES) \
    $(REPO)/rtl/kvq/kvq_engine.sv \
    $(REPO)/rtl/top/glue/apex_kvq_bank.sv \
    $(BANK) \
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

VOPTS = --binary --timing --assert -j 0 -Wall --timescale 1ns/1ps $(WAIVERS)

CFG_gqa4 := -GD=64 -GN_ENG=4 -GDEPTH=256 -GG=16 -GSSETS=4 \
            -GCFGNAME='"gqa4"'
CHK_gqa4 := 1859

.PHONY: all lint lint_top_on vectors build run \
        mut_mS1 mut_mS2 mut_mS3 gate clean

all: lint lint_top_on vectors build run mut_mS1 mut_mS2 mut_mS3 gate

lint:
	$(VERILATOR) --lint-only -Wall --timing $(WAIVERS) \
	  --top-module apex_kvq_gqa_bank $(BANK) $(ENG_SRCS)
	@echo "LINT CLEAN (bank -Wall; waivers scoped to vendored engine internals only)"

# the S4b elaboration-gate proof: apex_top ELABORATES at KVQ_GQA_NENG=4
# (GQA branch live, walker-side + l_kv_map consumers wired), warnings FATAL.
# Default-parameter (OFF) lint stays owned by verif/top/{smoke,l3}.
lint_top_on:
	$(VERILATOR) --lint-only -Wall --timing \
	  -I$(REPO)/rtl/rope -I$(REPO)/rtl/asu -I$(REPO)/verif/common \
	  $(REPO)/verif/top/smoke/lint_waivers.vlt \
	  -GKVQ_GQA_NENG=4 -GKVQ_DEPTH=256 \
	  --top-module apex_top $(TILE_RTL)
	@echo "LINT CLEAN (apex_top @ KVQ_GQA_NENG=4/KVQ_DEPTH=256 — parameter-ON elaboration)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_gqa_vectors.py | tee $(BUILD)/gen.log

build: vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 30; done
	mkdir -p $(BUILD)/obj_gqa4
	$(VERILATOR) $(VOPTS) --top-module tb_kvq_gqa -Mdir $(BUILD)/obj_gqa4 \
	  -o sim $(CFG_gqa4) $(BANK) $(ENG_SRCS) $(TB_SRCS)

run: build
	@rm -f $(BUILD)/fail_gqa4
	@set -o pipefail; \
	  ./$(BUILD)/obj_gqa4/sim +vecdir=$(BUILD)/vec 2>&1 \
	    | tee $(BUILD)/run_gqa4.log \
	  || { echo "RUN FAIL [gqa4]"; touch $(BUILD)/fail_gqa4; }
	@if grep -q "\[SVA" $(BUILD)/run_gqa4.log; \
	  then echo "SVA GATE FAIL [gqa4]"; touch $(BUILD)/fail_gqa4; fi
	@if ! grep -q "checks=$(CHK_gqa4) fails=0" $(BUILD)/run_gqa4.log; \
	  then echo "COUNT GATE FAIL [gqa4] (pin $(CHK_gqa4))"; \
	       touch $(BUILD)/fail_gqa4; fi

# ── mutation gate: patched COPIES of the bank (rtl/ never touched); each
#    mutant must fail WITH ITS SIGNATURE (mutate_gqa.py check) ───────────────
define RUN_MUT
mut_$(1):
	@rm -rf $(BUILD)/mut_$(1) $(BUILD)/obj_mut_$(1); rm -f $(BUILD)/fail_mut_$(1)
	$(PYTHON) mutate_gqa.py patch $(1) $(BUILD)/mut_$(1)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 30; done
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) $(VOPTS) --top-module tb_kvq_gqa -Mdir $(BUILD)/obj_mut_$(1) \
	  -o sim $(CFG_gqa4) $(BUILD)/mut_$(1)/apex_kvq_gqa_bank.sv \
	  $(ENG_SRCS) $(TB_SRCS)
	-./$(BUILD)/obj_mut_$(1)/sim +vecdir=$(BUILD)/vec +watchdog=200000 \
	  > $(BUILD)/run_mut_$(1).log 2>&1
	@set -o pipefail; $(PYTHON) mutate_gqa.py check $(1) \
	  $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt \
	  || touch $(BUILD)/fail_mut_$(1)
endef
$(eval $(call RUN_MUT,mS1))
$(eval $(call RUN_MUT,mS2))
$(eval $(call RUN_MUT,mS3))

gate:
	@fails=0; \
	for f in $(BUILD)/fail_gqa4 $(BUILD)/fail_mut_mS1 \
	         $(BUILD)/fail_mut_mS2 $(BUILD)/fail_mut_mS3; do \
	  if [ -e $$f ]; then echo "GATE: $$f"; fails=1; fi; done; \
	if [ $$fails -ne 0 ]; then echo "GQA SUITE: FAIL"; exit 1; fi; \
	echo "GQA SUITE: ALL PASS (gqa4 $(CHK_gqa4) checks, 3/3 signature mutants caught)" \
	  | tee $(BUILD)/gate.txt

clean:
	rm -rf $(BUILD)
