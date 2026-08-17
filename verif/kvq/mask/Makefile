# =============================================================================
# verif/kvq/mask/Makefile — S12/D-027 loadable-outlier-mask suite.
#
#   make all   - lint, golden vectors, rom+csr builds, both runs (count-
#                pinned), 2 fix-break mutation checks, final gate
#                (exit != 0 on ANY failure)
#   make clean
#
# Golden arbiter: gen_mask_vectors.py imports golden/apex_golden/cq_codec —
# records/banks/fp32 from the golden packers, never re-derived. The §5 SVA
# pack ../sb/kvq_sb_sva.sv is instantiated by the TB in every build; every
# run log is grep-gated for "SVA-VIOL" (D-012).
# Waivers: verif/kvq/smoke/lint_waivers.vlt (per-line, rtl/kvq/cores only).
#
# CHECK-COUNT PINS (derived from the TB structure, then confirmed by run —
# the no-coverage-regression gate; readback = D data beats + 1 tlast = 65;
# stage 4 added the mask_valid PORT checks: +1 in R1/C1/C3/C3b):
#   rom: R1 6 + run1(12 rec + 2 bank + 12*65) = 6 + 794          =  800
#   csr: C1 6 + C2 9 + C3 7 + C3b 4 + run1 794 + C5 3
#        + C6 (8+1+8*65)=529 + C7 (3+8+1+8*65)=532 + C8 2 + C9 1 = 1887
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
KVQ_SRCS  := $(REPO)/rtl/kvq/kvq_engine.sv $(KVQ_CORES)
TB_SRCS   := tb_kvq_mask.sv ../sb/kvq_sb_sva.sv
WAIVERS   := $(REPO)/verif/kvq/smoke/lint_waivers.vlt

VOPTS = --binary --timing --assert -j 0 -Wall --timescale 1ns/1ps $(WAIVERS)

CFG_rom := -GD=64 -GTIER=2 -GG=8 -GK_OUT=2 -GDEPTH=64 -GSSETS=4 -GIS_CSR=0 \
           -GCFGNAME='"mask_rom"' -GMASKFILE='"$(BUILD)/vec/mask_m1.u8.hex"'
CFG_csr := -GD=64 -GTIER=2 -GG=8 -GK_OUT=2 -GDEPTH=64 -GSSETS=4 -GIS_CSR=1 \
           -GCFGNAME='"mask_csr"' -GMASKFILE='""'

CHK_rom := 800
CHK_csr := 1887

.PHONY: all lint vectors build_rom build_csr run_rom run_csr \
        mut_bypass mut_stuck gate clean

all: lint vectors build_rom build_csr run_rom run_csr mut_bypass mut_stuck gate

lint:
	$(VERILATOR) --lint-only -Wall --timing $(WAIVERS) \
	  --top-module kvq_engine $(KVQ_SRCS)
	@echo "LINT CLEAN (-Wall; waivers scoped per-line to rtl/kvq/cores only)"

vectors:
	mkdir -p $(BUILD)
	$(PYTHON) gen_mask_vectors.py

build_rom: vectors
	mkdir -p $(BUILD)/obj_rom
	$(VERILATOR) $(VOPTS) --top-module tb_kvq_mask -Mdir $(BUILD)/obj_rom \
	  -o sim $(CFG_rom) $(KVQ_SRCS) $(TB_SRCS)

build_csr: vectors
	mkdir -p $(BUILD)/obj_csr
	$(VERILATOR) $(VOPTS) --top-module tb_kvq_mask -Mdir $(BUILD)/obj_csr \
	  -o sim $(CFG_csr) $(KVQ_SRCS) $(TB_SRCS)

# RUN: log + tee; fail marker on nonzero exit, SVA violation, or count drift
define RUN_CFG
run_$(1): build_$(1)
	@rm -f $(BUILD)/fail_$(1)
	@set -o pipefail; \
	  ./$(BUILD)/obj_$(1)/sim +vecdir=$(BUILD)/vec 2>&1 \
	    | tee $(BUILD)/run_$(1).log \
	  || { echo "RUN FAIL [$(1)]"; touch $(BUILD)/fail_$(1); }
	@if grep -q "SVA-VIOL" $(BUILD)/run_$(1).log; \
	  then echo "SVA GATE FAIL [$(1)]"; touch $(BUILD)/fail_$(1); fi
	@if ! grep -q "checks=$(CHK_$(1)) fails=0" $(BUILD)/run_$(1).log; \
	  then echo "COUNT GATE FAIL [$(1)] (pin $(CHK_$(1)))"; \
	       touch $(BUILD)/fail_$(1); fi
endef
$(eval $(call RUN_CFG,rom))
$(eval $(call RUN_CFG,csr))

# ---- fix-break mutation checks ----------------------------------------------
# Each copies rtl/kvq into build/, breaks one D-027 commit property via
# patch_mask_mutant.py (verified applied exactly once), rebuilds the csr
# config against the copy and reruns. It MUST fail.
#   bypass: drop the popcount legality gate (illegal commits take effect)
#   stuck : commit sets ownership but never loads the staged mask
define MUT_CHECK
mut_$(1):
	@rm -rf $(BUILD)/mut_$(1) $(BUILD)/obj_mut_$(1) $(BUILD)/fail_mut_$(1); \
	mkdir -p $(BUILD)/mut_$(1)/cores
	cp $(REPO)/rtl/kvq/kvq_engine.sv $(BUILD)/mut_$(1)/
	cp $(REPO)/rtl/kvq/cores/*.sv $(BUILD)/mut_$(1)/cores/
	$(PYTHON) patch_mask_mutant.py $(1) $(BUILD)/mut_$(1)
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) $(VOPTS) --top-module tb_kvq_mask -Mdir $(BUILD)/obj_mut_$(1) \
	  -o sim $(CFG_csr) $(BUILD)/mut_$(1)/kvq_engine.sv \
	  $(patsubst $(REPO)/rtl/kvq/%,$(BUILD)/mut_$(1)/%,$(KVQ_CORES)) $(TB_SRCS)
	@set -o pipefail; \
	  if ./$(BUILD)/obj_mut_$(1)/sim +vecdir=$(BUILD)/vec 2>&1 \
	       | tee $(BUILD)/run_mut_$(1).log; \
	  then echo "MUTANT ESCAPED [$(1)]"; touch $(BUILD)/fail_mut_$(1); \
	  else echo "mutant caught [$(1)]"; fi
endef
$(eval $(call MUT_CHECK,bypass))
$(eval $(call MUT_CHECK,stuck))

gate:
	@fails=0; \
	for f in $(BUILD)/fail_rom $(BUILD)/fail_csr \
	         $(BUILD)/fail_mut_bypass $(BUILD)/fail_mut_stuck; do \
	  if [ -e $$f ]; then echo "GATE: $$f"; fails=1; fi; done; \
	if [ $$fails -ne 0 ]; then echo "MASK SUITE: FAIL"; exit 1; fi; \
	echo "MASK SUITE: ALL PASS (rom $(CHK_rom) + csr $(CHK_csr) checks, 2 mutants caught)" \
	  | tee $(BUILD)/gate.txt

clean:
	rm -rf $(BUILD)
