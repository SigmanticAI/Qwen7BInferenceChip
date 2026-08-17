# =============================================================================
# verif/top/wcomp/Makefile — F6(i) per-KV-head composite scale-cache BANK
# suite (apex_wcomp_bank: N_ENG seq_walker_comp instances behind eng_sel).
#
#   make all   - bank lint + golden vectors + both geometry builds +
#                count-pinned runs (D=64 and D=128 — the GQA tile is a
#                head_dim=128 build) + 4 signature-required mutants
#                (mutate_wcomp.py discipline — never any-nonzero-exit) +
#                final gate (exit != 0 on ANY failure)
#   make clean
#
# SCOPE: this suite proves ROUTING and ISOLATION, not composite arithmetic —
# that is verif/seq_walker's comp sweep (30,720 vectors per geometry, both
# builds, mutant-gated), and it is not duplicated here. Expected words still
# come from the ONE arbiter: verif/top/l3/walker_composite_golden.py through
# gen_wcomp_vectors.py, with every scale entering via the REAL snoop path
# (golden/apex_golden/cq_codec). The apex_top ELABORATION proof at
# KVQ_GQA_NENG=4 lives in verif/kvq/gqa (lint_top_on), which compiles this
# bank as part of the tile.
#
# The P1 precondition SVA (verif/seq_walker/walker_comp_sva.svh, bound into
# seq_walker_comp) rides into ALL N_ENG instances; a violation prints
# %Error and fails the run.
# Waivers: verif/seq_walker/lint_waivers.vlt (PACKAGE-only UNUSEDPARAM — no
# waiver on apex_wcomp_bank.sv itself, which is -Wall clean).
#
# MACHINE RULE (LEVEL_C §9.1, all lanes): at most ONE Verilator build at a
# time across lanes — every build target waits on the hardened pgrep.
#
# CHECK-COUNT PIN (derived from the TB structure, then confirmed by run;
# identical at both geometries — routing is D-independent by construction):
#   A-store  err-pulse quiet                         =   2
#   A-check  32 rows x (cs resp + cs val + qs resp + qs val) = 128
#   B        32 rows x (cs resp + cs val)            =  64
#   S        refusal + OR-reduce + writer resp/cs/qs =   5
#   F        no err_frame + resp + cs + qs           =   4
#                                             total  = 203
# =============================================================================
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

N_ENG     := 4

RTL_SRCS  := \
  $(REPO)/rtl/apex_pkg.sv \
  $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
  $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
  $(REPO)/rtl/seq/seq_walker_pkg.sv \
  $(REPO)/rtl/seq/seq_walker_comp.sv
BANK      := $(REPO)/rtl/top/glue/apex_wcomp_bank.sv
TB_SRCS   := tb_wcomp_bank.sv
WAIVERS   := $(REPO)/verif/seq_walker/lint_waivers.vlt

VOPTS = --binary --timing --assert -Wall --timescale 1ns/1ps \
        -I$(REPO)/verif/seq_walker $(WAIVERS)

CHK := 203

.PHONY: all lint vectors build build64 build128 run run64 run128 mutlog \
        mut_mW1 mut_mW2 mut_mW3 mut_mW4 gate clean

all: lint vectors build run mutlog mut_mW1 mut_mW2 mut_mW3 mut_mW4 gate

# the mutation ledger is append-only (tee -a, so a partial re-run adds to
# it); `all` resets it first so build/mutation.txt always reads as exactly
# one suite pass — the l3/l4 discipline
mutlog:
	@mkdir -p $(BUILD); rm -f $(BUILD)/mutation.txt

lint:
	$(VERILATOR) --lint-only -Wall --timing $(WAIVERS) \
	  --top-module apex_wcomp_bank $(RTL_SRCS) $(BANK)
	@echo "LINT CLEAN (bank -Wall; waivers scoped to packages only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_wcomp_vectors.py $(BUILD) $(N_ENG) \
	  | tee $(BUILD)/gen.log

build: build64 build128

define BUILD_D
build$(1): vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 30; done
	mkdir -p $(BUILD)/obj$(1)
	$(VERILATOR) $(VOPTS) -DCFG_D_TB=$(1) -DN_ENG_TB=$(N_ENG) \
	  --top-module tb_wcomp_bank -Mdir $(BUILD)/obj$(1) -o sim \
	  $(RTL_SRCS) $(BANK) $(TB_SRCS)
endef
$(eval $(call BUILD_D,64))
$(eval $(call BUILD_D,128))

run: run64 run128

define RUN_D
run$(1): build$(1)
	@rm -f $(BUILD)/fail_run$(1)
	@set -o pipefail; \
	  ./$(BUILD)/obj$(1)/sim +vectors=$(BUILD)/vec/wcomp_d$(1).txt \
	    2>&1 | tee $(BUILD)/run_d$(1).log \
	  || { echo "RUN FAIL [d$(1)]"; touch $(BUILD)/fail_run$(1); }
	@if ! grep -q "WCOMP BANK: ALL PASS" $(BUILD)/run_d$(1).log; \
	  then echo "RUN GATE FAIL [d$(1)]"; touch $(BUILD)/fail_run$(1); fi
	@if ! grep -q "checks=$(CHK) fails=0" $(BUILD)/run_d$(1).log; \
	  then echo "COUNT GATE FAIL [d$(1)] (pin $(CHK))"; \
	       touch $(BUILD)/fail_run$(1); fi
endef
$(eval $(call RUN_D,64))
$(eval $(call RUN_D,128))

# ── mutation gate: patched COPIES of the bank (rtl/ never touched); each
#    mutant must fail WITH ITS SIGNATURE (mutate_wcomp.py check). Run at
#    D=64 — the mutations are in the routing, which is geometry-blind. ─────
define RUN_MUT
mut_$(1):
	@rm -rf $(BUILD)/mut_$(1) $(BUILD)/obj_mut_$(1); rm -f $(BUILD)/fail_mut_$(1)
	$(PYTHON) mutate_wcomp.py patch $(1) $(BUILD)/mut_$(1)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 30; done
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) $(VOPTS) -DCFG_D_TB=64 -DN_ENG_TB=$(N_ENG) \
	  --top-module tb_wcomp_bank -Mdir $(BUILD)/obj_mut_$(1) -o sim \
	  $(RTL_SRCS) $(BUILD)/mut_$(1)/apex_wcomp_bank.sv $(TB_SRCS)
	-./$(BUILD)/obj_mut_$(1)/sim +vectors=$(BUILD)/vec/wcomp_d64.txt \
	  +watchdog=200000 > $(BUILD)/run_mut_$(1).log 2>&1
	@set -o pipefail; $(PYTHON) mutate_wcomp.py check $(1) \
	  $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt \
	  || touch $(BUILD)/fail_mut_$(1)
endef
$(eval $(call RUN_MUT,mW1))
$(eval $(call RUN_MUT,mW2))
$(eval $(call RUN_MUT,mW3))
$(eval $(call RUN_MUT,mW4))

gate:
	@fails=0; \
	for f in $(BUILD)/fail_run64 $(BUILD)/fail_run128 $(BUILD)/fail_mut_mW1 \
	         $(BUILD)/fail_mut_mW2 $(BUILD)/fail_mut_mW3 \
	         $(BUILD)/fail_mut_mW4; do \
	  if [ -e $$f ]; then echo "GATE: $$f"; fails=1; fi; done; \
	if [ $$fails -ne 0 ]; then echo "WCOMP SUITE: FAIL"; exit 1; fi; \
	echo "WCOMP SUITE: ALL PASS (d64 + d128, $(CHK) checks each, 4/4 signature mutants caught)" \
	  | tee $(BUILD)/gate.txt

clean:
	rm -rf $(BUILD)
