# verif/audit_csr/Makefile — independent adversarial audit of csr_regs.
#   make run     - build + run (PERF_W=32 and PERF_W=8 saturation build)
#   make mutant  - audit-checker credibility: perf_double mutant MUST fail

SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FILES = apex_pkg.sv csr/csr_regs.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))

VOPTS = --timing --assert --timescale 1ns/1ps -j 4 -I$(REPO)/verif/csr

.PHONY: run build mutant clean

build: $(BUILD)/obj_w32/Vtb_csr_attack $(BUILD)/obj_w8/Vtb_csr_attack

$(BUILD)/obj_w32/Vtb_csr_attack: $(RTL_SRCS) tb_csr_attack.sv
	mkdir -p $(BUILD)/obj_w32
	$(VERILATOR) --binary $(VOPTS) --top-module tb_csr_attack \
	    -Mdir $(BUILD)/obj_w32 $(RTL_SRCS) tb_csr_attack.sv

$(BUILD)/obj_w8/Vtb_csr_attack: $(RTL_SRCS) tb_csr_attack.sv
	mkdir -p $(BUILD)/obj_w8
	$(VERILATOR) --binary $(VOPTS) --top-module tb_csr_attack -GPERF_W=8 \
	    -Mdir $(BUILD)/obj_w8 $(RTL_SRCS) tb_csr_attack.sv

run: build
	set -o pipefail; $(BUILD)/obj_w32/Vtb_csr_attack +seed=901 \
	    | tee $(BUILD)/run_w32.log
	set -o pipefail; $(BUILD)/obj_w8/Vtb_csr_attack +seed=902 \
	    | tee $(BUILD)/run_w8.log

# mutant: PERF_BUSY counters double-count -> the cycle-accurate monitor must bite
mutant:
	$(PYTHON) mutate_audit.py perf_double $(REPO)/rtl $(BUILD)/mut_perf_double/rtl
	mkdir -p $(BUILD)/mut_perf_double/obj
	$(VERILATOR) --binary $(VOPTS) --top-module tb_csr_attack \
	    -Mdir $(BUILD)/mut_perf_double/obj \
	    $(addprefix $(BUILD)/mut_perf_double/rtl/,$(RTL_FILES)) tb_csr_attack.sv
	@if $(BUILD)/mut_perf_double/obj/Vtb_csr_attack +seed=903 \
	      > $(BUILD)/mut_perf_double/run.log 2>&1; then \
	    echo "MUTANT perf_double: NOT CAUGHT — audit checks are decorative"; exit 1; \
	  else \
	    echo "MUTANT perf_double CAUGHT:"; \
	    grep -m 3 -E "FAIL|%Error" $(BUILD)/mut_perf_double/run.log; \
	  fi

clean:
	rm -rf $(BUILD)
