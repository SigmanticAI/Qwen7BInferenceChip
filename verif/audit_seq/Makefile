# verif/audit_seq/Makefile — independent adversarial audit of seq_walker.
# Reuses the deliverable's mxe_stub + SVA packs unchanged (read-only).
#   make run       - build + run the attack TB on the shipped RTL
#   make mutant    - audit-checker credibility: busy_skid mutant MUST fail

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FILES = apex_pkg.sv xbr/stream_skid.sv seq/seq_walker.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))
TB_SRCS   = $(REPO)/verif/seq/mxe_stub.sv tb_seq_attack.sv

VOPTS = --timing --assert --timescale 1ns/1ps -j 4 \
        -I$(REPO)/verif/common -I$(REPO)/verif/seq

.PHONY: run build mutant clean

build: $(BUILD)/obj/Vtb_seq_attack

$(BUILD)/obj/Vtb_seq_attack: $(RTL_SRCS) tb_seq_attack.sv
	mkdir -p $(BUILD)/obj
	$(VERILATOR) --binary $(VOPTS) --top-module tb_seq_attack \
	    -Mdir $(BUILD)/obj $(RTL_SRCS) $(TB_SRCS)

run: build
	set -o pipefail; $(BUILD)/obj/Vtb_seq_attack +seed=777 +lat_mode=1 \
	    +n_aborts=400 | tee $(BUILD)/run_attack.log

# mutant: busy no longer covers a beat held only in the input skid.
# Kill requires the mutant's OWN failure signature (mutate_audit.py check,
# house verif/top/l3 shape) — an infra failure (watchdog, drain timeout)
# is a gate FAIL, never a kill.
mutant:
	$(PYTHON) mutate_audit.py patch busy_skid $(REPO)/rtl $(BUILD)/mut_busy_skid/rtl
	mkdir -p $(BUILD)/mut_busy_skid/obj
	$(VERILATOR) --binary $(VOPTS) --top-module tb_seq_attack \
	    -Mdir $(BUILD)/mut_busy_skid/obj \
	    $(addprefix $(BUILD)/mut_busy_skid/rtl/,$(RTL_FILES)) $(TB_SRCS)
	-$(BUILD)/mut_busy_skid/obj/Vtb_seq_attack +seed=778 +lat_mode=1 \
	    +n_aborts=20 > $(BUILD)/mut_busy_skid/run.log 2>&1
	$(PYTHON) mutate_audit.py check busy_skid $(BUILD)/mut_busy_skid/run.log

clean:
	rm -rf $(BUILD)
