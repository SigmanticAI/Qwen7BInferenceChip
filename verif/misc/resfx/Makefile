# =============================================================================
# verif/misc/resfx/Makefile — residual_add_fx (IB-LAYER S2) verification:
# the synthesizable integer C-6 residual primitive vs (a) golden-fp expected
# vectors and (b) the behavioral rtl/misc/residual_add.sv co-simulated on
# every vector, plus the mutation gate.
#
# Targets: all | lint | vectors | build | run | mutate | clean
# pipefail is load-bearing on every gate line (w4 Makefile precedent).
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL = \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/misc/residual_add_fx.sv \
    $(REPO)/rtl/misc/residual_add.sv

# S5 datapath closure: apex_residual unit (BUS_ON arbiter tensors)
RRTL = \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/top/glue/apex_residual.sv

RRUN = set -o pipefail; $(BUILD)/obj_resid/Vtb_resid_unit

VOPTS = -Wall --timing --assert --timescale 1ns/1ps

RUN = set -o pipefail; $(BUILD)/obj_resfx/Vtb_resfx

.PHONY: all lint vectors build run mutate resid_vectors resid_build resid_run resid_mutate clean

all: lint vectors build run mutate resid_vectors resid_build resid_run resid_mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module residual_add_fx \
	    $(REPO)/rtl/misc/f16_arith_pkg.sv $(REPO)/rtl/misc/residual_add_fx.sv
	@echo "LINT CLEAN (-Wall, no waivers)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_resfx_vectors.py $(BUILD) | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_resfx
	$(VERILATOR) --binary $(VOPTS) --top-module tb_resfx \
	    -Mdir $(BUILD)/obj_resfx $(RTL) tb_resfx.sv

run:
	$(RUN) +vectors=$(BUILD)/vectors_resfx.txt +insim=2000000 +seed=3115 \
	    | tee $(BUILD)/run_resfx.log

mutate:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutation.txt

resid_vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_resid_vectors.py $(BUILD) | tee $(BUILD)/gen_resid.log

resid_build:
	mkdir -p $(BUILD)/obj_resid
	$(VERILATOR) --binary $(VOPTS) --top-module tb_resid_unit \
	    -Mdir $(BUILD)/obj_resid $(RRTL) tb_resid_unit.sv

resid_run:
	set -o pipefail; $(RRUN) +stall_mode=1 +seed=511 | tee $(BUILD)/run_resid_stall.log
	set -o pipefail; $(RRUN) +stall_mode=0 +seed=512 | tee $(BUILD)/run_resid_fast.log
	set -o pipefail; $(RRUN) +stall_mode=1 +seed=513 +resets=1 | tee $(BUILD)/run_resid_reset.log

resid_mutate:
	set -o pipefail; $(PYTHON) mutation_check_resid.py | tee $(BUILD)/mutation_resid.txt

clean:
	rm -rf $(BUILD)
