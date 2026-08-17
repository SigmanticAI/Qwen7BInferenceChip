# =============================================================================
# verif/asu/swiglu/Makefile — apex_layer_deq + asu_swiglu (IB-LAYER S3):
# golden-composition vectors (silu_apply*up arbiter; exact-f32 deq arbiter),
# engineered Q5.10/product RNE ties, clamp/ovfl/zero batteries, legality +
# frame + exact-or-refused paths, regimes, mid-op resets, mutation gate.
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

DEQ_RTL = \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/top/glue/apex_layer_deq.sv

SWG_RTL = \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/asu_silu.sv \
    $(REPO)/rtl/asu/asu_swiglu.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/asu

RUND = set -o pipefail; $(BUILD)/obj_deq/Vtb_layer_deq
RUNS = set -o pipefail; $(BUILD)/obj_swg/Vtb_swiglu

.PHONY: all lint vectors build run mutate clean

all: lint vectors build run mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module apex_layer_deq $(DEQ_RTL)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_swiglu $(SWG_RTL)
	@echo "LINT CLEAN (-Wall, no waivers)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_swiglu_vectors.py $(BUILD) | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_deq $(BUILD)/obj_swg
	$(VERILATOR) --binary $(VOPTS) --top-module tb_layer_deq \
	    -Mdir $(BUILD)/obj_deq $(DEQ_RTL) tb_layer_deq.sv
	$(VERILATOR) --binary $(VOPTS) --top-module tb_swiglu \
	    -Mdir $(BUILD)/obj_swg $(SWG_RTL) tb_swiglu.sv

run:
	set -o pipefail; $(RUND) +bp_mode=1 +stall_mode=1 +seed=301 | tee $(BUILD)/run_deq_bp.log
	set -o pipefail; $(RUND) +bp_mode=0 +stall_mode=0 +seed=302 | tee $(BUILD)/run_deq_fast.log
	set -o pipefail; $(RUND) +bp_mode=1 +stall_mode=0 +seed=303 +resets=5 | tee $(BUILD)/run_deq_reset.log
	set -o pipefail; $(RUNS) +bp_mode=1 +stall_mode=1 +seed=304 | tee $(BUILD)/run_swg_bp.log
	set -o pipefail; $(RUNS) +bp_mode=0 +stall_mode=0 +seed=305 | tee $(BUILD)/run_swg_fast.log
	set -o pipefail; $(RUNS) +bp_mode=2 +stall_mode=2 +seed=306 | tee $(BUILD)/run_swg_storm.log
	set -o pipefail; $(RUNS) +bp_mode=1 +stall_mode=0 +seed=307 +resets=7 | tee $(BUILD)/run_swg_reset.log

mutate:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutation.txt

clean:
	rm -rf $(BUILD)
