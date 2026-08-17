# =============================================================================
# verif/rope/row/Makefile — rope_row + rope_pair_fx (IB-LAYER S2): golden-fp
# expected vectors (gen self-checked vs rope_fx) + SHADOW behavioral rope.sv
# co-check on every beat + frame-error paths + regimes + mid-op resets +
# mutation gate. Both build points (D=64, D=128).
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL = \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/rope/rope_pair_fx.sv \
    $(REPO)/rtl/rope/rope_row.sv \
    $(REPO)/rtl/rope/rope.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/rope

RUN64  = set -o pipefail; $(BUILD)/obj_d64/Vtb_rope_row
RUN128 = set -o pipefail; $(BUILD)/obj_d128/Vtb_rope_row

.PHONY: all lint vectors build run mutate clean

all: lint vectors build run mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module rope_row \
	    $(REPO)/rtl/misc/f16_arith_pkg.sv $(REPO)/rtl/xbr/stream_skid.sv \
	    $(REPO)/rtl/rope/rope_pair_fx.sv $(REPO)/rtl/rope/rope_row.sv
	@echo "LINT CLEAN (-Wall, no waivers)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_rope_row_vectors.py $(BUILD) 64 | tee $(BUILD)/gen64.log
	set -o pipefail; $(PYTHON) gen_rope_row_vectors.py $(BUILD) 128 | tee $(BUILD)/gen128.log

build:
	mkdir -p $(BUILD)/obj_d64 $(BUILD)/obj_d128
	$(VERILATOR) --binary $(VOPTS) --top-module tb_rope_row \
	    -Mdir $(BUILD)/obj_d64 $(RTL) tb_rope_row.sv
	$(VERILATOR) --binary $(VOPTS) -GD=128 --top-module tb_rope_row \
	    -Mdir $(BUILD)/obj_d128 $(RTL) tb_rope_row.sv

run:
	set -o pipefail; $(RUN64) +bp_mode=1 +stall_mode=1 +seed=101 | tee $(BUILD)/run_d64_bp.log
	set -o pipefail; $(RUN64) +bp_mode=0 +stall_mode=0 +seed=102 | tee $(BUILD)/run_d64_fast.log
	set -o pipefail; $(RUN64) +bp_mode=2 +stall_mode=2 +seed=103 | tee $(BUILD)/run_d64_storm.log
	set -o pipefail; $(RUN64) +bp_mode=1 +stall_mode=0 +seed=104 +resets=17 | tee $(BUILD)/run_d64_reset.log
	set -o pipefail; $(RUN128) +bp_mode=1 +stall_mode=1 +seed=105 | tee $(BUILD)/run_d128_bp.log

mutate:
	set -o pipefail; $(PYTHON) mutation_check.py | tee $(BUILD)/mutation.txt

clean:
	rm -rf $(BUILD)
