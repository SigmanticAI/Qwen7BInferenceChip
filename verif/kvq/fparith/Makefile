# =============================================================================
# verif/kvq/fparith/Makefile — synthesizable integer fp-arith proof suite.
#
# Gates the `real`->integer rewrite of rtl/kvq/cores/cq_fp_pkg.sv:
#
#   make prove  - Python-level: int_model.py (the op-for-op mirror of the SV)
#                 proven EXHAUSTIVELY vs golden/apex_golden (prove.py)
#   make gen    - golden-derived vectors for the SV TB (gen_fparith_vectors.py)
#   make run    - tb_fparith.sv: exhaustive HW sweeps of the SHIPPING pkg +
#                 unit instances vs the golden vectors (see TB header)
#   make mut    - the 2 seeded-error checks, both MUST be caught:
#                   mut_tie  RNE tie flipped to half-up
#                   mut_eps  EPS=2^-14 floor removed
#   make synth  - synthesizability proof: sv2v + yosys (proc/opt/stat) on the
#                 full kvq_engine synthesis filelist; the flattened netlist
#                 must contain ZERO `real` / $bitstoreal remnants
#   make all    - everything above (the fparith gate)
# =============================================================================

SHELL      := /bin/bash
.SHELLFLAGS := -o pipefail -c
REPO       := ../../..
include $(REPO)/verif/tools.mk
PYTHON ?= python3

CORES      := $(REPO)/rtl/kvq/cores
# synthesis filelist: package FIRST, engine top LAST (mirrors smoke KVQ_SRCS)
SYN_FILES  := cq_fp_pkg.sv cq_scale_unit.sv cq_quant_unit.sv \
              cq_dequant_unit.sv scale_bank.sv \
              cq_rne_div_pipe.sv cq_quant_pipe.sv cq_scale_pipe.sv \
              residual_buffer.sv \
              sram_controller.sv cq_value_path.sv cq_key_path.sv
SYN_SRCS   := $(addprefix $(CORES)/, $(SYN_FILES)) $(REPO)/rtl/kvq/kvq_engine.sv

# TB only needs the pkg + the three unit wrappers
TB_SRCS    := $(CORES)/cq_fp_pkg.sv $(CORES)/cq_scale_unit.sv \
              $(CORES)/cq_quant_unit.sv $(CORES)/cq_dequant_unit.sv

VFLAGS     := --binary --timing --assert -j 0 --timescale 1ns/1ps -Wall \
              +incdir+build/vec

.PHONY: all prove gen build run mut mut_tie mut_eps synth clean

all: prove gen run mut synth
	@echo "FPARITH GATE: PASS"

prove:
	@mkdir -p build
	set -o pipefail; $(PYTHON) prove.py 2>&1 | tee build/prove.log

gen:
	@mkdir -p build/vec
	set -o pipefail; $(PYTHON) gen_fparith_vectors.py 2>&1 | tee build/gen.log
	@grep -q "GEN OK" build/gen.log

build: build/obj_fparith/Vtb_fparith

build/obj_fparith/Vtb_fparith: tb_fparith.sv $(TB_SRCS) build/vec/fparith_sizes.svh
	$(VERILATOR) $(VFLAGS) --Mdir build/obj_fparith --top-module tb_fparith \
	    $(TB_SRCS) tb_fparith.sv

run: build
	set -o pipefail; ./build/obj_fparith/Vtb_fparith 2>&1 | tee build/run.log
	@grep -q "FPARITH TB: ALL PASS" build/run.log

# ── seeded-error checks: the TB must FAIL on each mutant ─────────────────────
mut: mut_tie mut_eps
	@echo "MUTATION GATE: 2/2 seeded errors caught (mut_tie, mut_eps)" \
	    | tee build/mutation.txt

mut_tie mut_eps:
	@mkdir -p build/$@
	cp $(CORES)/cq_fp_pkg.sv build/$@/cq_fp_pkg.sv
	$(PYTHON) patch_fparith_mutant.py $@ build/$@/cq_fp_pkg.sv
	$(VERILATOR) --binary --timing --assert -j 0 --timescale 1ns/1ps \
	    +incdir+build/vec --Mdir build/obj_$@ --top-module tb_fparith \
	    build/$@/cq_fp_pkg.sv $(CORES)/cq_scale_unit.sv \
	    $(CORES)/cq_quant_unit.sv $(CORES)/cq_dequant_unit.sv tb_fparith.sv
	@echo "running seeded-error check $@ (must FAIL)..."
	@if ./build/obj_$@/Vtb_fparith > build/run_$@.log 2>&1; then \
	    echo "MUTATION ESCAPED: $@ was NOT caught"; exit 1; \
	 else \
	    echo "$@ CAUGHT (TB exited non-zero):"; \
	    grep -m 3 "FAIL" build/run_$@.log; \
	 fi

# ── synthesizability proof (sv2v + yosys, zero real remnants) ────────────────
synth:
	@mkdir -p build
	$(SV2V) --define=SYNTHESIS $(SYN_SRCS) > build/kvq_flat.v
	@echo "checking flattened netlist for real-math remnants..."
	@if grep -nE '(^|[^a-zA-Z_$$])real([^a-zA-Z_]|$$)|\$$bitstoreal|\$$realtobits|\$$itor|\$$rtoi|shortreal' \
	     build/kvq_flat.v; then \
	    echo "REAL-MATH REMNANT in build/kvq_flat.v"; exit 1; \
	 else echo "  none — clean"; fi
	$(YOSYS) -p "read_verilog build/kvq_flat.v; hierarchy -top kvq_engine; proc; opt; stat" \
	    > build/yosys.log 2>&1 || { tail -20 build/yosys.log; exit 1; }
	@grep -q "Printing statistics" build/yosys.log
	@grep -q "kvq_engine" build/yosys.log
	@! grep -iE "ERROR:|TOK_REAL" build/yosys.log
	@echo "SYNTH CHECK: PASS (yosys stat completed, zero real remnants)" \
	    | tee build/synth.txt

clean:
	rm -rf build
