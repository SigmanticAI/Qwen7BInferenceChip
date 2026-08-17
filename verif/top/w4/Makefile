# =============================================================================
# verif/top/w4/Makefile — the W4 INGEST LANE tile suite (D-031 combine;
# docs/design/W4_DATAPATH.md).
#
# Reuses the L3 harness (../l3/tb_apex_l3.sv) built at the b64 point with
# -GW4EN=1 (apex_top W4_LANE=1): one case file drives 9 sub-jobs — W4B
# GEMMs (raw + requant + ragged tails + odd beats), the K-split stripe
# with ONE shared s8, passthrough framing, the lane-off INT8 baseline
# equivalence, and the W4_CNTI/W4_CNTO ingest-proof counter checks — every
# MXE result beat bit-exact vs golden/apex_golden/w4_lane.py.
#
#   make all   - lint + vectors + build + run + mutants
#   make clean
#
# pipefail is load-bearing on every gate line (the tee lesson).
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/rope \
        -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I../l3 ../l3/lint_waivers.vlt

.PHONY: all lint vectors build run mutants clean

all: lint vectors build run mutants

# both elaborations must stay -Wall clean: the lane on, and (the default
# every other suite builds) the lane absent
lint:
	$(VERILATOR) --lint-only $(VOPTS) -f tile_rtl.f --top-module apex_top
	$(VERILATOR) --lint-only $(VOPTS) -f tile_rtl.f -GW4_LANE=1 \
	    --top-module apex_top
	@echo "LINT CLEAN (W4_LANE=0 and =1; -Wall, waivers scoped as in l3)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_w4_tile_vectors.py $(BUILD) \
	    | tee $(BUILD)/gen.log

build:
	mkdir -p $(BUILD)/obj_w4
	$(VERILATOR) --binary $(VOPTS) -GCFG_D=64 -GG_CFG=16 -GDEPTH=256 \
	    -GOUTK=0 -GW4EN=1 --top-module tb_apex_l3 -Mdir $(BUILD)/obj_w4 \
	    -f tile_rtl.f ../l3/tb_apex_l3.sv

# throwaway mutant build (mutants.sh only — the real gate binary in
# obj_w4 is never touched by a mutant build). -Wno-fatal: a mutant only
# has to be KILLED by the checks, not stay lint-clean.
build_mut:
	mkdir -p $(BUILD)/obj_mut
	$(VERILATOR) --binary $(VOPTS) -Wno-fatal -GCFG_D=64 -GG_CFG=16 \
	    -GDEPTH=256 -GOUTK=0 -GW4EN=1 --top-module tb_apex_l3 \
	    -Mdir $(BUILD)/obj_mut -f tile_rtl.f ../l3/tb_apex_l3.sv

run:
	set -o pipefail; $(BUILD)/obj_w4/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/w4_gemm_b64.txt +watchdog=8000000 \
	    | tee $(BUILD)/run_w4.log
	@grep -q "L3 PASS" $(BUILD)/run_w4.log \
	    || { echo "W4 TILE GATE: FAIL"; exit 1; }
	@echo "W4 TILE GATE: PASS"

# 3 single-line mutations of the NEW glue, each applied alone to the real
# tree (EXIT-trap reverted), built into a throwaway obj, and judged by the
# sim's OWN exit code (the f2sim mutants.sh discipline: a clean CONTROL
# must pass first, so a fail-everything harness cannot launder a survivor)
mutants:
	./mutants.sh

clean:
	rm -rf $(BUILD)
