# =============================================================================
# verif/csr/Makefile — CSR (tile register window) smoke suite (house
# verif/mxe/sb / verif/tip/sb pattern):
#   vector-driven scoreboard TB against an independent golden register model
#   (gen_csr_vectors.py) + bus-level PERF-counter mirror + CSR contract SVA
#   (csr_sb_sva.svh, --assert build gate D-012) + manual coverage buckets +
#   checker-credibility mutations.
#
# Targets:
#   make all      - lint, vectors, build (PERF_W=32 and PERF_W=8), 4 runs,
#                   coverage aggregation gate, 2 mutation-kill checks
#   make lint     - verilator --lint-only -Wall on csr_regs
#   make vectors  - golden model + vector generation
#   make mutants  - checker-credibility gate: every mutant run must FAIL
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on csr_regs.sv, the SVA checker, or the TB.
# =============================================================================

.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FILES = apex_pkg.sv csr/csr_regs.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))

TB_SRCS = tb_csr_sb.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -j 4 lint_waivers.vlt

RUN32 = set -o pipefail; $(BUILD)/obj_w32/Vtb_csr_sb
RUN8  = set -o pipefail; $(BUILD)/obj_w8/Vtb_csr_sb

.PHONY: all lint vectors build run_directed run_random run_random2 run_sat \
        coverage mutants mut_sticky_race mut_perf_nogate clean

all: lint vectors build run_directed run_random run_random2 run_sat \
     coverage mutants
	@echo "==== CSR SUITE: ALL TARGETS COMPLETE ===="

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module csr_regs $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_csr_vectors.py | tee $(BUILD)/vectors.log

build: $(BUILD)/obj_w32/Vtb_csr_sb $(BUILD)/obj_w8/Vtb_csr_sb

$(BUILD)/obj_w32/Vtb_csr_sb: $(RTL_SRCS) $(TB_SRCS) csr_sb_sva.svh
	mkdir -p $(BUILD)/obj_w32
	$(VERILATOR) --binary $(VOPTS) --top-module tb_csr_sb \
	    -Mdir $(BUILD)/obj_w32 $(RTL_SRCS) $(TB_SRCS)

$(BUILD)/obj_w8/Vtb_csr_sb: $(RTL_SRCS) $(TB_SRCS) csr_sb_sva.svh
	mkdir -p $(BUILD)/obj_w8
	$(VERILATOR) --binary $(VOPTS) --top-module tb_csr_sb -GPERF_W=8 \
	    -Mdir $(BUILD)/obj_w8 $(RTL_SRCS) $(TB_SRCS)

run_directed:
	$(RUN32) +vectors=$(BUILD)/vectors_directed.txt +seed=201 \
	    | tee $(BUILD)/run_directed.log

run_random:
	$(RUN32) +vectors=$(BUILD)/vectors_random.txt +seed=202 \
	    | tee $(BUILD)/run_random.log

# same random file, different bus pacing (request gaps reshuffle)
run_random2:
	$(RUN32) +vectors=$(BUILD)/vectors_random.txt +seed=303 \
	    | tee $(BUILD)/run_random2.log

# PERF_W=8 build: counters must PEG at 0xFF, never wrap
run_sat:
	$(RUN8) +vectors=$(BUILD)/vectors_sat.txt +seed=204 \
	    | tee $(BUILD)/run_sat.log

coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log | tee $(BUILD)/coverage.txt

# ── checker-credibility mutations (scratch-copy RTL; suite MUST fail with
#    the mutant's OWN signature — mutate.py check, house verif/top/l3 shape;
#    an infra failure (missing vectors, watchdog $fatal) is a gate FAIL,
#    never a kill) ─────────────────────────────────────────────────────────
define MUT_RULE
	$(PYTHON) mutate.py patch $(1) $(REPO)/rtl $(BUILD)/mut_$(1)/rtl
	mkdir -p $(BUILD)/mut_$(1)/obj
	$(VERILATOR) --binary --timing --assert --timescale 1ns/1ps -j 4 \
	    lint_waivers.vlt --top-module tb_csr_sb \
	    -Mdir $(BUILD)/mut_$(1)/obj \
	    $(addprefix $(BUILD)/mut_$(1)/rtl/,$(RTL_FILES)) $(TB_SRCS)
	-$(BUILD)/mut_$(1)/obj/Vtb_csr_sb +vectors=$(BUILD)/vectors_$(2).txt \
	    +seed=99$(3) > $(BUILD)/mut_$(1)/run.log 2>&1
	$(PYTHON) mutate.py check $(1) $(BUILD)/mut_$(1)/run.log
endef

# mutation targets cannot run before their vectors exist — an unreadable
# vector file is a $fatal (infra failure) and must never look like a kill
$(BUILD)/vectors_directed.txt: gen_csr_vectors.py
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_csr_vectors.py | tee $(BUILD)/vectors.log

mut_sticky_race: $(BUILD)/vectors_directed.txt
	$(call MUT_RULE,sticky_race,directed,1)

mut_perf_nogate: $(BUILD)/vectors_directed.txt
	$(call MUT_RULE,perf_nogate,directed,2)

mutants: mut_sticky_race mut_perf_nogate
	@echo "MUTATION GATE: 2/2 mutants caught"

clean:
	rm -rf $(BUILD)
