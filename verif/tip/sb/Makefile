# =============================================================================
# verif/tip/sb/Makefile — TIP INDEPENDENT verification suite:
#   vector-driven scoreboard TB (house verif/mxe/sb pattern) against an
#   independent golden model + house §5 stream SVA (apex_stream1_sva) +
#   TIP conservation/busy/F-2 checker + protocol adversaries + phase-targeted
#   mid-operation reset + manual coverage buckets + checker mutations.
#
# Targets:
#   make all      - lint, vectors, build (N=64 and N=4096), 7 run configs,
#                   coverage aggregation gate, 4 mutation-kill checks
#   make lint     - verilator --lint-only -Wall on tip_top
#   make vectors  - independent golden model + vector generation (also
#                   re-proves frozen-143 equivalence independently)
#   make mutants  - checker-credibility gate: every mutant run must FAIL
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any TIP RTL file, the SVA checkers, or the TB.
# =============================================================================

.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FILES = apex_pkg.sv xbr/stream_skid.sv tip/tip_decide.sv \
            tip/tip_importance.sv tip/tip_top.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))

TB_SRCS = tb_tip_sb.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -j 4 \
        -I$(REPO)/verif/common lint_waivers.vlt

RUN64   = set -o pipefail; $(BUILD)/obj_n64/Vtb_tip_sb
RUN4096 = set -o pipefail; $(BUILD)/obj_n4096/Vtb_tip_sb

.PHONY: all lint vectors build run_directed run_directed_fast run_random \
        run_storm run_stall run_reset run_n4096 coverage mutants \
        mut_tie_ge mut_skid_xor mut_sat_wrap mut_busy_hole clean

all: lint vectors build run_directed run_directed_fast run_random run_storm \
     run_stall run_reset run_n4096 coverage mutants
	@echo "==== TIP INDEPENDENT SUITE: ALL TARGETS COMPLETE ===="

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module tip_top $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_tip_sb_vectors.py | tee $(BUILD)/vectors.log

build: $(BUILD)/obj_n64/Vtb_tip_sb $(BUILD)/obj_n4096/Vtb_tip_sb

$(BUILD)/obj_n64/Vtb_tip_sb: $(RTL_SRCS) $(TB_SRCS) tip_sb_sva.svh \
                             $(REPO)/verif/common/apex_stream1_sva.svh
	mkdir -p $(BUILD)/obj_n64
	$(VERILATOR) --binary $(VOPTS) --top-module tb_tip_sb \
	    -GBLOCK_M=8 -GBLOCK_N=8 \
	    -Mdir $(BUILD)/obj_n64 $(RTL_SRCS) $(TB_SRCS)

$(BUILD)/obj_n4096/Vtb_tip_sb: $(RTL_SRCS) $(TB_SRCS) tip_sb_sva.svh \
                               $(REPO)/verif/common/apex_stream1_sva.svh
	mkdir -p $(BUILD)/obj_n4096
	$(VERILATOR) --binary $(VOPTS) --top-module tb_tip_sb \
	    -GBLOCK_M=64 -GBLOCK_N=64 \
	    -Mdir $(BUILD)/obj_n4096 $(RTL_SRCS) $(TB_SRCS)

# directed edges under random gaps + random backpressure
run_directed:
	$(RUN64) $$(cat $(BUILD)/directed.args) +bp_mode=1 +gap_mode=1 +seed=101 \
	    | tee $(BUILD)/run_directed.log

# same directed set at full rate (no gaps, no backpressure)
run_directed_fast:
	$(RUN64) $$(cat $(BUILD)/directed.args) +bp_mode=0 +gap_mode=0 +seed=102 \
	    | tee $(BUILD)/run_directed_fast.log

# >=500 seeded random tiles (1216) + interleaved overrun frames
run_random:
	$(RUN64) $$(cat $(BUILD)/random.args) +bp_mode=1 +gap_mode=1 +seed=103 \
	    | tee $(BUILD)/run_random.log

# decision-stream backpressure STORM + bursty stalled feeds
run_storm:
	$(RUN64) $$(cat $(BUILD)/storm.args) +bp_mode=2 +gap_mode=2 +seed=104 \
	    | tee $(BUILD)/run_storm.log

# stalled feeds with NO decision backpressure
run_stall:
	$(RUN64) $$(cat $(BUILD)/storm.args) +bp_mode=0 +gap_mode=2 +seed=105 \
	    | tee $(BUILD)/run_stall.log

# phase-targeted mid-operation resets (the Layer-1 hole the smoke left open)
run_reset:
	$(RUN64) $$(cat $(BUILD)/reset.args) +bp_mode=1 +gap_mode=0 +seed=106 \
	    | tee $(BUILD)/run_reset.log

# N=4096 build: frozen-143 subset round-trip + big-N directed/overrun
run_n4096:
	$(RUN4096) $$(cat $(BUILD)/n4096.args) +bp_mode=1 +gap_mode=1 +seed=107 \
	    | tee $(BUILD)/run_n4096.log

coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log | tee $(BUILD)/coverage.txt

# ── checker-credibility mutations (scratch-copy RTL; suite MUST fail) ───────
# each: copy+mutate RTL into build/mut_<name>/rtl, rebuild, run one config,
# then gate via 'mutate.py check' (house shape: verif/top/l3/mutate.py):
# the run log must NOT contain the TB PASS banner AND must contain the
# mutant-specific failure signature. A bare non-zero exit (missing vectors,
# harness crash, watchdog) is an infra FAILURE, never a caught mutant.
# rtl/ in the tree is never touched.
define MUT_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/mut_$(1)/rtl
	mkdir -p $(BUILD)/mut_$(1)/obj
	$(VERILATOR) --binary --timing --assert --timescale 1ns/1ps -j 4 \
	    lint_waivers.vlt -I$(REPO)/verif/common --top-module tb_tip_sb \
	    -GBLOCK_M=8 -GBLOCK_N=8 -Mdir $(BUILD)/mut_$(1)/obj \
	    $(addprefix $(BUILD)/mut_$(1)/rtl/,$(RTL_FILES)) $(TB_SRCS)
	@$(BUILD)/mut_$(1)/obj/Vtb_tip_sb +vectors=$(BUILD)/vectors_$(2).txt \
	      +imp_hi=$(3) +imp_lo=$(4) +bp_mode=$(5) +gap_mode=$(6) +seed=99$(7) \
	      > $(BUILD)/mut_$(1)/run.log 2>&1; \
	  $(PYTHON) mutate.py check $(1) $(BUILD)/mut_$(1)/run.log
endef

# vectors is a hard prerequisite: without it a missing vector file would
# $fatal the TB and the old any-failure gate counted that as "caught"
mut_tie_ge: vectors
	$(call MUT_RULE,tie_ge,directed,64000,6400,1,1,1)

mut_skid_xor: vectors
	$(call MUT_RULE,skid_xor,directed,64000,6400,1,1,2)

mut_sat_wrap: vectors
	$(call MUT_RULE,sat_wrap,directed,64000,6400,1,1,3)

mut_busy_hole: vectors
	$(call MUT_RULE,busy_hole,storm,1500,300,2,2,4)

mutants: mut_tie_ge mut_skid_xor mut_sat_wrap mut_busy_hole
	@echo "MUTATION GATE: 4/4 mutants caught"

clean:
	rm -rf $(BUILD)
