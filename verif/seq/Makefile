# =============================================================================
# verif/seq/Makefile — SEQ (seq_walker) smoke suite (house verif/mxe/sb /
# verif/tip/sb pattern):
#   vector-driven scoreboard TB against an independent golden queue model
#   (gen_seq_vectors.py, >=200 randomized descriptors) + a D-006-contract
#   behavioral MXE stub (mxe_stub.sv, randomized latencies/backpressure;
#   MXE itself is a verified black box) + house §5 stream SVA
#   (apex_stream1_sva on ds/md/result) + SEQ contract SVA (seq_sb_sva.svh:
#   D-006 serialization, busy honesty, queue bookkeeping, abort drain) +
#   phase-targeted mid-op resets + manual coverage buckets + mutations.
#
# Targets:
#   make all      - lint, vectors, build, 6 run configs, coverage gate,
#                   2 mutation-kill checks
#   make lint     - verilator --lint-only -Wall on seq_walker
#   make vectors  - independent golden model + vector generation
#   make mutants  - checker-credibility gate: every mutant run must FAIL
#   make clean
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on seq_walker.sv, mxe_stub.sv, the SVA, or the TB.
# =============================================================================

.SHELLFLAGS := -o pipefail -c
# NOTE: .SHELLFLAGS requires GNU make >= 3.82. Stock macOS make is 3.81 and
# SILENTLY IGNORES it, so every piped gate recipe below also carries an
# inline 'set -o pipefail;' — the gate must bite under either make.
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_FILES = apex_pkg.sv xbr/stream_skid.sv seq/seq_walker.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))

TB_SRCS = mxe_stub.sv tb_seq_sb.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -j 4 \
        -I$(REPO)/verif/common lint_waivers.vlt

RUN = set -o pipefail; $(BUILD)/obj/Vtb_seq_sb

.PHONY: all lint vectors build run_directed run_fast run_random run_storm \
        run_stall run_reset coverage mutants mut_d006_early mut_fifo_skip \
        clean

all: lint vectors build run_directed run_fast run_random run_storm \
     run_stall run_reset coverage mutants
	@echo "==== SEQ SUITE: ALL TARGETS COMPLETE ===="

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module seq_walker $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_seq_vectors.py | tee $(BUILD)/vectors.log

build: $(BUILD)/obj/Vtb_seq_sb

$(BUILD)/obj/Vtb_seq_sb: $(RTL_SRCS) $(TB_SRCS) seq_sb_sva.svh \
                         $(REPO)/verif/common/apex_stream1_sva.svh
	mkdir -p $(BUILD)/obj
	$(VERILATOR) --binary $(VOPTS) --top-module tb_seq_sb \
	    -Mdir $(BUILD)/obj $(RTL_SRCS) $(TB_SRCS)

# directed edges: random stub latency + feed gaps + result backpressure
run_directed:
	$(RUN) +vectors=$(BUILD)/vectors_directed.txt +lat_mode=1 +bp_mode=1 \
	    +gap_mode=1 +seed=401 | tee $(BUILD)/run_directed.log

# same directed set at full rate (zero-latency stub, no gaps, no bp)
run_fast:
	$(RUN) +vectors=$(BUILD)/vectors_directed.txt +lat_mode=0 +bp_mode=0 \
	    +gap_mode=0 +seed=402 | tee $(BUILD)/run_fast.log

# >=200 randomized descriptors with aborts/gating interleaved
run_random:
	$(RUN) +vectors=$(BUILD)/vectors_random.txt +lat_mode=1 +bp_mode=1 \
	    +gap_mode=1 +seed=403 | tee $(BUILD)/run_random.log

# storm everything: slow stub accept, ~6% result duty, bursty feed
run_storm:
	$(RUN) +vectors=$(BUILD)/vectors_random.txt +lat_mode=2 +bp_mode=2 \
	    +gap_mode=2 +seed=404 | tee $(BUILD)/run_storm.log

# slow stub with a free-running result side (queue pressure, no res bp)
run_stall:
	$(RUN) +vectors=$(BUILD)/vectors_random.txt +lat_mode=2 +bp_mode=0 \
	    +gap_mode=2 +seed=405 | tee $(BUILD)/run_stall.log

# phase-targeted mid-operation hard resets (Layer-1 requirement)
run_reset:
	$(RUN) +vectors=$(BUILD)/vectors_reset.txt +lat_mode=1 +bp_mode=1 \
	    +gap_mode=0 +seed=406 | tee $(BUILD)/run_reset.log

coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD)/run_*.log | tee $(BUILD)/coverage.txt

# ── checker-credibility mutations (scratch-copy RTL; suite MUST fail) ───────
# House verif/top/l3 shape: the mutant sim's exit status is IGNORED ('-');
# the gate is 'mutate.py check', which requires the ABSENCE of the TB PASS
# banner AND the PRESENCE of the mutant-specific failure signature. A run
# that merely crashes (missing vectors, watchdog, segfault) FAILS the gate
# instead of masquerading as a kill.
define MUT_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/mut_$(1)/rtl
	mkdir -p $(BUILD)/mut_$(1)/obj
	$(VERILATOR) --binary --timing --assert --timescale 1ns/1ps -j 4 \
	    lint_waivers.vlt -I$(REPO)/verif/common --top-module tb_seq_sb \
	    -Mdir $(BUILD)/mut_$(1)/obj \
	    $(addprefix $(BUILD)/mut_$(1)/rtl/,$(RTL_FILES)) $(TB_SRCS)
	-$(BUILD)/mut_$(1)/obj/Vtb_seq_sb \
	    +vectors=$(BUILD)/vectors_$(2).txt +lat_mode=1 +bp_mode=1 \
	    +gap_mode=1 +seed=99$(3) > $(BUILD)/mut_$(1)/run.log 2>&1
	$(PYTHON) mutate.py check $(1) $(BUILD)/mut_$(1)/run.log
endef

# 'vectors' prerequisite: a mutant run must never start without (or against
# stale) stimulus — the generator reruns, and with pipefail a generator
# crash stops the gate instead of the sim consuming last run's vectors.
mut_d006_early: vectors
	$(call MUT_RULE,d006_early,directed,1)

mut_fifo_skip: vectors
	$(call MUT_RULE,fifo_skip,random,2)

mutants: mut_d006_early mut_fifo_skip
	@echo "MUTATION GATE: 2/2 mutants caught"

clean:
	rm -rf $(BUILD)
