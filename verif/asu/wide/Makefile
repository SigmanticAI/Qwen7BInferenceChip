# =============================================================================
# verif/asu/wide/Makefile — WIDE-D RMSNorm verification (C-RMSW,
# docs/design/WIDE_RMSNORM.md §4): golden-arbiter vectors + two wide
# elaborations (RMS_D_MAX 3584 / 8192) + adversarial runs + mid-op resets +
# provenance gate on the MU/S params + mutation gate.
#
# Targets:
#   make all          - params-check, vectors, lint, both builds, 7 run
#                       configurations, mutation gate
#   make params-check - regenerate asu_wide_rms_params.svh from the golden
#                       arbiter (_wide_mu) and diff the checked-in rtl copy
#   make vectors      - regenerate stimulus/expected from rmsnorm_fx_wide
#   make lint         - verilator --lint-only -Wall at both wide elaborations
#   make mutants      - prove the checkers catch injected bugs (M3 reuse,
#                       MU+1, shift-1, SUM_W=27; scratch copies only)
#   make clean
#
# Waivers: lint_waivers.vlt scopes waivers to the FROZEN rtl/apex_pkg.sv only.
# The frozen D<=128 anchor suites are verif/asu/smoke + verif/asu/sb —
# READ-ONLY from this suite (regression stage 3 of the contract).
# =============================================================================

.SHELLFLAGS := -o pipefail -c
SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_RMS = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/asu/rsqrt.sv \
    $(REPO)/rtl/asu/asu_rmsnorm.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps lint_waivers.vlt \
        -I$(REPO)/verif/common +incdir+$(REPO)/rtl/asu

RUN_W   = set -o pipefail; $(BUILD)/obj_wide/Vtb_asu_rmsnorm_wide
RUN_W8K = set -o pipefail; $(BUILD)/obj_wide8k/Vtb_asu_rmsnorm_wide

.PHONY: all params-check vectors lint build_wide build_wide8k \
        run_wide_directed run_wide_fast run_wide_random run_wide_storm \
        run_wide_geager run_wide_reset run_wide8k \
        mutants waves_wide clean

all: params-check vectors lint build_wide build_wide8k \
     run_wide_directed run_wide_fast run_wide_random run_wide_storm \
     run_wide_geager run_wide_reset run_wide8k \
     mutants
	@echo "ASU WIDE: full suite passed (params provenance, bit-exact vs rmsnorm_fx_wide, rejects, resets, mutation gate)"

# ── provenance gate: checked-in MU/S params == regenerated-from-golden ──────
params-check:
	mkdir -p $(BUILD)
	$(PYTHON) $(REPO)/rtl/asu/gen_wide_rms_params.py --out $(BUILD)/asu_wide_rms_params.regen.svh
	diff -q $(BUILD)/asu_wide_rms_params.regen.svh $(REPO)/rtl/asu/asu_wide_rms_params.svh
	@echo "PARAMS CHECK: rtl/asu/asu_wide_rms_params.svh matches the golden arbiter (_wide_mu)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_wide_rms_vectors.py --outdir $(BUILD) | tee $(BUILD)/gen.log

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_rmsnorm \
	    -GRMS_D_MAX=3584 $(RTL_RMS)
	$(VERILATOR) --lint-only $(VOPTS) --top-module asu_rmsnorm \
	    -GRMS_D_MAX=8192 $(RTL_RMS)
	@echo "LINT CLEAN (-Wall, wide elaborations 3584 + 8192; waivers scoped to frozen apex_pkg.sv only)"

build_wide:
	mkdir -p $(BUILD)/obj_wide
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_asu_rmsnorm_wide \
	    -Mdir $(BUILD)/obj_wide $(RTL_RMS) tb_asu_rmsnorm_wide.sv

build_wide8k:
	mkdir -p $(BUILD)/obj_wide8k
	$(VERILATOR) --binary $(VOPTS) --trace-fst --top-module tb_asu_rmsnorm_wide \
	    -GD_MAX=8192 \
	    -Mdir $(BUILD)/obj_wide8k $(RTL_RMS) tb_asu_rmsnorm_wide.sv

# ── runs (D_MAX = 3584) ──────────────────────────────────────────────────────
run_wide_directed:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_directed.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=0 +seed=31 | tee $(BUILD)/run_wide_directed.log

run_wide_fast:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_directed.txt +bp_mode=0 \
	    +stall_mode=0 +g_mode=0 +seed=32 | tee $(BUILD)/run_wide_fast.log

# gamma LATE adversary on the random set
run_wide_random:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_random.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=1 +seed=33 | tee $(BUILD)/run_wide_random.log

# backpressure storm + bursty x feed + eager gamma (g-skid parked over WAIT)
run_wide_storm:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_directed.txt +bp_mode=2 \
	    +stall_mode=2 +g_mode=2 +seed=34 | tee $(BUILD)/run_wide_storm.log

run_wide_geager:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_random.txt +bp_mode=1 \
	    +stall_mode=0 +g_mode=2 +seed=35 | tee $(BUILD)/run_wide_geager.log

run_wide_reset:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_reset.txt +bp_mode=1 \
	    +stall_mode=0 +g_mode=0 +seed=36 | tee $(BUILD)/run_wide_reset.log

# ── the 8192 elaboration: sum2 == 2^27 corner ────────────────────────────────
run_wide8k:
	$(RUN_W8K) +vectors=$(BUILD)/vectors_wide8k.txt +bp_mode=1 \
	    +stall_mode=0 +g_mode=0 +seed=37 | tee $(BUILD)/run_wide8k.log

# ── gates ────────────────────────────────────────────────────────────────────
mutants:
	set -o pipefail; $(PYTHON) mutation_check_wide.py | tee $(BUILD)/mutants.txt

waves_wide:
	$(RUN_W) +vectors=$(BUILD)/vectors_wide_directed.txt +bp_mode=1 \
	    +stall_mode=1 +g_mode=0 +seed=31 +dump | tee $(BUILD)/run_wide_waves.log

clean:
	rm -rf $(BUILD) dump_rms_wide.fst
