# =============================================================================
# verif/tip/smoke/Makefile — TIP smoke: lint + golden vectors + 7 runs.
#
# Targets:
#   make smoke   - lint, vectors, build both N configs, run everything (default)
#   make lint    - verilator --lint-only -Wall on the TIP RTL (tip_top)
#   make vectors - golden model + vector generation (gen_tip_vectors.py;
#                  also re-proves frozen-143 equivalence at generation time)
#   make clean
#
# Runs (each self-gates via the TB's $fatal / ALL TESTS PASSED):
#   run_replay   - V0.3 frozen 143-case upstream vectors at T=10, N=4096
#                  (upstream-equivalence replay), random gaps + backpressure
#   run_sweep_t1/t5/t31 - programmable-threshold sweeps at N=64 vs the golden
#   run_clamp_t0 - +threshold=0 vs T=1 golden (hardware clamp proof)
#   run_imp      - importance directed set (tier boundaries, saturation)
#   run_frame    - V0.3 F-2 framing-guard regression (N+1 scores -> error
#                  pulse + sticky, no decision, clean state after)
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the FROZEN rtl/apex_pkg.sv
# only. No waivers on any new TIP file, the SVA checker, or the TB.
# =============================================================================

SHELL     := /bin/bash
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_SRCS = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/tip/tip_decide.sv \
    $(REPO)/rtl/tip/tip_importance.sv \
    $(REPO)/rtl/tip/tip_top.sv

TB_SRCS = tb_tip_smoke.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -j 4 lint_waivers.vlt

RUN64   = set -o pipefail; $(BUILD)/obj_n64/Vtb_tip_smoke
RUN4096 = set -o pipefail; $(BUILD)/obj_n4096/Vtb_tip_smoke

.PHONY: smoke lint vectors build run_replay run_sweep_t1 run_sweep_t5 \
        run_sweep_t31 run_clamp_t0 run_imp run_frame clean

smoke: lint vectors build run_replay run_sweep_t1 run_sweep_t5 run_sweep_t31 \
       run_clamp_t0 run_imp run_frame
	@echo "==== TIP SMOKE: ALL TARGETS COMPLETE ===="

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module tip_top $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall, waivers scoped to frozen apex_pkg.sv only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_tip_vectors.py | tee $(BUILD)/vectors.log

build: $(BUILD)/obj_n64/Vtb_tip_smoke $(BUILD)/obj_n4096/Vtb_tip_smoke

$(BUILD)/obj_n64/Vtb_tip_smoke: $(RTL_SRCS) $(TB_SRCS) tip_sva.svh
	mkdir -p $(BUILD)/obj_n64
	$(VERILATOR) --binary $(VOPTS) --top-module tb_tip_smoke \
	    -GBLOCK_M=8 -GBLOCK_N=8 \
	    -Mdir $(BUILD)/obj_n64 $(RTL_SRCS) $(TB_SRCS)

$(BUILD)/obj_n4096/Vtb_tip_smoke: $(RTL_SRCS) $(TB_SRCS) tip_sva.svh
	mkdir -p $(BUILD)/obj_n4096
	$(VERILATOR) --binary $(VOPTS) --top-module tb_tip_smoke \
	    -GBLOCK_M=64 -GBLOCK_N=64 -GMAX_SCORES=600000 \
	    -Mdir $(BUILD)/obj_n4096 $(RTL_SRCS) $(TB_SRCS)

# frozen V0.3 143-case upstream replay at T=10 (equivalence), N=4096
run_replay:
	$(RUN4096) $$(cat $(BUILD)/selfcheck_t10.args) +gaps=1 +bp=1 +seed=11 \
	    | tee $(BUILD)/run_replay.log

# programmable-threshold sweeps vs the golden model, N=64
run_sweep_t1:
	$(RUN64) $$(cat $(BUILD)/sweep_t1.args) +gaps=1 +bp=1 +seed=21 \
	    | tee $(BUILD)/run_sweep_t1.log

run_sweep_t5:
	$(RUN64) $$(cat $(BUILD)/sweep_t5.args) +gaps=1 +bp=1 +seed=22 \
	    | tee $(BUILD)/run_sweep_t5.log

run_sweep_t31:
	$(RUN64) $$(cat $(BUILD)/sweep_t31.args) +gaps=1 +bp=1 +seed=23 \
	    | tee $(BUILD)/run_sweep_t31.log

# threshold=0 clamps to 1 (D-017 thr_eff): T=1 golden expectations
run_clamp_t0:
	$(RUN64) $$(cat $(BUILD)/clamp_t0.args) +gaps=0 +bp=1 +seed=24 \
	    | tee $(BUILD)/run_clamp_t0.log

# importance accumulator directed set (tier boundary equality, saturation)
run_imp:
	$(RUN64) $$(cat $(BUILD)/imp_directed.args) +gaps=0 +bp=1 +seed=25 \
	    | tee $(BUILD)/run_imp.log

# V0.3 F-2 framing-guard regression (directed, no vector files)
run_frame:
	$(RUN64) +frame_test=1 +threshold=10 +imp_hi=65535 +imp_lo=32768 \
	    +seed=26 +bp=1 | tee $(BUILD)/run_frame.log

clean:
	rm -rf $(BUILD)
