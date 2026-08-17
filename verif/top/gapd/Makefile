# =============================================================================
# verif/top/gapd/Makefile — I-C GAP-D (IB_LAYER.md §3c-3): the CFG_D/CFG_DM
# width-family split gate.
#
#   make lint       - apex_top -Wall (warnings fatal) at BOTH directions:
#                     the two default single-width points (64, 128) and the
#                     SPLIT point (CFG_D=64 head_dim, CFG_DM=128 D_model),
#                     plus the W-G3 wide/GQA point
#   make lint7b     - NON-GATING probe: the 7B split point (128/3584) — must
#                     REFUSE today on the feeder/stage-buf D-021 legality
#                     (the stage-6 wide-feeder coupling, measured)
#   make vectors    - golden-gated split-build op streams for the FIVE L4
#                     cases GAP D blocked (gen_gapd_case.py self-checks every
#                     expectation against decoder_layer_fx BUS_ON first)
#   make build      - tb_apex_l3 at the split point (+ the PROJ_BIAS_EN=1
#                     sibling for l4_bias); the TB is verif/top/l3's,
#                     UNCHANGED (the wg3/qpath idiom: own -G point, no edits)
#   make run        - the five cases, each "L3 PASS" bit-exact vs the arbiter
#   make compose    - IB-LAYER's OWN segment-2 L4 suite (tb_l4_compose +
#                     compose_seg2.ops, both unchanged) on the SPLIT tile
#   make mutate     - 3 signature-required mutants proving CFG_DM is
#                     load-bearing in BOTH directions (patched COPIES; the
#                     RTL is never edited)
#   make all        - lint vectors build run compose mutate
#
# Machine rule (LEVEL_C_INTEGRATION.md §9.1): at most ONE Verilator build at
# a time — every build waits on `pgrep -f '[v]erilator_bin'`. pipefail is
# load-bearing on every gate line (GNU make 3.81 ignores .SHELLFLAGS).
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build
TB_L3     := $(REPO)/verif/top/l3
TB_L4     := $(REPO)/verif/top/l4

# the tile file list is the L3 suite's RTL_CORE, SCRAPED rather than copied
# so the two cannot drift (the wg3/qpath idiom)
RTL_CORE := $(shell sed -n '/^RTL_CORE = /,/apex_scale_quant.sv/p' \
              $(TB_L3)/Makefile | grep -o '\$$(REPO)/[a-z0-9_/]*\.sv' \
              | sed 's|\$$(REPO)|$(REPO)|')
RTL       = $(RTL_CORE) $(REPO)/rtl/top/apex_top.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu \
        -I$(TB_L3) $(TB_L3)/lint_waivers.vlt

# the split build point: CFG_D=64 (head_dim of the five blocked L4 cases),
# CFG_DM=128 (their D_model), CQ-8 only
SPLIT_G = -GCFG_D=64 -GCFG_DM=128 -GG_CFG=16 -GDEPTH=256 -GOUTK=0
# apex_top's own parameter names for the direct lint elaborations
LINT_SPLIT = -GCFG_D=64 -GCFG_DM=128 -GKVQ_G=16 -GKVQ_DEPTH=256
LINT_WIDE  = -GCFG_D=128 -GKVQ_GQA_NENG=4 -GKVQ_DEPTH=256 \
             -GRMS_D_MAX=3584 -GLAYER_DM_MAX=3584
LINT_7B    = -GCFG_D=128 -GCFG_DM=3584 -GRMS_D_MAX=3584 -GLAYER_DM_MAX=3584

CASES      = l4_h2_hd64 l4_gqa_h2kv1 l4_qwen_theta l4_selfinc
BIAS_CASE  = l4_bias

.PHONY: all lint lint7b vectors build build_bias run compose mutate clean

all: lint vectors build run compose mutate

define LINT_AT
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	$(VERILATOR) --lint-only $(VOPTS) $(1) --top-module apex_top $(RTL)

endef

lint:
	@test -n "$(RTL_CORE)" || { echo "lint: RTL_CORE scrape came back EMPTY — refusing to lint nothing"; exit 1; }
	$(call LINT_AT,-GCFG_D=64)
	$(call LINT_AT,-GCFG_D=128)
	$(call LINT_AT,$(LINT_SPLIT))
	$(call LINT_AT,$(LINT_WIDE))
	@echo "GAPD LINT CLEAN (-Wall warnings-fatal): default d64, default d128, SPLIT 64/128, W-G3 wide/GQA"

# NON-GATING measured probe: 7B model width refuses on the D-021 legality of
# the feeder and stage buffers — the stage-6 wide-feeder coupling. rc!=0 is
# the EXPECTED result; the gate is that the refusal is THESE $errors.
lint7b:
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	-$(VERILATOR) --lint-only $(VOPTS) $(LINT_7B) --top-module apex_top \
	    $(RTL) > $(BUILD)/lint7b.log 2>&1; true
	@set -o pipefail; grep -c "must be 64 or 128" $(BUILD)/lint7b.log \
	    | grep -q "^[1-9]" \
	    || { echo "LINT7B: expected the D-021 legality refusal, got something else"; \
	         tail -20 $(BUILD)/lint7b.log; exit 1; }
	@echo "LINT7B: 7B split point (128/3584) REFUSED on feeder/stage-buf D-021 legality (stage-6 wide-feeder item) — measured, as designed"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_gapd_case.py --build $(BUILD) \
	    | tee $(BUILD)/gen.log
	set -o pipefail; $(PYTHON) gen_gapd_case.py --build $(BUILD) --bias \
	    | tee -a $(BUILD)/gen.log
	@set -o pipefail; grep -c "GAPD CASE: PASS" $(BUILD)/gen.log \
	    | grep -q "^2$$" || { echo "GAPD VECTORS: FAIL"; exit 1; }

$(BUILD)/obj_split/Vtb_apex_l3: $(TB_L3)/tb_apex_l3.sv $(REPO)/rtl/top/apex_top.sv $(RTL_CORE)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_split
	$(VERILATOR) --binary $(VOPTS) $(SPLIT_G) --top-module tb_apex_l3 \
	    -Mdir $(BUILD)/obj_split $(RTL) $(TB_L3)/tb_apex_l3.sv

$(BUILD)/obj_bias/Vtb_apex_l3: $(TB_L3)/tb_apex_l3.sv $(REPO)/rtl/top/apex_top.sv $(RTL_CORE)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_bias
	$(VERILATOR) --binary $(VOPTS) $(SPLIT_G) -GPBIAS=1 \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_bias \
	    $(RTL) $(TB_L3)/tb_apex_l3.sv

build: $(BUILD)/obj_split/Vtb_apex_l3
build_bias: $(BUILD)/obj_bias/Vtb_apex_l3

run: vectors build build_bias
	@rm -f $(BUILD)/run_summary.txt
	@for c in $(CASES); do \
	  set -o pipefail; \
	  $(BUILD)/obj_split/Vtb_apex_l3 \
	      +vectors=$(BUILD)/cases/gapd_$$c.txt +watchdog=40000000 \
	      2>&1 | tee $(BUILD)/run_$$c.log | grep -E "L3 RESULT|L3 PASS" \
	      | sed "s/^/[$$c] /" | tee -a $(BUILD)/run_summary.txt \
	      || { echo "[$$c] FAIL"; exit 1; }; \
	done
	@set -o pipefail; \
	$(BUILD)/obj_bias/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/gapd_$(BIAS_CASE).txt +watchdog=40000000 \
	    2>&1 | tee $(BUILD)/run_$(BIAS_CASE).log \
	    | grep -E "L3 RESULT|L3 PASS" | sed "s/^/[$(BIAS_CASE)] /" \
	    | tee -a $(BUILD)/run_summary.txt \
	    || { echo "[$(BIAS_CASE)] FAIL"; exit 1; }
	@set -o pipefail; grep -c "L3 PASS" $(BUILD)/run_summary.txt \
	    | grep -q "^5$$" || { echo "GAPD RUN: FAIL"; exit 1; }
	@echo "GAPD RUN: PASS (5/5 previously-blocked L4 cases bit-exact on the SPLIT tile: K-rope/V/q-rope per head vs decoder_layer_fx BUS_ON)"

# IB-LAYER's own segment-2 suite (generator AND TB unchanged) on the split
# tile: proves the model-wide family (XR rows, feeder C-1, act stage) rides
# CFG_DM. Reuses ../l4's vectors; only the -G point differs (-GCFG_HD=64).
compose:
	$(MAKE) -C $(TB_L4) vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_cmp_split
	$(VERILATOR) --binary -Wall -Wno-PINCONNECTEMPTY --timing --assert \
	    --timescale 1ns/1ps -I$(REPO)/rtl/rope -I$(REPO)/rtl/asu \
	    -I$(REPO)/verif/common $(REPO)/verif/top/smoke/lint_waivers.vlt \
	    $(TB_L4)/lint_waivers.vlt -GCFG_D=128 -GCFG_HD=64 -GG_CFG=16 \
	    -GDEPTH=256 --top-module tb_l4_compose -Mdir $(BUILD)/obj_cmp_split \
	    $(RTL) $(TB_L4)/tb_l4_compose.sv
	set -o pipefail; $(BUILD)/obj_cmp_split/Vtb_l4_compose \
	    +vectors=$(TB_L4)/$(BUILD)/compose_seg2.ops \
	    | tee $(BUILD)/run_compose_split.log
	@set -o pipefail; grep -q "L4COMPOSE PASS" $(BUILD)/run_compose_split.log \
	    || { echo "GAPD COMPOSE-SPLIT: FAIL"; exit 1; }
	@echo "GAPD COMPOSE-SPLIT: PASS (IB-LAYER segment-2 suite, unchanged, on the CFG_D=64/CFG_DM=128 tile)"

# ── signature-required mutants (l3 discipline: a mutant counts ONLY on its
#    own FAIL signature, never any nonzero exit; patched COPIES) ────────────
define RUN_MUT
	$(PYTHON) mutate_gapd.py patch $(1) $(BUILD)/mut_$(1)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) $(SPLIT_G) --top-module tb_apex_l3 \
	    -Mdir $(BUILD)/obj_mut_$(1) $(RTL_CORE) $(BUILD)/mut_$(1)/apex_top.sv \
	    $(TB_L3)/tb_apex_l3.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/gapd_l4_h2_hd64.txt +watchdog=4000000 \
	    > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate_gapd.py check $(1) \
	    $(BUILD)/run_mut_$(1).log $(BUILD)/cases/gapd_l4_h2_hd64.json \
	    | tee -a $(BUILD)/mutation.txt
	rm -rf $(BUILD)/obj_mut_$(1)

endef

mutate: build vectors
	rm -f $(BUILD)/mutation.txt
	$(call RUN_MUT,mDM1)
	$(call RUN_MUT,mDM2)
	$(call RUN_MUT,mDM3)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mutation.txt | grep -q "^3$$" \
	    || { echo "GAPD MUTATION GATE: FAIL"; exit 1; }
	@echo "GAPD MUTATION GATE: PASS (3/3 width-family mutants caught on their own signatures)"

clean:
	rm -rf $(BUILD)
