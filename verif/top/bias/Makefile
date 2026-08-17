# =============================================================================
# verif/top/bias/Makefile — IC-BIAS (I-C gap B) suite.
#
#   make vectors   - golden-gated stimulus (gen_bias_vectors.py self-checks
#                    every expectation against decoder_layer_fx BUS_ON first)
#   make unit      - apex_proj_bias unit suite, co-simulated against the REAL
#                    apex_scale_quant MODE_F16
#   make tile      - the l4_bias q/K/V projections through a real apex_top
#                    (PROJ_BIAS_EN=1), checked at the dbg_f16_* seam
#   make lint      - -Wall (warnings fatal) with the path ENABLED, at the tile
#                    build point and at the wide I-B CL build point
#   make mutate    - signature-required mutation gate (l3 discipline: a mutant
#                    counts only on its OWN FAIL signature, never any nonzero
#                    exit; the RTL is never edited, mutants are patched COPIES)
#   make all       - lint vectors unit tile mutate
#
# Machine rule (LEVEL_C_INTEGRATION.md §9.1): at most ONE Verilator build at a
# time across lanes — every build waits on `pgrep -f '[v]erilator_bin'`.
# pipefail is load-bearing on every gate line (the tee lesson).
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_CORE := $(shell sed -n '/^RTL_CORE = /,/apex_scale_quant.sv/p' \
              ../l3/Makefile | grep -o '\$$(REPO)/[a-z0-9_/]*\.sv' \
              | sed 's|\$$(REPO)|$(REPO)|')
RTL       = $(RTL_CORE) $(REPO)/rtl/top/apex_top.sv

UNIT_RTL = $(REPO)/rtl/apex_pkg.sv $(REPO)/rtl/misc/f16_arith_pkg.sv \
           $(REPO)/rtl/xbr/stream_skid.sv \
           $(REPO)/rtl/top/glue/apex_proj_bias.sv \
           $(REPO)/rtl/top/glue/apex_scale_quant.sv

VOPTS = -Wall -Wno-PINCONNECTEMPTY --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/rtl/rope -I$(REPO)/rtl/asu -I$(REPO)/verif/common \
        $(REPO)/verif/top/smoke/lint_waivers.vlt lint_waivers.vlt

# the tile build point: CFG_D=128 (D_model of the L4 cases), CQ-8 KVQ,
# STAGE_R_MAX >= T+1, and the bias path ENABLED.
TILE_G = -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 -GPROJ_BIAS_EN=1 \
         -GLAYER_DM_MAX=128 -GFEED_ROWS_MAX=31 -GSTAGE_R_MAX=31
# the same build point expressed in apex_top's OWN parameter names (the lint
# elaborates apex_top directly, not the TB wrapper)
LINT_G = -GCFG_D=128 -GKVQ_G=16 -GKVQ_DEPTH=256 -GPROJ_BIAS_EN=1 \
         -GLAYER_DM_MAX=128 -GFEED_ROWS_MAX=31 -GSTAGE_R_MAX=31

.PHONY: all lint vectors unit tile mutate clean

all: lint vectors unit tile mutate

# the ENABLED-path elaboration lint, warnings fatal, at both build points
lint:
	@test -n "$(RTL_CORE)" || { echo "lint: RTL_CORE scrape from ../l3/Makefile came back EMPTY — refusing to lint nothing"; exit 1; }
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	$(VERILATOR) --lint-only -Wall --timing --assert --timescale 1ns/1ps \
	    -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. \
	    ../l3/lint_waivers.vlt $(LINT_G) --top-module apex_top $(RTL)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	$(VERILATOR) --lint-only -Wall --timing --assert --timescale 1ns/1ps \
	    -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. \
	    ../l3/lint_waivers.vlt -GCFG_D=128 -GKVQ_DEPTH=256 -GKVQ_GQA_NENG=4 \
	    -GRMS_D_MAX=3584 -GLAYER_DM_MAX=3584 -GPROJ_BIAS_EN=1 \
	    --top-module apex_top $(RTL)
	@echo "IC-BIAS LINT CLEAN (-Wall, warnings fatal; PROJ_BIAS_EN=1 at the tile and wide-CL build points)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_bias_vectors.py --outdir $(BUILD) \
	    | tee $(BUILD)/gen.log
	@grep -q "IC-BIAS VECTORS: PASS" $(BUILD)/gen.log \
	    || { echo "IC-BIAS VECTOR GATE: FAIL"; exit 1; }

unit: vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_unit
	$(VERILATOR) --binary $(VOPTS) --top-module tb_pbias_unit \
	    -Mdir $(BUILD)/obj_unit $(UNIT_RTL) tb_pbias_unit.sv
	set -o pipefail; $(BUILD)/obj_unit/Vtb_pbias_unit \
	    +vectors=$(BUILD)/pbias_unit.vec | tee $(BUILD)/run_unit.log
	@grep -q "PBIAS PASS" $(BUILD)/run_unit.log \
	    || { echo "IC-BIAS UNIT GATE: FAIL"; exit 1; }

tile: vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_tile
	$(VERILATOR) --binary $(VOPTS) $(TILE_G) --top-module tb_bias_tile \
	    -Mdir $(BUILD)/obj_tile $(RTL) tb_bias_tile.sv
	set -o pipefail; $(BUILD)/obj_tile/Vtb_bias_tile \
	    +vectors=$(BUILD)/bias_tile.ops | tee $(BUILD)/run_tile.log
	@grep -q "BIASTILE PASS" $(BUILD)/run_tile.log \
	    || { echo "IC-BIAS TILE GATE: FAIL"; exit 1; }

# ── mutation gate: patched COPIES (rtl/ never touched), each caught ONLY on
#    its own FAIL signature — the l3/mutate.py discipline. Both mutable files
#    (apex_proj_bias.sv, apex_top.sv) are substituted from the mutant dir so
#    one recipe serves mutants in either file.
MUT_RTL = $(filter-out %/apex_proj_bias.sv,$(RTL_CORE))

define RUN_MUT
	$(PYTHON) mutate_bias.py patch $(1) $(BUILD)/mut_$(1)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) $(TILE_G) --top-module tb_bias_tile \
	    -Mdir $(BUILD)/obj_mut_$(1) \
	    $(MUT_RTL) $(BUILD)/mut_$(1)/apex_proj_bias.sv \
	    $(BUILD)/mut_$(1)/apex_top.sv tb_bias_tile.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_bias_tile +vectors=$(BUILD)/bias_tile.ops \
	    +watchdog=4000000 > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate_bias.py check $(1) \
	    $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt
	rm -rf $(BUILD)/obj_mut_$(1)

endef

mutate: vectors
	rm -f $(BUILD)/mutation.txt
	$(call RUN_MUT,mB1)
	$(call RUN_MUT,mB2)
	$(call RUN_MUT,mB3)
	$(call RUN_MUT,mB4)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mutation.txt | grep -q "^4$$" \
	    || { echo "IC-BIAS MUTATION GATE: FAIL"; exit 1; }
	@echo "IC-BIAS MUTATION GATE: PASS (4/4 bias-path mutants caught)"

clean:
	rm -rf $(BUILD)
