# =============================================================================
# verif/top/qpath/Makefile — I-C IC-QPATH: the GAP A / GAP C closure gate.
#
#   make case     - render the three arms (self-checks Q1-Q5, no simulator)
#   make build    - the CFG_D=128 tile binary, driving the L3 TB
#   make run      - the SINK arm: MUST PASS, bit-exact vs decoder_layer_fx
#                   BUS_ON (D-030)
#   make discrim  - the two DISCRIMINATION arms: each MUST FAIL, on its own
#                   value signature (gap A = rotation, gap C = narrowing)
#   make mutate   - 3 signature-required mutants of the NEW routes
#   make all      - case + build + run + discrim + mutate
#
# The TB is verif/top/l3/tb_apex_l3.sv, UNCHANGED (the wg3 idiom: this lane
# builds it at its own -G point and never edits it).
#
# pipefail is load-bearing on every gate line (the tee lesson).
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON      ?= python3
REPO        := ../../..
include $(REPO)/verif/tools.mk
BUILD       := build
TB_L3       := $(REPO)/verif/top/l3
NAME        := qpath_l4_h1_hd128

# the tile file list is the L3 suite's RTL_CORE, SCRAPED rather than copied so
# the two cannot drift (the wg3 idiom)
TILE_RTL := $(shell sed -n '/^RTL_CORE = /,/apex_scale_quant.sv/p' \
              $(TB_L3)/Makefile | grep -o '\$$(REPO)/[a-z0-9_/]*\.sv' \
              | sed 's|\$$(REPO)|$(REPO)|')

VOPTS = -Wall --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu \
        -I$(TB_L3) $(TB_L3)/lint_waivers.vlt

.PHONY: all case build run discrim mutate lint clean

all: case build run discrim mutate

lint:
	@test -n "$(TILE_RTL)" || { echo "lint: RTL_CORE scrape came back EMPTY — refusing to lint nothing"; exit 1; }
	$(VERILATOR) --lint-only $(VOPTS) -GCFG_D=128 \
	    --top-module apex_top $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv
	@echo "QPATH LINT CLEAN (CFG_D=128)"

case:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_qpath_case.py --build $(BUILD) \
	    --arm sink | tee $(BUILD)/case_sink.log
	@grep -q "QPATH CASE: PASS" $(BUILD)/case_sink.log \
	    || { echo "QPATH CASE: FAIL"; exit 1; }
	set -o pipefail; $(PYTHON) gen_qpath_case.py --build $(BUILD) \
	    --arm norope   > $(BUILD)/case_norope.log 2>&1
	set -o pipefail; $(PYTHON) gen_qpath_case.py --build $(BUILD) \
	    --arm nonarrow > $(BUILD)/case_nonarrow.log 2>&1

$(BUILD)/obj_q128/Vtb_apex_l3: $(TB_L3)/tb_apex_l3.sv $(REPO)/rtl/top/apex_top.sv $(TILE_RTL)
	mkdir -p $(BUILD)/obj_q128
	$(VERILATOR) --binary $(VOPTS) -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 \
	    -GOUTK=0 --top-module tb_apex_l3 -Mdir $(BUILD)/obj_q128 \
	    $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv

build: $(BUILD)/obj_q128/Vtb_apex_l3

# ── the SINK arm: gap A + gap C closed, bit-exact vs the signed arbiter ─────
run: build case
	set -o pipefail; $(BUILD)/obj_q128/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(NAME)_sink.txt +watchdog=40000000 \
	    | tee $(BUILD)/run_sink.log
	@grep -q "L3 PASS" $(BUILD)/run_sink.log \
	    || { echo "QPATH SINK: FAIL"; exit 1; }
	@echo "QPATH SINK: PASS (q narrowed->rotated->quantized->staged, bit-exact vs decoder_layer_fx BUS_ON)"

# ── the DISCRIMINATION arms: same drive, one composition step removed from
#    the EXPECTATION. Each must be RED, on a value signature — if either goes
#    green the gate no longer tests that step.
discrim: build case
	-$(BUILD)/obj_q128/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(NAME)_norope.txt +watchdog=40000000 \
	    > $(BUILD)/run_norope.log 2>&1
	@grep -q "L3 PASS" $(BUILD)/run_norope.log \
	    && { echo "GAP-A DISCRIMINATION: un-rotated arm PASSED — the gate has lost its power"; exit 1; } \
	    || true
	@grep -q "tap f16" $(BUILD)/run_norope.log \
	    || { echo "GAP-A DISCRIMINATION: arm failed, but NOT on the tap signature — not a discrimination"; exit 1; }
	@echo "GAP-A DISCRIMINATION: PASS (un-rotated arm RED on the post-RoPE tap)"
	@grep -m1 "tap f16" $(BUILD)/run_norope.log
	-$(BUILD)/obj_q128/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(NAME)_nonarrow.txt +watchdog=40000000 \
	    > $(BUILD)/run_nonarrow.log 2>&1
	@grep -q "L3 PASS" $(BUILD)/run_nonarrow.log \
	    && { echo "GAP-C DISCRIMINATION: un-narrowed arm PASSED — the gate has lost its power"; exit 1; } \
	    || true
	@grep -q "tap f16" $(BUILD)/run_nonarrow.log \
	    || { echo "GAP-C DISCRIMINATION: arm failed, but NOT on the tap signature — not a discrimination"; exit 1; }
	@echo "GAP-C DISCRIMINATION: PASS (un-narrowed arm RED on the post-RoPE tap)"
	@grep -m1 "tap f16" $(BUILD)/run_nonarrow.log

# ── signature-required mutants of the NEW routes ────────────────────────────
define RUN_MUT
	$(PYTHON) mutate_qpath.py patch $(1) $(BUILD)/mut_$(1)
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 \
	    -GOUTK=0 --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mut_$(1) \
	    $(TILE_RTL) $(BUILD)/mut_$(1)/apex_top.sv $(TB_L3)/tb_apex_l3.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(NAME)_sink.txt +watchdog=40000000 \
	    > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate_qpath.py check $(1) \
	    $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt
	@# the mutant OBJECT tree is disposable once its log is captured; the
	@# patched source and the run log — the evidence — are kept.
	rm -rf $(BUILD)/obj_mut_$(1)
endef

mutate: build case
	rm -f $(BUILD)/mutation.txt
	$(call RUN_MUT,Q1)
	$(call RUN_MUT,Q2)
	$(call RUN_MUT,Q3)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mutation.txt | grep -q "^3$$" \
	    || { echo "QPATH MUTATION GATE: FAIL"; exit 1; }
	@echo "QPATH MUTATION GATE: PASS (3/3 q-sink mutants caught on their own signatures)"

clean:
	rm -rf $(BUILD)

# ═══════════════════════════════════════════════════════════════════════════
# F6 part (ii) — the MULTI-HEAD fmt=1 walk (LEVEL_C_INTEGRATION.md §9.1 F6).
#
# Build point: the W-G3 I-B CL point plus this lane's two additive knobs —
#   QSTAGE_H_MAX = NH   the per-head s_q file depth (default 1 = pre-I-C)
#   (STAGE_R_MAX is already 31 in the L3 TB — one q row per head fits)
# Everything else (CFG_D=128, GQANE=4, DEPTH=256, RMS/LAY 3584) is W-G3's.
#
#   make mh_case   render the multi-head case + its fmt=1 image
#   make mh_host   HOST arm: the NH q rows staged through the gap-A sink,
#                  every post-RoPE beat and every per-head s_q golden-gated
#   make mh_walk   WALKED arm: the same image driven by the WALKER
#   make mh_noq    DISCRIMINATION: the same case on a QSTAGE_H_MAX=1 tile
#                  (the pre-I-C single sticky). MUST FAIL.
#   make mh_mutate      Q4, the act-stage LOAD base row, against the HOST arm
#   make mh_mutate_walk X1/X2/X3, the three routes only the WALKED multi-head
#                  arm exercises. MUST all be RED on their OWN signature.
#   make mh        all of the above (KVG=0). `make mh KVG=1` re-runs the same
#                  machinery with every head in ONE KV group.
#
# I-C IC-MHWALK (this lane) closed the last blocker: the walk-start route
# level clear in seq_layer_walker's W_CHECK was a constraint-(d) violation
# that only a MULTI-head walk can reach, and it mis-routed each head's last
# PV result beat into the next head's score dequant. See the route-level note
# in rtl/seq/seq_layer_walker.sv. The walked arm additionally now carries the
# host generator's EFS expectations per head (340 feeder scales/head), which
# is what makes the per-head KV-group read path directly, not indirectly,
# gated — mutant X3 is caught by exactly those.
# ═══════════════════════════════════════════════════════════════════════════
NH       ?= 4
# KVG=0 (default): the NH heads sit one per KV group, so the walk crosses
# all NH GQA engines. KVG=1: all NH heads live in KV group 0 — same
# per-head machinery, ONE engine, which separates the per-head q/s_q
# staging from the per-engine read path.
KVG      ?= 0
MH_KV    := $(if $(filter 1,$(KVG)),1,$(NH))
MH_BASE  := qmh_s019_L19_nh$(NH)_g$(MH_KV)_hd128_T20
MH_HOST  := $(MH_BASE)_host
MH_WALK  := $(MH_BASE)_walk
MHOPTS   = -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 -GOUTK=0 -GGQANE=4 \
           -GRMSDM=3584 -GLAYDM=3584

.PHONY: mh_case mh_host mh_walk mh_noq mh

mh: mh_case mh_host mh_walk mh_noq mh_mutate mh_mutate_walk

mh_case:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_qmh_case.py --build $(BUILD) \
	    --heads $(NH) --kv-groups $(KVG) --mode host \
	    | tee $(BUILD)/mh_case.log
	set -o pipefail; $(PYTHON) gen_qmh_case.py --build $(BUILD) \
	    --heads $(NH) --kv-groups $(KVG) --mode walk \
	    >> $(BUILD)/mh_case.log 2>&1
	@grep -q "QMH CASE: PASS" $(BUILD)/mh_case.log \
	    || { echo "QMH CASE: FAIL"; exit 1; }

$(BUILD)/obj_mh/Vtb_apex_l3: $(TB_L3)/tb_apex_l3.sv $(REPO)/rtl/top/apex_top.sv $(TILE_RTL)
	mkdir -p $(BUILD)/obj_mh
	$(VERILATOR) --binary $(VOPTS) $(MHOPTS) -GQSTAGE=$(NH) \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mh \
	    $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv

mh_host: mh_case $(BUILD)/obj_mh/Vtb_apex_l3
	set -o pipefail; $(BUILD)/obj_mh/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(MH_HOST).txt +watchdog=60000000 \
	    | tee $(BUILD)/run_mh_host.log
	@grep -q "L3 PASS" $(BUILD)/run_mh_host.log \
	    || { echo "QMH HOST: FAIL"; exit 1; }
	@echo "QMH HOST: PASS ($(NH) real 7B heads staged through the gap-A sink)"

mh_walk: mh_case $(BUILD)/obj_mh/Vtb_apex_l3
	set -o pipefail; $(BUILD)/obj_mh/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(MH_WALK).txt +walker=1 +walkfmt2=1 \
	    +walkdesc=$(BUILD)/walker/$(MH_WALK).desc \
	    +walkdesc2=$(BUILD)/walker/$(MH_WALK).desc2 +watchdog=60000000 \
	    | tee $(BUILD)/run_mh_walk.log
	@grep -q "L3 PASS" $(BUILD)/run_mh_walk.log \
	    && grep -q "WALKFMT2: fmt=1 WALKED clean" $(BUILD)/run_mh_walk.log \
	    || { echo "QMH WALK: FAIL"; exit 1; }
	@echo "QMH WALK: PASS (MULTI-HEAD fmt=1 walk, n_heads=$(NH), real 7B geometry)"

mh_noq: mh_case
	mkdir -p $(BUILD)/obj_mh_noq
	$(VERILATOR) --binary $(VOPTS) $(MHOPTS) -GQSTAGE=1 \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mh_noq \
	    $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv
	-$(BUILD)/obj_mh_noq/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(MH_WALK).txt +walker=1 +walkfmt2=1 \
	    +walkdesc=$(BUILD)/walker/$(MH_WALK).desc \
	    +walkdesc2=$(BUILD)/walker/$(MH_WALK).desc2 +watchdog=60000000 \
	    > $(BUILD)/run_mh_noq.log 2>&1
	@grep -q "L3 PASS" $(BUILD)/run_mh_noq.log \
	    && { echo "F6(ii) DISCRIMINATION: the QSTAGE_H_MAX=1 tile PASSED a multi-head walk — the gate has lost its power"; exit 1; } \
	    || echo "F6(ii) DISCRIMINATION: PASS (pre-I-C single-sticky tile RED as required)"
	rm -rf $(BUILD)/obj_mh_noq

# Q4 mutates apex_stage_buf.sv (not apex_top.sv) and is gated by the
# MULTI-HEAD host arm's per-head readback, so it gets its own recipe: the
# patched copy REPLACES the stage buffer in the file list, by value.
mh_mutate: mh_case $(BUILD)/obj_mh/Vtb_apex_l3
	rm -f $(BUILD)/mh_mutation.txt
	$(PYTHON) mutate_qpath.py patch Q4 $(BUILD)/mut_Q4
	mkdir -p $(BUILD)/obj_mut_Q4
	$(VERILATOR) --binary $(VOPTS) $(MHOPTS) -GQSTAGE=$(NH) \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mut_Q4 \
	    $(subst $(REPO)/rtl/top/glue/apex_stage_buf.sv,$(BUILD)/mut_Q4/apex_stage_buf.sv,$(TILE_RTL)) \
	    $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv
	-$(BUILD)/obj_mut_Q4/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(MH_HOST).txt +watchdog=60000000 \
	    > $(BUILD)/run_mut_Q4.log 2>&1
	set -o pipefail; $(PYTHON) mutate_qpath.py check Q4 \
	    $(BUILD)/run_mut_Q4.log | tee -a $(BUILD)/mh_mutation.txt
	rm -rf $(BUILD)/obj_mut_Q4
	@grep -q "CAUGHT" $(BUILD)/mh_mutation.txt \
	    || { echo "QMH MUTATION GATE: FAIL"; exit 1; }
	@echo "QMH MUTATION GATE: PASS (act-stage LOAD base row caught on ERO)"

# ── I-C IC-MHWALK: the WALKED-arm mutation gate ────────────────────────────
# X1/X2/X3 mutate rtl/seq (not apex_top), so like Q4 the patched copy REPLACES
# its file in the list by VALUE. Each is run against the WALKED 4-head case
# and must fail on its OWN measured signature — never on "some non-zero exit".
define RUN_MUT_MH
	$(PYTHON) mutate_qpath.py patch $(1) $(BUILD)/mut_$(1)
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) $(MHOPTS) -GQSTAGE=$(NH) \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mut_$(1) \
	    $(subst $(REPO)/rtl/seq/$(2),$(BUILD)/mut_$(1)/$(2),$(TILE_RTL)) \
	    $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(MH_WALK).txt +walker=1 +walkfmt2=1 \
	    +walkdesc=$(BUILD)/walker/$(MH_WALK).desc \
	    +walkdesc2=$(BUILD)/walker/$(MH_WALK).desc2 +watchdog=60000000 \
	    > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate_qpath.py check $(1) \
	    $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mhwalk_mutation.txt
	rm -rf $(BUILD)/obj_mut_$(1)
endef

.PHONY: mh_mutate_walk

mh_mutate_walk: mh_case $(BUILD)/obj_mh/Vtb_apex_l3
	rm -f $(BUILD)/mhwalk_mutation.txt
	$(call RUN_MUT_MH,X1,seq_layer_walker.sv)
	$(call RUN_MUT_MH,X2,seq_layer_walker2.sv)
	$(call RUN_MUT_MH,X3,seq_layer_walker2.sv)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mhwalk_mutation.txt \
	    | grep -q "^3$$" \
	    || { echo "QMH WALK MUTATION GATE: FAIL"; exit 1; }
	@echo "QMH WALK MUTATION GATE: PASS (3/3 multi-head walk mutants caught on their own signatures)"
