# =============================================================================
# verif/top/l3/Makefile — Layer-3: named + random attention cases replayed
# through the REAL apex_top (host-sequenced v0.1 boundary), bit-exact vs
# golden/apex_golden/attention.py, plus a 3-mutant tile-level mutation gate.
#
#   make all      - lint + vectors + build (D=64 and D=128 tiles) + F-5
#                   stage-buf unit regression + all manifest cases +
#                   coverage gate + mutation gate
#   make waves CASE=<name>  - rerun one case with FST dump
#   make clean
#
# Feasibility register (F-1..F-5; F-1/F-5 CLOSED 2026-07-09 with
# kept-passing regressions) lives in gen_l3_vectors.py + manifest.json.
# pipefail is load-bearing on every gate line (the tee lesson).
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

RTL_CORE = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/misc/f16_arith_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/xbr/stream_skid.sv \
    $(REPO)/rtl/mxe/mxe_pe.sv \
    $(REPO)/rtl/mxe/mxe_array.sv \
    $(REPO)/rtl/mxe/mxe_buf.sv \
    $(REPO)/rtl/mxe/mxe_requant.sv \
    $(REPO)/rtl/mxe/mxe_ctrl.sv \
    $(REPO)/rtl/mxe/mxe_top.sv \
    $(REPO)/rtl/asu/asu_exp_lut.sv \
    $(REPO)/rtl/asu/asu_softmax.sv \
    $(REPO)/rtl/asu/rsqrt.sv \
    $(REPO)/rtl/asu/asu_rmsnorm.sv \
    $(REPO)/rtl/rope/rope_pair_fx.sv \
    $(REPO)/rtl/rope/rope_row.sv \
    $(REPO)/rtl/top/glue/apex_layer_deq.sv \
    $(REPO)/rtl/top/glue/apex_residual.sv \
    $(REPO)/rtl/asu/asu_swiglu.sv \
    $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
    $(REPO)/rtl/kvq/cores/cq_scale_unit.sv \
    $(REPO)/rtl/kvq/cores/cq_quant_unit.sv \
    $(REPO)/rtl/kvq/cores/cq_dequant_unit.sv \
    $(REPO)/rtl/kvq/cores/scale_bank.sv \
    $(REPO)/rtl/kvq/cores/scale_bank_store.sv \
    $(REPO)/rtl/kvq/cores/cq_rne_div_pipe.sv \
    $(REPO)/rtl/kvq/cores/cq_quant_pipe.sv \
    $(REPO)/rtl/kvq/cores/cq_scale_pipe.sv \
    $(REPO)/rtl/kvq/cores/residual_buffer.sv \
    $(REPO)/rtl/kvq/cores/sram_controller.sv \
    $(REPO)/rtl/kvq/cores/cq_value_path.sv \
    $(REPO)/rtl/kvq/cores/cq_key_path.sv \
    $(REPO)/rtl/kvq/kvq_engine.sv \
    $(REPO)/rtl/top/glue/apex_kvq_bank.sv \
    $(REPO)/rtl/top/glue/apex_kvq_gqa_bank.sv \
    $(REPO)/rtl/top/glue/apex_wcomp_bank.sv \
    $(REPO)/rtl/tip/tip_decide.sv \
    $(REPO)/rtl/tip/tip_importance.sv \
    $(REPO)/rtl/tip/tip_top.sv \
    $(REPO)/rtl/seq/seq_walker.sv \
    $(REPO)/rtl/seq/seq_walker_pkg.sv \
    $(REPO)/rtl/seq/seq_walker_comp.sv \
    $(REPO)/rtl/seq/seq_layer_walker.sv \
    $(REPO)/rtl/seq/seq_layer_walker2.sv \
    $(REPO)/rtl/csr/csr_regs.sv \
    $(REPO)/rtl/seam/seam_feeder_quant.sv \
    $(REPO)/rtl/seam/seam_score_dequant.sv \
    $(REPO)/rtl/top/glue/apex_q78_to_fp32.sv \
    $(REPO)/rtl/top/glue/apex_lane32_ser.sv \
    $(REPO)/rtl/top/glue/apex_lane8_unpack.sv \
    $(REPO)/rtl/top/glue/apex_gam_unpack.sv \
    $(REPO)/rtl/top/glue/apex_stage_buf.sv \
    $(REPO)/rtl/top/glue/apex_score_fork.sv \
    $(REPO)/rtl/top/glue/apex_proj_bias.sv \
    $(REPO)/rtl/top/glue/apex_scale_quant.sv

RTL      = $(RTL_CORE) $(REPO)/rtl/top/apex_top.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/rope \
        -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. lint_waivers.vlt

.PHONY: all lint vectors composite build unit run coverage mutate \
        walkdesc run_walker walkdesc2 run_walkfmt2 waves clean

all: lint vectors composite build unit run coverage walkdesc run_walker \
     mutate

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module apex_top $(RTL)
	@echo "LINT CLEAN (-Wall; waivers scoped to frozen apex_pkg + vendored files only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_l3_vectors.py $(BUILD) | tee $(BUILD)/gen.log

# B1 stage-0 gate (no Verilator, ~2 s): re-derive every scale composite in the
# op stream from the scales the stream itself publishes, and re-project each
# case to its canonical per-channel control trace. Catches drift between the
# golden/spec and the walker's arithmetic oracle. The self-test enforces a
# COVERAGE floor, so a broken provenance resolver fails instead of passing empty.
composite:
	set -o pipefail; $(PYTHON) walker_composite_golden.py $(BUILD) \
	    | tee $(BUILD)/composite.log
	set -o pipefail; $(PYTHON) extract_trace.py $(BUILD) \
	    | tee $(BUILD)/trace_scope.txt
	@grep -q "COMPOSITE REPLICA: PASS" $(BUILD)/composite.log \
	    || { echo "B1 COMPOSITE GATE: FAIL"; exit 1; }
	@echo "B1 COMPOSITE GATE: PASS"

build:
	mkdir -p $(BUILD)/obj_b64 $(BUILD)/obj_b128
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GCFG_D=64 -GG_CFG=16 \
	    -GDEPTH=256 -GOUTK=2 -GMASKF='"$(BUILD)/mask_l3_b64.hex"' \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_b64 \
	    $(RTL) tb_apex_l3.sv
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GCFG_D=128 -GG_CFG=16 \
	    -GDEPTH=256 -GOUTK=2 --top-module tb_apex_l3 -Mdir $(BUILD)/obj_b128 \
	    $(RTL) tb_apex_l3.sv

# F-5 directed unit regression: apex_stage_buf @ D=128, all PAT_D column
# blocks 0..15 byte-exact + nb=BPR=16 legality. Discrimination proven
# 2026-07-09 against a reverted-index mutant (fails at sel=8 beat=0).
unit:
	mkdir -p $(BUILD)/obj_unit
	$(VERILATOR) --binary $(VOPTS) --top-module tb_stagebuf_patd \
	    -Mdir $(BUILD)/obj_unit $(REPO)/rtl/apex_pkg.sv \
	    $(REPO)/rtl/xbr/stream_skid.sv \
	    $(REPO)/rtl/top/glue/apex_stage_buf.sv tb_stagebuf_patd.sv
	set -o pipefail; $(BUILD)/obj_unit/Vtb_stagebuf_patd \
	    | tee $(BUILD)/run_unit_stagebuf.log
	@grep -q "STAGEBUF PATD PASS" $(BUILD)/run_unit_stagebuf.log \
	    || { echo "F-5 UNIT REGRESSION: FAIL"; exit 1; }

run:
	set -o pipefail; $(PYTHON) run_cases.py $(BUILD) | tee $(BUILD)/run_all.txt

coverage:
	set -o pipefail; $(PYTHON) coverage_report.py $(BUILD) | tee $(BUILD)/coverage.txt

# ── B1 stage 5: walker-mode L3 ─────────────────────────────────────────────
# The compact layer descriptor a real host loads once per step. A NEW sibling
# of gen_l3_vectors.py - the spec generator is never edited (B1_WALKER.md §7).
walkdesc:
	set -o pipefail; $(PYTHON) gen_walker_desc.py $(BUILD) \
	    | tee $(BUILD)/walkdesc.log

# The §A-1 gate: every CQ-8 case must WALK (a silent host fallback is a FAIL),
# grouped-tier cases must be REFUSED with WALK_ERR_TIER and still pass in host
# mode, no case may LOSE checks vs host mode, and the tier split is PRINTED.
# `run` (host mode) stays byte-identical and is the compat proof.
run_walker: walkdesc build
	set -o pipefail; $(PYTHON) run_walker_cases.py $(BUILD) \
	    | tee $(BUILD)/run_walker_all.txt
	@grep -q "WALKER-MODE L3: all cases passed" $(BUILD)/run_walker_all.txt \
	    || { echo "WALKER-MODE L3: FAIL"; exit 1; }

# ── IB-WALK stage 4 (D-029 glue): the grown-window probe ────────────────────
# Separate target so the 27-case walker-mode counts stay untouched. One CQ-8
# case runs with +walkfmt: FMT_SUP read, deep DPTR, tile-level fmt=1 refusal
# + W1C, then the normal walk must still pass bit-exact.
run_walkfmt: walkdesc build
	set -o pipefail; $(BUILD)/obj_b64/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/rand_d64_T8.txt +walker=1 +walkfmt=1 \
	    +walkdesc=$(BUILD)/walker/rand_d64_T8.desc +watchdog=12000000 \
	    | tee $(BUILD)/run_walkfmt.log
	@grep -q "L3 PASS" $(BUILD)/run_walkfmt.log \
	    && grep -q "WALKFMT: FMT_SUP=0011" $(BUILD)/run_walkfmt.log \
	    || { echo "WALKFMT: FAIL"; exit 1; }

# ── W-G2 first rung (+walkfmt2): the first WALKABLE single-engine fmt=1
# tile case. gen_walkfmt2_desc.py compiles rand_d64_T8's golden-gated v1
# descriptor into a mirror-checked, per-head-v1-EQUIVALENT fmt=1 image
# (l64-class: n_heads=1, n_kv_heads=1, head_dim=64); the case then runs the
# FULL walker-mode scoreboard through the fmt=1 window path — deep SRAM
# words 21/22 get their first direct read via the walk. Separate target so
# the 28-case walker-mode counts stay untouched (the §B-1 rule).
walkdesc2: walkdesc
	set -o pipefail; $(PYTHON) gen_walkfmt2_desc.py $(BUILD) \
	    | tee $(BUILD)/walkdesc2.log

run_walkfmt2: walkdesc2 build
	set -o pipefail; $(BUILD)/obj_b64/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/rand_d64_T8.txt +walker=1 +walkfmt2=1 \
	    +walkdesc=$(BUILD)/walker/rand_d64_T8.desc \
	    +walkdesc2=$(BUILD)/walker/rand_d64_T8.desc2 +watchdog=12000000 \
	    | tee $(BUILD)/run_walkfmt2.log
	@grep -q "L3 PASS" $(BUILD)/run_walkfmt2.log \
	    && grep -q "WALKFMT2: fmt=1 WALKED clean" $(BUILD)/run_walkfmt2.log \
	    || { echo "WALKFMT2: FAIL"; exit 1; }

# ── mutation gate: patched COPIES of apex_top.sv (rtl/ never touched) ───────
define RUN_MUT
	$(PYTHON) mutate.py patch $(1) $(BUILD)/mut_$(1)
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) --trace-fst -GCFG_D=64 -GG_CFG=16 \
	    -GDEPTH=256 -GOUTK=2 -GMASKF='"$(BUILD)/mask_l3_b64.hex"' \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_mut_$(1) \
	    $(RTL_CORE) $(BUILD)/mut_$(1)/apex_top.sv tb_apex_l3.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(2).txt +watchdog=12000000 \
	    > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate.py check $(1) \
	    $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt

endef

mutate:
	rm -f $(BUILD)/mutation.txt
	$(call RUN_MUT,M1,rand_d64_T8)
	$(call RUN_MUT,M2,adv_T1)
	$(call RUN_MUT,M3,rand_d64_T8)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mutation.txt | grep -q "^3$$" \
	    || { echo "MUTATION GATE: FAIL"; exit 1; }
	@echo "MUTATION GATE: PASS (3/3 tile-level mutants caught)"

# make waves CASE=<name> [BIN=b64|b128]  (FST evidence; e.g. the F-5 fix
#   regression: make waves CASE=bug_d128_stagebuf_nb BIN=b128 — now expected
#   to PASS end-to-end; the pre-fix wedge ('drive_g stall', aj_nb=16
#   truncated to 0) is preserved in logs/prefix_f5a_keptfail_before_fix.log)
BIN ?= b64
waves:
	-$(BUILD)/obj_$(BIN)/Vtb_apex_l3 +vectors=$(BUILD)/cases/$(CASE).txt \
	    +watchdog=60000000 +dump | tee $(BUILD)/run_$(CASE)_waves.log
	mv dump.fst $(BUILD)/$(CASE).fst
	@echo "FST: $(BUILD)/$(CASE).fst"

clean:
	rm -rf $(BUILD) dump.fst
