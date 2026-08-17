# =============================================================================
# verif/top/l4/Makefile — IB-LAYER L4 harness. S4 content: the LEVEL-
# PROPAGATION probe (tb_l4_levels — the FUEL UNDRIVEN lesson made a gate:
# every exported LAYER level observed at its CONSUMER, hierarchically).
# The full host-mode composition suite lands at S5 (gen_l4_vectors.py is
# the S1 skeleton; its op-stream emitter unstubs against the frozen §3b
# table).
#
# Targets: levels (build+run the probe) | vectors (S1 skeleton) | clean
# =============================================================================

SHELL     := /bin/bash
PYTHON    ?= python3
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
    $(REPO)/rtl/seq/seq_walker_pkg.sv \
    $(REPO)/rtl/seq/seq_walker.sv \
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

# apex_top is split out of RTL_CORE so the `mutate` target can substitute a
# patched copy (build/mut_<M>/apex_top.sv) in its place — the RTL is NEVER
# edited; mutants are patched COPIES (the l3/mutate.py discipline).
RTL = $(RTL_CORE) $(REPO)/rtl/top/apex_top.sv

# -Wno-PINCONNECTEMPTY is TB-scoped: the TBs deliberately leave unobserved
# DUT outputs open; the RTL's own -Wall-fatal gate is the smoke/l3 lint
# (UNDRIVEN etc. stay fatal there).
VOPTS = -Wall -Wno-PINCONNECTEMPTY --timing --assert --timescale 1ns/1ps \
        -I$(REPO)/rtl/rope -I$(REPO)/rtl/asu -I$(REPO)/verif/common \
        $(REPO)/verif/top/smoke/lint_waivers.vlt lint_waivers.vlt

.PHONY: levels vectors compose mutate clean

levels:
	mkdir -p $(BUILD)/obj_lvl
	$(VERILATOR) --binary $(VOPTS) --top-module tb_l4_levels \
	    -Mdir $(BUILD)/obj_lvl $(RTL) tb_l4_levels.sv
	set -o pipefail; $(BUILD)/obj_lvl/Vtb_l4_levels | tee $(BUILD)/run_levels.log

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_l4_vectors.py --outdir $(BUILD) | tee $(BUILD)/gen.log

# SEGMENT-2 L4 host-composition suite: RMSNorm-1 front-half through the REAL
# apex_top at CFG_D=128 (D_model), checked at the feeder-scale bus vs r.s_h.
# Depends on `vectors` (compose_seg2.ops). Machine rule: at most ONE Verilator
# build on this Mac — wait for any peer build before launching.
compose: vectors
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_cmp
	$(VERILATOR) --binary $(VOPTS) -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 \
	    --top-module tb_l4_compose -Mdir $(BUILD)/obj_cmp \
	    $(RTL) tb_l4_compose.sv
	set -o pipefail; $(BUILD)/obj_cmp/Vtb_l4_compose +vectors=$(BUILD)/compose_seg2.ops \
	    | tee $(BUILD)/run_compose.log

# SEGMENT-2 mutation gate: 3 front-half mutants, each caught ONLY on its
# specific FAIL signature (mutate_compose.py — the l3 signature-required
# discipline; the RTL is never edited, mutants are patched COPIES). Machine
# rule: serialize behind any peer Verilator build.
define RUN_MUT_CMP
	$(PYTHON) mutate_compose.py patch $(1) $(BUILD)/mut_$(1)
	until ! pgrep -f '[v]erilator_bin' >/dev/null; do sleep 15; done
	mkdir -p $(BUILD)/obj_mut_$(1)
	$(VERILATOR) --binary $(VOPTS) -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 \
	    --top-module tb_l4_compose -Mdir $(BUILD)/obj_mut_$(1) \
	    $(RTL_CORE) $(BUILD)/mut_$(1)/apex_top.sv tb_l4_compose.sv
	-$(BUILD)/obj_mut_$(1)/Vtb_l4_compose +vectors=$(BUILD)/compose_seg2.ops \
	    +watchdog=1000000 > $(BUILD)/run_mut_$(1).log 2>&1
	set -o pipefail; $(PYTHON) mutate_compose.py check $(1) \
	    $(BUILD)/run_mut_$(1).log | tee -a $(BUILD)/mutation.txt

endef

mutate: vectors
	rm -f $(BUILD)/mutation.txt
	$(call RUN_MUT_CMP,mC1)
	$(call RUN_MUT_CMP,mC2)
	$(call RUN_MUT_CMP,mC3)
	@set -o pipefail; grep -c "CAUGHT" $(BUILD)/mutation.txt | grep -q "^3$$" \
	    || { echo "L4 SEG2 MUTATION GATE: FAIL"; exit 1; }
	@echo "L4 SEG2 MUTATION GATE: PASS (3/3 front-half mutants caught)"

clean:
	rm -rf $(BUILD)
