# =============================================================================
# verif/kvq/sb/Makefile — independent KVQ verification suite (house pattern of
# verif/mxe/sb): vector-driven scoreboard TB with the golden arbiter
# golden/apex_golden, bound kvq_sb_sva §5 pack, adversarial protocol runs
# (backpressure storm / stalled feeds / malformed framing), mid-operation
# reset in every reachable FSM phase, manual coverage buckets (required holes
# fail the build) and mutation checks proving the checkers bite.
#
# Targets:
#   make all        - lint, vectors, build, full run matrix, coverage,
#                     mutations, final gate (exit != 0 on ANY failure)
#   make lint       - verilator --lint-only -Wall on rtl/kvq (independent rerun)
#   make vectors    - regenerate stimulus/expected from the golden model
#   make gate       - evaluate accumulated fail markers
#   make clean
#
# Every run tees to build/run_*.log and drops build/fail_<name> on failure
# instead of aborting make, so coverage aggregates ALL runs and `gate` gives
# one honest verdict at the end. Functional runs additionally grep-gate
# "SVA-VIOL" (D-012); the reset run intentionally does not (violations there
# are collected as DUT-bug evidence and already fail the TB).
#
# Waivers: reuses verif/kvq/smoke/lint_waivers.vlt (scoped per-line to the
# VENDORED rtl/kvq/vendor files only). No waivers on the TB or SVA pack.
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

# clean-room cores (rtl/kvq/cores); package FIRST. -Wall clean, no waivers.
# A2/D-026 added scale_bank_store.sv (persistent per-group scale rows).
CORE_FILES := cq_fp_pkg.sv cq_scale_unit.sv cq_quant_unit.sv \
              cq_dequant_unit.sv scale_bank.sv scale_bank_store.sv \
              cq_rne_div_pipe.sv cq_quant_pipe.sv cq_scale_pipe.sv \
              residual_buffer.sv \
              sram_controller.sv cq_value_path.sv cq_key_path.sv
KVQ_CORES  := $(addprefix $(REPO)/rtl/kvq/cores/, $(CORE_FILES))
KVQ_SRCS   := $(REPO)/rtl/kvq/kvq_engine.sv $(KVQ_CORES)
TB_SRCS    := tb_kvq_sb.sv kvq_sb_sva.sv

VOPTS = --binary --timing --assert -j 0 -Wall --timescale 1ns/1ps

CFG_cq4  := -GD=16 -GTIER=1 -GG=4 -GK_OUT=0 -GDEPTH=64 -GCFGNAME='"cq4"'
CFG_cq8  := -GD=16 -GTIER=0 -GG=2 -GK_OUT=0 -GDEPTH=64 -GCFGNAME='"cq8"'
CFG_cq4p := -GD=16 -GTIER=2 -GG=4 -GK_OUT=2 -GDEPTH=64 -GCFGNAME='"cq4p"' \
            -GMASKFILE='"$(BUILD)/vec_cq4p/mask.u8.hex"'
CFG_d64  := -GD=64 -GTIER=1 -GG=8 -GK_OUT=0 -GDEPTH=32 -GCFGNAME='"d64"'
# A2/D-026: the k=5 tier point (6.125 b/v at shipped G=128) — 5 fp16 outlier
# lanes at D=64 make KEY_REC_RAW (344) dominate the row (384b)
CFG_k5   := -GD=64 -GTIER=2 -GG=8 -GK_OUT=5 -GDEPTH=32 -GCFGNAME='"k5"' \
            -GMASKFILE='"$(BUILD)/vec_k5/mask.u8.hex"'

# ---- check-count derivation (A2/D-026 mirror update, anti-fabrication) --------
# Per-command TB check formulas (from tb_kvq_sb.sv tasks):
#   V / K(w<2) / G / F0 = 1     R = 2+2D     P = 1     E = 8     O = S = 1
#   B = 1 (bank row)   Q = 1 (SB_OVWR)   M: m1=5+2D m2=4+2D m3=6+2D
#   X k0 = 1+L+2+(4+2D) ; k1 = 1+L+S+(4+2D) ; k2 = 3+(4+2D) ; k3 = 4+(4+2D) ;
#     k4 = 1+L+2   with L(p)=1 (2 for p in {4,5,6,11}),
#     S = 2 parked / 4+2D outburst (p6/p11)      Y = 1 + G*(3+2D)
#   + 1 epilogue (stability-violation summary check) per run
# Formula table VALIDATED against the S3 baseline (build/s3_baseline/kvq_sb.log):
#   reset  = kind0 p1..p11 (11*39 + SUM L=15 -> 444) + kind1 (4*40 + 2*75 = 310)
#            + kind2 (39) + kind3 (40) + kind4 (4) + Y (141) + ep (1) = 979
#            = the exact pre-A2 logged count
#   framing = M1 (37) + M2 (36) + M3 (38) + tail V+P+R (36) + ep (1) = 148 ✓
# D-026 mirrors insert ONLY B and Q lines (generator RNG call order unchanged,
# so every pre-existing command survives 1:1). New expected counts:
#   cq4  main: 4309 + 7 B (checked groups) + 4 B (bank sweep) + 4 Q = 4324
#   cq8  main: 4737 + 0  (value-only: no key records, no bank)       = 4737
#   cq4p main: 1497 + 1 B + 4 B (bank sweep)                         = 1502
#   d64  main: 3708 + 3 B + 4 B (bank sweep)                         = 3715
#   reset    :  979 + 1 B (post-Y recommit row)                      =  980
#   framing  :  148                                                  =  148
#   k5   main (NEW config, D=64): sum of the formulas over the generated
#     script = 13 V + 52 K(w<2) + 5 F0 + 46 P + 46 R(2+2D=130 -> 5980)
#     + 1 E (8) + 7 B + 1 ep = 13+52+5+46+5980+8+7+1 = 6112
.PHONY: all lint vectors build_all gate coverage clean \
        build_cq4 build_cq8 build_cq4p build_d64 build_k5 \
        run_cq4 run_cq4_fast run_cq4_storm run_cq8 run_cq4p run_d64 run_k5 \
        run_framing run_reset mut_data mut_round mut_last mut_ssid mut_bank

all: lint vectors build_all \
     run_cq4 run_cq4_fast run_cq4_storm run_cq8 run_cq4p run_d64 run_k5 \
     run_framing run_reset coverage \
     mut_data mut_round mut_last mut_ssid mut_bank gate

lint:
	$(VERILATOR) --lint-only -Wall --timing \
	  --top-module kvq_engine $(KVQ_SRCS)
	@echo "LINT CLEAN (-Wall; rtl/kvq/cores is clean with NO waivers)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_kvq_sb_vectors.py $(BUILD) | tee $(BUILD)/gen.log
	@grep -q "GEN OK" $(BUILD)/gen.log

# ---- builds ------------------------------------------------------------------
define BUILD_CFG
build_$(1):
	mkdir -p $(BUILD)/obj_$(1)
	$$(VERILATOR) $$(VOPTS) --top-module tb_kvq_sb -Mdir $(BUILD)/obj_$(1) \
	  -o sim $$(CFG_$(1)) $$(KVQ_SRCS) $$(TB_SRCS)
endef
$(eval $(call BUILD_CFG,cq4))
$(eval $(call BUILD_CFG,cq8))
$(eval $(call BUILD_CFG,cq4p))
$(eval $(call BUILD_CFG,d64))
$(eval $(call BUILD_CFG,k5))

build_all: build_cq4 build_cq8 build_cq4p build_d64 build_k5

# ---- runs ----------------------------------------------------------------------
# RUN_CFG: 1=name 2=cfg 3=script 4=bp 5=stall 6=seed 7=gate_sva|no_sva_gate
define RUN_CFG
run_$(1):
	@mkdir -p $(BUILD)
	@set -o pipefail; \
	  ./$(BUILD)/obj_$(2)/sim +vecdir=$(BUILD)/vec_$(2) \
	    +script=$(BUILD)/vec_$(2)/script_$(3).txt \
	    +bp_mode=$(4) +stall_mode=$(5) +seed=$(6) 2>&1 \
	    | tee $(BUILD)/run_$(1).log \
	  || { echo "RUN FAIL [$(1)]"; touch $(BUILD)/fail_$(1); }
	@if [ "$(7)" = "gate_sva" ] && grep -q "SVA-VIOL" $(BUILD)/run_$(1).log; \
	  then echo "SVA GATE FAIL [$(1)]"; touch $(BUILD)/fail_$(1); fi
endef

$(eval $(call RUN_CFG,cq4,cq4,main,1,1,11,gate_sva))
$(eval $(call RUN_CFG,cq4_fast,cq4,main,0,0,12,gate_sva))
$(eval $(call RUN_CFG,cq4_storm,cq4,main,2,2,13,gate_sva))
$(eval $(call RUN_CFG,cq8,cq8,main,1,1,14,gate_sva))
$(eval $(call RUN_CFG,cq4p,cq4p,main,1,1,15,gate_sva))
$(eval $(call RUN_CFG,d64,d64,main,1,2,16,gate_sva))
$(eval $(call RUN_CFG,k5,k5,main,1,1,19,gate_sva))
$(eval $(call RUN_CFG,framing,cq4,framing,0,0,17,gate_sva))
# reset run: SVA/stability violations are DUT-bug EVIDENCE, gated via TB fails
$(eval $(call RUN_CFG,reset,cq4,reset,0,0,18,no_sva_gate))

# ---- coverage (required holes fail the build) ----------------------------------
coverage:
	@set -o pipefail; \
	  $(PYTHON) coverage_report.py $(BUILD)/run_*.log $(BUILD)/vec_*/cov_gen.txt \
	    $(BUILD)/cov_proofs.txt 2>&1 | tee $(BUILD)/coverage.txt \
	  || touch $(BUILD)/fail_coverage

# ---- mutation checks: prove the checkers catch injected bugs -------------------
# Each copies rtl/kvq into build/, injects one bug via sed (verified applied),
# rebuilds and reruns the directed cq4 config. The run MUST fail WITH the
# mutant-specific failure signature (house pattern: verif/top/l3/mutate.py
# check()): a PASS banner or clean exit = ESCAPED; a non-zero exit without
# "TB FAIL"/%Fatal/%Error = "run neither passed nor failed" (infra) = gate
# FAIL; a failure without the expected signature = gate FAIL. Every mut_*
# target depends on `vectors` so a standalone invocation can never score the
# TB's missing-+vecdir $fatal as a catch.
# (a) data corruption: flip payload bit 0 of every stored value record
# (b) KVQ contract violation — ROUNDING DRIFT: RNE tie -> round-half-up in the
#     shared quant/scale rounding core cq_fp_pkg::rne_ll (directed tie tokens
#     make this deterministic)
# (c) framing: tlast also asserted on beat 1 (must trip SVA ap_m_last_pos)
# (d) D-026 ssid allocator stuck: alloc_ptr never advances -> group 2+ tags
#     carry the wrong ssid AND every group commits onto set 0 (P + B + R bite)
# (e) D-026 bank corruption: scale_bank_store flips wr_row bit 0 on write ->
#     channel-0 group scale corrupt (B row compare + key readback R bite)
# Anchor audit vs the A2 engine rewrite (ce53188): (a) `sram_wr_data <=
# val_wr_data;` still unique (ST_STORE; the KEMIT store writes key_wr_data),
# (b) cq_fp_pkg.sv untouched by A2, (c) `m_axis_kv_tlast  <= (out_count ==
# LAST_ELEM);` still unique in ST_OUTPUT — none needed retargeting; the OLD
# key scatter block (fp16 field walk) is gone, replaced by g_lanes/g_nolanes,
# and no mutant anchored there. (d)/(e) are NEW anchors in the D-026 code.
COMMA := ,

# MUT_CHECK: 1=name 2=sed-old 3=sed-new 4=file 5=applied-grep 6=expected
# failure signature (ERE; the catch is only scored when this matches)
define MUT_CHECK
mut_$(1): vectors
	@rm -rf $(BUILD)/mut_$(1) $(BUILD)/obj_mut_$(1) $(BUILD)/fail_mut_$(1); \
	mkdir -p $(BUILD)/mut_$(1); \
	cp -r $(REPO)/rtl/kvq/. $(BUILD)/mut_$(1)/; \
	sed "s/$(2)/$(3)/" $(BUILD)/mut_$(1)/$(4) > $(BUILD)/mut_$(1)/$(4).tmp \
	  && mv $(BUILD)/mut_$(1)/$(4).tmp $(BUILD)/mut_$(1)/$(4); \
	if ! grep -q "$(5)" $(BUILD)/mut_$(1)/$(4); then \
	  echo "MUT [$(1)]: sed did not apply"; touch $(BUILD)/fail_mut_$(1); \
	elif ! $(VERILATOR) --binary --timing --assert -j 0 -Wno-fatal \
	    --timescale 1ns/1ps --top-module tb_kvq_sb \
	    -Mdir $(BUILD)/obj_mut_$(1) -o sim $(CFG_cq4) \
	    $(BUILD)/mut_$(1)/kvq_engine.sv \
	    $(addprefix $(BUILD)/mut_$(1)/cores/,$(CORE_FILES)) \
	    $(TB_SRCS) > $(BUILD)/build_mut_$(1).log 2>&1; then \
	  echo "MUT [$(1)]: mutant did not build (see build_mut_$(1).log)"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ./$(BUILD)/obj_mut_$(1)/sim +vecdir=$(BUILD)/vec_cq4 \
	    +script=$(BUILD)/vec_cq4/script_main.txt +bp_mode=0 +stall_mode=0 \
	    +seed=99 > $(BUILD)/run_mut_$(1).log 2>&1 \
	    || grep -q "TB PASS" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUTATION ESCAPED [$(1)] — checkers did NOT catch the injected bug"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ! grep -Eq "TB FAIL|%Fatal|%Error" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUT [$(1)]: run neither passed nor failed?! (infra error — see run_mut_$(1).log) Gate FAIL"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ! grep -Eq "$(6)" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUT [$(1)]: run FAILED but without the expected signature <$(6)> — inspect run_mut_$(1).log. Gate FAIL"; \
	  touch $(BUILD)/fail_mut_$(1); \
	else \
	  echo "MUTATION CAUGHT [$(1)] — signature evidence:"; \
	  grep -m 3 -E "$(6)" $(BUILD)/run_mut_$(1).log | sed 's/^/    /'; \
	fi
endef

$(eval $(call MUT_CHECK,data,sram_wr_data <= val_wr_data;,sram_wr_data <= val_wr_data ^ (SRAM_WIDTH'(1) << 24);,kvq_engine.sv,val_wr_data ^,R addr=.*: got|record mismatch))
$(eval $(call MUT_CHECK,round,else if ((r2 == dd) \&\& q\[0\]) q = q + 13'd1;,else if (r2 == dd) q = q + 13'd1;,cores/cq_fp_pkg.sv,else if (r2 == dd) q = q,R addr=.*: got|record mismatch|bank row mismatch))
$(eval $(call MUT_CHECK,last,m_axis_kv_tlast  <= (out_count == LAST_ELEM);,m_axis_kv_tlast  <= (out_count == LAST_ELEM) || (out_count == CNT_W'(1));,kvq_engine.sv,CNT_W'(1),tlast=. wrong|SVA-VIOL \[ap_m_last_pos\]))
$(eval $(call MUT_CHECK,ssid,alloc_ptr + 1'b1;,alloc_ptr;,kvq_engine.sv,<= alloc_ptr;,record mismatch|bank row mismatch|R addr=.*: got|SB_OVWR))
$(eval $(call MUT_CHECK,bank,mem\[wr_set\] <= wr_row;,mem[wr_set] <= wr_row ^ 1;,cores/scale_bank_store.sv,wr_row \^ 1,bank row mismatch|R addr=.*: got))

# ---- final gate ----------------------------------------------------------------
gate:
	@echo ""
	@echo "==================== KVQ SB GATE ===================="
	@for f in $(BUILD)/run_*.log; do \
	  echo "--- $$f"; \
	  grep -E "^(CONFIG|TB PASS|TB FAIL)|MUTATION|COVERAGE GATE" $$f | sed 's/^/    /' || true; \
	done
	@grep -E "COVERAGE GATE" $(BUILD)/coverage.txt | sed 's/^/    /' || true
	@if ls $(BUILD)/fail_* >/dev/null 2>&1; then \
	  echo "GATE: FAIL — markers: $$(cd $(BUILD) && ls fail_*)"; exit 1; \
	else \
	  echo "GATE: PASS — all runs, coverage and mutation checks clean"; \
	fi

clean:
	rm -rf $(BUILD) dump.fst
