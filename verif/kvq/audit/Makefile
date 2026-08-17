# =============================================================================
# verif/kvq/audit/Makefile — adversarial re-audit of the D-020-fixed KVQ
# (attacks the B-1..B-4 fixes of verif/kvq/RESULT.md specifically).
#
# Targets:
#   make all   - lint, vectors, builds, reset-attack matrix (>=260 randomized
#                soft resets across every FSM state x pending-beat x mid-token
#                x mid-group x collision conditions, each followed by a clean
#                job checked bit-exact vs golden/apex_golden), -0.0 directed
#                runs (D=16 and D=64, outlier AND non-outlier), coverage gate,
#                fix-revert mutation checks (B-1 required; B-2/B-4 bonus),
#                final gate (exit != 0 on ANY failure)
#   make clean
#
# Every audit run (INCLUDING the reset attacks — the fixed RTL must be
# violation-free under reset storm) grep-gates "SVA-VIOL" (D-012); the §5 pack
# ../sb/kvq_sb_sva.sv rides along in every build via bind.
# Waivers: verif/kvq/smoke/lint_waivers.vlt (per-line, rtl/kvq/vendor only).
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON ?= python3
REPO      := ../../..
include $(REPO)/verif/tools.mk
BUILD     := build

# clean-room cores, package FIRST (repaired: the KVQ_CORES opener had been
# clobbered by a stale vendored-list remnant — Makefile was unparseable)
KVQ_CORES := \
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
$(REPO)/rtl/kvq/cores/cq_key_path.sv
KVQ_SRCS   := $(REPO)/rtl/kvq/kvq_engine.sv $(KVQ_CORES)
TB_SRCS    := tb_kvq_audit.sv ../sb/kvq_sb_sva.sv
WAIVERS    := $(REPO)/verif/kvq/smoke/lint_waivers.vlt

VOPTS = --binary --timing --assert -j 0 -Wall --timescale 1ns/1ps $(WAIVERS)

CFG_rst_cq4  := -GD=16 -GTIER=1 -GG=4 -GK_OUT=0 -GDEPTH=64 -GNV=24 -GNK=8 \
                -GCFGNAME='"rst_cq4"'
CFG_rst_cq4p := -GD=16 -GTIER=2 -GG=4 -GK_OUT=2 -GDEPTH=64 -GNV=12 -GNK=8 \
                -GCFGNAME='"rst_cq4p"' -GMASKFILE='"$(BUILD)/vec_rst_cq4p/mask.u8.hex"'
CFG_rst_d64  := -GD=64 -GTIER=1 -GG=8 -GK_OUT=0 -GDEPTH=32 -GNV=8 -GNK=4 \
                -GCFGNAME='"rst_d64"'
CFG_rst_cq8  := -GD=16 -GTIER=0 -GG=2 -GK_OUT=0 -GDEPTH=64 -GNV=12 -GNK=0 \
                -GCFGNAME='"rst_cq8"'
CFG_negz_d64p := -GD=64 -GTIER=2 -GG=8 -GK_OUT=4 -GDEPTH=32 -GNV=6 -GNK=4 \
                -GCFGNAME='"negz_d64p"' -GMASKFILE='"$(BUILD)/vec_negz_d64p/mask.u8.hex"'

.PHONY: all lint vectors build_all gate coverage clean \
        build_rst_cq4 build_rst_cq4p build_rst_d64 build_rst_cq8 build_negz_d64p \
        run_rst_cq4 run_rst_cq4p run_rst_d64 run_rst_cq8 run_rst_d64p \
        run_negz_cq4p run_negz_d64p mut_b1 mut_b2 mut_b4

all: lint vectors build_all \
     run_rst_cq4 run_rst_cq4p run_rst_d64 run_rst_cq8 run_rst_d64p \
     run_negz_cq4p run_negz_d64p coverage mut_b1 mut_b2 mut_b4 gate

lint:
	$(VERILATOR) --lint-only -Wall --timing $(WAIVERS) \
	  --top-module kvq_engine $(KVQ_SRCS)
	@echo "LINT CLEAN (-Wall; waivers scoped per-line to rtl/kvq/vendor only)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_audit_vectors.py $(BUILD) | tee $(BUILD)/gen.log
	@grep -q "GEN OK" $(BUILD)/gen.log

# ---- builds ------------------------------------------------------------------
define BUILD_CFG
build_$(1):
	mkdir -p $(BUILD)/obj_$(1)
	$$(VERILATOR) $$(VOPTS) --top-module tb_kvq_audit -Mdir $(BUILD)/obj_$(1) \
	  -o sim $$(CFG_$(1)) $$(KVQ_SRCS) $$(TB_SRCS)
endef
$(eval $(call BUILD_CFG,rst_cq4))
$(eval $(call BUILD_CFG,rst_cq4p))
$(eval $(call BUILD_CFG,rst_d64))
$(eval $(call BUILD_CFG,rst_cq8))
$(eval $(call BUILD_CFG,negz_d64p))

build_all: build_rst_cq4 build_rst_cq4p build_rst_d64 build_rst_cq8 build_negz_d64p

# ---- runs ----------------------------------------------------------------------
# RUN: 1=name 2=build 3=vecdir 4=mode 5=n_rst 6=bp 7=stall 8=seed
#
# A2/D-026 check-count derivation (old -> new; baseline = build/s3_baseline):
#  * key-record peek: OLD do_peek = 1 check (whole record: 1 tag + D fp16
#    fields + D codes in one image compare). NEW do_kpeek = 2 checks:
#    [tag bit0==1 && tag[7:1]==live ssid] + [image with tag[7:1] masked ==
#    golden: k fp16 lanes + D codes (outlier sentinels 4'd1) + pad].
#  * per committed group: NEW do_bankpeek = D checks (nk keep scales + k
#    outlier lanes pinned 16'h0000) vs the golden pack_key_records bank row.
#  * clean_kjob checks: OLD 1 + G*(1 + (2D+2))  ->  NEW 1 + G*(2 + (2D+2)) + D.
#  * NEW bank_reset_attack (N_BANK=6 per mode-0 KEYG run) = 2 x the kjob body
#    (pre- and post-ctrl_reset) = 2*(1 + G*(2D+4) + D) checks; adds per event:
#    +1 reset, +2G peeks, +2G reads, +2 bank peeks.
#  * NEW sb_ovwr_check (1 per mode-0 KEYG run) = 13 checks (6 flushed-partial
#    commits x 1 wait_idle + 7 IRQ_STATUS/W1C/irq-line probes); no reads/peeks.
#  * summary counters (reads = peeks = 1 prologue + clean_v + G*clean_k +
#    2*G*N_BANK + 1 epilogue; resets = n_rst + N_BANK; clean_v/clean_k
#    unchanged — the new phases run after the attack loop and draw no rnd()):
#      rst_cq4  : resets 140->146, reads/peeks 466 + 2*4*6 = 514, bank_peeks  81+12=93
#      rst_cq4p : resets  40->46,  reads/peeks 150 + 2*4*6 = 198, bank_peeks  27+12=39
#      rst_d64  : resets  40->46,  reads/peeks 242 + 2*8*6 = 338, bank_peeks  25+12=37
#      rst_cq8  : unchanged 20/20/0/22/22 (KEYG=0: no bank/SB_OVWR phases)
#      rst_d64p : resets  20->26,  reads/peeks 110 + 2*8*6 = 206, bank_peeks  11+12=23
#      negz_cq4p (mode1): 45/45 unchanged, bank_peeks = clean_k = 8
#      negz_d64p (mode1): 39/39 unchanged, bank_peeks = clean_k = 4
define RUN_CFG
run_$(1):
	@mkdir -p $(BUILD)
	@rm -f $(BUILD)/fail_$(1)
	@set -o pipefail; \
	  ./$(BUILD)/obj_$(2)/sim +vecdir=$(BUILD)/vec_$(3) +mode=$(4) \
	    +n_rst=$(5) +bp_mode=$(6) +stall_mode=$(7) +seed=$(8) 2>&1 \
	    | tee $(BUILD)/run_$(1).log \
	  || { echo "RUN FAIL [$(1)]"; touch $(BUILD)/fail_$(1); }
	@if grep -q "SVA-VIOL" $(BUILD)/run_$(1).log; \
	  then echo "SVA GATE FAIL [$(1)]"; touch $(BUILD)/fail_$(1); fi
endef

$(eval $(call RUN_CFG,rst_cq4,rst_cq4,rst_cq4,0,140,1,1,21))
$(eval $(call RUN_CFG,rst_cq4p,rst_cq4p,rst_cq4p,0,40,1,1,22))
$(eval $(call RUN_CFG,rst_d64,rst_d64,rst_d64,0,40,1,1,24))
$(eval $(call RUN_CFG,rst_cq8,rst_cq8,rst_cq8,0,20,1,1,25))
$(eval $(call RUN_CFG,rst_d64p,negz_d64p,negz_d64p,0,20,2,1,27))
# directed -0.0 runs (mode=1: every pool job once, D=16 and D=64 CQ-4+)
$(eval $(call RUN_CFG,negz_cq4p,rst_cq4p,rst_cq4p,1,0,1,1,23))
$(eval $(call RUN_CFG,negz_d64p,negz_d64p,negz_d64p,1,0,1,1,26))

# ---- coverage (required holes fail the build) ----------------------------------
coverage:
	@rm -f $(BUILD)/fail_coverage
	@set -o pipefail; \
	  $(PYTHON) audit_coverage.py $(BUILD)/run_*.log $(BUILD)/vec_*/cov_gen.txt \
	    2>&1 | tee $(BUILD)/coverage.txt \
	  || touch $(BUILD)/fail_coverage

# ---- fix-revert mutation checks -------------------------------------------------
# Each copies rtl/kvq into build/, REVERTS one D-020 fix via patch_mutant.py
# (verified applied), rebuilds and reruns the matching audit run. It MUST fail
# WITH the mutant-specific failure signature (house pattern:
# verif/top/l3/mutate.py check()): a PASS banner or clean exit = ESCAPED; a
# non-zero exit without "TB FAIL"/%Fatal/%Error = "run neither passed nor
# failed" (infra) = gate FAIL; a failure without the expected signature =
# gate FAIL. Every mut_* target depends on `vectors` so a standalone
# invocation can never score the TB's missing-+vecdir $fatal as a catch.
#   b1: reintroduce the lost-soft-reset (case overrides ctrl_reset)  [REQUIRED]
#   b2: tie off dp_clear (datapaths keep walking through soft reset)
#   b4: drop the outlier sign force (-0.0 outlier reads back +0.0). D-026:
#       the sign source is now key_cur_scale (the stored raw fp16 lane on
#       outlier beats, ex-key_field_rd); the kp_dec_hat_sx declaration line
#       the anchor pins is unchanged — see patch_mutant.py.
# Mutant sources mirror KVQ_CORES from the copied tree (the old vendor/*.sv
# glob predates the clean-room cores/ move and matched nothing).
# MUT: 1=name 2=build-cfg 3=vecdir 4=mode 5=n_rst 6=bp 7=stall 8=seed
#      9=expected failure signature (ERE; the catch is only scored when it matches)
MUT_CORES := $(patsubst $(REPO)/rtl/kvq/%,%,$(KVQ_CORES))
define MUT_CHECK
mut_$(1): vectors
	@rm -rf $(BUILD)/mut_$(1) $(BUILD)/obj_mut_$(1) $(BUILD)/fail_mut_$(1); \
	mkdir -p $(BUILD)/mut_$(1); \
	cp -r $(REPO)/rtl/kvq/. $(BUILD)/mut_$(1)/; \
	if ! $(PYTHON) patch_mutant.py $(1) $(BUILD)/mut_$(1) \
	     > $(BUILD)/patch_mut_$(1).log 2>&1; then \
	  echo "MUT [$(1)]: patch did not apply"; touch $(BUILD)/fail_mut_$(1); \
	elif ! $(VERILATOR) --binary --timing --assert -j 0 -Wno-fatal \
	    --timescale 1ns/1ps --top-module tb_kvq_audit \
	    -Mdir $(BUILD)/obj_mut_$(1) -o sim $(CFG_$(2)) \
	    $(BUILD)/mut_$(1)/kvq_engine.sv \
	    $(addprefix $(BUILD)/mut_$(1)/,$(MUT_CORES)) \
	    $(TB_SRCS) > $(BUILD)/build_mut_$(1).log 2>&1; then \
	  echo "MUT [$(1)]: mutant did not build (see build_mut_$(1).log)"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ./$(BUILD)/obj_mut_$(1)/sim +vecdir=$(BUILD)/vec_$(3) +mode=$(4) \
	    +n_rst=$(5) +bp_mode=$(6) +stall_mode=$(7) +seed=$(8) \
	    > $(BUILD)/run_mut_$(1).log 2>&1 \
	    || grep -q "TB PASS" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUTATION ESCAPED [$(1)] — the audit did NOT catch the reverted fix"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ! grep -Eq "TB FAIL|%Fatal|%Error" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUT [$(1)]: run neither passed nor failed?! (infra error — see run_mut_$(1).log) Gate FAIL"; \
	  touch $(BUILD)/fail_mut_$(1); \
	elif ! grep -Eq "$(9)" $(BUILD)/run_mut_$(1).log; then \
	  echo "MUT [$(1)]: run FAILED but without the expected signature <$(9)> — inspect run_mut_$(1).log. Gate FAIL"; \
	  touch $(BUILD)/fail_mut_$(1); \
	else \
	  echo "MUTATION CAUGHT [$(1)] — signature evidence:"; \
	  grep -m 3 -E "$(9)" $(BUILD)/run_mut_$(1).log | sed 's/^/    /'; \
	fi
endef

$(eval $(call MUT_CHECK,b1,rst_cq4,rst_cq4,0,40,1,1,31,soft reset NOT honored|LEAKED after soft reset|idle flag LIES|never returned to IDLE))
$(eval $(call MUT_CHECK,b2,rst_cq4,rst_cq4,0,40,1,1,32,datapath still walking after soft reset|IDLE-LIE|idle lie))
$(eval $(call MUT_CHECK,b4,rst_cq4p,rst_cq4p,1,0,1,1,33,R addr=.*: got|record body mismatch|: scale .* exp))

# ---- final gate ----------------------------------------------------------------
gate:
	@echo ""
	@echo "==================== KVQ AUDIT GATE ===================="
	@for f in $(BUILD)/run_*.log; do \
	  echo "--- $$f"; \
	  grep -E "^(CONFIG|TB PASS|TB FAIL)|MUTATION" $$f | sed 's/^/    /' || true; \
	done
	@grep -E "COVERAGE GATE" $(BUILD)/coverage.txt | sed 's/^/    /' || true
	@if ls $(BUILD)/fail_* >/dev/null 2>&1; then \
	  echo "GATE: FAIL — markers: $$(cd $(BUILD) && ls fail_*)"; exit 1; \
	else \
	  echo "GATE: PASS — all runs, coverage and fix-revert mutations clean"; \
	fi

clean:
	rm -rf $(BUILD) dump.fst
