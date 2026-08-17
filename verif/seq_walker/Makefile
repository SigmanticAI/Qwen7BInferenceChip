# =============================================================================
# verif/seq_walker/Makefile — B1 layer-walker unit suite (house verif/seq
# pattern): vector-driven scoreboard against the L3-extracted reference trace
# (gen_walker_vectors.py -> extract_trace.py, the spec per OPTIMIZATION.md:38)
# + randomized job-consumer backpressure + a KVQ AXI-Lite slave stub that
# enforces the poll-before-RADDR invariant + the §A-1 refusal gate +
# mutation-kill checks.
#
# Targets:
#   make all      - lint, vectors, both builds, 5 run configs, mutants
#   make vectors  - project the L3 cases to reference traces (needs the L3
#                   vectors: make -C ../top/l3 vectors)
#   make mutants  - checker-credibility gate: every mutant run must FAIL
#                   with its expected signature (mutate.py check; infra
#                   failures are gate failures, vectors are prerequisites)
#
# Waivers: lint_waivers.vlt scopes UNUSEDPARAM to the frozen apex_pkg.sv and
# the vendored cq_fp_pkg.sv only. NO waivers on any B1-owned RTL or the TB.
# =============================================================================

SHELL     := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON    ?= python3
REPO      := ../..
include $(REPO)/verif/tools.mk
BUILD     := build

# mxe_cfg_pkg.sv is the TILE's implemented-capacity package. It is listed
# ahead of seq_walker_pkg (which imports M_TILE_MAX from it for the D-029
# erratum bounds) and read by both TBs' MXE-legality checks, so the check
# cannot drift from the consumer it models.
RTL_FILES = apex_pkg.sv kvq/cores/cq_fp_pkg.sv mxe/mxe_cfg_pkg.sv \
            seq/seq_walker_pkg.sv \
            seq/seq_walker_comp.sv seq/seq_layer_walker.sv \
            seq/seq_layer_walker2.sv
RTL_SRCS  = $(addprefix $(REPO)/rtl/,$(RTL_FILES))
TB_SRCS   = tb_walker_sb.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -j 4 \
        -I$(REPO)/verif/common -I. lint_waivers.vlt

.PHONY: all lint vectors build run_small run_split run_d128 run_fast \
        run_refuse comp_vectors comp_build run_comp64 run_comp128 \
        run_comp_dir mutants layer_vectors build2 run_l64 run_l128 run_l7b \
        run_l05b \
        run_qstage run_refuse2 mutants2 mutants4 mutants5 mutants6 \
        check_vectors build_chk run_check \
        mutants3 clean

all: lint vectors build run_small run_split run_d128 run_fast run_refuse \
     comp_vectors comp_build run_comp64 run_comp128 run_comp_dir mutants \
     layer_vectors build2 run_l64 run_l128 run_l7b run_l05b run_qstage \
     run_refuse2 \
     mutants2 mutants4 mutants5 mutants6 check_vectors build_chk run_check \
     mutants3
	@echo "==== B1 WALKER SUITE: ALL TARGETS COMPLETE ===="

lint:
	$(VERILATOR) --lint-only $(VOPTS) --top-module seq_layer_walker $(RTL_SRCS)
	$(VERILATOR) --lint-only $(VOPTS) --top-module seq_walker_comp $(RTL_SRCS)
	$(VERILATOR) --lint-only $(VOPTS) --top-module seq_layer_walker2 $(RTL_SRCS)
	@echo "LINT CLEAN (-Wall; waivers scoped to apex_pkg, vendored cq_fp_pkg, mxe_cfg_pkg)"

vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_walker_vectors.py | tee $(BUILD)/vectors.log

build: $(BUILD)/obj64/Vtb_walker_sb $(BUILD)/obj128/Vtb_walker_sb

$(BUILD)/obj64/Vtb_walker_sb: $(RTL_SRCS) $(TB_SRCS)
	mkdir -p $(BUILD)/obj64
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=64 --top-module tb_walker_sb \
	    -Mdir $(BUILD)/obj64 $(RTL_SRCS) $(TB_SRCS)

$(BUILD)/obj128/Vtb_walker_sb: $(RTL_SRCS) $(TB_SRCS)
	mkdir -p $(BUILD)/obj128
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=128 --top-module tb_walker_sb \
	    -Mdir $(BUILD)/obj128 $(RTL_SRCS) $(TB_SRCS)

define RUN64
	set -o pipefail; $(BUILD)/obj64/Vtb_walker_sb +vectors=$(BUILD)/vectors_$(1).txt \
	    +lat_mode=$(2) +bp_mode=$(3) +seed=$(4) | tee $(BUILD)/run_$(1)$(5).log
	@grep -q "WALKER PASS" $(BUILD)/run_$(1)$(5).log || { echo "RUN $(1)$(5): FAIL"; exit 1; }
endef

# small shapes, randomized backpressure + engine latency
run_small:
	$(call RUN64,small,1,1,401,)

# the split path (ceil(T/8) > BPR) — only 3 of 24 CQ-8 cases reach it
run_split:
	$(call RUN64,split,1,1,402,)

# same small set at full rate (zero stalls) — ordering must not depend on it
run_fast:
	$(call RUN64,small,0,0,403,_fast)

# §A-1: a grouped-tier descriptor must be REFUSED with WALK_ERR_TIER
run_refuse:
	$(call RUN64,refuse,1,1,404,)

# D=128 build: the only place the composite's RNE rounding is observable
run_d128:
	set -o pipefail; $(BUILD)/obj128/Vtb_walker_sb \
	    +vectors=$(BUILD)/vectors_d128.txt +lat_mode=2 +bp_mode=2 +seed=405 \
	    | tee $(BUILD)/run_d128.log
	@grep -q "WALKER PASS" $(BUILD)/run_d128.log || { echo "RUN d128: FAIL"; exit 1; }

# ── stage 3: composite-unit arithmetic vs the stage-0 oracle ───────────────
comp_vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_comp_vectors.py | tee $(BUILD)/comp_vectors.log

comp_build: $(BUILD)/cobj64/Vtb_comp_sb $(BUILD)/cobj128/Vtb_comp_sb

$(BUILD)/cobj64/Vtb_comp_sb: $(RTL_SRCS) tb_comp_sb.sv walker_comp_sva.svh
	mkdir -p $(BUILD)/cobj64
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=64 --top-module tb_comp_sb \
	    -Mdir $(BUILD)/cobj64 $(RTL_SRCS) tb_comp_sb.sv

$(BUILD)/cobj128/Vtb_comp_sb: $(RTL_SRCS) tb_comp_sb.sv walker_comp_sva.svh
	mkdir -p $(BUILD)/cobj128
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=128 --top-module tb_comp_sb \
	    -Mdir $(BUILD)/cobj128 $(RTL_SRCS) tb_comp_sb.sv

# exhaustive element sweep -> 23,377 distinct reachable scales
run_comp64:
	set -o pipefail; $(BUILD)/cobj64/Vtb_comp_sb \
	    +vectors=$(BUILD)/comp_sweep_d64.txt | tee $(BUILD)/run_comp64.log
	@grep -q "COMP PASS" $(BUILD)/run_comp64.log || { echo "COMP64: FAIL"; exit 1; }

run_comp128:
	set -o pipefail; $(BUILD)/cobj128/Vtb_comp_sb \
	    +vectors=$(BUILD)/comp_sweep_d128.txt | tee $(BUILD)/run_comp128.log
	@grep -q "COMP PASS" $(BUILD)/run_comp128.log || { echo "COMP128: FAIL"; exit 1; }

# directed corners: the f32-vs-f64 constant divergence (D=128 only)
run_comp_dir:
	set -o pipefail; $(BUILD)/cobj64/Vtb_comp_sb \
	    +vectors=$(BUILD)/comp_dir_d64.txt | tee $(BUILD)/run_compdir64.log
	@grep -q "COMP PASS" $(BUILD)/run_compdir64.log || { echo "COMPDIR64: FAIL"; exit 1; }
	set -o pipefail; $(BUILD)/cobj128/Vtb_comp_sb \
	    +vectors=$(BUILD)/comp_dir_d128.txt | tee $(BUILD)/run_compdir128.log
	@grep -q "COMP PASS" $(BUILD)/run_compdir128.log || { echo "COMPDIR128: FAIL"; exit 1; }

# ── IB-WALK stage 2: fmt=1 sequencer vs the layer-trace spec ────────────────
# The reference is golden-gated at generation time (gen_layer_trace.py
# asserts every tensor-dependent field against decoder_layer_fx), so a
# walker2-vs-spec mismatch is a WALKER bug by construction.
layer_vectors:
	mkdir -p $(BUILD)/layer
	set -o pipefail; $(PYTHON) gen_layer_trace.py | tee $(BUILD)/layer_trace.log

build2: $(BUILD)/obj64w2/Vtb_walker2_sb $(BUILD)/obj128w2/Vtb_walker2_sb

$(BUILD)/obj64w2/Vtb_walker2_sb: $(RTL_SRCS) tb_walker2_sb.sv
	mkdir -p $(BUILD)/obj64w2
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=64 --top-module tb_walker2_sb \
	    -Mdir $(BUILD)/obj64w2 $(RTL_SRCS) tb_walker2_sb.sv

$(BUILD)/obj128w2/Vtb_walker2_sb: $(RTL_SRCS) tb_walker2_sb.sv
	mkdir -p $(BUILD)/obj128w2
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=128 --top-module tb_walker2_sb \
	    -Mdir $(BUILD)/obj128w2 $(RTL_SRCS) tb_walker2_sb.sv

# $(1) vector file stem  $(2) obj dir  $(3) log stem  $(4-6) lat/bp/seed
define RUN2
	set -o pipefail; $(BUILD)/$(2)/Vtb_walker2_sb \
	    +vectors=$(BUILD)/layer/$(1) +lat_mode=$(4) +bp_mode=$(5) +seed=$(6) \
	    | tee $(BUILD)/run2_$(3).log
	@grep -q "WALKER2 PASS" $(BUILD)/run2_$(3).log \
	    || { echo "RUN2 $(3): FAIL"; exit 1; }
endef

# !! SECOND ERRATUM (2026-07-30 audit; RTL fix landed later the same day,
# !! R1): the spec chunks swiglu LAYER_JOBs at asu_swiglu's own COLS_MAX
# !! (64), and after B1's host fix these targets were deliberately RED —
# !! seq_layer_walker2 still chunked at WALK2_N_JOB (4095, the 12-bit
# !! field): `expected 'LUJOB 1 40' got 'LUJOB 1 <d_ffn or fff>'`. Before
# !! B1 BOTH sides used the field bound and this gate was GREEN on a stream
# !! apex_top would mis-deliver — the walker and its oracle agreeing on the
# !! same wrong constant, the D-029 blind spot one lane over. The R1 RTL
# !! fix (seq_walker_pkg walk2_lu_bound per-unit chunk + walker2 lu_bound;
# !! apex_top now REFUSES swiglu cols > COLS_MAX, err_code 2, instead of
# !! truncating) re-greened all three WITHOUT touching the expectations.
# !! The rule stands: DO NOT re-point the expectations at the walker.
run_l64:
	$(call RUN2,layer_h2_hd64.sub.ops,obj64w2,l64,1,1,501)

run_l128:
	$(call RUN2,layer_h1_hd128.sub.ops,obj128w2,l128,1,1,502)
	# E-3 (E2E_TOY_LANE.md §4): the SAME toy-128 geometry with the in-tile
	# norm-feed step armed (mask bit EN_NFEED). The legacy stream above is
	# byte-identical to its pre-E-3 self; this one adds exactly the three
	# emissions the walked residual->norm chain needs — the level word with
	# feeder source code 4 (which a 2-bit walker level could not encode),
	# the C-1 feeder job, and the LAYER_JOB unit-3 NORM/EGRESS push.
	$(call RUN2,layer_nf_hd128.sub.ops,obj128w2,l128nf,1,1,512)

# the 7B geometry: multi-group GQA (28/4, group 7), K-split accumulate
# chains (3584/18944), N-split (18944) — the shapes the m5/m6 mutants need
run_l7b:
	$(call RUN2,qwen7b_T8.sub.ops,obj128w2,l7b,1,1,503)

# THIRD ERRATUM (W1, 2026-08-05): the Qwen2.5-0.5B geometry at D=64 — the
# walk whose down projection (k_total = d_ffn = 4864) exercises the D-AWARE
# k chunk: k_job(64) = 1984 (31 whole stage rows), split [1984, 1984, 896]
# with accumulate 0/1/1. Before the erratum the walker chunked at K_JOB=2048
# = 32 stage rows > apex_stage_buf R_MAX(31) — a TILE-ILLEGAL act family on
# every D=64 build. The spec side is held by the mutants6 stageability gate.
run_l05b:
	$(call RUN2,layer_05b_hd64.sub.ops,obj64w2,l05b,1,1,531)

run_refuse2:
	$(call RUN2,refuse2.ops,obj64w2,refuse2,1,1,504)

# E-3b (E2E_TOY_LANE.md §4): the walker-staged-q cases — the E-4 walk shape
# {ROPE, STOREKV, QSTAGE, SCORE, PV} (+NFEED+NSRC at 128 — E-4b: the
# walker OWNS l_nsrc there, so the whole chain is ONE kick and the NFEED
# level word carries bit 19). The QSTAGE emissions
# are the HOST's own staging choreography, emitted by the sequencer: FJOB,
# MODE_F16 QJOB, the per-step composite xD, the serializer job (the NEW sj
# channel), D/8 x {loader-row act EMIT, k2 WS DESC}, act-bank-1 LOAD row h,
# with the q-sink level armed before and restored after.
run_qstage:
	$(call RUN2,qstage_h1_hd64.sub.ops,obj64w2,qs64,1,1,521)
	$(call RUN2,qstage_h1_hd128.sub.ops,obj128w2,qs128,2,2,522)

# ── checker-credibility mutations (scratch RTL copy; suite MUST fail) ───────
# Classification is signature-required (mutate.py check, the house
# verif/top/l3/mutate.py check() shape): a PASS banner, an infra failure
# (cannot-open / missing +vectors / watchdog), a crashed run with no banner,
# or a failure without the mutant's expected signature all FAIL the gate.
# $(1) mutant  $(2) vector set  $(3) build dir  $(4) -DCFG_D_TB
define MUT_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/mut_$(1)/rtl
	mkdir -p $(BUILD)/mut_$(1)/obj
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=$(4) --top-module tb_walker_sb \
	    -Mdir $(BUILD)/mut_$(1)/obj \
	    $(addprefix $(BUILD)/mut_$(1)/rtl/,$(RTL_FILES)) $(TB_SRCS)
	@$(BUILD)/mut_$(1)/obj/Vtb_walker_sb +vectors=$(BUILD)/vectors_$(2).txt \
	    +lat_mode=1 +bp_mode=1 +seed=991 > $(BUILD)/mut_$(1)/run.log 2>&1 || true
	@$(PYTHON) mutate.py check walk $(1) $(BUILD)/mut_$(1)/run.log
endef

# the same constant mutant, but against the EXHAUSTIVE composite sweep — an
# independent, far stronger kill than the 100-composite d128 walker set
define CMUT_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/cmut_$(1)/rtl
	mkdir -p $(BUILD)/cmut_$(1)/obj
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=$(3) --top-module tb_comp_sb \
	    -Mdir $(BUILD)/cmut_$(1)/obj \
	    $(addprefix $(BUILD)/cmut_$(1)/rtl/,$(RTL_FILES)) tb_comp_sb.sv
	@$(BUILD)/cmut_$(1)/obj/Vtb_comp_sb +vectors=$(BUILD)/$(2) \
	    > $(BUILD)/cmut_$(1)/run.log 2>&1 || true
	@$(PYTHON) mutate.py check comp $(1) $(BUILD)/cmut_$(1)/run.log
endef

mutants: vectors comp_vectors
	$(call MUT_RULE,m1_kvskip,small,mut,64)
	$(call MUT_RULE,m3_pvorder,split,mut,64)
	$(call MUT_RULE,m2_narrow,d128,mut,128)
	$(call CMUT_RULE,m2_narrow,comp_sweep_d128.txt,128)
	$(call MUT_RULE,m4_fmtskip,refuse,mut,64)
	@echo "MUTATION GATE: 5/5 mutant runs caught"

# ── stage-2 mutants: tb_walker2_sb against the layer-trace spec ─────────────
# $(1) mutant  $(2) vector file  $(3) -DCFG_D_TB
define MUT2_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/mut2_$(1)/rtl
	mkdir -p $(BUILD)/mut2_$(1)/obj
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=$(3) --top-module tb_walker2_sb \
	    -Mdir $(BUILD)/mut2_$(1)/obj \
	    $(addprefix $(BUILD)/mut2_$(1)/rtl/,$(RTL_FILES)) tb_walker2_sb.sv
	@$(BUILD)/mut2_$(1)/obj/Vtb_walker2_sb \
	    +vectors=$(BUILD)/layer/$(2) +lat_mode=1 +bp_mode=1 +seed=991 \
	    > $(BUILD)/mut2_$(1)/run.log 2>&1 || true
	@$(PYTHON) mutate.py check walk2 $(1) $(BUILD)/mut2_$(1)/run.log
endef

# D-029 erratum: SIGNATURE-REQUIRED variant. A mutant that merely makes the
# suite fail proves nothing here — the scoreboard trips on almost anything.
# This rule demands the INDEPENDENT MXE-legality check's own line, so the
# gate fails if that check is deleted, waived, or made to read the walker's
# decomposition bounds instead of mxe_cfg_pkg's.
# $(1) mutant  $(2) vector file  $(3) -DCFG_D_TB  $(4) required signature
define MUT2_SIG_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/mut2_$(1)/rtl
	mkdir -p $(BUILD)/mut2_$(1)/obj
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=$(3) --top-module tb_walker2_sb \
	    -Mdir $(BUILD)/mut2_$(1)/obj \
	    $(addprefix $(BUILD)/mut2_$(1)/rtl/,$(RTL_FILES)) tb_walker2_sb.sv
	@$(BUILD)/mut2_$(1)/obj/Vtb_walker2_sb \
	      +vectors=$(BUILD)/layer/$(2) +lat_mode=1 +bp_mode=1 +seed=991 \
	      > $(BUILD)/mut2_$(1)/run.log 2>&1 || true
	@if grep -q "WALKER2 PASS" $(BUILD)/mut2_$(1)/run.log; then \
	    echo "MUTANT $(1): NOT CAUGHT — checkers are decorative"; exit 1; \
	  elif ! grep -q "$(4)" $(BUILD)/mut2_$(1)/run.log; then \
	    echo "MUTANT $(1): caught, but NOT by the required check ($(4))"; \
	    echo "  the independent MXE-legality check is missing or disabled"; \
	    exit 1; \
	  else \
	    echo "MUTANT $(1) CAUGHT by the required signature:"; \
	    grep -m 1 -E "$(4)" $(BUILD)/mut2_$(1)/run.log; \
	  fi
endef

mutants2: layer_vectors
	$(call MUT2_RULE,m5_gqamap,qwen7b_T8.sub.ops,128)
	$(call MUT2_RULE,m6_accskip,qwen7b_T8.sub.ops,128)
	$(call MUT2_RULE,m7_rqslot,qwen7b_T8.sub.ops,128)
	$(call MUT2_RULE,m8_fetchbase,layer_h2_hd64.sub.ops,64)
	$(call MUT2_RULE,m10_jcorder,layer_h2_hd64.sub.ops,64)
	$(call MUT2_RULE,m11_ropepos,layer_h2_hd64.sub.ops,64)
	$(call MUT2_RULE,m12_waddr,layer_h2_hd64.sub.ops,64)
	$(call MUT2_RULE,m14_qsrestore,qstage_h1_hd64.sub.ops,64)
	$(call MUT2_RULE,m15_nsrcarm,qstage_h1_hd128.sub.ops,128)
	$(call MUT2_SIG_RULE,m13_nmxe,qwen7b_T8.sub.ops,128,MXE-ILLEGAL DESC)
	@echo "MUTATION GATE 2: 10/10 stage-2+5 mutants caught (m13 signature-required)"

# ── D-029 erratum, gate 4: the SPEC-side half of the blind spot ─────────────
# m13_nmxe moves the bound in the RTL only, so the scoreboard disagrees and
# the run fails twice over. The defect this erratum fixes was NOT like that:
# walker and spec agreed, because both read one overloaded constant, and the
# scoreboard cannot see an error both sides make. This target reproduces THAT
# shape — the spec's own n-split bound moved, RTL untouched — and requires
# gen_layer_trace to hard-error before a single vector is written.
mutants4:
	mkdir -p $(BUILD)
	@if $(PYTHON) gen_layer_trace.py --mutate-n-mxe 4095 \
	      --out $(BUILD)/mutf > $(BUILD)/mutf.log 2>&1; then \
	    echo "SPEC MUTANT m13_nmxe_spec: NOT CAUGHT — the generator blessed"; \
	    echo "  MXE-illegal descriptors; the oracle is self-consistent again"; \
	    exit 1; \
	  elif ! grep -q "MXE-ILLEGAL DESC" $(BUILD)/mutf.log; then \
	    echo "SPEC MUTANT m13_nmxe_spec: failed for the WRONG reason:"; \
	    tail -3 $(BUILD)/mutf.log; exit 1; \
	  else \
	    echo "SPEC MUTANT m13_nmxe_spec CAUGHT by the required signature:"; \
	    grep -m 1 "MXE-ILLEGAL DESC" $(BUILD)/mutf.log; \
	  fi
	@echo "MUTATION GATE 4: 1/1 spec-side legality mutant caught"

# ── SECOND ERRATUM, gate 5: the same blind spot in the LAYER lane ───────────
# LAYER_JOB cols were chunked at the 12-bit FIELD for every unit; apex_top
# wires 7 of those bits to asu_swiglu #(.COLS_MAX(64)), so the d_ffn=18944
# tail chunk (2564) arrived as 4 and was ACCEPTED — silent wrong answer, no
# job_error, walker and spec in perfect agreement. This target restores the
# field bound in the SPEC only and requires the LAYER_JOB legality sweep to
# hard-error before a vector is written. Also gated: swiglu_cols_check.py,
# which re-derives the products against the golden SwiGLU chain.
mutants5:
	mkdir -p $(BUILD)
	@if $(PYTHON) gen_layer_trace.py --mutate-lu-swiglu 4095 \
	      --out $(BUILD)/mutlu > $(BUILD)/mutlu.log 2>&1; then \
	    echo "SPEC MUTANT m14_luswg_spec: NOT CAUGHT — the generator blessed"; \
	    echo "  LAYER_JOBs the swiglu unit cannot read"; \
	    exit 1; \
	  elif ! grep -q "LAYER_JOB-ILLEGAL" $(BUILD)/mutlu.log; then \
	    echo "SPEC MUTANT m14_luswg_spec: failed for the WRONG reason:"; \
	    tail -3 $(BUILD)/mutlu.log; exit 1; \
	  else \
	    echo "SPEC MUTANT m14_luswg_spec CAUGHT by the required signature:"; \
	    grep -m 1 "LAYER_JOB-ILLEGAL" $(BUILD)/mutlu.log; \
	  fi
	set -o pipefail; $(PYTHON) swiglu_cols_check.py \
	    | tee $(BUILD)/swiglu_cols_check.log
	@echo "MUTATION GATE 5: 1/1 spec-side LAYER_JOB mutant caught"

# ── THIRD ERRATUM, gate 6: the same blind spot in the k lane ────────────────
# The k chunk was the frozen K_JOB (2048) for EVERY D, but a k-deep job's act
# family is k/CFG_D stage rows and apex_stage_buf caps R_MAX at 31 — so a
# D=64 walk of the 0.5B down projection blessed 32-row act families no build
# can stage, walker and spec in perfect agreement (the D-029 shape, third
# lane). This target restores the D-blind cap in the SPEC only
# (--mutate-k-rows 32 -> k_job(64) = 2048) and requires the INDEPENDENT
# stageability sweep (seq_walker_fmt.assert_act_stageable, which reads only
# the stage buffer's own SB_* constants) to hard-error on the 0.5B case.
mutants6:
	mkdir -p $(BUILD)
	@if $(PYTHON) gen_layer_trace.py --mutate-k-rows 32 \
	      --out $(BUILD)/mutk > $(BUILD)/mutk.log 2>&1; then \
	    echo "SPEC MUTANT m16_kjob_spec: NOT CAUGHT — the generator blessed"; \
	    echo "  act families no legal stage buffer can hold (k=2048 @ D=64)"; \
	    exit 1; \
	  elif ! grep -q "ACT-UNSTAGEABLE" $(BUILD)/mutk.log; then \
	    echo "SPEC MUTANT m16_kjob_spec: failed for the WRONG reason:"; \
	    tail -3 $(BUILD)/mutk.log; exit 1; \
	  else \
	    echo "SPEC MUTANT m16_kjob_spec CAUGHT by the required signature:"; \
	    grep -m 1 "ACT-UNSTAGEABLE" $(BUILD)/mutk.log; \
	  fi
	@echo "MUTATION GATE 6: 1/1 spec-side stageability mutant caught"

# ── IB-WALK stage 3: SV-vs-mirror descriptor-check equivalence ──────────────
check_vectors:
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_check_vectors.py | tee $(BUILD)/check_vectors.log

build_chk: $(BUILD)/objchk/Vtb_check_sb

$(BUILD)/objchk/Vtb_check_sb: $(RTL_SRCS) tb_check_sb.sv
	mkdir -p $(BUILD)/objchk
	$(VERILATOR) --binary $(VOPTS) --top-module tb_check_sb \
	    -Mdir $(BUILD)/objchk $(RTL_SRCS) tb_check_sb.sv

run_check:
	set -o pipefail; $(BUILD)/objchk/Vtb_check_sb \
	    +vectors=$(BUILD)/check_vectors.txt | tee $(BUILD)/run_check.log
	@grep -q "CHECKEQ PASS" $(BUILD)/run_check.log \
	    || { echo "RUN check: FAIL"; exit 1; }

# $(1) mutant name — the replay must FAIL against a mutated pkg, and it must
# fail with a C1/C2 row mismatch (signature-required via mutate.py check)
define CHKMUT_RULE
	$(PYTHON) mutate.py $(1) $(REPO)/rtl $(BUILD)/chkmut_$(1)/rtl
	mkdir -p $(BUILD)/chkmut_$(1)/obj
	$(VERILATOR) --binary $(VOPTS) --top-module tb_check_sb \
	    -Mdir $(BUILD)/chkmut_$(1)/obj \
	    $(addprefix $(BUILD)/chkmut_$(1)/rtl/,$(RTL_FILES)) tb_check_sb.sv
	@$(BUILD)/chkmut_$(1)/obj/Vtb_check_sb \
	    +vectors=$(BUILD)/check_vectors.txt \
	    > $(BUILD)/chkmut_$(1)/run.log 2>&1 || true
	@$(PYTHON) mutate.py check chk $(1) $(BUILD)/chkmut_$(1)/run.log
endef

mutants3: check_vectors
	$(call CHKMUT_RULE,m4_fmtskip)
	$(call CHKMUT_RULE,m9_resvskip)
	@echo "MUTATION GATE 3: 2/2 check-equivalence mutants caught"

clean:
	rm -rf $(BUILD)
