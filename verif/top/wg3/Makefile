# =============================================================================
# verif/top/wg3/Makefile — W-G3 executor cases.
#
#   make case     - render the executor cases from the stage-B artifacts
#                   (verif/walkgold/gen_wg3_case.py; self-checks C1-C5)
#   make fenced   - the FENCED (mask 0x03D) real-7B walk at unit level:
#                   H=28, H_kv=4, GQA-4 banking, 16,139 emissions vs golden
#   make topo     - the TOPOLOGY A/B: is apex_top's composite/interlock
#                   topology (NOT the walker, NOT the vectors) what blocks a
#                   multi-head multi-group fmt=1 attention walk?
#                   arm A (control) must PASS, arm B (apex_top as built) must
#                   FAIL — a discrimination gate, same shape as the repo's
#                   mutation gates: an arm that stopped failing would mean the
#                   experiment had lost its power.
#   make all      - case + fenced + topo
#
# Prerequisites (gitignored, regenerate in THIS order):
#   python3 verif/walkgold/trace_layer_golden.py --step 19 --layer 19
#   python3 verif/walkgold/gen_walk_golden.py
#
# pipefail is load-bearing on every gate line (the tee lesson).
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
PYTHON      ?= python3
REPO        := ../../..
include $(REPO)/verif/tools.mk
BUILD       := build
GOLD        := $(REPO)/build/walkgold
CASE        := $(GOLD)/fenced_h28.sub.ops

# the walker + its composite unit, exactly as verif/seq_walker builds them
RTL = \
    $(REPO)/rtl/apex_pkg.sv \
    $(REPO)/rtl/mxe/mxe_cfg_pkg.sv \
    $(REPO)/rtl/kvq/cores/cq_fp_pkg.sv \
    $(REPO)/rtl/seq/seq_walker_pkg.sv \
    $(REPO)/rtl/seq/seq_walker.sv \
    $(REPO)/rtl/seq/seq_walker_comp.sv \
    $(REPO)/rtl/seq/seq_layer_walker.sv \
    $(REPO)/rtl/seq/seq_layer_walker2.sv

VOPTS = -Wall --timing --assert --timescale 1ns/1ps -I$(REPO)/rtl/asu -I. \
        $(REPO)/verif/seq_walker/lint_waivers.vlt

.PHONY: all case fenced topo lint clean tilecase tilebuild tilehost tilewalk tile krope fuel fuelddr

all: case fenced topo tile

# ═══════════════════════════════════════════════════════════════════════════
# The TILE-level walked half — integration-lead ruling (2026-07-26), option 3.
#
# SCOPE, stated precisely: a SINGLE-HEAD walked attention step at real 7B
# PER-HEAD geometry (head_dim=128, T=20, CQ-8), driven by the REAL traced
# Qwen2.5-7B head 3 of step 19 / layer 19 — the head with the committed S8
# record. The MULTI-HEAD walk (H=28, H_kv=4) is BLOCKED by F6 (i) apex_top's
# single seq_walker_comp instance and (ii) its single-arm hd_ready sticky;
# the A/B in `make topo` is the evidence. Option 1 (per-KV-head caches) is
# owned by a separate lane — this lane does not touch apex_top.
# ═══════════════════════════════════════════════════════════════════════════
CASE_NAME  := wg3_s019_L19_h03_hd128_T20
KROPE_NAME := wg3_krope_gqa4_s019_L19
TB_L3     := $(REPO)/verif/top/l3

tilecase:
	@test -f $(GOLD)/layer_s019_L19.npz \
	  || { echo "missing stage-A bundle — run verif/walkgold/ first"; exit 1; }
	mkdir -p $(BUILD)
	set -o pipefail; $(PYTHON) gen_wg3_tile_case.py --build $(BUILD) \
	    | tee $(BUILD)/tilecase.log
	@grep -q "WG3 TILE CASE: PASS" $(BUILD)/tilecase.log \
	    || { echo "WG3 TILE CASE: FAIL"; exit 1; }

# the I-B CL build point, driving the L3 TB (additively parameterised:
# GQANE/RMSDM/LAYDM default to apex_top's own defaults, so every existing L3
# build stays byte-identical — proven by re-running `make -C verif/top/l3 all`)
$(BUILD)/obj_tile/Vtb_apex_l3: $(TB_L3)/tb_apex_l3.sv
	mkdir -p $(BUILD)/obj_tile
	$(VERILATOR) --binary -Wall --timing --assert --timescale 1ns/1ps \
	    -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu \
	    -I$(TB_L3) $(TB_L3)/lint_waivers.vlt \
	    -GCFG_D=128 -GG_CFG=16 -GDEPTH=256 -GOUTK=0 \
	    -GGQANE=4 -GRMSDM=3584 -GLAYDM=3584 \
	    --top-module tb_apex_l3 -Mdir $(BUILD)/obj_tile \
	    $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv $(TB_L3)/tb_apex_l3.sv

tilebuild: $(BUILD)/obj_tile/Vtb_apex_l3

# (a) HOST mode — the compat baseline: the real head replays bit-exact vs
#     golden through the real tile with the host driving every phase.
tilehost: tilebuild tilecase
	set -o pipefail; $(BUILD)/obj_tile/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(CASE_NAME).txt +watchdog=40000000 \
	    | tee $(BUILD)/run_tilehost.log
	@grep -q "L3 PASS" $(BUILD)/run_tilehost.log \
	    || { echo "WG3 TILE HOST: FAIL"; exit 1; }
	@echo "WG3 TILE HOST: PASS (real 7B head, host-sequenced)"

# (b) WALKED — the same case, same scoreboard, but the fmt=1 image is loaded
#     through the REAL WALK CSR window and the WALKER drives score+pv.
tilewalk: tilebuild tilecase
	set -o pipefail; $(BUILD)/obj_tile/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(CASE_NAME).txt +walker=1 +walkfmt2=1 \
	    +walkdesc=$(BUILD)/walker/$(CASE_NAME).desc \
	    +walkdesc2=$(BUILD)/walker/$(CASE_NAME).desc2 +watchdog=40000000 \
	    | tee $(BUILD)/run_tilewalk.log
	@grep -q "L3 PASS" $(BUILD)/run_tilewalk.log \
	    && grep -q "WALKFMT2: fmt=1 WALKED clean" $(BUILD)/run_tilewalk.log \
	    || { echo "WG3 TILE WALK: FAIL"; exit 1; }
	@echo "WG3 TILE WALK: PASS (real 7B head, SINGLE-HEAD walked via fmt=1)"

tile: tilehost tilewalk krope fuel fuelddr

# (c) HOST-MODE GQA-4 K-rope coverage — needs no walk, so F6 does not block it.
#     80 real Qwen K rows staged PRE-RoPE across all 4 GQA engines; the
#     dbg_f16 tap carries the POST-RoPE beat and is gated against the D-030
#     arbiter (F5: never the legacy K_f16 field).
krope: tilebuild
	set -o pipefail; $(PYTHON) gen_wg3_krope_case.py --build $(BUILD) \
	    | tee $(BUILD)/kropecase.log
	@grep -q "WG3 KROPE CASE: PASS" $(BUILD)/kropecase.log \
	    || { echo "WG3 KROPE CASE: FAIL"; exit 1; }
	set -o pipefail; $(BUILD)/obj_tile/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(KROPE_NAME).txt +watchdog=60000000 \
	    | tee $(BUILD)/run_krope.log
	@grep -q "L3 PASS" $(BUILD)/run_krope.log \
	    || { echo "WG3 KROPE: FAIL"; exit 1; }
	@echo "WG3 KROPE: PASS (80 real K rows, 4 GQA engines, D-030 arbiter)"
	@#  F5 DISCRIMINATION: the same case pinned to the LEGACY K_f16 field
	@#  must FAIL. If it ever stops failing, the gate has lost its power and
	@#  the D-030-vs-legacy distinction is no longer being tested.
	set -o pipefail; $(PYTHON) gen_wg3_krope_case.py --build $(BUILD) \
	    --legacy-arbiter | tee $(BUILD)/kropecase_legacy.log
	-$(BUILD)/obj_tile/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(KROPE_NAME)_legacy.txt \
	    +watchdog=60000000 > $(BUILD)/run_krope_legacy.log 2>&1
	@grep -q "L3 PASS" $(BUILD)/run_krope_legacy.log \
	    && { echo "F5 DISCRIMINATION: legacy-pinned arm PASSED — the gate has lost its power"; exit 1; } \
	    || echo "F5 DISCRIMINATION: PASS (legacy-pinned arm RED as predicted)"
	@grep -m1 "tap f16" $(BUILD)/run_krope_legacy.log

# (d) FUEL half, TILE leg: a host-driven projection GEMM at the legal n=8 over
#     REAL Qwen Wq weights arriving on the EXTERNAL xw port — the port the
#     IB-FUEL reader feeds. The pushed byte stream is asserted byte-identical
#     to the compact DDR case image, so this leg joins the f2sim DDR leg on a
#     proven identity. The DDR leg itself is NOT yet wired (RESULT.md 5.3).
FUEL_NAME := wg3_fuel_wq_k2944_hd128

fuel: tilebuild
	set -o pipefail; $(PYTHON) gen_wg3_fuel_case.py --build $(BUILD) \
	    | tee $(BUILD)/fuelcase.log
	@grep -q "WG3 FUEL CASE: PASS" $(BUILD)/fuelcase.log \
	    || { echo "WG3 FUEL CASE: FAIL"; exit 1; }
	set -o pipefail; $(BUILD)/obj_tile/Vtb_apex_l3 \
	    +vectors=$(BUILD)/cases/$(FUEL_NAME).txt +watchdog=40000000 \
	    | tee $(BUILD)/run_fuel.log
	@grep -q "L3 PASS" $(BUILD)/run_fuel.log \
	    || { echo "WG3 FUEL: FAIL"; exit 1; }
	@echo "WG3 FUEL (tile leg): PASS (real Qwen Wq via the external xw port)"

# (e) FUEL half, DDR leg: the SAME projection GEMM with the weight beats
#     sourced from DDR (loader -> sh_ddr -> apex_fuel_reader -> afifo -> xw)
#     instead of the host mailbox. Three arms:
#       host  — mailbox weights through the full cl_apex executor (baseline)
#       fuel  — all 128 WB triples REMOVED, beats from the compact DDR image;
#               same INT32 reference, plus FUEL_STAT.last_tag + FUEL_ERR==0
#       bad   — the same regops against a DDR image with ONE INT8 lane
#               perturbed (manifest sha updated so it LOADS cleanly). MUST
#               FAIL: if it passes, the weights did not come from DDR.
F2SIM  := $(REPO)/verif/f2sim
FUELDIR := $(BUILD)/fuel

fuelddr: fuel
	$(MAKE) -C $(F2SIM) build
	set -o pipefail; $(F2SIM)/obj_d128_ddr1/f2sim \
	    $(FUELDIR)/$(FUEL_NAME).regops.jsonl | tee $(BUILD)/run_fuel_host.log
	@grep -q "F2SIM RESULT: files=1 checks=30 fails=0 -> PASS" $(BUILD)/run_fuel_host.log \
	    || { echo "FUEL DDR: host arm FAIL"; exit 1; }
	set -o pipefail; $(F2SIM)/obj_d128_ddr1/f2sim \
	    +ddr_image=$(FUELDIR)/fuel.bin \
	    +ddr_regions=$(FUELDIR)/fuel.regions.jsonl +fuel_audit \
	    $(FUELDIR)/$(FUEL_NAME).fuel.regops.jsonl | tee $(BUILD)/run_fuel_ddr.log
	@grep -q "F2SIM RESULT: files=1 checks=32 fails=0 -> PASS" $(BUILD)/run_fuel_ddr.log \
	    && grep -q "DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS" $(BUILD)/run_fuel_ddr.log \
	    && grep -q "FUELAUDIT.*err=00000000.*-> ok" $(BUILD)/run_fuel_ddr.log \
	    || { echo "FUEL DDR: fuel arm FAIL"; exit 1; }
	-$(F2SIM)/obj_d128_ddr1/f2sim \
	    +ddr_image=$(FUELDIR)/fuel_bad.bin \
	    +ddr_regions=$(FUELDIR)/fuel_bad.regions.jsonl +fuel_audit \
	    $(FUELDIR)/$(FUEL_NAME).fuel.regops.jsonl > $(BUILD)/run_fuel_bad.log 2>&1
	@grep -q "F2SIM RESULT: files=1 checks=32 fails=1 -> FAIL" $(BUILD)/run_fuel_bad.log \
	    || { echo "FUEL DDR: perturbed arm did NOT fail — the result does not depend on DDR contents"; exit 1; }
	@grep -q "DDRLOAD RESULT: regions=1 words=16 rb_fails=0 sha_fails=0 -> PASS" $(BUILD)/run_fuel_bad.log \
	    || { echo "FUEL DDR: perturbed arm failed at LOAD, not at the result — weaker evidence"; exit 1; }
	@echo "WG3 FUEL (DDR leg): PASS (weights from DDR, golden-gated; perturbed-DDR arm RED as required)"
	@grep -hE "DDRLOAD RESULT|FUELAUDIT|F2SIM RESULT" $(BUILD)/run_fuel_ddr.log $(BUILD)/run_fuel_bad.log

# ── the build point this lane targets, proven to elaborate -Wall clean ───────
# The tile file list is the L3 suite's RTL_CORE, scraped rather than copied so
# the two cannot drift; lint_waivers.vlt here is L3's (frozen/vendored only).
TILE_RTL := $(shell sed -n '/^RTL_CORE = /,/apex_scale_quant.sv/p' \
              ../l3/Makefile | grep -o '\$$(REPO)/[a-z0-9_/]*\.sv' \
              | sed 's|\$$(REPO)|$(REPO)|')

lint:
	@test -n "$(TILE_RTL)" || { echo "lint: RTL_CORE scrape from ../l3/Makefile came back EMPTY — refusing to lint nothing"; exit 1; }
	$(VERILATOR) --lint-only -Wall --timing --assert --timescale 1ns/1ps \
	    -I$(REPO)/rtl/rope -I$(REPO)/verif/common -I$(REPO)/rtl/asu -I. \
	    lint_waivers.vlt \
	    -GCFG_D=128 -GKVQ_DEPTH=256 -GKVQ_GQA_NENG=4 \
	    -GRMS_D_MAX=3584 -GLAYER_DM_MAX=3584 \
	    --top-module apex_top $(TILE_RTL) $(REPO)/rtl/top/apex_top.sv
	@echo "WG3 BUILD POINT LINT CLEAN (CFG_D=128 KVQ_DEPTH=256 KVQ_GQA_NENG=4 RMS_D_MAX=3584 LAYER_DM_MAX=3584)"

case:
	mkdir -p $(BUILD)
	@test -f $(GOLD)/layer_fenced_walk.ops \
	  || { echo "missing stage-B artifacts — run verif/walkgold/ first"; exit 1; }
	set -o pipefail; $(PYTHON) $(REPO)/verif/walkgold/gen_wg3_case.py \
	    | tee $(BUILD)/case.log
	@grep -q "WG3 CASE RENDER: PASS" $(BUILD)/case.log \
	    || { echo "WG3 CASE RENDER: FAIL"; exit 1; }

$(BUILD)/objab/Vtb_wg3_topo_ab: $(RTL) tb_wg3_topo_ab.sv
	mkdir -p $(BUILD)/objab
	$(VERILATOR) --binary $(VOPTS) -DCFG_D_TB=128 \
	    --top-module tb_wg3_topo_ab -Mdir $(BUILD)/objab \
	    $(RTL) tb_wg3_topo_ab.sv

# the FENCED real-7B case against the verified harness topology
fenced: $(BUILD)/objab/Vtb_wg3_topo_ab
	@test -f $(CASE) || { echo "missing $(CASE) — run 'make case'"; exit 1; }
	set -o pipefail; $(BUILD)/objab/Vtb_wg3_topo_ab +vectors=$(CASE) \
	    +lat_mode=1 +bp_mode=1 +seed=503 +topo=0 | tee $(BUILD)/run_fenced.log
	@grep -q "WG3-TOPO PASS" $(BUILD)/run_fenced.log \
	    || { echo "WG3 FENCED: FAIL"; exit 1; }
	@echo "WG3 FENCED: PASS"

# discrimination gate: arm A green, arm B RED (see the header)
topo: $(BUILD)/objab/Vtb_wg3_topo_ab
	@test -f $(CASE) || { echo "missing $(CASE) — run 'make case'"; exit 1; }
	set -o pipefail; $(BUILD)/objab/Vtb_wg3_topo_ab +vectors=$(CASE) \
	    +lat_mode=1 +bp_mode=1 +seed=503 +topo=0 | tee $(BUILD)/run_topoA.log
	-$(BUILD)/objab/Vtb_wg3_topo_ab +vectors=$(CASE) \
	    +lat_mode=1 +bp_mode=1 +seed=503 +topo=1 > $(BUILD)/run_topoB.log 2>&1
	@grep -q "WG3-TOPO PASS" $(BUILD)/run_topoA.log \
	    || { echo "TOPO A/B: control arm did not pass"; exit 1; }
	@grep -q "WG3-TOPO FAIL" $(BUILD)/run_topoB.log \
	    || { echo "TOPO A/B: the pre-F6(i) arm did NOT fail — the witness has lost its power"; exit 1; }
	@echo "TOPO A/B: PASS (control green, pre-F6(i) single-cache arm RED as required)"
	@grep -h "WG3-TOPO\[" $(BUILD)/run_topoA.log $(BUILD)/run_topoB.log

clean:
	rm -rf $(BUILD)
