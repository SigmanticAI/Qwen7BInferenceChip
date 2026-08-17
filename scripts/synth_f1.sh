#!/bin/bash
# =============================================================================
# scripts/synth_f1.sh — F1 exit gate: full-tile synthesis clean
#
# Flow: ordered filelist (packages first) -> sv2v (single flat .v)
#       -> yosys: hierarchy -top apex_top; proc; fsm; opt; memory -nomap; opt; stat
#
# PASS criteria (script exits 0 only if ALL hold):
#   * sv2v exits 0
#   * flattened .v has zero real-math remnants (real/$bitstoreal/$itor/...)
#   * yosys exits 0 with apex_top fully elaborated
#   * no "ERROR:" and no TOK_REAL in the yosys log
#   * no unresolved/blackbox modules after hierarchy
#
# Outputs (under build/synth_f1/):
#   apex_flat.v   — sv2v output
#   yosys.log     — full yosys run (stat section = per-module cell/FF counts)
#   yosys_stat.json — machine-readable stat
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
SV2V=${SV2V:-sv2v}
YOSYS=${YOSYS:-yosys}
OUT=build/synth_f1
mkdir -p "$OUT"

# ── ordered filelist: packages first, then leaf->top (order past packages is
#    cosmetic — sv2v takes the whole list at once — but kept readable) ────────
SRCS=(
  # packages
  rtl/apex_pkg.sv
  rtl/kvq/cores/cq_fp_pkg.sv
  rtl/mxe/mxe_cfg_pkg.sv
  # xbr
  rtl/xbr/stream_skid.sv
  # kvq cores + engine
  rtl/kvq/cores/cq_scale_unit.sv
  rtl/kvq/cores/cq_quant_unit.sv
  rtl/kvq/cores/cq_dequant_unit.sv
  rtl/kvq/cores/cq_rne_div_pipe.sv
  rtl/kvq/cores/cq_quant_pipe.sv
  rtl/kvq/cores/cq_scale_pipe.sv
  rtl/kvq/cores/cq_key_path.sv
  rtl/kvq/cores/cq_value_path.sv
  rtl/kvq/cores/residual_buffer.sv
  rtl/kvq/cores/scale_bank.sv
  rtl/kvq/cores/scale_bank_store.sv
  rtl/kvq/cores/sram_controller.sv
  rtl/kvq/kvq_engine.sv
  # asu
  rtl/asu/rsqrt.sv
  rtl/asu/asu_exp_lut.sv
  rtl/asu/asu_softmax.sv
  rtl/asu/asu_rmsnorm.sv
  # mxe
  rtl/mxe/mxe_pe.sv
  rtl/mxe/mxe_array.sv
  rtl/mxe/mxe_buf.sv
  rtl/mxe/mxe_ctrl.sv
  rtl/mxe/mxe_requant.sv
  rtl/mxe/mxe_top.sv
  # seam / seq / tip / csr
  rtl/seam/seam_feeder_quant.sv
  rtl/seam/seam_score_dequant.sv
  rtl/seq/seq_walker.sv
  rtl/tip/tip_importance.sv
  rtl/tip/tip_decide.sv
  rtl/tip/tip_top.sv
  rtl/csr/csr_regs.sv
  # top glue + tile
  rtl/top/glue/apex_q78_to_fp32.sv
  rtl/top/glue/apex_lane32_ser.sv
  rtl/top/glue/apex_proj_bias.sv
  rtl/top/glue/apex_scale_quant.sv
  rtl/top/glue/apex_score_fork.sv
  rtl/top/glue/apex_stage_buf.sv
  rtl/top/glue/apex_kvq_bank.sv
  rtl/top/glue/apex_kvq_gqa_bank.sv
  rtl/top/glue/apex_wcomp_bank.sv
  rtl/top/apex_top.sv
)

echo "== [1/3] sv2v (${#SRCS[@]} files -> $OUT/apex_flat.v)"
"$SV2V" --define=SYNTHESIS -I rtl/asu "${SRCS[@]}" > "$OUT/apex_flat.v"

echo "== [2/3] real-math remnant scan"
if grep -nE '(^|[^a-zA-Z_$])real([^a-zA-Z_]|$)|\$bitstoreal|\$realtobits|\$itor|\$rtoi|shortreal' \
     "$OUT/apex_flat.v" | grep -v '^\s*//' | grep -vE '^\s*[0-9]+:\s*//'; then
  echo "FAIL: real-math remnant in $OUT/apex_flat.v" >&2
  exit 1
fi
echo "   none — clean"

echo "== [3/3] yosys elaborate + coarse synth + stat"
"$YOSYS" -p "
  read_verilog $OUT/apex_flat.v;
  hierarchy -top apex_top;
  proc; fsm; opt;
  memory -nomap; opt;
  stat;
  tee -o $OUT/yosys_stat.json stat -json;
  tee -o $OUT/mem_cells.rpt dump t:\$mem_v2
" > "$OUT/yosys.log" 2>&1 || { echo "FAIL: yosys exited non-zero — tail of log:"; tail -30 "$OUT/yosys.log"; exit 1; }

# memories the frontend demoted to flip-flops (BRAM-mapping risk list for F2)
grep "Replacing memory" "$OUT/yosys.log" | sort -u > "$OUT/mem_demoted.rpt" || true

# gate checks on the log
fail=0
if grep -iE "^ERROR:|TOK_REAL" "$OUT/yosys.log"; then
  echo "FAIL: ERROR/TOK_REAL in yosys log" >&2; fail=1
fi
if ! grep -q "Printing statistics" "$OUT/yosys.log"; then
  echo "FAIL: yosys stat did not complete" >&2; fail=1
fi
if ! grep -qE "=== apex_top ===" "$OUT/yosys.log"; then
  echo "FAIL: apex_top not elaborated" >&2; fail=1
fi
# unresolved modules => yosys prints them as blackbox/abstract cells in stat,
# and hierarchy warns "module ... not found"
if grep -qE "not found|is a blackbox" "$OUT/yosys.log"; then
  echo "FAIL: unresolved/blackbox module after hierarchy:" >&2
  grep -E "not found|is a blackbox" "$OUT/yosys.log" >&2; fail=1
fi
[ "$fail" -eq 0 ] || exit 1

echo "SYNTH F1: PASS — apex_top fully elaborated, coarse-synth stat complete"
echo "  netlist: $OUT/apex_flat.v"
echo "  log:     $OUT/yosys.log"
echo "  stat:    $OUT/yosys_stat.json"
