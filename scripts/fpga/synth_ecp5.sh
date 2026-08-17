#!/bin/bash
# =============================================================================
# scripts/fpga/synth_ecp5.sh — real ECP5 flow for the apex tile
#   sv2v (same ordered filelist as synth_f1.sh, + FPGA wrapper)
#   -> yosys synth_ecp5 -> nextpnr-ecp5 (LFE5U-85F CABGA381) -> ecppack
#
# TOP selects what is synthesized/placed:
#   apex_top       — the tile itself (synthesis stats; too many pins for P&R)
#   apex_ecp5_top  — thin shift-register pin wrapper around an UNCHANGED
#                    apex_top instance (P&R + bitstream target)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/tools/oss-cad-suite/bin:$PATH"
SV2V=${SV2V:-sv2v}
TOP="${1:-apex_ecp5_top}"
FREQ="${2:-65}"
OUT=build/fpga
mkdir -p "$OUT"

# ordered filelist — mirrors scripts/synth_f1.sh (packages first)
SRCS=(
  rtl/apex_pkg.sv
  rtl/kvq/cores/cq_fp_pkg.sv
  rtl/mxe/mxe_cfg_pkg.sv
  rtl/xbr/stream_skid.sv
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
  rtl/asu/rsqrt.sv
  rtl/asu/asu_exp_lut.sv
  rtl/asu/asu_softmax.sv
  rtl/asu/asu_rmsnorm.sv
  rtl/mxe/mxe_pe.sv
  rtl/mxe/mxe_array.sv
  rtl/mxe/mxe_buf.sv
  rtl/mxe/mxe_ctrl.sv
  rtl/mxe/mxe_requant.sv
  rtl/mxe/mxe_top.sv
  rtl/seam/seam_feeder_quant.sv
  rtl/seam/seam_score_dequant.sv
  rtl/seq/seq_walker.sv
  rtl/tip/tip_importance.sv
  rtl/tip/tip_decide.sv
  rtl/tip/tip_top.sv
  rtl/csr/csr_regs.sv
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
if [ "$TOP" = "apex_ecp5_top" ]; then
  SRCS+=(scripts/fpga/apex_ecp5_top.sv)
fi

echo "== [1/4] sv2v (${#SRCS[@]} files -> $OUT/${TOP}_flat.v)"
"$SV2V" --define=SYNTHESIS -I rtl/asu "${SRCS[@]}" > "$OUT/${TOP}_flat.v"

echo "== [2/4] yosys synth_ecp5 -top $TOP"
yosys -p "
  read_verilog $OUT/${TOP}_flat.v;
  synth_ecp5 -top $TOP -json $OUT/${TOP}.json;
  tee -o $OUT/${TOP}_stat.json stat -json
" > "$OUT/yosys_${TOP}.log" 2>&1 || { echo "FAIL yosys — tail:"; tail -40 "$OUT/yosys_${TOP}.log"; exit 1; }
# print the final ECP5-mapped stat block
awk '/=== '"$TOP"' ===/{f=1} f' "$OUT/yosys_${TOP}.log" | head -60

if [ "$TOP" = "apex_top" ]; then
  echo "== apex_top synth-only run complete (P&R needs the wrapper)"; exit 0
fi

echo "== [3/4] nextpnr-ecp5 --85k --package CABGA381 --freq $FREQ"
nextpnr-ecp5 --85k --package CABGA381 --speed 6 \
  --json "$OUT/${TOP}.json" \
  --textcfg "$OUT/apex.config" \
  --freq "$FREQ" \
  --report "$OUT/rpt.json" \
  --seed 1 \
  > "$OUT/nextpnr.log" 2>&1 || { echo "FAIL nextpnr — tail:"; tail -60 "$OUT/nextpnr.log"; exit 1; }
grep -E "Max frequency|Device utilisation|logic slices|TRELLIS|DP16KD|MULT18X18D" "$OUT/nextpnr.log" | tail -40

echo "== [4/4] ecppack"
ecppack "$OUT/apex.config" "$OUT/apex.bit"
ls -l "$OUT/apex.bit"
echo "ECP5 FLOW: PASS"
