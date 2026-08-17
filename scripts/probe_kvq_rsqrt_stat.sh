#!/bin/bash
# =============================================================================
# scripts/probe_kvq_rsqrt_stat.sh — area probe: yosys stat for kvq_engine
# (three contract configs) + rsqrt_unit. ANALYSIS TOOL ONLY — no RTL touched.
#
# Flow (mirrors the F1 house flow): param wrapper top -> sv2v (--define=SYNTHESIS)
#   -> yosys: hierarchy -top <wrap>; proc; fsm; opt; memory -nomap; opt; stat
# `memory -nomap` keeps the record SRAM / residual buffer as $mem cells so the
# stat separates true datapath FFs from storage bits.
#
# Configs probed:
#   kvq_d64_cq8   D=64  TIER=0 G=128 DEPTH=256  (value/CQ-8 path only)
#   kvq_d64_cq4   D=64  TIER=1 G=64  DEPTH=256  (smoke config, grouped keys)
#   kvq_d128_cq4  D=128 TIER=1 G=128 DEPTH=256  (shipped envelope scale point)
#   rsqrt         rsqrt_unit standalone
#
# Outputs under build/probe_stat/<cfg>/: flat.v, yosys.log, stat.json
# and a combined summary at build/probe_stat/SUMMARY.txt
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
SV2V=${SV2V:-sv2v}
YOSYS=${YOSYS:-yosys}
OUT=build/probe_stat
mkdir -p "$OUT"

CORES=rtl/kvq/cores
KVQ_SRCS=(
  "$CORES/cq_fp_pkg.sv" "$CORES/cq_scale_unit.sv"
  "$CORES/cq_quant_unit.sv" "$CORES/cq_dequant_unit.sv" "$CORES/scale_bank.sv" "$CORES/scale_bank_store.sv"
  "$CORES/cq_rne_div_pipe.sv" "$CORES/cq_quant_pipe.sv" "$CORES/cq_scale_pipe.sv"
  "$CORES/residual_buffer.sv" "$CORES/sram_controller.sv"
  "$CORES/cq_value_path.sv" "$CORES/cq_key_path.sv" rtl/kvq/kvq_engine.sv
)

probe_kvq () { # $1 cfg name, $2 D, $3 TIER, $4 G, $5 DEPTH
  local cfg=$1 D=$2 TIER=$3 G=$4 DEPTH=$5
  local dir="$OUT/$cfg"; mkdir -p "$dir"
  cat > "$dir/wrap.sv" <<EOF
// generated probe wrapper — pins the $cfg contract config
module probe_top (
  input  wire clk, input  wire rst_n,
  input  wire [7:0] axil_awaddr, input wire axil_awvalid, output wire axil_awready,
  input  wire [31:0] axil_wdata, input wire axil_wvalid, output wire axil_wready,
  output wire [1:0] axil_bresp, output wire axil_bvalid, input wire axil_bready,
  input  wire [7:0] axil_araddr, input wire axil_arvalid, output wire axil_arready,
  output wire [31:0] axil_rdata, output wire [1:0] axil_rresp,
  output wire axil_rvalid, input wire axil_rready,
  input  wire [15:0] s_axis_kv_tdata, input wire s_axis_kv_tvalid,
  output wire s_axis_kv_tready, input wire s_axis_kv_tlast, input wire s_axis_kv_tuser,
  output wire [31:0] m_axis_kv_tdata, output wire m_axis_kv_tvalid,
  input  wire m_axis_kv_tready, output wire m_axis_kv_tlast,
  input  wire flush_req, output wire irq,
  output wire evict_needed, output wire [\$clog2($DEPTH)-1:0] evict_addr
);
  kvq_engine #(
    .VECTOR_DIM($D), .TIER($TIER), .KEY_GROUP($G), .OUTLIER_K(0),
    .SRAM_DEPTH($DEPTH)
  ) u_dut (.*);
endmodule
EOF
  "$SV2V" --define=SYNTHESIS "${KVQ_SRCS[@]}" "$dir/wrap.sv" > "$dir/flat.v"
  "$YOSYS" -p "read_verilog $dir/flat.v; hierarchy -top probe_top; proc; fsm; opt; memory -nomap; opt; stat; flatten; opt_clean; stat; tee -o $dir/stat.json stat -json; write_json $dir/netlist.json" \
      > "$dir/yosys.log" 2>&1 || { tail -20 "$dir/yosys.log"; exit 1; }
  grep -q "Printing statistics" "$dir/yosys.log"
  ! grep -iE "^ERROR:|TOK_REAL" "$dir/yosys.log" >/dev/null
  echo "probe $cfg: OK  (log: $dir/yosys.log)"
}

probe_rsqrt () {
  local dir="$OUT/rsqrt"; mkdir -p "$dir"
  "$SV2V" --define=SYNTHESIS rtl/asu/rsqrt.sv > "$dir/flat.v"
  "$YOSYS" -p "read_verilog $dir/flat.v; hierarchy -top rsqrt_unit; proc; fsm; opt; stat; tee -o $dir/stat.json stat -json; write_json $dir/netlist.json" \
      > "$dir/yosys.log" 2>&1 || { tail -20 "$dir/yosys.log"; exit 1; }
  grep -q "Printing statistics" "$dir/yosys.log"
  ! grep -iE "^ERROR:|TOK_REAL" "$dir/yosys.log" >/dev/null
  echo "probe rsqrt: OK  (log: $dir/yosys.log)"
}

probe_kvq kvq_d64_cq8   64  0 128 256
probe_kvq kvq_d64_cq4   64  1 64  256
probe_kvq kvq_d128_cq4  128 1 128 256
probe_rsqrt

# ── summary: design totals per config ────────────────────────────────────────
{
  echo "yosys stat probe — kvq_engine + rsqrt_unit ($(date +%F))"
  echo "flow: sv2v -> yosys proc; fsm; opt; memory -nomap; opt; stat"
  for cfg in kvq_d64_cq8 kvq_d64_cq4 kvq_d128_cq4 rsqrt; do
    echo; echo "=== $cfg (flat totals: cells / FF-family / div / mul) ==="
    awk '/Printing statistics/{n++} n>=2' "$OUT/$cfg/yosys.log" \
      | grep -E '[0-9]+ +cells$|\$adffe?|\$s?dffe?|\$div|\$mul' || true
    # rsqrt is a single module: only one stat pass
    if ! awk '/Printing statistics/{n++} END{exit !(n>=2)}' "$OUT/$cfg/yosys.log"; then
      awk '/Printing statistics/{n++} n>=1' "$OUT/$cfg/yosys.log" \
        | grep -E '[0-9]+ +cells$|\$adffe?|\$s?dffe?|\$div|\$mul' || true
    fi
  done
} > "$OUT/SUMMARY.txt"
echo "PROBE OK — summary: $OUT/SUMMARY.txt"
