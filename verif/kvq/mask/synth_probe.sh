#!/bin/bash
# synth_probe.sh — S12 stage-5 sv2v+yosys probe of the D-027 engine code.
#
# fparith's synth row already gates the DEFAULT engine shape (OUTLIER_K=0 —
# the reserved-window arm). The two D-027-specific synthesis risks live in
# the K>0 arms and are what this probe elaborates through the real flow:
#   (1) the HAS_MASK_FILE string-parameter compare (generate condition)
#       through sv2v, in BOTH truth values;
#   (2) $readmemh under the split generate (g_mask_rom vs g_mask_zero),
#       with a real mask hex on the ROM arm.
# Probe A: OUTLIER_K=2, MASK_FILE=""      -> g_mask_zero + live CSR mask path
# Probe B: OUTLIER_K=2, MASK_FILE=<d64>   -> g_mask_rom byte-path preserved
# PASS = sv2v clean, yosys elaborates + stats both shapes, zero ERROR lines.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
BUILD=build/synth_probe
mkdir -p "$BUILD"

SRCS="$REPO/rtl/kvq/kvq_engine.sv
$REPO/rtl/kvq/cores/cq_fp_pkg.sv
$REPO/rtl/kvq/cores/cq_scale_unit.sv
$REPO/rtl/kvq/cores/cq_quant_unit.sv
$REPO/rtl/kvq/cores/cq_dequant_unit.sv
$REPO/rtl/kvq/cores/scale_bank.sv
$REPO/rtl/kvq/cores/scale_bank_store.sv
$REPO/rtl/kvq/cores/cq_rne_div_pipe.sv
$REPO/rtl/kvq/cores/cq_quant_pipe.sv
$REPO/rtl/kvq/cores/cq_scale_pipe.sv
$REPO/rtl/kvq/cores/residual_buffer.sv
$REPO/rtl/kvq/cores/sram_controller.sv
$REPO/rtl/kvq/cores/cq_value_path.sv
$REPO/rtl/kvq/cores/cq_key_path.sv"

MASKHEX="$REPO/golden/vectors/d64_T128_G64__CQ4plus/outlier_mask.u8.hex"
[ -f "$MASKHEX" ] || { echo "PROBE: mask hex missing: $MASKHEX"; exit 1; }

# shellcheck disable=SC2086
sv2v --define=SYNTHESIS $SRCS > "$BUILD/engine_flat.v"
echo "sv2v: flat netlist OK ($(wc -l < "$BUILD/engine_flat.v") lines)"

check_yosys () { # $1=tag
  grep -q "Printing statistics" "$BUILD/yosys_$1.log"
  if grep -iE "^ERROR|Warning: .*readmem" "$BUILD/yosys_$1.log"; then
    echo "PROBE [$1]: yosys error/readmem warning"; exit 1; fi
  echo "probe $1: yosys elaborate+stat OK"
}

# Probe A — maskless K>0 arm via chparam (integer params only)
yosys -p "read_verilog $BUILD/engine_flat.v;
          hierarchy -chparam VECTOR_DIM 64 -chparam TIER 2 \
                    -chparam KEY_GROUP 8 -chparam OUTLIER_K 2 \
                    -chparam SRAM_DEPTH 64 -chparam SCALE_SETS 4 \
                    -top kvq_engine;
          proc; opt; stat" > "$BUILD/yosys_maskless.log" 2>&1 \
  || { tail -20 "$BUILD/yosys_maskless.log"; exit 1; }
check_yosys maskless

# Probe B — ROM arm: MASK_FILE bound in a wrapper (string params bind in
# instantiations exactly as the real ECP5/F2 wrappers do — higher fidelity
# than yosys chparam string quoting, which sv2v never sees)
cat > "$BUILD/probe_rom_top.sv" << EOF
module probe_rom_top (
  input  wire        clk, rst_n,
  input  wire [7:0]  awaddr, araddr,
  input  wire [31:0] wdata,
  input  wire        awvalid, wvalid, bready, arvalid, rready,
  input  wire [15:0] s_tdata,
  input  wire        s_tvalid, s_tlast, s_tuser, m_tready, flush_req,
  output wire        awready, wready, bvalid, arready, rvalid,
  output wire [1:0]  bresp, rresp,
  output wire [31:0] rdata,
  output wire [31:0] m_tdata,
  output wire        m_tvalid, m_tlast, s_tready,
  output wire        irq, evict_needed, mask_valid,
  output wire [5:0]  evict_addr
);
  kvq_engine #(
    .VECTOR_DIM(64), .TIER(2), .KEY_GROUP(8), .OUTLIER_K(2),
    .SCALE_SETS(4), .SRAM_DEPTH(64),
    .MASK_FILE("$MASKHEX")
  ) u_e (
    .clk(clk), .rst_n(rst_n),
    .axil_awaddr(awaddr), .axil_awvalid(awvalid), .axil_awready(awready),
    .axil_wdata(wdata), .axil_wvalid(wvalid), .axil_wready(wready),
    .axil_bresp(bresp), .axil_bvalid(bvalid), .axil_bready(bready),
    .axil_araddr(araddr), .axil_arvalid(arvalid), .axil_arready(arready),
    .axil_rdata(rdata), .axil_rresp(rresp), .axil_rvalid(rvalid),
    .axil_rready(rready),
    .s_axis_kv_tdata(s_tdata), .s_axis_kv_tvalid(s_tvalid),
    .s_axis_kv_tready(s_tready), .s_axis_kv_tlast(s_tlast),
    .s_axis_kv_tuser(s_tuser),
    .m_axis_kv_tdata(m_tdata), .m_axis_kv_tvalid(m_tvalid),
    .m_axis_kv_tready(m_tready), .m_axis_kv_tlast(m_tlast),
    .flush_req(flush_req), .irq(irq),
    .evict_needed(evict_needed), .evict_addr(evict_addr),
    .mask_valid(mask_valid)
  );
endmodule
EOF
# shellcheck disable=SC2086
sv2v --define=SYNTHESIS "$BUILD/probe_rom_top.sv" $SRCS > "$BUILD/engine_rom_flat.v"
yosys -p "read_verilog $BUILD/engine_rom_flat.v;
          hierarchy -top probe_rom_top; proc; opt; stat" \
  > "$BUILD/yosys_rom.log" 2>&1 \
  || { tail -20 "$BUILD/yosys_rom.log"; exit 1; }
check_yosys rom

echo "SYNTH PROBE: PASS (sv2v + yosys, K=2 maskless AND ROM arms)"
