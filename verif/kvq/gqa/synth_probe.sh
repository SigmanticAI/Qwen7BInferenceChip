#!/bin/bash
# synth_probe.sh — IB-LAYER S4b ×H_kv yosys probe (IB_LAYER.md §1b flow;
# obligation from LEVEL_C §9.1 R3-AMENDED: "a KVQ-engine synth probe joins
# the §4 stage-6 probe list so the ×H_kv figure lands MEASURED, not argued").
#
# Elaborates apex_kvq_gqa_bank at the I-B GQA geometry — CFG_D=128 (7B
# head_dim), KVQ_DEPTH=256 (the R3 2T<=256 sizing point / verified L3-f2
# depth), KVQ_G=16, KVQ_SETS=8 (the tile derivation at G=16/T_ROW_MAX=128) —
# at N_ENG=1 and N_ENG=4, through the house sv2v→yosys flow on BOTH targets:
#   US+  : synth_xilinx -family xcup -abc9 ; stat
#   ECP5 : synth_ecp5 -abc9 ; stat
# The N_ENG=4 minus N_ENG=1 delta IS the measured ×H_kv cost (3 extra CQ-8
# engines + the select fabric). Scope caveats, mirroring §1b verbatim: yosys
# mapping evidence only — NO P&R, NO timing (timing comes only from the I-B
# Vivado build report); the post-synth netlist is not re-simulated.
# PASS = sv2v clean, all four syntheses complete ("Printing statistics"),
# zero ERROR lines; the cell table is extracted into build/synth_probe/.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
BUILD=build/synth_probe
mkdir -p "$BUILD"

SRCS="$REPO/rtl/top/glue/apex_kvq_gqa_bank.sv
$REPO/rtl/kvq/kvq_engine.sv
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

# parameters bind in a generated WRAPPER instantiation (the §1b feeder-probe
# / mask-probe-B idiom — chparam on a bank top that re-parameterizes its
# engine generate trips a yosys 0.66 rtlil reprocessing assert, measured)
make_wrapper () { # $1=n_eng $2=eng_w
  cat > "$BUILD/gqa_probe_n$1.sv" << EOF
module gqa_probe_n$1 (
  input  wire        clk, rst_n,
  input  wire [$2:0] eng_sel,
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
  output wire        irq, evict_needed, m_pending,
  output wire [7:0]  evict_addr
);
  apex_kvq_gqa_bank #(
    .N_ENG($1), .CFG_D(128), .KVQ_G(16), .KVQ_DEPTH(256), .KVQ_SETS(8)
  ) u_bank (
    .clk(clk), .rst_n(rst_n), .eng_sel(eng_sel),
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
    .m_pending(m_pending)
  );
endmodule
EOF
}

make_wrapper 1 0
make_wrapper 4 1

flatten_cfg () { # $1=n_eng
  # shellcheck disable=SC2086
  sv2v --define=SYNTHESIS "$BUILD/gqa_probe_n$1.sv" $SRCS \
    > "$BUILD/gqa_n$1_flat.v"
  echo "sv2v n$1: flat netlist OK ($(wc -l < "$BUILD/gqa_n$1_flat.v") lines)"
}
flatten_cfg 1
flatten_cfg 4

run_probe () { # $1=tag $2=n_eng $3=synth_cmd
  yosys -p "read_verilog $BUILD/gqa_n$2_flat.v;
            hierarchy -top gqa_probe_n$2;
            $3 -abc9;
            stat" > "$BUILD/yosys_$1.log" 2>&1 \
    || { tail -20 "$BUILD/yosys_$1.log"; exit 1; }
  grep -q "Printing statistics" "$BUILD/yosys_$1.log"
  if grep -iE "^ERROR" "$BUILD/yosys_$1.log"; then
    echo "PROBE [$1]: yosys error"; exit 1; fi
  echo "probe $1: yosys synth+stat OK"
}

run_probe xcup_n1 1 "synth_xilinx -family xcup"
run_probe xcup_n4 4 "synth_xilinx -family xcup"
run_probe ecp5_n1 1 "synth_ecp5"
run_probe ecp5_n4 4 "synth_ecp5"

# cell-table extraction. yosys `stat` cell rows are "<count> <cell>"; the
# DESIGN TOTALS ("Count including submodules" on xcup, the flattened single
# module on ecp5) are the LAST occurrence of each cell name in the log, so
# a keep-last awk over the whole log reads exactly the totals block.
cell_n () { # $1=log $2=cell
  awk -v c="$2" '$2==c {v=$1} END {print v+0}' "$BUILD/yosys_$1.log"
}

report () { # $1=tag $2..=cells
  local tag="$1" out
  out="$BUILD/cells_$1.txt"
  shift
  : > "$out"
  for c in "$@"; do
    printf "%-12s %8d\n" "$c" "$(cell_n "$tag" "$c")" >> "$out"
  done
  echo "== $tag =="; cat "$out"
}

xcup_luts () { # LUT2..LUT6 total
  local s=0 c
  for c in LUT2 LUT3 LUT4 LUT5 LUT6; do s=$((s + $(cell_n "$1" "$c"))); done
  echo "$s"
}
xcup_ffs () { # FDCE+FDPE+FDRE+FDSE (the engines are async-reset: FDCE/FDPE)
  local s=0 c
  for c in FDCE FDPE FDRE FDSE; do s=$((s + $(cell_n "$1" "$c"))); done
  echo "$s"
}

for t in xcup_n1 xcup_n4; do
  report "$t" RAMB36E2 RAMB18E2 DSP48E2 CARRY4 MUXF7 MUXF8 INV
  echo "LUT2-6 total: $(xcup_luts "$t")" | tee -a "$BUILD/cells_$t.txt"
  echo "FF total:     $(xcup_ffs "$t")"  | tee -a "$BUILD/cells_$t.txt"
done
for t in ecp5_n1 ecp5_n4; do
  report "$t" DP16KD MULT18X18D LUT4 TRELLIS_FF CCU2C PFUMX L6MUX21
done

# §1b-style verbatim mapping evidence (record SRAM must infer block RAM)
grep -h "mapping memory" "$BUILD"/yosys_xcup_n1.log "$BUILD"/yosys_xcup_n4.log \
  "$BUILD"/yosys_ecp5_n1.log "$BUILD"/yosys_ecp5_n4.log \
  | tee "$BUILD/mapping_evidence.txt"

echo "SYNTH PROBE: PASS (sv2v + yosys, N_ENG=1 and N_ENG=4 on xcup AND ecp5)"
