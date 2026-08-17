#!/bin/bash
# s10_baseline.sh — S10 Phase 0: honest post-route Fmax re-baseline of the KVQ
# engine BEFORE any pipelining edit. Two configs, both post-D-026 RTL:
#   B0  proxy  (sv2v param defaults: D=16 TIER=1 G=2 K=0 DEPTH=2) — apples-to-
#       apples with the stale 9.77 MHz number that was measured pre-D-026
#   B1  ship   (D=64 TIER=2 G=128 K_OUT=2 DEPTH=128 — the apex_top params)
# Quote ONLY the post-route "Warning: Max frequency" line, config-labeled.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/tools/oss-cad-suite/bin:$PATH"
mkdir -p build/fpga/s10

# Timing constraint for the P&R step. 100 was the original target; the
# pipelined ship config tops out ~35-45 MHz, and constraining 2.5x beyond
# reach sent nextpnr's timing-driven router into a non-converging congestion
# spiral (35 h, killed 2026-07-16). Route at a reachable target and quote the
# achieved post-route Fmax: FREQ=40 bash scripts/fpga/s10_baseline.sh
FREQ="${FREQ:-100}"

CORES="rtl/kvq/cores/cq_fp_pkg.sv rtl/kvq/cores/cq_scale_unit.sv \
rtl/kvq/cores/cq_quant_unit.sv rtl/kvq/cores/cq_dequant_unit.sv \
rtl/kvq/cores/scale_bank.sv rtl/kvq/cores/scale_bank_store.sv \
rtl/kvq/cores/cq_rne_div_pipe.sv rtl/kvq/cores/cq_quant_pipe.sv \
rtl/kvq/cores/cq_scale_pipe.sv \
rtl/kvq/cores/residual_buffer.sv rtl/kvq/cores/sram_controller.sv \
rtl/kvq/cores/cq_value_path.sv rtl/kvq/cores/cq_key_path.sv"

run_cfg () {  # $1=name $2=extra sv2v defines/files (wrapper) $3=top
  local name=$1 top=$3
  echo "== [$name] sv2v"
  # shellcheck disable=SC2086
  sv2v --define=SYNTHESIS $2 $CORES rtl/kvq/kvq_engine.sv \
    > "build/fpga/s10/${name}_flat.v"
  echo "== [$name] yosys synth_ecp5 -abc9"
  yosys -p "read_verilog build/fpga/s10/${name}_flat.v; \
            synth_ecp5 -abc9 -top $top -json build/fpga/s10/${name}.json; stat" \
    > "build/fpga/s10/${name}_yosys.log" 2>&1
  echo "== [$name] nextpnr-ecp5 (post-route timing)"
  # SEED matters: seed 1 is pathological for the ship config (rip-up loop,
  # killed at 11-35 h twice); seed 2 routes it in minutes. s10_fmax/RESULT.md.
  nextpnr-ecp5 --85k --package CABGA381 --speed 6 --freq "$FREQ" --seed "${SEED:-1}" \
    --json "build/fpga/s10/${name}.json" \
    --textcfg "build/fpga/s10/${name}.cfg" \
    > "build/fpga/s10/${name}_nextpnr.log" 2>&1 || true
  rc=$?
  log="build/fpga/s10/${name}_nextpnr.log"
  # Do NOT key on the exit code alone. nextpnr exits 1 for a route that
  # COMPLETES but misses --freq ("ERROR: Max frequency ... (FAIL at N MHz)",
  # "0 warnings, 1 error", then "Program finished normally.") — that is the
  # normal, documented outcome here: both shipped configs miss timing at every
  # documented FREQ, and the reported Fmax is exactly the honest number this
  # baseline exists to record. Treating rc!=0 as fatal would make the script
  # abort on every legitimate invocation — a gate that always fails, which is
  # as useless as one that never fails.
  # The real defect being closed is quoting a CRASHED or KILLED route's log as
  # if it were a post-route result. So gate on the completion SIGNATURE (house
  # pattern: verif/top/l3/mutate.py — require the expected marker, not "went
  # red"), plus signal deaths, which is how the two long ship routes died.
  if [ "$rc" -ge 128 ]; then
      echo "== [$name] FATAL: nextpnr-ecp5 killed by signal (rc=$rc, e.g. OOM/timeout)." >&2
      echo "== A killed route has no honest post-route Fmax; refusing to quote it." >&2
      tail -5 "$log" >&2
      exit "$rc"
  fi
  if ! grep -q "Program finished normally\." "$log"; then
      echo "== [$name] FATAL: nextpnr-ecp5 did not finish (rc=$rc, no completion" >&2
      echo "== footer) — crashed or aborted mid-route; refusing to quote it." >&2
      tail -5 "$log" >&2
      exit "${rc:-1}"
  fi
  # Route completed: rc may be 0 (met timing) or 1 (completed, missed --freq).
  grep -E "Max frequency" "$log" | tail -2 || {
      echo "== [$name] FATAL: route completed but log has no 'Max frequency' line" >&2
      exit 1
    }
}

run_cfg b0_proxy "" kvq_engine

# B1: ship-config wrapper (mask ROM via $readmemh at synth time)
cat > build/fpga/s10/kvq_ship_top.sv <<'EOF'
// S10 B1 wrapper: kvq_engine at the apex_top ship parameters (cq4p engine).
module kvq_ship_top (
    input  wire clk, rst_n,
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
    input  wire flush_req, output wire irq, output wire evict_needed,
    output wire [6:0] evict_addr
);
    kvq_engine #(
        .VECTOR_DIM(64), .TIER(2), .KEY_GROUP(128), .OUTLIER_K(2),
        .SCALE_SETS(8), .SRAM_DEPTH(128),
        .MASK_FILE("golden/vectors/d64_T128_G64__CQ4plus/outlier_mask.u8.hex")
    ) u_eng (.*);
endmodule
EOF
run_cfg b1_ship "build/fpga/s10/kvq_ship_top.sv" kvq_ship_top

echo "== S10 BASELINE DONE $(date)"
