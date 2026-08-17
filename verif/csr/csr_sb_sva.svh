// csr_sb_sva.svh — CSR contract checker, bound INTO csr_regs by tb_csr_sb.sv
// (whitebox bind: w1c_fire / wr_srst / wr_flush are csr_regs internals).
// Compiled under --assert in every build — a gate, not decoration (D-012).
//
// Written for the Verilator 5.x SVA subset: simple concurrent assertions
// with |-> / |=> / $past only.
//
// Checked contracts (csr_regs.sv header):
//   - bus response discipline: ready pulses exactly one cycle after every
//     request, and never without one;
//   - desc_error sticky: SET on the pulse (set wins over a same-cycle W1C),
//     HELD while no W1C, CLEARED by an uncontested W1C;
//   - soft_reset / flush pulse outputs mirror their write strobes exactly
//     (1-cycle, self-clearing, never spurious).

`ifndef CSR_SB_SVA_SVH
`define CSR_SB_SVA_SVH

module csr_sb_sva (
  input logic clk,
  input logic rst_n,
  input logic read,
  input logic write,
  input logic ready,
  input logic desc_error_i,
  input logic desc_error_sticky,
  input logic w1c_fire,
  input logic soft_reset,
  input logic flush,
  input logic wr_srst,
  input logic wr_flush
);

  ap_ready_follows: assert property (@(posedge clk) disable iff (!rst_n)
    (read || write) |=> ready)
    else $error("[SVA csr] request not acknowledged one cycle later");

  ap_ready_causal: assert property (@(posedge clk) disable iff (!rst_n)
    ready |-> $past(read || write))
    else $error("[SVA csr] spurious ready (no request the cycle before)");

  // set WINS over a same-cycle W1C — never lose an error
  ap_sticky_set: assert property (@(posedge clk) disable iff (!rst_n)
    desc_error_i |=> desc_error_sticky)
    else $error("[SVA csr] desc_error pulse did not set the sticky bit");

  ap_sticky_hold: assert property (@(posedge clk) disable iff (!rst_n)
    (desc_error_sticky && !w1c_fire) |=> desc_error_sticky)
    else $error("[SVA csr] sticky bit dropped without a W1C write");

  ap_sticky_clear: assert property (@(posedge clk) disable iff (!rst_n)
    (w1c_fire && !desc_error_i) |=> !desc_error_sticky)
    else $error("[SVA csr] uncontested W1C did not clear the sticky bit");

  ap_srst_pulse: assert property (@(posedge clk) disable iff (!rst_n)
    soft_reset == $past(wr_srst))
    else $error("[SVA csr] soft_reset pulse does not mirror its write strobe");

  ap_flush_pulse: assert property (@(posedge clk) disable iff (!rst_n)
    flush == $past(wr_flush))
    else $error("[SVA csr] flush pulse does not mirror its write strobe");

endmodule

`endif // CSR_SB_SVA_SVH
