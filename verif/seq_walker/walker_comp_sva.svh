// walker_comp_sva.svh — B1 composite-unit precondition checker, bound INTO
// seq_walker_comp by tb_comp_sb.sv. Compiled under --assert (D-012).
//
// Written for the Verilator 5.x SVA subset (simple concurrent assertions).
//
// THE CONTRACT IT GUARDS (seq_walker_comp.sv P1): every consumed scale is a
// POSITIVE NORMAL fp16. The composite unpack takes that as given — it adds the
// hidden bit and rebiases the exponent directly, with no denormal or sign
// handling. A denormal, zero, negative or inf/nan input would corrupt the
// result silently while STILL PASSING the software oracle's own asserts (those
// reject only sign<=0 and inf/nan; all 1,023 positive denormals pass them).
// The exclusion of denormals comes from the EPS=2^-14 floor every scale
// producer applies, NOT from the oracle — so it must be asserted here, at the
// consumer, or it is not checked anywhere.
`ifndef WALKER_COMP_SVA_SVH
`define WALKER_COMP_SVA_SVH

module walker_comp_sva (
  input logic        clk,
  input logic        rst_n,
  input logic        in_mul,     // composite is consuming its operands
  input logic [15:0] sk_q,       // cached record scale (s_k[t] or s_v[t])
  input logic [15:0] s_q_q       // latched s_q
);

  // only sign+exponent decide positive-normality; the mantissa is irrelevant
  function automatic logic pos_normal(input logic [15:10] f);
    pos_normal = !f[15] && (f[14:10] != 5'd0) && (f[14:10] != 5'd31);
  endfunction

  ap_sk_pos_normal: assert property (@(posedge clk) disable iff (!rst_n)
    in_mul |-> pos_normal(sk_q[15:10]))
    else $error("composite consumed a non-positive-normal cached scale: %04h",
                sk_q);

  ap_sq_pos_normal: assert property (@(posedge clk) disable iff (!rst_n)
    in_mul |-> pos_normal(s_q_q[15:10]))
    else $error("composite consumed a non-positive-normal s_q: %04h", s_q_q);

endmodule

`endif // WALKER_COMP_SVA_SVH
