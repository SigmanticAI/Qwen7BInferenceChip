// tb_layer.sv — FULL decoder-layer COMPOSITION testbench.
//
// Drives the three NEW RTL blocks built this session — rope, asu_silu,
// residual_add — with the bit-exact stage tensors of a real APEX decoder-layer
// decode step (golden apex_golden.transformer.decoder_layer_fx, emitted by
// gen_layer_vectors.py) IN LAYER-DATAFLOW ORDER:
//
//   ATTENTION HALF : per HF half-split channel pair, the decode-token query
//     (m=T) and every cache key row (m=t) are RoPE-rotated on the fp16 Q/K
//     write bus by `rope`; checked === vs golden rope_fx.
//   FFN HALF       : every d_ffn SwiGLU gate activation, quantized to the Q5.10
//     feeder, is passed through `asu_silu`; checked === vs golden silu_fx.
//   RESIDUAL       : both C-6 fp16 residual adds of the layer (r1 = X[T]+attn,
//     r2 = r1+ffn) are computed element-wise by `residual_add`; checked === vs
//     the golden fp16 residual primitive.
//
// The GEMM / RMSNorm / attention-core / KVQ stages between these are supplied
// from the SAME golden decoder_layer_fx intermediates (they are already RTL-
// verified bit-exact in verif/mxe, verif/asu, verif/top/l3), so this is a
// layer-level composition of the new ops on real layer tensors.
//
// All three DUTs are combinational; every vector is applied and compared with
// no tolerance (===). A time-based $fatal watchdog bounds the run.
//
// Sims: Verilator (--binary --timing --assert) PRIMARY; Icarus SECONDARY.

module tb_layer;

  `include "layer_nvec.svh"   // LAYER_ROPE_NVEC / LAYER_SILU_NVEC / LAYER_RES_NVEC

  // ── DUTs ───────────────────────────────────────────────────────────────────
  logic [13:0] rp_u_q;
  logic [15:0] rp_x_i, rp_x_ih, rp_y_i, rp_y_ih;
  rope u_rope (.u_q(rp_u_q), .x_i(rp_x_i), .x_ih(rp_x_ih),
               .y_i(rp_y_i), .y_ih(rp_y_ih));

  logic signed [15:0] su_xq, su_yq;
  asu_silu u_silu (.x_q(su_xq), .y_q(su_yq));

  logic [15:0] rs_a, rs_b, rs_y;
  residual_add u_res (.a(rs_a), .b(rs_b), .y(rs_y));

  // ── oracle streams ─────────────────────────────────────────────────────────
  logic [13:0] v_rp_u   [0:LAYER_ROPE_NVEC-1];
  logic [15:0] v_rp_xi  [0:LAYER_ROPE_NVEC-1];
  logic [15:0] v_rp_xih [0:LAYER_ROPE_NVEC-1];
  logic [15:0] v_rp_ei  [0:LAYER_ROPE_NVEC-1];
  logic [15:0] v_rp_eih [0:LAYER_ROPE_NVEC-1];
  logic [15:0] v_su_xq  [0:LAYER_SILU_NVEC-1];
  logic [15:0] v_su_exp [0:LAYER_SILU_NVEC-1];
  logic [15:0] v_rs_a   [0:LAYER_RES_NVEC-1];
  logic [15:0] v_rs_b   [0:LAYER_RES_NVEC-1];
  logic [15:0] v_rs_exp [0:LAYER_RES_NVEC-1];

  int e_rope, e_silu, e_res;

  initial begin
    #40_000_000;
    $fatal(1, "WATCHDOG: layer composition did not finish in time");
  end

  initial begin
    e_rope = 0; e_silu = 0; e_res = 0;
    $readmemh("build/layer_rope_u_q.hex",   v_rp_u);
    $readmemh("build/layer_rope_x_i.hex",   v_rp_xi);
    $readmemh("build/layer_rope_x_ih.hex",  v_rp_xih);
    $readmemh("build/layer_rope_exp_i.hex", v_rp_ei);
    $readmemh("build/layer_rope_exp_ih.hex", v_rp_eih);
    $readmemh("build/layer_silu_xq.hex",    v_su_xq);
    $readmemh("build/layer_silu_exp.hex",   v_su_exp);
    $readmemh("build/layer_res_a.hex",      v_rs_a);
    $readmemh("build/layer_res_b.hex",      v_rs_b);
    $readmemh("build/layer_res_exp.hex",    v_rs_exp);

    // ── ATTENTION HALF: RoPE on the fp16 Q/K bus ──────────────────────────────
    for (int k = 0; k < LAYER_ROPE_NVEC; k++) begin
      rp_u_q  = v_rp_u[k];
      rp_x_i  = v_rp_xi[k];
      rp_x_ih = v_rp_xih[k];
      #1;
      if (rp_y_i !== v_rp_ei[k] || rp_y_ih !== v_rp_eih[k]) begin
        e_rope++;
        if (e_rope <= 10)
          $display("FAIL [rope] k=%0d u_q=%0d x=(%04x,%04x): got (%04x,%04x) exp (%04x,%04x)",
                   k, rp_u_q, rp_x_i, rp_x_ih, rp_y_i, rp_y_ih,
                   v_rp_ei[k], v_rp_eih[k]);
      end
    end

    // ── FFN HALF: SiLU gate ───────────────────────────────────────────────────
    for (int k = 0; k < LAYER_SILU_NVEC; k++) begin
      su_xq = v_su_xq[k];
      #1;
      if (su_yq !== $signed(v_su_exp[k])) begin
        e_silu++;
        if (e_silu <= 10)
          $display("FAIL [silu] k=%0d x_q=%0d: got %04x exp %04x",
                   k, su_xq, su_yq, v_su_exp[k]);
      end
    end

    // ── RESIDUAL: both C-6 fp16 residual adds ─────────────────────────────────
    for (int k = 0; k < LAYER_RES_NVEC; k++) begin
      rs_a = v_rs_a[k];
      rs_b = v_rs_b[k];
      #1;
      if (rs_y !== v_rs_exp[k]) begin
        e_res++;
        if (e_res <= 10)
          $display("FAIL [res] k=%0d a=%04x b=%04x: got %04x exp %04x",
                   k, rs_a, rs_b, rs_y, v_rs_exp[k]);
      end
    end

    // ── verdict ───────────────────────────────────────────────────────────────
    $display("---------------------------------------------------------------");
    $display("LAYER COMPOSITION: rope %0d/%0d  silu %0d/%0d  residual %0d/%0d",
             LAYER_ROPE_NVEC - e_rope, LAYER_ROPE_NVEC,
             LAYER_SILU_NVEC - e_silu, LAYER_SILU_NVEC,
             LAYER_RES_NVEC  - e_res,  LAYER_RES_NVEC);
    if (e_rope == 0 && e_silu == 0 && e_res == 0) begin
      $display("LAYER COMPOSITION: ALL TESTS PASSED (t=%0t)", $time);
      $display("  attention-half RoPE + FFN-half SiLU + both residual adds");
      $display("  bit-exact vs golden decoder_layer_fx over 2 configs (H2/hd64/T8, H1/hd128/T16)");
    end else begin
      $display("LAYER COMPOSITION: FAILURES rope=%0d silu=%0d res=%0d",
               e_rope, e_silu, e_res);
      $fatal(1, "layer composition mismatch");
    end
    $finish;
  end

endmodule
