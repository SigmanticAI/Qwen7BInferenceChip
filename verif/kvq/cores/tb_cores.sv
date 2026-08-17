// tb_cores.sv — clean-room unit testbench for the KVQ per-channel INT4 datapath
// cores (cq_value_path + cq_key_path and the fp arithmetic cores they compose).
// Streams FRESH golden vectors (gen_cores_vectors.py, arbiter cq_codec.py)
// and checks bit-exact: per-token value scale/payload/fp32 readback, and per-
// channel grouped-key scales/codes/records/fp32 readback including the FP16
// outlier identity lane (-0.0 -> 0x8000_0000). One elaboration per D/tier/G.
`timescale 1ns/1ps
`default_nettype none
module tb_cores;

  parameter int    D        = 64;
  parameter int    DW       = 16;
  parameter int    TIER     = 1;      // 0=CQ-8, 1=CQ-4, 2=CQ-4+
  parameter int    G        = 8;
  parameter int    K_OUT    = 0;
  parameter        VECDIR   = "build/d64_cq4";
  parameter        CFGNAME  = "d64_cq4";
  parameter int    MAX_CYCLES = 5_000_000;

  localparam int BITS      = (TIER == 0) ? 8 : 4;
  localparam int PAY_BYTES = (TIER == 0) ? D : D/2;
  localparam int PAY_BITS  = D * BITS;
  localparam int IDXW      = $clog2(D);

  // ── clock / reset / watchdog ────────────────────────────────────────────
  reg clk;
  initial begin clk = 1'b0; forever #5 clk = ~clk; end
  reg rst_n;
  initial begin
    #(MAX_CYCLES * 10);
    $fatal(1, "WATCHDOG: %s exceeded %0d cycles", CFGNAME, MAX_CYCLES);
  end

  integer checks = 0, fails = 0;
  task automatic note_fail(input string m);
    begin fails = fails + 1; if (fails <= 30) $display("  FAIL: %s", m); end
  endtask

  // ── outlier mask ─────────────────────────────────────────────────────────
  reg [D-1:0] omask;
  reg [7:0]   mask_mem [0:D-1];

  // ── value path ─────────────────────────────────────────────────────────--
  reg             cqv_in_valid;
  reg [D*DW-1:0]  vtok;
  wire            cqv_busy, cqv_ov;
  wire [15:0]     cqv_scale;
  wire [D*8-1:0]  cqv_codes, cqv_pay;
  reg [D*8-1:0]   vdec_codes;
  reg [15:0]      vdec_scale;
  reg [IDXW-1:0]  vdec_idx;
  wire [31:0]     vdec_hat;

  cq_value_path #(.D(D), .DW(DW)) u_vp (
    .clk(clk), .rst_n(rst_n), .clear(1'b0), .bits(4'(BITS)),
    .in_valid(cqv_in_valid), .in_vec(vtok), .busy(cqv_busy),
    .out_valid(cqv_ov), .out_scale(cqv_scale),
    .out_codes(cqv_codes), .out_pay(cqv_pay),
    .dec_codes(vdec_codes), .dec_scale(vdec_scale),
    .dec_idx(vdec_idx), .dec_hat(vdec_hat)
  );

  // ── key path (always elaborated; idle for TIER 0) ────────────────────────
  reg             kp_in_valid, kp_gstart, kp_glast, kp_flush;
  reg [D*DW-1:0]  ktok;
  wire            kp_busy, kp_gvalid, kp_tvalid;
  wire [$clog2(G)-1:0]   kp_tidx;
  wire [$clog2(G+1)-1:0] kp_gout;
  wire [D*DW-1:0] kp_scales, kp_emit;
  wire [(D/2)*8-1:0] kp_tpay;
  wire [D*8-1:0]  kp_tcodes;
  reg [D*8-1:0]   kdec_codes;
  reg [D*DW-1:0]  kdec_scales;
  reg [IDXW-1:0]  kdec_idx;
  wire [31:0]     kdec_hat;

  cq_key_path #(.D(D), .DW(DW), .G(G)) u_kp (
    .clk(clk), .rst_n(rst_n), .clear(1'b0), .outlier_mask(omask),
    .in_valid(kp_in_valid), .in_vec(ktok),
    .group_start(kp_gstart), .group_last(kp_glast), .flush(kp_flush),
    .busy(kp_busy), .group_valid(kp_gvalid),
    .scales_bus(kp_scales), .g_out(kp_gout),
    .tok_valid(kp_tvalid), .tok_idx(kp_tidx), .tok_pay(kp_tpay),
    .tok_codes(kp_tcodes), .emit_vec(kp_emit),
    .dec_codes(kdec_codes), .dec_scales(kdec_scales),
    .dec_idx(kdec_idx), .dec_hat(kdec_hat)
  );

  // ── scratch storage for a group ──────────────────────────────────────────
  reg [DW-1:0] graw   [0:G-1][0:D-1];
  reg [DW-1:0] gfield [0:D-1];
  reg [7:0]    gcode  [0:G-1][0:D-1];
  reg [DW-1:0] gsfield[0:G-1][0:D-1];
  reg [31:0]   ghat   [0:G-1][0:D-1];

  integer fd, i, j, t, g, full;
  reg [15:0] w16; reg [7:0] w8; reg [31:0] w32; reg [15:0] s_exp;
  reg [PAY_BITS-1:0] pay_exp; reg [D*8-1:0] code_exp;
  string vpath, kpath, mpath;

  // sink structurally-unused observation bits (TB is not a synthesis target)
  wire _unused_ok = &{1'b0, cqv_pay, cqv_codes, kp_tpay, kp_gout,
                      cqv_busy, kp_busy, kp_gvalid};

  // ── value token drive ────────────────────────────────────────────────────
  task automatic run_value(int nchan_hat);
    begin
      @(negedge clk); cqv_in_valid = 1'b1;
      @(negedge clk); cqv_in_valid = 1'b0;
      while (!cqv_ov) @(negedge clk);            // registered out_valid pulse
      checks = checks + 1;
      if (cqv_scale !== s_exp)
        note_fail($sformatf("val scale got=%h exp=%h", cqv_scale, s_exp));
      checks = checks + 1;
      if (cqv_pay[PAY_BITS-1:0] !== pay_exp)
        note_fail($sformatf("val pay got=%h exp=%h", cqv_pay[PAY_BITS-1:0], pay_exp));
      // dequant sweep
      vdec_scale = s_exp; vdec_codes = code_exp;
      for (j = 0; j < nchan_hat; j = j + 1) begin
        vdec_idx = IDXW'(j); #1;
        checks = checks + 1;
        if (vdec_hat !== ghat[0][j])
          note_fail($sformatf("val hat[%0d] got=%h exp=%h", j, vdec_hat, ghat[0][j]));
      end
    end
  endtask

  // ── key group drive + collect ────────────────────────────────────────────
  task automatic run_key_group(int gg, int is_full);
    integer b, kk, c;
    begin
      for (i = 0; i < gg; i = i + 1) begin
        @(negedge clk);
        kp_in_valid = 1'b1;
        for (c = 0; c < D; c = c + 1) ktok[c*DW +: DW] = graw[i][c];
        kp_gstart = (i == 0);
        kp_glast  = ((is_full != 0) && (i == gg-1));
      end
      @(negedge clk); kp_in_valid = 1'b0; kp_gstart = 1'b0; kp_glast = 1'b0;
      if (is_full == 0) begin
        @(negedge clk); kp_flush = 1'b1;
        @(negedge clk); kp_flush = 1'b0;
      end
      // collect gg emit beats
      b = 0;
      while (b < gg) begin
        @(negedge clk);
        if (kp_tvalid) begin
          checks = checks + 1;
          if (kp_tidx !== ($clog2(G))'(b))
            note_fail($sformatf("key tidx got=%0d exp=%0d", kp_tidx, b));
          // emit_vec == raw token
          for (c = 0; c < D; c = c + 1) begin
            checks = checks + 1;
            if (kp_emit[c*DW +: DW] !== graw[b][c])
              note_fail($sformatf("key emit t%0d c%0d got=%h exp=%h",
                        b, c, kp_emit[c*DW +: DW], graw[b][c]));
          end
          // keep-channel scale field + compacted keep codes
          kk = 0;
          for (c = 0; c < D; c = c + 1) begin
            if (!omask[c]) begin
              checks = checks + 1;
              if (kp_scales[c*DW +: DW] !== gfield[c])
                note_fail($sformatf("key scale c%0d got=%h exp=%h",
                          c, kp_scales[c*DW +: DW], gfield[c]));
              checks = checks + 1;
              if (kp_tcodes[kk*8 +: 4] !== gcode[b][c][3:0])
                note_fail($sformatf("key code t%0d c%0d got=%h exp=%h",
                          b, c, kp_tcodes[kk*8 +: 4], gcode[b][c][3:0]));
              kk = kk + 1;
            end
          end
          b = b + 1;
        end
      end
      // dequant readback per token (per-channel stored code + field)
      for (b = 0; b < gg; b = b + 1) begin
        for (c = 0; c < D; c = c + 1) begin
          kdec_codes[c*8 +: 8]   = gcode[b][c];
          kdec_scales[c*DW +: DW] = gsfield[b][c];
        end
        for (c = 0; c < D; c = c + 1) begin
          kdec_idx = IDXW'(c); #1;
          checks = checks + 1;
          if (kdec_hat !== ghat[b][c])
            note_fail($sformatf("key hat t%0d c%0d got=%h exp=%h",
                      b, c, kdec_hat, ghat[b][c]));
        end
      end
      @(negedge clk);      // settle before next group
    end
  endtask

  initial begin
    cqv_in_valid = 0; vtok = '0; vdec_codes = '0; vdec_scale = '0; vdec_idx = '0;
    kp_in_valid = 0; kp_gstart = 0; kp_glast = 0; kp_flush = 0; ktok = '0;
    kdec_codes = '0; kdec_scales = '0; kdec_idx = '0; omask = '0;

    if (K_OUT > 0) begin
      mpath = {VECDIR, "/mask.u8.hex"};
      $readmemh(mpath, mask_mem);
      for (i = 0; i < D; i = i + 1) omask[i] = mask_mem[i][0];
    end

    rst_n = 0; repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);
    $display("==== tb_cores %s: D=%0d TIER=%0d G=%0d K_OUT=%0d", CFGNAME, D, TIER, G, K_OUT);

    // ── VALUE tests ─────────────────────────────────────────────────────────
    vpath = {VECDIR, "/val.txt"};
    fd = $fopen(vpath, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vpath);
    forever begin
      // raw token
      if ($fscanf(fd, "%h", w16) != 1) break;
      vtok[0 +: DW] = w16;
      for (j = 1; j < D; j = j + 1) begin
        void'($fscanf(fd, "%h", w16)); vtok[j*DW +: DW] = w16;
      end
      void'($fscanf(fd, "%h", s_exp));
      pay_exp = '0;
      for (j = 0; j < PAY_BYTES; j = j + 1) begin
        void'($fscanf(fd, "%h", w8)); pay_exp[j*8 +: 8] = w8;
      end
      code_exp = '0;
      for (j = 0; j < D; j = j + 1) begin
        void'($fscanf(fd, "%h", w8)); code_exp[j*8 +: 8] = w8;
      end
      for (j = 0; j < D; j = j + 1) begin
        void'($fscanf(fd, "%h", w32)); ghat[0][j] = w32;
      end
      run_value(D);
    end
    $fclose(fd);

    // ── KEY tests (INT4 tiers only) ────────────────────────────────────────
    if (TIER != 0) begin
      kpath = {VECDIR, "/key.txt"};
      fd = $fopen(kpath, "r");
      if (fd == 0) $fatal(1, "cannot open %s", kpath);
      forever begin
        if ($fscanf(fd, "%d", g) != 1) break;
        void'($fscanf(fd, "%d", full));
        for (t = 0; t < g; t = t + 1)
          for (j = 0; j < D; j = j + 1) begin
            void'($fscanf(fd, "%h", w16)); graw[t][j] = w16;
          end
        for (j = 0; j < D; j = j + 1) begin
          void'($fscanf(fd, "%h", w16)); gfield[j] = w16;
        end
        for (t = 0; t < g; t = t + 1)
          for (j = 0; j < D; j = j + 1) begin
            void'($fscanf(fd, "%h", w8)); gcode[t][j] = w8;
          end
        for (t = 0; t < g; t = t + 1)
          for (j = 0; j < D; j = j + 1) begin
            void'($fscanf(fd, "%h", w16)); gsfield[t][j] = w16;
          end
        for (t = 0; t < g; t = t + 1)
          for (j = 0; j < D; j = j + 1) begin
            void'($fscanf(fd, "%h", w32)); ghat[t][j] = w32;
          end
        run_key_group(g, full);
      end
      $fclose(fd);
    end

    $display("============================================================");
    $display("CONFIG %s: checks=%0d fails=%0d", CFGNAME, checks, fails);
    if (fails == 0) $display("CORES RESULT [%s]: PASS", CFGNAME);
    else            $display("CORES RESULT [%s]: FAIL", CFGNAME);
    $finish;
  end

endmodule
`default_nettype wire
