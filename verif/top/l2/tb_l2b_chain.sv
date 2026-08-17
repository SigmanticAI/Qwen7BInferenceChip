// tb_l2b_chain.sv — Layer-2 chain (b): REAL kvq_engine -> seam_feeder_quant
// -> mxe_top, wired exactly as rtl/top/apex_top.sv wires them (KVQ fp32 read
// bus into the feeder input, feeder INT8 code beats into the MXE activation
// stream; in apex_top the wstage sits between feeder and MXE as a pure
// staging buffer — bypassed here so the D-021 numeric seam itself is the
// thing under test).
//
// What this chain proves:
//   * ALL THREE TIERS (CQ-8 / CQ-4 / CQ-4+ incl. outlier masks) through the
//     REAL engine: write path (key groups incl. PARTIAL-GROUP FLUSH, value
//     tokens), fp32 readback bursts, feeder requant, MXE accumulation —
//     bit-exact vs apex_golden (cq_codec + quant_rows_i8 + gemm_i8);
//   * k8/v8 code planes dumped THROUGH the MXE with INT8 identity weights
//     (every code bit-exact), q8·k8 score accumulators with a q8 weight
//     column (the L3 Q·K̂ᵀ shape);
//   * backpressure storms on the MXE result stream and the feeder scale
//     sideband, stall storms on every TB-driven stream;
//   * mid-operation reset (rst_n mid write-burst / mid readback) + full
//     clean re-run; KVQ D-020 soft reset (CTRL) mid-token, then clean.
//
// Command grammar:
//   KVW a d / KVP a m e   engine AXI-Lite write / poll-until-(v&m)==e
//   KT <D vals> / VT <D vals>   one key/value token (fp16 beats, tuser=0/1)
//   FLUSH                 pulse flush_req (D-008)
//   FJ rows               feeder job
//   RD addr               readback: write READ_ADDR (burst feeds the feeder)
//   DESC/WB               MXE descriptor / weight beat
//   EFS n <n {f16,last}>  expected feeder scales
//   ERO l0..l7 last       expected MXE result beat
//   STK mask exp          {fq_job, mxe_desc}
//   IDLE / RST / ENDTAB / DONE
//   +bp_mode / +stall_mode / +seed as tb_l2a_chain

`include "apex_stream_sva.svh"
`include "apex_stream1_sva.svh"

module tb_l2b_chain;

  import apex_pkg::*;

  parameter int unsigned D_CFG     = 64;
  parameter int unsigned TIER_CFG  = 0;
  parameter int unsigned G_CFG     = 16;
  parameter int unsigned OUTK_CFG  = 0;
  parameter int unsigned DEPTH_CFG = 512;
  parameter MASKF                  = "";

  localparam int unsigned WAIT_LIMIT = 400000;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 30_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: l2b did not finish within %0d cycles", watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_l2b_chain);
    end
  end

  int unsigned prng = 32'hC0FFEE02;
  function automatic int unsigned rnd();
    prng = prng ^ (prng << 13);
    prng = prng ^ (prng >> 17);
    prng = prng ^ (prng << 5);
    return prng;
  endfunction

  int unsigned bp_mode = 0;
  int unsigned stall_mode = 0;

  // ── KVQ engine ────────────────────────────────────────────────────────────
  logic [7:0]  kv_awaddr, kv_araddr;
  logic        kv_awvalid, kv_awready, kv_wvalid, kv_wready;
  logic [31:0] kv_wdata, kv_rdata;
  logic [1:0]  kv_bresp, kv_rresp;
  logic        kv_bvalid, kv_bready, kv_arvalid, kv_arready;
  logic        kv_rvalid, kv_rready;
  logic [15:0] s_tdata;
  logic        s_tvalid, s_tready, s_tlast, s_tuser;
  logic [31:0] m_tdata;
  logic        m_tvalid, m_tready, m_tlast;
  logic        flush_r, kv_irq, kv_evict_needed;
  logic [$clog2(DEPTH_CFG)-1:0] kv_evict_addr;
  wire         kv_mask_valid;   // D-027 stage-4 export (sunk; verif/kvq/mask checks it)

  kvq_engine #(
    .VECTOR_DIM (int'(D_CFG)),
    .TIER       (int'(TIER_CFG)),
    .KEY_GROUP  (int'(G_CFG)),
    .OUTLIER_K  (int'(OUTK_CFG)),
    .SCALE_WIDTH(16),
    .SRAM_DEPTH (int'(DEPTH_CFG)),
    .COORD_WIDTH(16),
    .OUT_WIDTH  (32),
    .MASK_FILE  (MASKF)
  ) u_kvq (
    .clk              (clk),
    .rst_n            (rst_n),
    .axil_awaddr      (kv_awaddr),
    .axil_awvalid     (kv_awvalid),
    .axil_awready     (kv_awready),
    .axil_wdata       (kv_wdata),
    .axil_wvalid      (kv_wvalid),
    .axil_wready      (kv_wready),
    .axil_bresp       (kv_bresp),
    .axil_bvalid      (kv_bvalid),
    .axil_bready      (kv_bready),
    .axil_araddr      (kv_araddr),
    .axil_arvalid     (kv_arvalid),
    .axil_arready     (kv_arready),
    .axil_rdata       (kv_rdata),
    .axil_rresp       (kv_rresp),
    .axil_rvalid      (kv_rvalid),
    .axil_rready      (kv_rready),
    .s_axis_kv_tdata  (s_tdata),
    .s_axis_kv_tvalid (s_tvalid),
    .s_axis_kv_tready (s_tready),
    .s_axis_kv_tlast  (s_tlast),
    .s_axis_kv_tuser  (s_tuser),
    .m_axis_kv_tdata  (m_tdata),
    .m_axis_kv_tvalid (m_tvalid),
    .m_axis_kv_tready (m_tready),
    .m_axis_kv_tlast  (m_tlast),
    .flush_req        (flush_r),
    .irq              (kv_irq),
    .evict_needed     (kv_evict_needed),
    .evict_addr       (kv_evict_addr),
    .mask_valid       (kv_mask_valid)
  );

  // ── feeder (KVQ fp32 read bus -> INT8 codes + fp16 scales) ───────────────
  logic             fj_valid, fj_ready;
  logic [DIM_W-1:0] fj_rows;
  logic fq_job_error, fq_job_error_sticky, fq_busy, fq_done;
  logic        fq_out_valid, fq_out_ready;
  lane8_beat_t fq_out_beat;
  logic        fs_valid, fs_ready, fs_last;
  logic [15:0] fs_data;

  seam_feeder_quant #(.D(D_CFG), .ROWS_MAX(64)) u_feeder (
    .clk              (clk),
    .rst_n            (rst_n),
    .job_valid        (fj_valid),
    .job_ready        (fj_ready),
    .job_rows         (fj_rows),
    .job_error        (fq_job_error),
    .job_error_sticky (fq_job_error_sticky),
    .busy             (fq_busy),
    .done             (fq_done),
    .in_valid         (m_tvalid),
    .in_ready         (m_tready),
    .in_data          (m_tdata),
    .out_valid        (fq_out_valid),
    .out_ready        (fq_out_ready),
    .out_beat         (fq_out_beat),
    .scl_valid        (fs_valid),
    .scl_ready        (fs_ready),
    .scl_data         (fs_data),
    .scl_last         (fs_last)
  );

  // ── MXE ───────────────────────────────────────────────────────────────────
  logic         desc_valid, desc_ready;
  logic [127:0] desc_r;
  logic         mxe_desc_error, mxe_desc_error_sticky, mxe_busy, mxe_done;
  logic         wgt_valid, wgt_ready;
  logic [63:0]  wgt_data;
  logic         res_valid, res_ready;
  lane32_beat_t res_beat;

  mxe_top u_mxe (
    .clk               (clk),
    .rst_n             (rst_n),
    .desc_valid        (desc_valid),
    .desc_ready        (desc_ready),
    .desc              (mxe_desc_t'(desc_r)),
    .desc_error        (mxe_desc_error),
    .desc_error_sticky (mxe_desc_error_sticky),
    .busy              (mxe_busy),
    .done              (mxe_done),
    .act_valid         (fq_out_valid),
    .act_ready         (fq_out_ready),
    .act_beat          (fq_out_beat),
    .wgt_valid         (wgt_valid),
    .wgt_ready         (wgt_ready),
    .wgt_beat          (lane8_beat_t'({wgt_data, 1'b0})),
    .res_valid         (res_valid),
    .res_ready         (res_ready),
    .res_beat          (res_beat)
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, kv_mask_valid, kv_bresp, kv_rresp, kv_awready, kv_wready,
                       kv_arready, kv_irq, kv_evict_needed, kv_evict_addr,
                       m_tlast, fq_done, mxe_done, fq_job_error,
                       mxe_desc_error, fs_last};

  // ── SVA (§5/D-006 compiled — D-012 gate) ─────────────────────────────────
  bind mxe_top apex_stream_sva u_sva_mxe (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(desc_valid), .desc_ready(desc_ready), .desc(desc),
    .desc_error(desc_error), .desc_error_sticky(desc_error_sticky),
    .busy(busy), .done(done),
    .act_valid(act_valid), .act_ready(act_ready), .act_beat(act_beat),
    .wgt_valid(wgt_valid), .wgt_ready(wgt_ready), .wgt_beat(wgt_beat),
    .res_valid(res_valid), .res_ready(res_ready), .res_beat(res_beat));

  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(65), .NAME("fq.out"))
    u_sva_o (.clk(clk), .rst_n(rst_n), .valid(out_valid), .ready(out_ready),
             .data({out_beat.data, out_beat.last}));
  bind seam_feeder_quant apex_stream1_sva #(.WIDTH(17), .NAME("fq.scl"))
    u_sva_s (.clk(clk), .rst_n(rst_n), .valid(scl_valid), .ready(scl_ready),
             .data({scl_data, scl_last}));
  // §5 on the KVQ read bus (the historical D-007 bug surface)
  bind kvq_engine apex_stream1_sva #(.WIDTH(33), .NAME("kvq.m_axis"))
    u_sva_m (.clk(clk), .rst_n(rst_n), .valid(m_axis_kv_tvalid),
             .ready(m_axis_kv_tready),
             .data({m_axis_kv_tdata, m_axis_kv_tlast}));

  // ── backpressure: res_ready + fs_ready ───────────────────────────────────
  int unsigned bp_hold, bp_hold2;
  always @(posedge clk) begin
    if (!rst_n) begin
      res_ready <= 1'b1;
      fs_ready  <= 1'b1;
      bp_hold   <= 0;
      bp_hold2  <= 0;
    end else begin
      unique case (bp_mode)
        0: begin
          res_ready <= 1'b1;
          fs_ready  <= 1'b1;
        end
        1: begin
          res_ready <= rnd()[0];
          fs_ready  <= rnd()[1];
        end
        default: begin
          if (bp_hold != 0) begin
            bp_hold   <= bp_hold - 1;
            res_ready <= 1'b0;
          end else begin
            res_ready <= 1'b1;
            if (rnd() % 12 == 0) bp_hold <= 20 + (rnd() % 180);
          end
          if (bp_hold2 != 0) begin
            bp_hold2 <= bp_hold2 - 1;
            fs_ready <= 1'b0;
          end else begin
            fs_ready <= 1'b1;
            if (rnd() % 16 == 0) bp_hold2 <= 20 + (rnd() % 120);
          end
        end
      endcase
    end
  end

  task automatic stall_gap();
    int unsigned g;
    unique case (stall_mode)
      0: g = 0;
      1: g = rnd() % 4;
      default: g = (rnd() % 10 == 0) ? (20 + rnd() % 120) : (rnd() % 3);
    endcase
    repeat (g) @(negedge clk);
  endtask

  // ── scoreboard collectors ─────────────────────────────────────────────────
  int n_checks, n_errors;
  int tap_checks, tap_errors;

  logic [16:0]  efs_exp [8192];
  int efs_n, efs_i;
  logic [256:0] ro_q [$];

  always @(posedge clk) begin
    if (rst_n && fs_valid && fs_ready) begin
      if (efs_i >= efs_n) begin
        tap_errors <= tap_errors + 1;
        $error("[efs] unexpected scale %04x @%0d", fs_data, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if ({fs_data, fs_last} !== efs_exp[efs_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[efs] idx %0d: got %04x/%0b exp %04x/%0b", efs_i,
                 fs_data, fs_last, efs_exp[efs_i][16:1], efs_exp[efs_i][0]);
        end
      end
      efs_i <= efs_i + 1;
    end
    if (rst_n && res_valid && res_ready)
      ro_q.push_back({res_beat.data, res_beat.last});
  end

  // ── drivers ───────────────────────────────────────────────────────────────
  task automatic kv_wr(input logic [7:0] a, input logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    kv_awaddr  = a;
    kv_wdata   = d;
    kv_awvalid = 1'b1;
    kv_wvalid  = 1'b1;
    @(negedge clk);
    kv_awvalid = 1'b0;
    kv_wvalid  = 1'b0;
    while (!kv_bvalid) begin
      @(negedge clk);
      wd++;
      if (wd > 100) $fatal(1, "[kv] write %02x: no bvalid", a);
    end
  endtask

  task automatic kv_rd(input logic [7:0] a, output logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    kv_araddr  = a;
    kv_arvalid = 1'b1;
    @(negedge clk);
    kv_arvalid = 1'b0;
    while (!kv_rvalid) begin
      @(negedge clk);
      wd++;
      if (wd > 100) $fatal(1, "[kv] read %02x: no rvalid", a);
    end
    d = kv_rdata;
  endtask

  task automatic kv_poll(input logic [7:0] a, input logic [31:0] m,
                         input logic [31:0] e);
    int wd = 0;
    logic [31:0] v;
    forever begin
      kv_rd(a, v);
      if ((v & m) === e) break;
      repeat (20) @(posedge clk);
      wd++;
      if (wd > 50000) $fatal(1, "KVP %02x timeout (got %08x) @%0d", a, v, cyc);
    end
  endtask

  task automatic drive_tok(input logic [15:0] vals [], input logic user);
    int wd;
    for (int i = 0; i < int'(D_CFG); i++) begin
      wd = 0;
      stall_gap();
      @(negedge clk);
      s_tdata  = vals[i];
      s_tuser  = user;
      s_tlast  = (i == int'(D_CFG) - 1);
      s_tvalid = 1'b1;
      while (!s_tready) begin
        @(negedge clk);
        wd++;
        if (wd > WAIT_LIMIT) $fatal(1, "drive_tok stall @%0d", cyc);
      end
      @(negedge clk);
      s_tvalid = 1'b0;
      s_tlast  = 1'b0;
    end
  endtask

  task automatic drive_fj(input logic [DIM_W-1:0] rows);
    int wd = 0;
    @(negedge clk);
    fj_rows  = rows;
    fj_valid = 1'b1;
    while (!fj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_fj stall @%0d", cyc);
    end
    @(negedge clk);
    fj_valid = 1'b0;
  endtask

  task automatic drive_desc(input logic [127:0] d);
    int wd = 0;
    @(negedge clk);
    desc_r     = d;
    desc_valid = 1'b1;
    while (!desc_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_desc stall @%0d", cyc);
    end
    @(negedge clk);
    desc_valid = 1'b0;
  endtask

  task automatic drive_wb(input logic [63:0] w);
    int wd = 0;
    stall_gap();
    @(negedge clk);
    wgt_data  = w;
    wgt_valid = 1'b1;
    while (!wgt_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_wb stall @%0d", cyc);
    end
    @(negedge clk);
    wgt_valid = 1'b0;
  endtask

  task automatic expect_ro(input logic [255:0] lanes, input logic lst);
    int wd = 0;
    logic [256:0] got;
    while (ro_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "ERO timeout @%0d", cyc);
    end
    got = ro_q.pop_front();
    n_checks++;
    if (got !== {lanes, lst}) begin
      n_errors++;
      $error("[ERO] got %064x/%0b exp %064x/%0b",
             got[256:1], got[0], lanes, lst);
    end
  endtask

  task automatic wait_idle();
    int wd = 0;
    @(posedge clk);
    while (mxe_busy || fq_busy || m_tvalid || fq_out_valid) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "IDLE timeout @%0d", cyc);
    end
  endtask

  function automatic void chk32(input string what, input logic [31:0] got,
                                input logic [31:0] exp);
    n_checks++;
    if (got !== exp) begin
      n_errors++;
      $error("[%s] got %08x exp %08x", what, got, exp);
    end
  endfunction

  // ── interpreter ────────────────────────────────────────────────────────────
  initial begin : interp
    string vec_path, cmd, rest;
    bit done_seen;
    int fd, i;
    logic [31:0] m, e, d;
    logic [63:0] w64;
    logic [31:0] dw [4];
    logic [255:0] lanes;
    logic [15:0] tok [];
    logic [7:0]  a;

    tok = new[int'(D_CFG)];
    done_seen = 1'b0;
    rst_n = 1'b0;
    kv_awaddr = '0; kv_wdata = '0; kv_awvalid = 1'b0; kv_wvalid = 1'b0;
    kv_bready = 1'b1; kv_araddr = '0; kv_arvalid = 1'b0; kv_rready = 1'b1;
    s_tdata = '0; s_tvalid = 1'b0; s_tlast = 1'b0; s_tuser = 1'b0;
    flush_r = 1'b0;
    fj_valid = 1'b0; fj_rows = '0;
    desc_valid = 1'b0; desc_r = '0;
    wgt_valid = 1'b0; wgt_data = '0;
    n_checks = 0; n_errors = 0;
    efs_n = 0; efs_i = 0;

    void'($value$plusargs("bp_mode=%d", bp_mode));
    void'($value$plusargs("stall_mode=%d", stall_mode));
    void'($value$plusargs("seed=%d", prng));
    if (prng == 0) prng = 32'hC0FFEE02;

    vec_path = "build/vectors_l2b_t0.txt";
    void'($value$plusargs("vectors=%s", vec_path));
    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    while ($fscanf(fd, "%s", cmd) == 1) begin
      if (cmd.len() >= 2 && cmd.substr(0, 1) == "//") begin
        void'($fgets(rest, fd));
        $display("[%0d] %s %s", cyc, cmd, rest);
        continue;
      end
      case (cmd)
        "KVW": begin
          void'($fscanf(fd, "%h %h", a, d));
          kv_wr(a, d);
        end
        "KVP": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          kv_poll(a, m, e);
        end
        "KT", "VT": begin
          for (i = 0; i < int'(D_CFG); i++) begin
            void'($fscanf(fd, "%h", d));
            tok[i] = d[15:0];
          end
          drive_tok(tok, cmd == "VT");
        end
        "PT": begin
          // partial value token (mid-op reset stimulus): n beats, no tlast
          void'($fscanf(fd, "%h", e));
          for (i = 0; i < int'(e); i++) begin
            void'($fscanf(fd, "%h", d));
            @(negedge clk);
            s_tdata  = d[15:0];
            s_tuser  = 1'b1;
            s_tlast  = 1'b0;
            s_tvalid = 1'b1;
            m = 0;
            while (!s_tready) begin
              @(negedge clk);
              m++;
              if (m > WAIT_LIMIT) $fatal(1, "PT stall @%0d", cyc);
            end
            @(negedge clk);
            s_tvalid = 1'b0;
          end
        end
        "FLUSH": begin
          @(negedge clk);
          flush_r = 1'b1;
          @(negedge clk);
          flush_r = 1'b0;
        end
        "FJ": begin
          void'($fscanf(fd, "%h", d));
          drive_fj(d[DIM_W-1:0]);
        end
        "RD": begin
          void'($fscanf(fd, "%h", d));
          kv_wr(8'h2C, d);            // READ_ADDR
        end
        "DESC": begin
          void'($fscanf(fd, "%h %h %h %h", dw[3], dw[2], dw[1], dw[0]));
          drive_desc({dw[3], dw[2], dw[1], dw[0]});
        end
        "WB": begin
          void'($fscanf(fd, "%h", w64));
          drive_wb(w64);
        end
        "EFS": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h %h", m, e));
            efs_exp[efs_n + i] = {m[15:0], e[0]};
          end
          efs_n += int'(d);
        end
        "ERO": begin
          for (i = 0; i < 8; i++) begin
            void'($fscanf(fd, "%h", d));
            lanes[32*i +: 32] = d;
          end
          void'($fscanf(fd, "%h", d));
          expect_ro(lanes, d[0]);
        end
        "STK": begin
          void'($fscanf(fd, "%h %h", m, e));
          chk32("STK", 32'({fq_job_error_sticky, mxe_desc_error_sticky}) & m,
                e);
        end
        "IDLE": wait_idle();
        "RST": begin
          @(negedge clk);
          kv_awvalid = 1'b0; kv_wvalid = 1'b0; kv_arvalid = 1'b0;
          s_tvalid = 1'b0; s_tlast = 1'b0; flush_r = 1'b0;
          fj_valid = 1'b0; desc_valid = 1'b0; wgt_valid = 1'b0;
          rst_n = 1'b0;
          repeat (4) @(negedge clk);
          rst_n = 1'b1;
          repeat (3) @(negedge clk);
          chk32("RST efs drain", 32'(efs_i), 32'(efs_n));
          chk32("RST ro drain", 32'(ro_q.size()), 0);
          ro_q.delete();
          $display("[%0d] mid-operation reset applied", cyc);
        end
        "ENDTAB": begin
          chk32("ENDTAB efs", 32'(efs_i), 32'(efs_n));
          chk32("ENDTAB ro", 32'(ro_q.size()), 0);
        end
        "DONE": begin
          wait_idle();
          n_checks += tap_checks;
          n_errors += tap_errors;
          $display("L2B RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "L2B FAIL: %0d errors", n_errors);
          $display("L2B PASS");
          done_seen = 1'b1;
          $finish;
        end
        default: $fatal(1, "unknown command '%s'", cmd);
      endcase
      if (done_seen) break;
    end
    if (!done_seen) $fatal(1, "vector file ended without DONE");
  end

endmodule
