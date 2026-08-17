// tb_bias_tile.sv — IC-BIAS (gap B) TILE suite: the l4_bias case's q/K/V
// projections driven through a REAL apex_top with PROJ_BIAS_EN=1.
//
// Drive (the L3 phase-B/phase-C idiom, generator gen_bias_vectors.py):
//   x8 -> u_rms -> u_widen -> u_feeder (C-1) -> act stage        [phase B]
//   act row t x Wk/Wv/Wq (OP_GEMM_WS, K=128, n=8) -> serializer
//     -> apex_proj_bias (composite + staged fp16 bias) -> S-2 f16 seam
//     -> (l_rope_en=0) -> KVQ store                              [A1/A2/A3]
// CHECK: every accepted f16 beat at the dbg_f16_* tap, against
//   f64_to_f16_bits(r.K_real / r.V_real / r.q_real) from
//   decoder_layer_fx(bus=BUS_ON) — i.e. ONE RNE of (acc*composite + bias),
//   the golden pre-narrowing bias order (transformer.py:509-514, :521-522).
//
// Arms: A1 biased q/K/V bit-exact · A2 l_bias_en=0 reproduces the UNBIASED
// narrowing exactly (the route mux is real) · A3 exact-or-refused and the
// C2/job refusals land in LAYER_STATUS[8] with their codes and NEVER in
// err_sticky[15] (the B1 rule).
//
// Vector file via +vectors=. Exit:
//   "BIASTILE RESULT: checks=<n> errors=0 -> PASS" + "BIASTILE PASS".

module tb_bias_tile;

  parameter int unsigned CFG_D = 128;
  parameter int unsigned G_CFG = 16;
  parameter int unsigned DEPTH = 256;
  parameter bit          PROJ_BIAS_EN = 1'b1;
  parameter int unsigned FEED_ROWS_MAX = 31;
  parameter int unsigned STAGE_R_MAX = 31;
  parameter int unsigned LAYER_DM_MAX = 128;

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 400000;
  localparam int unsigned NB_W = $clog2(CFG_D / 8) + 1;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin clk = 1'b0; forever #5 clk = ~clk; end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 20_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: bias tile did not finish within %0d cycles",
           watchdog_cyc);
  end

  // ── DUT signals ────────────────────────────────────────────────────────────
  logic [7:0]   csr_addr;
  logic [31:0]  csr_wdata, csr_rdata;
  logic         csr_write, csr_read, csr_ready;

  logic [7:0]   kv_awaddr, kv_araddr;
  logic         kv_awvalid, kv_wvalid, kv_arvalid;
  logic [31:0]  kv_wdata, kv_rdata;
  logic         kv_awready, kv_wready, kv_bvalid, kv_arready, kv_rvalid;

  logic [127:0] ds_desc_r;
  logic         ds_valid, ds_ready;
  logic         xw_valid, xw_ready;
  logic [63:0]  xw_data;

  logic         xa_valid, xa_ready, xa_last;
  logic [7:0]   xa_xb;
  logic         xg_valid, xg_ready;
  logic [15:0]  xg_gb;

  logic         qs_valid, qs_ready;
  logic [31:0]  qs_data;

  logic             fj_valid, fj_ready;
  logic [DIM_W-1:0] fj_rows;
  logic             qj_valid, qj_ready, qj_mode;
  logic [DIM_W-1:0] qj_cols;
  logic             lj_valid, lj_ready;
  logic [7:0]       lj_beats;
  logic [3:0]       lj_lanes;
  logic             aj_valid, aj_ready, aj_op, aj_bank;
  logic [1:0]       aj_pat;
  logic [4:0]       aj_rows, aj_sel;
  logic [NB_W-1:0]  aj_nb;

  logic [15:0]  route_r;

  logic         fs_valid, fs_last;
  logic [15:0]  fs_data;
  logic [15:0]  err_sticky;
  logic         dbg_f16_v, dbg_f16_last;
  logic [15:0]  dbg_f16_data;

  apex_top #(
    .CFG_D (CFG_D), .KVQ_G (G_CFG), .KVQ_DEPTH (DEPTH),
    .FEED_ROWS_MAX (FEED_ROWS_MAX), .STAGE_R_MAX (STAGE_R_MAX),
    .LAYER_DM_MAX (LAYER_DM_MAX), .PROJ_BIAS_EN (PROJ_BIAS_EN)
  ) u_dut (
    .clk (clk), .rst_n (rst_n),
    .csr_addr (csr_addr), .csr_wdata (csr_wdata), .csr_write (csr_write),
    .csr_read (csr_read), .csr_rdata (csr_rdata), .csr_ready (csr_ready),
    .kv_awaddr (kv_awaddr), .kv_awvalid (kv_awvalid), .kv_awready (kv_awready),
    .kv_wdata (kv_wdata), .kv_wvalid (kv_wvalid), .kv_wready (kv_wready),
    .kv_bresp (), .kv_bvalid (kv_bvalid), .kv_bready (1'b1),
    .kv_araddr (kv_araddr), .kv_arvalid (kv_arvalid), .kv_arready (kv_arready),
    .kv_rdata (kv_rdata), .kv_rresp (), .kv_rvalid (kv_rvalid),
    .kv_rready (1'b1),
    .kv_irq (), .kv_evict_needed (), .kv_evict_addr (),
    .ds_valid (ds_valid), .ds_ready (ds_ready),
    .ds_desc (mxe_desc_t'(ds_desc_r)),
    .xw_valid (xw_valid), .xw_ready (xw_ready),
    .xw_beat (lane8_beat_t'({xw_data, 1'b0})),
    .xa_valid (xa_valid), .xa_ready (xa_ready), .xa_x (signed'(xa_xb)),
    .xa_last (xa_last),
    .xg_valid (xg_valid), .xg_ready (xg_ready), .xg_gamma (signed'(xg_gb)),
    .qs_valid (qs_valid), .qs_ready (qs_ready), .qs_data (qs_data),
    .cs_valid (1'b0), .cs_ready (), .cs_data ('0),
    .fj_valid (fj_valid), .fj_ready (fj_ready), .fj_rows (fj_rows),
    .qj_valid (qj_valid), .qj_ready (qj_ready), .qj_mode (qj_mode),
    .qj_cols (qj_cols),
    .dj_valid (1'b0), .dj_ready (), .dj_cols ('0),
    .lj_valid (lj_valid), .lj_ready (lj_ready), .lj_beats (lj_beats),
    .lj_lanes (lj_lanes),
    .aj_valid (aj_valid), .aj_ready (aj_ready), .aj_op (aj_op),
    .aj_bank (aj_bank), .aj_pat (aj_pat), .aj_rows (aj_rows),
    .aj_nb (aj_nb), .aj_sel (aj_sel),
    .wj_valid (1'b0), .wj_ready (), .wj_op (1'b0), .wj_bank (1'b0),
    .wj_pat ('0), .wj_rows ('0), .wj_nb ('0), .wj_sel ('0),
    .rt_feeder_src (route_r[0]), .rt_feeder_dst (route_r[1]),
    .rt_act_src (route_r[2]), .rt_wgt_src (route_r[3]),
    .rt_res_dst (route_r[5:4]), .rt_squant_src (route_r[6]),
    .rt_kv_user (route_r[7]), .rt_tip_blk (route_r[14:8]),
    .rt_imp_hi ('0), .rt_imp_lo ('0), .rt_imp_clear (1'b0),
    .wf_valid (), .wf_ready (1'b1), .wf_req (),
    .fs_valid (fs_valid), .fs_ready (1'b1), .fs_data (fs_data),
    .fs_last (fs_last),
    .ss_valid (), .ss_ready (1'b1), .ss_data (), .ss_last (),
    .td_valid (), .td_ready (1'b1), .td_fp16 (), .td_tier (), .td_blk (),
    .ro_valid (), .ro_ready (1'b1), .ro_beat (),
    .dn_mxe (), .dn_feeder (), .dn_squant (), .dn_scored (), .dn_ser (),
    .dn_astage (), .dn_wstage (), .dn_rms (), .dn_asu (),
    .err_sticky (err_sticky),
    .dbg_f16_v (dbg_f16_v), .dbg_f16_data (dbg_f16_data),
    .dbg_f16_last (dbg_f16_last),
    .dbg_sc_v (), .dbg_sc_data (), .dbg_sc_last (),
    .dbg_pr_v (), .dbg_pr_data (), .dbg_pr_last ()
  );

  logic unused_ok;
  assign unused_ok = &{1'b0, kv_awready, kv_wready, kv_arready, csr_ready,
                       route_r[15], dbg_f16_last, fs_last, err_sticky};

  int n_checks, n_errors;

  // ── feeder-scale collector (EFS) ─────────────────────────────────────────
  logic [16:0] fs_q [$];
  always @(posedge clk) if (rst_n && fs_valid) fs_q.push_back({fs_data, fs_last});

  // ── the S-2 f16 tap: EVERY accepted beat is checked in order ─────────────
  // cur_tag names the ARM a beat belongs to so a mutant's FAIL signature is
  // specific to the integration point it broke (the l3 mutate.py discipline).
  logic [15:0] tapf16_exp [8192];
  int tapf16_n, tapf16_i;
  string cur_tag = "none";
  always @(posedge clk) begin
    if (!rst_n) begin
      tapf16_i <= 0;
    end else if (dbg_f16_v) begin
      if (tapf16_i >= tapf16_n) begin
        n_errors <= n_errors + 1;
        $error("[tap f16][%s] overflow at %0d (table %0d) @%0d", cur_tag,
               tapf16_i, tapf16_n, cyc);
      end else begin
        n_checks <= n_checks + 1;
        if (dbg_f16_data !== tapf16_exp[tapf16_i]) begin
          n_errors <= n_errors + 1;
          $error("[tap f16][%s] idx %0d: got %04x exp %04x @%0d", cur_tag,
                 tapf16_i, dbg_f16_data, tapf16_exp[tapf16_i], cyc);
        end
      end
      tapf16_i <= tapf16_i + 1;
    end
  end

  // ── bus / stream tasks (negedge discipline — §3b) ────────────────────────
  task automatic csr_wr(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk);
    csr_addr = a; csr_wdata = d; csr_write = 1'b1;
    @(negedge clk);
    csr_write = 1'b0;
  endtask

  task automatic csr_rd(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk);
    csr_addr = a; csr_read = 1'b1;
    @(negedge clk);            // 1-CYCLE sample discipline (§3b rule 5)
    csr_read = 1'b0;
    d = csr_rdata;
  endtask

  task automatic csr_poll(input logic [7:0] a, input logic [31:0] m,
                          input logic [31:0] e);
    int wd = 0; logic [31:0] v;
    forever begin
      csr_rd(a, v);
      if ((v & m) === e) break;
      repeat (20) @(posedge clk);
      wd++;
      if (wd > 50000) $fatal(1, "CSRP %02x timeout (got %08x) @%0d", a, v, cyc);
    end
  endtask

  task automatic kv_wr(input logic [7:0] a, input logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    kv_awaddr = a; kv_wdata = d; kv_awvalid = 1'b1; kv_wvalid = 1'b1;
    @(negedge clk);
    kv_awvalid = 1'b0; kv_wvalid = 1'b0;
    while (!kv_bvalid) begin
      @(negedge clk); wd++;
      if (wd > 100) $fatal(1, "[kv] write %02x: no bvalid", a);
    end
  endtask

  task automatic kv_rd(input logic [7:0] a, output logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    kv_araddr = a; kv_arvalid = 1'b1;
    @(negedge clk);
    kv_arvalid = 1'b0;
    while (!kv_rvalid) begin
      @(negedge clk); wd++;
      if (wd > 100) $fatal(1, "[kv] read %02x: no rvalid", a);
    end
    d = kv_rdata;
  endtask

  task automatic kv_poll(input logic [7:0] a, input logic [31:0] m,
                         input logic [31:0] e);
    int wd = 0; logic [31:0] v;
    forever begin
      kv_rd(a, v);
      if ((v & m) === e) break;
      repeat (20) @(posedge clk);
      wd++;
      if (wd > 50000) $fatal(1, "KVP %02x timeout (got %08x) @%0d", a, v, cyc);
    end
  endtask

  task automatic drive_x(input logic [7:0] b, input logic lst);
    int wd = 0;
    @(negedge clk);
    xa_xb = b; xa_last = lst; xa_valid = 1'b1;
    while (!xa_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_x stall @%0d", cyc);
    end
    @(negedge clk); xa_valid = 1'b0;
  endtask

  task automatic drive_g(input logic [15:0] g);
    int wd = 0;
    @(negedge clk);
    xg_gb = g; xg_valid = 1'b1;
    while (!xg_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_g stall @%0d", cyc);
    end
    @(negedge clk); xg_valid = 1'b0;
  endtask

  task automatic drive_w(input logic [63:0] w);
    int wd = 0;
    @(negedge clk);
    xw_data = w; xw_valid = 1'b1;
    while (!xw_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_w stall @%0d", cyc);
    end
    @(negedge clk); xw_valid = 1'b0;
  endtask

  task automatic drive_ds(input logic [127:0] d);
    int wd = 0;
    @(negedge clk);
    ds_desc_r = d; ds_valid = 1'b1;
    while (!ds_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_ds stall @%0d", cyc);
    end
    @(negedge clk); ds_valid = 1'b0;
  endtask

  task automatic drive_qs(input logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    qs_data = d; qs_valid = 1'b1;
    while (!qs_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_qs stall @%0d", cyc);
    end
    @(negedge clk); qs_valid = 1'b0;
  endtask

  task automatic drive_fj(input logic [DIM_W-1:0] rows);
    int wd = 0;
    @(negedge clk);
    fj_rows = rows; fj_valid = 1'b1;
    while (!fj_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_fj stall @%0d", cyc);
    end
    @(negedge clk); fj_valid = 1'b0;
  endtask

  task automatic drive_qj(input logic m, input logic [DIM_W-1:0] cols);
    int wd = 0;
    @(negedge clk);
    qj_mode = m; qj_cols = cols; qj_valid = 1'b1;
    while (!qj_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_qj stall @%0d", cyc);
    end
    @(negedge clk); qj_valid = 1'b0;
  endtask

  task automatic drive_lj(input logic [7:0] b, input logic [3:0] l);
    int wd = 0;
    @(negedge clk);
    lj_beats = b; lj_lanes = l; lj_valid = 1'b1;
    while (!lj_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_lj stall @%0d", cyc);
    end
    @(negedge clk); lj_valid = 1'b0;
  endtask

  task automatic drive_aj(input logic op, input logic bank,
                          input logic [1:0] pat, input logic [4:0] rows,
                          input logic [NB_W-1:0] nb, input logic [4:0] sel);
    int wd = 0;
    @(negedge clk);
    aj_op = op; aj_bank = bank; aj_pat = pat; aj_rows = rows;
    aj_nb = nb; aj_sel = sel; aj_valid = 1'b1;
    while (!aj_ready) begin
      @(negedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_aj stall @%0d", cyc);
    end
    @(negedge clk); aj_valid = 1'b0;
  endtask

  task automatic expect_fs(input logic [15:0] d, input logic lst);
    int wd = 0; logic [16:0] got;
    while (fs_q.size() == 0) begin
      @(posedge clk); wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "EFS timeout (exp %04x) @%0d", d, cyc);
    end
    got = fs_q.pop_front();
    n_checks++;
    if (got !== {d, lst}) begin
      n_errors++;
      $error("[EFS] got %04x/%0b exp %04x/%0b", got[16:1], got[0], d, lst);
    end
  endtask

  task automatic chk32(input string what, input logic [31:0] got,
                       input logic [31:0] exp);
    n_checks++;
    if (got !== exp) begin
      n_errors++;
      $error("[%s] got %08x exp %08x @%0d", what, got, exp, cyc);
    end
  endtask

  // ── interpreter ───────────────────────────────────────────────────────────
  initial begin : interp
    string vec_path, cmd, rest;
    bit done_seen;
    int fd, i;
    logic [7:0]  a;
    logic [31:0] m, e, d;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] v32;   // small-token scratch (AJ fields / EFS last)
    /* verilator lint_on UNUSEDSIGNAL */
    logic [127:0] dw;
    logic [63:0]  w64;
    logic [31:0]  d0, d1, d2, d3;
    logic        t_op, t_bank;
    logic [1:0]  t_pat;
    logic [4:0]  t_rows, t_sel;
    logic [NB_W-1:0] t_nb;

    rst_n = 1'b0;
    csr_addr = '0; csr_wdata = '0; csr_write = 1'b0; csr_read = 1'b0;
    kv_awaddr = '0; kv_araddr = '0; kv_wdata = '0;
    kv_awvalid = 1'b0; kv_wvalid = 1'b0; kv_arvalid = 1'b0;
    ds_valid = 1'b0; ds_desc_r = '0;
    xw_valid = 1'b0; xw_data = '0;
    xa_valid = 1'b0; xa_xb = '0; xa_last = 1'b0;
    xg_valid = 1'b0; xg_gb = '0;
    qs_valid = 1'b0; qs_data = '0;
    fj_valid = 1'b0; fj_rows = '0;
    qj_valid = 1'b0; qj_mode = 1'b0; qj_cols = '0;
    lj_valid = 1'b0; lj_beats = '0; lj_lanes = '0;
    aj_valid = 1'b0; aj_op = 1'b0; aj_bank = 1'b0; aj_pat = '0;
    aj_rows = '0; aj_nb = '0; aj_sel = '0;
    route_r = '0;
    n_checks = 0; n_errors = 0; tapf16_n = 0;

    vec_path = "build/bias_tile.ops";
    void'($value$plusargs("vectors=%s", vec_path));
    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    while ($fscanf(fd, "%s", cmd) == 1) begin
      if (cmd.len() >= 2 && cmd.substr(0, 1) == "//") begin
        void'($fgets(rest, fd));
        $display("[%0d] %s%s", cyc, cmd, rest);
        continue;
      end
      case (cmd)
        "CSRW": begin
          void'($fscanf(fd, "%h %h", a, d));
          csr_wr(a[7:0], d);
        end
        "CSRR": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          csr_rd(a[7:0], d);
          chk32($sformatf("CSRR %02x", a[7:0]), d & m, e);
        end
        "CSRP": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          csr_poll(a[7:0], m, e);
        end
        "KVW": begin
          void'($fscanf(fd, "%h %h", a, d));
          kv_wr(a[7:0], d);
        end
        "KVP": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          kv_poll(a[7:0], m, e);
        end
        "ROUTE": begin
          void'($fscanf(fd, "%h", d));
          @(negedge clk);
          route_r = d[15:0];
          @(negedge clk);
        end
        "AJ": begin
          void'($fscanf(fd, "%h", v32)); t_op   = v32[0];
          void'($fscanf(fd, "%h", v32)); t_bank = v32[0];
          void'($fscanf(fd, "%h", v32)); t_pat  = v32[1:0];
          void'($fscanf(fd, "%h", v32)); t_rows = v32[4:0];
          void'($fscanf(fd, "%h", v32)); t_nb   = v32[NB_W-1:0];
          void'($fscanf(fd, "%h", v32)); t_sel  = v32[4:0];
          drive_aj(t_op, t_bank, t_pat, t_rows, t_nb, t_sel);
        end
        "FJOB": begin
          void'($fscanf(fd, "%h", d));
          drive_fj(d[DIM_W-1:0]);
        end
        "QJOB": begin
          void'($fscanf(fd, "%h %h", a, d));
          drive_qj(a[0], d[DIM_W-1:0]);
        end
        "LJOB": begin
          void'($fscanf(fd, "%h %h", a, d));
          drive_lj(a[7:0], d[3:0]);
        end
        "QS": begin
          void'($fscanf(fd, "%h", d));
          drive_qs(d);
        end
        "DESC": begin
          void'($fscanf(fd, "%h %h %h %h", d0, d1, d2, d3));
          dw = {d0, d1, d2, d3};
          drive_ds(dw);
        end
        "WB": begin
          void'($fscanf(fd, "%h", w64));
          drive_w(w64);
        end
        "XR": begin
          for (i = 0; i < int'(CFG_D); i++) begin
            void'($fscanf(fd, "%h", d));
            drive_x(d[7:0], i == int'(CFG_D) - 1);
          end
        end
        "GR": begin
          for (i = 0; i < int'(CFG_D); i++) begin
            void'($fscanf(fd, "%h", d));
            drive_g(d[15:0]);
          end
        end
        "EFS": begin
          void'($fscanf(fd, "%h %h", d, v32));
          expect_fs(d[15:0], v32[0]);
        end
        "TAPF16": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h", v32));
            tapf16_exp[tapf16_n + i] = v32[15:0];
          end
          tapf16_n += int'(d);
        end
        "TAG": begin
          void'($fscanf(fd, "%s", cur_tag));
        end
        "ENDTAPS": begin
          chk32("ENDTAPS f16", 32'(tapf16_i), 32'(tapf16_n));
        end
        "DONE": begin
          n_checks++;
          if (fs_q.size() != 0) begin
            n_errors++;
            $error("[chk] leftover feeder scales: %0d", fs_q.size());
          end
          $display("BIASTILE RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "BIASTILE FAIL: %0d errors", n_errors);
          $display("%s%s", "BIASTILE PASS (l4_bias q/K/V biased projections ",
                   "bit-exact vs decoder_layer_fx BUS_ON at the S-2 seam)");
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
