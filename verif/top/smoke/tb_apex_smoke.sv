// tb_apex_smoke.sv — apex_top v0.1 full-chain smoke TB (command interpreter).
//
// The stimulus AND every expected value come from gen_top_vectors.py, which
// runs the golden arbiter (golden/apex_golden/attention.py attention_fx) on
// the same inputs; this TB only drives the tile's host interfaces, collects,
// and compares BIT-EXACT. It plays the v0.1 "host" role documented in
// rtl/top/apex_top.sv (descriptors, weights, glue jobs, routes, composites,
// KVQ AXI-Lite sequencing) — one whitespace-separated command per action:
//
//   CSRW a d / CSRR a m e / CSRP a m e (poll) / CSRN a m (expect nonzero)
//   KVW a d / KVP a m e                  KVQ AXI-Lite write / poll
//   ROUTE r                              rt_* levels (bit map in the DUT inst)
//   DESC w3 w2 w1 w0                     descriptor -> SEQ ds_* stream
//   XR <D bytes> / GR <D halfwords>      x8 row / gamma row (D = CFG_D)
//   WB w64                               one external weight beat
//   FJOB r / QJOB m c / SJOB c / LJOB b l / AJ .. / WJ ..   glue jobs
//   QS w32 / CS w32                      sideband composites
//   EFS h16 l / ESS h16 l                expect feeder / scale_quant scale
//   ERO l0..l7 last                      expect result beat (bit-exact o8)
//   ETIP blk                             expect one TIP decision beat
//   TAPF16 n <n vals> (also TAPSC/TAPPR) passive-tap expectation tables
//   ENDTAPS / ESTK m e / DONE
//
// Passive taps (dbg_* accepted-beat mirrors) check the KVQ write-port fp16
// values, the S-3 scores, and the ASU probabilities ON THE FLY — every
// mid-chain tensor is compared against golden, not just the final o8.
//
// §5/D-006 are additionally checked every cycle by apex_stream_sva bound
// into the MXE and apex_stream1_sva bound into every new glue block (data
// stability under backpressure), compiled under --assert (D-012 gate).
// Watchdogged ($fatal) — an honest hang, never a silent pass.

`include "apex_stream_sva.svh"
`include "apex_stream1_sva.svh"

module tb_apex_smoke;

  // build-time tile configuration (D=64 default; a -GCFG_D=128 build runs
  // the same grammar against the D=128 tile — F-5 closure smoke)
  parameter int unsigned CFG_D = 64;
  parameter int unsigned OUTK  = 0;    // CQ-4+ outlier lanes (tier bank e2)
  parameter              MASKF = "";   // outlier-mask ROM hex

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 400000;
  // F-5a: stage-buffer nb field width, derived exactly as in apex_top
  localparam int unsigned NB_W = $clog2(CFG_D / 8) + 1;

  // ── clock / reset / watchdog ───────────────────────────────────────────────
  logic clk, rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 8_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: smoke did not finish within %0d cycles", watchdog_cyc);
  end

  initial begin
    if ($test$plusargs("dump")) begin
      $dumpfile("dump.fst");
      $dumpvars(0, tb_apex_smoke);
    end
  end

  // ── DUT signals ────────────────────────────────────────────────────────────
  logic [7:0]   csr_addr;
  logic [31:0]  csr_wdata;
  logic         csr_write, csr_read;
  logic [31:0]  csr_rdata;
  logic         csr_ready;

  logic [7:0]   kv_awaddr, kv_araddr;
  logic         kv_awvalid, kv_awready, kv_wvalid, kv_wready;
  logic [31:0]  kv_wdata, kv_rdata;
  logic [1:0]   kv_bresp, kv_rresp;
  logic         kv_bvalid, kv_bready, kv_arvalid, kv_arready;
  logic         kv_rvalid, kv_rready, kv_irq, kv_evict_needed;
  logic [6:0]   kv_evict_addr;

  logic         ds_valid, ds_ready;
  logic [127:0] ds_desc_r;
  logic         xw_valid, xw_ready;
  logic [63:0]  xw_data;
  logic         xa_valid, xa_ready, xa_last;
  logic [7:0]   xa_xb;
  logic         xg_valid, xg_ready;
  logic [15:0]  xg_gb;
  logic         qs_valid, qs_ready;
  logic [31:0]  qs_data;
  logic         cs_valid, cs_ready;
  logic [31:0]  cs_data;

  logic             fj_valid, fj_ready;
  logic [DIM_W-1:0] fj_rows;
  logic             qj_valid, qj_ready, qj_mode;
  logic [DIM_W-1:0] qj_cols;
  logic             dj_valid, dj_ready;
  logic [DIM_W-1:0] dj_cols;
  logic             lj_valid, lj_ready;
  logic [7:0]       lj_beats;
  logic [3:0]       lj_lanes;
  logic             aj_valid, aj_ready, aj_op, aj_bank;
  logic [1:0]       aj_pat;
  logic [4:0]       aj_rows, aj_sel;
  logic [NB_W-1:0]  aj_nb;
  logic             wj_valid, wj_ready, wj_op, wj_bank;
  logic [1:0]       wj_pat;
  logic [4:0]       wj_rows, wj_sel;
  logic [NB_W-1:0]  wj_nb;

  logic [15:0]  route_r;

  logic         fs_valid, fs_ready, fs_last;
  logic [15:0]  fs_data;
  logic         ss_valid, ss_ready, ss_last;
  logic [15:0]  ss_data;
  logic         td_valid, td_ready, td_fp16;
  kvq_tier_e    td_tier;
  logic [6:0]   td_blk;
  logic         ro_valid, ro_ready;
  lane32_beat_t ro_beat;
  logic         wf_valid;              // W-G3 flip: walker fetch channel
  logic [63:0]  wf_req;

  logic dn_mxe, dn_feeder, dn_squant, dn_scored, dn_ser;
  logic dn_astage, dn_wstage, dn_rms, dn_asu;
  logic [15:0] err_sticky;

  logic        dbg_f16_v, dbg_f16_last, dbg_sc_v, dbg_sc_last;
  logic        dbg_pr_v, dbg_pr_last;
  logic [15:0] dbg_f16_data, dbg_pr_data;
  logic [31:0] dbg_sc_data;

  apex_top #(.CFG_D(CFG_D), .KVQ_OUTLIER_K(OUTK),
             .KVQ_MASK_FILE(MASKF)) dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .csr_addr        (csr_addr),
    .csr_wdata       (csr_wdata),
    .csr_write       (csr_write),
    .csr_read        (csr_read),
    .csr_rdata       (csr_rdata),
    .csr_ready       (csr_ready),
    .kv_awaddr       (kv_awaddr),
    .kv_awvalid      (kv_awvalid),
    .kv_awready      (kv_awready),
    .kv_wdata        (kv_wdata),
    .kv_wvalid       (kv_wvalid),
    .kv_wready       (kv_wready),
    .kv_bresp        (kv_bresp),
    .kv_bvalid       (kv_bvalid),
    .kv_bready       (kv_bready),
    .kv_araddr       (kv_araddr),
    .kv_arvalid      (kv_arvalid),
    .kv_arready      (kv_arready),
    .kv_rdata        (kv_rdata),
    .kv_rresp        (kv_rresp),
    .kv_rvalid       (kv_rvalid),
    .kv_rready       (kv_rready),
    .kv_irq          (kv_irq),
    .kv_evict_needed (kv_evict_needed),
    .kv_evict_addr   (kv_evict_addr),
    .ds_valid        (ds_valid),
    .ds_ready        (ds_ready),
    .ds_desc         (mxe_desc_t'(ds_desc_r)),
    .xw_valid        (xw_valid),
    .xw_ready        (xw_ready),
    .xw_beat         (lane8_beat_t'({xw_data, 1'b0})),
    .xa_valid        (xa_valid),
    .xa_ready        (xa_ready),
    .xa_x            (signed'(xa_xb)),
    .xa_last         (xa_last),
    .xg_valid        (xg_valid),
    .xg_ready        (xg_ready),
    .xg_gamma        (signed'(xg_gb)),
    .qs_valid        (qs_valid),
    .qs_ready        (qs_ready),
    .qs_data         (qs_data),
    .cs_valid        (cs_valid),
    .cs_ready        (cs_ready),
    .cs_data         (cs_data),
    .fj_valid        (fj_valid),
    .fj_ready        (fj_ready),
    .fj_rows         (fj_rows),
    .qj_valid        (qj_valid),
    .qj_ready        (qj_ready),
    .qj_mode         (qj_mode),
    .qj_cols         (qj_cols),
    .dj_valid        (dj_valid),
    .dj_ready        (dj_ready),
    .dj_cols         (dj_cols),
    .lj_valid        (lj_valid),
    .lj_ready        (lj_ready),
    .lj_beats        (lj_beats),
    .lj_lanes        (lj_lanes),
    .aj_valid        (aj_valid),
    .aj_ready        (aj_ready),
    .aj_op           (aj_op),
    .aj_bank         (aj_bank),
    .aj_pat          (aj_pat),
    .aj_rows         (aj_rows),
    .aj_nb           (aj_nb),
    .aj_sel          (aj_sel),
    .wj_valid        (wj_valid),
    .wj_ready        (wj_ready),
    .wj_op           (wj_op),
    .wj_bank         (wj_bank),
    .wj_pat          (wj_pat),
    .wj_rows         (wj_rows),
    .wj_nb           (wj_nb),
    .wj_sel          (wj_sel),
    .rt_feeder_src   (route_r[0]),
    .rt_feeder_dst   (route_r[1]),
    .rt_act_src      (route_r[2]),
    .rt_wgt_src      (route_r[3]),
    .rt_res_dst      (route_r[5:4]),
    .rt_squant_src   (route_r[6]),
    .rt_kv_user      (route_r[7]),      // F-2: KVQ tuser (0=key, 1=value)
    .rt_tip_blk      (route_r[14:8]),
    .rt_imp_hi       (16'hFFFF),
    .rt_imp_lo       (16'h0000),
    .rt_imp_clear    (1'b0),
    .fs_valid        (fs_valid),
    .fs_ready        (fs_ready),
    .fs_data         (fs_data),
    .fs_last         (fs_last),
    .ss_valid        (ss_valid),
    .ss_ready        (ss_ready),
    .ss_data         (ss_data),
    .ss_last         (ss_last),
    .td_valid        (td_valid),
    .td_ready        (td_ready),
    .td_fp16         (td_fp16),
    .td_tier         (td_tier),
    .td_blk          (td_blk),
    .ro_valid        (ro_valid),
    .ro_ready        (ro_ready),
    .ro_beat         (ro_beat),
    // W-G3 flip: walker fetch-request channel — never fires in smoke (no
    // fmt=1 walk); ready tied 1 so a future case cannot wedge at S2_FETCH
    .wf_valid        (wf_valid),
    .wf_ready        (1'b1),
    .wf_req          (wf_req),
    .dn_mxe          (dn_mxe),
    .dn_feeder       (dn_feeder),
    .dn_squant       (dn_squant),
    .dn_scored       (dn_scored),
    .dn_ser          (dn_ser),
    .dn_astage       (dn_astage),
    .dn_wstage       (dn_wstage),
    .dn_rms          (dn_rms),
    .dn_asu          (dn_asu),
    .err_sticky      (err_sticky),
    .dbg_f16_v       (dbg_f16_v),
    .dbg_f16_data    (dbg_f16_data),
    .dbg_f16_last    (dbg_f16_last),
    .dbg_sc_v        (dbg_sc_v),
    .dbg_sc_data     (dbg_sc_data),
    .dbg_sc_last     (dbg_sc_last),
    .dbg_pr_v        (dbg_pr_v),
    .dbg_pr_data     (dbg_pr_data),
    .dbg_pr_last     (dbg_pr_last)
  );

  // dangling observability outputs, consumed for -Wall cleanliness
  logic unused_ok;
  assign unused_ok = &{1'b0, kv_awready, kv_wready, kv_arready,
                       kv_bresp, kv_rresp, route_r[15],
                       kv_irq, kv_evict_needed, kv_evict_addr,
                       dn_mxe, dn_feeder, dn_squant, dn_scored, dn_ser,
                       dn_astage, dn_wstage, dn_rms, dn_asu, td_fp16,
                       dbg_f16_last, dbg_sc_last, dbg_pr_last,
                       wf_valid, wf_req};

  // ── SVA: §5/D-006 gates compiled into the build (D-012) ───────────────────
  bind mxe_top apex_stream_sva u_sva_mxe (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(desc_valid), .desc_ready(desc_ready), .desc(desc),
    .desc_error(desc_error), .desc_error_sticky(desc_error_sticky),
    .busy(busy), .done(done),
    .act_valid(act_valid), .act_ready(act_ready), .act_beat(act_beat),
    .wgt_valid(wgt_valid), .wgt_ready(wgt_ready), .wgt_beat(wgt_beat),
    .res_valid(res_valid), .res_ready(res_ready), .res_beat(res_beat));

  bind apex_scale_quant apex_stream1_sva #(.WIDTH(33), .NAME("sq.v"))
    u_sva_v (.clk(clk), .rst_n(rst_n), .valid(v_valid), .ready(v_ready),
             .data({unsigned'(v_data), v_last}));
  bind apex_scale_quant apex_stream1_sva #(.WIDTH(17), .NAME("sq.f16"))
    u_sva_f (.clk(clk), .rst_n(rst_n), .valid(f_valid), .ready(f_ready),
             .data({f_data, f_last}));
  bind apex_scale_quant apex_stream1_sva #(.WIDTH(65), .NAME("sq.q"))
    u_sva_q (.clk(clk), .rst_n(rst_n), .valid(q_valid), .ready(q_ready),
             .data({q_beat.data, q_beat.last}));
  bind apex_scale_quant apex_stream1_sva #(.WIDTH(17), .NAME("sq.s"))
    u_sva_s (.clk(clk), .rst_n(rst_n), .valid(s_valid), .ready(s_ready),
             .data({s_data, s_last}));
  bind apex_stage_buf apex_stream1_sva #(.WIDTH(65), .NAME("stage.em"))
    u_sva_em (.clk(clk), .rst_n(rst_n), .valid(em_valid), .ready(em_ready),
              .data({em_beat.data, em_beat.last}));
  bind apex_lane32_ser apex_stream1_sva #(.WIDTH(33), .NAME("ser.out"))
    u_sva_o (.clk(clk), .rst_n(rst_n), .valid(out_valid), .ready(out_ready),
             .data({unsigned'(out_data), out_last}));
  bind apex_score_fork apex_stream1_sva #(.WIDTH(33), .NAME("fork.a"))
    u_sva_a (.clk(clk), .rst_n(rst_n), .valid(a_valid), .ready(a_ready),
             .data({unsigned'(a_data), a_last}));
  bind apex_score_fork apex_stream1_sva #(.WIDTH(33), .NAME("fork.b"))
    u_sva_b (.clk(clk), .rst_n(rst_n), .valid(b_valid), .ready(b_ready),
             .data({unsigned'(b_data), b_last}));

  // ── scoreboard state ───────────────────────────────────────────────────────
  int n_checks, n_errors;          // interpreter-thread counters
  int tap_checks, tap_errors;      // always-block (NBA) counters, merged at DONE

  // always-ready output collectors (backpressure sweeps are the per-block
  // suites' job; this TB proves the chain values)
  logic [16:0]  fs_q [$];
  logic [16:0]  ss_q [$];
  logic [9:0]   tip_q [$];
  logic [256:0] ro_q [$];

  assign fs_ready = 1'b1;
  assign ss_ready = 1'b1;
  assign td_ready = 1'b1;
  assign ro_ready = 1'b1;

  always @(posedge clk) begin
    if (rst_n) begin
      if (fs_valid) fs_q.push_back({fs_data, fs_last});
      if (ss_valid) ss_q.push_back({ss_data, ss_last});
      if (td_valid) tip_q.push_back({td_blk, 2'(td_tier), td_fp16});
      if (ro_valid) ro_q.push_back({ro_beat.data, ro_beat.last});
    end
  end

  // passive tap expectation tables (loaded by TAPF16/TAPSC/TAPPR).
  // tapf16 sizing: 2 rounds x 2*T*D fp16 values = 4096 at D=128.
  logic [15:0] tapf16_exp [8192];
  logic [31:0] tapsc_exp  [256];
  logic [15:0] tappr_exp  [256];
  int tapf16_n, tapf16_i, tapsc_n, tapsc_i, tappr_n, tappr_i;

  always @(posedge clk) begin
    if (rst_n && dbg_f16_v) begin
      if (tapf16_i >= tapf16_n) begin
        tap_errors <= tap_errors + 1;
        $error("[tap f16] unexpected beat %04x @%0d", dbg_f16_data, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if (dbg_f16_data !== tapf16_exp[tapf16_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[tap f16] idx %0d: got %04x exp %04x", tapf16_i,
                 dbg_f16_data, tapf16_exp[tapf16_i]);
        end
      end
      tapf16_i <= tapf16_i + 1;
    end
    if (rst_n && dbg_sc_v) begin
      if (tapsc_i >= tapsc_n) begin
        tap_errors <= tap_errors + 1;
        $error("[tap score] unexpected beat %08x @%0d", dbg_sc_data, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if (dbg_sc_data !== tapsc_exp[tapsc_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[tap score] idx %0d: got %08x exp %08x", tapsc_i,
                 dbg_sc_data, tapsc_exp[tapsc_i]);
        end
      end
      tapsc_i <= tapsc_i + 1;
    end
    if (rst_n && dbg_pr_v) begin
      if (tappr_i >= tappr_n) begin
        tap_errors <= tap_errors + 1;
        $error("[tap prob] unexpected beat %04x @%0d", dbg_pr_data, cyc);
      end else begin
        tap_checks <= tap_checks + 1;
        if (dbg_pr_data !== tappr_exp[tappr_i]) begin
          tap_errors <= tap_errors + 1;
          $error("[tap prob] idx %0d: got %04x exp %04x", tappr_i,
                 dbg_pr_data, tappr_exp[tappr_i]);
        end
      end
      tappr_i <= tappr_i + 1;
    end
  end

  // ── bus tasks (csr_regs / kvq AXI-Lite timing per their headers) ──────────
  task automatic csr_wr(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk);
    csr_addr  = a;
    csr_wdata = d;
    csr_write = 1'b1;
    @(negedge clk);
    csr_write = 1'b0;
    if (!csr_ready) begin
      n_errors++;
      $error("[csr] write %02x: no ready", a);
    end
  endtask

  task automatic csr_rd(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk);
    csr_addr = a;
    csr_read = 1'b1;
    @(negedge clk);
    csr_read = 1'b0;
    if (!csr_ready) begin
      n_errors++;
      $error("[csr] read %02x: no ready", a);
    end
    d = csr_rdata;
  endtask

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

  // ── stream drive tasks (negedge discipline; ready never depends on our
  //    valid combinationally — every DUT-side ready is registered/FSM) ──────
  task automatic drive_x(input logic [7:0] b, input logic lst);
    int wd = 0;
    @(negedge clk);
    xa_xb    = b;
    xa_last  = lst;
    xa_valid = 1'b1;
    while (!xa_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_x stall @%0d", cyc);
    end
    @(negedge clk);
    xa_valid = 1'b0;
  endtask

  task automatic drive_g(input logic [15:0] g);
    int wd = 0;
    @(negedge clk);
    xg_gb    = g;
    xg_valid = 1'b1;
    while (!xg_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_g stall @%0d", cyc);
    end
    @(negedge clk);
    xg_valid = 1'b0;
  endtask

  task automatic drive_w(input logic [63:0] w);
    int wd = 0;
    @(negedge clk);
    xw_data  = w;
    xw_valid = 1'b1;
    while (!xw_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_w stall @%0d", cyc);
    end
    @(negedge clk);
    xw_valid = 1'b0;
  endtask

  task automatic drive_ds(input logic [127:0] d);
    int wd = 0;
    @(negedge clk);
    ds_desc_r = d;
    ds_valid  = 1'b1;
    while (!ds_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_ds stall @%0d", cyc);
    end
    @(negedge clk);
    ds_valid = 1'b0;
  endtask

  task automatic drive_qs(input logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    qs_data  = d;
    qs_valid = 1'b1;
    while (!qs_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_qs stall @%0d", cyc);
    end
    @(negedge clk);
    qs_valid = 1'b0;
  endtask

  task automatic drive_cs(input logic [31:0] d);
    int wd = 0;
    @(negedge clk);
    cs_data  = d;
    cs_valid = 1'b1;
    while (!cs_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_cs stall @%0d", cyc);
    end
    @(negedge clk);
    cs_valid = 1'b0;
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

  task automatic drive_qj(input logic m, input logic [DIM_W-1:0] cols);
    int wd = 0;
    @(negedge clk);
    qj_mode  = m;
    qj_cols  = cols;
    qj_valid = 1'b1;
    while (!qj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_qj stall @%0d", cyc);
    end
    @(negedge clk);
    qj_valid = 1'b0;
  endtask

  task automatic drive_dj(input logic [DIM_W-1:0] cols);
    int wd = 0;
    @(negedge clk);
    dj_cols  = cols;
    dj_valid = 1'b1;
    while (!dj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_dj stall @%0d", cyc);
    end
    @(negedge clk);
    dj_valid = 1'b0;
  endtask

  task automatic drive_lj(input logic [7:0] b, input logic [3:0] l);
    int wd = 0;
    @(negedge clk);
    lj_beats = b;
    lj_lanes = l;
    lj_valid = 1'b1;
    while (!lj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_lj stall @%0d", cyc);
    end
    @(negedge clk);
    lj_valid = 1'b0;
  endtask

  task automatic drive_aj(input logic op, input logic bank,
                          input logic [1:0] pat, input logic [4:0] rows,
                          input logic [NB_W-1:0] nb, input logic [4:0] sel);
    int wd = 0;
    @(negedge clk);
    aj_op    = op;
    aj_bank  = bank;
    aj_pat   = pat;
    aj_rows  = rows;
    aj_nb    = nb;
    aj_sel   = sel;
    aj_valid = 1'b1;
    while (!aj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_aj stall @%0d", cyc);
    end
    @(negedge clk);
    aj_valid = 1'b0;
  endtask

  task automatic drive_wj(input logic op, input logic bank,
                          input logic [1:0] pat, input logic [4:0] rows,
                          input logic [NB_W-1:0] nb, input logic [4:0] sel);
    int wd = 0;
    @(negedge clk);
    wj_op    = op;
    wj_bank  = bank;
    wj_pat   = pat;
    wj_rows  = rows;
    wj_nb    = nb;
    wj_sel   = sel;
    wj_valid = 1'b1;
    while (!wj_ready) begin
      @(negedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "drive_wj stall @%0d", cyc);
    end
    @(negedge clk);
    wj_valid = 1'b0;
  endtask

  // ── expectation pops ───────────────────────────────────────────────────────
  task automatic expect_fs(input logic [15:0] d, input logic lst);
    int wd = 0;
    logic [16:0] got;
    while (fs_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "EFS timeout (exp %04x) @%0d", d, cyc);
    end
    got = fs_q.pop_front();
    n_checks++;
    if (got !== {d, lst}) begin
      n_errors++;
      $error("[EFS] got %04x/%0b exp %04x/%0b", got[16:1], got[0], d, lst);
    end
  endtask

  task automatic expect_ss(input logic [15:0] d, input logic lst);
    int wd = 0;
    logic [16:0] got;
    while (ss_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "ESS timeout (exp %04x) @%0d", d, cyc);
    end
    got = ss_q.pop_front();
    n_checks++;
    if (got !== {d, lst}) begin
      n_errors++;
      $error("[ESS] got %04x/%0b exp %04x/%0b", got[16:1], got[0], d, lst);
    end
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

  task automatic expect_tip(input logic [6:0] blk);
    int wd = 0;
    logic [9:0] got;
    while (tip_q.size() == 0) begin
      @(posedge clk);
      wd++;
      if (wd > WAIT_LIMIT) $fatal(1, "ETIP timeout @%0d", cyc);
    end
    got = tip_q.pop_front();
    n_checks++;
    if (got[9:3] !== blk) begin
      n_errors++;
      $error("[ETIP] blk got %0d exp %0d", got[9:3], blk);
    end else begin
      $display("[ETIP] decision: blk=%0d tier=%0d fp16=%0b (value informational;",
               got[9:3], got[2:1], got[0]);
      $display("       TIP decision numerics are covered by verif/tip)");
    end
  endtask

  // ── polls ─────────────────────────────────────────────────────────────────
  task automatic csr_poll(input logic [7:0] a, input logic [31:0] m,
                          input logic [31:0] e);
    int wd = 0;
    logic [31:0] v;
    forever begin
      csr_rd(a, v);
      if ((v & m) === e) break;
      repeat (20) @(posedge clk);
      wd++;
      if (wd > 50000) $fatal(1, "CSRP %02x timeout (got %08x) @%0d", a, v, cyc);
    end
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

  // ── interpreter ────────────────────────────────────────────────────────────
  function automatic void chk32(input string what, input logic [31:0] got,
                                input logic [31:0] exp);
    n_checks++;
    if (got !== exp) begin
      n_errors++;
      $error("[%s] got %08x exp %08x", what, got, exp);
    end
  endfunction

  initial begin : interp
    string vec_path, cmd, rest;
    bit done_seen;
    int fd, i;
    logic [7:0]  a;
    logic [31:0] m, e, d, v32;
    logic [63:0] w64;
    logic [31:0] dw [4];
    logic [255:0] lanes;
    logic         t_op, t_bank;
    logic [1:0]   t_pat;
    logic [4:0]   t_rows, t_sel;
    logic [NB_W-1:0] t_nb;

    // init
    rst_n = 1'b0;
    csr_addr = '0; csr_wdata = '0; csr_write = 1'b0; csr_read = 1'b0;
    kv_awaddr = '0; kv_wdata = '0; kv_awvalid = 1'b0; kv_wvalid = 1'b0;
    kv_bready = 1'b1; kv_araddr = '0; kv_arvalid = 1'b0; kv_rready = 1'b1;
    ds_valid = 1'b0; ds_desc_r = '0;
    xw_valid = 1'b0; xw_data = '0;
    xa_valid = 1'b0; xa_xb = '0; xa_last = 1'b0;
    xg_valid = 1'b0; xg_gb = '0;
    qs_valid = 1'b0; qs_data = '0; cs_valid = 1'b0; cs_data = '0;
    fj_valid = 1'b0; fj_rows = '0;
    qj_valid = 1'b0; qj_mode = 1'b0; qj_cols = '0;
    dj_valid = 1'b0; dj_cols = '0;
    lj_valid = 1'b0; lj_beats = '0; lj_lanes = '0;
    aj_valid = 1'b0; aj_op = 1'b0; aj_bank = 1'b0; aj_pat = '0;
    aj_rows = '0; aj_nb = '0; aj_sel = '0;
    wj_valid = 1'b0; wj_op = 1'b0; wj_bank = 1'b0; wj_pat = '0;
    wj_rows = '0; wj_nb = '0; wj_sel = '0;
    route_r = '0;
    n_checks = 0; n_errors = 0;
    tapf16_n = 0; tapsc_n = 0; tappr_n = 0;
    tapf16_i = 0; tapsc_i = 0; tappr_i = 0;

    vec_path = "build/vectors_smoke.txt";
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
        "CSRW": begin
          void'($fscanf(fd, "%h %h", a, d));
          csr_wr(a[7:0], d);
        end
        "CSRR": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          csr_rd(a[7:0], v32);
          chk32($sformatf("CSRR %02x", a[7:0]), v32 & m, e);
        end
        "CSRP": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          csr_poll(a[7:0], m, e);
        end
        "CSRN": begin
          void'($fscanf(fd, "%h %h", a, m));
          csr_rd(a[7:0], v32);
          n_checks++;
          if ((v32 & m) === 32'h0) begin
            n_errors++;
            $error("[CSRN %02x] expected nonzero, got %08x", a[7:0], v32);
          end
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
        "DESC": begin
          void'($fscanf(fd, "%h %h %h %h", dw[3], dw[2], dw[1], dw[0]));
          drive_ds({dw[3], dw[2], dw[1], dw[0]});
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
        "WB": begin
          void'($fscanf(fd, "%h", w64));
          drive_w(w64);
        end
        "FJOB": begin
          void'($fscanf(fd, "%h", d));
          drive_fj(d[DIM_W-1:0]);
        end
        "QJOB": begin
          void'($fscanf(fd, "%h %h", a, d));
          drive_qj(a[0], d[DIM_W-1:0]);
        end
        "SJOB": begin
          void'($fscanf(fd, "%h", d));
          drive_dj(d[DIM_W-1:0]);
        end
        "LJOB": begin
          void'($fscanf(fd, "%h %h", a, d));
          drive_lj(a[7:0], d[3:0]);
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
        "WJ": begin
          void'($fscanf(fd, "%h", v32)); t_op   = v32[0];
          void'($fscanf(fd, "%h", v32)); t_bank = v32[0];
          void'($fscanf(fd, "%h", v32)); t_pat  = v32[1:0];
          void'($fscanf(fd, "%h", v32)); t_rows = v32[4:0];
          void'($fscanf(fd, "%h", v32)); t_nb   = v32[NB_W-1:0];
          void'($fscanf(fd, "%h", v32)); t_sel  = v32[4:0];
          drive_wj(t_op, t_bank, t_pat, t_rows, t_nb, t_sel);
        end
        "QS": begin
          void'($fscanf(fd, "%h", d));
          drive_qs(d);
        end
        "CS": begin
          void'($fscanf(fd, "%h", d));
          drive_cs(d);
        end
        "EFS": begin
          void'($fscanf(fd, "%h %h", d, v32));
          expect_fs(d[15:0], v32[0]);
        end
        "ESS": begin
          void'($fscanf(fd, "%h %h", d, v32));
          expect_ss(d[15:0], v32[0]);
        end
        "ERO": begin
          for (i = 0; i < 8; i++) begin
            void'($fscanf(fd, "%h", d));
            lanes[32*i +: 32] = d;
          end
          void'($fscanf(fd, "%h", v32));
          expect_ro(lanes, v32[0]);
        end
        "ETIP": begin
          void'($fscanf(fd, "%h", d));
          expect_tip(d[6:0]);
        end
        "TAPF16": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h", v32));
            tapf16_exp[tapf16_n + i] = v32[15:0];
          end
          tapf16_n += int'(d);
        end
        "TAPSC": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h", v32));
            tapsc_exp[tapsc_n + i] = v32;
          end
          tapsc_n += int'(d);
        end
        "TAPPR": begin
          void'($fscanf(fd, "%h", d));
          for (i = 0; i < int'(d); i++) begin
            void'($fscanf(fd, "%h", v32));
            tappr_exp[tappr_n + i] = v32[15:0];
          end
          tappr_n += int'(d);
        end
        "ENDTAPS": begin
          chk32("ENDTAPS f16", 32'(tapf16_i), 32'(tapf16_n));
          chk32("ENDTAPS score", 32'(tapsc_i), 32'(tapsc_n));
          chk32("ENDTAPS prob", 32'(tappr_i), 32'(tappr_n));
        end
        "ESTK": begin
          void'($fscanf(fd, "%h %h", m, e));
          chk32("ESTK", 32'(err_sticky) & m, e);
        end
        "DONE": begin
          // leftover queue check: every scripted expectation must be consumed
          chk32("leftover fs", 32'(fs_q.size()), 0);
          chk32("leftover ss", 32'(ss_q.size()), 0);
          chk32("leftover tip", 32'(tip_q.size()), 0);
          chk32("leftover ro", 32'(ro_q.size()), 0);
          n_checks += tap_checks;
          n_errors += tap_errors;
          $display("TOPSMOKE RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "TOPSMOKE FAIL: %0d errors", n_errors);
          $display("TOPSMOKE PASS");
          done_seen = 1'b1;
          $finish;
        end
        default: $fatal(1, "unknown command '%s'", cmd);
      endcase
      // under --timing, execution continues past $finish inside this time
      // slot (observed 2026-07-09: the loop reached EOF and the no-DONE
      // $fatal below clobbered the PASS exit status) — leave the loop
      // explicitly so the fatal is reachable only on a truly truncated
      // vector file.
      if (done_seen) break;
    end
    if (!done_seen) $fatal(1, "vector file ended without DONE");
  end

endmodule
