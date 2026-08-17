// tb_l4_compose.sv — IB-LAYER L4 host-composition suite through the REAL
// apex_top (IB_LAYER.md §3c). This is the full L4 harness: the generator
// (gen_l4_vectors.py) emits a host OP STREAM from the golden arbiter
// (decoder_layer_fx BUS_ON, D-030), this TB replays it against a real
// apex_top and compares at the tile boundaries. Vector file via +vectors=.
//
// SEGMENT 2 (this file's live coverage): the RMSNorm-1 front-half
// xa -> u_rms -> u_widen -> u_feeder -> u_astage, checked at the feeder
// SCALE bus (fs_*, the EFS op) == r.s_h per row — the C-1 row scale, which
// exercises the whole rmsnorm+feeder amax path. Directly host-drivable (the
// L3 phase-B idiom), CFG_D=128 (D_model of every L4 case).
//
// h8 (the feeder INT8 codes) bit-exactness is a DISCLOSED CARVE-OUT here,
// deferred to SEGMENT 1 (IB_LAYER.md §3c): there the o-proj GEMM consumes the
// h8 codes against the re-derived o8, so any h8 error necessarily breaks the
// o8/r1 match — a real, indirect check. Segment 1 (the o-proj -> r1 back-half
// composition) is the named next slice with its drive plan in §3c.
//
// Exit: "L4COMPOSE RESULT: checks=<n> errors=0 -> PASS" + "L4COMPOSE PASS".

module tb_l4_compose;

  parameter int unsigned CFG_D = 128;     // D_model of the L4 cases
  // I-C GAP-D (additive): the DUT's PER-HEAD width. Default CFG_D keeps the
  // pre-split single-width build byte-identical; the GAP-D split runs pass
  // -GCFG_HD=64 so the SAME segment-2 stream drives a CFG_D=64/CFG_DM=128
  // tile (XR rows, feeder C-1 and the act stage are all MODEL-wide).
  parameter int unsigned CFG_HD = CFG_D;
  parameter int unsigned G_CFG = 16;
  parameter int unsigned DEPTH = 256;

  import apex_pkg::*;

  localparam int unsigned WAIT_LIMIT = 400000;
  localparam int unsigned NB_W = $clog2(CFG_D / 8) + 1;

  logic clk, rst_n;
  int unsigned cyc;
  initial begin clk = 1'b0; forever #5 clk = ~clk; end
  always @(posedge clk) cyc <= cyc + 1;

  int unsigned watchdog_cyc = 8_000_000;
  initial begin
    void'($value$plusargs("watchdog=%d", watchdog_cyc));
    @(posedge clk);
    repeat (watchdog_cyc) @(posedge clk);
    $fatal(1, "WATCHDOG: l4compose did not finish within %0d cycles", watchdog_cyc);
  end

  // ── DUT signals ────────────────────────────────────────────────────────────
  logic [7:0]   csr_addr;
  logic [31:0]  csr_wdata, csr_rdata;
  logic         csr_write, csr_read, csr_ready;

  logic         xa_valid, xa_ready, xa_last;
  logic [7:0]   xa_xb;
  logic         xg_valid, xg_ready;
  logic [15:0]  xg_gb;

  logic             fj_valid, fj_ready;
  logic [DIM_W-1:0] fj_rows;
  logic             aj_valid, aj_ready, aj_op, aj_bank;
  logic [1:0]       aj_pat;
  logic [4:0]       aj_rows, aj_sel;
  logic [NB_W-1:0]  aj_nb;

  logic [15:0]  route_r;

  logic         fs_valid, fs_ready, fs_last;
  logic [15:0]  fs_data;
  logic         ss_valid, ss_last;
  logic [15:0]  ss_data;
  logic         td_valid, td_fp16;
  kvq_tier_e    td_tier;
  logic [6:0]   td_blk;
  logic         ro_valid;
  lane32_beat_t ro_beat;
  logic [15:0]  err_sticky;

  // ── DUT: apex_top at the segment-2 tile config (CFG_D=128) ────────────────
  apex_top #(
    .CFG_D (CFG_HD), .CFG_DM (CFG_D), .KVQ_G (G_CFG), .KVQ_DEPTH (DEPTH),
    .FEED_ROWS_MAX (31), .STAGE_R_MAX (31)
  ) u_dut (
    .clk (clk), .rst_n (rst_n),
    .csr_addr (csr_addr), .csr_wdata (csr_wdata), .csr_write (csr_write),
    .csr_read (csr_read), .csr_rdata (csr_rdata), .csr_ready (csr_ready),
    .kv_awaddr ('0), .kv_awvalid (1'b0), .kv_awready (),
    .kv_wdata ('0), .kv_wvalid (1'b0), .kv_wready (),
    .kv_bresp (), .kv_bvalid (), .kv_bready (1'b0),
    .kv_araddr ('0), .kv_arvalid (1'b0), .kv_arready (),
    .kv_rdata (), .kv_rresp (), .kv_rvalid (), .kv_rready (1'b0),
    .kv_irq (), .kv_evict_needed (), .kv_evict_addr (),
    .ds_valid (1'b0), .ds_ready (), .ds_desc ('0),
    .xw_valid (1'b0), .xw_ready (), .xw_beat ('0),
    .xa_valid (xa_valid), .xa_ready (xa_ready), .xa_x (signed'(xa_xb)),
    .xa_last (xa_last),
    .xg_valid (xg_valid), .xg_ready (xg_ready), .xg_gamma (signed'(xg_gb)),
    .qs_valid (1'b0), .qs_ready (), .qs_data ('0),
    .cs_valid (1'b0), .cs_ready (), .cs_data ('0),
    .fj_valid (fj_valid), .fj_ready (fj_ready), .fj_rows (fj_rows),
    .qj_valid (1'b0), .qj_ready (), .qj_mode (1'b0), .qj_cols ('0),
    .dj_valid (1'b0), .dj_ready (), .dj_cols ('0),
    .lj_valid (1'b0), .lj_ready (), .lj_beats ('0), .lj_lanes ('0),
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
    .wf_valid (), .wf_ready (1'b1), .wf_req (),   // W-G3 combine flip: no fuel fetch in seg-2
    .fs_valid (fs_valid), .fs_ready (fs_ready), .fs_data (fs_data),
    .fs_last (fs_last),
    .ss_valid (ss_valid), .ss_ready (1'b1), .ss_data (ss_data),
    .ss_last (ss_last),
    .td_valid (td_valid), .td_ready (1'b1), .td_fp16 (td_fp16),
    .td_tier (td_tier), .td_blk (td_blk),
    .ro_valid (ro_valid), .ro_ready (1'b1), .ro_beat (ro_beat),
    .dn_mxe (), .dn_feeder (), .dn_squant (), .dn_scored (), .dn_ser (),
    .dn_astage (), .dn_wstage (), .dn_rms (), .dn_asu (),
    .err_sticky (err_sticky),
    .dbg_f16_v (), .dbg_f16_data (), .dbg_f16_last (),
    .dbg_sc_v (), .dbg_sc_data (), .dbg_sc_last (),
    .dbg_pr_v (), .dbg_pr_data (), .dbg_pr_last ()
  );

  // consume dangling observability outputs (-Wall cleanliness)
  logic unused_ok;
  assign unused_ok = &{1'b0, ss_valid, ss_data, ss_last, td_valid, td_fp16,
                       2'(td_tier), td_blk, ro_valid, ro_beat.data,
                       ro_beat.last, err_sticky, csr_ready, route_r[15]};

  // ── feeder-scale collector (the S_H checkpoint bus) ──────────────────────
  int n_checks, n_errors;
  logic [16:0] fs_q [$];
  assign fs_ready = 1'b1;
  always @(posedge clk) if (rst_n && fs_valid) fs_q.push_back({fs_data, fs_last});

  // ── bus / stream tasks (negedge-driven; the S4 lesson — same-slot blocking
  //    drives race the DUT sample) ─────────────────────────────────────────
  task automatic csr_wr(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk);
    csr_addr = a; csr_wdata = d; csr_write = 1'b1;
    @(negedge clk);
    csr_write = 1'b0;
    if (!csr_ready) begin n_errors++; $error("[csr] write %02x: no ready", a); end
  endtask

  task automatic csr_rd(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk);
    csr_addr = a; csr_read = 1'b1;
    @(negedge clk);
    csr_read = 1'b0;
    if (!csr_ready) begin n_errors++; $error("[csr] read %02x: no ready", a); end
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

  task automatic chk(input logic cond, input string what);
    n_checks++;
    if (!cond) begin n_errors++; $error("[chk] %s", what); end
  endtask

  // ── interpreter ────────────────────────────────────────────────────────────
  initial begin : interp
    string vec_path, cmd, rest;
    bit done_seen;
    int fd, i;
    logic [7:0]  a;
    logic [31:0] m, e, d;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] v32;   // small-token scratch (AJ fields / EFS last); high bits unused
    /* verilator lint_on UNUSEDSIGNAL */
    logic        t_op, t_bank;
    logic [1:0]  t_pat;
    logic [4:0]  t_rows, t_sel;
    logic [NB_W-1:0] t_nb;

    rst_n = 1'b0;
    csr_addr = '0; csr_wdata = '0; csr_write = 1'b0; csr_read = 1'b0;
    xa_valid = 1'b0; xa_xb = '0; xa_last = 1'b0;
    xg_valid = 1'b0; xg_gb = '0;
    fj_valid = 1'b0; fj_rows = '0;
    aj_valid = 1'b0; aj_op = 1'b0; aj_bank = 1'b0; aj_pat = '0;
    aj_rows = '0; aj_nb = '0; aj_sel = '0;
    route_r = '0;
    n_checks = 0; n_errors = 0;

    vec_path = "build/compose_seg2.ops";
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
        "CSRP": begin
          void'($fscanf(fd, "%h %h %h", a, m, e));
          csr_poll(a[7:0], m, e);
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
        "DONE": begin
          chk(fs_q.size() == 0, "leftover fs (unconsumed feeder scales)");
          $display("L4COMPOSE RESULT: cycles=%0d checks=%0d errors=%0d",
                   cyc, n_checks, n_errors);
          if (n_errors != 0) $fatal(1, "L4COMPOSE FAIL: %0d errors", n_errors);
          $display("L4COMPOSE PASS (segment 2 RMSNorm-1 front-half; h8 deferred to segment 1, IB_LAYER sec 3c)");
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
