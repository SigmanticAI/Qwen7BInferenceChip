// tb_csr_sb.sv — CSR vector-driven scoreboard TB (house pattern, verif/mxe/sb
// lineage). Verilator --binary --timing --assert.
//
// All expected read data for the architectural registers comes from the
// independent golden model in gen_csr_vectors.py. The PERF_* cycle counters
// are cycle-accurate, so they are checked against a TB-side BUS-LEVEL mirror
// (V records): the mirror re-implements the documented counter contract
// (enable gate, clear-wins, saturate-never-wrap) from the bus signals only.
// Register OUTPUT ports (enable/threshold/tier_sel/tip_override) are
// cross-checked against every successful read-back of their register.
//
// The CSR bus is request/response, not a §5 valid/ready stream, so
// apex_stream1_sva does not apply here; the CSR-shaped contract (response
// discipline, sticky semantics, pulse outputs) is bound from csr_sb_sva.svh
// and compiled under --assert in every build (D-012).
//
// Plusargs: +vectors=<file> +seed=<n> (pacing rng)
// Records: see gen_csr_vectors.py header (W/R/S/Y/B/P/Z/T/V/X/E).
//
// TB discipline (V0/mxe lineage): stimulus at NEGEDGE, checks sampled 1 ns
// after the edge; $fatal watchdog bounds the run.

`include "csr_sb_sva.svh"

module tb_csr_sb #(
    parameter int unsigned PERF_W = 32
);

  import apex_pkg::*;

  localparam int unsigned N_BLOCKS = 8;
  localparam int unsigned IMP_AW   = 7;

  // ── clock / cycle counter / watchdog ──────────────────────────────────────
  logic clk;
  logic rst_n;
  int unsigned cyc;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    repeat (10_000_000) @(posedge clk);
    $fatal(1, "WATCHDOG: simulation did not finish (cyc=%0d)", cyc);
  end

  // ── DUT ────────────────────────────────────────────────────────────────────
  logic [7:0]          bus_addr;
  logic [31:0]         bus_wdata;
  logic                bus_wr, bus_rd;
  logic [31:0]         rdata;
  logic                ready;
  logic                enable, soft_reset;
  logic [N_BLOCKS-1:0] busy_lvl;
  logic                derr;
  logic                desc_error_sticky;
  kvq_tier_e           tier_sel;
  logic                tip_override;
  logic [4:0]          threshold;
  logic                flush;
  logic [IMP_AW-1:0]   imp_rd_addr;
  logic [15:0]         imp_rd_data_i;
  logic [1:0]          imp_rd_tier_i;

  csr_regs #(
    .N_BLOCKS (N_BLOCKS),
    .IMP_AW   (IMP_AW),
    .PERF_W   (PERF_W)
  ) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .addr              (bus_addr),
    .wdata             (bus_wdata),
    .write             (bus_wr),
    .read              (bus_rd),
    .rdata             (rdata),
    .ready             (ready),
    .enable            (enable),
    .soft_reset        (soft_reset),
    .block_busy_i      (busy_lvl),
    .desc_error_i      (derr),
    .desc_error_sticky (desc_error_sticky),
    .tier_sel          (tier_sel),
    .tip_override      (tip_override),
    .threshold         (threshold),
    .flush             (flush),
    .imp_rd_addr       (imp_rd_addr),
    .imp_rd_data_i     (imp_rd_data_i),
    .imp_rd_tier_i     (imp_rd_tier_i)
  );

  // ── fake TIP importance window (MUST match gen_csr_vectors.py imp_data) ───
  function automatic logic [15:0] imp_data(input logic [IMP_AW-1:0] a);
    return 16'((32'(a) * 32'h9E37) ^ 32'h1234);
  endfunction
  assign imp_rd_data_i = imp_data(imp_rd_addr);
  assign imp_rd_tier_i = 2'(32'(imp_rd_addr) % 3);

  // ── CSR contract SVA (whitebox bind; --assert build gate, D-012) ──────────
  bind csr_regs csr_sb_sva u_sva (
    .clk               (clk),
    .rst_n             (rst_n),
    .read              (read),
    .write             (write),
    .ready             (ready),
    .desc_error_i      (desc_error_i),
    .desc_error_sticky (desc_error_sticky),
    .w1c_fire          (w1c_fire),
    .soft_reset        (soft_reset),
    .flush             (flush),
    .wr_srst           (wr_srst),
    .wr_flush          (wr_flush)
  );

  // ── PERF bus-level mirror (independent re-implementation of the contract) ─
  localparam logic [PERF_W-1:0] MAXV = {PERF_W{1'b1}};
  logic                mir_en;
  logic [PERF_W-1:0]   mir_cyc;
  logic [PERF_W-1:0]   mir_cnt [N_BLOCKS];
  logic [31:0]         mir_rd;
  logic                mwr;
  assign mwr = bus_wr && (bus_addr[1:0] == 2'b00);

  always @(posedge clk) begin
    if (!rst_n) begin
      mir_en  <= 1'b0;
      mir_cyc <= '0;
      mir_rd  <= '0;
      for (int unsigned i = 0; i < N_BLOCKS; i++) mir_cnt[i] <= '0;
    end else begin
      if (bus_rd && bus_addr[1:0] == 2'b00) begin
        if (bus_addr[7:2] == 6'h0D) mir_rd <= 32'(mir_cyc);
        for (int unsigned i = 0; i < N_BLOCKS; i++)
          if (bus_addr[7:2] == 6'h0E + 6'(i)) mir_rd <= 32'(mir_cnt[i]);
      end
      if (mwr && bus_addr[7:2] == 6'h0C) mir_en <= bus_wdata[0];
      if (mwr && bus_addr[7:2] == 6'h0C && bus_wdata[1]) begin
        mir_cyc <= '0;
        for (int unsigned i = 0; i < N_BLOCKS; i++) mir_cnt[i] <= '0;
      end else if (mir_en) begin
        if (mir_cyc != MAXV) mir_cyc <= mir_cyc + 1'b1;
        for (int unsigned i = 0; i < N_BLOCKS; i++)
          if (busy_lvl[i] && mir_cnt[i] != MAXV) mir_cnt[i] <= mir_cnt[i] + 1'b1;
      end
    end
  end

  // ── pulse monitors (each write-strobe must yield EXACTLY one high cycle) ──
  int srst_seen, flush_seen, exp_srst, exp_flush;
  always @(posedge clk) begin
    if (rst_n) begin
      if (soft_reset) srst_seen  <= srst_seen + 1;
      if (flush)      flush_seen <= flush_seen + 1;
    end
  end

  // ── bookkeeping / coverage (manual, run2 pattern; "COV <name> <n>") ───────
  int errors, checked;
  int unsigned seed = 32'hC5A_0001;
  int unsigned rng_state;
  string vec_path;

  int cov_rd [0:63];
  int cov_wr [0:63];
  int cov_rsvd_rd, cov_unaligned_rd, cov_rsvd_wr;
  int cov_w1c_write, cov_sticky_set, cov_sticky_race;
  int cov_tier_clamp, cov_thresh_mask, cov_imp_mask;
  int cov_busy_all, cov_busy_some, cov_idle_seen;
  int cov_perf_en, cov_perf_clear, cov_perf_clr_en, cov_perf_dis_hold;
  int cov_perf_sat, cov_perf_nonzero;
  int cov_rw_simul, cov_b2b_rd, cov_hard_reset;
  int cov_out_enable, cov_out_thresh, cov_out_tier;
  int cov_pw32, cov_pw8;

  localparam int unsigned NAMED_W = 22;
  localparam logic [5:0] NAMED [NAMED_W] = '{
    6'h00, 6'h01, 6'h02, 6'h03, 6'h04, 6'h05, 6'h06, 6'h08, 6'h09, 6'h0A,
    6'h0B, 6'h0C, 6'h0D, 6'h0E, 6'h0F, 6'h10, 6'h11, 6'h12, 6'h13, 6'h14,
    6'h15, 6'h07};

  function automatic void print_coverage();
    for (int i = 0; i < int'(NAMED_W); i++) begin
      $display("COV rd_%02x %0d", 32'(NAMED[i]) * 4, cov_rd[NAMED[i]]);
      $display("COV wr_%02x %0d", 32'(NAMED[i]) * 4, cov_wr[NAMED[i]]);
    end
    $display("COV rsvd_rd %0d",       cov_rsvd_rd);
    $display("COV unaligned_rd %0d",  cov_unaligned_rd);
    $display("COV rsvd_wr %0d",       cov_rsvd_wr);
    $display("COV w1c_write %0d",     cov_w1c_write);
    $display("COV sticky_set %0d",    cov_sticky_set);
    $display("COV sticky_race %0d",   cov_sticky_race);
    $display("COV tier_clamp %0d",    cov_tier_clamp);
    $display("COV thresh_mask %0d",   cov_thresh_mask);
    $display("COV imp_mask %0d",      cov_imp_mask);
    $display("COV busy_all %0d",      cov_busy_all);
    $display("COV busy_some %0d",     cov_busy_some);
    $display("COV idle_seen %0d",     cov_idle_seen);
    $display("COV perf_en %0d",       cov_perf_en);
    $display("COV perf_clear %0d",    cov_perf_clear);
    $display("COV perf_clr_en %0d",   cov_perf_clr_en);
    $display("COV perf_dis_hold %0d", cov_perf_dis_hold);
    $display("COV perf_sat %0d",      cov_perf_sat);
    $display("COV perf_nonzero %0d",  cov_perf_nonzero);
    $display("COV rw_simul %0d",      cov_rw_simul);
    $display("COV b2b_rd %0d",        cov_b2b_rd);
    $display("COV hard_reset %0d",    cov_hard_reset);
    $display("COV out_enable_chk %0d", cov_out_enable);
    $display("COV out_thresh_chk %0d", cov_out_thresh);
    $display("COV out_tier_chk %0d",  cov_out_tier);
    $display("COV perfw_32 %0d",      cov_pw32);
    $display("COV perfw_8 %0d",       cov_pw8);
  endfunction

  function automatic int unsigned rnd();
    rng_state = rng_state * 32'd1664525 + 32'd1013904223;
    return rng_state;
  endfunction

  // ── record decode helpers (pulse expectations + coverage) ─────────────────
  function automatic void note_write(input logic [7:0] a, input logic [31:0] d);
    logic [5:0] w;
    w = a[7:2];
    if (a[1:0] != 2'b00) begin
      cov_rsvd_wr++;
      return;                                  // unaligned: ignored by DUT
    end
    cov_wr[w]++;
    if (w == 6'h00 && d[1]) exp_srst++;
    if (w == 6'h0A)         exp_flush++;
    if (w == 6'h01 && d[1]) cov_w1c_write++;
    if (w == 6'h08 && d[1:0] == 2'b11) cov_tier_clamp++;
    if (w == 6'h09 && d[31:5] != '0)   cov_thresh_mask++;
    if (w == 6'h0B && d[31:IMP_AW] != '0) cov_imp_mask++;
    if (w == 6'h0C) begin
      if (d[0]) cov_perf_en++;
      if (d[1] && !d[0]) cov_perf_clear++;
      if (d[1] && d[0])  cov_perf_clr_en++;
    end
    if (w == 6'h07 || w > 6'h15) cov_rsvd_wr++;
    begin
      logic unused_ok;
      unused_ok = &{1'b0, d[4:2]};   // fields covered via masked compares
    end
  endfunction

  function automatic void note_read(input logic [7:0] a);
    logic [5:0] w;
    w = a[7:2];
    if (a[1:0] != 2'b00) begin
      cov_unaligned_rd++;
      return;
    end
    cov_rd[w]++;
    if (w == 6'h07 || w > 6'h15) cov_rsvd_rd++;
  endfunction

  // cross-check register OUTPUT ports against a just-verified read-back
  function automatic void check_outputs(input logic [7:0] a,
                                        input logic [31:0] rd);
    if (a[1:0] != 2'b00) return;
    case (a[7:2])
      6'h00: begin
        if (enable !== rd[0]) begin
          errors++;
          $display("FAIL cyc=%0d: enable port %0b != CTRL[0] %0b",
                   cyc, enable, rd[0]);
        end
        cov_out_enable++;
      end
      6'h08: begin
        if ({tip_override, 2'(tier_sel)} !== {rd[8], rd[1:0]}) begin
          errors++;
          $display("FAIL cyc=%0d: tier/override ports %0b/%0d != TIER_CTRL %08x",
                   cyc, tip_override, 2'(tier_sel), rd);
        end
        cov_out_tier++;
      end
      6'h09: begin
        if (threshold !== rd[4:0]) begin
          errors++;
          $display("FAIL cyc=%0d: threshold port %0d != THRESHOLD %0d",
                   cyc, threshold, rd[4:0]);
        end
        cov_out_thresh++;
      end
      default: ;
    endcase
  endfunction

  // ── bus primitives (negedge discipline) ────────────────────────────────────
  task automatic pace_gap();
    int unsigned g;
    g = ((rnd() & 3) == 0) ? (rnd() % 3) : 0;
    repeat (g) @(negedge clk);
  endtask

  // single request; response checked one cycle later; ready must then drop
  task automatic bus_op(input bit do_rd, input bit do_wr,
                        input logic [7:0] a, input logic [31:0] d,
                        input bit do_derr, input bit chk,
                        input logic [31:0] exp, input string what);
    pace_gap();
    @(negedge clk);
    bus_rd    = do_rd;
    bus_wr    = do_wr;
    bus_addr  = a;
    bus_wdata = d;
    derr      = do_derr;
    @(negedge clk);
    bus_rd    = 1'b0;
    bus_wr    = 1'b0;
    bus_addr  = '0;
    bus_wdata = '0;
    derr      = 1'b0;
    #1;
    if (ready !== 1'b1) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: no ready one cycle after request @%02x",
               cyc, what, a);
    end
    if (chk && rdata !== exp) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: read @%02x got %08x exp %08x",
               cyc, what, a, rdata, exp);
    end
    if (chk) begin
      // rdata is the PRE-write value on simultaneous read+write, so the
      // output-port cross-check is only meaningful on pure reads
      if (!do_wr) check_outputs(a, rdata);
      checked++;
    end
    @(negedge clk);
    #1;
    if (ready !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d [%s]: ready stuck after response", cyc, what);
    end
  endtask

  // two back-to-back pipelined reads (one response per cycle)
  task automatic bus_b2b(input logic [7:0] a1, input logic [31:0] e1,
                         input logic [7:0] a2, input logic [31:0] e2);
    pace_gap();
    @(negedge clk);
    bus_rd   = 1'b1;
    bus_addr = a1;
    @(negedge clk);
    bus_addr = a2;
    #1;
    if (ready !== 1'b1 || rdata !== e1) begin
      errors++;
      $display("FAIL cyc=%0d [b2b1]: @%02x got rdy=%0b %08x exp %08x",
               cyc, a1, ready, rdata, e1);
    end
    @(negedge clk);
    bus_rd   = 1'b0;
    bus_addr = '0;
    #1;
    if (ready !== 1'b1 || rdata !== e2) begin
      errors++;
      $display("FAIL cyc=%0d [b2b2]: @%02x got rdy=%0b %08x exp %08x",
               cyc, a2, ready, rdata, e2);
    end
    checked += 2;
    cov_b2b_rd++;
    @(negedge clk);
    #1;
    if (ready !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d [b2b]: ready stuck after pipelined reads", cyc);
    end
  endtask

  task automatic pulse_derr();
    @(negedge clk);
    derr = 1'b1;
    @(negedge clk);
    derr = 1'b0;
    cov_sticky_set++;
  endtask

  task automatic hard_reset();
    @(negedge clk);
    rst_n     = 1'b0;
    bus_rd    = 1'b0;
    bus_wr    = 1'b0;
    derr      = 1'b0;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);
    #1;
    if (ready !== 1'b0 || desc_error_sticky !== 1'b0 || enable !== 1'b0) begin
      errors++;
      $display("FAIL cyc=%0d: dirty state out of hard reset", cyc);
    end
    cov_hard_reset++;
  endtask

  // ── vector-file reader ─────────────────────────────────────────────────────
  int    fd;
  string cur_line;

  function automatic bit next_line();
    while ($fgets(cur_line, fd) > 0) begin
      if (cur_line.len() > 1 && cur_line.getc(0) != "#") return 1'b1;
    end
    return 1'b0;
  endfunction

  // ── main ───────────────────────────────────────────────────────────────────
  initial begin : main
    string tag;
    logic [7:0] a, a2;
    logic [31:0] d, e, e2;
    int n, cnt;
    bit ended;

    rst_n     = 1'b0;
    bus_rd    = 1'b0;
    bus_wr    = 1'b0;
    bus_addr  = '0;
    bus_wdata = '0;
    busy_lvl  = '0;
    derr      = 1'b0;
    errors    = 0;
    checked   = 0;

    void'($value$plusargs("seed=%d", seed));
    if (!$value$plusargs("vectors=%s", vec_path))
      $fatal(1, "missing +vectors=<file>");
    rng_state = seed ^ 32'hC5AD_BEEF;

    if (PERF_W == 32) cov_pw32++;
    if (PERF_W == 8)  cov_pw8++;

    fd = $fopen(vec_path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", vec_path);
    $display("[TB] PERF_W=%0d vectors=%s seed=%0d", PERF_W, vec_path, seed);

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    ended = 1'b0;
    while (!ended && next_line()) begin
      case (cur_line.getc(0))
        "W": begin
          cnt = $sscanf(cur_line, "%s %h %h", tag, a, d);
          if (cnt != 3 || tag != "W") $fatal(1, "bad W line: %s", cur_line);
          note_write(a, d);
          bus_op(0, 1, a, d, 0, 0, '0, "W");
        end
        "R": begin
          cnt = $sscanf(cur_line, "%s %h %h", tag, a, e);
          if (cnt != 3) $fatal(1, "bad R line: %s", cur_line);
          note_read(a);
          bus_op(1, 0, a, '0, 0, 1, e, "R");
        end
        "S": begin
          cnt = $sscanf(cur_line, "%s %h %h %h", tag, a, d, e);
          if (cnt != 4) $fatal(1, "bad S line: %s", cur_line);
          note_write(a, d);
          note_read(a);
          bus_op(1, 1, a, d, 0, 1, e, "S");
          cov_rw_simul++;
        end
        "Y": begin
          cnt = $sscanf(cur_line, "%s %h %h %h %h", tag, a, e, a2, e2);
          if (cnt != 5) $fatal(1, "bad Y line: %s", cur_line);
          note_read(a);
          note_read(a2);
          bus_b2b(a, e, a2, e2);
        end
        "B": begin
          cnt = $sscanf(cur_line, "%s %h", tag, d);
          if (cnt != 2) $fatal(1, "bad B line: %s", cur_line);
          @(negedge clk);
          busy_lvl = d[N_BLOCKS-1:0];
          if (d[N_BLOCKS-1:0] == '1) cov_busy_all++;
          else if (d[N_BLOCKS-1:0] != '0) cov_busy_some++;
          else cov_idle_seen++;
        end
        "P": pulse_derr();
        "Z": begin
          cnt = $sscanf(cur_line, "%s %h", tag, d);
          if (cnt != 2) $fatal(1, "bad Z line: %s", cur_line);
          note_write(8'h04, d);
          bus_op(0, 1, 8'h04, d, 1, 0, '0, "Z");   // W1C races the set
          cov_sticky_race++;
        end
        "T": begin
          cnt = $sscanf(cur_line, "%s %d", tag, n);
          if (cnt != 2) $fatal(1, "bad T line: %s", cur_line);
          repeat (n) @(negedge clk);
        end
        "V": begin
          cnt = $sscanf(cur_line, "%s %h", tag, a);
          if (cnt != 2) $fatal(1, "bad V line: %s", cur_line);
          note_read(a);
          // response is compared against the mirror value latched at the
          // same request edge; do the compare inline (mir_rd is settled at
          // the #1 sample point of bus_op's response cycle)
          begin
            pace_gap();
            @(negedge clk);
            bus_rd   = 1'b1;
            bus_addr = a;
            @(negedge clk);
            bus_rd   = 1'b0;
            bus_addr = '0;
            #1;
            if (ready !== 1'b1 || rdata !== mir_rd) begin
              errors++;
              $display("FAIL cyc=%0d [V]: perf @%02x got %08x mirror %08x",
                       cyc, a, rdata, mir_rd);
            end
            checked++;
            if (mir_rd == 32'(MAXV)) cov_perf_sat++;
            if (mir_rd != '0)        cov_perf_nonzero++;
            if (!mir_en && mir_rd != '0) cov_perf_dis_hold++;
            @(negedge clk);
          end
        end
        "X": hard_reset();
        "E": begin
          ended = 1'b1;      // before $finish: post-$finish stmts don't run
          repeat (4) @(negedge clk);
          #1;
          if (srst_seen != exp_srst) begin
            errors++;
            $display("FAIL: soft_reset pulse cycles %0d != write strobes %0d",
                     srst_seen, exp_srst);
          end
          if (flush_seen != exp_flush) begin
            errors++;
            $display("FAIL: flush pulse cycles %0d != write strobes %0d",
                     flush_seen, exp_flush);
          end
          print_coverage();
          if (errors == 0) begin
            $display("TB PASS: %0d checked reads, %0d soft_reset + %0d flush pulses, 0 errors (PERF_W=%0d seed=%0d)",
                     checked, srst_seen, flush_seen, PERF_W, seed);
            $finish;
          end else begin
            $fatal(1, "TB FAIL: %0d error(s), %0d checked reads (PERF_W=%0d)",
                   errors, checked, PERF_W);
          end
        end
        default: $fatal(1, "unexpected record: %s", cur_line);
      endcase
    end
    if (!ended) $fatal(1, "vector file ended without an E record");
  end

endmodule
