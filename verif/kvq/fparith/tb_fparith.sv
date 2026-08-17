// tb_fparith.sv — EXHAUSTIVE bit-exactness proof of the synthesizable integer
// cq_fp_pkg datapath math against the GOLDEN arbiter (golden/apex_golden,
// via the vectors emitted by gen_fparith_vectors.py; nothing here trusts
// int_model.py).
//
// Sweeps (inf/nan exp==31 excluded everywhere — out of contract, the
// golden's own int cast on them is undefined; +-0 scales excluded from the
// QUANT sweep only, x/0 is undefined):
//
//   SCALE   scale_from_amax: ALL 63488 finite fp16 amax patterns (incl.
//           defensive negatives) x qmax {7,127}, explicit golden expected,
//           function AND cq_scale_unit instance — fully exhaustive at both
//           levels.
//   WIDEN   f16->f32 widen == dequant_one(code=+1) (the CQ-4+ outlier lane):
//           ALL finite fp16 patterns, explicit golden expected, function AND
//           cq_dequant_unit instance — fully exhaustive at both levels.
//   DEQUANT ALL finite fp16 scale patterns (incl. +-0/subnormal/negative) x
//           codes -256..255 (full 9-bit signed port range): function-level
//           exhaustive via per-scale order-sensitive 64-bit checksums of the
//           golden fp32 bits; cq_dequant_unit instance re-checked vs the
//           function on a strided scale sample.
//   QUANT   ALL finite fp16 x-values x the DENSE scale sets (every distinct
//           scale the golden can produce for each qmax + boundary/EPS/
//           subnormal/negative extras): function-level exhaustive via
//           per-scale checksums of the golden codes; cq_quant_unit instance
//           re-checked vs the function on a strided scale sample plus the
//           directed boundary scales.
//
// Checksum (must match gen_fparith_vectors.py exactly):
//   acc += (v + K) * w; w *= R   (mod 2^64, order-sensitive)
//
// Units are one-line combinational wrappers over the very functions swept
// exhaustively here, and the strided/exhaustive instance checks pin the
// wiring; system-level bit-exactness of the units in situ is re-proven by
// verif/kvq/{smoke,sb,cores} against the same golden.
// NOTE (Verilator 5.044 workaround): `continue` inside a for-loop whose body
// also contains a #delay is MISCOMPILED (later statements in the loop body
// are silently corrupted — reproduced standalone). Every sweep loop below
// therefore uses `if (finite...) begin ... end` guards, never `continue`.
`default_nettype none

module tb_fparith;
    import cq_fp_pkg::*;

    `include "fparith_sizes.svh"

    localparam int NF = 63488;                     // finite fp16 patterns
    localparam logic [63:0] CK_K = 64'h9E3779B97F4A7C15;
    localparam logic [63:0] CK_R = 64'hD1B54A32D192ED03;
    localparam int FAIL_MAX = 50;                  // early abort for mutants

    // golden vectors
    logic [15:0] scale_exp4 [0:NF-1];
    logic [15:0] scale_exp8 [0:NF-1];
    logic [31:0] widen_exp  [0:NF-1];
    logic [15:0] qscale4    [0:N_QSCALE4-1];
    logic [63:0] qsum4      [0:N_QSCALE4-1];
    logic [15:0] qscale8    [0:N_QSCALE8-1];
    logic [63:0] qsum8      [0:N_QSCALE8-1];
    logic [63:0] dsum       [0:NF-1];

    // DUT unit instances (the shipping combinational wrappers)
    logic [15:0]       u_amax;
    logic [3:0]        u_sbits;
    wire  [15:0]       u_scale;
    cq_scale_unit u_su (.amax(u_amax), .bits(u_sbits), .scale(u_scale));

    logic [15:0]       u_x, u_qs;
    logic [3:0]        u_qbits;
    wire  [8:0]        u_code;
    cq_quant_unit u_qu (.x(u_x), .s(u_qs), .bits(u_qbits), .code(u_code));

    logic signed [8:0] u_dcode;
    logic [15:0]       u_ds;
    wire  [31:0]       u_hat;
    cq_dequant_unit u_du (.code(u_dcode), .s(u_ds), .hat(u_hat));

    longint checks = 0;
    longint fails  = 0;

    task automatic note_fail(input string msg);
        fails = fails + 1;
        $display("FAIL %s", msg);
        if (fails >= longint'(FAIL_MAX))
            $fatal(1, "FPARITH TB: FAIL (aborted at %0d mismatches)", fails);
    endtask

    // is this fp16 exponent field finite (exp != 31)?
    function automatic bit finite16(input logic [4:0] e);
        finite16 = (e != 5'h1F);
    endfunction

    logic [15:0] b16, sb16, exp16, got16;
    logic [31:0] exp32, got32;
    logic [63:0] acc, w, v;
    longint      qq;
    int          i, si, c, xi;

    initial begin
        $readmemh("build/vec/scale_exp_b4.hex", scale_exp4);
        $readmemh("build/vec/scale_exp_b8.hex", scale_exp8);
        $readmemh("build/vec/widen_exp.hex",    widen_exp);
        $readmemh("build/vec/qscale_b4.hex",    qscale4);
        $readmemh("build/vec/qsum_b4.hex",      qsum4);
        $readmemh("build/vec/qscale_b8.hex",    qscale8);
        $readmemh("build/vec/qsum_b8.hex",      qsum8);
        $readmemh("build/vec/dsum.hex",         dsum);

        // ── SCALE: exhaustive, function + unit, both qmax ──────────────────
        i = 0;
        for (int bb = 0; bb < 65536; bb++) begin
            b16 = 16'(bb);
            if (finite16(b16[14:10])) begin
            // bits = 4
            exp16 = scale_exp4[i];
            got16 = scale_from_amax(b16, 4);
            if (got16 !== exp16)
                note_fail($sformatf("scale fn b4 amax=%04h exp=%04h got=%04h",
                                    b16, exp16, got16));
            u_amax = b16; u_sbits = 4'd4; #1;
            if (u_scale !== exp16)
                note_fail($sformatf("scale UNIT b4 amax=%04h exp=%04h got=%04h",
                                    b16, exp16, u_scale));
            // bits = 8
            exp16 = scale_exp8[i];
            got16 = scale_from_amax(b16, 8);
            if (got16 !== exp16)
                note_fail($sformatf("scale fn b8 amax=%04h exp=%04h got=%04h",
                                    b16, exp16, got16));
            u_sbits = 4'd8; #1;
            if (u_scale !== exp16)
                note_fail($sformatf("scale UNIT b8 amax=%04h exp=%04h got=%04h",
                                    b16, exp16, u_scale));
            checks += 4;
            i++;
            end
        end
        $display("SCALE   done: %0d finite amax x 2 qmax, fn+unit (checks=%0d)",
                 i, checks);

        // ── WIDEN (outlier lane, dequant code=+1): exhaustive fn + unit ────
        i = 0;
        for (int bb = 0; bb < 65536; bb++) begin
            b16 = 16'(bb);
            if (finite16(b16[14:10])) begin
            exp32 = widen_exp[i];
            got32 = dequant_one(9'sd1, b16);
            if (got32 !== exp32)
                note_fail($sformatf("widen fn x=%04h exp=%08h got=%08h",
                                    b16, exp32, got32));
            u_dcode = 9'sd1; u_ds = b16; #1;
            if (u_hat !== exp32)
                note_fail($sformatf("widen UNIT x=%04h exp=%08h got=%08h",
                                    b16, exp32, u_hat));
            checks += 2;
            i++;
            end
        end
        $display("WIDEN   done: %0d finite patterns, fn+unit (checks=%0d)",
                 i, checks);

        // ── DEQUANT: all finite scales x codes -256..255, checksum ─────────
        i = 0;
        for (int bb = 0; bb < 65536; bb++) begin
            sb16 = 16'(bb);
            if (finite16(sb16[14:10])) begin
            acc = 64'd0; w = 64'd1;
            for (c = -256; c <= 255; c++) begin
                v   = 64'({32'd0, dequant_one(9'(c), sb16)});
                acc = acc + ((v + CK_K) * w);
                w   = w * CK_R;
            end
            if (acc !== dsum[i])
                note_fail($sformatf("dequant cksum s=%04h exp=%016h got=%016h",
                                    sb16, dsum[i], acc));
            checks += 1;
            // strided unit-vs-function instance check
            if ((i % 251) == 0) begin
                for (c = -256; c <= 255; c++) begin
                    u_dcode = 9'(c); u_ds = sb16; #1;
                    if (u_hat !== dequant_one(9'(c), sb16))
                        note_fail($sformatf("dequant UNIT s=%04h c=%0d", sb16, c));
                    checks += 1;
                end
            end
            i++;
            end
        end
        $display("DEQUANT done: %0d finite scales x 512 codes (checks=%0d)",
                 i, checks);

        // ── QUANT bits=4: dense scale set x all finite x, checksum ─────────
        for (si = 0; si < N_QSCALE4; si++) begin
            sb16 = qscale4[si];
            acc = 64'd0; w = 64'd1;
            for (int bb = 0; bb < 65536; bb++) begin
                b16 = 16'(bb);
                if (finite16(b16[14:10])) begin
                    qq  = quant_one(b16, sb16, 4);
                    acc = acc + ((64'(qq) + CK_K) * w);
                    w   = w * CK_R;
                end
            end
            if (acc !== qsum4[si])
                note_fail($sformatf("quant b4 cksum s=%04h exp=%016h got=%016h",
                                    sb16, qsum4[si], acc));
            checks += 1;
            if (((si % 397) == 0) || (sb16 == 16'h0400) || (sb16 == 16'h0001)
                || (sb16 == 16'h7BFF) || (sb16 == 16'h8400)) begin
                xi = 0;
                for (int bb = 0; bb < 65536; bb++) begin
                    b16 = 16'(bb);
                    if (finite16(b16[14:10])) begin
                        u_x = b16; u_qs = sb16; u_qbits = 4'd4; #1;
                        if (u_code !== 9'(quant_one(b16, sb16, 4)))
                            note_fail($sformatf("quant UNIT b4 x=%04h s=%04h",
                                                b16, sb16));
                        xi++;
                    end
                end
                checks += longint'(xi);
            end
            if ((si % 4000) == 0)
                $display("  quant b4 progress: scale %0d/%0d", si, N_QSCALE4);
        end
        $display("QUANT4  done: %0d scales x 63488 x (checks=%0d)",
                 N_QSCALE4, checks);

        // ── QUANT bits=8: dense scale set x all finite x, checksum ─────────
        for (si = 0; si < N_QSCALE8; si++) begin
            sb16 = qscale8[si];
            acc = 64'd0; w = 64'd1;
            for (int bb = 0; bb < 65536; bb++) begin
                b16 = 16'(bb);
                if (finite16(b16[14:10])) begin
                    qq  = quant_one(b16, sb16, 8);
                    acc = acc + ((64'(qq) + CK_K) * w);
                    w   = w * CK_R;
                end
            end
            if (acc !== qsum8[si])
                note_fail($sformatf("quant b8 cksum s=%04h exp=%016h got=%016h",
                                    sb16, qsum8[si], acc));
            checks += 1;
            if (((si % 397) == 0) || (sb16 == 16'h0400) || (sb16 == 16'h0001)
                || (sb16 == 16'h7BFF) || (sb16 == 16'h8400)) begin
                xi = 0;
                for (int bb = 0; bb < 65536; bb++) begin
                    b16 = 16'(bb);
                    if (finite16(b16[14:10])) begin
                        u_x = b16; u_qs = sb16; u_qbits = 4'd8; #1;
                        if (u_code !== 9'(quant_one(b16, sb16, 8)))
                            note_fail($sformatf("quant UNIT b8 x=%04h s=%04h",
                                                b16, sb16));
                        xi++;
                    end
                end
                checks += longint'(xi);
            end
            if ((si % 4000) == 0)
                $display("  quant b8 progress: scale %0d/%0d", si, N_QSCALE8);
        end
        $display("QUANT8  done: %0d scales x 63488 x (checks=%0d)",
                 N_QSCALE8, checks);

        // ── verdict ─────────────────────────────────────────────────────────
        if (fails != 0)
            $fatal(1, "FPARITH TB: FAIL (checks=%0d fails=%0d)", checks, fails);
        $display("FPARITH TB: ALL PASS (checks=%0d fails=0)", checks);
        $finish;
    end
endmodule
`default_nettype wire
