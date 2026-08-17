// asu_wide_rms_params.svh — GENERATED FILE, DO NOT EDIT.
//
// Source of truth: apex_golden.attention._wide_mu (golden arbiter, C-RMSW),
// emitted by rtl/asu/gen_wide_rms_params.py. Regenerate + diff:
//   python3 gen_wide_rms_params.py
// Contract: docs/design/WIDE_RMSNORM.md §3 — wide mean path
//   mean2 = (sum2 · WIDE_RMS_MU[k]) >> (WIDE_RMS_S[k] + 16),  k = d / CHUNK.
// Index k = d/128, entry [0] unused (d=0 is not a row). K_MAX = 64 (d = 8192)
// = the golden accumulator proof bound (sum2 <= 2^27). For pow-2 d the entry
// is exactly 2^16, so the μ path reduces bit-exactly to `sum2 >> log2(d)`.
// Product width: sum2 (<= 2^27) · MU (< 2^17) < 2^44 — a 44-bit unsigned
// product; after >> (S+16) the result mean2 <= 2^14 (asserted per entry by
// the generator), so the rsqrt_unit 32-bit operand range is unchanged.
localparam int unsigned WIDE_RMS_CHUNK = 128;
localparam int unsigned WIDE_RMS_K_MAX = 64;

localparam logic [16:0] WIDE_RMS_MU [65] = '{
  17'd0, 17'd65536, 17'd65536, 17'd87381, 17'd65536, 17'd104858, 17'd87381, 17'd74898,
  17'd65536, 17'd116508, 17'd104858, 17'd95325, 17'd87381, 17'd80660, 17'd74898, 17'd69905,
  17'd65536, 17'd123362, 17'd116508, 17'd110376, 17'd104858, 17'd99864, 17'd95325, 17'd91181,
  17'd87381, 17'd83886, 17'd80660, 17'd77672, 17'd74898, 17'd72316, 17'd69905, 17'd67650,
  17'd65536, 17'd127100, 17'd123362, 17'd119837, 17'd116508, 17'd113360, 17'd110376, 17'd107546,
  17'd104858, 17'd102300, 17'd99864, 17'd97542, 17'd95325, 17'd93207, 17'd91181, 17'd89241,
  17'd87381, 17'd85598, 17'd83886, 17'd82241, 17'd80660, 17'd79138, 17'd77672, 17'd76260,
  17'd74898, 17'd73584, 17'd72316, 17'd71090, 17'd69905, 17'd68759, 17'd67650, 17'd66576,
  17'd65536
};

localparam logic [3:0] WIDE_RMS_S [65] = '{
  4'd0, 4'd7, 4'd8, 4'd9, 4'd9, 4'd10, 4'd10, 4'd10,
  4'd10, 4'd11, 4'd11, 4'd11, 4'd11, 4'd11, 4'd11, 4'd11,
  4'd11, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12,
  4'd12, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12, 4'd12,
  4'd12, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13,
  4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13,
  4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13,
  4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13, 4'd13,
  4'd13
};
