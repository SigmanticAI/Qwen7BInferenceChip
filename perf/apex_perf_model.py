#!/usr/bin/env python3
"""
apex_perf_model.py — S7 PERF-MODEL: the APEX-7B analytic performance model.

EVERY OUTPUT OF THIS PROGRAM IS A **PROJECTION**, NOT A MEASUREMENT.
No APEX silicon exists. The only measured quantities are the simulation /
place-and-route calibration anchors listed in CAL (each with its in-repo
source). Everything downstream is arithmetic on stated assumptions, and the
program prints every assumption it uses before it prints a single result.

What it models
  * decode tokens/s vs context length, array size, clock, memory system
  * prefill / time-to-first-token (TTFT)
  * energy per token vs context, KV compression on/off
  * multi-tile scaling
  * a cited comparator table (published single-stream numbers, other devices)

Design rules
  * pure Python 3.9 stdlib — reproducible from a clone, no pip installs
  * calibrated to MEASURED cycle data (docs/OPTIMIZATION.md, ECP5 P&R report,
    golden effective-bits gate) — the anchors are asserted in --check mode
  * the honest-accounting rule of golden/tests/test_effective_bits.py applies:
    KV traffic uses the *stored record* accounting (what the RTL actually
    writes to SRAM/DRAM), not the codec ceiling; the ceiling is shown as a
    separate labeled curve.

Usage
  python3 perf/apex_perf_model.py            # writes docs/results/perf_model/
  python3 perf/apex_perf_model.py --check    # validate calibration anchors only
"""

import argparse
import csv
import math
import os
import sys

# ============================================================================
# 1. CALIBRATION — the only measured numbers in this file.
#    Each entry: value + where the measurement lives in this repo.
# ============================================================================

CAL = {
    # docs/OPTIMIZATION.md (grounding header): "~300 MMIO round-trips/step,
    # tile idle ~90%" — measured on the host-sequenced L3 attention step.
    "mmio_per_l3_step": 300,
    "tile_idle_frac_measured": 0.90,

    # docs/OPTIMIZATION.md A1: ASU divider emission = 33 cycles/element.
    # At a T=128 step that is 33 * 128^2 = 540,672 cycles ("~540k")
    # vs ~65,000 cycles for all six GEMMs of the step combined ("MXE 65k").
    "asu_div_cyc_per_elem": 33,
    "mxe_all6_gemm_cyc_T128": 65_000,

    # docs/OPTIMIZATION.md B4: "the measured 3.3x-off-ideal gap"
    # -> measured MXE utilization ~= 1/3.27 = 30.6%.
    "mxe_util_measured": 0.306,

    # docs/results/kvq_ecp5_report.txt: KVQ engine P&R on ECP5-85F,
    # Fmax 9.04 MHz (un-pipelined fp16 divider limits the clock; task S10).
    "ecp5_fmax_mhz": 9.04,

    # rtl/apex_pkg.sv: MXE_N = 8 (8x8 array), K_MAX = 2048; the tile weight
    # port accepts one 8-lane INT8 beat per cycle = 8 bytes/cycle.
    "tile_array_n": 8,
    "k_max": 2048,
    "weight_port_bytes_per_cyc": 8,

    # golden/tests/test_effective_bits.py — pinned STORED and CODEC
    # accountings (D=128, the 7B head_dim), FP16 baseline = 16 b/v.
    # STORED = RTL record as written (tag+lanes+pad+scale bank, A2/D-026).
    "kv_bits_stored_kvq4_d128": 4.5625,   # 3.5068x
    "kv_bits_stored_kvq8_d128": 8.5,      # 1.8824x
    "kv_bits_codec_kvq4_d128": 4.125,     # 3.879x  (method ceiling)
    "kv_bits_codec_kvq4p_d128": 4.2178,   # 3.793x
}

# Weight-port roofline anchors (derivable from CAL): today's tile, INT8
# weights, decode weight-port-bound tok/s at f MHz
# = f*1e6 * 8 B/cyc / weight_bytes_per_token. Asserted in --check.
ANCHOR_TOKS_AT_MHZ = {9.04: 0.010, 100: 0.113, 400: 0.452}

# ============================================================================
# 2. ASSUMPTIONS — everything that is *not* measured. Printed verbatim into
#    the report. Change them here; nowhere else.
# ============================================================================

# --- workload: Qwen2.5-7B (the S5/S8 target model) -------------------------
MODEL = {
    "name": "Qwen2.5-7B",
    "n_layers": 28,
    "hidden": 3584,
    "n_heads": 28,
    "n_kv_heads": 4,          # GQA — NOT yet in golden/RTL (task S6); assumed
    "head_dim": 128,
    "ffn": 18944,             # SwiGLU: gate + up + down
    "vocab": 152064,
    "tie_embeddings": False,
}

# --- weight precisions ------------------------------------------------------
WEIGHT_MODES = {
    # today's verified datapath stores INT8 weights
    "INT8": 8.0,
    # B3 native-W4 path (docs/OPTIMIZATION.md Tier B) — NOT built yet.
    # 4-bit codes + fp16 scale per 128-group, same padding-inclusive
    # accounting we apply to KV: 4 + 16/128 = 4.125 b/w.
    "W4": 4.125,
    # B3 stage-6 ADOPTION GEOMETRY (docs/results/b3_w4_adoption/RESULTS.md,
    # 2026-07-21): tile-wide scales collapse the 7B model; the measured
    # adoption config is G=32 K-grouping — fp16 scale per 32 weights:
    # 4 + 16/32 = 4.5 b/w. The quality-viable W4 costs ~9% more weight
    # bytes than the G=128 amortization above.
    "W4-G32": 4.5,
}

# --- KV storage modes (bits/value as stored; D=128 pins from the gate) ------
KV_MODES = {
    "FP16":         {"bits": 16.0,  "label": "KV FP16 (uncompressed)"},
    "INT8-KV":      {"bits": 8.0,   "label": "KV INT8 (common GPU default)"},
    "KVQ8-stored":  {"bits": CAL["kv_bits_stored_kvq8_d128"],
                     "label": "KVQ8 as stored (RTL record)"},
    "KVQ4-stored":  {"bits": CAL["kv_bits_stored_kvq4_d128"],
                     "label": "KVQ4 as stored (RTL record, A2/D-026)"},
    "KVQ4-codec":   {"bits": CAL["kv_bits_codec_kvq4_d128"],
                     "label": "KVQ4 codec ceiling (not a record layout)"},
}

# --- memory systems ---------------------------------------------------------
MEM_SYSTEMS = {
    # peak GB/s = MT/s * bus_bytes. LPDDR5X-8533 x64 = 8533e6 * 8 = 68.26 GB/s
    "LPDDR5X x32 (34.1 GB/s)":  34.132,
    "LPDDR5X x64 (68.3 GB/s)":  68.264,   # the S13 spec point (dual x32)
    "LPDDR5X x128 (136.5 GB/s)": 136.528,
}
MEM_EFF_RANGE = (0.65, 0.70, 0.75)   # sustained/peak for single-stream reads
MEM_EFF_DEFAULT = 0.70

# --- array / clock scenarios -------------------------------------------------
ARRAY_SIZES = (8, 16, 32, 64, 128)          # N x N INT8 MACs
CLOCKS_GHZ = (0.2, 0.5, 0.8, 1.0, 1.2)
SPEC_ARRAY = 64                              # S13 robust config
SPEC_CLOCKS = (0.5, 0.8)

# MXE utilization scenarios for prefill (fraction of peak MACs sustained):
UTIL_SCENARIOS = {
    # measured today on the 8x8 host-sequenced tile (OPTIMIZATION.md 3.3x gap)
    "measured-today (30.6%)": CAL["mxe_util_measured"],
    # after A1 radix-4 divider + B4 ingest-under-compute land (targets, not
    # measurements — both tasks are on the OPTIMIZATION.md backlog)
    "post-A1/B4 target (65%)": 0.65,
    # ideal ceiling for a well-fed systolic array
    "ideal (90%)": 0.90,
}
UTIL_DEFAULT_KEY = "post-A1/B4 target (65%)"

# --- host sequencing ---------------------------------------------------------
HOST = {
    "mmio_round_trip_us": 1.5,   # PCIe/MMIO round-trip, typical posted-read
    # today: host sequences every tile job. MMIO per job calibrated from the
    # measured 300 MMIO / L3 attention step (~60 descriptor-scale jobs/step).
    "mmio_per_job_host_seq": 5.0,
    # B1 hardware layer-walker (OPTIMIZATION.md Tier B): <=3 MMIO per
    # layer-step. Assumed for all ASIC projections; printed as assumption.
    "mmio_per_layer_walker": 3.0,
}

# --- energy ------------------------------------------------------------------
ENERGY = {
    # DRAM system energy (device+PHY+controller). Published anchors:
    # DDR5 ~14-16 pJ/b, GDDR7 ~9-11 pJ/b, HBM2 ~3.9 pJ/b; LPDDR5X sits
    # between HBM and DDR5. We sweep 4-8 pJ/b and default to 6.
    "dram_pj_per_bit_sweep": (4.0, 6.0, 8.0),
    "dram_pj_per_bit_default": 6.0,
    # INT8 MAC incl. local operand movement, 16nm-class (Horowitz ISSCC'14
    # scaled): swept implicitly via the +/- band; default 0.35 pJ/MAC.
    "mac_pj": 0.35,
    # on-chip SRAM traffic folded into a flat overhead on MAC energy
    "sram_overhead_factor": 1.5,
    # always-on: LPDDR PHY+controller idle, PLLs, leakage
    "static_w": 0.5,
}

# --- misc --------------------------------------------------------------------
PREFILL_CHUNK_T = 128     # T<=128 per job (apex_top contract); weights are
                          # re-streamed once per chunk (no on-die weight cache)
CTX_SWEEP = [0, 1024, 2048, 4096, 8192, 16384, 21000, 32768, 49152, 65536,
             78000, 98304, 131072]
TOKS_FLOOR = 10.0         # the "doesn't-deeply-suck" decode floor (S13)
TTFT_PROMPT = 1024        # TTFT floor is spec'd at a 1k-token prompt
TILES_SWEEP = (1, 2, 4, 8, 16)

# ============================================================================
# 3. Derived workload counts (computed, printed — never hand-entered)
# ============================================================================


def model_counts(m=MODEL):
    h, hd = m["hidden"], m["head_dim"]
    q_dim = m["n_heads"] * hd
    kv_dim = m["n_kv_heads"] * hd
    attn_proj = h * q_dim + 2 * h * kv_dim + q_dim * h      # Q,K,V,O
    mlp = 3 * h * m["ffn"]                                   # gate,up,down
    per_layer = attn_proj + mlp
    lm_head = h * m["vocab"]
    weight_macs = m["n_layers"] * per_layer + lm_head
    kv_values_per_tok = 2 * m["n_kv_heads"] * hd * m["n_layers"]
    return {
        "weight_macs_per_tok": weight_macs,
        "per_layer_macs": per_layer,
        "lm_head_macs": lm_head,
        "kv_values_per_tok": kv_values_per_tok,
    }


COUNTS = model_counts()


def attn_macs_decode(ctx, m=MODEL):
    """Score + PV MACs for one decode token at context ctx (GQA)."""
    return 2 * m["n_heads"] * m["head_dim"] * ctx * m["n_layers"]


def attn_macs_prefill(prompt, m=MODEL):
    """Total attention MACs across a causal prefill of `prompt` tokens."""
    return m["n_heads"] * m["head_dim"] * prompt * prompt * m["n_layers"]


def weight_bytes(mode):
    return COUNTS["weight_macs_per_tok"] * WEIGHT_MODES[mode] / 8.0


def kv_bytes_per_tok(kv_mode):
    return COUNTS["kv_values_per_tok"] * KV_MODES[kv_mode]["bits"] / 8.0


def decode_jobs_per_token():
    """Minimum tile jobs for one decode token on today's contract:
    GEMV (M=1), K<=K_MAX rows, N<=MXE_N cols per job."""
    m, n8, kmax = MODEL, CAL["tile_array_n"], CAL["k_max"]
    h = m["hidden"]
    q_dim = m["n_heads"] * m["head_dim"]
    kv_dim = m["n_kv_heads"] * m["head_dim"]
    gemms = [(h, q_dim), (h, kv_dim), (h, kv_dim), (q_dim, h),
             (h, m["ffn"]), (h, m["ffn"]), (m["ffn"], h)]
    per_layer = sum(math.ceil(k / kmax) * math.ceil(n / n8) for k, n in gemms)
    lm = math.ceil(h / kmax) * math.ceil(m["vocab"] / n8)
    return m["n_layers"] * per_layer + lm


# ============================================================================
# 4. The model proper
# ============================================================================


def decode_time_s(ctx, kv_mode, weight_mode, mem_gbs, mem_eff,
                  array_n=SPEC_ARRAY, clock_ghz=0.8, util=0.65,
                  walker=True, n_tiles=1):
    """Seconds per decode token. Memory and compute overlap (max);
    host MMIO serializes (add)."""
    bw = mem_gbs * 1e9 * mem_eff
    bytes_tok = (weight_bytes(weight_mode)
                 + ctx * kv_bytes_per_tok(kv_mode)      # read whole KV
                 + kv_bytes_per_tok(kv_mode))           # write this token
    t_mem = bytes_tok / bw
    macs = COUNTS["weight_macs_per_tok"] + attn_macs_decode(ctx)
    t_compute = macs / (n_tiles * array_n * array_n * clock_ghz * 1e9 * util)
    if walker:
        t_host = (MODEL["n_layers"] * HOST["mmio_per_layer_walker"]
                  * HOST["mmio_round_trip_us"] * 1e-6)
    else:
        t_host = (decode_jobs_per_token() * HOST["mmio_per_job_host_seq"]
                  * HOST["mmio_round_trip_us"] * 1e-6)
    return max(t_mem, t_compute) + t_host, t_mem, t_compute, t_host


def decode_toks(ctx, kv_mode, weight_mode, mem_gbs, mem_eff, **kw):
    t, _, _, _ = decode_time_s(ctx, kv_mode, weight_mode, mem_gbs, mem_eff, **kw)
    return 1.0 / t


def prefill_time_s(prompt, weight_mode, mem_gbs, mem_eff,
                   array_n=SPEC_ARRAY, clock_ghz=0.8, util=0.65, n_tiles=1):
    """TTFT for a `prompt`-token prefill. Compute and weight-streaming
    overlap (max). Weights re-streamed once per T=128 chunk."""
    macs = prompt * COUNTS["weight_macs_per_tok"] + attn_macs_prefill(prompt)
    t_compute = macs / (n_tiles * array_n * array_n * clock_ghz * 1e9 * util)
    chunks = math.ceil(prompt / PREFILL_CHUNK_T)
    t_weights = chunks * weight_bytes(weight_mode) / (mem_gbs * 1e9 * mem_eff)
    t_host = (chunks * MODEL["n_layers"] * HOST["mmio_per_layer_walker"]
              * HOST["mmio_round_trip_us"] * 1e-6)
    return max(t_compute, t_weights) + t_host, t_compute, t_weights


def energy_per_token_j(ctx, kv_mode, weight_mode, mem_gbs, mem_eff,
                       pj_bit=None, **kw):
    """Decode energy per token = DRAM + MAC(+SRAM overhead) + static*time."""
    pj_bit = ENERGY["dram_pj_per_bit_default"] if pj_bit is None else pj_bit
    t_tok, t_mem, _, _ = decode_time_s(ctx, kv_mode, weight_mode,
                                       mem_gbs, mem_eff, **kw)
    bytes_tok = (weight_bytes(weight_mode)
                 + ctx * kv_bytes_per_tok(kv_mode) + kv_bytes_per_tok(kv_mode))
    e_dram = bytes_tok * 8 * pj_bit * 1e-12
    macs = COUNTS["weight_macs_per_tok"] + attn_macs_decode(ctx)
    e_mac = macs * ENERGY["mac_pj"] * ENERGY["sram_overhead_factor"] * 1e-12
    e_static = ENERGY["static_w"] * t_tok
    return e_dram + e_mac + e_static, t_tok


def ctx_at_floor(kv_mode, weight_mode, mem_gbs, mem_eff, floor=TOKS_FLOOR,
                 **kw):
    """Largest context (binary search) where decode still makes `floor` tok/s."""
    if decode_toks(0, kv_mode, weight_mode, mem_gbs, mem_eff, **kw) < floor:
        return 0
    lo, hi = 0, 1 << 22
    while hi - lo > 16:
        mid = (lo + hi) // 2
        if decode_toks(mid, kv_mode, weight_mode, mem_gbs, mem_eff, **kw) >= floor:
            lo = mid
        else:
            hi = mid
    return lo


def kv_weight_crossover_ctx(kv_mode, weight_mode):
    """Context where per-token KV read traffic equals weight traffic."""
    return weight_bytes(weight_mode) / kv_bytes_per_tok(kv_mode)


# ============================================================================
# 5. Calibration validation (--check)
# ============================================================================


def validate(verbose=True):
    fails = []

    def chk(name, got, exp, rel=0.05):
        ok = abs(got - exp) <= rel * abs(exp)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: model {got:.4g} "
                  f"vs anchor {exp:.4g}")
        if not ok:
            fails.append(name)

    if verbose:
        print("CALIBRATION VALIDATION (model must reproduce the measured "
              "anchors)")

    # 1. divider cycles at T=128: 33 cyc/elem * 128^2 = 540,672 (~540k)
    div = CAL["asu_div_cyc_per_elem"] * 128 * 128
    chk("ASU divider cycles @T=128 (~540k, OPTIMIZATION.md)", div, 540_000, 0.01)
    # ... and the 'divider runs ~8x longer than all six GEMMs' statement
    chk("divider/MXE cycle ratio (~8x, OPTIMIZATION.md)",
        div / CAL["mxe_all6_gemm_cyc_T128"], 8.3, 0.05)

    # 2. measured MXE utilization <-> the 3.3x-off-ideal gap
    chk("MXE util 30.6% <-> 3.3x-off-ideal (OPTIMIZATION.md)",
        1.0 / CAL["mxe_util_measured"], 3.3, 0.02)

    # 3. today-tile weight-port-bound decode roofline (INT8 weights):
    #    tok/s = f * 8 B/cyc / weight_bytes  — the §3 sanity numbers
    wb = weight_bytes("INT8")
    for mhz, exp in ANCHOR_TOKS_AT_MHZ.items():
        got = mhz * 1e6 * CAL["weight_port_bytes_per_cyc"] / wb
        chk(f"weight-port roofline @{mhz} MHz (tok/s)", got, exp, 0.05)

    # 4. derived workload counts are what the config says they should be
    chk("Qwen2.5-7B weight MACs/token (~7.07G)",
        COUNTS["weight_macs_per_tok"], 7.07e9, 0.01)
    chk("KV values/token (2*4*128*28 = 28,672)",
        COUNTS["kv_values_per_tok"], 28_672, 0.0)

    # 5. stored-record accounting matches the pinned effective-bits gate
    chk("KVQ4 D=128 stored ratio (3.5068x, effbits gate)",
        16.0 / KV_MODES["KVQ4-stored"]["bits"], 3.5068, 0.001)

    if verbose:
        print(f"  -> {'ALL ANCHORS REPRODUCED' if not fails else fails}")
    return not fails


# ============================================================================
# 6. SVG chart helper (stdlib-only line charts)
# ============================================================================

PALETTE = ["#1f6feb", "#d29922", "#3fb950", "#f85149", "#a371f7", "#8b949e"]


def svg_line_chart(path, title, xlabel, ylabel, series, hline=None,
                   w=760, h=440, y_max=None, legend_pos="tl"):
    """series = [(name, [(x, y), ...]), ...]; hline = (y, label) or None;
    legend_pos: 'tl' (top-left) or 'bl' (bottom-left)."""
    ml, mr, mt, mb = 70, 20, 46, 56
    pw, ph = w - ml - mr, h - mt - mb
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    if hline:
        ys.append(hline[0])
    x0, x1 = min(xs), max(xs)
    y0 = 0.0
    y1 = y_max if y_max is not None else max(ys) * 1.08
    if x1 == x0:
        x1 = x0 + 1

    def X(x):
        return ml + (x - x0) / (x1 - x0) * pw

    def Y(y):
        return mt + ph - (y - y0) / (y1 - y0) * ph

    def fmt(v):
        if v >= 1024 and abs(v / 1024 - round(v / 1024)) < 1e-9:
            return f"{int(round(v / 1024))}k"
        if v >= 1000:
            return f"{v / 1000:.3g}k"
        return f"{v:.3g}"

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" font-family="Menlo,monospace" '
           f'font-size="12">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="{w/2}" y="24" text-anchor="middle" font-size="14" '
           f'font-weight="bold" fill="#24292f">{title}</text>',
           f'<text x="{w/2}" y="38" text-anchor="middle" font-size="10" '
           f'fill="#d1242f">ALL VALUES PROJECTED — analytic model, '
           f'no silicon exists</text>']
    # axes + grid
    n_ticks = 6
    for i in range(n_ticks + 1):
        gx = x0 + (x1 - x0) * i / n_ticks
        gy = y0 + (y1 - y0) * i / n_ticks
        out.append(f'<line x1="{X(gx):.1f}" y1="{mt}" x2="{X(gx):.1f}" '
                   f'y2="{mt+ph}" stroke="#eaeef2"/>')
        out.append(f'<line x1="{ml}" y1="{Y(gy):.1f}" x2="{ml+pw}" '
                   f'y2="{Y(gy):.1f}" stroke="#eaeef2"/>')
        out.append(f'<text x="{X(gx):.1f}" y="{mt+ph+16}" text-anchor="middle" '
                   f'fill="#57606a">{fmt(gx)}</text>')
        out.append(f'<text x="{ml-8}" y="{Y(gy)+4:.1f}" text-anchor="end" '
                   f'fill="#57606a">{fmt(gy)}</text>')
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
               f'fill="none" stroke="#57606a"/>')
    out.append(f'<text x="{ml+pw/2}" y="{h-14}" text-anchor="middle" '
               f'fill="#24292f">{xlabel}</text>')
    out.append(f'<text x="18" y="{mt+ph/2}" text-anchor="middle" '
               f'fill="#24292f" transform="rotate(-90 18 {mt+ph/2})">'
               f'{ylabel}</text>')
    if hline:
        out.append(f'<line x1="{ml}" y1="{Y(hline[0]):.1f}" x2="{ml+pw}" '
                   f'y2="{Y(hline[0]):.1f}" stroke="#d1242f" '
                   f'stroke-dasharray="6 4"/>')
        out.append(f'<text x="{ml+pw-4}" y="{Y(hline[0])-6:.1f}" '
                   f'text-anchor="end" fill="#d1242f">{hline[1]}</text>')
    leg_y0 = (mt + 16) if legend_pos == "tl" else \
        (mt + ph - 10 - 15 * len(series))
    for i, (name, pts) in enumerate(series):
        col = PALETTE[i % len(PALETTE)]
        d = " ".join(f"{X(x):.1f},{Y(min(y, y1)):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" '
                   f'stroke-width="2"/>')
        ly = leg_y0 + 15 * i
        out.append(f'<line x1="{ml+10}" y1="{ly-4}" x2="{ml+34}" y2="{ly-4}" '
                   f'stroke="{col}" stroke-width="2"/>')
        out.append(f'<text x="{ml+40}" y="{ly}" fill="#24292f">{name}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


# ============================================================================
# 7. Report generation
# ============================================================================

CITATIONS = {
    "puget": ("Puget Systems, \"LLM Inference — Consumer GPU performance\" "
              "(llama.cpp, single stream)",
              "https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/"),
    "spheron4090": ("Spheron, \"RTX 4090 for AI/ML\" (250–320 W sustained "
                    "during LLM inference)",
                    "https://www.spheron.network/blog/rtx-4090-for-ai-ml/"),
    "hw-survey": ("Li et al., \"LLM Inference Acceleration: A Comprehensive "
                  "Hardware Perspective\" (arXiv:2410.04466; cross-platform "
                  "tok/s and tok/J incl. RTX 4090 Llama-2-7B INT4)",
                  "https://arxiv.org/html/2410.04466v3"),
    "vulkan-bench": ("llama.cpp discussion #10879, \"Performance of llama.cpp "
                     "with Vulkan\" (community multi-GPU 7B/8B Q4 table: "
                     "RTX 3090, RX 7900 XTX, RTX 4090)",
                     "https://github.com/ggml-org/llama.cpp/discussions/10879"),
    "koyeb": ("Koyeb, \"GPU LLM Performance Benchmarks\" (A100/H100 "
              "single-request latency & throughput)",
              "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
    "baseten": ("Baseten, \"Unlocking the full power of NVIDIA H100 GPUs for "
                "ML inference with TensorRT\" (Mistral-7B batch-1)",
                "https://www.baseten.co/blog/unlocking-the-full-power-of-nvidia-h100-gpus-for-ml-inference-with-tensorrt/"),
    "jetson": ("NVIDIA Developer Blog, \"Jetson Orin Nano Developer Kit Gets "
               "a Super Boost\" (Llama-3.1-8B INT4/MLC: 14.19 tok/s original "
               "15 W mode → 19.14 tok/s Super 25 W mode)",
               "https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/"),
    "jetson-lab": ("NVIDIA Jetson AI Lab benchmarks page (measured tok/s "
                   "tables incl. power modes)",
                   "https://www.jetson-ai-lab.com/archive/benchmarks.html"),
    "m4max": ("llama.cpp discussion #4167, \"Performance of llama.cpp on "
              "Apple Silicon M-series\" (M4 Max 7B Q4_0 TG = 83.06 tok/s; "
              "package power ~24–25 W is community powermetrics reporting, "
              "not an official spec)",
              "https://github.com/ggml-org/llama.cpp/discussions/4167"),
    "hailo": ("Hailo, \"Hailo-10H M.2 Generative AI Acceleration Module\" "
              "(vendor: Llama2-7B up to ~10 tok/s at <5 W module power)",
              "https://hailo.ai/products/ai-accelerators/hailo-10h-m-2-ai-acceleration-module/"),
    "hailo-eet": ("EE Times, \"LLMs in 2.5 Watts: Hailo Targets Lower Power "
                  "Market\"",
                  "https://www.eetimes.com/llms-in-2-5-watts-hailo-targets-lower-power-market/"),
    "dram-energy": ("Published DRAM system-energy anchors: DDR5 ~14–16 "
                    "pJ/bit, GDDR7 ~9–11 pJ/bit, HBM2 ~3.9 pJ/bit (e.g. "
                    "PatSnap Eureka HBM4 energy metrics survey); LPDDR5X "
                    "modeled at 4–8 pJ/bit between the HBM and DDR5 anchors",
                    "https://eureka.patsnap.com/report-hbm4-energy-metrics-bandwidth-per-watt-and-pj-bit-benchmarks"),
}


def fmt_g(x, nd=1):
    return f"{x:,.{nd}f}"


def build_report(outdir):
    os.makedirs(outdir, exist_ok=True)
    L = []                      # report lines
    add = L.append

    mem_name = "LPDDR5X x64 (68.3 GB/s)"
    mem = MEM_SYSTEMS[mem_name]
    eff = MEM_EFF_DEFAULT
    util = UTIL_SCENARIOS[UTIL_DEFAULT_KEY]

    # ---------------- header ------------------------------------------------
    add("# APEX-7B performance model — ALL NUMBERS PROJECTED")
    add("")
    add("> **THIS ENTIRE DOCUMENT IS A PROJECTION.** No APEX silicon exists; "
        "no board runs a 7B model today. This file is machine-generated by "
        "`perf/apex_perf_model.py` from (a) measured simulation / "
        "place-and-route calibration anchors, each cited to its in-repo "
        "artifact, and (b) explicitly listed assumptions. Regenerate with "
        "`python3 perf/apex_perf_model.py`. Nothing here is a benchmark of "
        "real hardware except the third-party comparator rows, which are "
        "cited to their sources.")
    add("")
    add("Verified-today baseline for contrast: the 8×8 attention tile is "
        "bit-exact in simulation and the KVQ engine is placed-and-routed on "
        "an ECP5-85F — 34.11 MHz at the shipping config after the divider "
        "pipelining (`docs/results/s10_fmax/RESULT.md`; this model's 9.04 MHz "
        "anchor is the retained pre-pipelining calibration measurement, "
        "documented in `docs/results/s10_fmax/BASELINE.md`). Everything "
        "beyond that — bigger arrays, LPDDR, W4 weights, GQA, the "
        "hardware layer-walker — is unbuilt and assumed below.")
    add("")

    # ---------------- calibration -------------------------------------------
    add("## 1. Measured calibration anchors (the only measurements here)")
    add("")
    add("| anchor | value | source |")
    add("|---|---|---|")
    add("| MMIO round-trips per host-sequenced attention step | ~300 "
        "(tile idle ~90%) | docs/OPTIMIZATION.md (measured, L3 choreography) |")
    add(f"| ASU divider emission | {CAL['asu_div_cyc_per_elem']} cyc/element "
        f"→ {CAL['asu_div_cyc_per_elem']*128*128:,} cyc @ T=128 | "
        "docs/OPTIMIZATION.md A1 |")
    add(f"| MXE, all six GEMMs of a T=128 step | ~{CAL['mxe_all6_gemm_cyc_T128']:,} "
        "cyc | docs/OPTIMIZATION.md (divider runs ~8× longer) |")
    add(f"| Measured MXE utilization | {CAL['mxe_util_measured']:.1%} "
        "(= the measured 3.3×-off-ideal gap) | docs/OPTIMIZATION.md B4 |")
    add(f"| KVQ engine ECP5-85F Fmax | {CAL['ecp5_fmax_mhz']} MHz "
        "(historical un-pipelined-divider measurement, retained as the "
        "calibration anchor; ship config now routes at 34.11 MHz) | "
        "docs/results/s10_fmax/BASELINE.md (pre-S10 baseline) |")
    add("| Tile weight port | 8 B/cycle (8-lane INT8 beat) | rtl/apex_pkg.sv "
        "MXE_N=8 |")
    add(f"| KVQ4 D=128 stored record | {CAL['kv_bits_stored_kvq4_d128']} b/v "
        f"= {16/CAL['kv_bits_stored_kvq4_d128']:.2f}× | "
        "golden/tests/test_effective_bits.py (pinned, runs in CI) |")
    add(f"| KVQ4 D=128 codec ceiling | {CAL['kv_bits_codec_kvq4_d128']} b/v "
        f"= {16/CAL['kv_bits_codec_kvq4_d128']:.2f}× | same gate |")
    add("")
    add("Validation — the model reproduces the derivable anchors "
        "(`--check` asserts these):")
    add("")
    wb8 = weight_bytes("INT8")
    add("| check | model | anchor |")
    add("|---|---|---|")
    for mhz, exp in sorted(ANCHOR_TOKS_AT_MHZ.items()):
        got = mhz * 1e6 * CAL["weight_port_bytes_per_cyc"] / wb8
        add(f"| today-tile weight-port-bound decode @ {mhz} MHz | "
            f"{got:.3f} tok/s | ~{exp} tok/s |")
    add(f"| divider/MXE cycle ratio @ T=128 | "
        f"{CAL['asu_div_cyc_per_elem']*128*128/CAL['mxe_all6_gemm_cyc_T128']:.1f}× "
        "| ~8× |")
    add("")

    # ---------------- assumptions -------------------------------------------
    add("## 2. Assumptions (everything that is NOT measured)")
    add("")
    add(f"Workload: **{MODEL['name']}** — {MODEL['n_layers']} layers, hidden "
        f"{MODEL['hidden']}, {MODEL['n_heads']} heads / "
        f"{MODEL['n_kv_heads']} KV heads (GQA), head_dim {MODEL['head_dim']}, "
        f"FFN {MODEL['ffn']}, vocab {MODEL['vocab']:,}.")
    add("")
    add(f"* Derived weight MACs/token: **{COUNTS['weight_macs_per_tok']/1e9:.2f} G** "
        f"(per-layer {COUNTS['per_layer_macs']/1e6:.1f} M × {MODEL['n_layers']} "
        f"+ LM head {COUNTS['lm_head_macs']/1e6:.0f} M). Attention adds "
        f"2·heads·head_dim·ctx·layers ≈ {attn_macs_decode(32768)/1e9:.2f} G at "
        "32k ctx.")
    add(f"* KV cache: **{COUNTS['kv_values_per_tok']:,} values/token** "
        f"(GQA), i.e. {kv_bytes_per_tok('FP16')/1024:.0f} KiB/token at FP16, "
        f"{kv_bytes_per_tok('KVQ4-stored')/1024:.1f} KiB/token as KVQ4 stored "
        "records.")
    add("* **UNBUILT features assumed by the projections** (each is a task "
        "on the backlog): GQA plumbing (S6), B1 hardware layer-walker "
        "(≤3 MMIO/layer-step; without it decode is transaction-bound), "
        "B3 native-W4 weight path (4.125 b/w incl. scales at the G=128 "
        "amortization; the B3 stage-6 adoption geometry is G=32 = 4.5 b/w — "
        "sensitivity in §3b½), A1 radix-4 "
        "divider + B4 ingest-under-compute (the 65% utilization scenario), "
        "wide-D RMSNorm for hidden=3584, T>128 chunk merge.")
    add(f"* Memory: peak GB/s = MT/s × bus bytes; sustained efficiency swept "
        f"{MEM_EFF_RANGE} (default {MEM_EFF_DEFAULT:.0%}) — single-stream "
        "mixed read traffic on LPDDR does not reach peak.")
    add(f"* Host: MMIO round-trip {HOST['mmio_round_trip_us']} µs; "
        f"host-sequenced mode {HOST['mmio_per_job_host_seq']:.0f} MMIO/job "
        "(calibrated: ~300 measured MMIO per ~60-job attention step); walker "
        f"mode {HOST['mmio_per_layer_walker']:.0f} MMIO/layer-step.")
    add(f"* Energy: DRAM {ENERGY['dram_pj_per_bit_sweep']} pJ/bit swept "
        f"(default {ENERGY['dram_pj_per_bit_default']}), see citation "
        "[dram-energy]; INT8 MAC "
        f"{ENERGY['mac_pj']} pJ × {ENERGY['sram_overhead_factor']}× on-chip "
        f"movement overhead; static floor {ENERGY['static_w']} W. DRAM "
        "energy dominates (60–75% of total) — it is both the biggest lever "
        "and the biggest uncertainty in every energy number below.")
    add(f"* Prefill: T={PREFILL_CHUNK_T} chunks; weights re-streamed once "
        "per chunk (no on-die weight residency); utilization scenarios: "
        + "; ".join(f"{k} = {v:.0%}" for k, v in UTIL_SCENARIOS.items())
        + f". Default scenario: **{UTIL_DEFAULT_KEY}**.")
    add("* KV traffic uses the **stored-record accounting** (what the RTL "
        "writes, tag+pad+scale-bank included) per the honest-accounting gate; "
        "the codec ceiling is plotted as a separate labeled curve, never as "
        "the headline.")
    add("")

    # ---------------- decode ------------------------------------------------
    add("## 3. PROJECTED decode throughput")
    add("")
    add(f"Reference configuration (the S13 candidate): **{SPEC_ARRAY}×"
        f"{SPEC_ARRAY} INT8 @ 0.5–0.8 GHz, {mem_name}, eff "
        f"{MEM_EFF_DEFAULT:.0%}, W4 weights, KVQ4 stored records, walker**. "
        "Decode is memory-bound: the array and clock barely matter; the "
        "memory system and bytes/token decide everything.")
    add("")
    add("### 3a. tok/s vs memory system and weight precision (ctx = 2k)")
    add("")
    add("| memory system | eff | W4 + KVQ4 | W4 + FP16 KV | INT8 + KVQ4 | "
        "INT8 + FP16 KV |")
    add("|---|---|---|---|---|---|")
    bw_rows = []
    for mname, mgbs in MEM_SYSTEMS.items():
        for e in MEM_EFF_RANGE:
            cells = []
            for wmode, kvmode in (("W4", "KVQ4-stored"), ("W4", "FP16"),
                                  ("INT8", "KVQ4-stored"), ("INT8", "FP16")):
                cells.append(decode_toks(2048, kvmode, wmode, mgbs, e))
            bw_rows.append((mname, e, *cells))
            if e == MEM_EFF_DEFAULT:
                add(f"| {mname} | {e:.0%} | **{cells[0]:.1f}** | {cells[1]:.1f} "
                    f"| {cells[2]:.1f} | {cells[3]:.1f} |")
    add("")
    x32 = decode_toks(2048, "KVQ4-stored", "W4",
                      MEM_SYSTEMS["LPDDR5X x32 (34.1 GB/s)"], MEM_EFF_DEFAULT)
    add(f"* A single ×32 LPDDR5X channel projects to **{x32:.1f} tok/s** — "
        f"below the {TOKS_FLOOR:.0f} tok/s floor. The spec must say ×64.")
    add(f"* INT8 weights + ×64 projects to "
        f"{decode_toks(2048, 'KVQ4-stored', 'INT8', mem, eff):.1f} tok/s — "
        "also below floor. **W4 (B3) is load-bearing for the decode floor, "
        "and it is unbuilt.**")
    add("")

    add("### 3b. tok/s vs context — the KVQ flat-region claim")
    add("")
    hdr = "| context | " + " | ".join(KV_MODES[k]["label"] for k in KV_MODES) + " |"
    add(hdr)
    add("|" + "---|" * (len(KV_MODES) + 1))
    ctx_csv = [("ctx",) + tuple(KV_MODES.keys())]
    for c in CTX_SWEEP:
        row = [decode_toks(c, k, "W4", mem, eff) for k in KV_MODES]
        ctx_csv.append((c, *[f"{v:.3f}" for v in row]))
        add(f"| {c:,} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    add("")
    floors = {k: ctx_at_floor(k, "W4", mem, eff) for k in KV_MODES}
    add(f"**≥{TOKS_FLOOR:.0f} tok/s flat region** (largest context holding "
        "the floor, reference config):")
    add("")
    add("| KV mode | flat region extends to | vs FP16 |")
    add("|---|---|---|")
    for k, v in floors.items():
        add(f"| {KV_MODES[k]['label']} | ~{v/1024:.0f}k ctx | "
            f"{v/max(floors['FP16'],1):.2f}× |")
    add("")
    add(f"KVQ stretches the ≥{TOKS_FLOOR:.0f} tok/s region from "
        f"**~{floors['FP16']/1024:.0f}k to ~{floors['KVQ4-stored']/1024:.0f}k "
        f"context** as stored today "
        f"(~{floors['KVQ4-codec']/1024:.0f}k at the codec ceiling). At 128k "
        f"ctx: {decode_toks(131072, 'KVQ4-stored', 'W4', mem, eff):.1f} vs "
        f"{decode_toks(131072, 'FP16', 'W4', mem, eff):.1f} tok/s — KVQ keeps "
        "long-context decode ~2× faster, PROJECTED.")
    add("")
    add("KV-read traffic crosses weight traffic at "
        f"**~{kv_weight_crossover_ctx('FP16','W4')/1024:.0f}k ctx "
        "uncompressed** vs "
        f"**~{kv_weight_crossover_ctx('KVQ4-stored','W4')/1024:.0f}k "
        "compressed (stored)** — with KVQ4 the KV stream never dominates "
        "inside a 128k window.")
    add("")
    add("![decode tok/s vs context](fig_decode_vs_ctx.svg)")
    add("")

    add("### 3b½. W4 grouping sensitivity — the B3 stage-6 adoption geometry")
    add("")
    add("The B3 stage-6 token-quality gate "
        "(`docs/results/b3_w4_adoption/RESULTS.md`, 2026-07-21) rejected "
        "tile-wide W4 scales (collapses the 7B model) and measured **G=32 "
        "K-grouping** as the quality-viable config — fp16 scale per 32 "
        "weights = **4.5 b/w** vs the 4.125 b/w (G=128 amortization) used "
        "elsewhere in this document. Decode is weight-BW-bound, so the "
        "shipped-W4 numbers shift down ~9%:")
    add("")
    add("| ctx | W4 4.125 b/w (G=128) | W4-G32 4.5 b/w (adoption) |")
    add("|---|---|---|")
    for c in (2048, 32768, 65536, 131072):
        add(f"| {c:,} | {decode_toks(c, 'KVQ4-stored', 'W4', mem, eff):.1f} "
            f"| {decode_toks(c, 'KVQ4-stored', 'W4-G32', mem, eff):.1f} |")
    add("")
    fl_g128 = ctx_at_floor('KVQ4-stored', 'W4', mem, eff)
    fl_g32 = ctx_at_floor('KVQ4-stored', 'W4-G32', mem, eff)
    add(f"The ≥{TOKS_FLOOR:.0f} tok/s flat region contracts from "
        f"~{fl_g128/1024:.0f}k to **~{fl_g32/1024:.0f}k context** at the "
        "adoption geometry (reference config, KVQ4 stored records). The "
        "decode floor still clears at 32k "
        f"({decode_toks(32768, 'KVQ4-stored', 'W4-G32', mem, eff):.1f} "
        "tok/s). Quote W4-dependent decode numbers from THIS section once "
        "B3 ships at G=32.")
    add("")

    add("### 3c. Host sequencing: why the walker (B1) is mandatory")
    add("")
    jobs = decode_jobs_per_token()
    t_host_seq = jobs * HOST["mmio_per_job_host_seq"] * \
        HOST["mmio_round_trip_us"] * 1e-6
    walker_t = decode_time_s(2048, "KVQ4-stored", "W4", mem, eff)[0]
    hostseq_t = decode_time_s(2048, "KVQ4-stored", "W4", mem, eff,
                              walker=False)[0]
    add(f"* Today's contract (GEMV M=1, K≤{CAL['k_max']}, N≤"
        f"{CAL['tile_array_n']}) needs **{jobs:,} tile jobs per decode "
        f"token** at 7B. Host-sequenced at "
        f"{HOST['mmio_per_job_host_seq']:.0f} MMIO/job × "
        f"{HOST['mmio_round_trip_us']} µs that is {t_host_seq:.1f} s/token of "
        f"pure transaction time → **{1/hostseq_t:.2f} tok/s** no matter how "
        "fast the memory is.")
    add(f"* With the B1 walker (≤{HOST['mmio_per_layer_walker']:.0f} "
        f"MMIO/layer-step): host overhead "
        f"{MODEL['n_layers']*HOST['mmio_per_layer_walker']*HOST['mmio_round_trip_us']:.0f} µs/token "
        f"(~{100*(MODEL['n_layers']*HOST['mmio_per_layer_walker']*HOST['mmio_round_trip_us']*1e-6)/walker_t:.1f}% "
        f"of the token) → {1/walker_t:.1f} tok/s.")
    add("")

    # ---------------- prefill ----------------------------------------------
    add("## 4. PROJECTED prefill / TTFT")
    add("")
    add(f"TTFT for a {TTFT_PROMPT:,}-token prompt (W4 weights, {mem_name}, "
        f"eff {MEM_EFF_DEFAULT:.0%}). Prefill is compute-bound: the array "
        "and its utilization decide TTFT; memory decides decode.")
    add("")
    add("| array | clock | " + " | ".join(UTIL_SCENARIOS) + " |")
    add("|---|---|" + "---|" * len(UTIL_SCENARIOS))
    ttft_csv = [("array_n", "clock_ghz")
                + tuple(f"ttft_s_util_{v:.3f}" for v in UTIL_SCENARIOS.values())]
    for n in ARRAY_SIZES:
        for f_ghz in CLOCKS_GHZ:
            if n < 32 and f_ghz not in (0.5, 1.0):
                continue
            row = [prefill_time_s(TTFT_PROMPT, "W4", mem, eff, array_n=n,
                                  clock_ghz=f_ghz, util=u)[0]
                   for u in UTIL_SCENARIOS.values()]
            ttft_csv.append((n, f_ghz, *[f"{t:.3f}" for t in row]))
            mark = " ←spec" if (n == SPEC_ARRAY and f_ghz in SPEC_CLOCKS) else ""
            add(f"| {n}×{n} | {f_ghz} GHz | "
                + " | ".join(f"{t:,.1f} s" for t in row) + f"{mark} |")
    add("")
    t_lo = prefill_time_s(TTFT_PROMPT, "W4", mem, eff, array_n=SPEC_ARRAY,
                          clock_ghz=SPEC_CLOCKS[0], util=util)[0]
    t_hi = prefill_time_s(TTFT_PROMPT, "W4", mem, eff, array_n=SPEC_ARRAY,
                          clock_ghz=SPEC_CLOCKS[1], util=util)[0]
    add(f"Reference config TTFT(1k): **{t_hi:.1f}–{t_lo:.1f} s** at "
        f"{SPEC_CLOCKS[1]}/{SPEC_CLOCKS[0]} GHz under the "
        f"{UTIL_DEFAULT_KEY} scenario. At today's *measured* 30.6% "
        "utilization the same array would need "
        f"{prefill_time_s(TTFT_PROMPT,'W4',mem,eff,array_n=SPEC_ARRAY,clock_ghz=0.8,util=CAL['mxe_util_measured'])[0]:.1f} s "
        "— A1 (radix-4 divider) and B4 (ingest-under-compute) are "
        "load-bearing for the TTFT floor.")
    add("")
    add(f"A 32×32 @ 1 GHz array projects to "
        f"{prefill_time_s(TTFT_PROMPT,'W4',mem,eff,array_n=32,clock_ghz=1.0,util=util)[0]:.1f} s "
        "TTFT at target utilization — marginal against a 5 s floor; "
        "the 64×64 array is sized by prefill, not decode.")
    add("")

    # ---------------- energy -----------------------------------------------
    add("## 5. PROJECTED energy per token vs context (KVQ on/off)")
    add("")
    add(f"Reference config; DRAM at {ENERGY['dram_pj_per_bit_default']} "
        "pJ/bit (band = 4–8 pJ/bit). J/token = DRAM traffic + MAC energy + "
        f"{ENERGY['static_w']} W static × token time.")
    add("")
    add("| context | J/tok KVQ4 (4/6/8 pJ/b) | J/tok FP16 KV (6 pJ/b) | "
        "power @ KVQ4 6 pJ/b | tok/s KVQ4 |")
    add("|---|---|---|---|---|")
    e_csv = [("ctx", "j_kvq4_4pj", "j_kvq4_6pj", "j_kvq4_8pj", "j_fp16_6pj",
              "w_kvq4_6pj", "toks_kvq4")]
    for c in CTX_SWEEP:
        js = [energy_per_token_j(c, "KVQ4-stored", "W4", mem, eff, pj_bit=p)[0]
              for p in ENERGY["dram_pj_per_bit_sweep"]]
        jf, tf = energy_per_token_j(c, "FP16", "W4", mem, eff)
        t_tok = decode_time_s(c, "KVQ4-stored", "W4", mem, eff)[0]
        watts = js[1] / t_tok
        e_csv.append((c, *[f"{j:.4f}" for j in js], f"{jf:.4f}",
                      f"{watts:.3f}", f"{1/t_tok:.3f}"))
        add(f"| {c:,} | {js[0]:.2f} / {js[1]:.2f} / {js[2]:.2f} | {jf:.2f} | "
            f"{watts:.1f} W | {1/t_tok:.1f} |")
    add("")
    t0 = decode_time_s(2048, "KVQ4-stored", "W4", mem, eff)[0]
    j0lo = energy_per_token_j(2048, "KVQ4-stored", "W4", mem, eff, pj_bit=4)[0]
    j0hi = energy_per_token_j(2048, "KVQ4-stored", "W4", mem, eff, pj_bit=8)[0]
    add(f"Short-context headline, PROJECTED: **~{j0lo:.2f}–{j0hi:.2f} "
        f"J/token at ~{j0lo/t0:.1f}–{j0hi/t0:.1f} W** (die+DRAM; excludes "
        "host SoC). The DRAM pJ/bit assumption alone moves this ±35%.")
    add("")
    add("![J/token vs context](fig_j_per_tok_vs_ctx.svg)")
    add("")

    # ---------------- comparator table --------------------------------------
    add("## 6. Comparator table — published single-stream 7B/8B Q4 decode")
    add("")
    add("Rules of this table: single-stream, single-user decode only; "
        "wall/board/module power as cited; quantization as cited. Batched "
        "GPU serving amortizes to far lower J/token (0.05–0.4) and is a "
        "different job — do not compare this table against batched or "
        "speculative-decoding numbers. The APEX row is a PROJECTION from "
        "this model; every other row is a third-party measurement or vendor "
        "claim, cited.")
    add("")
    add("| device | tok/s | power | J/token | basis |")
    add("|---|---|---|---|---|")
    add("| RTX 4090 (llama.cpp CUDA, 7B/8B Q4) | ~100–180 | ~250–320 W "
        "sustained | **~1.4–3.2** | measured [puget][spheron4090][hw-survey] |")
    add("| RTX 3090 (llama.cpp, 7B/8B Q4) | ~100–130 | ~275–350 W | "
        "**~2.1–3.5** | measured [puget][vulkan-bench] |")
    add("| RX 7900 XTX (llama.cpp, 7B/8B Q4) | ~95–130 | ~300–355 W | "
        "**~2.3–3.7** | measured [vulkan-bench] |")
    add("| A100 / H100 (batch-1, 7B) | ~120–145 | ~150–300 W | **~1.0–2.5** "
        "| measured [koyeb][baseten] |")
    add("| Jetson Orin Nano Super (Llama-3.1-8B INT4) | 14.2 @ 15 W mode / "
        "19.1 @ 25 W mode | 15 / 25 W (mode budget) | **~1.0–1.3** | "
        "measured [jetson][jetson-lab] |")
    add("| Apple M4 Max (llama.cpp Metal, 7B Q4_0) | 83.1 | ~24–25 W package "
        "(community powermetrics) | **~0.30** | measured [m4max] |")
    add("| Hailo-10H M.2 (Llama2-7B) | ~10 | <5 W module | **~0.45–0.5** | "
        "vendor claim [hailo][hailo-eet] |")
    add(f"| **APEX-7B (this model)** | {decode_toks(2048,'KVQ4-stored','W4',mem,MEM_EFF_RANGE[0]):.0f}–"
        f"{decode_toks(2048,'KVQ4-stored','W4',mem,MEM_EFF_RANGE[2]):.0f} "
        f"| ~{j0lo/t0:.1f}–{j0hi/t0:.1f} W (die+DRAM, modeled) | "
        f"**~{j0lo:.2f}–{j0hi:.2f}** | **PROJECTED — no hardware** |")
    add("")
    add("What the table does and does not support, stated plainly:")
    add("* Desktop discrete GPUs are ~8–14× *faster* than the APEX "
        "projection at batch-1. The projected win is **energy per token** "
        "(~5–10× vs stock desktop GPUs), never speed.")
    add("* Hailo-10H and Apple M4 Max already sit at 0.3–0.5 J/token — "
        "APEX does **not** claim first or only sub-0.5 J/token. The "
        "differentiating cell is J/token **held flat to long context** via "
        "in-datapath KV compression (§3b/§5), which none of the cited edge "
        "devices document.")
    add("* A power-capped 4090 or a TensorRT-LLM/speculative stack "
        "compresses the gap to ~2–6×.")
    add("")
    add("Citations:")
    add("")
    for key, (desc, url) in CITATIONS.items():
        add(f"* `[{key}]` {desc} — <{url}> (accessed 2026-07-14)")
    add("")

    # ---------------- multi-tile --------------------------------------------
    add("## 7. PROJECTED multi-tile scaling")
    add("")
    add("Tiles share one memory system (multi-tile NoC is explicitly out of "
        "current scope; this is the paper exercise). Decode is "
        "bandwidth-bound, so extra tiles buy ~nothing; prefill is "
        "compute-bound and scales until the weight stream saturates the "
        "memory. That asymmetry is the honest multi-tile story.")
    add("")
    add(f"| tiles ({SPEC_ARRAY}×{SPEC_ARRAY} @ 0.8 GHz each) | TTFT(1k) | "
        "prefill tok/s | prefill bound by | decode tok/s (2k ctx) |")
    add("|---|---|---|---|---|")
    mt_csv = [("tiles", "ttft_s", "prefill_toks", "decode_toks")]
    for nt in TILES_SWEEP:
        tt, tc, tw = prefill_time_s(TTFT_PROMPT, "W4", mem, eff,
                                    array_n=SPEC_ARRAY, clock_ghz=0.8,
                                    util=util, n_tiles=nt)
        dec = decode_toks(2048, "KVQ4-stored", "W4", mem, eff, n_tiles=nt)
        pf = TTFT_PROMPT / tt
        mt_csv.append((nt, f"{tt:.3f}", f"{pf:.1f}", f"{dec:.3f}"))
        bound = "compute" if tc >= tw else "**memory (BW ceiling)**"
        add(f"| {nt} | {tt:,.2f} s | {pf:,.0f} | {bound} | {dec:.1f} |")
    add("")
    add("![multi-tile scaling](fig_multitile.svg)")
    add("")

    # ---------------- floors ------------------------------------------------
    add("## 8. Spec floors check (S13: decode ≥10 tok/s to ≥32k ctx, "
        "TTFT(1k) ≤5 s, ≤5 W)")
    add("")
    add("| config | decode @32k | TTFT(1k) | power @32k | verdict |")
    add("|---|---|---|---|---|")
    for (n, f_ghz, mname) in [(64, 0.8, "LPDDR5X x64 (68.3 GB/s)"),
                              (64, 0.5, "LPDDR5X x64 (68.3 GB/s)"),
                              (32, 1.0, "LPDDR5X x64 (68.3 GB/s)"),
                              (64, 0.8, "LPDDR5X x32 (34.1 GB/s)")]:
        mg = MEM_SYSTEMS[mname]
        d32 = decode_toks(32768, "KVQ4-stored", "W4", mg, eff, array_n=n,
                          clock_ghz=f_ghz, util=util)
        tt = prefill_time_s(TTFT_PROMPT, "W4", mg, eff, array_n=n,
                            clock_ghz=f_ghz, util=util)[0]
        j32, ttok = energy_per_token_j(32768, "KVQ4-stored", "W4", mg, eff,
                                       array_n=n, clock_ghz=f_ghz, util=util)
        w32 = j32 / ttok
        ok = d32 >= TOKS_FLOOR and tt <= 5.0 and w32 <= 5.0
        add(f"| {n}×{n} @ {f_ghz} GHz, {mname} | {d32:.1f} tok/s | {tt:.1f} s "
            f"| {w32:.1f} W | {'**PASS**' if ok else 'FAIL'} |")
    add("")
    tt_05_90 = prefill_time_s(TTFT_PROMPT, "W4", mem, eff, array_n=64,
                              clock_ghz=0.5, util=0.90)[0]
    add(f"Utilization is the swing variable at the low-clock end: 64×64 @ "
        f"0.5 GHz misses the 5 s TTFT floor at 65% utilization but passes at "
        f"90% ({tt_05_90:.1f} s). Under this model's assumptions the robust "
        f"spec point is **64×64 @ ≥0.65 GHz with LPDDR5X ×64** — a single "
        "×32 channel fails decode outright, and a 32×32 array fails TTFT at "
        "any plausible clock.")
    add("")

    # ---------------- limitations -------------------------------------------
    add("## 9. Known limitations of this model")
    add("")
    add("* No DRAM controller, refresh, bank-conflict, or page-policy "
        "modeling — a flat efficiency factor stands in for all of it.")
    add("* Activation traffic between layers is assumed on-die; logits "
        "readout and sampling are ignored (<1% of traffic).")
    add("* The 65% utilization scenario assumes A1+B4 land as designed; "
        "the only *measured* utilization is 30.6%.")
    add("* The ASU divider is not modeled separately at ASIC scale. As "
        f"measured it emits {CAL['asu_div_cyc_per_elem']} cyc/element on one "
        "lane, which would bind prefill even at 64×64; the 65% scenario "
        "assumes the A1 radix-4 divider plus widened emission hide it under "
        "the MXE. That is a design assumption, not a measurement.")
    add("* Energy has no per-block breakdown from synthesis — pJ/MAC and "
        "pJ/bit are literature anchors, not extracted from our netlist.")
    add("* Comparator rows were not re-measured by us; a wall-metered "
        "single-stream 4090 baseline with a committed re-run script is an "
        "open follow-up.")
    add("")
    add("---")
    add("*Generated by `perf/apex_perf_model.py` — regenerate, never "
        "hand-edit. ALL NUMBERS PROJECTED.*")

    # ---------------- write files -------------------------------------------
    with open(os.path.join(outdir, "PERF_MODEL.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    def wcsv(name, rows):
        with open(os.path.join(outdir, name), "w", newline="") as f:
            csv.writer(f).writerows(rows)

    wcsv("decode_tok_s_vs_ctx.csv", ctx_csv)
    wcsv("bw_sweep.csv", [("mem", "eff", "w4_kvq4", "w4_fp16", "int8_kvq4",
                           "int8_fp16")] + bw_rows)
    wcsv("ttft_grid.csv", ttft_csv)
    wcsv("energy_vs_ctx.csv", e_csv)
    wcsv("multi_tile.csv", mt_csv)

    # ---------------- charts -------------------------------------------------
    ctxs = [c for c in range(0, 131073, 2048)]
    svg_line_chart(
        os.path.join(outdir, "fig_decode_vs_ctx.svg"),
        "PROJECTED decode tok/s vs context — 64×64, LPDDR5X ×64 @70%, W4",
        "context (tokens)", "tokens/s",
        [(KV_MODES[k]["label"],
          [(c, decode_toks(c, k, "W4", mem, eff)) for c in ctxs])
         for k in ("FP16", "INT8-KV", "KVQ8-stored", "KVQ4-stored",
                   "KVQ4-codec")],
        hline=(TOKS_FLOOR, "10 tok/s floor"), y_max=15, legend_pos="bl")
    svg_line_chart(
        os.path.join(outdir, "fig_j_per_tok_vs_ctx.svg"),
        "PROJECTED J/token vs context — 6 pJ/bit DRAM (band 4–8)",
        "context (tokens)", "J/token",
        [("KVQ4 stored", [(c, energy_per_token_j(c, "KVQ4-stored", "W4",
                                                 mem, eff)[0]) for c in ctxs]),
         ("FP16 KV", [(c, energy_per_token_j(c, "FP16", "W4",
                                             mem, eff)[0]) for c in ctxs]),
         ("KVQ4 stored, 4 pJ/b", [(c, energy_per_token_j(
             c, "KVQ4-stored", "W4", mem, eff, pj_bit=4)[0]) for c in ctxs]),
         ("KVQ4 stored, 8 pJ/b", [(c, energy_per_token_j(
             c, "KVQ4-stored", "W4", mem, eff, pj_bit=8)[0]) for c in ctxs])])
    svg_line_chart(
        os.path.join(outdir, "fig_multitile.svg"),
        "PROJECTED multi-tile scaling — prefill scales, decode does not",
        "tiles (64×64 @ 0.8 GHz)", "relative to 1 tile",
        [("prefill speedup",
          [(nt, prefill_time_s(TTFT_PROMPT, "W4", mem, eff, util=util,
                               clock_ghz=0.8)[0]
            / prefill_time_s(TTFT_PROMPT, "W4", mem, eff, util=util,
                             clock_ghz=0.8, n_tiles=nt)[0])
           for nt in TILES_SWEEP]),
         ("decode speedup",
          [(nt, decode_toks(2048, "KVQ4-stored", "W4", mem, eff, n_tiles=nt)
            / decode_toks(2048, "KVQ4-stored", "W4", mem, eff))
           for nt in TILES_SWEEP])])

    return L


# ============================================================================
# main
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate calibration anchors and exit")
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "docs", "results", "perf_model"))
    args = ap.parse_args()

    ok = validate(verbose=True)
    if not ok:
        print("CALIBRATION VALIDATION FAILED — refusing to generate report")
        return 1
    if args.check:
        return 0

    outdir = os.path.abspath(args.outdir)
    build_report(outdir)
    print(f"\nreport + CSVs + SVGs written to {outdir}")
    print("REMINDER: every number in the report is PROJECTED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
