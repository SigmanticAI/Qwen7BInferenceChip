// sim_main.cpp — F2 stage-2 regops executor (simulation side).
// Drives the verilated cl_apex's OCL AXI-Lite pins with the register-op
// stream compiled by scripts/fpga/f2/trace_to_regops.py — the SAME file the
// fpga_pci host runner executes on the F2 instance. One artifact, two
// executors; this one runs free and catches everything first.
//
// Usage: sim [+tile_div=N] [+cap_out=<path>] <regops.jsonl> [more.jsonl ...]
// Exit 0 = every check passed in every file.
//
// Op vocabulary (MUST stay bit-for-bit identical to f2_host_run.py; the
// authoritative table is trace_to_regops.py's header):
//   note | w a d | r a m e | rn a m | cap a m tag [e] | poll a m e | pw | jf
//
// CAPTURE EGRESS (+cap_out=<path>; host: --cap-out <path>). One JSON line per
// EXECUTED cap, file truncated at run start:
//   {"tag":"<str>","addr":"0x%08x","mask":"0x%08x","value":<int>,"i":<seq>}
// `i` is a global 0-based counter over the whole run (all files, argv order).
// The end-of-run line `F2SIM CAPTURES: n=<N> out=<path>` prints whenever the
// plusarg was given, n=0 included. The two executors' egress files are meant
// to be byte-identical — diffing them IS the parity check.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <memory>
#include "Vcl_apex.h"
#include "verilated.h"
#if VM_TRACE
// RUNG-3 wedge forensics: FST when the twin is verilated with --trace-fst
// (VM_TRACE_FST), else the legacy VCD. The dump can be WINDOWED with
// +trace_from_cyc=N / +trace_to_cyc=N (host-clock cycles, same counter the
// CYCMARK lines print) so a wedge run's multi-M-cycle poll spin does not
// drown the seam of interest.
#if VM_TRACE_FST
#include "verilated_fst_c.h"
static std::unique_ptr<VerilatedFstC> tfp;
#else
#include "verilated_vcd_c.h"
static std::unique_ptr<VerilatedVcdC> tfp;
#endif
static uint64_t trace_from_cyc = 0;
static uint64_t trace_to_cyc = ~0ULL;
#endif

static std::unique_ptr<Vcl_apex> top;
static uint64_t vtime = 0;
static uint64_t cyc = 0;
// cycles per poll op. Default unchanged (2M); overridable via +poll_limit=N
// because an E-6 walk's HOST-SILENT window puts the ENTIRE fuel-fed
// projection (112 jobs x ~21.7k cyc of DDR streaming at tile_div=5) under
// ONE done-poll — E-5's windows never exposed a single poll this long (its
// RO drains reset the budget per job). A raised limit is a per-run,
// disclosed choice; every legacy gate keeps the 2M stall detection.
static uint64_t POLL_LIMIT = 2'000'000;
static const uint64_t AXI_LIMIT  = 100'000;     // cycles per AXI handshake

#ifdef APEX_INGEST_MON
// ═══ perf/ingest-lane: IB-FUEL ingest monitor (+ingest_mon) ═══════════════
// Harness-side observation of the DDR->tile fuel line — ZERO RTL changes.
// Requires a build with `--public-flat-rw -CFLAGS -DAPEX_INGEST_MON` in
// VFLAGS_EXTRA (signals read through Vcl_apex___024root). Counters are
// CUMULATIVE; windows are snapshot deltas (verif/f2sim/ingest_report.py).
// Sampling contract: entry samples are taken at hi() entry — the settled
// low-phase values the imminent posedge consumes; tile-domain handshakes
// count only on shell cycles whose posedge carries a clk_tile rising edge
// (post-eval clk_tile high, entry clk_tile low). Every number this monitor
// prints is SIM-measured on the behavioral twin; none is a silicon claim.
#include "Vcl_apex___024root.h"
struct IngestMon {
    bool armed = false;                    // +ingest_mon
    // entry (pre-edge) samples
    bool s_wv=0, s_wr=0, s_arv=0, s_arr=0, s_rv=0, s_rr=0, s_rlast=0;
    bool s_frv=0, s_frr=0, s_busy=0, s_ack=0, s_ctile=0;
    bool s_req=0, s_wfv=0, s_wfr=0;
    uint8_t s_arlen=0;
    // edge trackers
    bool prev_ack=0, have_prev_ack=0;
    bool prev_req=0, have_prev_req=0;
    // cumulative counters
    uint64_t busy=0, arhs=0, arwords=0, rbeats=0, pushes=0, stallfull=0,
             pops=0, tedges=0, recs=0;
    // occupancy (write-domain view: committed pushes - committed pops)
    int64_t  occ=0; uint64_t occmax=0, occsum=0, hist[10]={0};
    // gaps: record framing (ack -> next AR) and inter-burst (rlast -> AR)
    bool pend_rec=0, pend_burst=0;
    uint64_t rec_t0=0, burst_t0=0;
    uint64_t recgapsum=0, recgapmax=0, recgapn=0;
    uint64_t burstgapsum=0, burstgapn=0;
    // framing split: ack -> req_tgl flip (ctl+CDC half of the gap), and
    // walker wf record handshakes (skid accept-ahead visibility)
    bool pend_req=0;
    uint64_t reqtglsum=0, reqtgln=0, wfhs=0;
};
static IngestMon mon;

static inline void mon_entry_sample() {
    if (!mon.armed) return;
    auto* r = top->rootp;
    mon.s_wv    = r->cl_apex__DOT__fifo_wvalid;
    mon.s_wr    = r->cl_apex__DOT__fifo_wready;
    mon.s_arv   = r->cl_apex__DOT__rdr_arvalid;
    mon.s_arr   = r->cl_apex__DOT__rdr_arready;
    mon.s_arlen = r->cl_apex__DOT__rdr_arlen;
    mon.s_rv    = r->cl_apex__DOT__rdr_rvalid;
    mon.s_rr    = r->cl_apex__DOT__rdr_rready;
    mon.s_rlast = r->cl_apex__DOT__ddr_rlast;
    mon.s_frv   = r->cl_apex__DOT__fifo_rvalid;
    mon.s_frr   = r->cl_apex__DOT__fifo_rready;
    mon.s_busy  = r->cl_apex__DOT__fuel_rdr_busy;
    mon.s_ack   = r->cl_apex__DOT__fuel_ack_tgl;
    mon.s_ctile = r->cl_apex__DOT__clk_tile;
    mon.s_req   = r->cl_apex__DOT__fuel_req_tgl;
    mon.s_wfv   = r->cl_apex__DOT__t_wf_valid;
    mon.s_wfr   = r->cl_apex__DOT__t_wf_ready;
}

static inline void mon_after_edge() {
    if (!mon.armed) return;
    auto* r = top->rootp;
    if (mon.s_busy) ++mon.busy;
    // AR handshake: burst issued
    if (mon.s_arv && mon.s_arr) {
        ++mon.arhs;
        mon.arwords += (uint64_t)mon.s_arlen + 1;
        if (mon.pend_rec) {
            uint64_t g = cyc - mon.rec_t0;
            mon.recgapsum += g;
            if (g > mon.recgapmax) mon.recgapmax = g;
            ++mon.recgapn;
            mon.pend_rec = 0;
            mon.pend_burst = 0;      // record gap consumed this AR
        } else if (mon.pend_burst) {
            mon.burstgapsum += cyc - mon.burst_t0;
            ++mon.burstgapn;
            mon.pend_burst = 0;
        }
    }
    // R beat retired at the reader
    if (mon.s_rv && mon.s_rr) {
        ++mon.rbeats;
        if (mon.s_rlast) { mon.pend_burst = 1; mon.burst_t0 = cyc; }
    }
    // afifo write side (shell domain)
    if (mon.s_wv) {
        if (mon.s_wr) { ++mon.pushes; ++mon.occ; }
        else            ++mon.stallfull;
    }
    // afifo read side commits only on a clk_tile posedge
    bool ctile_now = r->cl_apex__DOT__clk_tile;
    if (ctile_now && !mon.s_ctile) {
        ++mon.tedges;
        if (mon.s_frv && mon.s_frr) { ++mon.pops; --mon.occ; }
    }
    // walker record handshake into the fuel ctl skid (tile domain)
    bool ctile_rise = ctile_now && !mon.s_ctile;
    if (ctile_rise && mon.s_wfv && mon.s_wfr) ++mon.wfhs;
    // ctl half of the framing gap: ack -> next req toggle (tile domain reg,
    // sampled at shell rate; a toggle is a level change)
    if (mon.have_prev_req && (mon.s_req != mon.prev_req) && mon.pend_req) {
        mon.reqtglsum += cyc - mon.rec_t0;
        ++mon.reqtgln;
        mon.pend_req = 0;
    }
    mon.prev_req = mon.s_req; mon.have_prev_req = 1;
    // record completion: reader's ack toggle flipped (shell domain)
    if (mon.have_prev_ack && (mon.s_ack != mon.prev_ack)) {
        ++mon.recs;
        mon.pend_rec = 1;
        mon.pend_req = 1;
        mon.rec_t0 = cyc;
        printf("INGESTMON REC n=%llu cyc=%llu arwords=%llu pushes=%llu\n",
               (unsigned long long)mon.recs, (unsigned long long)cyc,
               (unsigned long long)mon.arwords,
               (unsigned long long)mon.pushes);
    }
    mon.prev_ack = mon.s_ack; mon.have_prev_ack = 1;
    // occupancy accumulation (per shell cycle, post-commit)
    if ((uint64_t)mon.occ > mon.occmax) mon.occmax = (uint64_t)mon.occ;
    mon.occsum += (uint64_t)mon.occ;
    int b = (mon.occ == 0) ? 0
          : (mon.occ >= 512) ? 9
          : (int)(1 + ((mon.occ - 1) >> 6));
    ++mon.hist[b];
}

static void mon_snapshot(const char* label) {
    if (!mon.armed) return;
    printf("INGESTMON SNAP %s cyc=%llu busy=%llu arhs=%llu arwords=%llu "
           "rbeats=%llu pushes=%llu stallfull=%llu pops=%llu tedges=%llu "
           "occ=%lld occmax=%llu occsum=%llu recs=%llu recgapsum=%llu "
           "recgapmax=%llu recgapn=%llu burstgapsum=%llu burstgapn=%llu "
           "reqtglsum=%llu reqtgln=%llu wfhs=%llu "
           "hist=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
           label, (unsigned long long)cyc,
           (unsigned long long)mon.busy, (unsigned long long)mon.arhs,
           (unsigned long long)mon.arwords, (unsigned long long)mon.rbeats,
           (unsigned long long)mon.pushes, (unsigned long long)mon.stallfull,
           (unsigned long long)mon.pops, (unsigned long long)mon.tedges,
           (long long)mon.occ, (unsigned long long)mon.occmax,
           (unsigned long long)mon.occsum, (unsigned long long)mon.recs,
           (unsigned long long)mon.recgapsum,
           (unsigned long long)mon.recgapmax,
           (unsigned long long)mon.recgapn,
           (unsigned long long)mon.burstgapsum,
           (unsigned long long)mon.burstgapn,
           (unsigned long long)mon.reqtglsum, (unsigned long long)mon.reqtgln,
           (unsigned long long)mon.wfhs,
           (unsigned long long)mon.hist[0], (unsigned long long)mon.hist[1],
           (unsigned long long)mon.hist[2], (unsigned long long)mon.hist[3],
           (unsigned long long)mon.hist[4], (unsigned long long)mon.hist[5],
           (unsigned long long)mon.hist[6], (unsigned long long)mon.hist[7],
           (unsigned long long)mon.hist[8], (unsigned long long)mon.hist[9]);
}
#endif  // APEX_INGEST_MON

// Two-phase clock so handshake outputs (combinational on inputs + st_q) can
// be SAMPLED on the low phase, before the rising edge that consumes them.
static void dump() {
#if VM_TRACE
    if (tfp && cyc >= trace_from_cyc && cyc <= trace_to_cyc)
        tfp->dump(vtime);
#endif
    ++vtime;
}
static void lo() { top->clk_main_a0 = 0; top->eval(); dump(); }
#ifdef APEX_INGEST_MON
static void hi() {
    mon_entry_sample();
    top->clk_main_a0 = 1; top->eval(); dump(); ++cyc;
    mon_after_edge();
}
#else
static void hi() { top->clk_main_a0 = 1; top->eval(); dump(); ++cyc; }
#endif

static void reset() {
    // 128 cycles per phase: the DECISION-LC-1 tile domain runs at
    // clk_main/(2*tile_div), and its 2FF reset synchronizer needs tile
    // posedges inside both phases (2 to assert, 2 to release) — 128 shell
    // cycles covers tile_div up to 16 with margin.
    top->rst_main_n = 0;
    for (int i = 0; i < 128; ++i) { lo(); hi(); }
    top->rst_main_n = 1;
    for (int i = 0; i < 128; ++i) { lo(); hi(); }
}

// +axi_wsplit: present AW alone first, W only after AW is accepted — the
// phase-decoupled ordering real shell AXI infrastructure is free to use
// (AXI gives the channels independent handshakes; nothing pairs them).
// The default (0) is the legacy same-cycle pairing every sim run to date
// has used — which is exactly why a sampler that ASSUMES pairing was
// sim-green and silicon-red (the 2026-08-07 walked-attention stale).
static int axi_wsplit = 0;

static bool axil_write(uint32_t a, uint32_t d) {
    uint64_t t0 = cyc;
    if (axi_wsplit) {
        top->ocl_cl_awaddr = a; top->ocl_cl_awvalid = 1;
        top->ocl_cl_wvalid = 0;
        for (;;) {                                 // AW phase alone
            lo();
            if (top->cl_ocl_awready) { hi(); break; }
            hi();
            if (cyc - t0 > AXI_LIMIT) { printf("AXI aw stall @%lu\n", cyc); return false; }
        }
        top->ocl_cl_awvalid = 0;
        top->ocl_cl_wdata = d; top->ocl_cl_wstrb = 0xF; top->ocl_cl_wvalid = 1;
        t0 = cyc;
        for (;;) {                                 // W phase, >=1 cycle later
            lo();
            if (top->cl_ocl_wready) { hi(); break; }
            hi();
            if (cyc - t0 > AXI_LIMIT) { printf("AXI w stall @%lu\n", cyc); return false; }
        }
        top->ocl_cl_wvalid = 0;
    } else {
        top->ocl_cl_awaddr = a; top->ocl_cl_awvalid = 1;
        top->ocl_cl_wdata = d;  top->ocl_cl_wstrb = 0xF; top->ocl_cl_wvalid = 1;
        for (;;) {                                 // aw/w accepted together
            lo();
            if (top->cl_ocl_awready) { hi(); break; }  // transfer on this edge
            hi();
            if (cyc - t0 > AXI_LIMIT) { printf("AXI aw stall @%lu\n", cyc); return false; }
        }
        top->ocl_cl_awvalid = 0; top->ocl_cl_wvalid = 0;
    }
    top->ocl_cl_bready = 1;
    t0 = cyc;
    for (;;) {
        lo();
        if (top->cl_ocl_bvalid) { hi(); break; }
        hi();
        if (cyc - t0 > AXI_LIMIT) { printf("AXI b stall @%lu\n", cyc); return false; }
    }
    top->ocl_cl_bready = 0; lo();
    return true;
}

static bool axil_read(uint32_t a, uint32_t* out) {
    top->ocl_cl_araddr = a; top->ocl_cl_arvalid = 1;
    uint64_t t0 = cyc;
    for (;;) {
        lo();
        if (top->cl_ocl_arready) { hi(); break; }
        hi();
        if (cyc - t0 > AXI_LIMIT) { printf("AXI ar stall @%lu\n", cyc); return false; }
    }
    top->ocl_cl_arvalid = 0;
    top->ocl_cl_rready = 1;
    t0 = cyc;
    for (;;) {
        lo();
        if (top->cl_ocl_rvalid) { *out = top->cl_ocl_rdata; hi(); break; }
        hi();
        if (cyc - t0 > AXI_LIMIT) { printf("AXI r stall @%lu\n", cyc); return false; }
    }
    top->ocl_cl_rready = 0; lo();
    return true;
}

static bool jint(const std::string& l, const char* key, long long* v);
static bool jstr(const std::string& l, const char* key, std::string* v);

// ═══ IB-FUEL s2: DDR image loader over PCIS (docs/design/IB_FUEL.md §1) ═══
// +ddr_image=<bin> +ddr_regions=<jsonl> loads the pre-swizzled weight image
// through the PCIS decode into (behavioral) DDR before any regops run, then
// reads every region back over PCIS, byte-compares against the file, and
// checks the SHA-256 of the READBACK stream against the manifest digest —
// loader -> DDR -> readback -> hash closes the integrity loop in-sim.

// compact self-contained SHA-256 (FIPS 180-4); self-tested at startup
struct Sha256 {
    uint32_t h[8]; uint64_t nbytes; uint8_t buf[64]; size_t fill;
    static uint32_t rr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void init() {
        static const uint32_t iv[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        memcpy(h, iv, sizeof h); nbytes = 0; fill = 0;
    }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
            0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
            0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
            0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
            0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
            0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16
                 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rr(w[i-15], 7) ^ rr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rr(w[i-2], 17) ^ rr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = rr(e,6) ^ rr(e,11) ^ rr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            uint32_t s0 = rr(a,2) ^ rr(a,13) ^ rr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = s0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const uint8_t* p, size_t n) {
        nbytes += n;
        while (n) {
            size_t take = 64 - fill; if (take > n) take = n;
            memcpy(buf + fill, p, take); fill += take; p += take; n -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    std::string hex() {
        uint64_t bits = nbytes * 8;
        uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0;
        while (fill != 56) update(&z, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = (uint8_t)(bits >> (56 - 8*i));
        update(lenb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) snprintf(out + 8*i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};
static std::string sha256_hex(const uint8_t* p, size_t n) {
    Sha256 s; s.init(); s.update(p, n); return s.hex();
}

// PCIS AXI4 master helpers (512-bit; same two-phase idiom as axil_*).
// exp_* lets the DECERR probe assert the ERROR responses (default OKAY).
static bool pcis_write_burst(uint64_t addr, uint16_t id,
                             const uint64_t* lanes, unsigned nbeats,
                             uint8_t exp_bresp = 0) {
    top->sh_cl_dma_pcis_awid = id;
    top->sh_cl_dma_pcis_awaddr = addr;
    top->sh_cl_dma_pcis_awlen = (uint8_t)(nbeats - 1);
    top->sh_cl_dma_pcis_awsize = 6;
    top->sh_cl_dma_pcis_awburst = 1;
    top->sh_cl_dma_pcis_awvalid = 1;
    uint64_t t0 = cyc;
    for (;;) {
        lo();
        bool rdy = top->cl_sh_dma_pcis_awready;
        hi();
        if (rdy) break;
        if (cyc - t0 > AXI_LIMIT) { printf("PCIS aw stall @%llx\n", (unsigned long long)addr); return false; }
    }
    top->sh_cl_dma_pcis_awvalid = 0;
    for (unsigned i = 0; i < nbeats; ++i) {
        for (int w = 0; w < 16; ++w)
            top->sh_cl_dma_pcis_wdata[w] =
                (uint32_t)(lanes[8*i + w/2] >> (32 * (w & 1)));
        top->sh_cl_dma_pcis_wstrb = ~0ull;
        top->sh_cl_dma_pcis_wlast = (i + 1 == nbeats);
        top->sh_cl_dma_pcis_wvalid = 1;
        t0 = cyc;
        for (;;) {
            lo();
            bool rdy = top->cl_sh_dma_pcis_wready;
            hi();
            if (rdy) break;
            if (cyc - t0 > AXI_LIMIT) { printf("PCIS w stall beat %u\n", i); return false; }
        }
    }
    top->sh_cl_dma_pcis_wvalid = 0;
    top->sh_cl_dma_pcis_wlast = 0;
    top->sh_cl_dma_pcis_bready = 1;
    t0 = cyc;
    for (;;) {
        lo();
        bool bv = top->cl_sh_dma_pcis_bvalid;
        uint8_t br = top->cl_sh_dma_pcis_bresp;
        hi();
        if (bv) {
            if (br != exp_bresp) { printf("PCIS bresp %d (want %d) @%llx\n", br, exp_bresp, (unsigned long long)addr); return false; }
            break;
        }
        if (cyc - t0 > AXI_LIMIT) { printf("PCIS b stall\n"); return false; }
    }
    top->sh_cl_dma_pcis_bready = 0;
    lo(); hi();
    return true;
}

static bool pcis_read_burst(uint64_t addr, uint16_t id,
                            uint64_t* lanes, unsigned nbeats,
                            uint8_t exp_rresp = 0) {
    top->sh_cl_dma_pcis_arid = id;
    top->sh_cl_dma_pcis_araddr = addr;
    top->sh_cl_dma_pcis_arlen = (uint8_t)(nbeats - 1);
    top->sh_cl_dma_pcis_arsize = 6;
    top->sh_cl_dma_pcis_arburst = 1;
    top->sh_cl_dma_pcis_arvalid = 1;
    uint64_t t0 = cyc;
    for (;;) {
        lo();
        bool rdy = top->cl_sh_dma_pcis_arready;
        hi();
        if (rdy) break;
        if (cyc - t0 > AXI_LIMIT) { printf("PCIS ar stall @%llx\n", (unsigned long long)addr); return false; }
    }
    top->sh_cl_dma_pcis_arvalid = 0;
    top->sh_cl_dma_pcis_rready = 1;
    unsigned got = 0;
    t0 = cyc;
    while (got < nbeats) {
        lo();
        bool rv = top->cl_sh_dma_pcis_rvalid;
        bool rl = top->cl_sh_dma_pcis_rlast;
        uint8_t rr_ = top->cl_sh_dma_pcis_rresp;
        if (rv)
            for (int w = 0; w < 16; ++w)
                lanes[8*got + w/2] = (w & 1)
                    ? (lanes[8*got + w/2] | ((uint64_t)top->cl_sh_dma_pcis_rdata[w] << 32))
                    : (uint64_t)top->cl_sh_dma_pcis_rdata[w];
        hi();
        if (rv) {
            if (rr_ != exp_rresp) { printf("PCIS rresp %d (want %d) beat %u\n", rr_, exp_rresp, got); return false; }
            bool want_last = (got + 1 == nbeats);
            if (rl != want_last) { printf("PCIS rlast misplace beat %u\n", got); return false; }
            ++got;
            t0 = cyc;
        }
        if (cyc - t0 > AXI_LIMIT) { printf("PCIS r stall beat %u\n", got); return false; }
    }
    top->sh_cl_dma_pcis_rready = 0;
    lo(); hi();
    return true;
}

// IB-FUEL s3: FUEL CSR BAR0 addresses (cl_apex bridge window 0x0)
static const uint32_t FUEL_CTRL = 0x0020, FUEL_STAT = 0x002C,
                      FUEL_ERR = 0x0030;

// s3 directed DECERR probe (IB_FUEL.md §1.3 disclosure closed here):
// out-of-range write + read answer DECERR with the poison payload and
// never touch DDR; RUN-mode (ar_owner=1) blocks PCIS reads with DECERR
// (the R8 never-wedge rule). Runs after the image load so a decode bug
// that corrupted real regions would also surface in the readback gate.
static int ddr_decerr_probe() {
    const uint64_t OOR = 1ull << 36;             // first byte past 64 GiB
    const uint64_t POISON = 0xDEC00DD0DEC00DD0ull;
    int fails = 0;
    uint64_t buf[8 * 2] = {0};
    // A: out-of-range write answered DECERR (bresp=3)
    if (!pcis_write_burst(OOR, 0x0EE0, buf, 1, 3)) { ++fails; printf("DDRPROBE A fail\n"); }
    // B: out-of-range 2-beat read: DECERR + poison payload + rlast placing
    if (!pcis_read_burst(OOR, 0x0EE1, buf, 2, 3)) { ++fails; printf("DDRPROBE B fail\n"); }
    else
        for (int i = 0; i < 16; ++i)
            if (buf[i] != POISON) { ++fails; printf("DDRPROBE: poison lane %d got %016llx\n", i, (unsigned long long)buf[i]); break; }
    // C: RUN mode blocks PCIS reads (in-range!) with DECERR; writes still
    // pass. SCRATCH is far above any image region so the write never
    // touches loaded weights.
    const uint64_t SCRATCH = 1ull << 30;         // 1 GiB, in-range
    uint32_t v = 0;
    if (!axil_write(FUEL_CTRL, 0x2)) { ++fails; printf("DDRPROBE C0 fail\n"); }
    { uint32_t c = 0, s = 0;
      axil_read(FUEL_CTRL, &c); axil_read(FUEL_STAT, &s);
      printf("DDRPROBE C: FUEL_CTRL=%08x FUEL_STAT=%08x\n", c, s); }
    if (!pcis_read_burst(SCRATCH, 0x0EE2, buf, 1, 3)) { ++fails; printf("DDRPROBE C1 fail (RUN-mode read not blocked?)\n"); }
    if (!pcis_write_burst(SCRATCH, 0x0EE3, buf, 1, 0)) { ++fails; printf("DDRPROBE C2 fail\n"); }
    if (!axil_write(FUEL_CTRL, 0x0)) { ++fails; printf("DDRPROBE C3 fail\n"); }
    if (!axil_read(FUEL_ERR, &v)) ++fails;
    else if ((v & 0xF) != 0) { ++fails; printf("DDRPROBE: FUEL_ERR %08x after probes\n", v); }
    printf("DDRPROBE RESULT: fails=%d -> %s\n", fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

// loader: returns 0 on success, 1 on any failure (region counts printed)
static int ddr_load_and_verify(const char* img_path, const char* reg_path) {
    // SHA self-test (empty string) — guards the local implementation
    if (sha256_hex((const uint8_t*)"", 0) !=
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        printf("DDRLOAD: sha256 self-test FAILED\n");
        return 1;
    }
    std::ifstream img(img_path, std::ios::binary);
    if (!img) { printf("DDRLOAD: cannot open %s\n", img_path); return 1; }
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(img)),
                              std::istreambuf_iterator<char>());
    std::ifstream rf(reg_path);
    if (!rf) { printf("DDRLOAD: cannot open %s\n", reg_path); return 1; }

    reset();
    long regions = 0, rb_fails = 0, sha_fails = 0;
    uint64_t words_total = 0;
    std::string line;
    while (std::getline(rf, line)) {
        long long b64 = 0, n64 = 0, tag = 0;
        std::string job, sha;
        if (!jint(line, "base_64B", &b64) || !jint(line, "beats_64B", &n64))
            continue;
        jint(line, "tag", &tag);
        jstr(line, "job", &job);
        jstr(line, "sha256", &sha);
        uint64_t base = (uint64_t)b64 * 64, words = (uint64_t)n64;
        if (base % 4096 != 0) { printf("DDRLOAD: %s base not 4KiB-aligned\n", job.c_str()); return 1; }
        if (base + words * 64 > file.size()) { printf("DDRLOAD: %s beyond file\n", job.c_str()); return 1; }
        // write in 64-beat (4 KiB) bursts
        for (uint64_t off = 0; off < words; off += 64) {
            unsigned n = (unsigned)((words - off > 64) ? 64 : (words - off));
            const uint64_t* src =
                reinterpret_cast<const uint64_t*>(file.data() + base) + 8 * off;
            if (!pcis_write_burst(base + 64 * off, (uint16_t)(0x0C00 + regions),
                                  src, n)) return 1;
        }
        // read back: byte-compare vs file AND hash the readback stream
        Sha256 rsha; rsha.init();
        std::vector<uint64_t> rb(8 * 64);
        bool region_ok = true;
        for (uint64_t off = 0; off < words; off += 64) {
            unsigned n = (unsigned)((words - off > 64) ? 64 : (words - off));
            std::fill(rb.begin(), rb.end(), 0);
            if (!pcis_read_burst(base + 64 * off, (uint16_t)(0x0D00 + regions),
                                 rb.data(), n)) return 1;
            const uint64_t* ref =
                reinterpret_cast<const uint64_t*>(file.data() + base) + 8 * off;
            if (memcmp(rb.data(), ref, 64ull * n) != 0) region_ok = false;
            rsha.update(reinterpret_cast<const uint8_t*>(rb.data()), 64ull * n);
        }
        if (!region_ok) ++rb_fails;
        std::string got = rsha.hex();
        if (!sha.empty() && got != sha) {
            ++sha_fails;
            printf("DDRLOAD: %s sha mismatch got %s want %s\n",
                   job.c_str(), got.c_str(), sha.c_str());
        }
        printf("DDRLOAD [%s] base=0x%llx words=%llu tag=%lld %s\n",
               job.c_str(), (unsigned long long)base,
               (unsigned long long)words, tag,
               (region_ok && (sha.empty() || got == sha)) ? "ok" : "MISMATCH");
        ++regions;
        words_total += words;
    }
    printf("DDRLOAD RESULT: regions=%ld words=%llu rb_fails=%ld sha_fails=%ld -> %s\n",
           regions, (unsigned long long)words_total, rb_fails, sha_fails,
           (rb_fails || sha_fails) ? "FAIL" : "PASS");
    return (rb_fails || sha_fails) ? 1 : 0;
}

// ── minimal flat-JSON field extraction (compiler emits flat objects) ───────
static bool jint(const std::string& l, const char* key, long long* v) {
    std::string pat = std::string("\"") + key + "\":";
    auto p = l.find(pat);
    if (p == std::string::npos) return false;
    *v = strtoll(l.c_str() + p + pat.size(), nullptr, 10);
    return true;
}
static bool jstr(const std::string& l, const char* key, std::string* v) {
    std::string pat = std::string("\"") + key + "\":\"";
    auto p = l.find(pat);
    if (p == std::string::npos) return false;
    auto e = l.find('"', p + pat.size());
    *v = l.substr(p + pat.size(), e - p - pat.size());
    return true;
}

struct Stats { long checks = 0, fails = 0, ops = 0;
               // read-back gate (see trace_to_regops.py `cap` / PROMPT_ON_CHIP.md §0)
               long cap_checked = 0, cap_mismatch = 0; };

// ── capture egress sink (+cap_out=<path>) ─────────────────────────────────
// Opened (truncating) at run start, one line per executed cap, closed and
// summarised by cap_finish() on EVERY exit path.
static FILE* cap_fp = nullptr;
static const char* cap_path = nullptr;
static long cap_i = 0;             // global cap sequence over the whole run

static void cap_finish() {
    if (cap_fp) { fclose(cap_fp); cap_fp = nullptr; }
    if (cap_path)                  // UNCONDITIONAL when asked for, n=0 too:
        printf("F2SIM CAPTURES: n=%ld out=%s\n", cap_i, cap_path);
}

static bool run_file(const char* path, Stats* st) {
    std::ifstream f(path);
    if (!f) { printf("cannot open %s\n", path); return false; }
    std::string line;
    long lineno = 0;
    while (std::getline(f, line)) {
        ++lineno; ++st->ops;
        std::string op; long long a = 0, d = 0, m = 0, e = 0;
        if (!jstr(line, "op", &op)) continue;
        jint(line, "a", &a); jint(line, "d", &d);
        jint(line, "m", &m);
        const bool has_e = jint(line, "e", &e);   // `cap` treats e as optional
        uint32_t v = 0;
        if (op == "note") {
            // E-6 measurement hook: a note whose text contains CYCMARK
            // prints the sim cycle counter — pure observability (no tile
            // access), so programs without the marker are byte-identical
            // in behaviour AND in output.
            std::string s;
            if (jstr(line, "s", &s) && s.find("CYCMARK") != std::string::npos) {
                printf("CYCMARK %s cyc=%llu\n",
                       s.substr(s.find("CYCMARK")).c_str(),
                       (unsigned long long)cyc);
#ifdef APEX_INGEST_MON
                // ingest snapshot at every CYCMARK: the walk window's
                // counters come from the PREGO/WALKDONE snapshot delta
                std::string lab = s.substr(s.find("CYCMARK"));
                for (auto& ch : lab) if (ch == ' ') ch = '_';
                mon_snapshot(lab.c_str());
#endif
            }
            continue;
        }
        if (op == "w") {
            if (!axil_write((uint32_t)a, (uint32_t)d)) return false;
        } else if (op == "pw") {                       // poll free, then write
            uint64_t t0 = cyc;
            for (;;) {
                if (!axil_read((uint32_t)a, &v)) return false;
                if ((v >> 9) & 0x7FFF) break;
                if (cyc - t0 > POLL_LIMIT) { printf("pw stall L%ld\n", lineno); return false; }
            }
            if (!axil_write((uint32_t)a, (uint32_t)d)) return false;
        } else if (op == "jf") {                       // poll ready, then fire
            uint64_t t0 = cyc;
            for (;;) {
                if (!axil_read((uint32_t)a, &v)) return false;
                if (v & 1) break;
                if (cyc - t0 > POLL_LIMIT) { printf("jf stall L%ld\n", lineno); return false; }
            }
            if (!axil_write((uint32_t)a, (uint32_t)d)) return false;
        } else if (op == "cap") {
            // READ AND KEEP — the only op that makes a tile-produced
            // number available to the caller rather than comparing it
            // against something the host already knows.
            //
            // `tag` is MANDATORY (same as f2_host_run.py): it is the value's
            // only identity — addresses are not identities (RO delivers 16
            // beats x 8 lanes through 8 addresses), so an untagged capture is
            // an anonymous number the grader cannot place.
            std::string tag;
            if (!jstr(line, "tag", &tag)) {
                printf("cap without tag L%ld @%04llx\n",
                       lineno, (unsigned long long)a);
                return false;
            }
            if (!axil_read((uint32_t)a, &v)) return false;
            uint32_t got = v & (uint32_t)m;
            if (cap_fp) {          // EGRESS record — format is contract
                fprintf(cap_fp,
                        "{\"tag\":\"%s\",\"addr\":\"0x%08x\",\"mask\":\"0x%08x\","
                        "\"value\":%u,\"i\":%ld}\n",
                        tag.c_str(), (uint32_t)a, (uint32_t)m, got, cap_i);
            }
            ++cap_i;
            if (has_e) {
                // STAGE-A SEMANTIC CHANGE (D-P1): a cap that CARRIES an
                // expectation and misses it now counts as a FAILURE, so the
                // file verdict and the process exit code go red. Before this
                // it only printed and the run still exited 0 — the capture
                // path could be wrong with every gate green. A cap WITHOUT
                // "e" still NEVER fails (that is Milestone B: capture where
                // no expectation exists).
                // BOTH sides masked — parity with f2_host_run.py, and the
                // emitter therefore need not pre-mask "e".
                ++st->cap_checked;
                const uint32_t exp = (uint32_t)e & (uint32_t)m;
                if (got != exp) {
                    ++st->cap_mismatch;
                    ++st->fails;
                    printf("CAPMISMATCH L%ld: [%04llx] tag=%s captured %08x "
                           "!= baked %08x\n",
                           lineno, (unsigned long long)a, tag.c_str(),
                           got, exp);
                }
            }
        } else if (op == "r") {
            if (!axil_read((uint32_t)a, &v)) return false;
            ++st->checks;
            if ((v & (uint32_t)m) != (uint32_t)e) {
                ++st->fails;
                printf("FAIL L%ld: [%04llx] got %08x want %08llx (mask %08llx)\n",
                       lineno, (unsigned long long)a, v,
                       (unsigned long long)e, (unsigned long long)m);
            }
        } else if (op == "rn") {
            if (!axil_read((uint32_t)a, &v)) return false;
            ++st->checks;
            if ((v & (uint32_t)m) == 0) {
                ++st->fails;
                printf("FAIL L%ld: [%04llx] got %08x expected nonzero under %08llx\n",
                       lineno, (unsigned long long)a, v, (unsigned long long)m);
            }
        } else if (op == "poll") {
            uint64_t t0 = cyc;
            for (;;) {
                if (!axil_read((uint32_t)a, &v)) return false;
                if ((v & (uint32_t)m) == (uint32_t)e) break;
                if (cyc - t0 > POLL_LIMIT) {
                    printf("poll stall L%ld: [%04llx] last %08x want %08llx/%08llx\n",
                           lineno, (unsigned long long)a, v,
                           (unsigned long long)e, (unsigned long long)m);
                    ++st->fails;
                    return false;
                }
            }
        } else {
            printf("unknown op '%s' L%ld\n", op.c_str(), lineno);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    top = std::make_unique<Vcl_apex>();
#if VM_TRACE
    Verilated::traceEverOn(true);
#if VM_TRACE_FST
    tfp = std::make_unique<VerilatedFstC>();
    top->trace(tfp.get(), 99);
    tfp->open("wave.fst");
#else
    tfp = std::make_unique<VerilatedVcdC>();
    top->trace(tfp.get(), 99);
    tfp->open("wave.vcd");
#endif
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "+trace_from_cyc=", 16))
            trace_from_cyc = strtoull(argv[i] + 16, nullptr, 10);
        if (!strncmp(argv[i], "+trace_to_cyc=", 14))
            trace_to_cyc = strtoull(argv[i] + 14, nullptr, 10);
    }
#endif
    // IB-FUEL s2: optional DDR image load+verify BEFORE any regops run.
    // DDR content survives the per-file reset()s below (the behavioral
    // model's memory is not reset-cleared, matching real DRAM across the
    // hardware flow's per-file TILE_RST which never resets the shell side).
    const char *ddr_img = nullptr, *ddr_reg = nullptr;
    bool decerr_probe = false, fuel_audit = false;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "+ddr_image=", 11)) ddr_img = argv[i] + 11;
        if (!strncmp(argv[i], "+ddr_regions=", 13)) ddr_reg = argv[i] + 13;
        if (!strcmp(argv[i], "+ddr_decerr_probe")) decerr_probe = true;
        if (!strcmp(argv[i], "+fuel_audit")) fuel_audit = true;
        if (!strcmp(argv[i], "+axi_wsplit")) axi_wsplit = 1;
        if (!strncmp(argv[i], "+cap_out=", 9)) cap_path = argv[i] + 9;
        if (!strncmp(argv[i], "+poll_limit=", 12)) {
            POLL_LIMIT = strtoull(argv[i] + 12, nullptr, 10);
            printf("POLL_LIMIT overridden: %llu cycles/poll op\n",
                   (unsigned long long)POLL_LIMIT);
        }
        if (!strcmp(argv[i], "+ingest_mon")) {
#ifdef APEX_INGEST_MON
            mon.armed = true;
            printf("INGESTMON armed (SIM-measured counters; behavioral twin"
                   " — no silicon claim)\n");
#else
            // fail-loud, never run a half-configured measurement
            printf("+ingest_mon needs a build with -DAPEX_INGEST_MON "
                   "(--public-flat-rw); this binary has no monitor\n");
            return 1;
#endif
        }
    }
    // Capture egress opened (TRUNCATING) at run start, before anything runs.
    if (cap_path) {
        cap_fp = fopen(cap_path, "w");
        if (!cap_fp) {
            printf("cannot open cap_out %s\n", cap_path);
            return 1;
        }
    }
    if ((ddr_img != nullptr) != (ddr_reg != nullptr)) {
        printf("need BOTH +ddr_image= and +ddr_regions=\n");
        cap_finish();
        return 1;
    }
    if (ddr_img && ddr_load_and_verify(ddr_img, ddr_reg)) {
        printf("F2SIM RESULT: files=0 checks=0 fails=1 -> FAIL\n");
        cap_finish();
        return 1;
    }
    if (decerr_probe && ddr_decerr_probe()) {
        printf("F2SIM RESULT: files=0 checks=0 fails=1 -> FAIL\n");
        cap_finish();
        return 1;
    }

    long total_checks = 0, total_fails = 0;
    long total_cap_checked = 0, total_cap_mismatch = 0;
    int files_run = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '+') continue;               // verilator plusargs
        reset();
        Stats st;
        bool ok = run_file(argv[i], &st);
        printf("[%s] %ld ops, %ld checks, %ld fails%s (cyc %lu)\n",
               argv[i], st.ops, st.checks, st.fails,
               ok ? "" : " [ABORTED]", (unsigned long)cyc);
#ifdef APEX_INGEST_MON
        mon_snapshot("FILE_END");
#endif
        total_checks += st.checks; total_fails += st.fails + (ok ? 0 : 1);
        total_cap_checked += st.cap_checked; total_cap_mismatch += st.cap_mismatch;
        // §9.1 R7 executor POSTLUDE — OUTSIDE the counted stream: after the
        // file, FUEL_ERR must be 0 and the fuel line quiescent (fifo empty,
        // no pending record). Failures fail the FILE, never touch `checks`.
        if (fuel_audit) {
            uint32_t fe = 0, fs = 0;
            bool aok = axil_read(FUEL_ERR, &fe) && axil_read(FUEL_STAT, &fs)
                       && ((fe & 0xF) == 0)      // no sticky fuel error
                       && ((fs & 0x1) == 1)      // fifo_empty
                       && ((fs & 0x2) == 0);     // no pending record
            printf("FUELAUDIT [%s] err=%08x stat=%08x -> %s\n",
                   argv[i], fe, fs, aok ? "ok" : "FAIL");
            if (!aok) ++total_fails;
        }
        ++files_run;
    }
    printf("F2SIM RESULT: files=%d checks=%ld fails=%ld -> %s\n",
           files_run, total_checks, total_fails,
           total_fails ? "FAIL" : "PASS");
    cap_finish();                  // closes the sink + prints CAPTURES n=/out=
    if (total_cap_checked) {
        // Read-back gate: every value the tile HANDED BACK equalled the
        // baked expectation the compare path was already validating.
        // Mismatches are ALSO inside total_fails above (D-P1).
        printf("F2SIM CAPGATE: captured=%ld mismatches=%ld -> %s\n",
               total_cap_checked, total_cap_mismatch,
               total_cap_mismatch ? "FAIL" : "PASS");
    }
#if VM_TRACE
    if (tfp) { tfp->close(); tfp.reset(); }
#endif
    top->final();
    // Release before main returns: these are file-scope smart pointers, so
    // static destruction would otherwise run after the Verilated context is
    // gone (SIGABRT on teardown, masking the exit status).
    top.reset();
    return total_fails ? 1 : 0;
}
