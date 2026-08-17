// tb_ddr.cpp — IB-FUEL s1 gate (a): directed AXI smoke for sh_ddr_beh.sv.
// Drives the behavioral model's cl_sh_ddr_axi_* pins directly (no CL, no
// kit includes). Same run twice from the Makefile: default knobs, then
// +ddr_stall=48879 +ddr_lat=3 — identical checks must pass under random
// ready backpressure and rvalid gaps.
//
// Checks per read beat: 512-bit data compare, rid, rlast placement; plus
// bid/bresp per write burst and the is_ready handshake. Deliberately NOT
// tested here: protocol violations (4 KiB straddle, bad size/burst) — the
// model $fatal's on those by design, which would kill the harness; they
// are covered by code inspection and by every master being our own RTL.
//
// Exit 0 = every check passed. Result line: DDRBEH RESULT: ...

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <array>
#include "Vsh_ddr.h"
#include "verilated.h"

static std::unique_ptr<Vsh_ddr> top;
static uint64_t cyc = 0;
static const uint64_t TMO = 200000;

using Beat = std::array<uint64_t, 8>;   // 8x64-bit lanes = 512 bits

static void lo() { top->clk = 0; top->stat_clk = 0; top->eval(); }
static void hi() { top->clk = 1; top->stat_clk = 1; top->eval(); ++cyc; }
static void tick() { lo(); hi(); }

static long checks = 0, fails = 0;
static void chk(bool ok, const char* what, uint64_t got, uint64_t want) {
    ++checks;
    if (!ok) {
        ++fails;
        printf("FAIL %s: got %016llx want %016llx (cyc %llu)\n", what,
               (unsigned long long)got, (unsigned long long)want,
               (unsigned long long)cyc);
    }
}

// deterministic patterns — mirror sh_ddr_beh.sv poison_lane exactly
static uint64_t poison_lane(uint64_t addr, int k) {
    uint64_t w = addr >> 6;
    return (0xD0ull << 56) | ((w & 0xFFFFFFFFFFFFull) << 8) | (uint64_t)(k & 7);
}
static uint64_t data64(int t, int i, int k) {
    return 0xC0DE000000000000ull | ((uint64_t)t << 40) | ((uint64_t)i << 16)
         | ((uint64_t)k << 8) | (uint64_t)((i * 7 + k * 13 + t) & 0xFF);
}

static void put_wdata(const Beat& b) {
    for (int w = 0; w < 16; ++w)
        top->cl_sh_ddr_axi_wdata[w] = (uint32_t)(b[w / 2] >> (32 * (w & 1)));
}
static Beat get_rdata() {
    Beat b{};
    for (int w = 0; w < 16; ++w)
        b[w / 2] |= (uint64_t)top->cl_sh_ddr_axi_rdata[w] << (32 * (w & 1));
    return b;
}

static bool wr_burst(uint64_t addr, uint16_t id,
                     const std::vector<Beat>& beats, uint64_t strb) {
    top->cl_sh_ddr_axi_awid    = id;
    top->cl_sh_ddr_axi_awaddr  = addr;
    top->cl_sh_ddr_axi_awlen   = (uint8_t)(beats.size() - 1);
    top->cl_sh_ddr_axi_awsize  = 6;
    top->cl_sh_ddr_axi_awburst = 1;
    top->cl_sh_ddr_axi_awvalid = 1;
    uint64_t t0 = cyc;
    for (;;) {
        lo();
        bool rdy = top->cl_sh_ddr_axi_awready;
        hi();
        if (rdy) break;
        if (cyc - t0 > TMO) { printf("TMO aw @%llx\n", (unsigned long long)addr); return false; }
    }
    top->cl_sh_ddr_axi_awvalid = 0;
    for (size_t i = 0; i < beats.size(); ++i) {
        put_wdata(beats[i]);
        top->cl_sh_ddr_axi_wstrb  = strb;
        top->cl_sh_ddr_axi_wlast  = (i + 1 == beats.size());
        top->cl_sh_ddr_axi_wvalid = 1;
        t0 = cyc;
        for (;;) {
            lo();
            bool rdy = top->cl_sh_ddr_axi_wready;
            hi();
            if (rdy) break;
            if (cyc - t0 > TMO) { printf("TMO w beat %zu\n", i); return false; }
        }
    }
    top->cl_sh_ddr_axi_wvalid = 0;
    top->cl_sh_ddr_axi_wlast  = 0;
    top->cl_sh_ddr_axi_bready = 1;
    t0 = cyc;
    for (;;) {
        lo();
        bool bv = top->cl_sh_ddr_axi_bvalid;
        uint16_t bid = top->cl_sh_ddr_axi_bid;
        uint8_t bresp = top->cl_sh_ddr_axi_bresp;
        hi();
        if (bv) {
            chk(bid == id, "bid", bid, id);
            chk(bresp == 0, "bresp", bresp, 0);
            break;
        }
        if (cyc - t0 > TMO) { printf("TMO b\n"); return false; }
    }
    top->cl_sh_ddr_axi_bready = 0;
    lo(); hi();
    return true;
}

static bool rd_burst(uint64_t addr, uint16_t id, size_t n,
                     std::vector<Beat>* out) {
    top->cl_sh_ddr_axi_arid    = id;
    top->cl_sh_ddr_axi_araddr  = addr;
    top->cl_sh_ddr_axi_arlen   = (uint8_t)(n - 1);
    top->cl_sh_ddr_axi_arsize  = 6;
    top->cl_sh_ddr_axi_arburst = 1;
    top->cl_sh_ddr_axi_arvalid = 1;
    uint64_t t0 = cyc;
    for (;;) {
        lo();
        bool rdy = top->cl_sh_ddr_axi_arready;
        hi();
        if (rdy) break;
        if (cyc - t0 > TMO) { printf("TMO ar @%llx\n", (unsigned long long)addr); return false; }
    }
    top->cl_sh_ddr_axi_arvalid = 0;
    top->cl_sh_ddr_axi_rready  = 1;
    out->clear();
    t0 = cyc;
    while (out->size() < n) {
        lo();
        bool rv = top->cl_sh_ddr_axi_rvalid;
        Beat b = rv ? get_rdata() : Beat{};
        uint16_t rid = top->cl_sh_ddr_axi_rid;
        bool rlast = top->cl_sh_ddr_axi_rlast;
        hi();
        if (rv) {
            chk(rid == id, "rid", rid, id);
            bool want_last = (out->size() + 1 == n);
            chk(rlast == want_last, "rlast", rlast, want_last);
            out->push_back(b);
            t0 = cyc;
        }
        if (cyc - t0 > TMO) { printf("TMO r beat %zu\n", out->size()); return false; }
    }
    top->cl_sh_ddr_axi_rready = 0;
    lo(); hi();
    return true;
}

static void cmp_beats(const char* what, const std::vector<Beat>& got,
                      const std::vector<Beat>& want) {
    for (size_t i = 0; i < want.size(); ++i)
        for (int k = 0; k < 8; ++k) {
            char nm[64];
            snprintf(nm, sizeof nm, "%s[%zu].l%d", what, i, k);
            chk(got[i][k] == want[i][k], nm, got[i][k], want[i][k]);
        }
}

static std::vector<Beat> pat(int t, size_t n) {
    std::vector<Beat> v(n);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 8; ++k) v[i][k] = data64(t, (int)i, k);
    return v;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    top = std::make_unique<Vsh_ddr>();

    top->rst_n = 0; top->stat_rst_n = 0;
    for (int i = 0; i < 20; ++i) tick();
    top->rst_n = 1; top->stat_rst_n = 1;

    // is_ready handshake (default +ddr_rdy=100)
    uint64_t t0 = cyc;
    while (!top->sh_cl_ddr_is_ready) {
        tick();
        if (cyc - t0 > TMO) { printf("TMO is_ready\n"); return 1; }
    }
    chk(true, "is_ready", 1, 1);

    std::vector<Beat> rb;
    bool ok = true;

    // T1: single beat @0x0
    auto w1 = pat(1, 1);
    ok = ok && wr_burst(0x0, 0x00A5, w1, ~0ull);
    ok = ok && rd_burst(0x0, 0x005A, 1, &rb);
    if (ok) cmp_beats("T1", rb, w1);

    // T2: 8-beat burst @0x1000
    auto w2 = pat(2, 8);
    ok = ok && wr_burst(0x1000, 0x0011, w2, ~0ull);
    ok = ok && rd_burst(0x1000, 0x0012, 8, &rb);
    if (ok) cmp_beats("T2", rb, w2);

    // T3: full 4 KiB burst (64 beats) @0x10000
    auto w3 = pat(3, 64);
    ok = ok && wr_burst(0x10000, 0x0022, w3, ~0ull);
    ok = ok && rd_burst(0x10000, 0x0023, 64, &rb);
    if (ok) cmp_beats("T3", rb, w3);

    // T4: unwritten region reads poison (deterministic, address-salted)
    ok = ok && rd_burst(0x200000, 0x0033, 4, &rb);
    if (ok) {
        std::vector<Beat> want(4);
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 8; ++k)
                want[i][k] = poison_lane(0x200000 + 64ull * i, k);
        cmp_beats("T4", rb, want);
    }

    // T5: partial strobe (lane 0 only) merges against poison
    auto w5 = pat(5, 1);
    ok = ok && wr_burst(0x8000, 0x0044, w5, 0xFFull);
    ok = ok && rd_burst(0x8000, 0x0045, 1, &rb);
    if (ok) {
        std::vector<Beat> want(1);
        want[0][0] = w5[0][0];
        for (int k = 1; k < 8; ++k) want[0][k] = poison_lane(0x8000, k);
        cmp_beats("T5", rb, want);
    }

    // T6: overwrite T1's word, read back the new value
    auto w6 = pat(6, 1);
    ok = ok && wr_burst(0x0, 0x0055, w6, ~0ull);
    ok = ok && rd_burst(0x0, 0x0056, 1, &rb);
    if (ok) cmp_beats("T6", rb, w6);

    if (!ok) ++fails;   // any TMO/abort counts as a failure
    printf("DDRBEH RESULT: checks=%ld fails=%ld -> %s\n", checks, fails,
           fails ? "FAIL" : "PASS");
    top->final();
    top.reset();
    return fails ? 1 : 0;
}
