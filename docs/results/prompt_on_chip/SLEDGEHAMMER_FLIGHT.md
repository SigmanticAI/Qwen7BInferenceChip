# SLEDGEHAMMER FLIGHT — the sweep is defeated, the stale is NOT: the sweep was a SYMPTOM

**Date:** 2026-08-10 · **Image:** `agfi-01ebc7b159ce88d56` (apex-sh-20260810,
@fb0bd44 — whole-bank `dont_touch`+`keep_hierarchy` + kept snoop write-enable
chain `wc_snp_valid_k`; WNS +0.426) · **Instance:** `i-010b97077ac3bc862`
(terminated+verified) · 20th image, 20th first-try ingestion.

## Pre-flight forensics (the fix DID what it was asked)

Batch-Vivado on `cl_apex.2026_08_10-184003.post_route.dcp`:

```
WCOMP_TOTAL_CELLS: 6085     (kill-shot image: ~24)
g_comp[0].u_comp:  2997     g_comp[1].u_comp: 3029
SC_VAL cells: 1288          SC_MEM cells: 4084
```

The composite scale cache physically exists in this netlist. The
Vivado sweep — five images' worth of ~10-cell ghosts — is DEFEATED.

## The flight (clean: root run, clkgen preflight green, A2 @15.62 verified)

- `walk_e6`: **STILL RED** — poll stall @0x1068 `0x3703` (WALK_ERR_SEQ), the
  historical signature exactly.
- `walk_e6_dbg2` term word: **`0x02000542` — BIT-IDENTICAL to the kill-shot
  and dbg5 flights.** snp=2, commits at addrs 0&1, engine 0, TERM_SC=1,
  TERM_SQ=0. With the cache present, `sc_val[0]` still reads 0 after two
  observed commits.
- Controls on the SAME card/image: `walk_e7ng` 26 checks 0 fails,
  `hostattn_fuelarm` 113 checks 0 fails → the sledgehammer attributes broke
  nothing; the image is healthy everywhere except walked attention.

## The diagnosis this forces

The sweep was **synthesis agreeing with the fault, not causing it**: Vivado
deleted the cache because it could PROVE the write never happens under the
conditions the netlist sees — and with storage forced into existence, the
write still doesn't happen (or is erased). The fold point and the silicon
fault are the SAME upstream fact about the commit conjunction
(`snp_valid && !snp_flush && snp_last && beat_is_last`).

Post-flight forensics 6 on the same DCP: `sc_val_reg[0]` is a real FDRE,
CE=VCC (gating folded into the D-LUT `sc_val[0]_i_1`), SR live, clk
connected. So the lie lives in the D-LUT input cone or the SR cone —
forensics 7 (running) enumerates every input net, its driver, and
constness. One of those nets embodies the term that silicon never asserts.

## Also fixed this session (operational)

`pci attach ... error 25: unresponsive` on a fresh card is **sysfs file
permissions**, not a hung device (`fpga_pci.c:243` raises UNRESPONSIVE on
`open()` failure). Fix: `sudo chmod 666 /sys/bus/pci/devices/<BDF>/resource*`
or run the executor as root. Prior cards had the SDK udev rule active.

## Cost

f2.6xlarge ~45 min ≈ $1.60. Terminated + verified.
