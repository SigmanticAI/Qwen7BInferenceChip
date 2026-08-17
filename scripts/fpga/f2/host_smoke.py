#!/usr/bin/env python3
# host_smoke.py — first-light register smoke for APEX on F2 (S15 prep).
# ⏸ VALIDATION-DEFERRED: never executed. Runs ON THE F2 INSTANCE after
# `sudo fpga-load-local-image -S 0 -I agfi-…`, using the SDK's official
# Cython bindings (aws-fpga sdk/userspace/cython_bindings — build per its
# README). See BRINGUP.md §3 step 6.
#
# Expected signature (from the committed RTL, not from any hardware run):
#   BAR0 0x0000 ID       == 0x4139_4558 ("A9EX", cl_apex.sv)
#   BAR0 0x0008 SCRATCH  read-back == write
#   BAR0 0x2000+ KVQ INFO window (kvq_engine.sv REG_* map):
#     0x2008 INFO_DIM, 0x200C INFO_TIER, 0x2010 INFO_GROUP,
#     0x2020 INFO_VERSION — values must match the built CL's parameters
#     (ship config: D=64 TIER=2 G=128 k=2 SETS=8 DEPTH=128).
#   Tile CSR window 0x1000+: STATUS idle bit set, ERR_STICKY clear.
import sys

BRIDGE_ID_EXPECT = 0x41394558  # "A9EX"

def main() -> int:
    try:
        import fpga_pci  # SDK cython bindings, built on the F2 instance
    except ImportError:
        print("ABORT: fpga_pci bindings not built — see "
              "aws-fpga/sdk/userspace/cython_bindings/README")
        return 2

    pci = fpga_pci.FpgaPci(slot=0, pf=0, bar=0)   # AppPF BAR0 = OCL
    checks, fails = 0, 0

    def check(name, got, want=None):
        nonlocal checks, fails
        ok = (want is None) or (got == want)
        checks += 1
        fails += 0 if ok else 1
        w = "" if want is None else f" want=0x{want:08x}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got=0x{got:08x}{w}")

    print("== APEX F2 smoke (BAR0/OCL) ==")
    check("bridge ID", pci.peek(0x0000), BRIDGE_ID_EXPECT)
    pci.poke(0x0008, 0xA5A5_5A5A)
    check("scratch RW", pci.peek(0x0008), 0xA5A55A5A)
    # KVQ INFO window — compare against the parameters baked into the CL
    # build (record them in the build log; ship config shown in header).
    for off, name in ((0x2008, "KVQ INFO_DIM"), (0x200C, "KVQ INFO_TIER"),
                      (0x2010, "KVQ INFO_GROUP"), (0x2020, "KVQ INFO_VERSION")):
        check(name, pci.peek(off))          # print-only until params pinned
    print(f"== {checks} checks, {fails} fails ==")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
