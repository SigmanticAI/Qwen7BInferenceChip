# F2 first light — the APEX tile answering from real hardware (AWS F2)

**Date:** 2026-07-17 · **AFI:** `afi-03232169ea962036d` / AGFI
`agfi-0f7c93ffa798ecc3f` · **Instance:** f2.6xlarge (VU47P), us-west-2,
terminated after the session · **Shell:** F2 Small Shell `0x10212415` ·
**CL:** `scripts/fpga/f2/cl_apex/` at commit eac38ee (tile config D=64,
G=128, DEPTH=128, OUTLIER_K=0)

## Build (evidence: [`cl_apex...post_route_timing.rpt`](cl_apex.2026_07_17-202346.post_route_timing.rpt))

The full attention tile (`apex_top`: systolic MXE, KVQ tier bank, ASU
softmax/RMSNorm, TIP, SEQ/CSR, seams, stage buffers) + the OCL bridge built
clean with Vivado 2025.2: synthesis + place-and-route in **15 m 39 s, zero
errors**, post-route timing **MET at 250 MHz** (`clk_main_a0`, worst slack
**+0.711 ns**) — no CDC, the tile runs at the shell clock (DECISION-F2-1(a)).
Total RTL changes needed for a codebase that had never met Vivado: **one
`wire` keyword** (eac38ee).

## First light (evidence: [`smoke_session.log`](smoke_session.log), verbatim)

`fpga-load-local-image -S 0 -I agfi-0f7c93ffa798ecc3f` → `loaded ... ok`,
and the AppPF re-enumerated with our CL ID (DeviceId `0x9048 → 0xf006`).
BAR0 probe over PCIe (C, SDK `fpga_pci`):

```
peek 0x0000 = 0x41394558   bridge ID "A9EX"
poke/peek 0x0008           scratch: 0xA5A55A5A then 0x5A5AA5A5, both exact
peek 0x2008 = 0x00000040   KVQ INFO_DIM   = 64   (ship D)
peek 0x200C = 0x00000001   KVQ INFO_TIER  = CQ-8 (engine 0 of the D-024 bank;
                           tier_sel tied 0 in the smoke CL — per-engine truth)
peek 0x2010 = 0x00000080   KVQ INFO_GROUP = 128  (ship G)
peek 0x2020 = 0x00020001   KVQ INFO_VERSION
peek 0x1004 = 0x00000001   tile CSR STATUS: idle
peek 0x0010 = 0x00000000   err_sticky[15:0] = 0 after load
PROBE: ALL PASS
```

The registers answering are the same CSRs the verification matrix exercises
in simulation — the verified RTL, on hardware, reporting its own build truth.

## Scope (what this is and is not)

- This is **register first light**: configuration/status reads and scratch
  writes over PCIe. No attention job has run on F2 hardware yet — that is
  bring-up stage 2 (mailbox bridge + replay of the committed S8 trace,
  design in `scripts/fpga/f2/BRINGUP.md` §2.3½).
- 250 MHz on VU47P is a **different part and toolchain** than the in-repo
  ECP5 open-flow story — the two numbers are never interchangeable.
- Cost of the whole arc (dev box + 4 build iterations + AFI + smoke
  session): ≈ **$12**.

## Reproduce

```sh
# dev box (FPGA Developer AMI Ubuntu): scp rtl/ + scripts/fpga/f2/ → ~/apex_src
bash apex_src/f2/devbox_setup.sh          # kit v2.3.3 + CL assembly + DCP build
bash scripts/fpga/f2/create_afi.sh <tarball>   # or the manual aws-cli fallback
# f2.6xlarge: bash f2_smoke_session.sh    # loads the AGFI, runs the BAR0 probe
```
