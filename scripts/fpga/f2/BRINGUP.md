# APEX on AWS F2 — bring-up runbook (S15 prep)

> **STATUS: PREP ONLY — NOTHING HERE HAS BEEN EXECUTED.** Every AWS-side step
> is **VALIDATION-DEFERRED** (marked ⏸): no AWS account action, no spend, no
> build has been run. This document is authored from primary sources (cited
> at the bottom, fetched 2026-07-16 with a verification pass) so that the
> first real F2 session is turnkey. Spend requires owner approval.

## 0. What this brings up

The full APEX attention tile (`rtl/top/apex_top.sv`) behind the AWS F2
shell: CSR + KV-engine register access from the host over PCIe first (smoke),
then scripted attention jobs replaying the L3 vectors against the golden
model on real cloud FPGA hardware. Target part: AMD Virtex UltraScale+ HBM
**VU47P** (16 GiB HBM + 64 GiB DDR4 per FPGA — neither needed for smoke; the
tile is host-sequenced with on-tile SRAM only).

**This is a port, not a bitstream copy:** the in-repo ECP5 artifact is a
Lattice open-flow build of the KVQ engine; F2 is an AMD part behind AWS's
shell, built with Vivado. F1-era AFIs/flows are explicitly incompatible with
F2 (different FPGA + redesigned shell) — everything goes through the F2 dev
kit.

## 1. Facts the design leans on (verified 2026-07-16)

| fact | value | why it matters |
|---|---|---|
| Dev kit | `aws/aws-fpga`, branch **`f2`** (repo default), release **v2.3.3** (2026-06-22) | pin the checkout |
| Vivado | 2024.1 / 2024.2 / 2025.1 / 2025.2 supported; FPGA Developer AMI 1.19.3 ships 2025.2 (license included for on-EC2 use, AMI itself free) | no local Vivado license needed |
| Build host | any ≥4 vCPU / 32 GiB EC2 (C/M family) — **an F2 instance is NOT needed for synthesis** | DCP builds cost ~$2, not F2 rates |
| Shell | **F2 Small Shell only** (v0x10212415). **No built-in DMA**; XDMA-shell CL builds fail per ERRATA; 88% of FPGA resources usable | streaming goes PCIS-mmap or SDE IP |
| CL interfaces | OCL AXI-Lite 32-bit → AppPF **BAR0** (64 MiB) · SDA AXI-Lite (MgmtPF BAR4) · PCIS 512-bit AXI4 inbound → AppPF **BAR4** (128 GiB, 8 µs timeout) · PCIM 512-bit outbound · 16 MSI-X (`cl_sh_apppf_irq_req/ack`) | CSRs on OCL; bulk data on PCIS |
| Clocking | `clk_main_a0` **fixed 250 MHz** for all shell interfaces; additional/slower clocks only via the AWS_CLK_GEN IP (recipes programmed over SDA; `fpga-load-clkgen-*` tools) | see DECISION-F2-1 |
| Port contract | `hdk/common/shell_stable/design/interfaces/cl_ports.vh` | wrapper port list source of truth |
| Examples | `cl_axil_reg_access` (OCL hello-world), `cl_dram_hbm_dma`, `cl_mem_perf`, `cl_sde` (~9 GB/s streaming), `CL_TEMPLATE` | start from CL_TEMPLATE |
| Build → run | `aws_build_dcp_from_cl.py` (30–90 min for kit examples) → tarball w/ manifest v2 → `hdk/scripts/create_afi.py` (or `aws ec2 create-fpga-image`, S3 same-region) → poll `describe-fpga-images` pending→available (**can take hours**) → `sudo fpga-load-local-image -S 0 -I agfi-…` | budget a day for the first AFI cycle |
| Host register access | SDK `fpga_pci` C lib (`fpga_pci_attach/peek/poke`, `FPGA_APP_PF=0`, `APP_PF_BAR0=0`) + **official Cython bindings** (`sdk/userspace/cython_bindings`, `fpga_pci_example.py`) | `host_smoke.py` uses these |
| Instances | f2.6xlarge = 1 FPGA / 24 vCPU / 256 GiB, ~**$1.98/hr** on-demand (third-party price page — re-check at spend time); 12xl/48xl = 2/8 FPGAs; 8 regions incl. us-east-1, us-west-2 | smoke sessions are ~$5–10 |
| Quota | "Running On-Demand F instances" **L-74FC7D96** defaults to **0 vCPUs** on fresh accounts → request ≥24 vCPUs BEFORE launch day (approval can take ~a day) | do this first |

## 2. Integration design

### 2.1 The tile's host surface (from `rtl/top/apex_top.sv`, ~126 ports)

- `csr_*` — simple 8-bit-addr / 32-bit-data register bus (tile CSR window §7).
- `kv_*` — full AXI-Lite (8-bit addr) into the KVQ engine + `kv_irq`,
  eviction sideband.
- **Streams** (valid/ready): descriptor `ds_*`, weights `xw_*` (lane8 beats),
  activations `xa_*`, gamma `xg_*`, sidebands `qs_/cs_*`, job pulses
  `fj_/qj_/dj_/lj_/aj_/wj_*`, route levels `rt_*`, scale taps out.
  Today the L3 testbench choreographs all of these; on F2 the **host
  program** takes the TB's role.

### 2.2 CL mapping (implemented in `cl_apex/design/cl_apex.sv` — stage-1,
Verilator-linted 0-errors against the v2.3.3 shell contract; Vivado build
deferred to the dev box)

| shell side | APEX side | mechanism |
|---|---|---|
| OCL AXI-Lite, BAR0 `0x0000–0x0FFF` | `apex_f2_bridge` register file | wrapper-owned: ID/version/scratch, tile reset, stream-mailbox + job-pulse registers (stage 2) |
| OCL AXI-Lite, BAR0 `0x1000–0x10FF` | tile `csr_*` bus | address-window translation (AXI-Lite → simple bus, 1 outstanding) |
| OCL AXI-Lite, BAR0 `0x2000–0x20FF` | `kv_*` AXI-Lite | direct passthrough |
| PCIS AXI4 (BAR4) | bulk stream feed (weights/activations) | stage 3: PCIS write-window → per-stream async FIFOs; SDE IP if mmap throughput is insufficient |
| `cl_sh_apppf_irq_req[0]` | `kv_irq` | MSI-X |

**Smoke needs only the OCL rows.** Registers are wide enough at AXI-Lite
rates to drive whole attention jobs slowly through the mailbox path — the
same job scripts L3 uses, throughput irrelevant for bit-exactness checks.

### 2.3½ Stage-2/3 design (authored 2026-07-17; RTL/software not started)

**Stage 2 — mailbox job driver (BAR0 window `0x3___`):** turns the host into
the L3 testbench. One OCL write = one stream beat or one job pulse; one OCL
read = one output beat + status. Register classes (exact field packing is
fixed against `apex_pkg` widths when the RTL lands):
- `STREAM_PUSH[s]` — one write pushes a beat into a small (≥4-deep) FIFO in
  front of stream *s* (`ds/xw/xa/xg/qs/cs`); wide beats (descriptor, lane8)
  assemble from N consecutive 32-bit writes, little-endian. Reads return
  `{fifo_free, accepted_count}` so the host never overruns.
- `JOB_FIRE[j]` — one write packs and pulses a job (`fj/qj/dj/lj/aj/wj`)
  with its fields in the write data; read returns `{ready, done_sticky}`.
- `ROUTE` — the `rt_*` level bundle as one register (change only while the
  touched paths are idle — same contract as the TB).
- `OUT_POP[o]` — reads pop `ro/fs/ss/td` beats from capture FIFOs;
  `{valid, data}` framing, `dn_*`/`err_sticky` already live at `0x0___`.
The host driver is a port of the L3 job scripts: replay the committed
`docs/results/s8_7b_token/artifact_trace` jobs register-by-register and
compare outputs bit-exact against the stored golden results — the S8 chain,
on silicon. Throughput is irrelevant for this proof (AXI-Lite rates fine).

**Stage 3 — the "type hello" demo (`demo/chat_kv.py`, authored):** a chat
REPL whose per-layer KV cache round-trips through a pluggable codec backend —
`twin` (the S4/S5-certified software twin; runnable on a quiet machine
today) or `fpga` (same `qdq_values/qdq_keys` interface served by the KVQ
engine over this bridge; wired at bring-up). KV-in-hardware chat is
interactive-speed and honest ("the conversation's memory lives in our
hardware"); full attention-on-tile chat stays a patience-demo until the B1
walker exists (~0.27 tok/s host-sequenced ceiling — perf model §3c).

### 2.3 DECISION-F2-1 (open): tile clock

`clk_main_a0` is fixed at 250 MHz. The tile's routed ceiling on ECP5 fabric
is ~35–45 MHz, but VU47P is 16 nm UltraScale+ — the same RTL plausibly closes
at 250 MHz there. Plan: **(a)** first DCP attempt runs the tile directly on
`clk_main_a0` (no CDC — simplest correct design); **(b)** if timing fails,
generate a slower tile clock with AWS_CLK_GEN and add async-FIFO CDC on the
OCL/PCIS crossings (more RTL, stage-3 scope). Do not guess — let the first
Vivado timing report decide. ⏸ deferred to the first build.

## 3. Runbook (⏸ = deferred, needs AWS account/spend)

1. ⏸ **Account prep (day −1):** request Service Quota **L-74FC7D96 ≥ 24
   vCPUs** in us-east-1 (and us-west-2 as fallback). Create an S3 bucket in
   the same region for DCP tarballs + AFI logs.
2. ⏸ **Dev box:** launch a c6a.8xlarge (or ≥4 vCPU/32 GiB) from **FPGA
   Developer AMI 1.19.3**; `git clone aws-fpga` (branch `f2`, tag v2.3.3);
   `source hdk_setup.sh`.
3. ⏸ **CL setup:** copy `CL_TEMPLATE` → `cl_apex`; drop in this repo's
   `rtl/` + `scripts/fpga/f2/cl_apex_wrapper.sv`; port list per
   `cl_ports.vh`. Run the kit's CL simulation smoke if available.
4. ⏸ **DCP:** `bash scripts/fpga/f2/build_dcp.sh` (wraps
   `aws_build_dcp_from_cl.py`). Read the timing report → resolve
   DECISION-F2-1. 30–90 min class runtime.
5. ⏸ **AFI:** `bash scripts/fpga/f2/create_afi.sh` (wraps
   `hdk/scripts/create_afi.py`; polls to `available` — hours).
6. ⏸ **Smoke (first F2 spend):** launch f2.6xlarge; `sudo
   fpga-load-local-image -S 0 -I agfi-…`; build SDK Cython bindings; run
   `python3 scripts/fpga/f2/host_smoke.py` → expects the INFO_* register
   signature (see script) and scratch write/read. Terminate instance.
7. ⏸ **Stage 2+ (post-smoke):** bridge mailbox job driving (L3 vector
   replay vs golden on hardware), then PCIS/SDE bulk path.

## 4. Cost + time budget (owner approves before any step runs)

| item | est. cost | est. wall time |
|---|---|---|
| Quota request | $0 | ~1 day approval |
| DCP build session (c6a.8xlarge ~$1.22/hr) | ~$2–4/spin | 1–2 h/spin |
| AFI ingestion | $0 | hours (poll) |
| F2 smoke session (f2.6xlarge ~$1.98/hr) | ~$5–10 | 2–4 h |
| **First full cycle (2–3 DCP spins)** | **~$15–30** | ~2 elapsed days |

## 5. What is NOT claimed

No AFI exists; nothing has run on F2; the wrapper is an unsynthesized
skeleton; the 250 MHz question is open. Any public wording about F2 stays at
"turnkey bring-up scripts prepared" until a load + smoke log is committed
here (same anti-fabrication rule as everything else: no PASS without pasted
output).

## 6. Sources (fetched + verified 2026-07-16)

- github.com/aws/aws-fpga (branch `f2`, release v2.3.3) — shell version,
  interfaces, examples, `supported_vivado_versions.txt`, ERRATA.md (XDMA
  unsupported), `hdk/scripts/create_afi.py`, SDK `fpga_pci` + Cython bindings
- awsdocs-fpga-f2.readthedocs-hosted.com — F2 dev-kit docs mirror
- aws.amazon.com/ec2/instance-types/f2 — instance sizes, VU47P, HBM/DDR4
- AWS Service Quotas — L-74FC7D96 (On-Demand F instances)
- instances.vantage.sh — on-demand pricing (third-party; re-check at spend)
