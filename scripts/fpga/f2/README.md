# scripts/fpga/f2 — APEX on AWS F2 (S15 prep, ⏸ validation-deferred)

Turnkey bring-up package for putting the full APEX tile behind the AWS F2
Small Shell. **Nothing here has been executed — no AWS action, no spend, no
build.** Owner approves spend before any ⏸ step runs.

| file | what |
|---|---|
| [`BRINGUP.md`](BRINGUP.md) | the runbook: verified F2 facts (2026-07-16), integration design, step-by-step flow, cost table, open DECISION-F2-1 (clocking) |
| [`cl_apex/`](cl_apex/) | **stage-1 CL design (supersedes the earlier skeleton):** `design/cl_apex.sv` — OCL AXI-Lite FSM (BAR0 windows: bridge regs / tile CSR / KVQ AXI-Lite) + the **full apex_top instantiation**, Verilator-linted 0-errors against the v2.3.3 shell contract (`cl_ports.vh` + unused-interface templates + `sh_ddr` stub; remaining warnings are kit-internal) · `design/cl_id_defines.vh` · `design/apex_sources.f` (ordered RTL list) · `build/scripts/synth_cl_apex.tcl`. ⏸ Vivado build itself still deferred to the dev box |
| [`build_dcp.sh`](build_dcp.sh) | CL → post-route DCP tarball (runs on an EC2 dev box, not locally) |
| [`create_afi.sh`](create_afi.sh) | DCP tarball → AFI/AGFI via the kit's `create_afi.py` (+ manual fallback) |
| [`host_smoke.py`](host_smoke.py) | first-light register smoke on the F2 instance (SDK Cython `fpga_pci`) |

Claim discipline: until a load + smoke log is committed here, the only
public wording is "turnkey F2 bring-up scripts prepared" — never "runs on
AWS F2".
