#!/usr/bin/env python3
# f2_host_run.py — F2 stage-2 regops executor, HOST side (real silicon).
#
# The counterpart to verif/f2sim/sim_main.cpp: it executes the SAME
# regops.jsonl files that trace_to_regops.py compiles from the S8 trace, but
# over the real FPGA's BAR0 via the SDK fpga_pci bindings instead of
# Verilator AXI-Lite pins. "One artifact, two executors" (that contract is
# stated verbatim in sim_main.cpp) — this closes the loop: when the sim goes
# green, this replays the identical jobs on the loaded AGFI and every
# result-check either passes on hardware or names the first divergence.
#
# ⚠ RUNS ON THE F2 INSTANCE only, after `sudo fpga-load-local-image -S 0 -I
# agfi-...` (see scripts/fpga/f2/f2_smoke_session.sh). Needs the SDK Cython
# bindings built (aws-fpga/sdk/userspace/cython_bindings). NOT yet run —
# same claim discipline as everything F2: no PASS reported without a pasted
# session log committed under docs/results/.
#
# Op semantics MUST stay bit-for-bit identical to sim_main.cpp:
#   note                     annotation (printed)
#   w    a d                 BAR0 write
#   r    a m e               read, check (val & m) == e   [counted]
#   rn   a m                 read, check (val & m) != 0   [counted]
#   cap  a m tag [e]         read and KEEP (val & m), keyed by `tag`; with the
#                            optional `e` also check (val & m) == (e & m) and
#                            count a mismatch as a FAIL  [not a counted check]
#   poll a m e               read until (val & m) == e (bounded)
#   pw   a d                 stream push: poll free ((val>>9)&0x7FFF)>0, write
#   jf   a d                 job fire: poll ready (bit0), write
# POLL_LIMIT / AXI semantics are host-timed here (wall-clock bounded), since
# a real read is ~1 us not one sim cycle.
#
# CAPTURE EGRESS (--cap-out <path>; sim: +cap_out=<path>). Both executors
# write the SAME format, one JSON line per EXECUTED cap, file truncated at
# run start:
#   {"tag":"<str>","addr":"0x%08x","mask":"0x%08x","value":<int>,"i":<seq>}
# `i` is a global 0-based counter over the whole run (all files, in argv
# order), addr/mask are lowercase 8-digit hex. Byte-identical egress from the
# two executors IS the parity check — diff the two files.
#
# ══ TRANSPORT (2026-08-14): THE COMPILED / BURST EXECUTOR ═══════════════════
# At 62.5 MHz the token wall is executor transport: every op above used to be
# (a) a json.loads on the hot path and (b) its own Python→Cython→PCIe access.
# A walked token is 24 programs × ~13k ops — hundreds of thousands of
# per-op round trips. The default path now:
#   1. COMPILES each regops file ONCE into a flat node list (typed tuples) —
#      no json/dict work between PCIe accesses;
#   2. coalesces runs of consecutive `w` lines whose addresses ascend by 4
#      into ONE pci_write_burst call (see ORDERING below);
#   3. batches `pw` FIFO polls: the free-count read is a LOWER BOUND on
#      space, so one read prepays up to PW_BUDGET_CAP pushes (see PW BUDGET);
#   4. (opt-in --cap-gather) reads each run of consecutive `cap` lines via
#      one C-side loop over the mmap'd BAR (pci_get_address + numpy.take).
# SEMANTICS ARE IDENTICAL: every check/poll/cap computes the same predicate
# on the same value in the same order; the WRITE SEQUENCE reaching the CL is
# dword-for-dword the order the file states; the egress bytes and the
# summary contract are unchanged. --selfcheck proves compiled == interpreted
# op-for-op against a mock PCI layer; --legacy keeps the original loop.
#
# ORDERING (why bursts are safe): BAR0 is attached WITHOUT write-combining
# (flags=0 — the OCL bar is not BURST_CAPABLE and must never be WC-mapped:
# WC may merge/reorder). fpga_pci_write_burst on a non-WC handle is a C loop
# of plain 32-bit stores through the same UC mapping pci_poke uses: same
# TLPs, same order, one binding crossing instead of N. Reads (polls) cannot
# pass posted writes on PCIe, exactly as with pokes — no fence is needed.
# Bursts NEVER cross a non-`w` op (and not even a `note`), so every
# read/poll/cap boundary sees the identical write history.
#
# PW BUDGET (why batched pushes are safe): the free count at a stream CSR
# only DECREASES via host writes to that address; the tile consuming records
# only increases it. So an observed free=F prepays F pushes: before each
# one, actual free >= remaining budget >= 1 — the exact predicate the old
# per-push poll established. The budget is invalidated by ANY other write
# (w/jf) to that address (none exist in real programs; checked at compile
# time), capped at PW_BUDGET_CAP, and dropped between files (TILE_RST).
# A stalled FIFO still stalls: the refresh poll has the same 5 s bound and
# prints the same "pw stall" line.
#
# --cap-gather is OPT-IN (default off, and forced off under --stop-on-fail):
# it reads with numpy element loads through pci_get_address instead of
# pci_peek. Same 32-bit reads, same order; flip it on only after one card
# flight diffs its egress against the default path.
#
# --mmio-direct is the OPT-IN endpoint of the same idea for EVERY access:
# a numpy uint32 view over pci_get_address(handle, 0, span) — the very
# pointer peek/poke dereference — with element loads/stores replacing the
# per-access binding crossing (~1-2 us each on the F2 host). Access
# sequence, order and 32-bit width are identical by construction (element
# indexing never widens; slice/memcpy forms are deliberately NOT used).
# Qualify on card by diffing a token's cap egress: default vs --mmio-direct.

import argparse
import json
import sys
import time
from pathlib import Path

POLL_SECONDS = 5.0          # bound per poll/pw/jf wait on real hardware
PW_BUDGET_CAP = 128         # max pushes prepaid by ONE free-count read

# ── per-file TILE_RST (see _tile_reset) ─────────────────────────────────────
TILE_RST_A     = 0x000C     # bridge CSR: bit0 LEVEL-holds tile in reset
TILE_STATUS_A  = 0x1004     # tile CSR STATUS (csr_regs.sv 0x04)
TILE_READY_M   = 0x0000FF03 # busy[15:8] | desc_error[1] | idle[0]
TILE_READY_E   = 0x00000001 # idle=1, desc_error clear, every block idle
TILE_RST_TMO_S = 0.050      # readiness poll bound — a miss is a hard fault
TILE_RST_TICK  = 10e-6      # retry granularity between readiness reads

# ── compiled-program op codes ───────────────────────────────────────────────
OPC_W, OPC_WB, OPC_PW, OPC_JF, OPC_POLL, OPC_CAPRUN, OPC_R, OPC_RN, \
    OPC_BAD, OPC_BADCAP = range(10)

_OPC_NAME = {OPC_W: "w", OPC_WB: "w(burst)", OPC_PW: "pw", OPC_JF: "jf",
             OPC_POLL: "poll", OPC_CAPRUN: "cap", OPC_R: "r", OPC_RN: "rn",
             OPC_BAD: "bad", OPC_BADCAP: "badcap"}


def _wait_clkgen_lock(slot: int, timeout_s: int = 30) -> bool:
    """DECISION-LC-1 bring-up order (AWS_CLK_GEN_spec.md:165): the tile
    clock (clk_extra_a1 — recipe A2 = 15.625 MHz on every image to date, A0
    = 62.5 MHz on the CLOCK_LADDER track; the ORDER rule is identical) is
    HELD STATIC LOW until MMCM lock.
    Touching BAR0 before lock trips apex_ocl_cdc's dead-clock guard, which
    poisons the bridge (every access SLVERR) until reset — BY DESIGN. So:
    poll `fpga-describe-clkgen` until it reports lock, BEFORE any BAR0 op.
    KNOWN VACUOUS as a lock check (the substring matches the tool's own
    header — docs/results/f2_stage2_hw/RESULT.md:64-68); the NUMERIC,
    image-keyed gate lives in remote_hw_exec.check_tile_clock, which
    replaces this preflight on every remote invocation."""
    import subprocess, time
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["fpga-describe-clkgen", "-S", str(slot)],
                capture_output=True, text=True, timeout=10).stdout
            last = out
            if "lock" in out.lower() and "not" not in out.lower():
                print(f"[clkgen] MMCM locked:\n{out.strip()}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            last = str(e)
        time.sleep(1)
    print(f"[clkgen] WARNING: no lock confirmation within {timeout_s}s "
          f"(last: {last.strip()[:200]}) — BAR0 traffic may poison the CDC "
          f"bridge; pass --skip-clkgen-wait only for single-clock CLs")
    return False


# ── transport layer ─────────────────────────────────────────────────────────
class HwIO:
    """The silicon transport: SDK peek/poke plus the two batched entries.

    write_run(a0, data) is ONE binding crossing for a run of consecutive `w`
    lines with ascending +4 addresses — see ORDERING in the header. gather()
    (opt-in) is one C-side element-load loop per cap run."""

    def __init__(self, kit, handle, span_dwords: int = 0):
        self.kit, self.h = kit, handle
        self.n_rd = self.n_wr = self.n_calls = 0
        self._span = span_dwords
        self._view = None

    def read(self, a):
        self.n_rd += 1
        self.n_calls += 1
        return self.kit.pci_peek(self.h, a) & 0xFFFFFFFF

    def write(self, a, d):
        self.n_wr += 1
        self.n_calls += 1
        self.kit.pci_poke(self.h, a, d)

    def write_run(self, a0, data):
        self.n_wr += len(data)
        self.n_calls += 1
        self.kit.pci_write_burst(self.h, a0, data, len(data))

    def _ensure_view(self):
        """numpy uint32 view over the mmap'd BAR (the SAME pointer peek/poke
        dereference — pci_get_address). Element indexing on it is a single
        32-bit UC load/store, never widened."""
        if self._view is None:
            try:
                import ctypes
                import numpy as np
                ptr = self.kit.pci_get_address(self.h, 0, self._span)
                buf = (ctypes.c_uint32 * self._span).from_address(ptr)
                self._view = np.frombuffer(buf, dtype=np.uint32)
                self._np = np
            except Exception as e:      # pragma: no cover - hw-only path
                print(f"[mmio] view unavailable ({e}) — using SDK calls")
                self._view = False
        return self._view

    def gather(self, addrs):
        """Read len(addrs) dwords in file order via ONE numpy take over the
        BAR view (per-element 32-bit loads). Falls back to the peek loop."""
        self.n_rd += len(addrs)
        self.n_calls += 1
        v = self._ensure_view()
        if v is False:
            return [self.kit.pci_peek(self.h, a) & 0xFFFFFFFF for a in addrs]
        idx = self._np.fromiter((a >> 2 for a in addrs), dtype=self._np.int64,
                                count=len(addrs))
        return [int(x) for x in v.take(idx)]

    def activate_direct(self) -> bool:
        """--mmio-direct: swap every accessor to element ops on the BAR view,
        removing the per-access binding crossing entirely. Access order,
        width (32-bit) and count are IDENTICAL to the SDK path — x86 UC
        stores/loads through the same mapping, retired in program order.
        n_calls then counts binding crossings only (i.e. stops advancing)."""
        v = self._ensure_view()
        if v is False:
            return False
        np_ = self._np
        me = self

        def read(a):
            me.n_rd += 1
            return int(v[a >> 2])

        def write(a, d):
            me.n_wr += 1
            v[a >> 2] = d & 0xFFFFFFFF

        def write_run(a0, data):
            me.n_wr += len(data)
            i = a0 >> 2
            for d in data:              # per-element: never a wide store
                v[i] = d
                i += 1

        def gather(addrs):
            me.n_rd += len(addrs)
            idx = np_.fromiter((a >> 2 for a in addrs), dtype=np_.int64,
                               count=len(addrs))
            return [int(x) for x in v.take(idx)]

        self.read, self.write = read, write
        self.write_run, self.gather = write_run, gather
        return True


class _CapEgress:
    """The capture-egress channel: format is contract, see header. The
    global sequence `i` advances for every executed cap even with no file
    (parity with the original cap_i)."""

    def __init__(self, fp):
        self.fp = fp
        self.i = 0

    def emit(self, tag, a, m, v):
        if self.fp is not None:
            self.fp.write('{"tag":"%s","addr":"0x%08x","mask":"0x%08x",'
                          '"value":%d,"i":%d}\n' % (tag, a, m, v, self.i))
        self.i += 1


# ── the compiler ────────────────────────────────────────────────────────────
def compile_file(path):
    """Parse a regops.jsonl ONCE into the flat node list executed by
    exec_compiled — no json/dict work remains on the transport path.

    Node tuples (n[1] is the value the `ops` counter must hold when the node
    ABORTS, i.e. the count of op lines — notes included — through the node's
    aborting line; CAPRUN carries the count BEFORE its first cap instead):
      (OPC_W,    ops_after, lineno, a, d, inval)
      (OPC_WB,   ops_after, lineno0, a0, [d...], inval)   consecutive `w`
                 lines, addresses ascending by exactly 4, never crossing any
                 other line (not even `note`)
      (OPC_PW,   ops_after, lineno, a, d)
      (OPC_JF,   ops_after, lineno, a, d)
      (OPC_POLL, ops_after, lineno, a, m, e)
      (OPC_CAPRUN, ops_before, [(lineno, a, m, tag, e, has_e), ...])
      (OPC_R,    ops_after, lineno, a, m, e)
      (OPC_RN,   ops_after, lineno, a, m, None)
      (OPC_BAD,  ops_after, lineno, opname)   unknown op — abort HERE
      (OPC_BADCAP, ops_after, lineno, a)      cap without tag — abort HERE
    inval: tuple of pw-FIFO addresses this write touches (budget
    invalidation; empty in every real program to date).

    Returns (nodes, total_ops, census) — census maps op name -> line count.
    """
    # One extraction pass. The whole file parses as ONE json array (a single
    # C call, ~3x the per-line loads); everything after runs over parallel
    # lists of ints/strings, not dicts.
    lns, keep = [], []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if line:
            lns.append(lineno)
            keep.append(line)
    if keep:
        try:                          # ~5x the stdlib parse when present
            import orjson
            raw = orjson.loads(("[" + ",".join(keep) + "]").encode())
        except ImportError:
            raw = json.loads("[" + ",".join(keep) + "]")
    else:
        raw = []
    del keep
    opl = [d["op"] for d in raw]
    al = [d.get("a", 0) for d in raw]
    census = {}
    for op in opl:
        census[op] = census.get(op, 0) + 1

    pw_addrs = {al[k] for k in range(len(opl)) if opl[k] == "pw"}

    nodes = []
    count = 0                         # the running `ops` counter value
    i, n = 0, len(raw)
    while i < n:
        op = opl[i]
        lineno = lns[i]
        a = al[i]
        d = raw[i]
        if op == "note":
            count += 1
            i += 1
            continue
        if op == "w":
            j = i                     # extend over consecutive `w` LINES
            ln, aa = lineno, a
            while (j + 1 < n and opl[j + 1] == "w"
                   and lns[j + 1] == ln + 1 and al[j + 1] == aa + 4):
                j += 1
                ln += 1
                aa += 4
            if j > i:
                data = [raw[k]["d"] & 0xFFFFFFFF for k in range(i, j + 1)]
                inval = tuple(x for x in al[i:j + 1] if x in pw_addrs) \
                    if pw_addrs else ()
                count += j - i + 1
                nodes.append((OPC_WB, count, lineno, a, data, inval))
                i = j + 1
            else:
                count += 1
                inval = (a,) if a in pw_addrs else ()
                nodes.append((OPC_W, count, lineno, a,
                              d["d"] & 0xFFFFFFFF, inval))
                i += 1
        elif op == "cap":
            if "tag" not in d:
                count += 1
                nodes.append((OPC_BADCAP, count, lineno, a))
                i += 1
                continue
            j = i                     # run of consecutive tagged cap LINES
            ln = lineno
            while (j + 1 < n and opl[j + 1] == "cap"
                   and lns[j + 1] == ln + 1 and "tag" in raw[j + 1]):
                j += 1
                ln += 1
            entries = []
            for k in range(i, j + 1):
                dd = raw[k]
                m = dd.get("m", 0)    # sim default when absent: 0
                entries.append((lns[k], al[k], m, dd["tag"],
                                dd.get("e"), "e" in dd))
            nodes.append((OPC_CAPRUN, count, entries))
            count += j - i + 1
            i = j + 1
        elif op == "pw":
            count += 1
            nodes.append((OPC_PW, count, lineno, a, d["d"] & 0xFFFFFFFF))
            i += 1
        elif op == "jf":
            count += 1
            nodes.append((OPC_JF, count, lineno, a, d["d"] & 0xFFFFFFFF))
            i += 1
        elif op == "poll":
            count += 1
            nodes.append((OPC_POLL, count, lineno, a, d["m"], d["e"]))
            i += 1
        elif op == "r":
            count += 1
            nodes.append((OPC_R, count, lineno, a, d["m"], d["e"]))
            i += 1
        elif op == "rn":
            count += 1
            nodes.append((OPC_RN, count, lineno, a, d["m"], None))
            i += 1
        else:
            count += 1
            nodes.append((OPC_BAD, count, lineno, op))
            i += 1
    return nodes, count, census


# ── the executors (return identical tuples; print identical lines) ──────────
def exec_interpreted(path, io, stop_on_fail, cap):
    """The ORIGINAL per-line loop, verbatim semantics (kept as the --legacy
    path and as the selfcheck reference). Returns
    (ops, checks, fails, captured, cap_checked, cap_mismatch, aborted)."""
    def rd(a):
        return io.read(a)

    def wr(a, d):
        io.write(a, d & 0xFFFFFFFF)

    checks = fails = ops = 0
    captured = {}                 # tag -> [values the TILE produced]
    cap_checked = cap_mismatch = 0   # bring-up gate only (see `cap`)
    aborted = False
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        op = d["op"]
        a = d.get("a", 0)
        ops += 1
        if op == "note":
            continue
        if op == "w":
            wr(a, d["d"])
        elif op == "pw":
            t0 = time.time()
            while not ((rd(a) >> 9) & 0x7FFF):
                if time.time() - t0 > POLL_SECONDS:
                    print(f"pw stall L{lineno} @{a:#06x}")
                    aborted = True
                    break
            if aborted:
                break
            wr(a, d["d"])
        elif op == "jf":
            t0 = time.time()
            while not (rd(a) & 1):
                if time.time() - t0 > POLL_SECONDS:
                    print(f"jf stall L{lineno} @{a:#06x}")
                    aborted = True
                    break
            if aborted:
                break
            wr(a, d["d"])
        elif op == "poll":
            t0 = time.time()
            while (rd(a) & d["m"]) != d["e"]:
                if time.time() - t0 > POLL_SECONDS:
                    print(f"poll stall L{lineno} @{a:#06x}: "
                          f"last {rd(a):#010x} want {d['e']:#x}/{d['m']:#x}")
                    fails += 1
                    aborted = True
                    break
            if aborted:
                break
        elif op == "cap":
            # READ AND KEEP. The only op in this vocabulary that makes a
            # tile-produced number reach a host variable — see
            # docs/design/PROMPT_ON_CHIP.md §0.
            #
            # `tag` is MANDATORY (same in sim_main.cpp): it is the value's
            # only identity — an untagged capture is an anonymous number
            # the grader cannot place.
            if "tag" not in d:
                print(f"cap without tag L{lineno} @{a:#06x}")
                aborted = True
                break
            m = d.get("m", 0)               # sim default when absent: 0
            v = rd(a) & m
            captured.setdefault(d["tag"], []).append(v)
            cap.emit(d["tag"], a, m, v)
            if "e" in d:
                # STAGE-A SEMANTIC CHANGE (D-P1): a cap that CARRIES an
                # expectation and misses it now counts as a FAILURE, so
                # the run's exit code / verdict goes red. Before this, a
                # mismatch only printed and the process still exited 0 —
                # a capture path could be wrong and every gate green.
                # A cap WITHOUT `e` still NEVER fails (that is Milestone
                # B: capture where no expectation exists).
                # Both sides masked — PARITY with sim_main.cpp, and the
                # emitter therefore need not pre-mask `e`.
                cap_checked += 1
                if v != (d["e"] & m):
                    cap_mismatch += 1
                    fails += 1
                    print(f"CAPMISMATCH L{lineno}: [{a:#06x}] tag={d['tag']} "
                          f"captured {v:#010x} != baked {d['e'] & m:#010x}")
                    if stop_on_fail:
                        aborted = True
                        break

        elif op in ("r", "rn"):
            v = rd(a)
            checks += 1
            if op == "r":
                ok = (v & d["m"]) == d["e"]
                if not ok:
                    fails += 1
                    print(f"FAIL L{lineno}: [{a:#06x}] got {v:#010x} "
                          f"want {d['e']:#010x} (mask {d['m']:#x})")
            else:                                    # rn
                if (v & d["m"]) == 0:
                    fails += 1
                    print(f"FAIL L{lineno}: [{a:#06x}] got {v:#010x} "
                          f"expected nonzero under {d['m']:#x}")
            if fails and stop_on_fail:
                aborted = True
                break
        else:
            print(f"unknown op '{op}' L{lineno}")
            aborted = True
            break
    return ops, checks, fails, captured, cap_checked, cap_mismatch, aborted


def exec_compiled(nodes, total_ops, io, stop_on_fail, cap, *,
                  use_burst=True, use_gather=False,
                  budget_cap=PW_BUDGET_CAP, times=None):
    """Execute one compiled program. Same return tuple, same printed lines,
    same access ORDER as exec_interpreted — fewer binding crossings and
    fewer FIFO free-count reads (see PW BUDGET in the header).

    times: None, or a dict op-code -> [seconds, ops, reads, writes] to
    accumulate --time-report samples into."""
    rd, wr = io.read, io.write
    checks = fails = 0
    captured = {}
    cap_checked = cap_mismatch = 0
    aborted = False
    ops = 0
    budget = {}                   # pw addr -> prepaid free-slot lower bound
    perf = time.perf_counter if times is not None else None

    for node in nodes:
        code = node[0]
        if perf is not None:
            t0 = perf()
            rd0, wr0 = io.n_rd, io.n_wr
        if code == OPC_W:
            ops = node[1]
            a = node[3]
            wr(a, node[4])
            if node[5]:
                for x in node[5]:
                    budget[x] = 0
        elif code == OPC_WB:
            ops = node[1]
            if use_burst:
                io.write_run(node[3], node[4])
            else:
                a = node[3]
                for d in node[4]:
                    wr(a, d)
                    a += 4
            if node[5]:
                for x in node[5]:
                    budget[x] = 0
        elif code == OPC_PW:
            ops = node[1]
            a = node[3]
            if budget.get(a, 0) <= 0:
                t0p = time.time()
                free = (rd(a) >> 9) & 0x7FFF
                while not free:
                    if time.time() - t0p > POLL_SECONDS:
                        print(f"pw stall L{node[2]} @{a:#06x}")
                        aborted = True
                        break
                    free = (rd(a) >> 9) & 0x7FFF
                if aborted:
                    break
                budget[a] = min(free, budget_cap)
            wr(a, node[4])
            budget[a] -= 1
        elif code == OPC_JF:
            ops = node[1]
            a = node[3]
            t0p = time.time()
            while not (rd(a) & 1):
                if time.time() - t0p > POLL_SECONDS:
                    print(f"jf stall L{node[2]} @{a:#06x}")
                    aborted = True
                    break
            if aborted:
                break
            wr(a, node[4])
        elif code == OPC_POLL:
            ops = node[1]
            _, _, lineno, a, m, e = node
            t0p = time.time()
            while (rd(a) & m) != e:
                if time.time() - t0p > POLL_SECONDS:
                    print(f"poll stall L{lineno} @{a:#06x}: "
                          f"last {rd(a):#010x} want {e:#x}/{m:#x}")
                    fails += 1
                    aborted = True
                    break
            if aborted:
                break
        elif code == OPC_CAPRUN:
            base = node[1]
            entries = node[2]
            ops = base + len(entries)
            if use_gather and len(entries) > 1:
                vals = io.gather([en[1] for en in entries])
            else:
                vals = None
            for j, (lineno, a, m, tag, e, has_e) in enumerate(entries):
                v = (vals[j] if vals is not None else rd(a)) & m
                captured.setdefault(tag, []).append(v)
                cap.emit(tag, a, m, v)
                if has_e:
                    cap_checked += 1
                    if v != (e & m):
                        cap_mismatch += 1
                        fails += 1
                        print(f"CAPMISMATCH L{lineno}: [{a:#06x}] tag={tag} "
                              f"captured {v:#010x} != baked {e & m:#010x}")
                        if stop_on_fail:
                            ops = base + j + 1
                            aborted = True
                            break
            if aborted:
                break
        elif code == OPC_R:
            ops = node[1]
            _, _, lineno, a, m, e = node
            v = rd(a)
            checks += 1
            if (v & m) != e:
                fails += 1
                print(f"FAIL L{lineno}: [{a:#06x}] got {v:#010x} "
                      f"want {e:#010x} (mask {m:#x})")
            if fails and stop_on_fail:
                aborted = True
                break
        elif code == OPC_RN:
            ops = node[1]
            _, _, lineno, a, m, _e = node
            v = rd(a)
            checks += 1
            if (v & m) == 0:
                fails += 1
                print(f"FAIL L{lineno}: [{a:#06x}] got {v:#010x} "
                      f"expected nonzero under {m:#x}")
            if fails and stop_on_fail:
                aborted = True
                break
        elif code == OPC_BADCAP:
            ops = node[1]
            print(f"cap without tag L{node[2]} @{node[3]:#06x}")
            aborted = True
            break
        else:                                        # OPC_BAD
            ops = node[1]
            print(f"unknown op '{node[3]}' L{node[2]}")
            aborted = True
            break
        if perf is not None:
            acc = times.setdefault(code, [0.0, 0, 0, 0])
            acc[0] += perf() - t0
            acc[1] += 1
            acc[2] += io.n_rd - rd0
            acc[3] += io.n_wr - wr0
    if not aborted:
        ops = total_ops
    elif perf is not None:
        acc = times.setdefault(code, [0.0, 0, 0, 0])
        acc[0] += perf() - t0
        acc[1] += 1
        acc[2] += io.n_rd - rd0
        acc[3] += io.n_wr - wr0
    return ops, checks, fails, captured, cap_checked, cap_mismatch, aborted


def _print_time_report(times, census_total, io, wall_s):
    print("F2HOST TIMES: class      nodes     reads    writes   seconds")
    for code in sorted(times, key=lambda c: -times[c][0]):
        s, nn, nr, nw = times[code]
        print(f"F2HOST TIMES: {_OPC_NAME[code]:<9s} {nn:9d} {nr:9d} "
              f"{nw:9d}   {s:7.3f}")
    print(f"F2HOST TIMES: total ops={census_total} pci_reads={io.n_rd} "
          f"pci_writes={io.n_wr} binding_calls={io.n_calls} "
          f"wall={wall_s:.3f}s")


def _print_file_times(file_times, legacy):
    """--time-report per-file PHASE rows — attributes each file's wall to
    reset (the TILE_RST pulse + readiness poll; polls = STATUS reads, 0
    under --legacy-rst-sleep), the op loop, and cap-drain (time spent in
    the file's cap reads, i.e. draining tile results — sliced from the
    op-class accumulator, so '-' under --legacy, which has no per-class
    attribution). The row set is what lets a card run name the residual
    per-file cost instead of guessing at it."""
    print("F2HOST FILETIMES:   reset_s  polls  oploop_s  capdrain_s  file")
    t_rst = t_exec = t_cap = 0.0
    for path, rst_s, polls, exec_s, cap_s in file_times:
        cd = "         -" if legacy else f"{cap_s:10.4f}"
        print(f"F2HOST FILETIMES: {rst_s:9.6f} {polls:6d} {exec_s:9.4f} "
              f"{cd}  {Path(path).name}")
        t_rst += rst_s
        t_exec += exec_s
        t_cap += cap_s
    cd = "-" if legacy else f"{t_cap:.4f}s"
    print(f"F2HOST FILETIMES: total files={len(file_times)} "
          f"reset={t_rst:.6f}s oploop={t_exec:.4f}s capdrain={cd}")


# ── selfcheck: compiled vs interpreted on a MOCK pci layer ──────────────────
class MockIO:
    """Records every dword access; models FIFOs, poll scripts and static
    registers. write_run expands to the identical per-dword write events, so
    trace equality IS the ordering proof."""

    def __init__(self, fifo=(), fifo_cap=4, fifo_drain_on_read=True,
                 regs=None, poll_script=None, ready=(), never_ready=()):
        self.trace = []               # ('r', a) / ('w', a, d) in access order
        self.wtrace = []              # writes only — the hw-mutation trace
        self.n_rd = self.n_wr = self.n_calls = 0
        self.fifo = {a: 0 for a in fifo}          # addr -> occupancy
        self.fifo_cap = fifo_cap
        self.fifo_drain = fifo_drain_on_read
        self.regs = dict(regs or {})
        self.poll_script = {a: list(v) for a, v in (poll_script or {}).items()}
        self.ready = set(ready)
        self.never_ready = set(never_ready)
        self.overflow = 0             # pushes observed with no space
        self.tile_rst = 0             # TILE_RST level (bridge CSR 0x000C)

    def read(self, a):
        self.n_rd += 1
        self.n_calls += 1
        self.trace.append(("r", a))
        if a in self.fifo:
            if self.fifo_drain and self.fifo[a] > 0:
                self.fifo[a] -= 1                 # the tile consumed one
            return ((self.fifo_cap - self.fifo[a]) & 0x7FFF) << 9
        if a in self.never_ready:
            return 0
        if a in self.poll_script:
            q = self.poll_script[a]
            return q.pop(0) if q else 0xFFFFFFFF
        if a in self.ready:
            return 0xFFFFFFFF
        if a == TILE_STATUS_A and a not in self.regs:
            # reset-readiness model (mirrors cl_apex.sv + csr_regs.sv):
            # STATUS.idle==1 with busy/desc_error clear once TILE_RST is
            # deasserted; 0 (not ready) while the tile is held in reset.
            # An explicit regs/poll_script/ready entry for 0x1004 wins —
            # fixtures that script STATUS keep their behavior.
            return 0 if self.tile_rst else TILE_READY_E

        return self.regs.get(a, 0)

    def write(self, a, d):
        self.n_wr += 1
        self.n_calls += 1
        self.trace.append(("w", a, d & 0xFFFFFFFF))
        self.wtrace.append((a, d & 0xFFFFFFFF))
        if a == TILE_RST_A:
            self.tile_rst = d & 1
        if a in self.fifo:
            if self.fifo[a] >= self.fifo_cap:
                self.overflow += 1                # the invariant the budget
            self.fifo[a] += 1                     # math must never break

    def write_run(self, a0, data):
        assert len(data) >= 2, "burst of <2 dwords"
        for k, d in enumerate(data):
            self.write(a0 + 4 * k, d)
        self.n_calls -= len(data) - 1             # count as one crossing

    def gather(self, addrs):
        return [self.read(a) for a in addrs]


def _run_pair(regops_path, mkmock, stop_on_fail=False, use_gather=False,
              budget_cap=PW_BUDGET_CAP):
    """Run interpreted and compiled on twin mocks; return both result dicts."""
    import io as _io
    import contextlib
    out = []
    for which in ("interp", "compiled"):
        mock = mkmock()
        cap_lines = _io.StringIO()
        cap = _CapEgress(cap_lines)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            if which == "interp":
                r = exec_interpreted(regops_path, mock, stop_on_fail, cap)
            else:
                nodes, total, _ = compile_file(regops_path)
                r = exec_compiled(nodes, total, mock, stop_on_fail, cap,
                                  use_burst=True, use_gather=use_gather,
                                  budget_cap=budget_cap)
        out.append({"res": r, "mock": mock, "egress": cap_lines.getvalue(),
                    "stdout": buf.getvalue(), "cap_n": cap.i})
    return out


def _sc_fixture():
    """The base selfcheck program + register model — ONE source, shared by
    --selfcheck (in-process, compiled vs interpreted) and
    --selfcheck-server (single-shot CLI vs the --server transport).
    Returns (J, F, base_lines, regs)."""
    def J(**kw):
        return json.dumps(kw)

    F = 0x3000                                   # the mock stream FIFO
    base = [J(op="note", s="selfcheck"),
            J(op="w", a=0x1000, d=1),
            J(op="w", a=0x2000, d=0xAAAA0000),   # contiguous pair -> burst
            J(op="w", a=0x2004, d=0xAAAA0001),
            J(op="note", s="a note must BREAK the burst run"),
            J(op="w", a=0x2008, d=0xAAAA0002),
            J(op="w", a=0x2010, d=3),            # gap -> two singles
            J(op="w", a=0x2100, d=1), J(op="w", a=0x2104, d=2),
            J(op="w", a=0x2108, d=3), J(op="w", a=0x210C, d=4),
            J(op="w", a=0x2110, d=5),            # 5-dword burst
            J(op="poll", a=0x200C, m=1, e=1),
            J(op="pw", a=F, d=0x11), J(op="pw", a=F, d=0x12),
            J(op="pw", a=F, d=0x13), J(op="pw", a=F, d=0x14),
            J(op="pw", a=F, d=0x15), J(op="pw", a=F, d=0x16),
            J(op="pw", a=F, d=0x17),             # 7 pushes through cap-4 FIFO
            J(op="jf", a=0x3100, d=1),
            J(op="cap", a=0x3204, m=0xFFFFFFFF, tag="ro_w0_0"),
            J(op="cap", a=0x3208, m=0xFFFF, tag="ro_w1_1"),
            J(op="cap", a=0x320C, m=0xFFFFFFFF, tag="ro_w2_2", e=0xBEEF),
            J(op="cap", a=0x3210, m=0xFFFFFFFF, tag="ro_w3_3", e=0xDEAD),
            J(op="r", a=0x3204, m=0xFFFF, e=0xBEEF),
            J(op="r", a=0x3204, m=0xFFFF, e=0x1111),      # counted FAIL
            J(op="rn", a=0x3208, m=0xFF), J(op="rn", a=0x1FFC, m=0xFF)]

    regs = {0x3204: 0xBEEF, 0x3208: 0x40, 0x320C: 0xBEEF, 0x3210: 0xDEAD,
            0x1FFC: 0}
    return J, F, base, regs


def _selfcheck(extra_files=()) -> int:
    """Prove compiled == interpreted op-for-op on a mock pci layer:
    the WRITE trace (every dword reaching the CL, in order) must be
    IDENTICAL; every observable (ops/checks/fails/captures/egress
    bytes/abort) must be IDENTICAL; the FIFO occupancy must never exceed
    capacity under the pw budget. Read COUNT is allowed to shrink (that is
    the point); read ORDER of non-pw classes is covered by the value model
    (a differing order would produce differing values, checks and egress).
    """
    global POLL_SECONDS
    import tempfile, os
    fails = 0
    saved_poll = POLL_SECONDS

    J, F, base, regs = _sc_fixture()

    def scenario(name, lines, mkmock, stop_on_fail=False, use_gather=False,
                 budget_cap=PW_BUDGET_CAP, expect_abort=None,
                 expect_overflow=0):
        nonlocal fails
        fd, p = tempfile.mkstemp(prefix="apex_sc_", suffix=".jsonl")
        os.close(fd)
        Path(p).write_text("\n".join(lines) + "\n")
        try:
            a, b = _run_pair(p, mkmock, stop_on_fail, use_gather, budget_cap)
        finally:
            os.unlink(p)
        bad = []
        if a["mock"].wtrace != b["mock"].wtrace:
            for k, (x, y) in enumerate(zip(a["mock"].wtrace,
                                           b["mock"].wtrace)):
                if x != y:
                    bad.append(f"write trace diverges at #{k}: {x} vs {y}")
                    break
            else:
                bad.append(f"write trace length {len(a['mock'].wtrace)} vs "
                           f"{len(b['mock'].wtrace)}")
        if a["res"] != b["res"]:
            bad.append(f"result tuple {a['res']} vs {b['res']}")
        if a["egress"] != b["egress"]:
            bad.append("cap egress bytes differ")
        if a["cap_n"] != b["cap_n"]:
            bad.append(f"cap_i {a['cap_n']} vs {b['cap_n']}")
        if a["stdout"] != b["stdout"]:
            bad.append(f"stdout differs:\n--interp--\n{a['stdout']}"
                       f"--compiled--\n{b['stdout']}")
        for w, r in (("interp", a), ("compiled", b)):
            if r["mock"].overflow != expect_overflow:
                bad.append(f"{w}: {r['mock'].overflow} FIFO pushes with no "
                           f"space (expected {expect_overflow})")
            if expect_abort is not None and r["res"][6] != expect_abort:
                bad.append(f"{w}: aborted={r['res'][6]} expected "
                           f"{expect_abort}")
        rr = (f"reads {a['mock'].n_rd}->{b['mock'].n_rd} "
              f"calls {a['mock'].n_calls}->{b['mock'].n_calls}")
        if bad:
            fails += 1
            print(f"  [{name}] FAIL  ({rr})")
            for x in bad:
                print(f"      {x}")
        else:
            print(f"  [{name}] ok    ops={a['res'][0]} fails={a['res'][2]} "
                  f"{rr}")

    def mk(**kw):
        d = dict(fifo=(F,), fifo_cap=4, regs=regs,
                 poll_script={0x200C: [0, 0, 1]}, ready=(0x3100,))
        d.update(kw)
        return lambda: MockIO(**d)

    print("SELFCHECK 1: happy path (bursts, pw budget, cap run, fails "
          "counted)")
    scenario("1a default", base, mk(), expect_abort=False)
    scenario("1b gather", base, mk(), use_gather=True, expect_abort=False)
    scenario("1c budget=1 (poll per push, legacy read pattern)", base, mk(),
             budget_cap=1, expect_abort=False)

    print("SELFCHECK 2: abort parity")
    scenario("2a unknown op mid-file",
             base[:6] + [J(op="zz", a=0)] + base[6:], mk(),
             expect_abort=True)
    scenario("2b cap without tag",
             base[:12] + [J(op="cap", a=0x3204, m=1)] + base[12:], mk(),
             expect_abort=True)
    scenario("2c stop-on-fail r", base, mk(), stop_on_fail=True,
             expect_abort=True)
    scenario("2d stop-on-fail cap mismatch",
             [J(op="cap", a=0x3204, m=0xFFFFFFFF, tag="x_0", e=1),
              J(op="w", a=0x1000, d=1)], mk(), stop_on_fail=True,
             expect_abort=True)
    POLL_SECONDS = 0.05
    try:
        scenario("2e poll stall",
                 [J(op="w", a=0x1000, d=1),
                  J(op="poll", a=0x1FF0, m=1, e=1),
                  J(op="w", a=0x1004, d=2)],
                 mk(never_ready=(0x1FF0,)), expect_abort=True)
        scenario("2f pw stall (FIFO wedged full)",
                 [J(op="pw", a=F, d=k) for k in range(6)],
                 mk(fifo_drain_on_read=False), expect_abort=True)
    finally:
        POLL_SECONDS = saved_poll

    for path in extra_files:
        print(f"SELFCHECK 3: real-file trace diff: {path}")
        nodes, _tot, _c = compile_file(path)
        pw_a = set()
        polls = {}
        for nd in nodes:
            if nd[0] == OPC_PW:
                pw_a.add(nd[3])
            elif nd[0] == OPC_POLL:
                polls.setdefault(nd[3], []).append(
                    nd[5] if (nd[5] & ~nd[4]) == 0 else (nd[5] & nd[4]))
        jf_a = {nd[3] for nd in nodes if nd[0] == OPC_JF}
        assert not (pw_a & set(polls)) and not (pw_a & jf_a), \
            "pw FIFO addrs overlap poll/jf addrs — permissive mock invalid"

        def mkreal():
            return MockIO(fifo=tuple(pw_a), fifo_cap=PW_BUDGET_CAP,
                          poll_script={a: list(v) for a, v in polls.items()},
                          ready=tuple(jf_a))
        scenario(Path(path).name, Path(path).read_text().splitlines(),
                 mkreal)

    print(f"F2HOST SELFCHECK: {'FAIL' if fails else 'PASS'} (fails={fails})")
    return 1 if fails else 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface — module-level because --server re-parses every
    job's argv with THIS parser (one flag surface, nothing to drift)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("regops", nargs="*", help="regops.jsonl files")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="abort a file at its first failing check")
    ap.add_argument("--skip-clkgen-wait", action="store_true",
                    help="skip the MMCM-lock preflight (single-clock CLs "
                         "only — the LC-1 two-clock CL REQUIRES the wait)")
    ap.add_argument("--cap-out", metavar="PATH",
                    help="write one JSON line per executed `cap` op to PATH "
                         "(truncated at run start) — the capture-egress "
                         "channel; same format as the sim's +cap_out=")
    ap.add_argument("--legacy", action="store_true",
                    help="run the original per-line interpreter (no "
                         "compile, no bursts, no pw budget)")
    ap.add_argument("--no-burst", action="store_true",
                    help="compiled path, but pci_poke per write (no "
                         "pci_write_burst coalescing)")
    ap.add_argument("--cap-gather", action="store_true",
                    help="OPT-IN: read cap runs via one C-side loop over "
                         "the mmap'd BAR (pci_get_address + numpy) instead "
                         "of per-op pci_peek; forced off by --stop-on-fail")
    ap.add_argument("--mmio-direct", action="store_true",
                    help="OPT-IN: every access as a 32-bit numpy element "
                         "load/store on the mmap'd BAR (pci_get_address) — "
                         "no per-access binding crossing; same accesses, "
                         "same order, same widths as the SDK path")
    ap.add_argument("--pw-budget", type=int, default=PW_BUDGET_CAP,
                    metavar="N",
                    help="max stream pushes prepaid by one free-count read "
                         "(1 = poll before every push, the legacy pattern; "
                         f"default {PW_BUDGET_CAP})")
    ap.add_argument("--legacy-rst-sleep", action="store_true",
                    help="per-file TILE_RST settles by the pre-sleepcut "
                         "2 x 10 ms open-loop sleeps instead of the STATUS "
                         "readiness poll (revert/compare knob; the poll is "
                         "the default — see _tile_reset)")
    ap.add_argument("--time-report", action="store_true",
                    help="print per-op-class totals (ops, pci reads/writes, "
                         "seconds) after the run summary, plus per-file "
                         "phase rows (reset / op loop / cap drain)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="no hardware: prove compiled == interpreted "
                         "op-for-op on a mock pci layer (any regops given "
                         "are also trace-diffed) and exit")
    ap.add_argument("--server", action="store_true",
                    help="PERSISTENT EXECUTOR: attach the card ONCE, then "
                         "execute one job per stdin JSON line "
                         '({"argv": [...]} -> {"rc": ..., "log": ...}); '
                         "stdout carries protocol lines only (every job "
                         "print comes back in the reply's log). "
                         "tile_exec_bridge starts/reuses this for "
                         "executor='hw' — the per-invocation "
                         "python+clkgen+attach cost is paid once per "
                         "session instead of once per job")
    ap.add_argument("--selfcheck-server", action="store_true",
                    help="no hardware: prove the --server transport is "
                         "observably IDENTICAL to the single-shot CLI on "
                         "the mock pci layer (rc, job log, cap egress "
                         "bytes, dword write trace; plus a two-job server "
                         "reuse gate) and exit")
    return ap


# ── the attach seam (single-shot: per run; --server: once, then amortized) ──
def _attach(args):
    """kit + BAR0 handle: SDK import, LC-1 clkgen preflight, pci_attach —
    the whole per-invocation fixed cost. Shared verbatim by the single-shot
    CLI and --server (which calls it exactly once and amortizes it over
    every job). APEX_F2HOST_FAKE_PCI=<spec.json> swaps in the mock-pci kit
    (selfcheck machinery, never a hardware path). Raises SystemExit(2)
    with the same ABORT line as before when the bindings are missing."""
    import os
    fake = os.environ.get("APEX_F2HOST_FAKE_PCI")
    if fake:
        kit = _FakeKit(fake)
        return kit, kit.pci_attach(args.slot, 0, 0, 0)
    # The kit's real python surface (validated on hardware 2026-07-22) is
    # fpga_pci_wrapper.FpgaPCI from sdk/userspace/cython_bindings — built by
    # sdk_setup.sh. Adapter kept inline so the runner works from a clone.
    for _kit_root in (os.environ.get("AWS_FPGA_REPO_DIR", ""),
                      os.path.expanduser("~/aws-fpga"),
                      "/home/ubuntu/aws-fpga"):   # sudo: ~ is /root
        _p = os.path.join(_kit_root, "sdk/userspace/cython_bindings")
        if _kit_root and os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    try:
        from fpga_pci_wrapper import FpgaPCI as _KitPci
    except ImportError:
        print("ABORT: fpga_pci bindings not built — source "
              "aws-fpga/sdk_setup.sh (builds cython_bindings)")
        raise SystemExit(2)
    if not args.skip_clkgen_wait:
        _wait_clkgen_lock(args.slot)
    kit = _KitPci()
    handle = kit.pci_attach(args.slot, 0, 0, 0)   # AppPF BAR0 = OCL, no WC
    return kit, handle


class _FakeKit:
    """APEX_F2HOST_FAKE_PCI=<spec.json>: the mock pci layer behind the
    kit's own API (pci_attach/peek/poke/write_burst), so the --server
    transport can be proven observably identical to the single-shot CLI
    with zero hardware. The spec mirrors MockIO's knobs (address keys are
    decimal-or-'0x' strings); 'poll_seconds' shortens the stall bound the
    same way _selfcheck patches POLL_SECONDS. APEX_F2HOST_FAKE_TRACE=<p>
    dumps the accumulated dword WRITE trace at process exit (= the job
    boundary for a single-shot run); --server instead slices the trace per
    job and returns it in the reply ('fake_wtrace')."""

    def __init__(self, spec_path):
        global POLL_SECONDS
        spec = json.loads(Path(spec_path).read_text())
        if "poll_seconds" in spec:
            POLL_SECONDS = float(spec["poll_seconds"])

        def ai(x):
            return int(x, 0) if isinstance(x, str) else int(x)
        self.mock = MockIO(
            fifo=tuple(ai(a) for a in spec.get("fifo", ())),
            fifo_cap=int(spec.get("fifo_cap", 4)),
            fifo_drain_on_read=bool(spec.get("fifo_drain_on_read", True)),
            regs={ai(a): v for a, v in spec.get("regs", {}).items()},
            poll_script={ai(a): list(v)
                         for a, v in spec.get("poll_script", {}).items()},
            ready=tuple(ai(a) for a in spec.get("ready", ())),
            never_ready=tuple(ai(a) for a in spec.get("never_ready", ())))
        import atexit
        import os
        tr = os.environ.get("APEX_F2HOST_FAKE_TRACE")
        if tr:
            atexit.register(lambda: Path(tr).write_text(
                json.dumps([list(w) for w in self.mock.wtrace])))

    def pci_attach(self, slot, pf, bar, flags):
        return 0

    def pci_peek(self, h, a):
        return self.mock.read(a)

    def pci_poke(self, h, a, d):
        self.mock.write(a, d)

    def pci_write_burst(self, h, a0, data, n):
        self.mock.write_run(a0, list(data)[:n])

    def pci_get_address(self, h, off, span):
        raise RuntimeError("fake kit models registers, not a BAR mapping")


def _tile_reset(io, path, legacy_sleep: bool):
    """Per-file TILE_RST: drop tile/mailbox/fuel-CSR state before a regops
    file (executor parity with sim_main.cpp reset(); found necessary on
    silicon 2026-07-22 — see the call site).

    RESET SEMANTICS (cl_apex.sv): TILE_RST 0x000C bit0 is a bridge-level
    LEVEL register — rst_n = rst_tile_n & ~tile_rst_q for the tile, the
    mailbox and the fuel CSR block; the bridge itself stays live. Both
    writes commit in the tile-clock domain BEFORE their AXI response
    returns through the single-outstanding apex_ocl_cdc, and host UC
    accesses are ordered, so the assert is in force when write(1) returns
    and the pulse width is the full write(1)-response + write(0)-request
    CDC round trip — thousands of tile clocks, vastly more than the one
    posedge a synchronous reset needs. No MMCM/clock is touched (lock is
    an attach-time precondition, _wait_clkgen_lock).

    READINESS: reset-done is OBSERVABLE, not a wait. Tile CSR STATUS
    (0x1004, csr_regs.sv 0x04) reads idle=~|block_busy_i combinationally
    (bit0), live per-block busy in [15:8], and the desc_error sticky in
    bit1 — which only this hard reset clears, so (STATUS & 0xFF03) ==
    0x0001 proves the reset LANDED and the tile is quiescent. FUEL_STAT
    ddr_ready (0x002C bit3) is deliberately NOT gated on: it sits on the
    GLOBAL reset domain (apex_fuel_ctl.sv "crossing state: GLOBAL reset
    only"), is untouched by TILE_RST, and reads 0 forever on
    CL_DDR_PRESENT=0 builds — polling it would hang non-DDR images.

    legacy_sleep (--legacy-rst-sleep) keeps the pre-sleepcut behavior:
    2 x 10 ms open-loop settle sleeps, no readiness read.

    Returns (seconds, status_reads). A readiness miss after
    TILE_RST_TMO_S is a hard fault: loud print + SystemExit(2) (under
    --server the job replies rc=2; the card needs looking at, not a
    silently wedged file loop)."""
    t0 = time.perf_counter()
    io.write(TILE_RST_A, 1)
    if legacy_sleep:
        time.sleep(0.01)
        io.write(TILE_RST_A, 0)
        time.sleep(0.01)
        return time.perf_counter() - t0, 0
    io.write(TILE_RST_A, 0)
    deadline = t0 + TILE_RST_TMO_S
    polls = 0
    while True:
        v = io.read(TILE_STATUS_A)
        polls += 1
        if (v & TILE_READY_M) == TILE_READY_E:
            return time.perf_counter() - t0, polls
        if time.perf_counter() >= deadline:
            print(f"ABORT: TILE_RST readiness stall before [{path}]: "
                  f"STATUS {v:#010x} & {TILE_READY_M:#x} != "
                  f"{TILE_READY_E:#x} after {TILE_RST_TMO_S * 1e3:.0f} ms "
                  f"({polls} reads) — tile did not come back from reset; "
                  f"card needs attention (--legacy-rst-sleep to compare)")
            raise SystemExit(2)
        time.sleep(TILE_RST_TICK)


def _execute_job(args, get_handle) -> int:
    """ONE job — the whole single-shot body after arg parsing, verbatim:
    compile FIRST (a malformed artifact aborts before any BAR0 op), attach
    via get_handle() (a fresh pci_attach for the CLI; --server's persistent
    handle), then the per-file TILE_RST + execute loop and the summary
    lines. HwIO is rebuilt per job on purpose: counters and the
    --mmio-direct/--cap-gather accessor swaps never leak between jobs."""
    programs = []
    if not args.legacy:
        for path in args.regops:
            programs.append(compile_file(path))
    span = 0
    if args.cap_gather or args.mmio_direct:
        span = 0x0010 // 4            # TILE_RST 0x000C at minimum
        for nodes, _t, _c in programs:
            for nd in nodes:
                if nd[0] == OPC_CAPRUN:
                    hi = max(en[1] for en in nd[2])
                elif nd[0] == OPC_WB:
                    hi = nd[3] + 4 * (len(nd[4]) - 1)
                elif nd[0] in (OPC_BAD, OPC_BADCAP):
                    continue
                else:
                    hi = nd[3]
                span = max(span, hi // 4 + 1)

    kit, handle = get_handle()
    io = HwIO(kit, handle, span_dwords=span)
    if args.mmio_direct:
        if args.legacy:
            print("[mmio] --mmio-direct ignored under --legacy")
        elif not io.activate_direct():
            print("[mmio] staying on SDK peek/poke")

    use_gather = args.cap_gather and not args.stop_on_fail
    if args.cap_gather and args.stop_on_fail:
        print("[gather] disabled: --stop-on-fail needs per-op read aborts")

    total_checks = total_fails = files_run = 0
    total_cap_checked = total_cap_mismatch = 0
    all_captured = {}             # tag -> [values across every file]
    # ── capture egress: truncate at RUN start, append as caps execute ───────
    cap_fp = open(args.cap_out, "w") if args.cap_out else None
    cap = _CapEgress(cap_fp)
    times = {} if args.time_report else None
    file_times = [] if args.time_report else None
    t_run0 = time.perf_counter()
    for fi, path in enumerate(args.regops):
        # Executor parity with verif/f2sim/sim_main.cpp:198 — the sim resets
        # the CL before EVERY regops file (each file's choreography assumes
        # fresh tile/KVQ state; chunk-c0 jobs deliberately leave their store
        # populated). Hardware equivalent: bridge TILE_RST CSR 0x000C holds
        # tile+mailbox in reset (cl_apex.sv: rst_n = rst_tile_n & ~tile_rst_q)
        # while the bridge stays live. Found on silicon 2026-07-22: without
        # this, both _c1 chunk files stall their OCC poll at 256 (=2*128
        # records left by their _c0 predecessor). Pulse + STATUS readiness
        # poll (semantics and the --legacy-rst-sleep revert: _tile_reset).
        rst_s, rst_polls = _tile_reset(io, path, args.legacy_rst_sleep)
        cap0 = times[OPC_CAPRUN][0] if times and OPC_CAPRUN in times else 0.0
        t_f0 = time.perf_counter()
        if args.legacy:
            (ops, checks, fails, captured, cap_checked, cap_mismatch,
             aborted) = exec_interpreted(path, io, args.stop_on_fail, cap)
        else:
            nodes, total_ops, _census = programs[fi]
            (ops, checks, fails, captured, cap_checked, cap_mismatch,
             aborted) = exec_compiled(
                nodes, total_ops, io, args.stop_on_fail, cap,
                use_burst=not args.no_burst, use_gather=use_gather,
                budget_cap=max(1, args.pw_budget), times=times)
        if file_times is not None:
            cap1 = (times[OPC_CAPRUN][0]
                    if times and OPC_CAPRUN in times else 0.0)
            file_times.append((path, rst_s, rst_polls,
                               time.perf_counter() - t_f0, cap1 - cap0))
        tag = " [ABORTED]" if aborted else ""
        print(f"[{path}] {ops} ops, {checks} checks, {fails} fails{tag}")
        total_checks += checks
        total_cap_checked += cap_checked
        total_cap_mismatch += cap_mismatch
        for _tag, _vals in captured.items():
            all_captured.setdefault(_tag, []).extend(_vals)
        total_fails += fails + (1 if aborted and not fails else 0)
        files_run += 1
    t_run = time.perf_counter() - t_run0

    if cap_fp is not None:
        cap_fp.close()

    verdict = "FAIL" if total_fails else "PASS"
    print(f"F2HOST RESULT: files={files_run} checks={total_checks} "
          f"fails={total_fails} -> {verdict}")
    if args.cap_out:
        # UNCONDITIONAL whenever --cap-out was given, n=0 included: a run
        # that captured nothing (e.g. regops compiled without --shadow-cap)
        # must NOT be indistinguishable from a clean capture gate.
        print(f"F2HOST CAPTURES: n={cap.i} out={args.cap_out}")
    if total_cap_checked:
        # The read-back gate: every captured value equalled the baked
        # expectation the compare path was already validating, i.e. the
        # tile returns exactly what we have been checking all along.
        # Mismatches are ALSO in total_fails above (D-P1).
        cverdict = "PASS" if total_cap_mismatch == 0 else "FAIL"
        print(f"F2HOST CAPGATE: captured={total_cap_checked} "
              f"mismatches={total_cap_mismatch} -> {cverdict}")
    if args.time_report:
        if times is None or args.legacy:
            print("F2HOST TIMES: (unavailable under --legacy)")
        else:
            _print_time_report(times, sum(p[1] for p in programs), io, t_run)
        if file_times:
            _print_file_times(file_times, args.legacy)
    return 1 if total_fails else 0


# ── the persistent executor (--server) ──────────────────────────────────────
def _serve(args) -> int:
    """--server: attach ONCE (SDK import + clkgen preflight + pci_attach —
    exactly the per-invocation fixed cost this mode amortizes), then run
    one job per stdin JSON line:

        {"argv": ["<regops>", ..., "--slot", "N", "--cap-out", "<p>"]}
            -> {"rc": <int>, "log": "<the job's exact stdout+stderr>"}
        {"op": "exit"}  (or stdin EOF)  -> clean shutdown, exit 0

    stdout carries ONLY protocol lines: {"ready": true, ...} after attach,
    then one reply per job. Every job print (per-file lines, the F2HOST
    RESULT / CAPTURES / CAPGATE / TIMES summaries) is captured into the
    reply's "log", so the bridge's summary parsing reads byte-identical
    text; non-job chatter (the clkgen preflight) goes to stderr. Per job
    the SAME _execute_job body runs: same compile, same per-file TILE_RST,
    same cap-out truncate-then-append semantics. Job argv is re-parsed
    with build_parser() — no second flag surface to drift. Refused per
    job: nested --server/--selfcheck*, a --slot different from the
    server's (one process = one attached card), an empty regops list.
    --skip-clkgen-wait in a job argv is moot (the preflight ran at
    attach). Under a fake kit the reply also carries 'fake_wtrace', the
    per-job dword write trace --selfcheck-server diffs."""
    import contextlib
    import io as _io
    import os
    import traceback

    proto = sys.stdout
    sys.stdout = sys.stderr       # stray prints must never corrupt protocol
    try:
        kit, handle = _attach(args)
    except SystemExit as e:
        proto.write(json.dumps({"ready": False,
                                "error": "attach failed (see stderr)"})
                    + "\n")
        proto.flush()
        return e.code if isinstance(e.code, int) else 2
    fake = isinstance(kit, _FakeKit)
    proto.write(json.dumps({"ready": True, "slot": args.slot,
                            "pid": os.getpid(), "fake": fake}) + "\n")
    proto.flush()
    ap = build_parser()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            # a corrupt request is the CLIENT's bug — reply, don't die
            proto.write(json.dumps(
                {"rc": 2, "log": f"ABORT: bad request line: {e}\n"}) + "\n")
            proto.flush()
            continue
        if req.get("op") == "exit":
            break
        mark = len(kit.mock.wtrace) if fake else 0
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            try:
                jargs = ap.parse_args([str(x) for x in req["argv"]])
                if jargs.server or jargs.selfcheck or jargs.selfcheck_server:
                    print("ABORT: --server/--selfcheck are not nestable "
                          "inside a server job")
                    rc = 2
                elif jargs.slot != args.slot:
                    print(f"ABORT: job slot {jargs.slot} != server slot "
                          f"{args.slot} — one server per attached card")
                    rc = 2
                elif not jargs.regops:
                    print("ABORT: server job carries no regops files")
                    rc = 2
                else:
                    rc = _execute_job(jargs, lambda: (kit, handle))
            except SystemExit as e:   # argparse error, attach ABORT, …
                if e.code is not None and not isinstance(e.code, int):
                    print(e.code)
                rc = e.code if isinstance(e.code, int) else 2
            except Exception:
                traceback.print_exc()
                rc = 1
        rep = {"rc": rc, "log": buf.getvalue()}
        if fake:
            rep["fake_wtrace"] = [list(w) for w in kit.mock.wtrace[mark:]]
        proto.write(json.dumps(rep) + "\n")
        proto.flush()
    return 0


def _selfcheck_server() -> int:
    """Gate: the --server transport is INVISIBLE. The same regops on the
    same mock-pci model must produce byte-identical observables through
    (1) the single-shot CLI and (2) the server protocol — rc, the full job
    log (per-file + summary lines), the cap egress bytes, and the
    dword-for-dword write trace the fake kit records — and a REUSED server
    (two jobs, ONE process) must repeat them. This is the mock-pci
    selfcheck run THROUGH the server path; compiled == interpreted itself
    stays --selfcheck's claim."""
    import os
    import shutil
    import subprocess
    import tempfile

    me = str(Path(__file__).resolve())
    J, F, base, regs = _sc_fixture()
    # base ends [r pass, r FAIL, rn(0x3208 ok), rn(0x1FFC reads 0 -> FAIL)]
    # — clean keeps only the passing tail, for an rc=0 scenario.
    clean = base[:-3] + [base[-2]]
    spec0 = {"fifo": [F], "fifo_cap": 4, "regs": regs,
             "poll_script": {"0x200C": [0, 0, 1]}, "ready": [0x3100]}
    stall_poll = [J(op="w", a=0x1000, d=1),
                  J(op="poll", a=0x1FF0, m=1, e=1),
                  J(op="w", a=0x1004, d=2)]
    stall_pw = [J(op="pw", a=F, d=k) for k in range(6)]
    scen = [
        ("s1 base (counted FAIL)", base, {}, [], 1),
        ("s2 clean PASS", clean, {}, [], 0),
        ("s3 budget=1", base, {}, ["--pw-budget", "1"], 1),
        ("s4 unknown op abort", base[:6] + [J(op="zz", a=0)] + base[6:],
         {}, [], 1),
        ("s5 stop-on-fail", base, {}, ["--stop-on-fail"], 1),
        ("s6 poll stall", stall_poll,
         {"never_ready": [0x1FF0], "poll_seconds": 0.05}, [], 1),
        ("s7 pw stall", stall_pw,
         {"fifo_drain_on_read": False, "poll_seconds": 0.05}, [], 1),
        ("s8 legacy interpreter", base, {}, ["--legacy"], 1),
    ]
    tmp = Path(tempfile.mkdtemp(prefix="apex_srv_st_"))
    fails = 0

    def server_start(env):
        p = subprocess.Popen([sys.executable, me, "--server", "--slot", "0"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1,
                             env=env)
        hello = json.loads(p.stdout.readline())
        assert hello.get("ready") and hello.get("fake"), hello
        return p

    def server_job(p, argv):
        p.stdin.write(json.dumps({"argv": argv}) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    def server_stop(p):
        try:
            p.stdin.write(json.dumps({"op": "exit"}) + "\n")
            p.stdin.flush()
            p.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            p.kill()

    try:
        for name, lines, over, extra, want_rc in scen:
            sdir = tmp / name.split()[0]
            sdir.mkdir()
            regf = sdir / "job.regops.jsonl"
            regf.write_text("\n".join(lines) + "\n")
            spec = sdir / "fake_pci.json"
            spec.write_text(json.dumps({**spec0, **over}))
            capA, capB = sdir / "capA.jsonl", sdir / "capB.jsonl"
            trA = sdir / "traceA.json"
            envA = {**os.environ, "APEX_F2HOST_FAKE_PCI": str(spec),
                    "APEX_F2HOST_FAKE_TRACE": str(trA)}
            one = subprocess.run(
                [sys.executable, me, str(regf), "--slot", "0",
                 "--cap-out", str(capA), *extra],
                capture_output=True, text=True, env=envA, timeout=120)
            envB = {**os.environ, "APEX_F2HOST_FAKE_PCI": str(spec)}
            p = server_start(envB)
            rep = server_job(p, [str(regf), "--slot", "0",
                                 "--cap-out", str(capB), *extra])
            server_stop(p)
            bad = []
            if one.returncode != want_rc:
                bad.append(f"single-shot rc {one.returncode} != {want_rc}")
            if rep["rc"] != want_rc:
                bad.append(f"server rc {rep['rc']} != {want_rc}")
            # the server log must be the single-shot stdout, byte for byte,
            # with cap-out path the ONLY licensed difference — normalize it.
            logA = one.stdout.replace(str(capA), "<cap>")
            logB = rep["log"].replace(str(capB), "<cap>")
            if logA != logB:
                bad.append(f"log differs:\n--single--\n{logA}"
                           f"--server--\n{logB}")
            if one.stderr:
                bad.append(f"single-shot stderr not empty: {one.stderr!r}")
            egA = capA.read_bytes() if capA.exists() else None
            egB = capB.read_bytes() if capB.exists() else None
            if egA != egB:
                bad.append(f"cap egress differs ({egA!r:.80} vs {egB!r:.80})")
            trace_one = json.loads(trA.read_text())
            if trace_one != rep.get("fake_wtrace"):
                bad.append(f"write trace differs (single {len(trace_one)} "
                           f"writes, server "
                           f"{len(rep.get('fake_wtrace') or [])})")
            if bad:
                fails += 1
                print(f"  [{name}] FAIL")
                for x in bad:
                    print(f"      {x}")
            else:
                print(f"  [{name}] ok    rc={want_rc} "
                      f"writes={len(trace_one)} "
                      f"egress={len(egA or b'')}B")

        # reuse: TWO jobs on ONE server must repeat every observable
        sdir = tmp / "reuse"
        sdir.mkdir()
        regf = sdir / "job.regops.jsonl"
        regf.write_text("\n".join(clean) + "\n")
        spec = sdir / "fake_pci.json"
        spec.write_text(json.dumps(spec0))
        env = {**os.environ, "APEX_F2HOST_FAKE_PCI": str(spec)}
        p = server_start(env)
        caps, reps, logs = [], [], []
        for k in (1, 2):
            capf = sdir / f"cap{k}.jsonl"
            reps.append(server_job(p, [str(regf), "--slot", "0",
                                       "--cap-out", str(capf)]))
            caps.append(capf.read_bytes() if capf.exists() else None)
            logs.append(reps[-1]["log"].replace(str(capf), "<cap>"))
        server_stop(p)
        bad = []
        if reps[0]["rc"] != 0 or reps[1]["rc"] != 0:
            bad.append(f"rc {reps[0]['rc']}/{reps[1]['rc']} != 0")
        if logs[0] != logs[1]:
            bad.append("job 2 log differs from job 1 on the same server")
        if caps[0] != caps[1]:
            bad.append("job 2 cap egress differs from job 1")
        if reps[0]["fake_wtrace"] != reps[1]["fake_wtrace"]:
            bad.append("job 2 write trace differs from job 1")
        if bad:
            fails += 1
            print("  [reuse x2] FAIL")
            for x in bad:
                print(f"      {x}")
        else:
            print(f"  [reuse x2] ok    one server pid, two jobs, "
                  f"identical rc/log/egress/trace "
                  f"({len(reps[0]['fake_wtrace'])} writes each)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"F2HOST SERVER SELFCHECK: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails})")
    return 1 if fails else 0


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.selfcheck:
        return _selfcheck(args.regops)
    if args.selfcheck_server:
        return _selfcheck_server()
    if args.server:
        return _serve(args)
    if not args.regops:
        ap.error("give regops file(s) or --selfcheck")
    return _execute_job(args, lambda: _attach(args))


if __name__ == "__main__":
    sys.exit(main())
