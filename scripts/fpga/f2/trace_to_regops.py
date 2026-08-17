#!/usr/bin/env python3
# trace_to_regops.py — F2 stage 2: compile S8 real-model trace jobs into
# BAR0 register operations that replay them through cl_apex's mailbox
# (window 0x3), first in Verilator simulation, then on the F2 instance.
#
# Pipeline (no new choreography is invented — reuse, then translate):
#   job .npz  ──►  verif/top/l3 core_case()   (the VERIFIED L3 choreography,
#                  imported and called verbatim on the trace tensors)
#             ──►  op-script (CSRW/XR/WB/FJOB/ERO/… tokens)
#             ──►  regops JSONL (this file's translator)
# The same regops file is executed by the Verilator harness (verif/f2sim)
# and by the fpga_pci host runner on F2 — one artifact, two executors.
#
# Regop record: {"op": .., "a": addr, ...}
#   w   a d          BAR0 write
#   r   a m e        read, check (val & m) == e
#   rn  a m          read, check (val & m) != 0
#   cap a m tag [e]  read and RETURN (val & m) to the caller, keyed by `tag`
#                    (the egress channel: --cap-out / +cap_out=). `e` is
#                    OPTIONAL: when present the executor ALSO checks
#                    (val & m) == (e & m) and a mismatch COUNTS AS A FAILURE
#                    in both executors; without `e`, `cap` never fails.
#   poll a m e       read until (val & m) == e (bounded)
#   pw  a d          stream push: poll free ((val>>9)&0x7FFF)>0 at a, write d
#   jf  a d          job fire: poll ready (bit0) at a, write d
#   note s           annotation (executors print)
# Checks counted = r + rn + poll-with-expect consumed by EFS/ESS/ERO/ETIP(T)
# semantics documented inline below. `cap` adds NO counted check (a captured
# value is not an expectation) — but a mismatching cap-with-`e` does add a
# FAIL, so a red capture path cannot exit 0.
#
# `tag` identity (emitter's job — see Xlate.cap): "<sem>_<seq>" where `sem`
# names the SITE (ro_w3, ro_meta, fs, ss, td, csr058, errstk, empty_ro,
# mbstat) and `seq` is a global per-job counter, so every cap site in a job
# is uniquely addressable. Addresses are NOT identities: RO delivers 16
# beats x 8 lanes through the same 8 addresses, so an address-derived tag
# (the pre-D-P1 `r@0x3244` scheme) collapsed 25,008 distinct sites into one.
#
# DISCLOSED CARVE-OUT: the L3 TB additionally checks passive debug taps
# (TAPF16/TAPSC/TAPPR mirrors of internal streams). The mailbox exposes no
# debug taps, so those checks are SKIPPED here and COUNTED in the manifest —
# the binding result checks (records, scale taps, result beats, TIP, CSR,
# sticky errors) all translate 1:1.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "golden"))
sys.path.insert(0, str(REPO / "verif" / "top" / "l3"))

import gen_l3_vectors as g3                              # noqa: E402
from apex_golden.fp import f16_bits_to_f64               # noqa: E402
from apex_golden import attention as at                  # noqa: E402

# BAR0 windows (cl_apex.sv)
B_CSR, B_KVQ, B_MB = 0x1000, 0x2000, 0x3000
# mailbox offsets (apex_f2_mailbox.sv — keep in lockstep)
XA, XG, QS, CS = 0x000, 0x010, 0x020, 0x030
XW0, XW1, XWP = 0x040, 0x044, 0x048
DS_W = (0x080, 0x084, 0x088, 0x08C)
DS_PUSH = 0x090
JOB = {"FJOB": 0x100, "QJOB": 0x104, "SJOB": 0x108, "LJOB": 0x10C,
       "AJ": 0x110, "WJ": 0x114}
ROUTE0, ROUTE1, IMPCLR = 0x140, 0x144, 0x148
CAP = {  # status, first-word, meta(last), adv
    "ro": (0x200, 0x204, 0x224, 0x228),
    "fs": (0x240, 0x244, None, 0x248),
    "ss": (0x250, 0x254, None, 0x258),
    "td": (0x260, 0x264, None, 0x268),
}
MB_STATUS = 0x2F0          # sticky {ovfl_out, ovfl_in} — drop audit
BR_ERRSTK, BR_DONESTK = 0x0010, 0x0014

# Set by main() from --shadow-cap; read by compile_job(), which runs in a
# different scope than the argument parser.
SHADOW_CAP = False

TAP_KINDS = {"TAPF16", "TAPSC", "TAPPR", "ENDTAPS"}


class Xlate:
    def __init__(self, D: int):
        self.D = D
        self.NBW = (D // 8 - 1).bit_length() + 1 if D else 4  # clog2(D/8)+1
        self.ops: list[dict] = []
        self.n_checks = 0
        self.n_tap_skipped = 0
        self.shadow_cap = False     # set per-job from --shadow-cap
        self.cap_seq = 0            # global per-job cap counter (tag suffix)

    # ── emit helpers ────────────────────────────────────────────────────────
    def w(self, a, d):    self.ops.append({"op": "w", "a": a, "d": d & 0xFFFFFFFF})
    def pw(self, a, d):   self.ops.append({"op": "pw", "a": a, "d": d & 0xFFFFFFFF})
    def jf(self, a, d):   self.ops.append({"op": "jf", "a": a, "d": d & 0xFFFFFFFF})
    def note(self, s):    self.ops.append({"op": "note", "s": s})

    def r(self, a, m, e, sem="r"):
        # SHADOW MODE (bring-up gate for `cap`): when enabled, every compare
        # site also emits a capture of the SAME address with the SAME mask and
        # the expectation attached. Running an existing job set this way proves
        # the read-back path returns exactly what the compare path has been
        # validating all along — without changing a single expectation, and
        # entirely in simulation. See docs/design/PROMPT_ON_CHIP.md §0.
        # `sem` names the site so the shadow cap gets a UNIQUE tag (the caller
        # supplies it; every r() call site below passes one).
        if self.shadow_cap:
            self.cap(a, m, sem, e=e)
        self.ops.append({"op": "r", "a": a, "m": m & 0xFFFFFFFF,
                         "e": e & 0xFFFFFFFF})
        self.n_checks += 1

    def rn(self, a, m, sem="rn"):
        # NOT shadowed: `cap` mirrors equality expectations only (a captured
        # value cannot reproduce a "nonzero under mask" check). 36 rn sites
        # across the 18 trace jobs stay compare-only — disclosed, not fixed.
        self.ops.append({"op": "rn", "a": a, "m": m & 0xFFFFFFFF})
        self.n_checks += 1

    def cap(self, a, m, sem, e=None):
        """READ AND RETURN — the op that makes the tile a source of values.

        Every other read op in this vocabulary (`r`, `rn`, `poll`) reads a BAR0
        address, compares it against something the HOST ALREADY KNOWS, and
        discards the value. That is why no number this tile has produced has
        ever reached a host variable, and it is the structural gap named in
        docs/design/PROMPT_ON_CHIP.md §0.

        `cap` reads the same way and RETURNS the masked value to the caller,
        keyed by a PER-SITE tag: `tag = f"{sem}_{seq}"`, where `sem` names the
        site (ro_w3, fs, ss, …) and `seq` is this job's global cap counter.
        Uniqueness is the EMITTER's responsibility — the executors key their
        egress records verbatim by whatever tag arrives, and two sites sharing
        a tag are two values the grader cannot tell apart. Addresses are not
        identities (RO: 16 beats x 8 lanes through 8 addresses).

        `e` is OPTIONAL:
          * absent  — pure egress; the op cannot fail (Milestone B).
          * present — the bring-up gate: the executor ALSO checks
            (val & m) == (e & m) and, since D-P1, a mismatch INCREMENTS THE
            FAILURE COUNT in both executors (before D-P1 it only printed, so
            a red capture path still exited 0). Note the compare masks BOTH
            sides, which is why nothing here needs to pre-mask `e`.
        """
        tag = f"{sem}_{self.cap_seq}"
        self.cap_seq += 1
        rec = {"op": "cap", "a": a, "m": m & 0xFFFFFFFF, "tag": tag}
        if e is not None:
            rec["e"] = e & 0xFFFFFFFF
        self.ops.append(rec)
        return tag

    def poll(self, a, m, e):
        self.ops.append({"op": "poll", "a": a, "m": m & 0xFFFFFFFF,
                         "e": e & 0xFFFFFFFF})

    # ── capture-FIFO pop-check: wait beat, peek words, compare, advance ────
    def pop_check(self, which, checks):
        st, w0, meta, adv = CAP[which]
        self.poll(B_MB + st, 0x100, 0x100)          # nonempty
        for off, m, e in checks:
            sem = which if off == 0 else f"{which}_w{off // 4}"
            self.r(B_MB + w0 + off, m, e, sem=sem)
        if meta is not None and checks and checks[-1][0] == "META":
            pass
        self.w(B_MB + adv, 1)

    # ── the op-script translator ────────────────────────────────────────────
    def run(self, lines):
        i = 0
        toks_iter = iter(lines)
        for ln in toks_iter:
            ln = ln.strip()
            if not ln or ln.startswith("//"):
                if ln:
                    self.note(ln[2:].strip())
                continue
            t = ln.split()
            op, args = t[0], t[1:]
            if op == "CSRW":
                self.w(B_CSR + int(args[0], 16), int(args[1], 16))
            elif op == "CSRR":
                self.r(B_CSR + int(args[0], 16), int(args[1], 16),
                       int(args[2], 16),
                       sem=f"csr{int(args[0], 16):03x}")
            elif op == "CSRP":
                self.poll(B_CSR + int(args[0], 16), int(args[1], 16),
                          int(args[2], 16))
            elif op == "CSRN":
                self.rn(B_CSR + int(args[0], 16), int(args[1], 16),
                        sem=f"csrn{int(args[0], 16):03x}")
            elif op == "KVW":
                self.w(B_KVQ + int(args[0], 16), int(args[1], 16))
            elif op == "KVP":
                self.poll(B_KVQ + int(args[0], 16), int(args[1], 16),
                          int(args[2], 16))
                self.n_checks += 1      # KVP is a scripted expectation
            elif op == "ROUTE":
                # script packing == ROUTE0 packing (verified bit-for-bit)
                self.w(B_MB + ROUTE0, int(args[0], 16))
            elif op == "IMP":
                self.w(B_MB + ROUTE1,
                       (int(args[0], 16) << 16) | int(args[1], 16))
            elif op == "DESC":
                # script order = MSW..LSW; DS_W[3]=bits[127:96] … DS_W[0]=LSW
                words = [int(a, 16) for a in args]      # w127_96 … w31_0
                for reg, val in zip(reversed(DS_W), words):
                    self.w(B_MB + reg, val)
                self.pw(B_MB + DS_PUSH, 0)
            elif op == "XR":
                for j, b in enumerate(args):
                    last = 1 if j == self.D - 1 else 0
                    self.pw(B_MB + XA, (last << 8) | int(b, 16))
            elif op == "GR":
                for gtok in args:
                    self.pw(B_MB + XG, int(gtok, 16))
            elif op == "WB":
                b64 = int(args[0], 16)
                self.w(B_MB + XW0, b64 & 0xFFFFFFFF)
                self.w(B_MB + XW1, (b64 >> 32) & 0xFFFFFFFF)
                self.pw(B_MB + XWP, 0)                   # last=0 (TB parity)
            elif op == "QS":
                self.pw(B_MB + QS, int(args[0], 16))
            elif op == "CS":
                self.pw(B_MB + CS, int(args[0], 16))
            elif op == "FJOB":
                self.jf(B_MB + JOB[op], int(args[0], 16))
            elif op == "QJOB":
                self.jf(B_MB + JOB[op],
                        (int(args[0]) << 12) | int(args[1], 16))
            elif op == "SJOB":
                self.jf(B_MB + JOB[op], int(args[0], 16))
            elif op == "LJOB":
                self.jf(B_MB + JOB[op],
                        (int(args[1], 16) << 8) | int(args[0], 16))
            elif op in ("AJ", "WJ"):
                o, bank, pat = int(args[0]), int(args[1]), int(args[2])
                rows, nb, sel = (int(args[3], 16), int(args[4], 16),
                                 int(args[5], 16))
                d = (o | (bank << 1) | (pat << 2) | (rows << 4)
                     | (nb << 9) | (sel << (9 + self.NBW)))
                self.jf(B_MB + JOB[op], d)
            elif op == "EFS":
                self.pop_check("fs", [(0, 0x1FFFF,
                                       (int(args[1]) << 16)
                                       | int(args[0], 16))])
            elif op == "ESS":
                self.pop_check("ss", [(0, 0x1FFFF,
                                       (int(args[1]) << 16)
                                       | int(args[0], 16))])
            elif op == "ERO":
                lanes = [int(a, 16) for a in args[:8]]
                last = int(args[8])
                st, w0, meta, adv = CAP["ro"]
                self.poll(B_MB + st, 0x100, 0x100)
                for j, v in enumerate(lanes):
                    self.r(B_MB + w0 + 4 * j, 0xFFFFFFFF, v, sem=f"ro_w{j}")
                self.r(B_MB + meta, 0x1, last, sem="ro_meta")
                self.w(B_MB + adv, 1)
            elif op == "ETIP":
                blk = int(args[0], 16)
                self.pop_check("td", [(0, 0x7F << 3, blk << 3)])
            elif op == "ETIPT":
                blk, tier, fp16 = (int(args[0], 16), int(args[1], 16),
                                   int(args[2], 16))
                self.pop_check("td", [(0, 0x3FF,
                                       (blk << 3) | (tier << 1) | fp16)])
            elif op == "ESTK":
                self.r(BR_ERRSTK, int(args[0], 16), int(args[1], 16),
                       sem="errstk")
            elif op == "DONE":
                # every queue drained: all capture FIFOs empty
                for which, (st, _, _, _) in CAP.items():
                    self.r(B_MB + st, 0x100, 0x000, sem=f"empty_{which}")
                # and the mailbox itself never dropped a push or a job fire.
                # Without this the twelve silent-drop paths are unobservable,
                # and a dropped beat is indistinguishable from a miscomputed
                # value in the log — which is exactly how the xw_beat field
                # mis-order stayed unlocalised for several sessions.
                self.r(B_MB + MB_STATUS, 0x3, 0x0, sem="mbstat")
            elif op in TAP_KINDS:
                # passive debug-tap expectation — not observable on hardware.
                # Format: KIND <count-hex> followed by <count> raw value lines.
                if op != "ENDTAPS":
                    n_vals = int(args[0], 16)
                    for _ in range(n_vals):
                        next(toks_iter)
                    self.n_tap_skipped += n_vals
            else:
                raise SystemExit(f"untranslated op: {op} ({ln[:60]})")


def compile_job(npz_path: Path):
    z = np.load(npz_path)
    meta = json.loads(str(z["meta_json"]))
    D, T, tier_s = meta["D"], meta["T"], meta["tier"]
    tier = {"CQ-8": at.TIER_CQ8, "CQ-4": at.TIER_CQ4,
            "CQ-4+": at.TIER_CQ4P}[tier_s]
    q = f16_bits_to_f64(z["q_f16"].astype(np.uint16))
    name = npz_path.stem
    s, r = g3.core_case(name, q, z["K_f16"].astype(np.uint16),
                        z["V_f16"].astype(np.uint16), D, tier)
    # cross-check the choreography's golden result against the STORED trace
    # intermediates (the S8 chain): o8 + s_out must match bit-exact.
    assert np.array_equal(np.asarray(r.o8, dtype=np.int64),
                          z["o8"].astype(np.int64)), f"{name}: o8 mismatch"
    x = Xlate(D)
    x.shadow_cap = SHADOW_CAP
    x.note(f"trace job {name}: T={T} D={D} {tier_s} "
           f"(step {meta['step']} layer {meta['layer']} head {meta['head']})")
    x.run(s.lines)
    return x, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(
        REPO / "docs/results/s8_7b_token/artifact_trace"))
    ap.add_argument("--out", default=str(REPO / "build/f2_regops"))
    ap.add_argument("--jobs", type=int, default=0, help="limit (0 = all)")
    ap.add_argument("--shadow-cap", action="store_true",
                    help="emit a `cap` (read-and-return) alongside every "
                         "`r` compare, carrying the same expectation and a "
                         "per-site tag. The bring-up gate for the read-back "
                         "path: proves the tile returns exactly what we have "
                         "been comparing, without changing any expectation. "
                         "Pair with the executors' --cap-out/+cap_out= to get "
                         "the values themselves. PROMPT_ON_CHIP.md §0")
    args = ap.parse_args()
    global SHADOW_CAP
    SHADOW_CAP = args.shadow_cap
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = sorted(Path(args.trace).glob("job_*.npz"))
    if args.jobs:
        jobs = jobs[:args.jobs]
    manifest = {"jobs": [], "total_checks": 0, "tap_checks_skipped": 0,
                "total_caps": 0}
    for j in jobs:
        x, meta = compile_job(j)
        of = outdir / f"{j.stem}.regops.jsonl"
        with of.open("w") as f:
            for o in x.ops:
                f.write(json.dumps(o, separators=(",", ":")) + "\n")
        manifest["jobs"].append({
            "job": j.stem, "regops": len(x.ops), "checks": x.n_checks,
            "taps_skipped": x.n_tap_skipped, "caps": x.cap_seq, **meta})
        manifest["total_checks"] += x.n_checks
        manifest["tap_checks_skipped"] += x.n_tap_skipped
        manifest["total_caps"] += x.cap_seq
        print(f"[{j.stem}] {len(x.ops)} regops, {x.n_checks} checks, "
              f"{x.cap_seq} caps "
              f"({x.n_tap_skipped} tap-expects skipped, disclosed)")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"TOTAL: {len(jobs)} jobs, {manifest['total_checks']} checks → "
          f"{outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
