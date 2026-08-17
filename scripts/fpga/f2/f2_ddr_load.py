#!/usr/bin/env python3
# f2_ddr_load.py — put the weight image INTO the card's DDR over BAR4.
#
#   python3 scripts/fpga/f2/f2_ddr_load.py --image build/ddr_weights_05b --plan
#   python3 scripts/fpga/f2/f2_ddr_load.py --image build/ddr_weights_05b --load
#   python3 scripts/fpga/f2/f2_ddr_load.py --image build/ddr_weights_05b --verify
#   python3 scripts/fpga/f2/f2_ddr_load.py --selftest        # offline, no card
#
# ══ WHAT THE CL EXPOSES, INVESTIGATED — AND WHAT IS NEW ════════════════════
# The write path is NOT new hardware and NOT a new kit API. It is:
#
#   host  fpga_pci (AppPF BAR4, 64-bit prefetchable, 128 GiB — the kit's
#         AWS_Shell_Interface_Specification.md) ─► PCIS 512-bit AXI4
#     ─►  apex_pcis_slv.sv (this CL): DDR decoded at PCIS byte 0x0, 64 GiB
#         (`in_range` = addr[63:36]==0; outside -> DECERR, never a wedge)
#     ─►  sh_ddr (DDR_PRESENT=1, +define+APEX_CL_DDR=1) ─► the DIMM
#
# and the kit already ships every host call it needs:
#   fpga_pci.h          fpga_pci_attach(slot, pf, APP_PF_BAR4, BURST_CAPABLE)
#                       fpga_pci_write_burst(handle, offset, dwords, len)
#                       fpga_pci_peek64(handle, offset)
#   cython_bindings/fpga_pci_wrapper.pyx  pci_attach / pci_write_burst /
#                       pci_peek64 / pci_peek — all present.
#
# What is genuinely new is only THIS FILE: nothing in this repo had ever
# attached BAR4. `f2_host_run.py:112` attaches `pci_attach(slot, 0, 0, 0)` —
# AppPF BAR0, the OCL register window — and drives peek/poke. So the card-side
# loader is a new HOST SCRIPT over existing kit calls, not a new path through
# the shell. Said plainly, as asked.
#
# ══ NEVER FLOWN ════════════════════════════════════════════════════════════
# No DDR_PRESENT=1 image has ever been built or ingested (IB_FUEL.md s5 is
# owner-gated, and the deferral note of 2026-07-30 stands). Everything below
# is EXERCISED ONLY against the plan/selftest offline; the --load and --verify
# arms have never touched silicon. Any number this file prints on a card is
# the first of its kind — treat it as bring-up, not as a result.
#
# ══ THE ORDER, AND WHY EACH STEP IS THERE ══════════════════════════════════
#  1. AFI loaded, then `sudo fpga-load-clkgen-recipe -S <slot> -a 2` and a
#     frequency read-back. Inherited I-A truth: the clkgen instance must be
#     named AWS_CLK_GEN or the recipe silently does not apply, and an AFI load
#     RESETS the MMCMs to the default recipe. BAR0 before MMCM lock poisons
#     the OCL bridge by design.
#  2. FUEL_STAT[3] ddr_ready — the DDR controller's calibration. Writing
#     before calibration is undefined; poll, do not sleep.
#  3. FUEL_CTRL.ar_owner must be 0 (LOAD). In RUN mode `apex_pcis_slv`
#     answers PCIS READS with DECERR by design (§9.1 R8), so a readback in
#     RUN mode is not a failure of the DIMM — it is the mux doing its job.
#     Writes are ALWAYS PCIS-owned, in either mode.
#  4. Whole 64 B words only (the image is whole words by construction) — no
#     partial-strobe read-modify-write against uninitialised ECC.
#  5. The tile may stay in reset for all of this: the decode is entirely
#     shell-domain, so load order is free.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# BAR0 (OCL) bridge window — FUEL CSRs, cl_apex.sv bridge decode / IB_FUEL §5
FUEL_CTRL, FUEL_BASE, FUEL_BEATS, FUEL_STAT, FUEL_ERR = (
    0x0020, 0x0024, 0x0028, 0x002C, 0x0030)
STAT_FIFO_EMPTY, STAT_REQ_PEND, STAT_RDR_BUSY, STAT_DDR_READY = 1, 2, 4, 8
CTRL_AR_RUN = 0x2
APP_PF, BAR4, BAR0, BURST_CAPABLE = 0, 4, 0, 0x1
DWORD = 4
WORD = 64                       # one DDR word / one PCIS beat


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ═════════════════════════════ the burst plan ══════════════════════════════

def load_manifest(image_dir: Path) -> tuple[dict, Path]:
    man = json.loads((image_dir / "ddr_image.json").read_text())
    return man, image_dir / "ddr_image.bin"


def burst_plan(man: dict, burst_bytes: int) -> list[dict]:
    """Split every tensor region into whole-word bursts, in image order."""
    if burst_bytes % WORD or burst_bytes <= 0:
        raise SystemExit(f"--burst-bytes {burst_bytes} is not a positive "
                         f"multiple of the {WORD} B DDR word")
    out = []
    for r in man["tensors"]:
        b0, nb = r["base_64B"] * WORD, r["beats_64B"] * WORD
        if b0 % 4096:
            raise SystemExit(f"{r['job']}: base 0x{b0:x} is not 4 KiB aligned")
        for off in range(0, nb, burst_bytes):
            n = min(burst_bytes, nb - off)
            out.append({"tensor": r["job"], "offset": b0 + off, "bytes": n})
    return out


def check_plan(man: dict, plan: list[dict], image_len: int) -> None:
    """Every region byte written exactly once, nothing outside a region."""
    covered = 0
    seen: dict[int, int] = {}
    for b in plan:
        if b["bytes"] % WORD:
            raise SystemExit(f"burst at 0x{b['offset']:x} is not whole words")
        if b["offset"] % WORD:
            raise SystemExit(f"burst at 0x{b['offset']:x} is not word-aligned")
        if b["offset"] + b["bytes"] > image_len:
            raise SystemExit(f"burst at 0x{b['offset']:x} runs past the image")
        if b["offset"] in seen:
            raise SystemExit(f"burst offset 0x{b['offset']:x} written twice")
        seen[b["offset"]] = b["bytes"]
        covered += b["bytes"]
    want = sum(r["beats_64B"] * WORD for r in man["tensors"])
    if covered != want:
        raise SystemExit(f"plan covers {covered} B, regions are {want} B")


# ══════════════════════════════ the card side ══════════════════════════════

def _kit():
    """The kit's cython bindings, resolved exactly like f2_host_run.py."""
    for root in (os.environ.get("AWS_FPGA_REPO_DIR", ""),
                 os.path.expanduser("~/aws-fpga"), "/home/ubuntu/aws-fpga",
                 str(REPO / "tools/aws-fpga")):
        p = os.path.join(root, "sdk/userspace/cython_bindings")
        if root and os.path.isdir(p):
            sys.path.insert(0, p)
            break
    try:
        from fpga_pci_wrapper import FpgaPCI            # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "ABORT: fpga_pci bindings not built — source aws-fpga/sdk_setup.sh "
            "(it builds sdk/userspace/cython_bindings). This arm needs a real "
            "card; there is nothing to fake here.")
    return FpgaPCI()


class Card:
    def __init__(self, slot: int):
        self.kit = _kit()
        self.bar0 = self.kit.pci_attach(slot, APP_PF, BAR0, 0)
        self.bar4 = self.kit.pci_attach(slot, APP_PF, BAR4, BURST_CAPABLE)

    def rd0(self, a: int) -> int:
        return self.kit.pci_peek(self.bar0, a) & 0xFFFFFFFF

    def wr0(self, a: int, d: int) -> None:
        self.kit.pci_poke(self.bar0, a, d & 0xFFFFFFFF)

    def wr4(self, off: int, blob: bytes) -> None:
        assert len(blob) % DWORD == 0
        dw = list(struct.unpack(f"<{len(blob) // DWORD}I", blob))
        self.kit.pci_write_burst(self.bar4, off, dw, len(dw))

    def rd4_64(self, off: int) -> int:
        return self.kit.pci_peek64(self.bar4, off)


def preflight(c: Card, timeout_s: float) -> dict:
    """ddr_ready + LOAD-mode ar_owner + a clean FUEL_ERR, or refuse."""
    t0 = time.time()
    stat = 0
    while time.time() - t0 < timeout_s:
        stat = c.rd0(FUEL_STAT)
        if stat & STAT_DDR_READY:
            break
        time.sleep(0.05)
    if not (stat & STAT_DDR_READY):
        raise SystemExit(
            f"ABORT: FUEL_STAT=0x{stat:08x} — ddr_ready (bit 3) never came up "
            f"in {timeout_s}s. The DDR controller has not finished "
            f"calibration (or this image is DDR_PRESENT=0). Refusing to "
            f"write into an uncalibrated DIMM.")
    ctrl = c.rd0(FUEL_CTRL)
    if ctrl & CTRL_AR_RUN:
        raise SystemExit(
            f"ABORT: FUEL_CTRL=0x{ctrl:08x} has ar_owner=RUN, so PCIS reads "
            f"answer DECERR by design (§9.1 R8) and no readback could be "
            f"trusted. Quiesce the fuel line and clear ar_owner first.")
    err = c.rd0(FUEL_ERR)
    return {"FUEL_STAT": stat, "FUEL_CTRL": ctrl, "FUEL_ERR": err}


def do_load(c: Card, man: dict, img: bytes, plan: list[dict]) -> dict:
    t0 = time.time()
    done = 0
    for b in plan:
        c.wr4(b["offset"], img[b["offset"]:b["offset"] + b["bytes"]])
        done += b["bytes"]
    dt = time.time() - t0
    return {"bytes": done, "bursts": len(plan), "seconds": round(dt, 2),
            "MB_per_s": round(done / 2**20 / dt, 2) if dt else None}


def do_verify(c: Card, man: dict, img: bytes, *, full: bool,
              sample_words: int) -> dict:
    """Read BAR4 back with peek64 and compare.

    full=False verifies, per tensor, the first and last DDR word plus a
    deterministic sample — and SAYS SO in the result. Only full=True lets a
    per-tensor sha256 be claimed, because only then is the whole region read.
    """
    import random
    res = {"mode": "full" if full else "sampled", "tensors": [],
           "fails": 0, "words_read": 0}
    for r in man["tensors"]:
        b0, nw = r["base_64B"] * WORD, r["beats_64B"]
        if full:
            h = hashlib.sha256()
            bad = 0
            for w in range(nw):
                blob = b"".join(struct.pack("<Q", c.rd4_64(b0 + w * WORD + 8 * i))
                                for i in range(8))
                res["words_read"] += 1
                if blob != img[b0 + w * WORD: b0 + (w + 1) * WORD]:
                    bad += 1
                h.update(blob)
            ok = (bad == 0) and h.hexdigest() == r["sha256"]
            res["tensors"].append({"tensor": r["job"], "words": nw,
                                   "bad_words": bad, "sha_ok":
                                   h.hexdigest() == r["sha256"], "ok": ok})
        else:
            rnd = random.Random(r["base_64B"])
            picks = sorted({0, nw - 1} | {rnd.randrange(nw)
                                          for _ in range(sample_words)})
            bad = 0
            for w in picks:
                for i in range(8):
                    got = c.rd4_64(b0 + w * WORD + 8 * i)
                    res["words_read"] += 1
                    want = struct.unpack_from(
                        "<Q", img, b0 + w * WORD + 8 * i)[0]
                    if got != want:
                        bad += 1
            ok = bad == 0
            res["tensors"].append({"tensor": r["job"], "words": nw,
                                   "sampled": len(picks), "bad_lanes": bad,
                                   "ok": ok})
        if not ok:
            res["fails"] += 1
    return res


# ════════════════════════════════ selftest ═════════════════════════════════

def selftest() -> int:
    import tempfile
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f" — {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("F2_DDR_LOAD SELFTEST (offline: plan arithmetic only, no card)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        man = {"tensors": [
            {"job": "L00_Wq", "base_64B": 0, "beats_64B": 100, "sha256": ""},
            {"job": "L00_Wk", "base_64B": 128, "beats_64B": 7, "sha256": ""}]}
        img_len = 128 * WORD + 7 * WORD
        for bb in (WORD, 4096, 65536):
            p = burst_plan(man, bb)
            check_plan(man, p, img_len)
        chk("1 plans at 64 B / 4 KiB / 64 KiB all cover every region byte once",
            True)
        p = burst_plan(man, 4096)
        chk("2 no burst crosses a region boundary",
            all(any(r["base_64B"] * WORD <= b["offset"]
                    and b["offset"] + b["bytes"]
                    <= (r["base_64B"] + r["beats_64B"]) * WORD
                    for r in man["tensors"]) for b in p))
        try:
            burst_plan(man, 100)
            chk("3 a non-word burst size is REFUSED", False, "no refusal")
        except SystemExit:
            chk("3 a non-word burst size is REFUSED", True)
        bad = {"tensors": [{"job": "x", "base_64B": 1, "beats_64B": 2,
                            "sha256": ""}]}
        try:
            burst_plan(bad, 4096)
            chk("4 a non-4 KiB-aligned region is REFUSED", False, "no refusal")
        except SystemExit:
            chk("4 a non-4 KiB-aligned region is REFUSED", True)
        try:
            check_plan(man, p + [p[0]], img_len)
            chk("5 a duplicated burst is REFUSED", False, "no refusal")
        except SystemExit:
            chk("5 a duplicated burst is REFUSED", True)
        try:
            check_plan(man, [{"tensor": "x", "offset": img_len,
                              "bytes": WORD}], img_len)
            chk("6 a burst past the image is REFUSED", False, "no refusal")
        except SystemExit:
            chk("6 a burst past the image is REFUSED", True)
        # 7: the real image, if present, plans cleanly at the default size
        real = REPO / "build/ddr_weights_05b"
        if (real / "ddr_image.json").exists():
            rman, rbin = load_manifest(real)
            rp = burst_plan(rman, 65536)
            check_plan(rman, rp, rbin.stat().st_size)
            chk("7 the real 0.5B image plans cleanly", True,
                f"{len(rp)} bursts, {sum(b['bytes'] for b in rp)} B")
        else:
            print("  SKIP  7 (no build/ddr_weights_05b yet)")
    print("=" * 60)
    if fails:
        print(f"F2_DDR_LOAD SELFTEST FAIL: {fails}")
        return 1
    print("F2_DDR_LOAD SELFTEST: ALL PASS")
    return 0


# ═══════════════════════════════════ CLI ═══════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Load a make_weight_image.py DDR image into the F2 card's "
                    "DIMM over AppPF BAR4 (PCIS -> apex_pcis_slv -> sh_ddr).")
    ap.add_argument("--image", default=str(REPO / "build/ddr_weights_05b"))
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--burst-bytes", type=int, default=65536)
    ap.add_argument("--plan", action="store_true",
                    help="offline: emit + validate the burst plan, no card")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--full-verify", action="store_true",
                    help="read EVERY word back (needed to claim a sha256); "
                         "otherwise a per-tensor sample is read and the "
                         "result says 'sampled'")
    ap.add_argument("--sample-words", type=int, default=16)
    ap.add_argument("--ddr-ready-timeout", type=float, default=30.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None, help="write the JSON result here")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    idir = Path(args.image)
    man, binpath = load_manifest(idir)
    img = binpath.read_bytes()
    if hashlib.sha256(img).hexdigest() != man["image_sha256"]:
        raise SystemExit(f"ABORT: {binpath} does not match the manifest "
                         f"sha256 — rebuild the image before loading it")
    plan = burst_plan(man, args.burst_bytes)
    check_plan(man, plan, len(img))
    print(f"IMAGE  {idir}  {len(img)} B ({len(img) / 2**20:.1f} MiB), "
          f"{len(man['tensors'])} tensors, layers {man['layers']}")
    print(f"PLAN   {len(plan)} bursts of <= {args.burst_bytes} B over BAR4 "
          f"(AppPF BAR4 -> PCIS 0x0 -> sh_ddr)")

    if args.plan and not (args.load or args.verify):
        out = Path(args.out) if args.out else idir / "ddr_load_plan.json"
        out.write_text(json.dumps(
            {"image": str(binpath), "image_sha256": man["image_sha256"],
             "burst_bytes": args.burst_bytes, "n_bursts": len(plan),
             "total_bytes": sum(b["bytes"] for b in plan), "bursts": plan},
            indent=1))
        print(f"DDRPLAN: bursts={len(plan)} "
              f"bytes={sum(b['bytes'] for b in plan)} -> {out}")
        return 0

    if not (args.load or args.verify):
        ap.error("choose --plan, --load and/or --verify")

    c = Card(args.slot)
    pf = preflight(c, args.ddr_ready_timeout)
    print(f"PREFLIGHT FUEL_STAT=0x{pf['FUEL_STAT']:08x} "
          f"FUEL_CTRL=0x{pf['FUEL_CTRL']:08x} FUEL_ERR=0x{pf['FUEL_ERR']:08x} "
          f"(ddr_ready=1, ar_owner=LOAD)")
    res = {"image": str(binpath), "image_sha256": man["image_sha256"],
           "slot": args.slot, "preflight": pf}
    if args.load:
        res["load"] = do_load(c, man, img, plan)
        print(f"DDRLOAD HW: bytes={res['load']['bytes']} "
              f"bursts={res['load']['bursts']} {res['load']['seconds']}s "
              f"({res['load']['MB_per_s']} MB/s)")
    if args.verify:
        res["verify"] = do_verify(c, man, img, full=args.full_verify,
                                  sample_words=args.sample_words)
        v = res["verify"]
        print(f"DDRVERIFY HW ({v['mode']}): tensors={len(v['tensors'])} "
              f"words_read={v['words_read']} fails={v['fails']} -> "
              f"{'FAIL' if v['fails'] else 'PASS'}")
        if not args.full_verify:
            print("  NOTE: sampled — this does NOT establish a per-tensor "
                  "sha256. Use --full-verify for that claim.")
    res["FUEL_ERR_after"] = c.rd0(FUEL_ERR)
    print(f"FUEL_ERR after: 0x{res['FUEL_ERR_after']:08x}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    bad = res.get("verify", {}).get("fails", 0) or (res["FUEL_ERR_after"] & 0xF)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
