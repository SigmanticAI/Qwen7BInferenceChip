#!/usr/bin/env python3
# make_ddr_image.py — IB-FUEL s2: build the pre-swizzled DDR weight image
# from the committed regops (docs/design/IB_FUEL.md §2 — "the image IS the
# wire format").
#
# Extracts each job's xw beat stream from its regops JSONL — the WB triples
# the mailbox path pushes today (XW0 lo, XW1 hi, XWP commit; the RTL
# semantics are mimicked exactly: assembly registers persist, XWP pushes
# {hi, lo}) — and lays the per-job beat sequences into one flat image:
#
#   * beat = little-endian u64 = lane8_beat_t.data (lane j = byte j)
#   * region base 4 KiB-aligned, gaps zero-filled
#   * the §2.3 divisibility invariant is a HARD ERROR if violated
#     (lane-beat count % 8 != 0 would stream padding into the tile)
#
# Outputs (--out dir):
#   ddr_image.bin           the DDR image, file offset == DDR byte address
#   ddr_image.json          human manifest {format:1, jobs:[...]}
#   ddr_image.regions.jsonl one flat record per region (the executor input:
#                           {"job","base_64B","beats_64B","tag","sha256"})
#
# --check re-reads the emitted .bin and re-derives every region from the
# regops, byte-comparing and re-hashing (the gate-(a) offline cross-check).

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

XW0, XW1, XWP = 0x3040, 0x3044, 0x3048   # BAR0 mailbox addrs (window 0x3)
ALIGN = 4096


def extract_beats(regops_path: Path) -> list[int]:
    """The job's xw wire stream, in push order (RTL-faithful WB replay)."""
    lo = hi = 0
    beats: list[int] = []
    with regops_path.open() as f:
        for ln in f:
            op = json.loads(ln)
            a = op.get("a")
            if a == XW0 and op["op"] == "w":
                lo = op["d"]
            elif a == XW1 and op["op"] == "w":
                hi = op["d"]
            elif a == XWP and op["op"] == "pw":
                assert op["d"] == 0, f"{regops_path.name}: XWP last!=0"
                beats.append((hi << 32) | lo)
    return beats


def build(jobs: list[Path], outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": 1, "jobs": []}
    regions = []
    base = 0
    for idx, jp in enumerate(jobs):
        beats = extract_beats(jp)
        if not beats:
            raise SystemExit(f"{jp.name}: no xw beats found")
        if len(beats) % 8 != 0:
            raise SystemExit(
                f"{jp.name}: lane-beat count {len(beats)} % 8 != 0 — "
                "IB_FUEL.md §2.3 invariant violated; this is a format bug, "
                "escalate to the integration lead (do NOT pad)")
        base = (base + ALIGN - 1) // ALIGN * ALIGN
        blob = struct.pack(f"<{len(beats)}Q", *beats)
        rec = {
            "job": jp.stem.replace(".regops", ""),
            "base_64B": base // 64,
            "beats_64B": len(beats) // 8,
            "tag": idx & 0xFF,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        regions.append((base, blob))
        manifest["jobs"].append(rec)
        base += len(blob)

    image = bytearray(base)
    for rbase, blob in regions:
        image[rbase:rbase + len(blob)] = blob

    (outdir / "ddr_image.bin").write_bytes(image)
    (outdir / "ddr_image.json").write_text(json.dumps(manifest, indent=1))
    with (outdir / "ddr_image.regions.jsonl").open("w") as f:
        for rec in manifest["jobs"]:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return manifest


def check(jobs: list[Path], outdir: Path) -> int:
    image = (outdir / "ddr_image.bin").read_bytes()
    manifest = json.loads((outdir / "ddr_image.json").read_text())
    by_job = {r["job"]: r for r in manifest["jobs"]}
    fails = 0
    for jp in jobs:
        name = jp.stem.replace(".regops", "")
        rec = by_job[name]
        beats = extract_beats(jp)
        blob = struct.pack(f"<{len(beats)}Q", *beats)
        b0 = rec["base_64B"] * 64
        sl = image[b0:b0 + len(blob)]
        ok_bytes = sl == blob
        ok_sha = hashlib.sha256(sl).hexdigest() == rec["sha256"]
        ok_len = rec["beats_64B"] * 8 == len(beats)
        if not (ok_bytes and ok_sha and ok_len):
            fails += 1
            print(f"[{name}] CHECK FAIL bytes={ok_bytes} sha={ok_sha} "
                  f"len={ok_len}")
        else:
            print(f"[{name}] check ok: {len(beats)} lane beats = "
                  f"{rec['beats_64B']} words @ 0x{b0:x}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regops", default=str(REPO / "build/f2_regops"))
    ap.add_argument("--out", default=str(REPO / "build/ddr_image"))
    ap.add_argument("--check", action="store_true",
                    help="re-read the emitted image and cross-check")
    args = ap.parse_args()
    jobs = sorted(Path(args.regops).glob("job_*.regops.jsonl"))
    if not jobs:
        raise SystemExit(f"no job_*.regops.jsonl under {args.regops}")
    outdir = Path(args.out)
    if args.check:
        fails = check(jobs, outdir)
        print(f"DDRIMG CHECK: jobs={len(jobs)} fails={fails} -> "
              f"{'FAIL' if fails else 'PASS'}")
        return 1 if fails else 0
    m = build(jobs, outdir)
    words = sum(r["beats_64B"] for r in m["jobs"])
    for r in m["jobs"]:
        print(f"[{r['job']}] base=0x{r['base_64B']*64:x} "
              f"words={r['beats_64B']} tag={r['tag']}")
    print(f"DDRIMG: jobs={len(m['jobs'])} words={words} "
          f"bytes={words*64} -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


