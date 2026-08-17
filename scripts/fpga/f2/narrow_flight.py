#!/usr/bin/env python3
"""N-lane flight driver (v13 N4): the whole narrow-image layer-op set, batched.

gen : build EVERY stage the narrow silicon claim rests on into --out:
        resid_probe            R3 geometry-refusal guard
        resid_r1 / resid_r2    residual UNIT demo over the 128-window
        norm2 (dm=128)         RMSNorm-2 UNIT demo (128-wide instance;
                               explicitly NOT the full-row 3584 op — that
                               claim stays gated on aws-fpga#799)
        swiglu_probe + swiglu  ALL 296 x 64-col chunks = the FULL d_ffn=18944
                               of the real step (full-fidelity op)
        rope h00..h27          ALL 28 heads, head_dim=128 (full-fidelity op)
        residfull r1/r2        the FULL D_model=3584 residual as 28 + 28
                               aligned 128-element slices (full-fidelity op)
      Every program is produce-mode (zero-output-expectation audit enforced
      here, same gate as the sweep driver).

fly : run the set through batch_exec (marker-demux + per-file manifests),
      demux captures per stage, grade each with gen_layer_ops' own graders,
      and write a result JSON. --executor sim IS the pre-flight proof;
      --executor hw is the claim run (remote_hw_exec verifies the A2 clock
      numerically per invocation).

      The full-row residual is graded TWICE: per slice (28 independent
      bit-exact grades per row, gen_layer_ops' own grade_resid at that
      slice's offset) and then by REASSEMBLY — the 28 slices' CAPTURED fp16
      words are concatenated into one 3584 row and compared against golden's
      r1/r2 verbatim. Per-slice greens with a reassembly red would mean the
      slices were right but not the row; both must be green.

══ norm2m: THE FULL-ROW RMSNorm-2 WITH NO HOST ARITHMETIC AT ALL ══════════
norm2c leaves the host doing dm/128 - 1 = 27 INT32 adds (golden's own step 2).
norm2m removes them. It is an ORDERED PAIR, because a regops program has no
verb that moves a captured value into a register — `cap` returns bits to the
HOST, and only the host can write a CSR:

  program 1  nf_sum2m   the MXE computes sum2 = x·x over the whole 3584 row
                        (the row is both the activation and the single weight
                        column) and CHAINS the K-split ON THE TILE: every
                        descriptor after the first carries OP_GEMM_OS
                        accumulate=1, so the array retains its accumulators
                        across descriptors (mxe_array.sv:36-52,
                        mxe_ctrl.sv:286) and the LAST RO beat's lane 0 is
                        already the total. The host adds NOTHING.
  host       one u32 read out of a capture register, written unchanged into
                        RMS_SUM2. assert_move_only() re-derives that claim
                        FROM THE EMITTED FILE: exactly one write to 0x90, its
                        payload byte-identical to the captured word, and no
                        read of 0x90 anywhere.
  program 2  nf_norm2m  build_norm2_ext(sum2_source="tile") — the R4 ext-armed
                        pass 2, graded EXACTLY like norm2c/norm2x (3584 codes
                        AND 28 row scales against golden's own h2).

The discriminator cannot be "+1": sum2 reaches the graded row only through the
integer r = 2^13 // isqrt(...), so golden itself returns a bit-identical row
for every sum2 in one r bucket (measured 4942 wide here). The perturbation
used is the FIRST delta that leaves that bucket, computed host-side before the
run by norm2_grade_sensitivity(), and a +1 arm is ALSO run as a control that
must stay green — the tile must be exactly as sensitive as the arbiter.

══ WHY 28 SLICES ARE THE FULL ROW (not an approximation) ══════════════════
The residual is ELEMENTWISE — row[i] <- f16(row[i] + b[i]), no cross-element
state (apex_residual indexes each beat independently; golden transformer.py
r = f16(x + y)). So an aligned 128-element slice computes EXACTLY the words
one wide pass would produce for those elements, and 28 of them tile the row
with no overlap and no gap. This is what lets a DM_MAX=128 image compute the
FULL 3584 residual; the host only reloads the row slice between passes (data
movement, no arithmetic). Proven end to end by
scripts/fpga/f2/resid_full_check.py (28/28 for r1 AND r2, reassembled rows
== golden) before it was wired in here.

Usage:
  python3 narrow_flight.py selftest                 # host-only grader proof
  python3 narrow_flight.py gen  --npz build/f2_layer/layer_step_full.npz
  python3 narrow_flight.py fly  --executor sim --binary verif/f2sim/obj_d128_ddr0/f2sim
  python3 narrow_flight.py fly  --executor hw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_layer_ops as gl                                      # noqa: E402
import gemm_job as gj                                           # noqa: E402
import batch_exec as bx                                         # noqa: E402
import cap_decode as cd                                         # noqa: E402
import tile_exec_bridge as bridge                               # noqa: E402
import trace_to_regops as t2r                                   # noqa: E402
import remote_hw_exec                                           # noqa: E402
from apex_golden import transformer as tf                       # noqa: E402

DEFAULT_OUT = REPO / "build" / "n_flight"
N_HEADS = 28
SWG_CHUNKS = 296                       # 296 x 64 = 18944 = the full d_ffn
RESID_SLICE = 128                      # the narrow image's DM_MAX
RESID_ROWS = ("r1", "r2")
RMS_SUM2_ADDR = t2r.B_CSR + gl.RMS_SUM2        # the CSR the moved u32 lands in


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def gen(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = Path(args.npz)
    z = np.load(npz, allow_pickle=False)
    step = {k: z[k] for k in z.files if k != "meta_json"}
    meta = json.loads(str(z["meta_json"]))
    step["d_ffn"] = meta["d_ffn"]
    step["head_dim"] = meta["head_dim"]
    assert int(meta["n_ffn"]) >= SWG_CHUNKS * 64, \
        f"npz n_ffn={meta['n_ffn']} < {SWG_CHUNKS * 64}: re-capture with " \
        f"--n-ffn 18944 (full-depth operands)"

    mans = []
    mans.append(gl.build_resid_probe(step, out_dir=out, name="nf_resid_probe"))
    m, _ = gl.build_resid_stage(step, which="r1", cols=128, out_dir=out,
                                name="nf_resid_r1")
    mans.append(m)
    mans.append(gl.build_norm2_stage(step, dm=128, out_dir=out,
                                     name="nf_norm2"))
    # ── R4: the FULL D_model RMSNorm-2 on this NARROW build ────────────────
    # norm2c is the self-contained two-pass form (tile exports each chunk's
    # sum2, host does k-1 INT32 adds — golden's own step 2 — then the ARMED
    # tile runs mu + rsqrt + all D_model per-element ops). norm2x is pass-2
    # only with sum2 supplied from outside, which is the shape the proven
    # MXE x.x source plugs into (sum2_mxe.py: 120498, bit-exact). norm2_probe
    # guards the refusal path. Together these lift RMSNorm-2 from a 128-wide
    # UNIT demo to the FULL-FIDELITY op with no wide image.
    dm_full = int(meta["D_model"])
    mans.append(gl.build_norm2_probe(step, out_dir=out, name="nf_norm2_probe"))
    mans.append(gl.build_norm2_chunked(step, dm=dm_full, out_dir=out,
                                       name="nf_norm2c"))
    mans.append(gl.build_norm2_ext(step, dm=dm_full, out_dir=out,
                                   name="nf_norm2x"))
    # ── norm2m: the SAME full-row norm with the 27 host adds REMOVED ───────
    # Half one is a program: the MXE computes the row's sum-of-squares as the
    # x·x dot product and CHAINS the K-split on the tile (OP_GEMM_OS
    # accumulate=1), so the last RO beat already IS the total. Half two cannot
    # be built here — its only input is that beat — so `fly` builds it AFTER
    # the run and proves the host step was a MOVE (see move_and_norm).
    mans.append(gl.build_sum2_chain(step, dm=dm_full, out_dir=out,
                                    name="nf_sum2m",
                                    rows_per_desc=args.sum2_rows_per_desc))
    m, _ = gl.build_resid_stage(step, which="r2", cols=128, out_dir=out,
                                name="nf_resid_r2")
    mans.append(m)
    mans.append(gl.build_swiglu_probe(step, out_dir=out,
                                      name="nf_swiglu_probe"))
    mans.append(gl.build_swiglu_stage(step, chunks=SWG_CHUNKS, out_dir=out,
                                      name="nf_swiglu"))
    for h in range(N_HEADS):
        mans.append(gl.build_rope_stage(step, head=h, out_dir=out,
                                        name=f"nf_rope_h{h:02d}"))

    # ── the FULL-ROW residual: D_model as aligned RESID_SLICE passes ───────
    dm = int(meta["D_model"])
    assert dm % RESID_SLICE == 0, \
        f"D_model={dm} is not a whole number of {RESID_SLICE}-element slices"
    n_sl = dm // RESID_SLICE
    resid_full = {"d_model": dm, "cols": RESID_SLICE, "n_slices": n_sl,
                  "rows": {}}
    for which in RESID_ROWS:
        names = []
        for k in range(n_sl):
            nm = f"nf_residfull_{which}_{k:02d}"
            m, _exp = gl.build_resid_stage(step, which=which,
                                           cols=RESID_SLICE, off=k * RESID_SLICE,
                                           out_dir=out, name=nm)
            assert int(m["off"]) == k * RESID_SLICE and \
                int(m["cols"]) == RESID_SLICE, f"{nm}: window bookkeeping"
            mans.append(m)
            names.append(nm)
        resid_full["rows"][which] = names
    # tiling proof, host-side and BEFORE anything runs: the slice offsets
    # cover [0, D_model) exactly once, no overlap, no gap.
    for which in RESID_ROWS:
        offs = sorted(int(m["off"]) for m in mans
                      if m["name"] in resid_full["rows"][which])
        assert offs == [k * RESID_SLICE for k in range(n_sl)], \
            f"{which}: slice offsets {offs[:4]}… do not tile {dm}"

    # the norm2m pair's second half is built at FLY time from a tile capture
    sens = gl.norm2_grade_sensitivity(step, dm_full)
    norm2m = dict(sum2_stage="nf_sum2m", dm=int(dm_full),
                  ext_name="nf_norm2m", disc_name="nf_norm2m_disc",
                  ctl_name="nf_norm2m_ctl", sensitivity=sens,
                  note="pass 2 is NOT in this plan on purpose: its only "
                       "input is the MXE capture, so building it at gen time "
                       "would mean the host computed the number")

    n_viol = sum(m["audit"]["violations"] for m in mans)
    plan = dict(npz=str(npz), meta=meta, stages=mans,
                audit_violations=int(n_viol),
                swiglu_chunks=SWG_CHUNKS, rope_heads=N_HEADS,
                resid_full=resid_full, norm2m=norm2m,
                claim_note="the nf_norm2 stage is a 128-wide UNIT demo; the "
                           "FULL-ROW norm is norm2c/norm2x/norm2m (R4 chunk "
                           "composition on this narrow build) — norm2m is the "
                           "same full row with the host's 27 INT32 adds "
                           "REMOVED: the MXE computes sum2 and accumulates it "
                           "across descriptors ON THE TILE, and the host only "
                           "moves one u32 into RMS_SUM2. The RESIDUAL is "
                           "full-fidelity: the whole D_model row as "
                           f"{n_sl} aligned {RESID_SLICE}-element slices per "
                           "row, graded per slice AND by reassembly against "
                           "golden's r1/r2")
    (out / "flight_plan.json").write_text(json.dumps(plan, indent=1))
    eprint(f"[gen] {len(mans)} stages -> {out}  "
           f"(audit violations: {n_viol} — MUST be 0; full-row residual: "
           f"{n_sl} + {n_sl} slices covering {dm} elements each of r1/r2)")
    sm = next(m for m in mans if m["name"] == "nf_sum2m")
    eprint(f"[gen] norm2m: sum2 chain = {sm['n_desc']} chained OP_GEMM_OS "
           f"descriptors over K_staged={sm['K_staged']} ({sm['staged_rows']} "
           f"staged rows), 0 host adds; grade sensitivity "
           f"delta_min={sens['delta_min']} (+1 absorbed by golden's own "
           f"tail: {sens['plus_one_absorbed']})")
    return 0 if n_viol == 0 else 2


def reassemble_resid_row(plan, by_path, which: str) -> dict:
    """Concatenate the CAPTURED fp16 words of one row's slices into the full
    D_model row and compare against golden verbatim.

    The bits come from the tile's own LAYER_RDATA captures (cap_decode
    .rdata_f16) — NEVER from the per-slice expectation — so a slice that
    graded green for the wrong reason cannot launder itself into the row.
    Coverage is tracked per element: a missing or short slice leaves holes,
    and holes are a FAIL even if every element that WAS written matches.
    """
    rf = plan["resid_full"]
    dm, cols = int(rf["d_model"]), int(rf["cols"])
    by_name = {m["name"]: m for m in plan["stages"]}
    got = np.zeros(dm, dtype=np.uint16)
    seen = np.zeros(dm, dtype=bool)
    notes, n_slices_ok = [], 0
    for nm in rf["rows"][which]:
        m = by_name[nm]
        f = by_path.get(m["path"])
        caps = f["captures"] if f else []
        bits = [c["bits"] for c in cd.rdata_f16(caps)]
        off = int(m["off"])
        if len(bits) != cols:
            notes.append(f"{nm}: {len(bits)}/{cols} RDATA captures")
            continue
        got[off:off + cols] = np.asarray(bits, dtype=np.uint16)
        seen[off:off + cols] = True
        n_slices_ok += 1
    return {"which": which, "d_model": dm, "n_slices": len(rf["rows"][which]),
            "slices_with_full_capture": n_slices_ok,
            "covered": int(seen.sum()), "notes": notes,
            "bits": got, "complete": bool(seen.all())}


def grade_resid_full(plan, by_path, step) -> dict:
    """Per-row reassembly verdict: the concatenated 3584 row == golden."""
    out = {}
    for which in plan["resid_full"]["rows"]:
        a = reassemble_resid_row(plan, by_path, which)
        gold = np.asarray(step[f"{which}_bits"], dtype=np.uint16)
        eq = bool(a["complete"] and a["bits"].shape == gold.shape
                  and np.array_equal(a["bits"], gold))
        d = np.flatnonzero(a["bits"] != gold)
        out[which] = {"equal": eq, "d_model": a["d_model"],
                      "n_slices": a["n_slices"],
                      "slices_with_full_capture":
                          a["slices_with_full_capture"],
                      "covered": a["covered"], "n_diff": int(d.size),
                      "first_idx": int(d[0]) if d.size else -1,
                      "notes": a["notes"]}
    return out


def assert_move_only(regops_path, sum2_u32: int) -> int:
    """PROVE the host's step was a MOVE, not a computation.

    Re-reads the EMITTED program and requires: exactly one write to
    RMS_SUM2 (0x90), whose 32-bit payload is BYTE-IDENTICAL to the untouched
    u32 the executor read out of the MXE's lane-0 capture register, and no
    read/poll of that address anywhere (the zero-output-expectation rule
    already covers RMS_SUM2's read view, gen_layer_ops.OUTPUT_ADDRS).
    Raises rather than returning a verdict: a failed move must stop the run.
    """
    ops = [json.loads(l) for l in Path(regops_path).read_text().splitlines()]
    ws = [o for o in ops if o["op"] == "w" and o.get("a") == RMS_SUM2_ADDR]
    assert len(ws) == 1, \
        f"{regops_path}: {len(ws)} writes to RMS_SUM2, expected exactly 1"
    got = int(ws[0]["d"]) & 0xFFFFFFFF
    want = int(sum2_u32) & 0xFFFFFFFF
    assert got == want, (
        f"{regops_path}: the word written to RMS_SUM2 is {got} "
        f"(0x{got:08x}) but the captured MXE accumulator is {want} "
        f"(0x{want:08x}) — the host TRANSFORMED the value instead of moving "
        f"it; this run does not support a zero-host-arithmetic claim")
    rd = [o for o in ops if o["op"] in ("r", "rn", "poll")
          and o.get("a") == RMS_SUM2_ADDR]
    assert not rd, f"{regops_path}: {len(rd)} read(s) of RMS_SUM2 survived"
    return got


def _run_one(path, args) -> dict:
    """One regops file through the same batched executor path as the flight."""
    rb = bx.run_jobs_batched([str(path)], executor=args.executor,
                             binary=args.binary, timeout_s=args.timeout_s)
    f = rb["files"][0] if rb["files"] else None
    return {"ok": bool(f and f["complete"] and f["attribution_ok"]
                       and rb["ok"]),
            "caps": f["captures"] if f else [], "notes": rb["notes"]}


def move_and_norm(plan, step, sum2_caps, out, args) -> dict:
    """The norm2m pair, phase 2: MOVE the captured INT32 into RMS_SUM2 and run
    the R4 ext-armed full-row norm — then prove the grade can go red.

    The ONLY host action between the two tile jobs is
        sum2_u32 = <the u32 read out of the MXE lane-0 capture register>
        <build a program whose single RMS_SUM2 write carries exactly those bits>
    with assert_move_only() re-deriving that claim FROM THE EMITTED FILE.
    """
    cfg = plan["norm2m"]
    dm = int(cfg["dm"])
    man_sum2 = next(m for m in plan["stages"] if m["name"] == cfg["sum2_stage"])
    ct = gj.chain_totals(sum2_caps, n_desc=int(man_sum2["n_desc"]))
    sum2_u32 = int(ct["raw_lane0_u32"])          # <- the moved bits, verbatim
    res = {"sum2_moved_u32": sum2_u32, "n_beats": int(ct["n_beats"]),
           "host_adds": 0, "sensitivity": cfg["sensitivity"]}

    runs = [("ext", cfg["ext_name"], 0, True),
            ("disc", cfg["disc_name"], int(cfg["sensitivity"]["delta_min"]),
             False),
            ("ctl", cfg["ctl_name"], 1, True)]
    for key, name, delta, want_green in runs:
        armed = sum2_u32 + delta
        man = gl.build_norm2_ext(step, dm=dm, out_dir=out, name=name,
                                 ext_sum2=armed, sum2_source="tile")
        wrote = assert_move_only(man["path"], armed)
        if delta == 0:
            # the load-bearing one: the bits that left the MXE are the bits
            # that reach the CSR, with nothing applied in between.
            assert wrote == sum2_u32, "move proof did not bind the real run"
        assert man["audit"]["violations"] == 0, \
            f"{name}: {man['audit']['violations']} output expectation(s)"
        r = _run_one(man["path"], args)
        g = gl._grade_stage(man, step, plan["meta"], r["caps"], plan)
        green = bool(r["ok"] and g.get("equal"))
        res[key] = {"name": name, "armed": armed, "delta": delta,
                    "csr_word": wrote, "run_ok": bool(r["ok"]),
                    "caps": len(r["caps"]), "grade": g, "green": green,
                    "as_expected": bool(green == want_green),
                    "notes": r["notes"]}
        eprint(f"  [{name}] armed={armed} (delta {delta:+d}) "
               f"run={'ok' if r['ok'] else 'FAIL'} "
               f"grade={'PASS' if g.get('equal') else 'FAIL'} "
               f"caps={len(r['caps'])} -> "
               f"{'as expected' if green == want_green else 'UNEXPECTED'}")
    res["discriminator_caught"] = bool(not res["disc"]["green"])
    res["plus_one_absorbed_measured"] = bool(res["ctl"]["green"])
    # all three must land where golden says they must: the moved value
    # reproduces the row, the first delta that leaves golden's r bucket does
    # NOT, and a delta INSIDE that bucket still does (the tile is exactly as
    # sensitive as the arbiter — no more, no less).
    res["ok"] = bool(res["ext"]["green"] and not res["disc"]["green"]
                     and res["ctl"]["green"])
    return res


def fly(args) -> int:
    out = Path(args.out)
    plan = json.loads((out / "flight_plan.json").read_text())
    if plan["audit_violations"] != 0:
        eprint("[fly] REFUSE: plan has audit violations")
        return 2
    # Route executor='hw' to the REMOTE F2 (env-gated; no-op for sim). Without
    # this the bridge tries to drive BAR0 from THIS machine and every batch
    # returns no cap file at all — which is exactly what happened on the
    # first two hw attempts (2026-07-31). The shim also re-verifies the A2
    # clock NUMERICALLY on every invocation.
    if args.executor == "hw":
        if not remote_hw_exec.attach(bridge, args):
            eprint("[fly] REFUSE: executor=hw but no remote config "
                   "(set APEX_F2_HOST / APEX_F2_KEY)")
            return 2
        eprint("[fly] remote hw shim attached")
    z = np.load(plan["npz"], allow_pickle=False)
    step = {k: z[k] for k in z.files if k != "meta_json"}
    meta = plan["meta"]
    step["d_ffn"] = meta["d_ffn"]
    step["head_dim"] = meta["head_dim"]

    # SIZE-BOUNDED BATCHES (2026-07-31): one batch of all 34 stages is 80 MB
    # because the full-depth swiglu stage alone is 74 MB (296 chunks ->
    # ~2.8M regops). That upload never completed and the batch produced no
    # cap file at all. Group by BYTES, not by count, and let any oversized
    # single file be its own batch — the same lesson the projection sweep
    # learned about per-file transport.
    files = [m["path"] for m in plan["stages"]]
    budget = int(args.max_batch_mb) * 1024 * 1024
    batches, cur, cur_b = [], [], 0
    for p in files:
        sz = Path(p).stat().st_size
        if cur and cur_b + sz > budget:
            batches.append(cur)
            cur, cur_b = [], 0
        cur.append(p)
        cur_b += sz
    if cur:
        batches.append(cur)
    eprint(f"[fly] {len(files)} stages -> {len(batches)} batch(es) "
           f"(<= {args.max_batch_mb} MB each)")

    t0 = time.time()
    merged_files, notes, all_ok = [], [], True
    for i, chunk in enumerate(batches, 1):
        mb = sum(Path(p).stat().st_size for p in chunk) / 1048576
        eprint(f"[fly] batch {i}/{len(batches)}: {len(chunk)} stage(s), "
               f"{mb:.1f} MB")
        rb = bx.run_jobs_batched(chunk, executor=args.executor,
                                 binary=args.binary,
                                 timeout_s=args.timeout_s)
        merged_files.extend(rb["files"])
        notes.extend(rb["batch"]["notes"] + rb["notes"])
        all_ok = all_ok and rb["ok"]
    r = {"ok": all_ok, "files": merged_files,
         "batch": {"notes": notes}, "notes": []}
    wall = time.time() - t0
    if not r["ok"]:
        for n in notes:
            eprint(f"[fly] note: {n}")

    rows, n_pass = [], 0
    by_path = {f["path"]: f for f in r["files"]}
    for m in plan["stages"]:
        f = by_path.get(m["path"])
        caps = f["captures"] if f else []
        ok = bool(f and f["complete"] and f["attribution_ok"])
        try:
            if m["name"] == "nf_norm2":
                # UNIT-DEMO grade: a dm=128 norm is a DIFFERENT op instance
                # than the full-row norm (different rms denominator), so the
                # expectation is the ARBITER'S OWN rmsnorm applied at 128
                # width to the same input codes — never the full-row h2.
                # (Codes alone would pass vacuously: row-quant codes are
                # scale-invariant; the row SCALE carries the denominator.)
                h2u, _, _ = tf.rmsnorm_fx_wide(
                    [int(v) for v in np.asarray(step["r1_8"])[:128]],
                    [int(v) for v in np.asarray(step["gamma2"])[:128]])
                exp = np.asarray(h2u, dtype=np.float64) / 256.0
                g = gl.grade_codes_scales(caps, exp, rows=1)
            else:
                g = gl._grade_stage(m, step, meta, caps, plan)
        except Exception as e:                                  # noqa: BLE001
            g = {"equal": False, "error": f"{type(e).__name__}: {e}"}
        good = ok and bool(g.get("equal"))
        n_pass += good
        rows.append(dict(name=m["name"], stage=m["stage"], run_ok=ok,
                         grade=g if not good else {"equal": True},
                         caps=len(caps)))
        eprint(f"  [{m['name']}] run={'ok' if ok else 'FAIL'} "
               f"grade={'PASS' if g.get('equal') else 'FAIL'} caps={len(caps)}")

    # ── the FULL-ROW residual's SECOND grade: reassembly ───────────────────
    rfg = grade_resid_full(plan, by_path, step) if "resid_full" in plan else {}
    rf_ok = all(v["equal"] for v in rfg.values()) if rfg else True
    for which, v in rfg.items():
        eprint(f"  [residfull {which}] slices "
               f"{v['slices_with_full_capture']}/{v['n_slices']} fully "
               f"captured, {v['covered']}/{v['d_model']} elements covered, "
               f"reassembled row == golden {which}: {v['equal']} "
               f"(n_diff={v['n_diff']})")
        for n in v["notes"]:
            eprint(f"    note: {n}")

    # ── norm2m phase 2: MOVE the captured INT32, then the ext-armed norm ────
    n2m = {}
    if "norm2m" in plan:
        sf = by_path.get(next(m["path"] for m in plan["stages"]
                              if m["name"] == plan["norm2m"]["sum2_stage"]))
        if sf is None or not sf["captures"]:
            n2m = {"ok": False, "error": "the sum2 chain produced no captures "
                                         "— nothing to move"}
            eprint("  [norm2m] REFUSE: no sum2 captures to move")
        else:
            eprint("  [norm2m] moving the MXE accumulator into RMS_SUM2 "
                   "(0 host adds, 0 host arithmetic)")
            n2m = move_and_norm(plan, step, sf["captures"], out, args)
            s = n2m["sensitivity"]
            eprint(f"  [norm2m] sum2 moved = {n2m['sum2_moved_u32']} "
                   f"(u32 verbatim, {n2m['n_beats']} chained beats); "
                   f"norm graded {'PASS' if n2m['ext']['green'] else 'FAIL'}; "
                   f"discriminator (+{s['delta_min']}, the first delta that "
                   f"leaves golden's r bucket r={s['r_base']}->"
                   f"{s['r_at_delta']}) "
                   f"{'CAUGHT' if n2m['discriminator_caught'] else 'MISSED'}; "
                   f"+1 control (inside the bucket) "
                   f"{'GREEN as golden predicts' if n2m['plus_one_absorbed_measured'] else 'RED — golden says it must not be'}")
    n2m_ok = bool(n2m.get("ok", True))

    verdict = (n_pass == len(rows)) and rf_ok and n2m_ok
    res = dict(executor=args.executor, n_stages=len(rows), n_pass=int(n_pass),
               wall_s=round(wall, 1), ok=bool(verdict), rows=rows,
               resid_full_reassembly=rfg, resid_full_ok=bool(rf_ok),
               norm2m=n2m, norm2m_ok=bool(n2m_ok),
               claim_note=plan["claim_note"])
    (out / f"flight_result_{args.executor}.json").write_text(
        json.dumps(res, indent=1))
    print(f"N-FLIGHT ({args.executor}): {n_pass}/{len(rows)} stages green, "
          f"full-row residual reassembly "
          f"{sum(1 for v in rfg.values() if v['equal'])}/{len(rfg)} rows "
          f"== golden, wall {wall/60:.1f} min -> "
          f"{'PASS' if verdict else 'FAIL'}")
    if n2m:
        print(f"NORM2M (0 host adds): sum2={n2m.get('sum2_moved_u32')} "
              f"computed and ACCUMULATED on the tile, moved verbatim into "
              f"RMS_SUM2; full-row norm "
              f"{'PASS' if n2m.get('ext', {}).get('green') else 'FAIL'}, "
              f"discriminator "
              f"{'CAUGHT' if n2m.get('discriminator_caught') else 'MISSED'} "
              f"-> {'PASS' if n2m_ok else 'FAIL'}")
    print(f"record -> {out}/flight_result_{args.executor}.json")
    return 0 if verdict else 1


def selftest(args) -> int:
    """The reassembly grader must be able to go RED. Host-only, no executor.

    A grader that always passes proves nothing (the discipline of
    gen_layer_ops §6b / cap_decode case 5). Drive grade_resid_full with
    SYNTHETIC captures and check all four failure shapes.
    """
    del args
    fails = 0

    def chk(n, cond, extra=""):
        nonlocal fails
        print(f"  [{n}] {'ok' if cond else 'FAIL'} {extra}")
        if not cond:
            fails += 1

    rng = np.random.default_rng(20260731)
    dm, cols = 3584, RESID_SLICE
    n_sl = dm // cols
    step = {f"{w}_bits": rng.integers(0, 0x3C00, dm, dtype=np.uint16)
            for w in RESID_ROWS}
    plan = {"stages": [], "resid_full": {"d_model": dm, "cols": cols,
                                         "n_slices": n_sl, "rows": {}}}
    for w in RESID_ROWS:
        names = []
        for k in range(n_sl):
            nm = f"x_{w}_{k:02d}"
            plan["stages"].append({"name": nm, "path": f"/tmp/{nm}",
                                   "stage": f"resid_{w}", "cols": cols,
                                   "off": k * cols})
            names.append(nm)
        plan["resid_full"]["rows"][w] = names

    def caps_for(w, off, bits=None):
        b = step[f"{w}_bits"][off:off + cols] if bits is None else bits
        return [{"tag": f"lrd_{i}", "sem": "lrd", "seq": i, "addr": 0x1088,
                 "mask": 0xFFFF, "value": int(v), "i": i}
                for i, v in enumerate(b)]

    def by_path(mut=None):
        d = {}
        for m in plan["stages"]:
            w = m["stage"].split("_")[1]
            caps = caps_for(w, m["off"])
            if mut:
                caps = mut(m, w, caps)
            if caps is None:
                continue
            d[m["path"]] = {"path": m["path"], "captures": caps,
                            "complete": True, "attribution_ok": True}
        return d

    g = grade_resid_full(plan, by_path(), step)
    chk("1 exact captures reassemble to golden",
        all(v["equal"] for v in g.values())
        and all(v["covered"] == dm for v in g.values()),
        f"r1 covered {g['r1']['covered']}/{dm}, r2 "
        f"{g['r2']['covered']}/{dm}")

    def flip(m, w, caps):
        if m["off"] == 17 * cols and w == "r1":
            caps = [dict(c) for c in caps]
            caps[5]["value"] ^= 1
        return caps
    g = grade_resid_full(plan, by_path(flip), step)
    chk("2 ONE flipped fp16 bit in slice 17 -> RED",
        (not g["r1"]["equal"]) and g["r2"]["equal"]
        and g["r1"]["first_idx"] == 17 * cols + 5,
        f"n_diff={g['r1']['n_diff']} at {g['r1']['first_idx']}")

    def drop(m, w, caps):
        return None if (w == "r2" and m["off"] == 3 * cols) else caps
    g = grade_resid_full(plan, by_path(drop), step)
    chk("3 a MISSING slice -> RED (hole, not silent pass)",
        (not g["r2"]["equal"]) and g["r2"]["covered"] == dm - cols
        and g["r2"]["slices_with_full_capture"] == n_sl - 1,
        f"covered {g['r2']['covered']}/{dm}, "
        f"{len(g['r2']['notes'])} note(s)")

    def short(m, w, caps):
        return caps[:-1] if (w == "r1" and m["off"] == 0) else caps
    g = grade_resid_full(plan, by_path(short), step)
    chk("4 a SHORT slice (127/128 caps) -> RED",
        not g["r1"]["equal"] and g["r1"]["covered"] == dm - cols,
        f"notes={g['r1']['notes'][:1]}")

    def swap(m, w, caps):
        # the exact shape the off-shadowing bug produced: every slice
        # carrying row-0 content
        return caps_for(w, 0) if w == "r1" else caps
    g = grade_resid_full(plan, by_path(swap), step)
    chk("5 every slice carrying slice-0 content -> RED",
        not g["r1"]["equal"] and g["r1"]["n_diff"] > cols,
        f"n_diff={g['r1']['n_diff']}")

    # ── 6-8: the norm2m MOVE proof must be able to REFUSE ──────────────────
    # assert_move_only is the entire evidence that the host did not compute
    # the number. If it cannot fail, the claim is decoration.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def _prog(writes):
            p = Path(td) / f"mv_{len(list(Path(td).iterdir()))}.jsonl"
            p.write_text("".join(json.dumps(o) + "\n" for o in writes))
            return p

        good = _prog([{"op": "note", "s": "x"},
                      {"op": "w", "a": RMS_SUM2_ADDR, "d": 120498},
                      {"op": "cap", "a": 0x3204, "m": 0xFFFFFFFF, "tag": "t"}])
        chk("6 a genuine move passes", assert_move_only(good, 120498) == 120498)
        cases = {
            "7a a TRANSFORMED value is refused":
                (_prog([{"op": "w", "a": RMS_SUM2_ADDR, "d": 120499}]), 120498),
            "7b a MISSING write is refused":
                (_prog([{"op": "w", "a": 0x1094, "d": 1}]), 120498),
            "7c a SECOND write is refused":
                (_prog([{"op": "w", "a": RMS_SUM2_ADDR, "d": 120498},
                        {"op": "w", "a": RMS_SUM2_ADDR, "d": 120498}]), 120498),
            "7d a READ-BACK of the armed value is refused":
                (_prog([{"op": "w", "a": RMS_SUM2_ADDR, "d": 120498},
                        {"op": "r", "a": RMS_SUM2_ADDR, "m": 0xFFFF,
                         "e": 0}]), 120498),
        }
        for label, (p, v) in cases.items():
            try:
                assert_move_only(p, v)
                chk(label, False, "the move proof accepted it")
            except AssertionError as e:
                chk(label, True, f"{str(e).split(': ')[-1][:56]}…")
        # 8: sign discipline — the u32 that leaves the MXE is what gets armed
        neg = _prog([{"op": "w", "a": RMS_SUM2_ADDR, "d": 0xFFFFFFFF}])
        chk("8 the raw u32 is compared, not a re-signed value",
            assert_move_only(neg, 0xFFFFFFFF) == 0xFFFFFFFF,
            "a negative accumulator would arrive as >= 2^31 and "
            "LayerScript.nsum2's [27:0] bound would refuse it upstream")

    print(f"NARROW_FLIGHT SELFTEST: {'FAIL' if fails else 'PASS'} "
          f"(fails={fails})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    g = sub.add_parser("gen")
    g.add_argument("--npz", default=str(REPO / "build/f2_layer/layer_step_full.npz"))
    g.add_argument("--out", default=str(DEFAULT_OUT))
    g.add_argument("--sum2-rows-per-desc", type=int, default=1,
                   help="staged rows per OP_GEMM_OS descriptor in the norm2m "
                        "sum2 chain (default 1 = the K=128 descriptor "
                        "geometry already proven on silicon; the on-tile "
                        "accumulation makes the choice free)")
    f = sub.add_parser("fly")
    f.add_argument("--executor", choices=("sim", "hw"), required=True)
    f.add_argument("--binary", default=None)
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--timeout-s", type=int, default=3600)
    f.add_argument("--max-batch-mb", type=int, default=8,
                   help="byte budget per executor invocation (a single "
                        "larger stage becomes its own batch)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a)
    return gen(a) if a.cmd == "gen" else fly(a)


if __name__ == "__main__":
    sys.exit(main())
