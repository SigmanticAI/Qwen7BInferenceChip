"""Prove: the FULL 3584 residual computed as 28 x 128 slices on the NARROW
twin is bit-exact vs golden's r1/r2 — i.e. the last-but-one op type needs NO
wide image."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "scripts/fpga/f2")
import gen_layer_ops as gl, tile_exec_bridge as bridge

npz = "build/f2_layer/layer_step_full.npz"
z = np.load(npz, allow_pickle=False)
step = {k: z[k] for k in z.files if k != "meta_json"}
meta = json.loads(str(z["meta_json"]))
step["d_ffn"] = meta["d_ffn"]; step["head_dim"] = meta["head_dim"]
DM = int(meta["D_model"]); C = 128
out = Path("build/resid_full"); out.mkdir(parents=True, exist_ok=True)
BIN = "verif/f2sim/obj_d128_ddr0/f2sim"

for which in ("r1", "r2"):
    got = np.zeros(DM, dtype=np.uint16); nbad = 0
    for k in range(DM // C):
        off = k * C
        man, exp = gl.build_resid_stage(step, which=which, cols=C, off=off,
                                        out_dir=out, name=f"rf_{which}_{k:02d}")
        r = bridge.run_job([man["path"]], executor="sim", binary=BIN,
                           cap_out=str(out / f"rf_{which}_{k:02d}.cap.jsonl"))
        g = gl.grade_resid(r["captures"], exp)
        if not g.get("equal"):
            nbad += 1
            if nbad <= 2: print(f"  slice {k} off={off} FAIL {g}")
        else:
            got[off:off+C] = exp
    full = np.asarray(step[f"{which}_bits"], dtype=np.uint16)
    ok = nbad == 0 and np.array_equal(got, full)
    print(f"{which}: slices {DM//C - nbad}/{DM//C} bit-exact; "
          f"reassembled row == golden {which}: {np.array_equal(got, full)} "
          f"-> {'PASS' if ok else 'FAIL'}")
