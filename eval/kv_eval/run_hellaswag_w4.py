#!/usr/bin/env python3
# run_hellaswag_w4.py — B3 stage-6 ADOPTION GATE: HellaSwag with Qwen2.5-7B
# under the B3 WEIGHT tiers (KV cache = identity hook on the S4/S5 code path,
# so deltas isolate the WEIGHT path — the mirror image of S5's design).
#
#   python run_hellaswag_w4.py --wtier w4-tile-a --limit 1000
#
# wtiers: base (stock mlx-4bit g64 = THE baseline, identical to S5's base)
#       | w-int8 (S8 golden feed) | w4-tile-a (B3 as landed)
#       | w4-g32-b | w4-g16-b (the finer-grouping lever, realization B form)
#
# GATES (all counts recorded in the JSON; refuses to run on any failure):
#   1. test_twin_bitexact.py — the S5 KV-codec twin gate (per S5 rule every
#      eval JSON carries its counts; the KV path here is identity but runs
#      through the same certified plumbing).
#   2. w4_weights_mlx.weight_twin_selftest — vectorized W4 chain vs the
#      golden weight_codec arbiter on synthetic tiles, exact.
#   3. w4_weights_mlx.mlx_pack_selftest — packed-representation bit order
#      proven against mx.quantize/mx.dequantize on this MLX build.
#   4. install-time golden stripe gate — sampled REAL stripes (incl. each
#      tensor's amax stripe) re-derived through golden compress/wfeed, exact.
#
# Scoring is byte-for-byte the S5 semantics (fresh cache, batch 1, one
# full-sequence forward per (context, ending)); --preflight-against proves
# harness equivalence by exact per-doc comparison with committed S5 samples.

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kvq_cache_mlx import make_kvq_caches  # noqa: E402
import w4_weights_mlx as W4                # noqa: E402


def certify_twin():
    r = subprocess.run([sys.executable, str(HERE / "test_twin_bitexact.py")],
                       capture_output=True, text=True)
    line = " | ".join((r.stdout.strip().splitlines() or ["<no output>"])[-2:])
    if r.returncode != 0:
        sys.exit(f"ABORT: twin bit-exact gate failed:\n{r.stdout}\n{r.stderr}")
    print(f"[gate] {line}")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-4bit")
    ap.add_argument("--wtier", required=True, choices=list(W4.WTIERS))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--weights-cache",
                    default=str(REPO / "build/s8_weights/Qwen2.5-7B-4bit"))
    ap.add_argument("--outdir", default=str(REPO / "docs/results/b3_w4_adoption"))
    ap.add_argument("--preflight-against", default=None,
                    help="committed S5 samples.json: assert exact per-doc "
                         "equality (harness-equivalence proof), tag output "
                         "as preflight")
    args = ap.parse_args()

    gate_line = certify_twin()
    n_wtwin = W4.weight_twin_selftest()
    print(f"[gate] WEIGHT TWIN GATE: checks={n_wtwin} fails=0")
    n_pack = W4.mlx_pack_selftest()
    print(f"[gate] MLX PACK GATE: checks={n_pack} fails=0")

    cache_dir = Path(args.weights_cache)
    if args.wtier != "base" and not (cache_dir / "meta.json").exists():
        sys.exit(f"ABORT: S8 weight cache not found at {cache_dir}")

    import lm_eval
    import mlx_lm
    from huggingface_hub import hf_hub_download
    from mlx_lm.evaluate import MLXLM
    from tqdm import tqdm

    class KVQMLXLM(MLXLM):
        tier = "identity"        # ALWAYS identity here: weight tiers only
        group = 128
        outliers = None

        def _score_full(self, toks):
            cache = make_kvq_caches(self._model, self.tier, self.group,
                                    self.outliers)
            inp = mx.array(toks[:-1])[None]
            logits = self._model(inp, cache=cache)
            logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
            targets = mx.array(toks[1:])[None]
            sc = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
            ig = mx.argmax(logits, axis=-1) == targets
            mx.eval(sc, ig)
            mx.clear_cache()
            return np.array(sc[0]), np.array(ig[0])

        def loglikelihood(self, requests):
            out = []
            for req in tqdm(requests):
                ctx, cont = req.args
                prefix = self._tokenize([ctx])[0]
                full = self._tokenize([ctx + cont])[0]
                assert 0 < len(prefix) < len(full) <= 4096, \
                    (len(prefix), len(full))
                sc, ig = self._score_full(full)
                out.append((float(sc[len(prefix) - 1:].sum()),
                            bool(ig[len(prefix) - 1:].all())))
            return out

    lm = KVQMLXLM(args.model, batch_size=1, use_chat_template=False)

    if args.wtier == "base":
        wrec = {"wtier": "base", "transform": "none (stock mlx-4bit g64)"}
    else:
        print(f"[install] transforming weights → {args.wtier}")
        wrec = W4.install_tier(lm._model, args.wtier, cache_dir,
                               expect_model=args.model)
        print(f"[gate] WEIGHT INSTALL GATE: tensors={wrec['tensors']} "
              f"golden_stripes={wrec['golden_stripes']} "
              f"stripe_checks={wrec['golden_stripe_checks']} fails=0 "
              f"cast_max_rel={wrec['composite_cast_max_rel']:.3e} "
              f"subnormal={wrec['subnormal_scales']} "
              f"dequant_probe_max_ulp={wrec['install_dequant_probe_max_ulp']:.3f}")

    cfg = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    res = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"],
                                  limit=args.limit, log_samples=True)
    h = res["results"]["hellaswag"]
    samples = [{"doc_id": s["doc_id"], "acc": s["acc"], "acc_norm": s["acc_norm"]}
               for s in res["samples"]["hellaswag"]]

    outdir = Path(args.outdir)
    preflight = None
    if args.preflight_against:
        want = {d["doc_id"]: d for d in
                json.loads(Path(args.preflight_against).read_text())}
        bad = [s["doc_id"] for s in samples
               if s["doc_id"] not in want
               or want[s["doc_id"]]["acc"] != s["acc"]
               or want[s["doc_id"]]["acc_norm"] != s["acc_norm"]]
        # version-equality vs the committed base RECORD (drift hardening)
        base_rec_p = Path(args.preflight_against.replace(".samples.json",
                                                         ".json"))
        vers_ok = None
        if base_rec_p.exists():
            bv = json.loads(base_rec_p.read_text()).get("versions", {})
            import mlx_lm as _mlxlm
            import lm_eval as _lmev
            now = {"mlx": mx.__version__, "mlx_lm": _mlxlm.__version__,
                   "lm_eval": _lmev.__version__, "numpy": np.__version__}
            vers_ok = {k: (bv.get(k), now[k]) for k in now
                       if bv.get(k) != now[k]}
            if vers_ok:
                sys.exit(f"ABORT: version drift vs committed base "
                         f"record: {vers_ok}")
        preflight = {"against": args.preflight_against,
                     "docs": len(samples), "mismatches": len(bad),
                     "mismatched_doc_ids": bad[:20],
                     "version_drift_vs_base_record":
                         vers_ok if vers_ok is not None
                         else "base record not found — versions unchecked"}
        verdict = "PASS" if not bad else "FAIL"
        print(f"PREFLIGHT HARNESS-EQUIVALENCE: docs={len(samples)} "
              f"mismatches={len(bad)} {verdict}")
        if bad:
            sys.exit("ABORT: harness does not reproduce the committed S5 "
                     f"base per-doc outcomes: {bad[:20]}")
    else:
        # tier run: pairing provenance = the preflight ARTIFACT of this
        # outdir, referenced by name+counts (never a bare prose claim)
        pf = sorted(p for p in outdir.glob("*_preflight_base_n*.json")
                    if not p.name.endswith(".samples.json"))
        if pf:
            pr = json.loads(pf[-1].read_text()).get("preflight") or {}
            preflight = {"artifact": pf[-1].name,
                         "docs": pr.get("docs"),
                         "mismatches": pr.get("mismatches"),
                         "timestamp": json.loads(
                             pf[-1].read_text()).get("timestamp")}
        else:
            preflight = ("NO preflight artifact found in outdir — harness "
                         "equivalence with the committed S5 base NOT "
                         "established for this environment; do not quote "
                         "paired deltas from this JSON alone")

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    record = {
        "model": args.model, "wtier": args.wtier, "kv_tier": "identity",
        "task": "hellaswag", "limit": args.limit, "device": "mlx-metal",
        "stderr_method": "lm-eval analytic binomial sqrt(p*(1-p)/(n-1))",
        "acc": h.get("acc,none"), "acc_stderr": h.get("acc_stderr,none"),
        "acc_norm": h.get("acc_norm,none"),
        "acc_norm_stderr": h.get("acc_norm_stderr,none"),
        "twin_gate": gate_line,
        "weight_twin_gate": f"checks={n_wtwin} fails=0",
        "mlx_pack_gate": f"checks={n_pack} fails=0",
        "weight_install_gate": wrec,
        "preflight": preflight,
        "weights": {
            "quantization_of_checkpoint": cfg.get("quantization"),
            "disclosure": "Baseline = the model's real shipped weights "
                          "(MLX 4-bit affine, group_size 64). Weight tiers "
                          "are DOUBLE-quantized on top of that checkpoint: "
                          "mlx-4bit dequant → per-tensor INT8 (the S8 golden "
                          "feed, bytes read from the committed cache) → for "
                          "W4 tiers the B3 weight_codec chain. Effective "
                          "values are installed in the model's own quantized "
                          "format with q/scales/biases constructed directly "
                          "from the golden chain (no re-quantization hop); "
                          "the only narrowing is the f64→fp16 RNE cast of "
                          "the composite scale (measured, recorded) — the "
                          "same 11-bit grade the hardware f16_grade applies. "
                          "All tiers including base share the same fp16 "
                          "dequant grid, so tier deltas are format-"
                          "symmetric. KV cache is an identity hook on the "
                          "S5 code path in every tier: deltas isolate the "
                          "weight path.",
        },
        "model_shape": {k: cfg.get(k) for k in
                        ["num_hidden_layers", "num_attention_heads",
                         "num_key_value_heads", "hidden_size"]},
        "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
                     "lm_eval": lm_eval.__version__,
                     "numpy": np.__version__},
        "timestamp": stamp,
        "method": "Identical scoring semantics to S5 "
                  "(docs/results/s5_eval7b): one full-sequence forward per "
                  "(context, ending), fresh cache, batch_size=1, mlx-lm "
                  "whole-string tokenization. Harness equivalence with the "
                  "committed S5 base is asserted ONLY by the 'preflight' "
                  "field of this JSON (an artifact reference with doc/"
                  "mismatch counts, or an explicit not-established "
                  "marker). Paired per-doc deltas vs the committed S5 base "
                  "samples are the comparable quantity; absolutes carry the "
                  "checkpoint's 4-bit weight penalty.",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "preflight_" if args.preflight_against else ""
    name = f"{args.model.split('/')[-1]}_{tag}{args.wtier}_n{args.limit}.json"
    (outdir / name).write_text(json.dumps(record, indent=1))
    (outdir / name.replace(".json", ".samples.json")).write_text(
        json.dumps(samples, separators=(",", ":")))
    print(f"RESULT {args.model} wtier={args.wtier} n={args.limit}: "
          f"acc={record['acc']:.4f} acc_norm={record['acc_norm']:.4f} "
          f"(stderr {record['acc_norm_stderr']:.4f}) -> {outdir / name}")


if __name__ == "__main__":
    main()
