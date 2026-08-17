#!/usr/bin/env python3
# token_chat.py — the interactive prompt CLI over token_loop's graded decode.
#
#   sudo python3 scripts/fpga/f2/token_chat.py --engine hw-walked \
#       --ddr-attested                       # on the F2 card
#   python3 scripts/fpga/f2/token_chat.py --engine host-golden   # anywhere
#
# You type a (SHORT) prompt; the model answers; with --engine hw-walked
# every sampling-relevant layer's walked {FPROJ, QKV, OPROJ, RES1} chain
# runs on the REAL tile and is graded bit-exact before any token is
# sampled. The grading discipline is token_loop's verbatim: a host-golden
# reference pass first, then the engine pass, token identity enforced.
#
# THE ENVELOPE (today's silicon, stated up front): per-engine T fences
# in T_FENCE below — hw-walked is T <= token_loop.T_HW_MAX (= 64) total
# steps (prompt + answer), the tile's walk-envelope budget; the ceiling-
# by-ceiling RTL citations live at T_HW_MAX's definition in token_loop.py.
# The REPL auto-shrinks the answer budget to fit and REFUSES over-long
# prompts. Raising 64 is RTL work (projection m-chunking), not a constant.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import token_loop as tl                                        # noqa: E402

rt = tl.rt
MacCensus = tl.MacCensus

# Per-engine session fences (prompt + answer), each the ENGINE'S OWN
# budget — token_loop/run_tinynpu constants, single-sourced (no number is
# restated here):
#   hw-walked : tl.T_HW_MAX (64) — the tile's walk-envelope budget. The
#               binding ceiling (MXE m-height: mxe_ctrl refuses m_dim >
#               M_TILE_MAX=64, and the walker's S2_CHECK refuses EN_QKV
#               descriptors with t_rows above it up front) and every
#               runner-up (walker descriptor + scale cache and KVQ CQ-8
#               at 128, host session at 384) are cited file:line at
#               T_HW_MAX's definition in token_loop.py. Raising it is
#               RTL work (projection m-chunking), not a constant edit.
#   sim-walked: tl.T_SIM_MAX (8) — a Verilator WALL budget (24 flights
#               per token), NOT a hardware limit; token_loop's
#               --probe-steps mode carries the deep-T sim evidence.
#   host-golden: the golden session cap (D-023-gated C-CHUNK range).
T_FENCE = {"hw-walked": tl.T_HW_MAX, "sim-walked": tl.T_SIM_MAX,
           "host-golden": rt.T_SESSION_MAX}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="interactive prompt CLI over token_loop's graded decode")
    ap.add_argument("--engine", default="hw-walked",
                    choices=sorted(tl.ENGINES))
    ap.add_argument("--tokens", type=int, default=16,
                    help="answer budget per prompt (auto-shrunk to the "
                         "engine's T fence)")
    ap.add_argument("--raw", action="store_true",
                    help="feed the text verbatim (completion mode) instead "
                         "of wrapping it in the model's chat template")
    ap.add_argument("--weights-dir", default=str(tl.DEFAULT_WEIGHTS))
    ap.add_argument("--image", default=str(tl.DEFAULT_IMAGE))
    ap.add_argument("--binary", default=None)
    ap.add_argument("--obj", default="obj_tokenloop")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--ddr-attested", action="store_true")
    ap.add_argument("--work", default=str(tl.DEFAULT_WORK))
    ap.add_argument("--tier", default="kvq8", choices=sorted(rt.TIER_MAP))
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print("loading model …", flush=True)
    model = rt.GoldenModel(Path(args.weights_dir))
    eos_raw = model.meta.get("eos_token_id")
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw]) - {None}
    tier = rt.TIER_MAP[args.tier]
    try:
        tok = rt.load_tokenizer(model.meta["model"])
    except Exception as e:                                     # noqa: BLE001
        raise SystemExit(
            f"REFUSE: the chat needs the tokenizer ({type(e).__name__}) — "
            f"pip install transformers (the token-id mode lives in "
            f"token_loop.py, not here)")

    t_max = T_FENCE[args.engine]
    print("=" * 70)
    print(f"APEX TOKEN CHAT — engine={args.engine}  (T<={t_max} "
          f"prompt+answer fence)")
    print("type a short prompt; 'exit' quits. Every walked value is graded")
    print("bit-exact against golden before a token is sampled.")
    print("=" * 70)

    while True:
        try:
            text = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        if args.raw:
            ids = [int(t) for t in tok.encode(text)]
        else:
            # the chat template: the role wrapper the model was TRAINED to
            # converse in ("user says X, now the assistant answers"). Raw
            # text puts it in autocomplete mode instead.
            ids = [int(t) for t in tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True,
                return_dict=False)]
        if len(ids) >= t_max:
            print(f"  [refused] prompt is {len(ids)} tokens; this "
                  f"engine's session fence is T<={t_max} INCLUDING the "
                  f"answer — try a shorter prompt")
            continue
        n_new = min(args.tokens, t_max - len(ids))
        if n_new < args.tokens:
            print(f"  [fence] answer budget shrunk to {n_new} "
                  f"(prompt {len(ids)} + {n_new} = {t_max})")

        # pass 1: host-golden reference (the arbiter for token identity)
        t0 = time.monotonic()
        ref_engine = tl.HostGoldenEngine()
        ref_ctx = {"model": model, "args": args, "work": work, "ids": ids}
        ref_engine.start(ref_ctx)
        ref = tl.decode_pass(model, ids, n_new, tier, args.group, eos,
                             ref_engine, ref_ctx, MacCensus(model.n_layers),
                             None, quiet=True)
        # pass 2: the requested engine, graded
        engine = tl.ENGINES[args.engine]()
        ctx = {"model": model, "args": args, "work": work, "ids": ids}
        engine.start(ctx)
        res = tl.decode_pass(model, ids, n_new, tier, args.group, eos,
                             engine, ctx, MacCensus(model.n_layers),
                             expect_ids=ref["ids"])
        wall = time.monotonic() - t0
        if res["ids"] != ref["ids"]:
            raise SystemExit(f"REFUSE: engine ids {res['ids']} != "
                             f"host-golden {ref['ids']}")

        answer = tok.decode(res["ids"], skip_special_tokens=True)
        n_chains = getattr(engine, "n_walked", 0)
        if args.raw:
            print(f"\n  → {text}{answer}")
        else:
            print(f"\n  → {answer}")
        print(f"    [{args.engine}] {len(res['ids'])} tokens in "
              f"{wall:.1f}s (both graded passes)"
              + (f"; {n_chains} walked chains bit-exact on silicon"
                 if n_chains else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
