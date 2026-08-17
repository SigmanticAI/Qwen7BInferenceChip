#!/usr/bin/env python3
# clock_key.py — the IMAGE-KEYED tile-clock expectation (single source of truth).
#
# ══ WHY THIS FILE EXISTS ═════════════════════════════════════════════════════
# Until 2026-08-05 exactly one tile frequency was legal anywhere (recipe A2 =
# 15.625 MHz), so the numeric anti-vacuity clock gate could be a pair of module
# constants (remote_hw_exec.TILE_CLK_MHZ). The A0 track (62.5 MHz,
# docs/design/CLOCK_LADDER.md) puts TWO legal frequencies in circulation, and
# the CL exposes NO recipe-identity register — the manifest's clock_recipe_a is
# not honored at AFI load (an AFI load resets the MMCMs to the A1 default,
# docs/results/f2_stage2_hw/RESULT.md:38-44). So the ONLY sound binding is:
#
#     the AGFI the host loaded  ->  the recipe that image's netlist was
#                                   TIMING-CONSTRAINED at  ->  the a1 MHz the
#                                   numeric gate must measure.
#
# A fast (A0, 16 ns) image flown at 15.625 is 4x SLOWER than sign-off:
# setup-safe by construction, legal as a deliberate A/B arm (run_recipe=2).
# A slow (A2, 64 ns) image flown at 62.5 is 4x OVER its closed clock: it
# computes garbage while every rc stays 0 — the exact silent-wrong-answer
# class this project refuses. Hence the asymmetric rule enforced here:
# UNDERCLOCK is legal only when explicitly requested; OVERCLOCK never is.
#
# ══ THE DISCIPLINE (same shape as tile_geom.IMAGES_WITH_BASE_ROW_FIX) ════════
# An AGFI on neither side of the table REFUSES — unknown is never "assume A2",
# because the first A0 image will by definition be unknown to an old checkout,
# and defaulting it to A2 would program 15.625 on it silently (harmless) while
# defaulting a NEW A2 image on a tree that grew an A0 default would fly it 4x
# over (fatal). Registration is a one-line commit with a build receipt, exactly
# like the base-row allowlist.
#
# CLI: python3 clock_key.py --selftest        (no instance, no spend)
#      python3 clock_key.py <agfi> [run_recipe]

from __future__ import annotations

import re
import sys

# ── group-A recipe -> clk_extra_a1 MHz ──────────────────────────────────────
# Ground truth is the SDK C table (sdk/userspace/fpga_libs/fpga_clkgen/
# fpga_clkgen_utils.c:75-79, input 100 MHz, a1 = 100*mult/div0):
#   index 0 (A0): mult 15,   div0 24 -> 62.5
#   index 1 (A1): mult 15,   div0 12 -> 125.0   (the load-time default)
#   index 2 (A2): mult 12.5, div0 80 -> 15.625
# byte-identical to the HDK build-side MMCM branches
# (hdk/common/shell_stable/build/scripts/aws_clock_properties.tcl:80-111).
# NEVER trust `fpga-load-clkgen-recipe --help` — its table prints A0's row for
# A2 (docs/results/f2_stage2_hw/RESULT.md:45-50).
RECIPE_A_A1_MHZ = {0: 62.5, 1: 125.0, 2: 15.625}

# `fpga-describe-clkgen` prints a1 rounded to 2 decimals (15.62 for A2 —
# docs/results/f2_stage2_hw/clkgen_final.txt), hence a tolerance rather than
# equality. 0.2 separates every pair of recipe frequencies by >50x the band.
A1_TOL = 0.2

_AGFI_RE = re.compile(r"agfi-[0-9a-f]+")


class ClockKeyRefusal(ValueError):
    """The image-keyed clock expectation could not be established. Callers
    MUST treat this as a refusal to fly (nothing contacted, nothing run)."""


def _fmt_mhz(m: float) -> str:
    return f"{m:g}"


def recipe_cmd(slot: int = 0, recipe_idx: int = 2, sudo: bool = True) -> str:
    """The exact runtime programming command for a group-A recipe index."""
    if recipe_idx not in RECIPE_A_A1_MHZ:
        raise ClockKeyRefusal(f"recipe index {recipe_idx!r} is not a stock "
                              f"group-A recipe {sorted(RECIPE_A_A1_MHZ)}")
    pre = "sudo -n " if sudo else ""
    return f"{pre}fpga-load-clkgen-recipe -S {int(slot)} -a {int(recipe_idx)}"


# ── the per-image table: AGFI -> (constrained recipe index, receipt) ─────────
# "constrained recipe" = the --clock_recipe_a the image's netlist was built
# (and therefore TIMED) with. Every image to date is A2; the first A0 image
# gets registered here (recipe 0) BEFORE it is ever flown — see
# docs/design/CLOCK_LADDER.md §7.2 and scripts/fpga/f2/build_a0_62p5.sh.
IMAGE_RECIPE: dict[str, tuple[int, str]] = {
    "agfi-0f7c93ffa798ecc3f": (2,
        "first-light D=64 image — SINGLE-CLOCK CL (tile on clk_main_a0 @ "
        "250 MHz, docs/results/f2_firstlight/RESULT.md); clkgen a1 does not "
        "clock this tile, A2 preserves the historical bring-up discipline "
        "harmlessly."),
    "agfi-0ae06ea568e5667ba": (2,
        "I-A / Milestone C-D D=128 two-clock image (built 05efb2a, "
        "--clock_recipe_a A2): 18-job canonical set 27,996 checks at "
        "15.625 MHz (docs/results/f2_stage2_hw/RESULT.md)."),
    "agfi-0cc7aa798fe3abce2": (2,
        "narrow image #1 (layer window) — A2 lineage "
        "(docs/results/prompt_on_chip/NARROW_LANE_RESULT.md)."),
    "agfi-09e947873048a6877": (2,
        "narrow image #2 (R4) — 93/93 flight at 15.625 MHz "
        "(docs/results/prompt_on_chip/SIX_OF_SIX_RESULT.md)."),
    "agfi-006b1314fcbbb3505": (2,
        "narrow image #3 (E-lane walker, @47a5a3a) "
        "(docs/results/prompt_on_chip/ELANE_WALKED_CHAIN_RESULT.md)."),
    "agfi-0ecab46b8a8376b21": (2,
        "b64v4 / b64_05b 0.5B image — 4,380 programs bit-exact at "
        "15.625 MHz, per-invocation gate ON (build/p05b_hw_check2.log, "
        "2026-08-03)."),
    "agfi-0a345ddb51285e847": (2,
        "DDR bring-up image apex-b64-ddr1-20260805 — TIMING MET WNS +0.392 "
        "at A2 (docs/results/prompt_on_chip/DDR_BRINGUP_RESULT.md)."),
    "agfi-0183a4b88c8d21163": (2,
        "E-5 convergence image apex-convergence-20260805 — Slack MET +0.277 "
        "at A2 (docs/results/prompt_on_chip/SELF_RUNNING_CARD_RESULT.md)."),
    "agfi-0bc20880b50f5faba": (2,
        "E-6 convergence2 image apex-convergence2-20260805 — Slack MET "
        "+0.297 at A2 (docs/results/prompt_on_chip/E6_ON_SILICON.md)."),
    "agfi-04236db8bb2899ac0": (2,
        "A2 COMBINED image apex-comb-20260814 (@8fb83b1) — WNS +0.349; "
        "manifest A2 (s3 dcps/comb-2026_08_14-210124.Developer_CL.tar). "
        "Carries the e7 emission fix (qs rebias flops) + all hardenings. "
        "NOTE: W4 did NOT elaborate (the synth tcl ignores APEX_CL_W4EN) "
        "- W4 debut needs the next spin with the knob actually wired."),
    "agfi-0ffb83a83f00524c0": (0,
        "A0 62.5MHz image apex-a0r-20260814 — WNS +0.015 TNS 0 (build 4, "
        "the 3-stage sc_mem write pipeline); clock_recipe_a=A0 from the "
        "build's OWN manifest (s3 dcps/a0r-2026_08_14-203318.Developer_CL"
        ".tar). Load with fpga-load-clkgen-recipe -S 0 -a 0 and VERIFY "
        "clk_extra_a1 = 62.50. THE 4x TILE CLOCK."),
    "agfi-030a812cd224b409d": (2,
        "SETTLE-FIXED image apex-settle-20260813 (@b4c53ca) — WNS +0.279 "
        "TNS 0; clock_recipe_a=A2 from the build's OWN manifest "
        "(s3 dcps/settle-2026_08_13-205459.Developer_CL.tar). S2_SKS "
        "walker settle state + hd_pres_q replay defer on the registered "
        "kv_eng_sel (fix-flight-1 verdict). The e7-in-token-loop fix, "
        "image 2."),
    "agfi-09c8f18fa5dd29c86": (2,
        "LKV-HARDENED image apex-lkv-20260812 (@fa603d3) — WNS +0.219 "
        "TNS 0; clock_recipe_a=A2 from the build's OWN manifest "
        "(s3 dcps/lkv-2026_08_12-181555.Developer_CL.tar). dont_touch "
        "l_kv_map_q + registered kv_eng_sel_q fanned to all three "
        "consumer cones (the D-033 flop pattern) + the STOREKV-paced "
        "e7 ATT builder. The e7-in-token-loop fix image."),
    "agfi-0e103916142fa216e": (2,
        "D-033 SNOOP-FLOP FIX image apex-snpflop-20260811 (@f2f2dc3) — "
        "WNS +0.356 TNS 0; clock_recipe_a=A2 from the build's OWN manifest "
        "(s3 dcps/snpflop-2026_08_11-080839.Developer_CL.tar). Bank "
        "snp_valid driven by wc_snp_valid_q_reg FDRE (forensics10) — the "
        "replica-fold failure mode is structurally impossible."),
    "agfi-0500f4afe435b5e71": (2,
        "MERGED image apex-full-20260811 (@a8689a6) — fix + rung-3 "
        "FFN-DOWN + W4 ingest + fuel prefetch; WNS +0.356; manifest A2 "
        "(s3 dcps/merged-2026_08_11-080839.Developer_CL.tar)."),
    "agfi-01ebc7b159ce88d56": (2,
        "SLEDGEHAMMER+KEEP image apex-sh-20260810 (@fb0bd44) — WNS +0.426 "
        "TNS 0; clock_recipe_a=A2 from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_10-184003."
        "Developer_CL.tar :: to_aws/2026_08_10-184003.manifest.txt). "
        "Whole-bank dont_touch + kept snoop write-enable chain "
        "(wc_snp_valid_k) — the cache-sweep fix, forensics-confirmed "
        "6085 composite cells (was 24). THE walked-attention fix."),
    "agfi-0bf7b3a02afd1c631": (2,
        "THE KILL-SHOT image apex-ks-20260810 (@0a14888) — WNS +0.397 "
        "TNS 0; clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_10-065443."
        "Developer_CL.tar :: to_aws/2026_08_10-065443.manifest.txt). "
        "ram_style=registers + dont_touch on sc_val — the first fix "
        "candidate against the Vivado cache-sweep (TERM_READ_VERDICT.md "
        "follow-up; netlist forensics passes 1-4)."),
    "agfi-080a4f90f14a51bca": (2,
        "DBG-v5 term-split image apex-dbg5-20260808 (@d1794be) — WNS "
        "+0.445 TNS 0; clock_recipe_a=A2 read from the build's OWN "
        "manifest (s3://apex-f2-dcp-099597653601/dcps/2026_08_08-080441."
        "Developer_CL.tar :: to_aws/2026_08_08-080441.manifest.txt). "
        "DBG[9:8]={sq,sc} term split + acceptance-latched ctx + "
        "per-engine stale — the sc-vs-sq decider."),
    "agfi-0ecb3a62da3fda9bd": (2,
        "THE FIX CANDIDATE apex-scfix-20260808 (@50088f4) — WNS +0.166 "
        "TNS 0; clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_08-060156."
        "Developer_CL.tar :: to_aws/2026_08_08-060156.manifest.txt). "
        "seq_walker_comp sc_val defensively rewritten (the walked-"
        "attention stale); carries DBG v2-v4 + the FFN interleave."),
    "agfi-0d125ca0e76988260": (2,
        "DBG-v4 image apex-dbg4-20260808 (@1a3b64f) — WNS +0.361 TNS 0; "
        "clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_08-003725."
        "Developer_CL.tar :: to_aws/2026_08_08-003725.manifest.txt). "
        "WALK_DBG v4 adds [5]=commit-engine OR, [4]=live l_kv_map[0] — "
        "the final attribution bit for the composite-cache stale."),
    "agfi-07db100f1c3e55e1d": (2,
        "DBG-v3 image apex-dbg3-20260807 (@962a4bc) — WNS +0.352 TNS 0; "
        "clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_07-224834."
        "Developer_CL.tar :: to_aws/2026_08_07-224834.manifest.txt). "
        "WALK_DBG v3: [31:24] snp count, [23:16] err req_idx, [15:10] OR "
        "of commit-time snoop addrs, [9:8] {qs,eng}, [3:2] commit@1/@0 "
        "seen, [1:0] {stale,frame} — the write-vs-read decider."),
    "agfi-03064a1c113e19c0b": (2,
        "DBG-v2 image apex-dbg2-20260807 (@8c1c8db tree + squant pipeline) "
        "— WNS +0.238 TNS 0; clock_recipe_a=A2 read from the build's OWN "
        "manifest (s3://apex-f2-dcp-099597653601/dcps/2026_08_07-194511."
        "Developer_CL.tar :: to_aws/2026_08_07-194511.manifest.txt). "
        "WALK_DBG v2: [31:24] snoop-row count, [23:16] first-error "
        "req_idx, [9:8] {qs,eng}, [1:0] {stale,frame} — the seam-naming "
        "instrument for the walked-attention stale differential."),
    "agfi-07245eb04d8bede9e": (2,
        "A2+WALK_DBG image apex-a2dbg-20260807 (@8ebdf52) — WNS +0.453 "
        "TNS 0; clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcps/2026_08_07-091121."
        "Developer_CL.tar :: to_aws/2026_08_07-091121.manifest.txt). "
        "Adds WALK_DBG (0x98) err_frame/err_stale split for the "
        "walked-attention differential; RTL otherwise the e7e8 tree."),
    "agfi-04cfe164ba90b8ab0": (2,
        "E-7/E-8 image apex-e7e8-20260806 (@a772713) — Slack MET +0.319; "
        "clock_recipe_a=A2 read from the build's OWN manifest "
        "(s3://apex-f2-dcp-099597653601/dcp/e7_2026_08_06-071435."
        "Developer_CL.tar :: to_aws/2026_08_06-071435.manifest.txt), not "
        "from the AFI description "
        "(docs/results/prompt_on_chip/WALKED_ATTENTION_DEFECT.md)."),
}

_REGISTER_HOWTO = (
    "Register it in scripts/fpga/f2/clock_key.py IMAGE_RECIPE with the "
    "recipe index its DCP was built at (--clock_recipe_a, echoed in the "
    "build log banner 'clock_recipe_a : Ax' and recorded in the manifest) "
    "plus a receipt (result doc / build log path). One line, one commit — "
    "same discipline as tile_geom.IMAGES_WITH_BASE_ROW_FIX.")


def expected_clock(agfi: str, *, run_recipe=None, table=None) -> dict:
    """The a1-MHz expectation for flying image `agfi`.

    run_recipe : optional group-A recipe index (int or int-string, e.g. from
        $APEX_F2_RUN_RECIPE) to DELIBERATELY run the image at, instead of the
        recipe it was constrained at. Legal only at or BELOW the constrained
        frequency (underclock — setup-safe by construction, the §7.3 A/B arm
        of docs/design/CLOCK_LADDER.md). Overclock REFUSES.
    table : test injection point only; production callers never pass it.

    Returns {'agfi','constrained_recipe','run_recipe','mhz','tol',
             'recipe_cmd_a','underclocked','why'}.
    Raises ClockKeyRefusal — callers must refuse to fly, loudly.
    """
    t = IMAGE_RECIPE if table is None else table
    a = (agfi or "").strip()
    if not a:
        raise ClockKeyRefusal(
            "the clock gate is IMAGE-KEYED and no AGFI was given: pass the "
            "image id (--agfi / $APEX_F2_AGFI) so the expected tile clock can "
            "be bound to the netlist that is actually flying. There is no "
            "safe default with two legal frequencies in circulation.")
    if not _AGFI_RE.fullmatch(a):
        raise ClockKeyRefusal(f"{a!r} does not look like an AGFI id")
    if a not in t:
        raise ClockKeyRefusal(
            f"AGFI {a} is not in clock_key.IMAGE_RECIPE — its constrained "
            f"tile clock is UNKNOWN, so no frequency the card reports can be "
            f"attested as correct. {_REGISTER_HOWTO}")
    cons, why = t[a]
    if cons not in RECIPE_A_A1_MHZ:
        raise ClockKeyRefusal(
            f"IMAGE_RECIPE[{a}] carries recipe index {cons!r}, which is not "
            f"a stock group-A recipe {sorted(RECIPE_A_A1_MHZ)} — fix the "
            f"table entry")
    run = cons
    if run_recipe is not None:
        try:
            run = int(run_recipe)
        except (TypeError, ValueError):
            raise ClockKeyRefusal(
                f"run_recipe {run_recipe!r} is not an integer recipe index")
        if run not in RECIPE_A_A1_MHZ:
            raise ClockKeyRefusal(
                f"run_recipe {run} is not a stock group-A recipe "
                f"{sorted(RECIPE_A_A1_MHZ)}")
        if RECIPE_A_A1_MHZ[run] > RECIPE_A_A1_MHZ[cons]:
            raise ClockKeyRefusal(
                f"REFUSING an OVERCLOCK: image {a} is constrained at recipe "
                f"A{cons} = {_fmt_mhz(RECIPE_A_A1_MHZ[cons])} MHz; "
                f"run_recipe A{run} = {_fmt_mhz(RECIPE_A_A1_MHZ[run])} MHz "
                f"is ABOVE its closed clock. Over sign-off the tile computes "
                f"garbage while every rc stays 0. No override exists for "
                f"this direction — build the image at the faster recipe "
                f"instead (docs/design/CLOCK_LADDER.md §7.1).")
    mhz = RECIPE_A_A1_MHZ[run]
    under = mhz < RECIPE_A_A1_MHZ[cons]
    return {
        "agfi": a,
        "constrained_recipe": cons,
        "run_recipe": run,
        "mhz": mhz,
        "tol": A1_TOL,
        "recipe_cmd_a": run,           # feed to recipe_cmd(slot, this)
        "underclocked": under,
        "why": (f"image {a} constrained at recipe A{cons} "
                f"({_fmt_mhz(RECIPE_A_A1_MHZ[cons])} MHz)"
                + (f"; DELIBERATE UNDERCLOCK ARM at recipe A{run} "
                   f"({_fmt_mhz(mhz)} MHz) — setup-safe, disclosed"
                   if under else "")
                + f" — {why}"),
    }


# ══════════════════════════════ selftest ════════════════════════════════════

def selftest() -> int:
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{' ' + extra if extra else ''}")
        if not cond:
            fails += 1

    def refuses(name, fn):
        try:
            fn()
            chk(name, False, "(did NOT refuse)")
        except ClockKeyRefusal as e:
            chk(name, True, f"-> {str(e)[:64]}…")

    print("CLOCK_KEY SELFTEST")
    # table integrity: every entry resolves and its cmd matches its index
    for a in IMAGE_RECIPE:
        e = expected_clock(a)
        okrow = (_AGFI_RE.fullmatch(a) is not None
                 and e["mhz"] == RECIPE_A_A1_MHZ[e["constrained_recipe"]]
                 and recipe_cmd(0, e["recipe_cmd_a"]).endswith(
                     f"-a {e['recipe_cmd_a']}")
                 and not e["underclocked"])
        chk(f"table: {a} -> A{e['constrained_recipe']} "
            f"{_fmt_mhz(e['mhz'])} MHz", okrow)
    # the recipe table itself (the values every claim hangs on)
    chk("recipes: A0=62.5 A1=125 A2=15.625",
        RECIPE_A_A1_MHZ == {0: 62.5, 1: 125.0, 2: 15.625})
    chk("A2 default resolves to 15.625 & '-a 2'",
        expected_clock("agfi-0ecab46b8a8376b21")["mhz"] == 15.625
        and recipe_cmd(0, 2) == "sudo -n fpga-load-clkgen-recipe -S 0 -a 2"
        and recipe_cmd(1, 2, sudo=False)
        == "fpga-load-clkgen-recipe -S 1 -a 2")
    # tolerance separates the printed values of every recipe pair
    seps = all(abs(RECIPE_A_A1_MHZ[i] - RECIPE_A_A1_MHZ[j]) > 10 * A1_TOL
               for i in RECIPE_A_A1_MHZ for j in RECIPE_A_A1_MHZ if i != j)
    chk("tolerance: no two recipes are confusable at ±0.2", seps)
    chk("tolerance: the tool's rounded 15.62 passes A2",
        abs(15.62 - expected_clock("agfi-0ecab46b8a8376b21")["mhz"]) <= A1_TOL)

    # refusals — the gate must BITE
    refuses("empty AGFI refused", lambda: expected_clock(""))
    refuses("None AGFI refused", lambda: expected_clock(None))
    refuses("malformed AGFI refused", lambda: expected_clock("not-an-agfi"))
    refuses("UNKNOWN AGFI refused (never defaults to A2)",
            lambda: expected_clock("agfi-00000000000000000"))
    refuses("A2 image at run_recipe 0 refused (OVERCLOCK)",
            lambda: expected_clock("agfi-0ecab46b8a8376b21", run_recipe=0))
    refuses("A2 image at run_recipe 1 refused (OVERCLOCK, load default)",
            lambda: expected_clock("agfi-0ecab46b8a8376b21", run_recipe=1))
    refuses("non-integer run_recipe refused",
            lambda: expected_clock("agfi-0ecab46b8a8376b21", run_recipe="x"))
    refuses("out-of-table run_recipe refused",
            lambda: expected_clock("agfi-0ecab46b8a8376b21", run_recipe=7))

    # a (hypothetical) registered A0 image — via the test injection table,
    # NEVER by mutating the real one
    fake = dict(IMAGE_RECIPE)
    fake["agfi-0aaaaaaaaaaaaaaaa"] = (0, "selftest-only fake A0 image")
    e0 = expected_clock("agfi-0aaaaaaaaaaaaaaaa", table=fake)
    chk("A0 image resolves to 62.5 & '-a 0'",
        e0["mhz"] == 62.5 and e0["recipe_cmd_a"] == 0
        and not e0["underclocked"]
        and recipe_cmd(0, 0).endswith("-a 0"))
    eu = expected_clock("agfi-0aaaaaaaaaaaaaaaa", run_recipe=2, table=fake)
    chk("A0 image at run_recipe 2 = legal UNDERCLOCK arm (15.625, flagged)",
        eu["mhz"] == 15.625 and eu["underclocked"]
        and "UNDERCLOCK" in eu["why"])
    chk("A0 image at run_recipe '0' (env string) = itself, not underclocked",
        expected_clock("agfi-0aaaaaaaaaaaaaaaa", run_recipe="0",
                       table=fake)["mhz"] == 62.5)
    refuses("A0 image at run_recipe 1 (125 MHz) refused — above ITS clock",
            lambda: expected_clock("agfi-0aaaaaaaaaaaaaaaa", run_recipe=1,
                                   table=fake))
    # a registered-but-garbage table row must refuse, not resolve
    fake["agfi-0bbbbbbbbbbbbbbbb"] = (9, "corrupt row")
    refuses("corrupt table row (recipe 9) refused",
            lambda: expected_clock("agfi-0bbbbbbbbbbbbbbbb", table=fake))
    # the real table was not mutated by any of the above
    chk("production table untouched by selftest",
        "agfi-0aaaaaaaaaaaaaaaa" not in IMAGE_RECIPE
        and all(v[0] in RECIPE_A_A1_MHZ for v in IMAGE_RECIPE.values()))

    print(f"CLOCK_KEY SELFTEST: {'FAIL' if fails else 'ALL PASS'} "
          f"(fails={fails}, {len(IMAGE_RECIPE)} images registered)")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    if not sys.argv[1:]:
        print(__doc__ or "clock_key.py <agfi> [run_recipe] | --selftest")
        sys.exit(2)
    try:
        e = expected_clock(sys.argv[1],
                           run_recipe=(sys.argv[2] if len(sys.argv) > 2
                                       else None))
    except ClockKeyRefusal as err:
        print(f"CLOCK_KEY REFUSAL: {err}", file=sys.stderr)
        sys.exit(78)                     # same EX_CONFIG the clock gate uses
    print(f"{e['agfi']}: expect clk_extra_a1 = {e['mhz']:g} MHz "
          f"± {e['tol']} (program: {recipe_cmd(0, e['recipe_cmd_a'])})")
    print(f"  {e['why']}")
    sys.exit(0)
