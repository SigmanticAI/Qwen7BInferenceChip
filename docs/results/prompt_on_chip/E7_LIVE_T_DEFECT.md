# E7-at-live-T: walked score/pv diverges on silicon (OPEN)

**2026-08-12.** First hw flight of the integrated `--mask e7` token loop
(image agfi-0500f4afe435b5e71, card i-0a3f (e7-measure), prompt T=5,
t004_L00): REFUSE — walked attention o8 NOT bit-exact, ALL 14 heads.
`--no-batch` fails identically (batching exonerated).

## The differential package (build/e7_differential/)

Same card-built program pair (walk_t004_L00_{att,qkv}), byte-identical,
both substrates:

| | silicon | twin (obj_tokenloop @a6f3da7) |
|---|---|---|
| program's own checks | **87/87 PASS** | 74 checks, **2 FAIL** (wedge) |
| captures | 3858 | 2962 (= exactly 896 short — the r1 drain never ran) |
| first divergence | capture 25 `ro_w0` = **0x3** | **0xa** (sim == golden per the pre-merge gate) |

So: sim==golden, hw≠golden, and the program's baked checks are
INSENSITIVE to the o8 values (they pass on the wrong-valued hw run) while
the twin wedges before finishing the pair — two DIFFERENT failure modes
from one program pair. The pre-merge `--mask e7` sim gate was green on
engine-built programs; this pair is card-built at the same commit.

## Suspects, in order

1. **Deep-T scale-cache envelope**: T=5 walks read sc entries at addrs
   2..4 — silicon has only ever proven addrs 0..1 (walk_e6/e7 flew
   snp=2). The T-fence-raise validation flight never happened.
2. The e7 ATT builder's KV/rope staging at live T vs the flown E-7
   program's staged form (builder-drift between what was proven and what
   the token loop now emits).
3. The twin wedge is a THIRD signal: if the pair wedges the twin at this
   T but the engine-built sim gate passed, diff the card-built vs an
   engine-built pair for the same (step, layer) — if they differ, the
   builder is environment-sensitive (paths/state), which is its own bug.

## Next debugging session, in order

1. `diff` card-built vs locally-engine-built walk_t004_L00 pair (byte
   compare; then per-op). 2. If identical: FST waveform of the twin wedge
   (the r1 drain stall) — the wedge and the o8 divergence likely share a
   root. 3. The T-sweep discriminator: run the e7 gate at T=2 (prompt 1
   id + 1 token) on BOTH substrates — if hw greens at T=2 and reds at
   T=3+, suspect (1) is confirmed and the fix is RTL-side (sc addressing
   depth), not the builder.

MASK_B remains fully green on silicon (5.4 s/token measured earlier
today) — the demo and the measured ladder are unaffected.

## Pattern analysis (2026-08-12, capture-stream forensics)

- Divergent words: 877/2962 — ALL in the `ro_w*` o8 readout families;
  every fs_/state capture MATCHES. The walk runs; the VALUES are wrong.
- hw magnitudes consistently SMALLER than sim (3v10, 7v17, -4v-18,
  20v41 …): the signature of PV summing FEWER context rows on silicon —
  partial KV staging (an engine's rows missing → ~half-scale o8 at
  GQA=2), NOT noise, NOT a scale constant.
- T-sweep: fails at T=1..5 identically → depth/sc-address hypothesis
  DEAD. The builder's programs are byte-identical card vs engine; sim
  gate green on those bytes.
- The proven walk_e7 program (27/27 on this same image) also
  host-stages KV + walks SCORE+PV. THE DELTA between walk_e7's staging/
  descriptor and _emit_program_e7's ATT program IS the defect surface.

## Next move (mechanical, no card needed)

Op-level structural diff: build/e7_flight/walk_e7.regops.jsonl (proven)
vs build/e7_differential/walk_t004_L00_att.regops.jsonl (broken):
KV-record staging ops (which engine CSRs, order, count), descriptor
words (L_CTRL rope leaves, engine map), phase table. The op class
present/different in the broken program that silicon ignores or
misroutes is the bug. Sim passing means the twin honors a staging form
silicon does not — the D-033 lesson says look for ENGINE-SELECT
sensitivity in the KV write path.

## ROOT CAUSE (2026-08-12, closed by program archaeology)

`build_att_program` stages KV per engine by flipping **L_CTRL bit 15**
(9 writes vs the proven programs' 0-3) as a KVQ CSR-window select. THE
TWIN HONORS THAT BIT; SILICON DOES NOT (reserved in the real l_ctrl
decode) → every KV record lands in engine 0, engine 1 empty → PV
under-accumulates ≈ half. All evidence consistent: only o8 rows wrong,
T-independent, magnitudes low.

Deeper: NO silicon-proven program host-stages KV into BOTH engines.
walk_e7 proved T=1/one record; hostattn is per-head single-window; the
July host-mode demo re-stages per head-job. Multi-engine KV staging
from the host is UNPROVEN SURFACE invented by this builder.

## THE FIX (design decision, next session)

Preferred (b): add EN_STOREKV to the walked mask — the WALKER stages KV
records itself through the snoop path (the D-033-fixed, silicon-proven
commit path; that is the architecture's intent for walked mode). The
ATT program then feeds K/V rows as walk inputs instead of host CSR
staging; engine routing happens in the walker's l_kv_map, proven by
every E-6/E-7 flight. Alternative (a): find/verify a REAL host-side
engine-select CSR in rtl (l_kv_map host access) — requires RTL reading
+ a validation flight either way. Both are builder+maybe-RTL work; the
sim twin must ALSO be fixed to NOT honor L_CTRL[15] (a twin-fidelity
bug: sim modeled a register silicon doesn't have — file with
verif/f2sim).

## ROOT CAUSE CORRECTED (2026-08-12, RTL read)

L_CTRL[17:15] IS real RTL: `l_kv_map_q` (apex_top.sv ~886), the
HOST-MODE engine select, muxed `kv_eng_sel = walk_en_q ? wk_kv_eng_sel
: l_kv_map_q`. The builder's mechanism is architecturally legitimate —
retract "twin-only register". The operative theory is one family older:
**the host branch of this engine-select mux is a D-033-class synthesis
casualty** — l_kv_map_q + the mux live at TILE level, OUTSIDE the
dont_touch bank instance, unprotected. Every green flight to date used
the walker branch (walked mode) or engine 0 (hostattn h6 → group 0;
host-mode demo per-head jobs) — a folded host branch was invisible
until the FIRST engine-1 host staging: this program. Identical RTL,
two build outcomes: sim honors the select, the routed netlist does not.

Consequences:
1. Builder fix UNCHANGED and stronger: EN_STOREKV walked staging uses
   the WALKER branch — proven by every E-6/E-7 flight — and needs NO
   new image.
2. RTL hardening item (next image spin): flop + dont_touch the
   l_kv_map/kv_eng_sel host branch exactly like the snoop bundle
   (D-033 pattern); optionally confirm the fold first by batch-vivado
   on the flying image's DCP (S3 dcps/merged-2026_08_11-080839).
3. The "twin-fidelity bug" is withdrawn — sim is faithful to the RTL;
   the netlist is not.

## Fix-flight 1 on the LKV-HARDENED image (2026-08-13, OPEN)

Image agfi-09c8f18fa5dd29c86 (l_kv_map_q dont_touch + kv_eng_sel_q
registered; STOREKV-paced builder @853f6a8). Card i-0e156fb4b2f9c4406
(terminated; artifacts in build/e7fix_diff/).

- Battery sanity: walk_e6/e7/e7ng/hostattn **193/193** — hardening
  regressed nothing.
- token_loop --mask e7: **NEW failure mode** — the o8 all-heads REFUSE
  is GONE; every layer's ATT file now ABORTS at its own FINAL poll
  (`0x3240 want 0x100, last 0x2000`, op ~10817/10820 — the builder's
  [6] ESTK/ECODE STOREKV verdict read); all 24 QKV files PASS 50/50.
  48 files, 1440 checks, 24 fails (one per ATT tail).
- The o8-correctness question is UNANSWERED: the run aborts before
  grading; my quick hw-vs-sim capture diff was MISALIGNED (the hw
  stream spans 48 files with an abort-shortened first file — align
  per-file by baked cap counts before comparing; artifacts:
  build/e7fix_diff/apex_cap_k7jl_xxg.jsonl + the L00 pair + a sim run
  of the pair).

Next: (1) align the L00 ATT hw caps (slice by tag sequence: the file's
caps start at its first fs_ tag after the previous file's last lrd) and
compare ro_w vs sim — if o8 is NOW CORRECT, the select fix WORKED and
the only bug is the new builder's final-poll spec (0x3240 bit8
semantics on silicon — likely ESTK encoding differs from the twin or
the poll targets a sim-only summary bit); fix the poll, re-fly. (2) If
o8 still wrong, the fold survived the hardening — netlist forensics on
the lkv DCP (S3 dcps/lkv-2026_08_12-181555.Developer_CL.tar).

## Fix-flight 2, settle image agfi-030a812cd224b409d (2026-08-13, OPEN)

Battery 193/193. The settle fixes WORK: every ATT walk now COMPLETES
(87/87 program checks on the L00 pair standalone, full capture ledger —
fix-flight-1's stall is gone, both hazards real and fixed). But the o8
grade REFUSES again: all 14 heads, t004_L00 — the ORIGINAL signature
class, sim-green/silicon-red on identical RTL, THIRD failure layer of
this onion. The registered+dont_touch kv_eng_sel_q did NOT cure the
value corruption.

Evidence package: build/settle_diff/ (settle_l00.caps.jsonl — COMPLETE
hw capture ledger incl. in-window ATT caps this time — + the L00 pair +
regions). Next: the verdict-agent analysis on the COMPLETE ledger —
grade hw ATT o8/fs/ss vs golden (the machinery run: grade_hw.py pattern
from /private/tmp/apex-e7verdict/), and critically the VALUE PATTERN:
half-scale (engine 1 still starved → the fold survives even the flop —
next step netlist forensics on THIS DCP: does kv_eng_sel_q's Q pin
actually drive the kvq bank mux, or did synthesis re-derive AGAIN?) vs
a new pattern (the hd_pres_q defer may have shifted s_q pairing — check
the fs/ss ladders first). The DCP: S3
dcps/settle-2026_08_13-205459.Developer_CL.tar.

## Fix-flight-2 VERDICT (2026-08-13, verdict agent 2 — layer 3 localized)

Grades on the complete L00 ledger: s_q 14/14, s_k 70/70, s_v 560/560
(each head matches its OWN group 5/5, other group 0/5 — **the
engine-select fold is CURED**; both prior fixes work), QKV 144/144,
r1 896/896. ONLY o8 fails (19/896 words) — and hw o8 is EXACTLY
requant(Σ c8'·v8, host RQ) with a solved integer c8' for all 14 heads
(896/896 reconstruction): the PV weight vector alone is corrupted.
Silicon's softmax is SUB-NORMALIZED (Σp' ≈ 0.1-0.5, row-dependent,
non-monotone; h11/h13 exactly c8/2; h08/h09 exact double-requant
numerology). Anything upstream of the ASU normalize divide is PROVABLY
excluded; 12+ transfer functions numerically killed (p/2, p², l²,
zero-dilution, LUT stuck bits, interleave, double-rescale...).

NEXT: the ASU softmax emission / PV-consumer / F6-replay cone — a
third D-033-class unprotected-at-tile-level fold candidate. Moves:
(1) netlist trace of the emission divide/rescale cone in the settle
DCP (S3 dcps/settle-2026_08_13-205459.Developer_CL.tar) — dont_touch
coverage vs the RTL cone (asu_softmax.sv + walker-side consumer +
apex_top ~1049-1090); (2) micro-flight: single-head ATT programs with
synthetic one-hot/controlled-gap scores to map silicon's p-transfer
function (prediction target = the solved c8' table;
/private/tmp/apex-verdict2/ scripts + state2.npz carry all internals).
Builder changes are NOT the fix — stop iterating there.

## NETLIST FORENSICS VERDICT (2026-08-14, devbox9): the netlist is INNOCENT

Full doc: /private/tmp/apex-forensics/VERDICT.md (+ raw q1-q5). Both
hardening spins targeted a provably innocent unit — analytical
kill-shot: (T=3,t=2) and (T=5,t=0) read the SAME cache index with
DIFFERENT weights (3/8 vs 1/16); no function of req_idx can do that.
asu_softmax: every p gets exactly 31 divider iterations (FSM decoded
from INITs). squant: no counter reaches any exponent arithmetic.
Replicas everywhere are faithful.

FIX CLASS: builder/walker SEQUENCING — silicon executes a different
S-4 stream (cs-word/v-beat pairing) than golden's single T-column
requant; the map predicts extra/duplicated final beats at T=2,3 and a
halved-rate pairing at T=5. NEXT (sim-only): trace job_cols + cs-load
+ v-beat sequences per head on the twin for the L00 ATT pair, diff vs
the builder's descriptor expectations. Do NOT spin more hardening
images for this defect.

Also measured today: daemon + mmio opt-ins saved ~0 (the engine cost
is per-file processing: 24 x ~60ms walks + 0.48s TILE_RST sleeps +
drain); anatomy at 3.1: golden 0.92 + engine 2.11 + head 0.19.
# S-4 pairing-schedule algebra notes (Sun Aug 16 18:22:51 PDT 2026)

## Partial algebra (in-session, 2026-08-16)
Decomposition: w(t,T) = 2^-k(t,T) + [t==T-1 && T in {2,3}] * 2^-(k+1)
i.e. base powers PLUS ONE EXTRA HALF-BEAT on the final row at T=2,3
(3/8 = 1/4 + 1/8) — confirms the verdict's "duplicated final beat".
Bases k: T=1:[1]; T=2:[2,2]; T=3:[2,2,2]; T=5:[4,3,3,2,2] = 4-ceil(t/2)
(pair-indexed from the start at T=5; constant at T<=3). No single
closed form over classes A/B fits — the law switches regime between
T<=3 and T=5, suggesting the racing update is BEAT-COUNT-dependent
(consumption overlaps updates only when the stream is long enough).
NEXT (fresh context): read the S-4 RTL beat semantics (what shifts the
scale per v-beat/pair) and derive k from the pipeline depth; then the
fence = hold v-stream until the scale settles (program-side poll).
