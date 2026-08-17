#!/usr/bin/env bash
# d_preflight.sh — MILESTONE D preflight. Step 0 of docs/design/MILESTONE_D_RUNBOOK.md
#
# Answers ONE question: "if we launch the f2.6xlarge right now, is every
# precondition already true, or are we about to pay $1.98/hr to discover a
# problem we could have found for free?"
#
# THIS SCRIPT SPENDS NOTHING AND CHANGES NOTHING.
#   * every AWS call is read-only (describe-* / get-* / sts / pricing)
#   * it never calls run-instances, never modifies a security group, never
#     touches the AFI, never writes outside $TMPDIR
#   * it requires NO instance to exist
# Exit status: 0 = clear to launch. Non-zero = do NOT launch (count of FAILs).
#
# Usage:
#   bash scripts/fpga/f2/d_preflight.sh              # ~30 s, all cheap checks
#   bash scripts/fpga/f2/d_preflight.sh --deep       # + re-run the real-7B sim
#                                                    #   demo end-to-end (~6 min
#                                                    #   of Mac CPU, still $0)
#   bash scripts/fpga/f2/d_preflight.sh --no-aws     # local checks only
#
# Overridable (defaults are the verified values — change only with evidence):
#   REGION AFI AGFI SHELL_VER AMI KEY_NAME PEM SG_NAME SUBNET AZ_OK
#   ITYPE MAX_PRICE REPO VENV_PY WEIGHTS
set -uo pipefail

# ─────────────────────────── configuration ──────────────────────────────────
REGION="${REGION:-us-west-2}"
AFI="${AFI:-afi-036d83cafa00d26ea}"
AGFI="${AGFI:-agfi-0ae06ea568e5667ba}"
SHELL_VER="${SHELL_VER:-0x10212415}"
AMI="${AMI:-ami-07a164f1a402ab274}"
KEY_NAME="${KEY_NAME:-apex-f2}"
PEM="${PEM:-$HOME/.ssh/apex-f2.pem}"
SG_NAME="${SG_NAME:-apex-f2-ssh}"
SUBNET="${SUBNET:-subnet-0ee519f2d304c99c9}"     # us-west-2b (f2 is offered there)
ITYPE="${ITYPE:-f2.6xlarge}"
MAX_PRICE="${MAX_PRICE:-2.50}"                   # sanity ceiling, $/hr on-demand
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VENV_PY="${VENV_PY:-$HOME/.venvs/apex-eval/bin/python}"
WEIGHTS="${WEIGHTS:-$REPO/build/s8_weights/Qwen2.5-7B-4bit}"
SIM_EVIDENCE="${SIM_EVIDENCE:-$REPO/build/prompt_offload/prompt_offload_result.json}"
AWSTO=(--cli-connect-timeout 15 --cli-read-timeout 45)

DEEP=0; DO_AWS=1
for a in "$@"; do
  case "$a" in
    --deep)   DEEP=1 ;;
    --no-aws) DO_AWS=0 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

FAILS=0; WARNS=0; PASSES=0
pass(){ PASSES=$((PASSES+1)); printf '  [PASS] %s\n' "$*"; }
fail(){ FAILS=$((FAILS+1));   printf '  [FAIL] %s\n' "$*"; }
warn(){ WARNS=$((WARNS+1));   printf '  [WARN] %s\n' "$*"; }
info(){                       printf '  [info] %s\n' "$*"; }
sect(){ printf '\n== %s\n' "$*"; }
# expect <label> <got> <want>
expect(){ if [ "$2" = "$3" ]; then pass "$1 = $2"; else fail "$1 = '$2' (want '$3')"; fi; }
have(){ command -v "$1" >/dev/null 2>&1; }
PY="$(command -v python3 || true)"

printf '==================================================================\n'
printf 'APEX MILESTONE D — PREFLIGHT (read-only, $0, no instance required)\n'
printf 'date   : %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ') UTC"
printf 'repo   : %s\n' "$REPO"
printf 'region : %s   afi: %s   type: %s\n' "$REGION" "$AFI" "$ITYPE"
printf 'mode   : %s\n' "$( [ $DEEP = 1 ] && echo 'deep (re-runs the sim demo)' || echo 'fast' )"
printf '==================================================================\n'

# ══════════════════ A. local tree — is the demo we are shipping intact? ══════
sect "A. LOCAL TREE"
if [ -d "$REPO/.git" ] || git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  BR="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  HEADSHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
  pass "git tip: $BR @ ${HEADSHA:0:7}  ($(git -C "$REPO" log -1 --format=%s | cut -c1-64))"
  DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$DIRTY" = "0" ]; then pass "working tree clean"
  else warn "working tree has $DIRTY modified/untracked path(s) — the session log MUST record what they are (parallel lanes are editing this tree)"; fi
else
  fail "not a git repo: $REPO"; HEADSHA=""; BR=""
fi

for f in scripts/fpga/f2/prompt_offload.py scripts/fpga/f2/compute_job.py \
         scripts/fpga/f2/tile_exec_bridge.py scripts/fpga/f2/cap_decode.py \
         scripts/fpga/f2/f2_host_run.py scripts/fpga/f2/trace_to_regops.py \
         run_tinynpu.py verif/top/l3/gen_l3_vectors.py \
         golden/apex_golden/transformer.py golden/apex_golden/attention.py; do
  if [ -f "$REPO/$f" ]; then pass "present: $f"; else fail "MISSING: $f"; fi
done

sect "A2. SELFTESTS (the demo's own gates, host-only, no executor)"
run_selftest(){ # <label> <script> [args...]
  local label="$1"; shift
  local out rc
  out="$("$PY" "$@" 2>&1)"; rc=$?
  if [ $rc -eq 0 ]; then pass "$label — $(printf '%s' "$out" | tail -1 | cut -c1-88)"
  else fail "$label exited $rc — $(printf '%s' "$out" | tail -3 | tr '\n' ' ' | cut -c1-160)"; fi
}
run_selftest "compute_job --selftest"      "$REPO/scripts/fpga/f2/compute_job.py" --selftest
run_selftest "tile_exec_bridge --selftest" "$REPO/scripts/fpga/f2/tile_exec_bridge.py" --selftest
run_selftest "prompt_offload --selftest"   "$REPO/scripts/fpga/f2/prompt_offload.py" --selftest
if "$PY" -c "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)" \
      "$REPO/scripts/fpga/f2/f2_host_run.py" >/dev/null 2>&1; then
  pass "f2_host_run.py compiles (it can only be *run* on the instance)"
else fail "f2_host_run.py does not compile"; fi

sect "A3. BANKED SIM EVIDENCE (Milestone C) — is it green, and does it still describe HEAD?"
if [ -f "$SIM_EVIDENCE" ]; then
  EV="$("$PY" - "$SIM_EVIDENCE" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
caps = d.get("captures") or []
bad = [c.get("name") for c in caps
       if not str(c.get("acc_source", "")).startswith("TILE")
       or not str(c.get("s_c_source", "")).startswith("TILE")]
print("|".join([
    str(d.get("token_identity")), str(d.get("milestone_c")), str(d.get("mode")),
    str(len(caps)), str(d.get("git")), str(d.get("text_on")),
    str(d.get("tokens_on")), ",".join(x or "?" for x in bad) or "-"]))
EOF
)"
  IFS='|' read -r EV_ID EV_C EV_MODE EV_N EV_GIT EV_TXT EV_TOK EV_BAD <<EOF
$EV
EOF
  expect "banked token_identity" "$EV_ID" "True"
  expect "banked milestone_c"    "$EV_C"  "True"
  info   "banked mode=$EV_MODE captures=$EV_N token=$EV_TOK text='$EV_TXT' git=$EV_GIT"
  if [ "$EV_BAD" = "-" ]; then pass "every banked capture has acc_source AND s_c_source = TILE (no golden fallback)"
  else fail "banked captures with a GOLDEN source: $EV_BAD — that run did not consume tile values end-to-end"; fi
  if [ -n "${HEADSHA:-}" ] && [ -n "$EV_GIT" ] && [ "$EV_GIT" != "None" ]; then
    if git -C "$REPO" merge-base --is-ancestor "$EV_GIT" HEAD 2>/dev/null; then
      CH="$(git -C "$REPO" diff --name-only "$EV_GIT" HEAD -- scripts/fpga/f2 golden run_tinynpu.py verif/top/l3 2>/dev/null | tr '\n' ' ')"
      if [ -z "$CH" ]; then pass "banked evidence commit == today's load-bearing code"
      else warn "evidence recorded git=$EV_GIT but these load-bearing files changed by HEAD: $CH  -> re-verify with --deep before spending (expected when the run pre-dated its own commit)"; fi
    else warn "evidence git=$EV_GIT is not an ancestor of HEAD — provenance unclear, run --deep"; fi
  fi
  N_RE="$(ls "$REPO"/build/prompt_offload/*.compute.regops.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  N_CA="$(ls "$REPO"/build/prompt_offload/*.cap.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  info "banked artifacts: $N_RE regops + $N_CA cap files under build/prompt_offload"
  [ "$N_RE" -ge 1 ] || fail "no compiled regops under build/prompt_offload"
else
  fail "no banked sim evidence at $SIM_EVIDENCE — Milestone C is unproven; do not buy silicon time"
fi

sect "A4. THE 7B MODEL (stays on this Mac in Path A)"
if [ -f "$WEIGHTS/meta.json" ]; then
  MD="$("$PY" - "$WEIGHTS/meta.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print("%s|%s|%s|%s" % (m.get("model"), m.get("n_layers"), m.get("H"), m.get("head_dim")))
EOF
)"
  IFS='|' read -r MD_MODEL MD_L MD_H MD_HD <<EOF
$MD
EOF
  info "weights: $MD_MODEL  L=$MD_L H=$MD_H head_dim=$MD_HD  ($WEIGHTS)"
  expect "head_dim (must match APEX_CL_D=128 in the AFI)" "$MD_HD" "128"
  NPY="$(ls "$WEIGHTS"/*.npy 2>/dev/null | wc -l | tr -d ' ')"
  [ "$NPY" -gt 100 ] && pass "$NPY .npy weight shards present" || fail "only $NPY .npy shards under $WEIGHTS"
else
  fail "weight cache missing: $WEIGHTS/meta.json  (--prepare is macOS/mlx-only and takes ~an hour — do NOT discover this with an instance running)"
fi
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import numpy, transformers" >/dev/null 2>&1; then
  pass "tokenizer venv OK: $VENV_PY (numpy + transformers)"
else
  warn "no usable tokenizer venv at $VENV_PY — a run must then use --ids 785 6722 315 9625 374 and the id->text mapping happens later, on this Mac"
fi
if [ -x "$REPO/verif/f2sim/obj_d128_ddr0/f2sim" ]; then pass "sim executor present (silicon twin, DDR=0) — --deep and the sim/hw A-B are possible"
else warn "no verif/f2sim/obj_d128_ddr0/f2sim — --deep cannot re-run the sim demo, and you lose the sim-vs-hw comparison"; fi

# ══════════════════ B. which RUN PATH is armed? ══════════════════════════════
sect "B. RUN PATH ARMING (how the Mac's golden reaches the instance's PCI device)"
PATH_A=0; PATH_B=0
RHE=""
for c in "$REPO/scripts/fpga/f2/remote_hw_exec.py" "$REPO/scripts/fpga/f2/remote_hw_exec.sh"; do
  [ -f "$c" ] && RHE="$c" && break
done
if [ -z "$RHE" ]; then
  warn "PATH A NOT ARMED: no scripts/fpga/f2/remote_hw_exec.{py,sh} — 'hw' would spawn f2_host_run.py on THIS Mac, which has no PCI device (runbook §5.1)"
else
  info "remote executor: $RHE"
  case "$RHE" in
    *.py)
      RO="$("$PY" "$RHE" --selftest 2>&1)"; RRC=$?
      if [ $RRC -eq 0 ]; then pass "remote_hw_exec --selftest — $(printf '%s' "$RO" | tail -1 | cut -c1-80)"
      else fail "remote_hw_exec --selftest exited $RRC — $(printf '%s' "$RO" | tail -3 | tr '\n' ' ' | cut -c1-160)"; fi
      # The zero-edit wiring the runbook uses: attach() must be a no-op with no
      # host, and must actually replace bridge.run_job when a host is set.
      W="$(cd "$REPO" && APEX_F2_HOST=ubuntu@203.0.113.1 APEX_F2_KEY="$PEM" "$PY" - <<'EOF' 2>&1
import sys
sys.path.insert(0, "scripts/fpga/f2")
import prompt_offload as po, remote_hw_exec as rhe
a = rhe.attach(po.bridge)
print("%s|%s" % (a, po.bridge.run_job.__module__))
EOF
)"
      if printf '%s' "$W" | grep -q '^True|remote_hw_exec'; then
        pass "PATH A ARMED: attach() re-points bridge.run_job at the remote executor (verified offline, no host contacted)"
        PATH_A=1
      else
        fail "PATH A wiring check failed — attach() did not re-point bridge.run_job: $(printf '%s' "$W" | tr '\n' ' ' | cut -c1-200)"
      fi
      grep -qE 'remote_hw_exec' "$REPO/scripts/fpga/f2/prompt_offload.py" 2>/dev/null \
        && info "prompt_offload.py already imports remote_hw_exec — the runbook's python -c wrapper is then optional" \
        || info "prompt_offload.py does NOT import remote_hw_exec (that 2-line edit belongs to its owner) — use the runbook §5.2 python -c wrapper instead; no file needs editing"
      grep -q 'check-clock' "$RHE" 2>/dev/null \
        && pass "remote executor exposes --check-clock (runbook §3.4 uses it as the second, independent clock gate)" \
        || warn "remote executor has no --check-clock — verify the tile clock by hand on the instance (runbook §3.3)"
      ;;
    *.sh)
      if bash -n "$RHE" 2>/dev/null; then PATH_A=1; pass "PATH A: $RHE syntax OK (no --selftest available; verify its contract by hand)"
      else fail "$RHE has a syntax error"; fi
      ;;
  esac
fi
if [ -f "$WEIGHTS/meta.json" ] && [ -f "$REPO/run_tinynpu.py" ] \
   && [ -f "$REPO/verif/top/l3/gen_l3_vectors.py" ]; then
  PATH_B=1; pass "PATH B ARMED: whole demo can run ON the instance (weights + golden + l3 generator all shippable; GoldenModel is mmap-numpy, no mlx needed)"
else
  warn "PATH B NOT ARMED: weights or golden/l3 sources missing"
fi
if [ "$PATH_A" = "0" ] && [ "$PATH_B" = "0" ]; then
  fail "NEITHER run path is armed — there is no way to reach the tile from a real prompt. Do not launch."
else
  info "session will run PATH $( [ "$PATH_A" = 1 ] && echo A || echo B ) (runbook §5)"
fi

# ══════════════════ C. AWS, read-only ════════════════════════════════════════
if [ "$DO_AWS" = "0" ]; then
  sect "C. AWS — SKIPPED (--no-aws)"; warn "AWS preconditions unverified"
else
sect "C. AWS PRECONDITIONS (all read-only)"
if ! have aws; then
  fail "aws cli not installed"
else
  info "$(aws --version 2>&1 | head -1)"
  ACCT="$(aws "${AWSTO[@]}" sts get-caller-identity --query Account --output text 2>&1)"
  ARN="$(aws "${AWSTO[@]}" sts get-caller-identity --query Arn --output text 2>&1)"
  if [ "$ACCT" = "099597653601" ]; then pass "identity: $ARN (account $ACCT)"
  else fail "unexpected AWS identity: $ACCT / $ARN (expected account 099597653601)"; fi

  # ── C1. THE AFI. This is the whole zero-rebuild premise. ──────────────────
  sect "C1. AFI LIVENESS — the exact command the runbook publishes"
  echo "  \$ aws ec2 describe-fpga-images --region $REGION --fpga-image-ids $AFI"
  AFIJ="$(aws "${AWSTO[@]}" ec2 describe-fpga-images --region "$REGION" \
          --fpga-image-ids "$AFI" \
          --query 'FpgaImages[0].[State.Code,FpgaImageGlobalId,Name,ShellVersion,CreateTime]' \
          --output text 2>&1)"
  if printf '%s' "$AFIJ" | grep -qiE 'error|not found|invalid'; then
    fail "describe-fpga-images failed: $(printf '%s' "$AFIJ" | tr '\n' ' ' | cut -c1-200)"
  else
    set -- $AFIJ
    A_STATE="${1:-}"; A_AGFI="${2:-}"; A_NAME="${3:-}"; A_SHELL="${4:-}"; A_CREATED="${5:-}"
    expect "AFI state"        "$A_STATE" "available"
    expect "AGFI id"          "$A_AGFI"  "$AGFI"
    expect "AFI shell"        "$A_SHELL" "$SHELL_VER"
    info   "AFI name=$A_NAME created=$A_CREATED"
  fi

  # ── C2. the things a launch needs ─────────────────────────────────────────
  sect "C2. LAUNCH INPUTS"
  KP="$(aws "${AWSTO[@]}" ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" \
        --query 'KeyPairs[0].KeyName' --output text 2>&1)"
  [ "$KP" = "$KEY_NAME" ] && pass "key pair '$KEY_NAME' exists in $REGION" \
                          || fail "key pair '$KEY_NAME' not found: $(printf '%s' "$KP" | cut -c1-120)"
  if [ -f "$PEM" ]; then
    MODE="$(stat -f '%Lp' "$PEM" 2>/dev/null || stat -c '%a' "$PEM" 2>/dev/null)"
    case "$MODE" in
      400|600) pass "private key $PEM present, mode $MODE" ;;
      *) fail "private key $PEM mode $MODE — ssh will refuse it (chmod 400)" ;;
    esac
  else fail "private key missing: $PEM"; fi

  AMI_STATE="$(aws "${AWSTO[@]}" ec2 describe-images --region "$REGION" --image-ids "$AMI" \
               --query 'Images[0].State' --output text 2>&1)"
  AMI_NAME="$(aws "${AWSTO[@]}" ec2 describe-images --region "$REGION" --image-ids "$AMI" \
              --query 'Images[0].[Name,RootDeviceName]' --output text 2>&1 | tr '\t' ' ')"
  if printf '%s' "$AMI_STATE" | grep -qiE 'error|invalid|not exist'; then
    fail "AMI $AMI not resolvable: $(printf '%s' "$AMI_STATE" | cut -c1-140)"
  else
    expect "AMI state" "$AMI_STATE" "available"; info "AMI $AMI = $AMI_NAME"
  fi

  # AZ trap: f2.6xlarge is NOT offered in every AZ of us-west-2.
  OFFERS="$(aws "${AWSTO[@]}" ec2 describe-instance-type-offerings --region "$REGION" \
            --location-type availability-zone \
            --filters "Name=instance-type,Values=$ITYPE" \
            --query 'InstanceTypeOfferings[].Location' --output text 2>&1 | tr '\t' ' ')"
  info "$ITYPE offered in: $OFFERS"
  SNAZ="$(aws "${AWSTO[@]}" ec2 describe-subnets --region "$REGION" --subnet-ids "$SUBNET" \
          --query 'Subnets[0].AvailabilityZone' --output text 2>&1)"
  if printf '%s' " $OFFERS " | grep -q " $SNAZ "; then
    pass "pinned --subnet-id $SUBNET is in $SNAZ, which offers $ITYPE"
  else
    fail "pinned subnet $SUBNET is in $SNAZ, which does NOT offer $ITYPE — run-instances without a pinned subnet can also land in a bad AZ (offered: $OFFERS)"
  fi

  SGID="$(aws "${AWSTO[@]}" ec2 describe-security-groups --region "$REGION" \
          --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' \
          --output text 2>&1)"
  if [ "${SGID:0:3}" = "sg-" ]; then
    pass "security group $SG_NAME = $SGID"
    CIDRS="$(aws "${AWSTO[@]}" ec2 describe-security-groups --region "$REGION" --group-ids "$SGID" \
             --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' \
             --output text 2>&1 | tr '\t' ' ')"
    info "port 22 allowed from: $CIDRS"
    MYIP="$(curl -s --max-time 8 https://checkip.amazonaws.com 2>/dev/null | tr -d '\r\n')"
    if [ -z "$MYIP" ]; then warn "could not determine this Mac's public IP — verify by hand that one of the CIDRs above covers it, or ssh will hang while the meter runs"
    else
      if "$PY" - "$MYIP" $CIDRS <<'EOF' >/dev/null 2>&1
import ipaddress, sys
ip = ipaddress.ip_address(sys.argv[1])
sys.exit(0 if any(ip in ipaddress.ip_network(c) for c in sys.argv[2:]) else 1)
EOF
      then pass "this Mac ($MYIP) is covered by an existing port-22 rule"
      else fail "this Mac ($MYIP) is NOT in any port-22 rule of $SGID — ssh will time out. Remediation (an account-settings change: get owner approval, do it BEFORE launching):
           aws ec2 authorize-security-group-ingress --region $REGION --group-id $SGID --protocol tcp --port 22 --cidr $MYIP/32"
      fi
    fi
  else fail "security group '$SG_NAME' not found: $(printf '%s' "$SGID" | cut -c1-120)"; fi

  # ── C3. quota + cost + spend leaks ───────────────────────────────────────
  sect "C3. QUOTA, COST, SPEND LEAKS"
  QV="$(aws "${AWSTO[@]}" service-quotas get-service-quota --region "$REGION" \
        --service-code ec2 --quota-code L-74FC7D96 --query 'Quota.Value' --output text 2>&1)"
  if printf '%s' "$QV" | grep -qE '^[0-9.]+$'; then
    if "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1])>=24 else 1)" "$QV"; then
      pass "quota L-74FC7D96 (Running On-Demand F instances) = $QV vCPU >= 24 needed by $ITYPE"
    else fail "quota L-74FC7D96 = $QV vCPU — $ITYPE needs 24; the launch WILL be refused"; fi
  else warn "could not read quota L-74FC7D96: $(printf '%s' "$QV" | cut -c1-120)"; fi

  PRICE="$(aws "${AWSTO[@]}" pricing get-products --region us-east-1 --service-code AmazonEC2 \
      --filters "Type=TERM_MATCH,Field=instanceType,Value=$ITYPE" \
                "Type=TERM_MATCH,Field=regionCode,Value=$REGION" \
                'Type=TERM_MATCH,Field=operatingSystem,Value=Linux' \
                'Type=TERM_MATCH,Field=tenancy,Value=Shared' \
                'Type=TERM_MATCH,Field=preInstalledSw,Value=NA' \
                'Type=TERM_MATCH,Field=capacitystatus,Value=Used' \
      --max-results 1 --output json 2>/dev/null | "$PY" -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); sys.exit()
for p in d.get("PriceList",[]):
    p=json.loads(p) if isinstance(p,str) else p
    for t in p["terms"].get("OnDemand",{}).values():
        for dim in t["priceDimensions"].values():
            print("%.4f"%float(dim["pricePerUnit"]["USD"])); sys.exit()
print("")')"
  if [ -n "$PRICE" ]; then
    if "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1])<=float(sys.argv[2]) else 1)" "$PRICE" "$MAX_PRICE"; then
      pass "$ITYPE on-demand = \$$PRICE/hr in $REGION (AWS Pricing API, primary source) <= ceiling \$$MAX_PRICE"
    else fail "$ITYPE on-demand = \$$PRICE/hr > ceiling \$$MAX_PRICE — re-budget before launching"; fi
    info "budget: a 1 h session ~= \$$PRICE ; the runbook's 2 h hard cap ~= \$$("$PY" -c "print('%.2f'%(2*float('$PRICE')))")"
  else warn "could not read the on-demand price from the Pricing API — use the in-repo figure (\$1.98/hr, BRINGUP.md §1/§4) and re-check at spend time"; fi

  RUN="$(aws "${AWSTO[@]}" ec2 describe-instances --region "$REGION" \
        --filters Name=instance-state-name,Values=running,pending \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' --output text 2>&1)"
  if [ -z "$RUN" ]; then pass "no running/pending instances in $REGION (no pre-existing spend)"
  else
    warn "instances already running/pending in $REGION — confirm none is yours to pay for, and NEVER touch the verifagent/Catapult boxes:"
    printf '%s\n' "$RUN" | sed 's/^/         /'
    if printf '%s' "$RUN" | grep -q 'f2\.'; then
      fail "an f2.* instance is ALREADY running — a previous session leaked. Reconcile spend before launching another."
    fi
  fi
fi
fi

# ══════════════════ D. deep: re-run the sim demo (still $0) ══════════════════
if [ "$DEEP" = "1" ]; then
  sect "D. DEEP RE-VERIFY — the real 7B prompt through the sim executor (~6 min, \$0)"
  DW="$REPO/build/prompt_offload_preflight"
  RUNPY="$VENV_PY"; [ -x "$RUNPY" ] || RUNPY="$PY"
  LOG="$DW/preflight_sim.log"; mkdir -p "$DW"
  echo "  \$ $RUNPY scripts/fpga/f2/prompt_offload.py --prompt 'The capital of France is' --max-tokens 1 --executor sim --work-dir $DW"
  ( cd "$REPO" && "$RUNPY" scripts/fpga/f2/prompt_offload.py \
      --prompt "The capital of France is" --max-tokens 1 \
      --offload-layer 0 --offload-head 0 --executor sim \
      --work-dir "$DW" >"$LOG" 2>&1 ); RC=$?
  if [ $RC -eq 0 ] && grep -q 'MILESTONE C   : PASS' "$LOG" \
     && grep -q 'TOKEN IDENTITY: PASS' "$LOG"; then
    pass "sim demo re-verified at ${HEADSHA:0:7}: $(grep -m1 'MILESTONE C' "$LOG" | sed 's/^ *//')"
    grep -m1 'acc source' "$LOG" | sed 's/^ */         /'
    info "log: $LOG (work dir kept separate from build/prompt_offload so the banked evidence is not clobbered)"
  else
    fail "sim demo re-verify FAILED (rc=$RC) — fix this for free before buying silicon time. Tail:"
    tail -12 "$LOG" | sed 's/^/         /'
  fi
else
  info "(--deep not requested: the sim demo was NOT re-run at this commit. Runbook §0.4 requires it before launch.)"
fi

# ─────────────────────────────── verdict ─────────────────────────────────────
printf '\n==================================================================\n'
printf 'PREFLIGHT: %d pass, %d warn, %d FAIL\n' "$PASSES" "$WARNS" "$FAILS"
if [ "$FAILS" -eq 0 ]; then
  printf 'VERDICT: CLEAR TO LAUNCH (runbook §1). Read every [WARN] above first.\n'
  printf 'Paste this whole output into the session log BEFORE run-instances.\n'
else
  printf 'VERDICT: DO NOT LAUNCH — %d precondition(s) failed. Every one of them\n' "$FAILS"
  printf 'is cheaper to fix now than with a $1.98/hr meter running.\n'
fi
printf '==================================================================\n'
exit $(( FAILS > 0 ? 1 : 0 ))
