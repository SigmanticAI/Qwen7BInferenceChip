#!/bin/bash
# apex_verify_box.sh — cheap AWS verify box for Verilator/yosys unit suites
# (LEVEL_C_PARALLEL §bottlenecks unlock; standing spend rule APPROVED by
# owner 2026-07-21 — see docs/design/LEVEL_C_INTEGRATION.md §6:
# c6a-class <= ~$0.70/hr, ~$40 I-A cap, tagged apex-*, stopped when idle,
# NEVER touch the verifagent/Catapult/g5 instances on the account).
#
# Usage (from any worktree):
#   scripts/aws/apex_verify_box.sh launch            # start (or reuse) the box
#   scripts/aws/apex_verify_box.sh push  <branch>    # git-bundle branch -> box
#                                                    # (bundle, NOT git push —
#                                                    #  lane branches stay off
#                                                    #  the public remote)
#   scripts/aws/apex_verify_box.sh run   <branch> "<make cmd>"
#                                                    # e.g. "make -C verif/seq_walker all"
#   scripts/aws/apex_verify_box.sh status | stop | terminate
#
# Multi-lane (S12 addition): the box is keyed by tag. Concurrent lanes MUST
# NOT share one box — `run` does `git checkout -f` in the box's single ~/apex
# clone and would clobber another lane's in-flight flow (S14's OpenLane/eqy
# run was live on the default box when S12 needed one). Per-lane boxes are
# within the owner-approved spend rule ("any lane may spin a c6a-class box").
# Override the tag per lane:
#   APEX_BOX_TAG=apex-s12-verify scripts/aws/apex_verify_box.sh launch
#
# Notes:
#  * Box verilator is the distro build (5.02x-class), NOT the pinned local
#    5.044: box verdicts are GREEN/RED unit gates; any byte-identical
#    anchor A/B claim must run on the local pinned toolchain.
#  * The /opt/homebrew/bin/verilator symlink this bootstrap used to create is
#    GONE, and so is the reason for it: the repo no longer hardcodes tool
#    paths. Makefiles include verif/tools.mk and mutation_check*.py resolve
#    VERILATOR from the environment, so PATH is enough.
#  * CONSEQUENCE, read this before assuming a box is usable: verif/tools.mk
#    now ENFORCES the pinned Verilator (ARCHITECTURE.md), so the apt package
#    above (5.02x-class) is REFUSED with a loud version-mismatch error instead
#    of being silently used. That is the point — the old symlink pointed the
#    de-facto "pin" at whatever apt shipped, so boxes were quietly running a
#    version the tile build is known to miscompile. To make a box usable,
#    build the pinned Verilator from source and put it on PATH, or pass
#    VERILATOR=/path/to/it. Deliberately accepting another build is an
#    explicit VERILATOR_REQ=<ver> override, never a silent default.
#  * Logs come back to build_box_logs/<ts>_<target>.log — paste from there.
set -euo pipefail

REGION=us-west-2
ITYPE="${APEX_BOX_TYPE:-c6a.xlarge}"          # ~$0.15/hr; 4 vCPU is plenty
KEY=apex-f2
PEM="$HOME/.ssh/apex-f2.pem"
SG_NAME=apex-f2-ssh
TAG="${APEX_BOX_TAG:-apex-verify-box}"
AMI_SSM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id

aws_() { aws --region "$REGION" "$@"; }

box_id() {
  aws_ ec2 describe-instances \
    --filters "Name=tag:Name,Values=$TAG" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[0].InstanceId' --output text | head -1
}
box_ip() {
  aws_ ec2 describe-instances --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
SSH() { ssh -i "$PEM" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$1" "${@:2}"; }

cmd="${1:-status}"

case "$cmd" in
  launch)
    ID=$(box_id)
    if [ -n "$ID" ] && [ "$ID" != "None" ]; then
      ST=$(aws_ ec2 describe-instances --instance-ids "$ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
      if [ "$ST" = "stopped" ]; then aws_ ec2 start-instances --instance-ids "$ID" >/dev/null; fi
      echo "reusing $ID ($ST -> running)"
    else
      AMI=$(aws_ ssm get-parameter --name "$AMI_SSM" --query Parameter.Value --output text)
      SG=$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
             --query 'SecurityGroups[0].GroupId' --output text)
      ID=$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
        --key-name "$KEY" --security-group-ids "$SG" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=apex,Value=verify}]" \
        --query 'Instances[0].InstanceId' --output text)
      echo "launched $ID ($ITYPE)"
    fi
    aws_ ec2 wait instance-running --instance-ids "$ID"
    IP=$(box_ip "$ID"); echo "ip: $IP"
    until SSH "$IP" true 2>/dev/null; do sleep 5; done
    SSH "$IP" 'set -e
      command -v verilator >/dev/null || {
        sudo apt-get -qq update
        sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
          verilator make git python3-numpy python3-venv yosys > /dev/null
      }
      verilator --version'
    echo "READY. Reminder: stop when idle (this script: stop)."
    ;;
  push)
    BR="${2:?branch}"; ID=$(box_id); IP=$(box_ip "$ID")
    git bundle create /tmp/apex_push.bundle "$BR" --branches="$BR" 2>/dev/null \
      || git bundle create /tmp/apex_push.bundle "$BR"
    scp -i "$PEM" -o StrictHostKeyChecking=accept-new /tmp/apex_push.bundle "ubuntu@$IP:~/"
    SSH "$IP" "set -e
      if [ -d apex ]; then cd apex && git checkout -q --detach \
        && git fetch ~/apex_push.bundle '$BR:refs/heads/$BR' -f
      else git clone -b '$BR' ~/apex_push.bundle apex; fi
      cd ~/apex && git checkout -f '$BR' && git log --oneline -1"
    ;;
  run)
    BR="${2:?branch}"; MAKECMD="${3:?make cmd}"; ID=$(box_id); IP=$(box_ip "$ID")
    TS=$(date +%Y%m%d_%H%M%S)
    SAFE=$(echo "$MAKECMD" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-60)
    mkdir -p build_box_logs
    SSH "$IP" "cd ~/apex && git checkout -f '$BR' >/dev/null 2>&1 && \
      $MAKECMD" 2>&1 | tee "build_box_logs/${TS}_${SAFE}.log"
    RC=${PIPESTATUS[0]}
    echo "box run rc=$RC — log: build_box_logs/${TS}_${SAFE}.log"
    exit "$RC"
    ;;
  status)
    ID=$(box_id)
    [ -z "$ID" ] || [ "$ID" = "None" ] && { echo "no verify box"; exit 0; }
    aws_ ec2 describe-instances --instance-ids "$ID" \
      --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]' \
      --output text
    ;;
  stop)
    ID=$(box_id); aws_ ec2 stop-instances --instance-ids "$ID" >/dev/null; echo "stopping $ID"
    ;;
  terminate)
    ID=$(box_id); aws_ ec2 terminate-instances --instance-ids "$ID" >/dev/null; echo "terminating $ID"
    ;;
  *) echo "usage: $0 launch|push <branch>|run <branch> '<make cmd>'|status|stop|terminate"; exit 1 ;;
esac
