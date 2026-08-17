#!/bin/bash
# run_walked_demo.sh — one-command walked-attention demo flight.
#
#   bash scripts/fpga/f2/run_walked_demo.sh <AGFI>
#
# Boots an f2.6xlarge, loads <AGFI> + the A2 clock recipe, ships the
# weights and walk programs, runs the walked chains (E-6 attention walk,
# E-7 composed walk, E-7ng control, host-attention control), prints a
# verdict table, and TERMINATES the card (verified) no matter what.
# Run from the repo root. Requires: aws cli configured (us-west-2),
# ~/.ssh/apex-f2.pem. Cost: ~$1-2 (card lives ~30-40 min).
set -euo pipefail

AGFI="${1:?usage: run_walked_demo.sh <agfi-id> (see scripts/fpga/f2/clock_key.py for registered images)}"
REGION=us-west-2
AMI=ami-07fd0100b41f6c579
SG=sg-0766e253ceeaa3b74
KEY=~/.ssh/apex-f2.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

echo "== [1/6] launching f2.6xlarge"
IID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type f2.6xlarge \
  --key-name apex-f2 --security-group-ids $SG \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=apex-walked-demo}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "   instance: $IID"
cleanup () {
  echo "== terminating $IID"
  aws ec2 terminate-instances --region $REGION --instance-ids "$IID" --output text >/dev/null
  aws ec2 wait instance-terminated --region $REGION --instance-ids "$IID" \
    && echo "   terminated (verified)"
}
trap cleanup EXIT

aws ec2 wait instance-running --region $REGION --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "   ip: $IP"
for i in $(seq 1 30); do $SSH "ubuntu@${IP}" true 2>/dev/null && break; sleep 20; done

echo "== [2/6] AFI + clock recipe"
$SSH "ubuntu@${IP}" "test -d ~/aws-fpga || git clone -q https://github.com/aws/aws-fpga.git ~/aws-fpga; \
  cd ~/aws-fpga && source sdk_setup.sh >/tmp/sdk.log 2>&1; \
  sudo fpga-load-local-image -S 0 -I $AGFI >/dev/null && \
  sudo fpga-load-clkgen-recipe -S 0 -a 2 >/dev/null && \
  sudo fpga-describe-clkgen -S 0 | grep -q '15.62' && echo '   A2 @15.62 verified' && \
  sudo chmod 666 /sys/bus/pci/devices/0000:34:00.0/resource* /sys/bus/pci/devices/0000:34:00.1/resource* 2>/dev/null || true"

echo "== [3/6] shipping repo + weights + programs"
TARBALL=$(mktemp -t apexrepo.XXXX).tar
git archive HEAD -o "$TARBALL"
$SSH "ubuntu@${IP}" "mkdir -p ~/apex/build/ddr_weights_05b ~/apex/build/e6c_bisect ~/apex/build/e7_flight ~/apex/build/e7ng"
scp -q -i $KEY -o StrictHostKeyChecking=no "$TARBALL" "ubuntu@${IP}:~/apex/repo.tar"
scp -q -i $KEY -o StrictHostKeyChecking=no build/ddr_weights_05b/* "ubuntu@${IP}:~/apex/build/ddr_weights_05b/"
scp -q -i $KEY -o StrictHostKeyChecking=no build/e6c_bisect/walk_e6.regops.jsonl "ubuntu@${IP}:~/apex/build/e6c_bisect/"
scp -q -i $KEY -o StrictHostKeyChecking=no build/e7_flight/walk_e7.regops.jsonl "ubuntu@${IP}:~/apex/build/e7_flight/"
scp -q -i $KEY -o StrictHostKeyChecking=no build/e7ng/walk_e7ng.regops.jsonl build/e7ng/hostattn_fuelarm.regops.jsonl "ubuntu@${IP}:~/apex/build/e7ng/"
rm -f "$TARBALL"
$SSH "ubuntu@${IP}" "cd ~/apex && tar -xf repo.tar"

echo "== [4/6] DDR weight load (full verify)"
$SSH "ubuntu@${IP}" "source ~/aws-fpga/sdk_setup.sh >/dev/null 2>&1; \
  python3 /home/ubuntu/apex/scripts/fpga/f2/f2_ddr_load.py \
    --image /home/ubuntu/apex/build/ddr_weights_05b --load --verify --full-verify 2>&1 | tail -2"

echo "== [5/6] FLYING the walked chains"
$SSH "ubuntu@${IP}" "sudo bash -c 'source /home/ubuntu/aws-fpga/sdk_setup.sh >/dev/null 2>&1; \
  cd /home/ubuntu/apex && python3 scripts/fpga/f2/f2_host_run.py \
    build/e6c_bisect/walk_e6.regops.jsonl \
    build/e7_flight/walk_e7.regops.jsonl \
    build/e7ng/walk_e7ng.regops.jsonl \
    build/e7ng/hostattn_fuelarm.regops.jsonl 2>&1' | grep -E '^\[|F2HOST RESULT'"

echo "== [6/6] verdict above: every file must show 'fails=0'."
echo "   walk_e6 = walked ATTENTION. walk_e7 = walked composed layer front."
echo "   (card terminates automatically now)"
