# MILESTONE D — the live-FPGA runbook

> **STATUS: PREPARED, NOT EXECUTED.** No instance was launched and no money was
> spent writing this. Everything below marked ✅ was *actually run* on this Mac
> against the real AWS account (read-only calls only) and its output is pasted
> in §10. Everything marked ⏸ needs the instance and therefore owner-approved
> spend. Same claim discipline as the rest of the repo: **no PASS without a
> pasted log.**

**D in one sentence.** Take the thing that is already green in simulation at
`6aac4fa` — a real Qwen2.5-7B prompt where one attention operation is computed
by the tile and the emitted token is proven identical to the pure-golden token
— and re-run it with the **real FPGA** in the loop instead of Verilator.

**What D does NOT need** (all settled by `docs/design/PROMPT_DEMO_AUDIT.md` §1
and re-verified in §10): no new AFI, no RTL change, no Vivado, no DCP build, no
DDR/fuel path, no walker, no GQA-4. The bitstream that already replayed 18 real
7B attention jobs bit-exactly on silicon (`docs/results/f2_stage2_hw/`) is
still `available` and is the *only* hardware artifact this milestone consumes.

---

## The five ways this session could produce a fake PASS

This runbook is structured around defeating these. Every one of them has been
seen or narrowly avoided in this project before.

| # | fake-pass mode | why it is possible | the defence in this runbook |
|---|---|---|---|
| **F1** | **wrong tile clock, green gates** | an AFI load **resets the clkgen MMCMs to the default recipe** (`a1 = 125.00 MHz`), the tile only closes timing at **recipe A2 = 15.625 MHz** (64 ns), and `f2_host_run.py`'s MMCM preflight accepts any output containing the substring `lock` — which the tool's own header line `Clock Group A Frequency (Mhz)` satisfies inside the word **C-lock**. It passes **vacuously, always, at any frequency.** The line `[clkgen] MMCM locked:` in the previous session's log is `f2_host_run.py` printing its *own* text, not a verdict. | §3.3 asserts the **number** `a1 ≈ 15.62` and aborts non-zero otherwise; §3.4 re-asserts it through a second, independent gate (`remote_hw_exec.py --check-clock`, rc 78 on refusal); §7.1 re-reads it *after* the run to prove it never changed mid-session |
| **F2** | **a `--dry-run` log filed as D** | `--dry-run` exercises the whole plumbing with golden supplying the value; the banner is otherwise identical | §6.2 requires `executor : hw` and `acc source : TILE via … (hw)` on **every** capture line |
| **F3** | **the scale silently came from golden** | `prompt_offload._tile_value` falls back to `s_c, sc_src = core.s_c, "GOLDEN (no ss tap captured)"` when no `ss` tap is captured — the run still says PASS | §6.2 requires `s_c source : TILE (…)` on every capture; a single `GOLDEN` voids the claim |
| **F4** | **`sim` relabelled as silicon** | `prompt_offload` discards the executor's stdout on success, so the banner alone cannot prove a PCI device was involved | §6.3 re-executes the *same* 5 regops files on the instance with `f2_host_run.py` directly and diffs the cap JSONL byte-for-byte against what the demo consumed; §6.1 requires the instance-side transcript (AGFI load, clkgen table, BAR0 probe, `F2HOST CAPTURES:` lines) in the same commit |
| **F5** | **the wrong bitstream** | `f2_smoke_session.sh` hardcodes `AGFI=agfi-0f7c93ffa798ecc3f` — the **first-light D=64** image, not ours. It would load, probe green on ID/scratch, and then compute nonsense for D=128 jobs | §3.1 forbids running that script as-is; §4 makes `KVQ INFO_DIM == 0x00000080` (D=128) a hard go/no-go — the D=64 image reads `0x40` there |

A sixth, cheaper failure mode — **spend leak** — is handled by §7 (terminate,
*verify* terminated, sweep for orphans) and by the dead-man switch in §1.4.

---

## §0 · PREFLIGHT — costs nothing, must be 100 % green before §1 ✅

### 0.1 Run the script

```sh
cd ~/Desktop/apex-promptdemo
bash scripts/fpga/f2/d_preflight.sh            # ~30 s, read-only, no instance
bash scripts/fpga/f2/d_preflight.sh --deep      # + re-runs the real-7B sim demo
```

It exits non-zero on any failure and prints `VERDICT: DO NOT LAUNCH`. It makes
**only** read-only AWS calls (`sts`, `describe-*`, `get-service-quota`,
`pricing`) — it cannot launch, modify or delete anything. Paste its whole output
into the session log **before** `run-instances`. Verified output: §10.1.

It checks, in order: git tip + working-tree state · the ten load-bearing source
files · the three host-side selftests (`compute_job`, `tile_exec_bridge`,
`prompt_offload`) · that `f2_host_run.py` at least compiles · the banked
Milestone-C result (`token_identity`, `milestone_c`, and that **every** capture's
`acc_source` **and** `s_c_source` start with `TILE`) · whether that evidence is
stale w.r.t. HEAD · the 7B weight cache (`head_dim == 128`, 339 shards) ·
the tokenizer venv · which run path is armed (§5) · AWS identity ·
**AFI liveness** · key pair + local `.pem` mode · AMI · **the AZ trap** ·
security-group coverage of this Mac's current public IP · the F-instance quota ·
the on-demand price against a ceiling · and pre-existing running instances.

### 0.2 The AFI liveness command, verbatim ✅

AFI liveness is otherwise only a **doc claim**. Run this yourself before
promising anything to anyone:

```sh
aws ec2 describe-fpga-images --region us-west-2 --fpga-image-ids afi-036d83cafa00d26ea
```

`State.Code` **must** be `available` and `FpgaImageGlobalId` **must** be
`agfi-0ae06ea568e5667ba`. Real output, 2026-07-30: §10.2.

### 0.3 Git tip

Record `git rev-parse HEAD` and the branch in the session log. The runbook was
written against **`comp/prompt-b-c` @ `6aac4fa`** ("MILESTONE B+C GREEN"). If
HEAD has moved, `--deep` (0.4) is not optional — it is the only thing that
proves the *current* code still produces the token.

### 0.4 Re-verify the sim demo at today's commit (still $0)

```sh
bash scripts/fpga/f2/d_preflight.sh --deep     # ~6 min of Mac CPU
```

This runs the **real** 7B prompt through the **sim** executor into a *separate*
work dir (`build/prompt_offload_preflight`) and requires
`TOKEN IDENTITY: PASS` + `MILESTONE C   : PASS`. Two reasons it is mandatory:
the banked `prompt_offload_result.json` records `git=0e2a18c` while HEAD is
`6aac4fa` (the run pre-dated its own commit — the preflight WARNs about exactly
this), and a green sim run is the reference the hardware run is compared to. If
the sim demo is red, **stop here**: silicon cannot fix it and the meter would
be running while you debug host software.

> ⚠ **Clobber hazard (audit N10, and it applies to us).** The demo's default
> `--work-dir` is `build/prompt_offload`, which holds the banked Milestone-C
> evidence (5 regops + 5 cap files + `prompt_offload_result.json`). Every run in
> this runbook passes an explicit, *different* `--work-dir`. Never let the
> hardware run write into the sim evidence directory — the comparison in §6.3
> depends on both surviving.

### 0.5 Gate

Do not proceed to §1 until: preflight exit status 0, every `[WARN]` read and
understood, `--deep` green, and the owner has re-confirmed spend. Budget the
session at **≤ 2 h ⇒ ≤ $4.00**.

---

## §1 · LAUNCH ⏸

### 1.1 Cost, from primary sources ✅

`f2.6xlarge` on-demand in `us-west-2` is **$1.98/hr** — the AWS Pricing API
returns `1.9800000000 USD/Hrs` (§10.3) and the in-repo figure agrees
(`scripts/fpga/f2/BRINGUP.md` §1 and §4: "f2.6xlarge … ~**$1.98/hr**"; the last
silicon session booked "f2.6xlarge ~1.3 h ≈ $2.60"). **The ~$1.65/hr figure in
the sprint brief is wrong — use $1.98/hr.** Linux billing is per-second with a
60 s minimum. 1 FPGA, 24 vCPU, 256 GiB.

### 1.2 The AZ trap ✅

`f2.6xlarge` is offered in **`us-west-2b` and `us-west-2c` only** (§10.4), but
the default VPC has subnets in **a/b/c/d**. `run-instances` *without*
`--subnet-id` may pick `us-west-2a` and fail. **Always pin the subnet.**
`subnet-0ee519f2d304c99c9` = `us-west-2b` (the AZ the successful 2026-07-22
session used) and it auto-assigns a public IP (`MapPublicIpOnLaunch=true`).
Fallback: `subnet-0dbb62b25dea9e7a3` (`us-west-2c`).

### 1.3 The command

```sh
export AWS_REGION=us-west-2
T_START=$(date -u +%s)
IID=$(aws ec2 run-instances --region us-west-2 \
  --image-id ami-07a164f1a402ab274 \
  --instance-type f2.6xlarge \
  --key-name apex-f2 \
  --security-group-ids sg-0766e253ceeaa3b74 \
  --subnet-id subnet-0ee519f2d304c99c9 \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=apex-milestone-d},{Key=Project,Value=apex},{Key=Purpose,Value=prompt-on-chip-D}]' \
  --count 1 --query 'Instances[0].InstanceId' --output text)
echo "IID=$IID T_START=$T_START"          # <-- both into the session log NOW

aws ec2 wait instance-running --region us-west-2 --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region us-west-2 --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "IP=$IP"
```

No IAM instance profile (none is needed, and the only profiles in this account
belong to Catapult — **never attach or touch those**). No block-device override:
the AMI already brings a 120 GiB gp3 root + 10 GiB `/dev/sdm`, both
`DeleteOnTermination=true` (§10.5), which is ample even for the Path-B weight
copy.

### 1.4 Dead-man switch (recommended)

`--instance-initiated-shutdown-behavior terminate` is already set above, so a
shutdown *terminates* rather than parking a stopped instance you keep paying
EBS for. Arm the timer as the first command on the box:

```sh
ssh -i ~/.ssh/apex-f2.pem -o StrictHostKeyChecking=accept-new ubuntu@$IP \
    'sudo shutdown -h +120 && uptime'      # hard $4 ceiling
```

Consequence to accept: at T+120 min the box dies with whatever is on it, so §6
pulls evidence back **as it is produced**, not at the end. Cancel with
`sudo shutdown -c` if you deliberately extend (and say so in the log).

---

## §2 · INSTANCE SETUP ⏸ — what does and does not get copied

**Path A (default, §5.2): nothing from this repo is copied by hand.** The
instance needs only the AWS FPGA kit. `remote_hw_exec.py` ships
`f2_host_run.py` itself into `~/apexrun` per job, and `f2_host_run.py` imports
nothing but the standard library plus the kit's Cython bindings — **no numpy,
no golden, no weights, no tokenizer**. The 7.6 GB model and the mlx/HF venv
**stay on this Mac**. Per-job traffic is one 80–240 KB regops file up and one
small cap JSONL back (§10.6).

```sh
ssh -i ~/.ssh/apex-f2.pem -o StrictHostKeyChecking=accept-new ubuntu@$IP
# ── on the instance ───────────────────────────────────────────────────────
git clone --depth 1 --branch f2 https://github.com/aws/aws-fpga.git ~/aws-fpga
cd ~/aws-fpga
set +u                      # the kit's scripts are not -u clean
source sdk_setup.sh 2>&1 | tail -20        # NEVER pipe `source` into a pager
# this is what builds sdk/userspace/cython_bindings — the surface f2_host_run.py needs
ls -la ~/aws-fpga/sdk/userspace/cython_bindings/*.so
sudo python3 -c "import sys; sys.path.insert(0,'/home/ubuntu/aws-fpga/sdk/userspace/cython_bindings'); import fpga_pci_wrapper; print('bindings OK')"
```

The clone **must** live at `/home/ubuntu/aws-fpga`: under `sudo`, `~` is
`/root`, and `f2_host_run.py`'s path resolution falls through to a hardcoded
`/home/ubuntu/aws-fpga`. (Alternative: pass
`--remote-env AWS_FPGA_REPO_DIR=/opt/aws-fpga`.)

**Path B (fallback, §5.3) additionally needs**, on the instance: numpy for
root, and a copy of the golden stack + weights.

```sh
# instance
sudo apt-get update -qq && sudo apt-get install -y python3-numpy
sudo python3 -c "import numpy; print('root numpy', numpy.__version__)"
# Mac, in a second terminal, in parallel with the kit build above
cd ~/Desktop/apex-promptdemo && git rev-parse HEAD > /tmp/COMMIT.txt
rsync -az -e "ssh -i ~/.ssh/apex-f2.pem" \
      run_tinynpu.py golden scripts/fpga/f2 verif/top/l3 /tmp/COMMIT.txt \
      ubuntu@$IP:~/apex_d/repo/
rsync -az --info=progress2 -e "ssh -i ~/.ssh/apex-f2.pem" \
      build/s8_weights/Qwen2.5-7B-4bit/ \
      ubuntu@$IP:~/apex_d/s8_weights/Qwen2.5-7B-4bit/     # 7.6 GB
```
Start the 7.6 GB copy immediately after ssh works so it overlaps the kit build
+ AFI load + probe (~15 min of work that costs nothing extra). If your uplink
makes the ETA exceed ~25 min, **abort Path B and fix Path A instead** — the
copy is pure burn. S3 is not a shortcut here: the instance has no credentials
and creating a role is an account change requiring separate approval.

---

## §3 · AFI LOAD + CLKGEN + THE REAL FREQUENCY VERIFICATION ⏸

### 3.1 Do not run `f2_smoke_session.sh` as-is

It hardcodes `AGFI=agfi-0f7c93ffa798ecc3f` — the **first-light D=64** image
(fake-pass mode F5). Its BAR0 probe body is good and §4 reuses it verbatim, but
the AGFI must be ours.

### 3.2 Load

```sh
sudo fpga-describe-local-image-slots -H
sudo fpga-load-local-image -S 0 -I agfi-0ae06ea568e5667ba
sudo fpga-describe-local-image -S 0 -H            # must echo OUR agfi + "loaded"
```

### 3.3 Program recipe A2, then **verify the frequency** — the load-bearing step

An AFI load resets the clkgen MMCMs to the **default** recipe (`a1 = 125 MHz`);
neither the bitstream's static MMCM properties nor the manifest's
`clock_recipe_a=A2` are honoured at load time. The host must program it before
any BAR0 traffic. And do not trust any "lock" message: recipe A2 = **15.625
MHz** comes from the SDK's own C table (`clkgen_a_recipes[2] = {mult 12.5, div
1, div0 80, …}` → 1250/80), while `fpga-load-clkgen-recipe --help` prints the
**wrong** row for A2.

```sh
sudo fpga-load-clkgen-recipe -S 0 -a 2
mkdir -p ~/apex_d && fpga-describe-clkgen -S 0 | tee ~/apex_d/clkgen_after_recipe.txt

# HARD GATE — the number, not the word "lock":
A1=$(awk '/Clock Group A/{g=1} g && /^\|/ && $0 !~ /clk_extra/ && $0 !~ /^\|-/ {gsub(/ /,""); split($0,c,"|"); print c[2]; exit}' ~/apex_d/clkgen_after_recipe.txt)
python3 -c "import sys;v=float(sys.argv[1]);sys.exit(0 if abs(v-15.625)<=0.05 else 1)" "$A1" \
  && echo "CLKGEN GATE PASS: clk_extra_a1 = $A1 MHz (recipe A2, the closed 64 ns tile clock)" \
  || { echo "CLKGEN GATE FAIL: clk_extra_a1 = $A1 MHz — ABORT, run no jobs (see §8 row C)"; exit 1; }
```

Expected table (this is exactly what the 2026-07-22 silicon session recorded in
`docs/results/f2_stage2_hw/clkgen_final.txt`):

```
Clock Group A Frequency (Mhz)
| clk_extra_a1 | clk_extra_a2 | clk_extra_a3 |
|--------------|--------------|--------------|
|      15.62   |     125.00   |      62.50   |
```

**Say it plainly in the session log:** *the MMCM-lock preflight inside
`f2_host_run.py` is vacuous — it matches the substring `lock` in the tool's own
`Clock Group` header and returns PASS at any frequency, including the 125 MHz
default that is 8× over the tile's closed clock. The frequency read-back above
is the only real assurance, and it is why this run is trustworthy.*

### 3.4 Second, independent clock gate (from the Mac) ⏸

```sh
cd ~/Desktop/apex-promptdemo
python3 scripts/fpga/f2/remote_hw_exec.py --check-clock \
        --host ubuntu@$IP --key ~/.ssh/apex-f2.pem
# must print: CLOCK GATE: PASS — ... ; rc 0. On refusal: rc 78, nothing is shipped.
```
This same gate runs automatically before **every** job in §5 (`verify_clock=True`
by default), so a mid-session clock change cannot go unnoticed. Never pass
`--no-verify-clock` / `APEX_F2_NO_CLOCK_CHECK=1` on a run you intend to publish.

---

## §4 · BAR0 IDENTITY PROBE ⏸ — the cheap go/no-go

Body lifted verbatim from the proven `f2_smoke_session.sh` probe (it compiled
and ran on both previous silicon sessions); only the expectations are tightened.
Run it **on the instance, from `~/aws-fpga`** (that relative `-I` and the
default library path are what worked before — do not "improve" the compile
line):

```sh
cd ~/aws-fpga
cat > /tmp/probe_d.c <<'EOC'
#include <stdio.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>
static uint32_t rd(pci_bar_handle_t h, uint64_t a){uint32_t v=0;int r=fpga_pci_peek(h,a,&v);
  printf("  peek 0x%04lx = 0x%08x%s\n",a,v,r?" (ERR)":"");return v;}
int main(void){
  uint32_t v; int fails=0;
  if (fpga_mgmt_init()) { printf("mgmt_init FAIL\n"); return 2; }
  pci_bar_handle_t h = PCI_BAR_HANDLE_INIT;
  if (fpga_pci_attach(0, FPGA_APP_PF, APP_PF_BAR0, 0, &h)) { printf("attach FAIL\n"); return 2; }
  printf("BAR0 attached\n bridge window:\n");
  v=rd(h,0x0000); if(v!=0x41394558){printf("  ^ HARD FAIL want A9EX 0x41394558\n");fails++;}
  v=rd(h,0x0004);                       /* APEX_CL_VER, expect 0x00000002 (record) */
  fpga_pci_poke(h,0x0008,0xA5A55A5A); v=rd(h,0x0008); if(v!=0xA5A55A5A){printf("  ^ HARD FAIL scratch\n");fails++;}
  fpga_pci_poke(h,0x0008,0x5A5AA5A5); v=rd(h,0x0008); if(v!=0x5A5AA5A5){printf("  ^ HARD FAIL scratch2\n");fails++;}
  v=rd(h,0x0010); if(v!=0x00000000){printf("  ^ HARD FAIL err_sticky must be 0 after load\n");fails++;}
  v=rd(h,0x0014);                       /* done_sticky (record) */
  v=rd(h,0x0018);                       /* kv evict/irq (record) */
  v=rd(h,0x0020);                       /* expect 0xDEAD0020 on the 05efb2a build (record) */
  printf(" KVQ INFO window:\n");
  v=rd(h,0x2008); if(v!=0x00000080){printf("  ^ HARD FAIL INFO_DIM must be 0x80 (D=128). 0x40 => the D=64 first-light AFI is loaded\n");fails++;}
  v=rd(h,0x200C);                       /* INFO_TIER, expect 0x1 = CQ-8 (record) */
  v=rd(h,0x2010);                       /* INFO_GROUP, expect 0x10 = KVQ_G 16 (record) */
  v=rd(h,0x2020);                       /* INFO_VERSION, expect 0x00020001 (record) */
  printf(" tile CSR window:\n");
  v=rd(h,0x1000); v=rd(h,0x1004);       /* STATUS, expect 0x1 idle (record) */
  printf(fails?"PROBE: %d HARD FAILS\n":"PROBE: ALL HARD CHECKS PASS\n",fails);
  return fails?1:0;
}
EOC
gcc -I sdk/userspace/include /tmp/probe_d.c -o /tmp/probe_d -lfpga_mgmt
sudo /tmp/probe_d | tee ~/apex_d/bar0_probe.txt ; echo "probe rc=$?"
```

| addr | expect | gate | provenance |
|---|---|---|---|
| `0x0000` | `0x41394558` "A9EX" | **HARD** | `cl_apex.sv:248` @ `05efb2a`; read from *this* AFI on 2026-07-22 |
| `0x0008` | RW `A5A55A5A` / `5A5AA5A5` | **HARD** | first light |
| `0x0010` | `0x00000000` | **HARD** | `err_sticky[15:0]`, clean after load (first light) |
| `0x2008` | `0x00000080` | **HARD** | `INFO_DIM = VECTOR_DIM` (`kvq_engine.sv:213/1083`) driven by `-verilog_define APEX_CL_D=128` (`synth_cl_apex.tcl:65` @ `05efb2a`). **The wrong-AFI detector**: first light's D=64 image reads `0x40` |
| `0x0004` | `0x00000002` | record | `APEX_CL_VER` @ `05efb2a`; never read on silicon before — a miss means *investigate*, not "known bad" |
| `0x0020` | `0xDEAD0020` | record | at `05efb2a` there is **no** `0x20` bridge register, so the `{24'hDEAD00, addr}` default answers. A live value here means the loaded bitstream is **not** the `05efb2a` build |
| `0x200C` / `0x2010` / `0x2020` | `0x1` / `0x10` / `0x00020001` | record | tier CQ-8; `KVQ_G(16)` (`cl_apex.sv:544` @ `05efb2a`); version as first light |
| `0x1000` / `0x1004` | — / `0x1` idle | record | tile CSR STATUS (first light) |

**Go/no-go:** any HARD failure ⇒ §8. All four HARD checks green ⇒ the two-clock
CL is answering over PCIe with the right head dimension, and the ~$1.98/hr is
now buying real work.

---

## §5 · THE PROMPT RUN ⏸

### 5.1 Why a transport exists at all

`tile_exec_bridge.run_job(executor="hw")` spawns `f2_host_run.py` as a **local**
subprocess. On this Mac that is a machine with no PCI device. The 7.6 GB weight
cache, the tokenizer and mlx cannot move (and `--prepare` is macOS-only), while
the device cannot move either — so the unit crossing the wire is **one whole
regops job** (~2.6k–7.8k ops, 80–240 KB), never one MMIO. Five jobs per prompt.

### 5.2 Path A — from the Mac, tile in the loop (default) ✅ *(wiring verified offline)*

`scripts/fpga/f2/remote_hw_exec.py` (sibling lane, present at session time,
`--selftest` 12/12 green — §10.7) implements exactly that transport: clock gate
first, then scp the regops + `f2_host_run.py` to `~/apexrun`, run it under
`sudo`, recover the runner's real rc through an `APEX_REMOTE_RC` sentinel, scp
the cap JSONL back, and hand `tile_exec_bridge.parse_cap_file` the same 11-key
dict the sim path returns.

It is wired in with **no file edited** — `attach()` re-points
`bridge.run_job`, and is a no-op unless `$APEX_F2_HOST` is set:

```sh
cd ~/Desktop/apex-promptdemo
export APEX_F2_HOST=ubuntu@$IP
export APEX_F2_KEY=~/.ssh/apex-f2.pem
mkdir -p /tmp/apex_d

~/.venvs/apex-eval/bin/python -c "
import sys
sys.path.insert(0, 'scripts/fpga/f2')
import prompt_offload as po, remote_hw_exec as rhe
assert rhe.attach(po.bridge), 'remote executor NOT armed — refusing to run'
sys.exit(po.main())
" --prompt "The capital of France is" --max-tokens 1 \
  --offload-layer 0 --offload-head 0 \
  --executor hw --poison 2.0 \
  --work-dir build/prompt_offload_hw \
  2>&1 | tee /tmp/apex_d/milestone_d_run.log
```

* `--work-dir build/prompt_offload_hw` — **never** the sim evidence dir (§0.4).
* `--poison 2.0` is not optional: it re-runs the ON decode with the tile's value
  scaled and reports `max|dlogit|`. If that is 0, the substitution was never
  load-bearing and **every** PASS above it is vacuous. It costs one extra
  hardware pass (5 more jobs) and one extra golden decode.
* Defaults `--tier kvq8 --group 128` match the banked sim run; leave them alone.
* Expect ~150 s per golden decode on the Mac (≈29 s/step × 5 prompt steps + the
  generated token) × 3 decodes (on / off / poison) ≈ **8 min**, plus hardware
  time. The 5 D jobs total **26,005 ops** — 1.3 % of the 1.95 M ops the
  18-job silicon replay already executed inside a 1.3 h session, so BAR0 time
  is minutes, not hours.
* The banner prints `executor : hw` and, per capture,
  `acc source : TILE via compute_job.grade_compute_job (hw)`.

### 5.3 Path B — everything on the instance (fallback)

Only if Path A cannot be armed. After §2's Path-B copy:

```sh
# on the instance
cd ~/apex_d/repo && cat COMMIT.txt          # <-- into the log
sudo -E python3 scripts/fpga/f2/prompt_offload.py \
  --ids 785 6722 315 9625 374 --max-tokens 1 \
  --offload-layer 0 --offload-head 0 \
  --executor hw --poison 2.0 \
  --weights-dir ~/apex_d/s8_weights/Qwen2.5-7B-4bit \
  --work-dir ~/apex_d/work 2>&1 | tee ~/apex_d/milestone_d_run.log
```

Path-B differences to disclose in the log, not hide:
* `sudo` is required for BAR0, so root's `python3` must import numpy.
* No tokenizer on the instance ⇒ `--ids 785 6722 315 9625 374` (the exact ids
  the banked run used for "The capital of France is") and the banner shows
  `prompt '<ids [...]>'`. The token → " Paris" decode happens back on the Mac.
* The A/B and the bit-exactness are still *within-host*, so the gate is intact.
  If the instance's golden picks a different token than the Mac's `12095`, that
  is a **host-numerics** disclosure (different BLAS under the float64 reference
  paths), not a tile failure — record both and say so.
* `f2_host_run.py`'s vacuous clock preflight is the *only* clock check on this
  path, so §3.3's frequency read-back must be re-run immediately before and
  after (Path A's per-job gate does not exist here).

### 5.4 Optional provenance anchor (~$1, only if time allows)

Replay the canonical 18 jobs on the instance and expect the published line
`F2HOST RESULT: files=18 checks=27996 fails=0 -> PASS` — the same numbers as
`docs/results/f2_stage2_hw/replay_silicon_PASS.log`. It proves the tile is
healthy independently of the new code and is the natural first bisector if §5
goes red (§8 row F).

---

## §6 · EVIDENCE CAPTURE ⏸ — what must be in the commit for the claim to stand

### 6.1 Files to pull back, into `docs/results/f2_milestone_d/`

```sh
# from the Mac
mkdir -p docs/results/f2_milestone_d
scp -i ~/.ssh/apex-f2.pem \
    ubuntu@$IP:'~/apex_d/clkgen_after_recipe.txt ~/apex_d/bar0_probe.txt' \
    docs/results/f2_milestone_d/
scp -i ~/.ssh/apex-f2.pem ubuntu@$IP:'~/apexrun/*.cap.jsonl' \
    docs/results/f2_milestone_d/remote_caps/          # Path A
cp /tmp/apex_d/milestone_d_run.log docs/results/f2_milestone_d/
cp build/prompt_offload_hw/prompt_offload_result.json docs/results/f2_milestone_d/
cp build/prompt_offload_hw/*.cap.jsonl build/prompt_offload_hw/*.manifest.json \
   docs/results/f2_milestone_d/
```
Plus: the §0.1 preflight output, the AFI describe output, the AGFI-load and
`fpga-describe-local-image` transcript, the instance id / AZ / launch+terminate
timestamps, and the teardown proof from §7.

Do **not** commit the 5 `*.compute.regops.jsonl` if size is a concern — but do
record their `sha256` and the per-job `inputs_sha256` from each manifest, since
§6.3's parity check refers to them.

### 6.2 The MUST-CONTAIN checklist — a missing line voids the claim

1. `executor        : hw` (not `sim`, not `golden`) — kills F2/F4.
2. For **every** capture: `acc source   : TILE via … (hw)` — kills F2.
3. For **every** capture: `s_c source   : TILE (…)`. A single
   `GOLDEN (no ss tap captured)` voids it — kills F3.
4. For **every** capture: `substituted  : True   consumed by model: True   core object kept: True`.
5. `tile vs golden  : out_hat bit-exact 5/5 ; acc_o bit-exact 5/5`.
6. `TOKEN IDENTITY: PASS` **and** `MILESTONE C   : PASS` (the banner reuses the
   C wording; the D-ness comes from line 1 plus the hardware transcript).
7. `discriminator   : tile value x2.0 -> …, max|dlogit|=<non-zero> (substitution IS load-bearing)`.
8. The clkgen table showing `clk_extra_a1 = 15.62`, captured **before** the run
   *and* re-read after it (§7.1) — kills F1.
9. `PROBE: ALL HARD CHECKS PASS` with `peek 0x2008 = 0x00000080` visible — kills F5.
10. Zero `REMOTE NOTE:` / `NOTE:` lines that are unexplained; every note quoted
    and answered in `RESULT.md`.

### 6.3 The provenance check that cannot be faked (kills F4)

The demo's banner cannot prove a PCI device existed. This can: re-execute the
*same* regops files the demo just sent, directly through the runner, and diff
the capture egress.

```sh
# Path A: the files are already on the instance from the demo run
ssh -i ~/.ssh/apex-f2.pem ubuntu@$IP \
  'cd ~/apexrun && sudo python3 ./f2_host_run.py ./poff_s000_L00_h00.compute.regops.jsonl \
      --cap-out ./reexec_s000.cap.jsonl' | tee -a /tmp/apex_d/reexec.log
scp -i ~/.ssh/apex-f2.pem ubuntu@$IP:'~/apexrun/reexec_s000.cap.jsonl' /tmp/apex_d/
diff /tmp/apex_d/reexec_s000.cap.jsonl \
     build/prompt_offload_hw/poff_s000_L00_h00.cap.jsonl && echo "REEXEC PARITY: byte-identical"
```

Byte-identical is the expected result (the tile is deterministic and
`f2_host_run.py` asserts TILE_RST before every file). If a handful of records
differ, they must be named and explained in `RESULT.md`, and **none** of them
may be `ro_lanes` — those are the INT32 accumulators the whole claim rests on.

Also worth pasting: the instance-side `F2HOST CAPTURES: n=… out=…` and
`F2HOST RESULT: … -> PASS` lines, which only a real runner on a real device
prints.

### 6.4 `RESULT.md` skeleton

Configuration (AFI/AGFI, instance id + AZ, CL commit `05efb2a`, D=128 / G=16 /
DEPTH=256 / GQA_NENG=1 / maskless, tile clock 15.625 MHz) · the verbatim
banner · the checklist above with each line quoted · the §6.3 parity result ·
every disclosure (the vacuous preflight, the skipped debug taps, the
0.13 % offload fraction, `--poison` numbers) · spend · and the exact claim
sentence from §9.

---

## §7 · TEARDOWN ⏸ — terminate, *verify*, record

### 7.1 Before you kill it

```sh
# instance: prove the clock never moved during the session
fpga-describe-clkgen -S 0 | tee ~/apex_d/clkgen_after_run.txt   # a1 must still be 15.62
sudo /tmp/probe_d | tee ~/apex_d/bar0_probe_after.txt           # err_sticky still 0
```
Then confirm every file in §6.1 is already on the Mac.

### 7.2 Terminate (never stop)

```sh
aws ec2 terminate-instances --region us-west-2 --instance-ids "$IID"
aws ec2 wait instance-terminated --region us-west-2 --instance-ids "$IID"
aws ec2 describe-instances --region us-west-2 --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text   # must print: terminated
```

`stop` keeps the EBS bill and the AFI-load state; terminate is the only correct
end state. The AFI is untouched — leave it (storage is free and D may be
re-run).

### 7.3 Sweep for leaks and record spend

```sh
aws ec2 describe-instances --region us-west-2 \
  --filters Name=instance-state-name,Values=running,pending Name=instance-type,Values=f2.6xlarge,f2.12xlarge,f2.48xlarge \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text   # must be EMPTY
aws ec2 describe-volumes --region us-west-2 --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output text        # must be EMPTY (no orphan EBS)
T_END=$(date -u +%s); H=$(python3 -c "print(($T_END-$T_START)/3600)")
python3 -c "print('f2.6xlarge %.2f h x \$1.98/hr = \$%.2f' % ($H, $H*1.98))"
```

Record in `RESULT.md`: wall hours, instance cost at $1.98/hr, any S3/EBS
extras, and the total. Previous silicon session for calibration: **~$5.50** all
in (devbox included); D has no devbox, so expect **$2–4**.

---

## §8 · ABORT LADDER

"Keep spending?" assumes a $1.98/hr meter and a 2 h cap. When in doubt:
terminate, fix on the Mac for free, relaunch. A relaunch costs ~5 min and
~$0.20; debugging on a live instance costs $0.033/min and tempts shortcuts.

| # | symptom | most likely cause | do this | keep spending? |
|---|---|---|---|---|
| **A** | AFI not `available` in §0.2 | AFI deleted / wrong id / wrong region | **STOP at §0 — nothing is launched.** Re-derive the AFI from `docs/results/f2_stage2_hw/afi_final.txt`; if it is truly gone, D needs a rebuild (devbox + DCP + ingestion, a different, day-scale session) | N/A ($0 spent) |
| **B** | `run-instances` → `Unsupported`/`InsufficientInstanceCapacity` | AZ has no f2 capacity, or the subnet is in `us-west-2a/d` (§1.2) | retry pinned to `subnet-0dbb62b25dea9e7a3` (`us-west-2c`); if both AZs are dry, wait or use `us-east-1` (needs the AFI to exist there — it does not; then STOP) | nothing launched yet |
| **C** | `fpga-describe-clkgen` shows `a1 = 125.00` (or anything ≠ 15.62) | the AFI load reset the MMCMs and `-a 2` was skipped or failed | re-run `sudo fpga-load-clkgen-recipe -S 0 -a 2`, re-read. Still wrong ⇒ reload the AGFI, program again, re-read. **Run zero jobs until the number is right** — at 125 MHz the tile is 8× over its closed clock and everything it produces is garbage | yes, ≤10 min; then terminate |
| **D** | BAR0 `0x0000` ≠ `A9EX`, or attach/mgmt_init fails | AFI not actually loaded; probe run before the recipe (the CDC dead-clock guard poisons the bridge until reset — by design) | re-check §3.2 output, program the recipe, reload the AGFI once, re-probe | yes, ≤15 min |
| **E** | `0x2008` = `0x00000040` | the **D=64 first-light AGFI** is loaded (F5) | load `agfi-0ae06ea568e5667ba`, redo §3.3 + §4 | yes, ≤10 min |
| **F** | `poll`/`pw`/`jf` stall, or `out_hat` DIFF vs golden | wrong clock (recheck C first), or a genuine job/compiler defect | 1) re-read the clock; 2) run §5.4's 18-job canonical replay — **green** ⇒ the tile is fine and the defect is in the new compute-mode job (host-side; terminate and fix free), **red** ⇒ hardware/bring-up problem, terminate and reproduce in the sim | ≤20 min for the bisect, then terminate |
| **G** | `CAPMISMATCH` lines | a `cap` carrying an expectation missed it | that is a real data failure — capture the log, terminate, reproduce in `f2sim` (`obj_d128_ddr0`). Do **not** relax the check | no |
| **H** | `TOKEN IDENTITY: FAIL` with `out_hat` bit-exact 5/5 | the substitution is fine but a *different* op diverged, or a stale work dir | keep the log (this is a real finding, not a fake pass), re-run once with a clean `--work-dir`; then terminate and debug in sim | ≤1 re-run |
| **I** | `s_c source : GOLDEN (no ss tap captured)` | the `ss` tap was not captured; the epilogue used golden's scale (F3) | the run does **not** support the claim. Terminate, fix the emitter/decoder on the Mac (the sim reproduces it), relaunch later | no |
| **J** | `REFUSED (clock gate) … rc=78` from `remote_hw_exec` | its independent gate caught a wrong/unreadable clock — **nothing was shipped** | fix per row C; never pass `--no-verify-clock` to get past it | yes, ≤10 min |
| **K** | ssh times out / drops | this Mac's public IP changed (the SG has 5 pinned CIDRs), or a transient | re-run §0.1 (it checks IP coverage explicitly); add an ingress rule **only with owner approval**; retry with `-o ServerAliveInterval=20`. Jobs are per-file and self-contained, so a re-run after a drop is safe | yes, ≤10 min |
| **L** | `sdk_setup.sh` / bindings missing (`ABORT: fpga_pci bindings not built`) | clone is not at `/home/ubuntu/aws-fpga`, or `source` failed under `set -u` | re-clone to that exact path, `set +u`, re-source, re-verify the `.so`; or pass `--remote-env AWS_FPGA_REPO_DIR=…` | yes, ≤15 min |
| **M** | Path-B weight copy ETA > 25 min | uplink | abort Path B, arm Path A (which ships 240 KB per job instead of 7.6 GB) | no — terminate rather than burn on a copy |
| **N** | 2 h cap approaching | anything | stop, pull evidence, terminate. Publish only what is already green; an unfinished D is a status, not a failure | no |

---

## §9 · CLAIM DISCIPLINE

### The session MAY publish, if and only if every §6.2 line is present

> On AWS F2 silicon (VU47P, AFI `afi-036d83cafa00d26ea` / AGFI
> `agfi-0ae06ea568e5667ba`, the two-clock Level-C CL with the tile on
> `clk_extra_a1` recipe A2 = **15.625 MHz**), **one attention operation of a
> real Qwen2.5-7B decode step was computed by the APEX tile and its value was
> used to produce the token.** The tile returned raw INT32 accumulators the
> host had not computed (requant disabled on-tile, `cap` egress), the host
> epilogue turned them into the layer's attention output through golden's own
> requant functions, and greedy decode emitted the **same token as the pure
> golden path** (`5/5` offloaded ops bit-exact vs golden; poison
> discriminator `max|Δlogit| > 0`, so the returned value was load-bearing).

Add, in the same breath, every time:

> One operation of 3,920 attention-core ops in this prompt (28 layers × 28
> heads × 5 steps) was offloaded — **0.13 %**. 15.625 MHz is a correctness
> clock, not a performance number. Debug-tap expectations (TAPF16/TAPSC/TAPPR)
> are skipped and counted, as in simulation.

### The session MAY NOT publish

* "Qwen2.5-7B runs on our chip" / "we ran a 7B model on the FPGA" — one
  attention op did; the other 99.87 % of attention, all projections, all MLPs,
  RoPE, norms, sampling and the KV cache ran in golden on a Mac.
* "a decoder layer runs on the FPGA" — no full layer has ever run on hardware
  (see `docs/design/IB_LAYER.md` and the I-B capability gaps: a walked FULL
  layer is not achievable on today's RTL).
* Any tokens/s, latency, throughput or speed-up number. Any comparison to a
  GPU/competitor. 15.625 MHz is a bring-up clock; the previous session's own
  wording ("a correctness clock, not a performance number") is binding here.
* Anything sourced from `--dry-run`, from `--executor sim`, from a run whose
  `s_c source` says `GOLDEN`, or from a run without the clkgen frequency
  read-back in the same log.
* "the prompt was computed on the FPGA" / "our chip answered the prompt".
* Any claim about the walker, DRAM weight streaming, GQA-4, wide-D, masked
  attention, or T > 128 (the seam hard-refuses chunked heads).
* Any number not present verbatim in a committed log.

---

## §10 · WHAT WAS ACTUALLY VERIFIED WHILE WRITING THIS ✅ ($0, no instance)

### 10.1 The preflight, run on this Mac (2026-07-30T01:50:23Z, fast mode)

Verbatim, complete, single run — nothing merged or reflowed:

```
==================================================================
APEX MILESTONE D — PREFLIGHT (read-only, $0, no instance required)
date   : 2026-07-30T01:50:23Z UTC
repo   : /Users/nabilabdelazizferhattaleb/Desktop/apex-promptdemo
region : us-west-2   afi: afi-036d83cafa00d26ea   type: f2.6xlarge
mode   : fast
==================================================================

== A. LOCAL TREE
  [PASS] git tip: comp/prompt-b-c @ 6aac4fa  (MILESTONE B+C GREEN: a real 7B prompt's attention op computed by)
  [WARN] working tree has 3 modified/untracked path(s) — the session log MUST record what they are (parallel lanes are editing this tree)
  [PASS] present: scripts/fpga/f2/prompt_offload.py
  [PASS] present: scripts/fpga/f2/compute_job.py
  [PASS] present: scripts/fpga/f2/tile_exec_bridge.py
  [PASS] present: scripts/fpga/f2/cap_decode.py
  [PASS] present: scripts/fpga/f2/f2_host_run.py
  [PASS] present: scripts/fpga/f2/trace_to_regops.py
  [PASS] present: run_tinynpu.py
  [PASS] present: verif/top/l3/gen_l3_vectors.py
  [PASS] present: golden/apex_golden/transformer.py
  [PASS] present: golden/apex_golden/attention.py

== A2. SELFTESTS (the demo's own gates, host-only, no executor)
  [PASS] compute_job --selftest — COMPUTE_JOB SELFTEST: PASS (fails=0; T=5 D=128, 7823 regops, 232 caps, 119 structural ch
  [PASS] tile_exec_bridge --selftest — BRIDGE SELFTEST: PASS (fails=0)
  [PASS] prompt_offload --selftest — PROMPT_OFFLOAD SELFTEST: ALL PASS
  [PASS] f2_host_run.py compiles (it can only be *run* on the instance)

== A3. BANKED SIM EVIDENCE (Milestone C) — is it green, and does it still describe HEAD?
  [PASS] banked token_identity = True
  [PASS] banked milestone_c = True
  [info] banked mode=sim captures=5 token=[12095] text=' Paris' git=0e2a18c
  [PASS] every banked capture has acc_source AND s_c_source = TILE (no golden fallback)
  [WARN] evidence recorded git=0e2a18c but these load-bearing files changed by HEAD: scripts/fpga/f2/compute_job.py scripts/fpga/f2/prompt_offload.py scripts/fpga/f2/smoke_bridge_sim.py   -> re-verify with --deep before spending (expected when the run pre-dated its own commit)
  [info] banked artifacts: 5 regops + 5 cap files under build/prompt_offload

== A4. THE 7B MODEL (stays on this Mac in Path A)
  [info] weights: mlx-community/Qwen2.5-7B-4bit  L=28 H=28 head_dim=128  (…/build/s8_weights/Qwen2.5-7B-4bit)
  [PASS] head_dim (must match APEX_CL_D=128 in the AFI) = 128
  [PASS] 339 .npy weight shards present
  [PASS] tokenizer venv OK: /Users/…/.venvs/apex-eval/bin/python (numpy + transformers)
  [PASS] sim executor present (silicon twin, DDR=0) — --deep and the sim/hw A-B are possible

== B. RUN PATH ARMING (how the Mac's golden reaches the instance's PCI device)
  [info] remote executor: …/scripts/fpga/f2/remote_hw_exec.py
  [PASS] remote_hw_exec --selftest — REMOTE_HW_EXEC SELFTEST: PASS (fails=0)
  [PASS] PATH A ARMED: attach() re-points bridge.run_job at the remote executor (verified offline, no host contacted)
  [info] prompt_offload.py does NOT import remote_hw_exec (that 2-line edit belongs to its owner) — use the runbook §5.2 python -c wrapper instead; no file needs editing
  [PASS] remote executor exposes --check-clock (runbook §3.4 uses it as the second, independent clock gate)
  [PASS] PATH B ARMED: whole demo can run ON the instance (weights + golden + l3 generator all shippable; GoldenModel is mmap-numpy, no mlx needed)
  [info] session will run PATH A (runbook §5)

== C. AWS PRECONDITIONS (all read-only)
  [info] aws-cli/2.33.22 Python/3.13.12 Darwin/24.6.0 source/arm64
  [PASS] identity: arn:aws:iam::099597653601:user/Aziz (account 099597653601)

== C1. AFI LIVENESS — the exact command the runbook publishes
  $ aws ec2 describe-fpga-images --region us-west-2 --fpga-image-ids afi-036d83cafa00d26ea
  [PASS] AFI state = available
  [PASS] AGFI id = agfi-0ae06ea568e5667ba
  [PASS] AFI shell = 0x10212415
  [info] AFI name=apex-lc1-a2-20260722 created=2026-07-22T22:34:25+00:00

== C2. LAUNCH INPUTS
  [PASS] key pair 'apex-f2' exists in us-west-2
  [PASS] private key /Users/…/.ssh/apex-f2.pem present, mode 400
  [PASS] AMI state = available
  [info] AMI ami-07a164f1a402ab274 = FPGA Developer AMI (Ubuntu) - 1.19.2-prod-rhng4b6alkhdq /dev/sda1
  [info] f2.6xlarge offered in: us-west-2c us-west-2b
  [PASS] pinned --subnet-id subnet-0ee519f2d304c99c9 is in us-west-2b, which offers f2.6xlarge
  [PASS] security group apex-f2-ssh = sg-0766e253ceeaa3b74
  [info] port 22 allowed from: 98.210.40.192/32 24.23.136.255/32 172.59.162.137/32 172.56.0.0/13 46.248.159.10/32
  [PASS] this Mac (98.210.40.192) is covered by an existing port-22 rule

== C3. QUOTA, COST, SPEND LEAKS
  [PASS] quota L-74FC7D96 (Running On-Demand F instances) = 128.0 vCPU >= 24 needed by f2.6xlarge
  [PASS] f2.6xlarge on-demand = $1.9800/hr in us-west-2 (AWS Pricing API, primary source) <= ceiling $2.50
  [info] budget: a 1 h session ~= $1.9800 ; the runbook's 2 h hard cap ~= $3.96
  [WARN] instances already running/pending in us-west-2 — confirm none is yours to pay for, and NEVER touch the verifagent/Catapult boxes:
         i-0e6661e09a5a350dc  t3.micro     2026-01-29T04:42:43+00:00
         i-0d477329467397592  c6a.xlarge   2026-02-07T19:03:00+00:00
         i-0a4cb41b9df82c69f  t3.micro     2026-03-29T06:22:24+00:00
         i-0bb4b6856db46fb95  c6a.xlarge   2026-03-28T07:10:04+00:00
         i-076d466a47b2dc331  c6a.xlarge   2026-03-29T06:20:58+00:00
  [info] (--deep not requested: the sim demo was NOT re-run at this commit. Runbook §0.4 requires it before launch.)

==================================================================
PREFLIGHT: 38 pass, 3 warn, 0 FAIL
VERDICT: CLEAR TO LAUNCH (runbook §1). Read every [WARN] above first.
Paste this whole output into the session log BEFORE run-instances.
==================================================================
```

The three `[WARN]`s, answered: (1) the 3 untracked paths are this runbook,
`d_preflight.sh` and the sibling lane's `remote_hw_exec.py`; (2) the banked
evidence pre-dates its own commit → §0.4 `--deep` is mandatory; (3) the 5
running instances are the long-lived verifagent/Catapult boxes — **no `f2.*`
is running**, so there is no pre-existing FPGA spend and nothing of theirs is
to be touched.

**The gate bites** (fault injection, same script, two values overridden):

```
$ AGFI=agfi-0000000000000000 SUBNET=subnet-08df413f13f18d476 bash scripts/fpga/f2/d_preflight.sh
  [FAIL] AGFI id = 'agfi-0ae06ea568e5667ba' (want 'agfi-0000000000000000')
  [FAIL] pinned subnet subnet-08df413f13f18d476 is in us-west-2a, which does NOT offer
         f2.6xlarge (offered: us-west-2b us-west-2c)
  … VERDICT: DO NOT LAUNCH — 2 precondition(s) failed
EXIT=1
```

### 10.2 AFI liveness — it is real, not a doc claim ✅

```
$ aws ec2 describe-fpga-images --region us-west-2 --fpga-image-ids afi-036d83cafa00d26ea
{ "FpgaImageId": "afi-036d83cafa00d26ea",
  "AgfiId": "agfi-0ae06ea568e5667ba",
  "Name": "apex-lc1-a2-20260722",
  "State": "available",
  "Created": "2026-07-22T22:34:25+00:00",
  "Public": false,
  "Shell": "0x10212415" }
```

### 10.3 Price ✅ — the brief's $1.65/hr is wrong

```
$ aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
    --filters …instanceType=f2.6xlarge …regionCode=us-west-2 …Linux/Shared/NA/Used
f2.6xlarge {'USD': '1.9800000000'} Hrs $1.98 per On Demand Linux f2.6xlarge Instance Hour
```
In-repo agreement: `BRINGUP.md` §1 ("~**$1.98/hr**") and §4; `f2_stage2_hw/RESULT.md`
("f2.6xlarge ~1.3 h ≈ $2.60").

### 10.4 The AZ trap ✅

```
$ aws ec2 describe-instance-type-offerings --region us-west-2 --location-type availability-zone \
    --filters Name=instance-type,Values=f2.6xlarge --query 'InstanceTypeOfferings[].Location'
[ "us-west-2c", "us-west-2b" ]
$ # default VPC vpc-098ca8e5e6de24c9a subnets: 2a, 2b, 2c, 2d (all MapPublicIpOnLaunch=true)
```

### 10.5 Launch inputs ✅

key pair `apex-f2` = `key-0920c2c03d7a2ddd6` (ed25519) · local
`~/.ssh/apex-f2.pem` mode 400 · SG `apex-f2-ssh` = `sg-0766e253ceeaa3b74`
(port 22 from 5 CIDRs incl. this Mac's `98.210.40.192/32`) · AMI
`ami-07a164f1a402ab274` = *FPGA Developer AMI (Ubuntu) 1.19.2-prod*,
root `/dev/sda1` 120 GiB gp3 + `/dev/sdm` 10 GiB, both
`DeleteOnTermination=true` · quota `L-74FC7D96` = **128** vCPU · no `f2.*`
instance running · no apex instance profile exists (only Catapult's — do not
touch).

### 10.6 The D workload, measured from the banked artifacts ✅

```
poff_s000_L00_h00.compute.regops.jsonl  2579 ops  { w 1099, pw 1092, jf 128, poll 67, r 18, rn 2, cap 164, note 9 }   82 KB
poff_s001…                              3890 ops  … cap 181                                                         122 KB
poff_s002…                              5201 ops  … cap 198                                                         162 KB
poff_s003…                              6512 ops  … cap 215                                                         202 KB
poff_s004…                              7823 ops  … cap 232                                                         242 KB
                                       ------                                                                       ------
                                       26,005 ops total, 990 cap records, ~0.8 MB
```
vs the proven silicon replay: 18 files, ~1.95 M ops, `checks=27996 fails=0`
inside a ~1.3 h session. D is **1.3 %** of that BAR0 traffic — per-job round
trips over ssh are comfortable; per-op round trips would not be (hence §5.1).
Per-job manifest confirms the compute-mode shape: `requant_en 0`, `D 128`,
`tier CQ-8`, `cap_census {'ro_lanes':128,'ro_meta':16,'fs':…,'ss':2}`,
`pv_desc_rq_cleared 16`, `inputs_sha256 …`.

### 10.7 The transport is real and self-testing ✅

```
$ python3 scripts/fpga/f2/remote_hw_exec.py --selftest
  [2] ok  a1=125.00 REFUSED rc=78 (probe only, 1 ssh call)
  [3a] ok a1='0.00' refused …    [3b] ok a1='garbage' refused
  [8] ok  shipped f2_host_run.py verbatim
  [10] ok same 11 keys + same types as run_job
  [12] ok clkgen_final.txt -> a1=15.62 MHz; header-indexed
REMOTE_HW_EXEC SELFTEST: PASS (fails=0)

$ APEX_F2_HOST=ubuntu@203.0.113.1 python3 -c "…rhe.attach(po.bridge)…"
[remote_hw_exec] executor 'hw' -> ubuntu@203.0.113.1:~/apexrun (clock gate ON)
attached = True ; run_job is now remote_hw_exec._dispatch
$ python3 -c "…same wrapper…" --selftest     ->  PROMPT_OFFLOAD SELFTEST: ALL PASS (exit 0)
```
(A fake host was used; nothing was contacted — `attach()` only re-points the
dispatcher.)

### 10.8 Register expectations, read out of the *bitstream's own* commit ✅

`git show 05efb2a:scripts/fpga/f2/cl_apex/design/cl_apex.sv` →
`APEX_CL_ID = 32'h4139_4558`, `APEX_CL_VER = 32'h0000_0002`, `.KVQ_G(16)`,
`CL_KVQ_DEPTH = 256`, `default: bridge_rd = {24'hDEAD00, a}`, **no** `8'h20`
register; `git show 05efb2a:…/synth_cl_apex.tcl` → `-verilog_define
APEX_CL_D=128`. `rtl/kvq/kvq_engine.sv:213/1083` → `REG_INFO_DIM = 8'h08`
returns `VECTOR_DIM`. That is where §4's `0x2008 == 0x00000080` gate comes from.

### 10.9 §3.3's clock gate, tested against the real recorded output ✅

Run on the *actual* table the 2026-07-22 silicon session committed, and on the
125 MHz default it must refuse:

```
$ A1=$(awk '…' docs/results/f2_stage2_hw/clkgen_final.txt); echo "A1='$A1'"
A1='15.62'
GATE PASS
$ # same parser, same gate, on a synthesized default-recipe table:
A1='125.00'
GATE FAIL (correct — refuses the 125 MHz default)
```
So the gate parses the tool's real format (header-indexed, not column-guessed),
accepts 15.62 (|15.62 − 15.625| = 0.005 ≤ 0.05) and rejects the exact value a
forgotten `-a 2` would leave behind. It is a real check, not decoration.

---

## §11 · OPEN DEPENDENCIES AND OWNERSHIP

| item | owner | state at `6aac4fa` |
|---|---|---|
| `docs/design/MILESTONE_D_RUNBOOK.md`, `scripts/fpga/f2/d_preflight.sh` | this lane | landed, preflight **run** (§10.1) |
| `scripts/fpga/f2/remote_hw_exec.py` | sibling lane | present, `--selftest` green, wiring verified offline (§10.7); **never exercised against a real instance** |
| the optional 2-line `import remote_hw_exec` / `attach()` edit in `prompt_offload.py` | that file's owner | not applied — **not needed**: §5.2's `python -c` wrapper does it without editing anything |
| `f2_host_run.py`'s vacuous MMCM preflight | that file's owner | still vacuous; §3.3/§3.4 replace it rather than trusting it. Fixing it (assert `a1 ≈ 15.62` directly) remains the follow-up chip opened by `f2_stage2_hw/RESULT.md` §5 |
| `f2_smoke_session.sh`'s stale `AGFI=agfi-0f7c93ffa798ecc3f` | that file's owner | still the D=64 first-light image — §3.1/§4 route around it |

**Never verified by anything in this document:** that the transport works
against a live instance, that BAR0 answers today, or that the tile reproduces
the 5 captures on silicon. Those are exactly what the live session buys, and
none of them may be claimed before their log exists.
