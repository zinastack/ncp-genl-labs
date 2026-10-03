# Running the GPU labs on Brev (or AWS)

Every lab has a CPU part (`exercises.py` + tests + quiz, runs on a laptop) and a GPU part
(`gpu_lab.py` and section `make` targets) meant for a **cheap** cloud GPU. You don't need
A100/H100s: every GPU lab is sized for a **24 GB L4** or a **16 GB T4**.

## Which instance for which section

| Section | Labs | Recommended | Budget option | Why |
|---|---|---|---|---|
| **1** Foundations & Prompting | 01, 02 | 1× **L4** 24 GB | 1× T4 16 GB | FlashAttention needs sm_80+ (L4 has it, T4 shows "unsupported"), bf16 |
| **2** Data & Fine-Tuning | 03, 04 | 1× **L4** 24 GB | 1× T4 16 GB | full FT vs LoRA vs QLoRA memory comparison of a 0.5B model |
| **3** Optimization & Acceleration | 05, 06 | **2× L4** (or 2× T4) | 1× L4 for Lab 05 only | Lab 06 needs ≥ 2 GPUs for real NCCL, DDP scaling and FSDP sharding |
| **4** Deployment & Monitoring | 07, 08 | 1× **L4** 24 GB, **200 GB disk** | 1× T4 (no TensorRT-LLM FP8, most NIMs won't fit) | Triton, TensorRT, TensorRT-LLM, NIM, k3s Kubernetes, Prometheus + Grafana + DCGM: ~150 GB of container images |
| **5** Evaluation & Responsible AI | 09, 10 | 1× **L4** 24 GB | 1× T4 16 GB | vLLM server + evaluation models + guardrails together |

Rough cost: T4 instances are typically **$0.35–0.60/h** and L4 **$0.70–1.00/h** (2× L4 ≈ 2×), so a
2–3 hour session per section costs about **$1–6**. Prices vary by provider and region; check
Brev's instance picker. **Stop the instance when you're done.** Disk: 100 GB (models, Docker images).

> T4 notes: no bf16 (the labs switch to fp16 automatically), no FlashAttention-2, no FP8.
> Everything still runs, and the differences are themselves exam material.

## Private instances from your laptop (Brev CLI, no GitHub needed)

The fastest private path: the root Makefile's `brev-*` targets (in `brev/brev.mk`) create an
instance **in your own Brev org**, copy your **local** repo to it (tracked and untracked files, nothing
git-ignored, so your filled-in exercises come along), copy your Hugging Face token if one is set,
and run `brev/setup.sh`. Nobody else can see or use these instances.

```bash
brev login                       # once
make brev-plan S=3               # preview the GPU types that would be tried (free)
make brev-up S=3                 # create + copy repo + setup (billing starts)
make brev-shell S=3              # work on it (or: make brev-open S=3 for VS Code)
make brev-exec S=3 CMD=gpu-06    # or run a target remotely
make brev-sync S=3               # push local edits again
make brev-forward S=4 PORT=3000  # Grafana on http://localhost:3000 (PORT=9090 for Prometheus)
make brev-get S=3 FILE=labs/3-optimization-acceleration/06-gpu-acceleration/ddp_profile.nsys-rep
make brev-stop S=3               # stop billing for compute (disk kept); make brev-delete S=3 to remove
```

Defaults per section (cheapest first, falling back to the next type if one isn't available):
1× L4 for Sections 1, 2, 4 and 5 (Section 4 prefers 8 vCPUs), 2× L4 then 2× T4 for Section 3. Override with
`TYPE=<brev type>` (see `brev search gpu`) or `INSTANCE=<name>`. `brev create` doesn't take a disk
size. If a type's default disk turns out too small for models and Docker images, pick a type with a
larger disk (`brev search gpu --min-disk 100`).

## Create the Launchables (one per section)

The code source is the public repo **https://github.com/zinastack/ncp-genl-labs**. Brev clones it to
`/home/ubuntu/ncp-genl-labs`. Brev's CLI can **deploy** a Launchable
(`make brev-launch S=3 LAUNCHABLE=env-...`) but can't **create** one; that happens in the web console.

**[`LAUNCHABLES.md`](LAUNCHABLES.md) has the exact settings for all five**: names, GPUs, launch
parameter, secure links, descriptions, and the setup script to paste
([`launchable-setup.sh`](launchable-setup.sh)). Keep each one on **"Only my organization"** while you
test it (new Launchables default to *Anyone with the link*), and switch visibility when you open them.

## Using an instance

```bash
brev shell <instance-name>          # or open Jupyter from the Brev console
cd ~/ncp-genl-labs
make doctor                         # GPU, driver, torch, docker
make s3                             # list Section 3's targets
make gpu-05                         # run Lab 05's GPU part
make s3-gpu-06                      # multi-GPU: NCCL check, all-reduce bandwidth, DDP scaling, DDP vs FSDP
make s3-profile-06                  # Nsight Systems + PyTorch profiler traces
make s4-triton-up && make gpu-07    # Triton + perf_analyzer
make s5-llm-up && make s5-guardrails
```

On a Launchable the repo is the public version (empty exercise stubs). To bring your own
solutions, use the CLI path above (`make brev-sync`), which copies your local working tree. Run the GPU parts with **your** code via
`USE_EXERCISES=1 make gpu-02`.

Port forwarding without secure links: `brev port-forward <instance> -p 3000:3000`.

## AWS instead of Brev

| Section | Instance | GPU | Approx. on-demand (us-east-1) |
|---|---|---|---|
| 1, 2, 5 | `g6.xlarge` | 1× L4 24 GB | ~$0.80/h |
| 4 | `g6.2xlarge` | 1× L4 24 GB, 8 vCPUs (200 GB gp3) | ~$0.98/h |
| 1, 2, 5 (budget) | `g4dn.xlarge` | 1× T4 16 GB | ~$0.53/h |
| 3 | `g6.12xlarge` / `g4dn.12xlarge` | 4× L4 / 4× T4 | ~$4.60 / ~$3.90 per h |

1. AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** (driver, Docker and the NVIDIA Container Toolkit preinstalled).
2. 100 GB gp3 root volume (200 GB for Section 4). Security group: SSH (22) from your IP only.
3. `ssh ubuntu@<ip>`, then `git clone https://github.com/zinastack/ncp-genl-labs.git && cd ncp-genl-labs && LAB_SECTION=3 bash brev/setup.sh`.
4. Reach Grafana and Prometheus through an SSH tunnel: `ssh -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<ip>`.
5. **Stop or terminate** the instance afterwards. Spot instances cut the price by about 60–70% for these interruptible labs.

## Optional: NVIDIA hosted NIMs

The prompting and guardrails labs can use NVIDIA-hosted NIM endpoints instead of the local vLLM
server. Get a free key at https://build.nvidia.com and `export NVIDIA_API_KEY=nvapi-...`.
`common/llm.py` switches automatically.
