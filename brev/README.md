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
| **4** Deployment & Monitoring | 07, 08 | 1× **T4** 16 GB | 1× T4 | Triton + DCGM + Prometheus + Grafana in Docker, with a small model |
| **5** Evaluation & Responsible AI | 09, 10 | 1× **L4** 24 GB | 1× T4 16 GB | vLLM server + evaluation models + guardrails together |

Rough cost: T4 instances are typically **$0.35–0.60/h** and L4 **$0.70–1.00/h** (2× L4 ≈ 2×), so a
2–3 hour session per section costs about **$1–6**. Prices vary by provider and region; check
Brev's instance picker. **Stop the instance when you're done.** Disk: 100 GB (models, Docker images).

> T4 notes: no bf16 (the labs switch to fp16 automatically), no FlashAttention-2, no FP8.
> Everything still runs, and the differences are themselves exam material.

## Create the Launchables (one per section)

Push this repo to GitHub first (public, or grant Brev access), then set `REPO_URL` in
`brev/setup.sh` to its URL. In the Brev console, **Launchables → Create Launchable**:

| Field | Value |
|---|---|
| **Code** | Git repository → `https://github.com/<you>/genl-labs` (Brev clones it to `/home/ubuntu/genl-labs`) |
| **Runtime** | **VM Mode** (Ubuntu 22.04 + Docker + NVIDIA driver + CUDA) |
| **Setup script** | paste the contents of `brev/setup.sh` |
| **Launch parameter** | `LAB_SECTION` = `1`…`5` (or `all`) |
| **Jupyter** | enabled (setup generates `gpu_lab.ipynb` next to every `gpu_lab.py`) |
| **Secure links** | Section 4 only: `grafana` → port **3000**, `prometheus` → port **9090** |
| **GPU** | from the table above |
| **Disk** | 100 GiB |
| **Name** | e.g. `NCP-GENL · Section 3 · Optimization & Acceleration (2×L4)` |

Create five Launchables that differ only in `LAB_SECTION`, GPU and name, and share their links.
Each one is a one-click, reproducible lab environment.

## Using an instance

```bash
brev shell <instance-name>          # or open Jupyter from the Brev console
cd ~/genl-labs
make doctor                         # GPU, driver, torch, docker
make s3                             # list Section 3's targets
make gpu-05                         # run Lab 05's GPU part
make s3-gpu-06                      # multi-GPU: NCCL check, all-reduce bandwidth, DDP scaling, DDP vs FSDP
make s3-profile-06                  # Nsight Systems + PyTorch profiler traces
make s4-triton-up && make gpu-07    # Triton + perf_analyzer
make s5-llm-up && make s5-guardrails
```

Your exercises are the same files as on your laptop. Commit and push from the laptop, then
`git pull` on the instance. Run the GPU parts with **your** code via
`USE_EXERCISES=1 make gpu-02`.

Port forwarding without secure links: `brev port-forward <instance> -p 3000:3000`.

## AWS instead of Brev

| Section | Instance | GPU | Approx. on-demand (us-east-1) |
|---|---|---|---|
| 1, 2, 5 | `g6.xlarge` | 1× L4 24 GB | ~$0.80/h |
| 1, 2, 4, 5 (budget) | `g4dn.xlarge` | 1× T4 16 GB | ~$0.53/h |
| 3 | `g6.12xlarge` / `g4dn.12xlarge` | 4× L4 / 4× T4 | ~$4.60 / ~$3.90 per h |

1. AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** (driver, Docker and the NVIDIA Container Toolkit preinstalled).
2. 100 GB gp3 root volume. Security group: SSH (22) from your IP only.
3. `ssh ubuntu@<ip>`, then `git clone <repo> genl-labs && cd genl-labs && LAB_SECTION=3 bash brev/setup.sh`.
4. Reach Grafana and Prometheus through an SSH tunnel: `ssh -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<ip>`.
5. **Stop or terminate** the instance afterwards. Spot instances cut the price by about 60–70% for these interruptible labs.

## Optional: NVIDIA hosted NIMs

The prompting and guardrails labs can use NVIDIA-hosted NIM endpoints instead of the local vLLM
server. Get a free key at https://build.nvidia.com and `export NVIDIA_API_KEY=nvapi-...`.
`common/llm.py` switches automatically.
