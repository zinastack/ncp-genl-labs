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
1× L4 for Sections 1, 2 and 5, 2× L4 then 2× T4 for Section 3, 1× T4 for Section 4. Override with
`TYPE=<brev type>` (see `brev search gpu`) or `INSTANCE=<name>`. `brev create` doesn't take a disk
size. If a type's default disk turns out too small for models and Docker images, pick a type with a
larger disk (`brev search gpu --min-disk 100`).

## Create the Launchables (one per section)

Brev's CLI can **deploy** a Launchable (`make brev-launch S=3 LAUNCHABLE=env-...`) but can't
**create** one: that happens in the web console's Launchable builder. Two things to know:

- **Keep it private:** set visibility to **"Only my organization"**. New Launchables default to
  **"Anyone with the link"**; the third option, "Everyone (published)", lists it publicly.
- **Code source:** Brev's docs describe a **public** Git repository URL as the code source. Until you
  publish the repo, use the CLI path above.

When the repo is on GitHub, set `REPO_URL` in `brev/setup.sh` to its URL. In the Brev console,
**Launchables → Create Launchable**:

| Field | Value |
|---|---|
| **Code** | Git repository → `https://github.com/<you>/genl-labs` (Brev clones it to `/home/ubuntu/genl-labs`) |
| **Runtime** | **VM Mode** (Ubuntu 22.04 + Docker + NVIDIA driver + CUDA) |
| **Setup script** | paste the contents of `brev/setup.sh` |
| **Launch parameter** | `LAB_SECTION` = `1`…`5` (or `all`) |
| **Launch parameter (optional)** | `HF_TOKEN`: each deployer's own Hugging Face token. Not needed (all lab models are public) but avoids download rate limits. Setup saves it to `~/.cache/huggingface/token`. Never bake your own token into the Launchable. |
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
