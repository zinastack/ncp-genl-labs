# NCP-GENL Labs: Launchable settings

Five Launchables, one per section, all built from **https://github.com/zinastack/ncp-genl-labs**.
Create them in the Brev console (**Launchables → Create Launchable**); the Brev CLI can deploy
Launchables but can't create them.

## Settings shared by all five

| Field | Value |
|---|---|
| **Code** | Git repository → `https://github.com/zinastack/ncp-genl-labs` |
| **Runtime** | **VM Mode** |
| **Setup script** | paste [`brev/launchable-setup.sh`](launchable-setup.sh) (a short bootstrap that runs `brev/setup.sh` from the repo). The console requires the first line to be exactly `#!/bin/bash` |
| **Launch parameter** | `LAB_SECTION` = the section number below |
| **Optional parameter** | `HF_TOKEN`: left empty; each deployer can add their own (all lab models are public) |
| **Jupyter** | enabled |
| **Disk** | 100 GiB (**300 GiB for S4**: Triton, TensorRT, TensorRT-LLM, vLLM, NIM and SDK images) |
| **Overview page** | Brev renders the repo's root `README.md` (the *Launched from Brev? Start here* section), so keep that section current |
| **Visibility** | **Only my organization** while you test; switch to *Anyone with the link* or *Everyone (published)* when you open them. New Launchables default to *Anyone with the link*, so change it at creation. |

## The five Launchables

| # | Name | GPU (first choice → fallback) | `LAB_SECTION` | Secure links | ~$/h |
|---|---|---|---|---|---|
| 1 | **NCP-GENL Labs · S1 Foundations & Prompting** | 1× L4 24 GB → 1× T4 | `1` | Jupyter | 0.85 |
| 2 | **NCP-GENL Labs · S2 Data & Fine-Tuning** | 1× L4 24 GB → 1× T4 | `2` | Jupyter | 0.85 |
| 3 | **NCP-GENL Labs · S3 Optimization & Acceleration** | **2× L4** → 2× T4 (Lab 06 needs 2 GPUs) | `3` | Jupyter | 2.39 / 0.98 |
| 4 | **NCP-GENL Labs · S4 Deployment & Monitoring** | 1× L4 24 GB, 8 vCPUs | `4` | Jupyter, `grafana` → 3000, `prometheus` → 9090 | 1.02 |
| 5 | **NCP-GENL Labs · S5 Evaluation & Responsible AI** | 1× L4 24 GB → 1× T4 | `5` | Jupyter | 0.85 |

Prices are Brev's listing at the time of writing (`make brev-plan S=N` shows current options).

## Descriptions (paste into each Launchable)

1. **S1 Foundations & Prompting**: Transformer internals and prompt engineering for the NVIDIA
   NCP-GENL exam. Build attention, RoPE, GQA, FlashAttention and MoE from scratch; drive a real model
   with your own sampling, constrained decoding, CoT/self-consistency, RAG and ReAct code. 1× L4.
2. **S2 Data & Fine-Tuning**: Curate data like NeMo Curator (dedup, MinHash-LSH, quality filters,
   PII, decontamination, BPE, packing) and fine-tune with LoRA, QLoRA (NF4), soft prompts, DPO and
   reward models; measure full FT vs LoRA vs QLoRA memory on a real model. 1× L4.
3. **S3 Optimization & Acceleration**: Memory math, quantization (INT8, NF4, SmoothQuant, FP8),
   paged KV cache, speculative decoding, activation checkpointing; then real multi-GPU NCCL, DDP vs
   FSDP, ring attention and Nsight Systems profiling. 2 GPUs.
4. **S4 Deployment & Monitoring**: Hands-on serving: tune Triton dynamic batching and instance
   groups with perf_analyzer, build TensorRT engines, chain an ensemble, run Model Analyzer, tune
   TensorRT-LLM (batch, KV cache, INT8) and a NIM; then Kubernetes on the same GPU (device plugin,
   probes, time-slicing, HPA on queue time, rollouts, canary) and Prometheus + Grafana + DCGM alerts,
   failure drills and drift. 1× L4.
5. **S5 Evaluation & Responsible AI**: Evaluation metrics, bootstrap CIs, debiased LLM-as-judge,
   lm-evaluation-harness; fairness audits, NeMo Guardrails, retrieval and topic rails, red-teaming and
   EU AI Act basics. 1× L4.

## After creating each one

1. Deploy it once (from the console, or `make brev-launch S=N LAUNCHABLE=env-…`).
2. On the instance: `make doctor`, then the section's GPU targets (`brev/README.md` lists them).
3. Stop or delete the test instance.
4. When everything works, change the visibility to share it.
