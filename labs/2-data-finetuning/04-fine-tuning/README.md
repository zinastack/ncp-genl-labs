# Lab 04 — Fine-Tuning (13% of exam)

> Blueprint: *domain customization, parameter-efficient techniques, task-specific adaptation.*

You will implement **LoRA from scratch** in PyTorch, inject it into a model the way PEFT and NeMo
do, write the SFT data collator (prompt-masked labels) and the causal-LM loss, implement the
**DPO** preference loss, and run a real (tiny) LoRA fine-tune to confirm the base weights never move.

```
make test-04          # YOUR exercises (run from the repo root)
make solutions-04     # reference solutions
make quiz-04          # exam-style questions
make gpu-04           # GPU part (gpu_lab.py) on the Brev/AWS instance
```

---

## 1. The customisation ladder

| Stage | What changes | Data | Typical use |
|---|---|---|---|
| **Continued / domain-adaptive pre-training (DAPT)** | All weights, next-token loss on raw domain text | Billions of unlabelled domain tokens | New domain vocabulary and knowledge (for example NVIDIA ChipNeMo on chip-design docs) |
| **Supervised fine-tuning (SFT) / instruction tuning** | All weights or PEFT, loss on responses only | 1k–100k prompt→response pairs | Follow instructions, task formats, style |
| **Preference alignment** | RLHF (reward model + PPO), **DPO**, RLAIF, SteerLM | Chosen/rejected pairs or ratings | Helpfulness, harmlessness, tone |
| **Task-specific heads** | Classification or regression head (+ optional PEFT) | Labelled examples | Fast, cheap classifiers on encoders |

NVIDIA stack: **NeMo Framework** (Megatron-based training: SFT, PEFT, DPO/RLHF via NeMo-RL /
NeMo-Aligner), **NeMo Customizer** (a microservice API for LoRA/SFT jobs), and **NIM** to serve
the base model plus LoRA adapters.

## 2. Parameter-efficient fine-tuning (PEFT)

| Method | Idea | Trainable params | Inference overhead |
|---|---|---|---|
| **LoRA** | `W' = W + (α/r)·B·A`, with `A ∈ ℝ^{r×d_in}` random and `B ∈ ℝ^{d_out×r}` **zero-initialised** so training starts from the base model | `r·(d_in+d_out)` per adapted matrix (≈0.1–1%) | **None after merging** `W + ΔW` |
| **QLoRA** | Frozen base quantised to **4-bit NF4** + double quantisation + paged optimisers, with LoRA adapters in bf16 | same as LoRA | dequantisation cost unless you merge |
| **DoRA** | Splits the weight into magnitude and direction and applies LoRA to the direction | ≈LoRA | none after merging |
| **Adapters (Houlsby)** | Small bottleneck MLPs inserted after sub-layers | ~1–3% | extra layers, adding latency |
| **IA³** | Learned vectors rescaling K, V and FFN activations | ~0.01% | negligible |
| **Prefix tuning** | Learned K/V vectors prepended at every layer | small | uses context slots |
| **Prompt / p-tuning** | Learned soft prompt embeddings at the input (p-tuning uses an LSTM/MLP prompt encoder) | tiny | uses context slots |

**LoRA hyperparameters:** rank `r` 8–64 (higher for harder or more different tasks), `α` often
`= r` or `2r` (the scale is `α/r`), dropout 0.05–0.1, **lr ≈ 1e-4 to 2e-4** (10× a full-FT lr),
target modules `q_proj, k_proj, v_proj, o_proj` and, for more capacity, the MLP
`gate_proj, up_proj, down_proj`.

**Why PEFT:** a tiny optimizer state (Adam states only for adapter params), so memory drops
dramatically. Adapters are MBs, so you can keep **many adapters for one base model**
(multi-LoRA serving in NIM / TensorRT-LLM). PEFT also causes **less catastrophic forgetting**.

## 3. SFT details that matter

- **Loss masking:** set labels to `-100` (`ignore_index`) on prompt/system tokens so the model is
  trained only to produce responses. Otherwise it learns to generate user prompts.
- **Causal shift:** logits at position *t* predict token *t+1*: `loss = CE(logits[:, :-1], labels[:, 1:])`.
- **Chat template:** train with the **same template** you'll use at inference.
- **Epochs:** 1–3. More overfits and increases forgetting. Watch validation loss plus a general benchmark.
- **Learning-rate schedule:** warm-up then cosine decay. Full-FT lr ≈ 1e-5 to 2e-5.

## 4. Preference alignment

- **RLHF:** SFT model → train a **reward model** on human preference pairs → optimise the policy
  with **PPO** against the reward, with a KL penalty to the reference model. Complex: four models in memory.
- **DPO:** skips the reward model and RL. The loss directly increases the margin of chosen over
  rejected log-probs relative to a frozen reference:

  `L = −log σ( β·[(log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x))] )`

  `β` (0.1–0.5) controls how far the policy may drift from the reference.
- **SteerLM (NVIDIA):** condition generation on attribute labels (helpfulness, humour and so on)
  so they can be adjusted at inference time.

## 5. Exam traps

- LoRA `B` is initialised to **zero**, so step 0 equals the base model exactly.
- Merged LoRA has **no extra latency**. Unmerged adapters add a small matmul, but let you swap adapters per request.
- QLoRA reduces **memory**, not FLOPs. It is usually a bit *slower* per step than LoRA in bf16.
- "Domain vocabulary and knowledge" calls for continued pre-training or RAG. "Behaviour, format and style" calls for SFT or PEFT. "Preferences and safety" calls for DPO or RLHF.
- Few examples (tens): use prompting. Thousands: PEFT. Very large and different: full FT or DAPT.
- Classification with an encoder (BERT) plus a linear head is often cheaper and better than generative SFT for fixed labels.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function / class | Concept |
|---|---|---|
| 1 | `LoRALinear` (forward + `merge`) | low-rank update, zero-init B, α/r scaling |
| 2 | `apply_lora` | inject adapters by module name, freeze the rest |
| 3 | `count_parameters` | trainable vs total |
| 4 | `build_sft_example` | prompt-masked labels |
| 5 | `causal_lm_loss` | shifted cross-entropy with ignore_index |
| 6 | `sequence_logprob` | per-sequence log-likelihood |
| 7 | `dpo_loss` | direct preference optimisation |
| 8 | `train` | a minimal optimisation loop over trainable params only |
| 9 | `nf4_quantize`, `nf4_dequantize` | QLoRA's 4-bit NormalFloat format, block-wise scales |
| 10 | `SoftPrompt` | prompt tuning / p-tuning: trainable virtual tokens, frozen model |
| 11 | `reward_model_loss`, `preference_accuracy` | the RLHF reward model (Bradley–Terry) |
| 12 | `MultiLoRALinear` | multi-LoRA serving: one base model, per-request adapters |
| 13 | `lr_at_step` | warm-up + cosine learning-rate schedule |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).
