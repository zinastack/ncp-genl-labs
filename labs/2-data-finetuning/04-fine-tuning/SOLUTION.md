# Lab 04 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: a pre-trained LLM already knows language. **Fine-tuning** nudges it toward a task,
style, domain or set of preferences. The obstacle is size: updating all 7 billion weights needs
gradients and optimizer state for every one of them (16 bytes per parameter, Lab 05). The whole
domain is about **changing behaviour while training as little as possible**.

```
                 frozen W (billions of weights)
x ──► ──────────────────────────────► (+) ──► y          y = W·x + (α/r)·B·A·x
  └─► A (r × d_in) ─► B (d_out × r) ──┘                 only A and B train (LoRA, ex. 1–3)
```

---

## First: choosing a fine-tuning method

| Situation | Method | Why |
|---|---|---|
| Tens of examples, no training allowed | prompting (Lab 02) | cheapest; no weights change |
| Thousands of prompt→response pairs; style, format, task behaviour | **SFT with LoRA** (ex. 1–5) | ~0.1–1% of parameters train; little forgetting |
| Same, but the model barely fits the GPU | **QLoRA** (ex. 9) | 4-bit frozen base plus LoRA adapters |
| Very few trainable parameters, or many tasks on one frozen model | **prompt tuning / p-tuning** (ex. 10) | only a few virtual-token embeddings train |
| New domain vocabulary and knowledge (billions of tokens) | **continued pre-training (DAPT)**, then SFT | knowledge comes from lots of raw text (NVIDIA ChipNeMo) |
| Human preferences, tone, safety | **DPO** (ex. 7) or **RLHF** (reward model, ex. 11, + PPO) | learns from chosen vs rejected answers |
| Fixed label set, lots of labels, low latency | **encoder + classification head** (e.g. DeBERTa) | far cheaper than a generative LLM |
| Many customers, one base model | **one LoRA per customer + multi-LoRA serving** (ex. 12) | adapters are megabytes and share the base |

Knowledge that **changes weekly** doesn't belong in weights at all: use RAG (Lab 02).

---

## Exercise 1 — `LoRALinear`: the low-rank update

### Why it exists

Full fine-tuning updates every weight, which is expensive (gradients plus Adam state for billions
of parameters), produces a full model copy per task, and can make the model **forget** general
skills. Research found that the weight *change* needed for adaptation has **low rank**: it can be
written as the product of two thin matrices. LoRA trains only those.

### How it works

| matrix | full ΔW | LoRA (r) | trainable share |
|---|---|---|---|
| 4096 × 4096, r = 8 | 16,777,216 | 65,536 | 0.39% |
| 4096 × 4096, r = 16 | 16,777,216 | 131,072 | 0.78% |
| 4096 × 11008 (MLP), r = 16 | 45,088,768 | 241,664 | 0.54% |

LoRA parameters per matrix = `r · (d_in + d_out)`.

```python
p.requires_grad = False                         # the base weight never changes
self.lora_A = nn.Parameter(torch.empty(r, in))  # random (kaiming)
self.lora_B = nn.Parameter(torch.zeros(out, r)) # ZERO
self.scaling = alpha / r
forward: base(x) + (dropout(x) @ Aᵀ @ Bᵀ) * scaling
```

- **B = 0** makes `B·A = 0`, so step 0 is **exactly** the pre-trained model. A is random so the gradient for B isn't zero (if both were zero, nothing would learn).
- **α/r** keeps the update's size stable when you change r, so r = 8 vs 32 needs no re-tuning of the learning rate.
- `x @ Aᵀ @ Bᵀ` never builds the full d_out × d_in matrix.
- **`merge()`** folds the update in: `W' = W + (α/r)·B·A`, a plain Linear with **no extra latency**.

### On the exam

*Why is B initialised to zero and A randomly?*

| option | verdict |
|---|---|
| So BA = 0 at the start (the model equals the pre-trained one) while gradients can still flow to B | ✅ |
| To make adapter weights sparse for faster inference | ❌ nothing is sparse; B fills in during training |
| Because A and B are merged before training | ❌ merging happens after training |
| To stop the base weights from updating | ❌ that's `requires_grad = False`, a separate mechanism |

---

## Exercise 2 — `apply_lora`: inject adapters by name

### Why it exists

Real models are deep module trees, and you choose *which* layers get adapters. In Llama the
attention projections are `q_proj, k_proj, v_proj, o_proj` and the MLP has `gate_proj, up_proj, down_proj`.
PEFT's `target_modules` selects layers **by attribute name**, and this exercise is that mechanism.

### How it works

Walk `named_modules()` (snapshot with `list(...)` before modifying the tree), swap matching
`nn.Linear` children with `setattr(parent, name, LoRALinear(child))`, then freeze everything whose name doesn't contain `lora_`, including heads and norms.

### On the exam

*LoRA with r = 8 on attention only under-fits a hard task. Increase capacity (Select TWO):*
**raise r** (32–64, adjusting α) and **also target the MLP matrices** (they hold ⅔ of the parameters, Lab 01).
Wrong: lowering r, freezing A, switching to 5-token prompt tuning (far less capacity).

---

## Exercise 3 — `count_parameters`: what you actually train

`sum(p.numel() for p in params if p.requires_grad)`. In the test model, LoRA on q, v (32→32) and
down (64→32) with r = 4 gives `4·64·2 + 4·96 = 896` trainable of 8,388 total. On a real 7B model,
q and v in all 32 layers at r = 16 is ~8.4M, **0.12%**.

### On the exam

*r = 16 on q_proj and v_proj (4096 × 4096) in 32 layers:* 64 matrices × 16 × 8,192 ≈ **8.4 M**.
Wrong answers usually forget one of the two matrices (4.2M) or multiply instead of adding the dimensions.

---

## Exercise 4 — `build_sft_example`: learn the answer, not the question

### Why it exists

SFT data is (prompt, response) pairs. The model should learn to **produce responses**, not to
write user prompts, so the loss must ignore prompt positions.

```
input_ids = [5, 6, 7,    8, 9, 2]         prompt + response + EOS
labels    = [-100,-100,-100, 8, 9, 2]     -100 = ignore_index: no loss here
```

**EOS stays in the labels**, so the model learns when to **stop**.

### On the exam

*After SFT the model sometimes writes the next user question instead of answering.* The loss
included **prompt tokens** (no -100 masking). Masking EOS causes the opposite symptom, a model that
never stops. A low learning rate under-fits but doesn't teach the model to write user turns.

---

## Exercise 5 — `causal_lm_loss`: the shift

### Why it exists

A causal LM at position *t* predicts token *t+1*, so logits and labels must be offset by one.

```python
F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
```

When logits favour the right token by 2 points, `p = e²/(e²+2) = 0.787`, costing `−ln 0.787 = 0.2395` per position:

| labels | mean loss |
|---|---|
| `[-100, 1, 2, 0]` | 0.2395 |
| `[-100, -100, 2, 0]` | 0.2395 (masked positions just don't count) |
| `[-100, 2, 2, 0]` | 0.9062 (one position disagrees and costs 2.24) |

Hugging Face does this shift internally when you pass `labels=`.

---

## Exercise 6 — `sequence_logprob`: how likely is a whole answer?

### Why it exists

Preference methods (DPO, ex. 7) compare how likely the model finds **whole** answers, so we need
`log p(response | prompt)` = the sum of token log-probabilities over the response.

`gather` picks log p(actual next token) at each position; `clamp(min=0)` makes -100 a valid index,
and the mask zeroes it. Example above: 3 × ln 0.787 = **−0.7186**; masking the first position gives −0.4791.

---

## Exercise 7 — `dpo_loss`: preferences without RL

### Why it exists

Classic **RLHF**: train a reward model on human preferences (ex. 11), then optimise the policy with
PPO against it, with a KL penalty. That keeps **four models** in memory (policy, reference, reward,
value) and is notoriously finicky. **DPO** reaches the same objective with **one classification-style loss** on preference pairs:

```
L = −log σ( β · [ (log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l)) ] )
```

At the start policy = reference, the margin is 0, and the loss is ln 2 = 0.693. **β** controls how far the policy may drift:

| margin | β = 0.1 | β = 0.5 |
|---|---|---|
| 0 | 0.693 | 0.693 |
| 4 | 0.513 | 0.127 |
| 8 | 0.371 | 0.018 |

`F.logsigmoid` is used because `log(sigmoid(z))` can hit `log(0)`.

### On the exam

*Main advantage of DPO over PPO-based RLHF?* It **optimises directly from preference pairs with a
simple loss: no separate reward model and no RL sampling loop.** Wrong: "doesn't need preference
data", "updates only embeddings", "removes the reference model" (DPO still uses a frozen reference).

---

## Exercise 8 — `train`: only the trainable parameters

`AdamW([p for p in model.parameters() if p.requires_grad])`. Adam keeps two extra fp32 numbers
per managed parameter, so **passing only adapters is where LoRA's memory saving comes from**. The test
checks every original weight is **bit-for-bit unchanged** after LoRA learns a new task.

---

## Exercise 9 — `nf4_quantize`: what "4-bit" in QLoRA means

### Why it exists

Even with LoRA, the **frozen base weights** must sit in GPU memory: 65B × 2 bytes = 130 GB in
bf16. **QLoRA** stores the frozen base in **4-bit NF4** and trains LoRA adapters in bf16, dequantizing
weights on the fly. A 65B fine-tune then fits on a single 48 GB GPU.

### How it works

1. Split the weights into **blocks of 64** and store each block's absmax (so one outlier only affects its own block).
2. Divide by absmax, so values are in [−1, 1].
3. Replace each value with the **index of the nearest of 16 NF4 levels**: 4 bits.

Why NF4 and not 16 evenly spaced levels? Trained weights are roughly **normally distributed**, with
most values near zero. NF4's levels are **normal-distribution quantiles**: dense near 0, sparse at
the tails. Measured on normally distributed weights: **NF4's error is about 17% lower** than uniform 4-bit at the same size.

**Memory:** 4 bits + a 32-bit absmax per 64 weights = 4.5 bits per weight. QLoRA's **double
quantization** also quantizes the absmax values (to 8 bits), giving about **4.13 bits per weight**:
a 65B model's weights take about 33.5 GB. **Paged optimizers** spill optimizer state to CPU memory during spikes.

### On the exam

*Fine-tune a 65B model on one 48 GB GPU:* **QLoRA.** Full fine-tuning in fp16 needs over 1 TB,
LoRA with an fp32 base needs 260 GB for weights alone, and 100 few-shot examples isn't fine-tuning.
Trade-off: QLoRA saves **memory, not time**, and dequantization makes each step somewhat slower than bf16 LoRA.

---

## Exercise 10 — `SoftPrompt`: prompt tuning and p-tuning

### Why it exists

Instead of changing *any* weight, learn a few **"virtual token" embeddings** prepended to every
input. The LLM stays completely frozen; only `n_virtual × d_model` numbers train (5 × 16 = 80 in the
test; about 20 × 4,096 = 82k for a 7B model). One frozen model can serve many tasks, each just a tiny prompt.

```
input embeddings (B, T, d) → [ learned prompt (n, d) | inputs ] → (B, n + T, d) → frozen LLM
```

The test checks that gradients reach **only** the soft prompt, never the frozen model.

### On the exam

| method | trains | used by |
|---|---|---|
| **prompt engineering** | nothing: you write text | anyone |
| **prompt tuning** | soft prompt embeddings at the input | Lester et al. |
| **p-tuning** | a small **prompt encoder** (LSTM/MLP) that *generates* the virtual tokens; after training they can be cached and the encoder dropped | NVIDIA NeMo |
| **prefix tuning** | learned key/value vectors at **every layer** | Li & Liang |

All three trained variants are **PEFT**: they need data and a training loop, even though
"prompt" is in the name. They use a few context positions, and have less capacity than LoRA for hard tasks.

---

## Exercise 11 — `reward_model_loss`: the "RM" in RLHF

### Why it exists

Humans can't label "the perfect answer", but they can easily say **which of two answers is better**.
A **reward model** learns a score `r(prompt, answer)` from such pairs, then guides PPO. The
**Bradley–Terry** model says the probability that the chosen answer wins is `σ(r_chosen − r_rejected)`, and we maximise it:

```
L = mean( −log σ(r_chosen − r_rejected) )
pairs (2.0 vs 0.0), (0.5 vs 1.0), (1.0 vs 1.0) → loss 0.598, preference accuracy 1/3 (ties don't count)
```

Only the **difference** matters: the reward scale is arbitrary. **DPO's loss is this same formula**
with the reward replaced by `β · log(π/π_ref)`, which is why DPO needs no separate reward model.

### On the exam

*Classic PPO-RLHF needs (Select TWO):* a **reward model trained on preference comparisons** and a
**frozen reference model for the KL penalty**. Wrong: a new tokenizer, a retrieval index, removing
the value network (PPO needs it; GRPO-style methods drop it). NVIDIA tools: NeMo-RL / NeMo-Aligner
for RLHF and DPO, **SteerLM** for attribute-conditioned alignment, and **Nemotron reward models** for scoring.

---

## Exercise 12 — `MultiLoRALinear`: many adapters, one base model

### Why it exists

40 customers each want their own fine-tuned 8B model. Forty full copies would need 40 × 16 GB of GPU
memory. With **one LoRA adapter per customer**, the 16 GB base is loaded **once** and each adapter
adds only megabytes. **A single batch can mix requests for different customers.**

### How it works

```python
out = base(x)                  # the expensive big matmul runs once for the whole mixed batch
for each row i with adapter a: # the cheap low-rank part is per request
    out[i] += (x[i] @ Aᵀ @ Bᵀ) · scaling
```

The test checks each row equals a standalone LoRA layer with that customer's adapter, and a row
with `None` equals the plain base model. NIM and TensorRT-LLM implement this with batched kernels.

### On the exam

*40 customer-specific variants of one 8B model, minimum GPU memory:* **one LoRA per customer,
multi-LoRA serving with per-request adapter selection.** Wrong: 40 full copies (40× memory),
merging all adapters into one set of weights (their behaviours mix), putting the customer's name in the prompt.
**Merge** (exercise 1) when you serve a **single** adapter and want zero overhead; **keep adapters
separate** when you need to switch per request.

---

## Exercise 13 — `lr_at_step`: warm-up + cosine schedule

### Why it exists

- **Warm-up:** at the start, Adam's moment estimates are based on very few steps and are noisy, and
  (for LoRA) the adapter output starts at exactly 0. Big early steps can destabilise training, so
  the learning rate ramps up linearly.
- **Cosine decay:** large steps make fast progress early; small steps at the end settle into a good minimum.

### How it works

`max_lr 2e-4`, 100 warm-up steps, 1,000 total, floor `2e-5`:

| step | lr |
|---|---|
| 0 | 2e-6 |
| 49 | 1e-4 |
| 99–100 | 2e-4 (peak) |
| 550 | 1.1e-4 (halfway through the decay) |
| 1000+ | 2e-5 (floor) |

### On the exam: typical learning rates

| setup | typical peak lr |
|---|---|
| **LoRA / QLoRA** | **1e-4 to 2e-4** (adapters start at zero and are small) |
| **full fine-tuning** | **1e-5 to 2e-5** (about 10× lower; high rates destroy pre-trained knowledge) |

Fine-tuning epochs: **1–3**. More epochs overfit and increase **catastrophic forgetting**, as do high
learning rates and narrow data. Remedies: PEFT, replaying general data (Lab 03, exercise 14), fewer epochs and a lower lr.

---

## Evaluating a fine-tune (exam favourite)

*What should the evaluation plan include? (Select TWO)*
- **task metrics on a held-out, deduplicated test set** (ROUGE, accuracy, faithfulness, LLM-judge; Lab 09);
- **regression checks on general benchmarks and safety tests**, to catch forgetting and lost alignment.

Training loss, scores on training examples and Wikipedia perplexity don't measure task quality or generalisation.

**Data size:** a small, **curated, diverse** set (LIMA: about 1k examples) can teach format and
style, because knowledge comes from pre-training. Duplicating examples causes memorisation, not learning.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise / section |
|---|---|
| LoRA initialisation, merging | 1 |
| LoRA hyperparameters | 1, 2 |
| LoRA parameter count | 1, 3 |
| Loss masking | 4, 5 |
| DPO vs RLHF | 6, 7, 11 |
| RLHF components | 11 |
| QLoRA | 9 |
| P-tuning | 10 |
| Multi-LoRA serving | 12 |
| Learning rates, catastrophic forgetting | 13 |
| Choosing a method, task-specific adaptation | "Choosing a fine-tuning method" |
| Evaluation after fine-tuning, SFT data size | "Evaluating a fine-tune" |
