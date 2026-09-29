# Lab 05 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link. This is the heaviest exam domain (17%), so the "why" matters
as much as the code.

The picture: every optimisation moves one of three numbers.

```
            MEMORY                      COMPUTE                      BANDWIDTH
   weights, grads, optimizer,      matmul FLOPs (prefill,        bytes read per step
   activations, KV cache           training)                      (decode = memory-bound)
   ex. 1, 2, 3, 4, 5               ex. 6, 7 (smaller models)       ex. 8, 9, 10
```

---

## Exercise 1 — `training_memory_bytes`: the 16 bytes/parameter rule

Mixed-precision training with Adam keeps, **per parameter**:

| what | bytes | why it exists |
|---|---|---|
| bf16 weights | 2 | used in forward/backward |
| bf16 gradients | 2 | result of backward |
| fp32 master weights | 4 | small updates vanish in bf16, so the optimizer updates an fp32 copy |
| Adam momentum `m` | 4 | running average of gradients |
| Adam variance `v` | 4 | running average of squared gradients |
| **total** | **16** | 7B model → **112 GB**, before activations |

The function splits this into **every** parameter's stored weight (`weight_bytes`) plus **14 bytes
per trainable** parameter (grads + master + m + v):

| setup | formula | result |
|---|---|---|
| full FT, 7B | 7e9 × 2 + 7e9 × 14 | 112 GB |
| LoRA, 7B, 20M trainable | 7e9 × 2 + 20e6 × 14 | 14.28 GB |
| QLoRA, 70B, 100M trainable, 4-bit | 70e9 × 0.5 + 100e6 × 14 | 36.4 GB |

That's why a 70B fine-tune fits on one 48 GB GPU with QLoRA and needs a cluster otherwise.
Activations come on top, and activation checkpointing (README) is their fix.

---

## Exercise 2 — `max_batch_size`: how many users fit on a GPU

At inference, memory = weights + **KV cache per sequence** × sequences + workspace.

```
80 GB GPU, keep 10% for activations → 72 GB usable
Llama-2-7B fp16 weights: 13.5 GB     → 58.5 GB left
KV per 4k sequence: 2 GiB (Lab 01)   → floor(58.5e9 / 2.147e9) = 27 concurrent sequences
```

`max(0, …)` handles "the weights don't even fit" (a 16 GB GPU gives 0). This is the calculation
behind GQA, FP8 KV cache and paged attention: each one raises that 27.

---

## Exercise 3 — `quantize_int8` / `dequantize`: fewer bytes per weight

Map floats to integers in `[-127, 127]` with a **scale**:

```
scale = max|w| / 127        q = round(w / scale)        w ≈ q · scale
```

The problem is **outliers**. Take a weight matrix with one big value:

```
w = [[ 0.1, -0.5, 0.3],
     [40.0, -2.0, 1.0]]
```

| | scale(s) | row 0 as int8 | row 0 recovered |
|---|---|---|---|
| **per-tensor** | one scale: 40/127 = 0.315 | `[0, -2, 1]` | `[0.0, -0.63, 0.31]` (0.1 became 0) |
| **per-channel** (per row) | 0.5/127 = 0.0039 and 40/127 = 0.315 | `[25, -127, 76]` | `[0.098, -0.50, 0.299]` ✓ |

With one scale, the step size is 0.315, so small weights fall between steps and round to 0.
Per-row scales let each row use its full 255 levels. The test builds a matrix with one outlier
row and requires per-channel error to be **10× smaller**. On real Qwen weights (GPU lab), per-channel
wins as well.

Code notes: `np.maximum(absmax, 1e-12)` avoids dividing by zero for an all-zero row, `np.clip`
keeps rounding inside the int8 range, and `keepdims=True` gives shape `(rows, 1)` so each row divides by its own scale.

**Exam link:** per-channel or per-group (e.g. 128) scales; SmoothQuant for *activation* outliers;
weight-only INT4 (AWQ/GPTQ) speeds up memory-bound decode.

---

## Exercise 4 — `DynamicLossScaler`: making FP16 training work

FP16 has a **small range**: the largest value is 65,504, and very small values round to 0. Real
gradients are often tiny: `float16(1e-8) = 0.0`, so the gradient is gone. **Loss scaling** multiplies
the loss (and so every gradient) by a big factor before backward: `1e-8 × 1024 = 1.02e-5`, which fits.
The optimizer divides it back out before updating.

The scale must be as big as possible without overflowing to `inf`, and nobody knows that value in
advance, so it **adapts** (scale 1024, growth every 3 good steps):

| step | overflow? | action | scale after |
|---|---|---|---|
| 1 | yes | skip the update, halve | 512 |
| 2 | no | good step 1 | 512 |
| 3 | no | good step 2 | 512 |
| 4 | no | good step 3 → double, reset counter | 1024 |
| 5 | no | good step 1 | 1024 |
| 6 | yes | skip, halve, reset counter | 512 |

- `update()` returns **False on overflow**, meaning skip `optimizer.step()`. A gradient containing `inf` would destroy the weights.
- Reset the counter on overflow so growth needs a fresh run of clean steps.

> **Exam link:** **BF16 has FP32's exponent range**, so loss scaling isn't needed. That's the main
> reason A100/H100 training uses BF16.

---

## Exercise 5 — `accumulated_gradients`: a big batch that doesn't fit

If batch 32 doesn't fit but batch 8 does, run 4 micro-batches and **add up their gradients**
before one optimizer step:

```python
model.zero_grad()                          # once, at the start
for xb, yb in chunks:                      # 4 micro-batches
    loss = mse(model(xb), yb) / len(chunks)   # ÷4 so the SUM equals the full-batch MEAN
    loss.backward()                        # PyTorch ADDS into .grad, it doesn't overwrite
```

- `backward()` **accumulates** into `.grad`. Zeroing between micro-batches would keep only the last one.
- Dividing by the number of chunks makes the summed gradient equal the gradient of the mean loss
  over all 32 examples. The test checks this to floating-point precision.
- Global batch = micro-batch × accumulation steps × data-parallel GPUs. It saves **memory, not time**.

---

## Exercise 6 — `distillation_loss`: a small student copies a big teacher

The teacher's full probability distribution carries more information than the single correct label.
For an image of a "7" the teacher might say 90% "7", 8% "1", 2% "9", which teaches that 7 looks like 1.

```
L = α · T² · KL( softmax(teacher/T) ‖ softmax(student/T) )  +  (1 − α) · CE(student, label)
```

**Temperature T softens the teacher**, the same softmax temperature as Lab 02. With teacher logits `[3, 1, -1]`:

| T | teacher probabilities |
|---|---|
| 1 | `[0.867, 0.117, 0.016]` (almost one-hot) |
| 2 | `[0.665, 0.245, 0.090]` |
| 4 | `[0.506, 0.307, 0.186]` (the "dark knowledge" is visible) |

- **Why T²?** Softening by T shrinks the gradients by about 1/T², so multiplying by T² keeps the
  KD term balanced against the hard-label term.
- `F.kl_div(log_student, teacher_probs, reduction="batchmean")` expects the student as
  **log-probabilities** and the teacher as **probabilities**. That argument order trips many people up.
- Identical logits give zero KD loss (the test checks this).

NVIDIA **Minitron** = prune a big model + distil from the original, which gives good small models cheaply.

---

## Exercise 7 — `prune_2_4`: sparsity the GPU can use

Randomly zeroing weights rarely speeds anything up on a GPU. **2:4 sparsity** is a fixed pattern
(in every group of 4 weights, exactly 2 are zero) that Ampere, Hopper and later **sparse tensor
cores** accelerate, up to 2× on matmuls.

```
[0.1, -0.9, 0.3, 0.05 | 2.0, -3.0, 0.0, 1.0]
 keep the 2 largest |w| in each group of 4:
[0,   -0.9, 0.3, 0    | 2.0, -3.0, 0,   0  ]
```

Code: reshape to `(-1, 4)` groups, `argsort(-|w|)[:, :2]` picks the two largest per group,
`put_along_axis` builds the keep-mask, and then reshape back. Exactly 50% zeros. Accuracy drops, so
production **fine-tunes after pruning** (the GPU lab shows the raw error).

---

## Exercise 8 — static vs in-flight batching

Eight requests with output lengths `[10, 200, 12, 15, 180, 9, 11, 14]` (451 tokens of real work), 4 GPU slots:

**Static batching:** requests are grouped in fours, and each group holds the GPU until its
*longest* member finishes.
```
batch 1: [10, 200, 12, 15] → 200 steps     (three slots idle for ~190 steps)
batch 2: [180, 9, 11, 14]  → 180 steps
total 380 steps
```

**In-flight (continuous) batching:** when any request finishes, the next one takes its slot.
```
slots start with 10, 200, 12, 15
t=10: slot 1 free → the 180-token request runs until 190
t=12: → the 9-token request until 21 · t=15: → 11 until 26 · t=21: → 14 until 35
last finish: 200 steps
```

Code: a **min-heap** of slot finish times. `heappop` gives the slot that frees first, the request
starts then, and its finish time is pushed back. Total time = the latest finish. That is the
scheduler inside TensorRT-LLM, vLLM and NIM, and it raises throughput and GPU utilisation (not
single-request latency).

---

## Exercise 9 — `speculative_expected_tokens`: guessing ahead

The big model is memory-bound at decode (next exercise), so checking 5 tokens costs about the
same as generating 1. A small **draft model** guesses γ tokens, and the big model verifies them
all in **one** pass, keeping the correct prefix plus one token of its own.

If each guess is accepted independently with probability α, the expected tokens per big-model pass
are a geometric series:

```
1 + α + α² + … + α^γ = (1 − α^(γ+1)) / (1 − α)
```

| α \ γ | 2 | 4 | 8 |
|---|---|---|---|
| 0.5 | 1.75 | 1.94 | 2.00 |
| 0.7 | 2.19 | 2.77 | 3.20 |
| 0.8 | 2.44 | 3.36 | 4.33 |
| 0.9 | 2.71 | 4.10 | 6.13 |

- α = 0 still yields **1** token (the big model's own correction), so it's never slower in tokens per pass.
- A longer draft helps only when α is high: at α = 0.5, going from 4 to 8 guesses adds almost nothing.
- The output is **identical** to the big model's (the GPU lab checks this). The acceptance rule guarantees it.
- `a == 1` needs a special case, because the formula would divide by zero.

---

## Exercise 10 — `decode_tokens_per_sec_bound`: the decode roofline

Each decode step reads **every weight once** to produce one token per sequence. At batch 1 the GPU
waits on memory, not math:

```
tokens/s ≤ memory_bandwidth / bytes_of_weights × batch
```

| model | GPU | bound |
|---|---|---|
| 70B fp16 (140 GB) | H100, 3.35 TB/s | ~24 tokens/s |
| 8B fp16 (16 GB) | L4, 300 GB/s | ~19 tokens/s |
| 8B INT4 (4 GB) | L4, 300 GB/s | ~75 tokens/s (4× fewer bytes → 4× faster) |

This one line explains most of inference optimisation:
- **batching:** one weight read serves many sequences, so throughput scales with batch (until compute or KV memory runs out);
- **weight quantization:** fewer bytes per step;
- **speculative decoding:** more tokens per weight read.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| memory for full fine-tuning | ~16 B/param (+ activations) |
| OOM from activations at long sequence | activation checkpointing (compute ↔ memory) |
| FP16 inf/NaN, scale keeps dropping | dynamic loss scaling; switch to BF16 |
| target batch doesn't fit | gradient accumulation (memory, not speed) |
| accuracy loss after INT8 PTQ | per-channel/group scales; SmoothQuant; QAT if still bad |
| variable output lengths waste GPU | in-flight (continuous) batching |
| long prompts slow to first token | prefill: prefix caching, chunked prefill, FP8 |
| slow token generation at batch 1 | memory-bound: quantize weights, speculative decoding, batch |
| KV cache limits concurrency | FP8 KV cache, paged KV, GQA |
| Ampere sparse speed-up | 2:4 structured sparsity |
