# Lab 05 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

This is the heaviest exam domain (17%). Nearly every question reduces to: **which resource is the
bottleneck, and which technique moves it?**

```
            MEMORY                          COMPUTE                         MEMORY BANDWIDTH
 weights, grads, optimizer (1),      matmul FLOPs: training,         bytes read per decode step
 activations (5, 11), KV cache       prefill (14)                    (decode is memory-bound, 10)
 (2, 12), precision (3, 4, 13, 15)   smaller models (6, 7)           batching (8), speculation (9)
```

| Symptom | Bottleneck | First levers |
|---|---|---|
| OOM during training | memory | mixed precision, activation checkpointing, grad accumulation, ZeRO/FSDP (Lab 06), LoRA/QLoRA |
| Slow first token on long prompts | prefill compute | prefix caching, chunked prefill, FP8, FlashAttention |
| Slow token generation at batch 1 | memory bandwidth | weight quantization, speculative decoding, batching |
| Few concurrent users fit | KV-cache memory | paged KV cache, FP8 KV, GQA |
| GPU idle between requests | scheduling | in-flight batching, dynamic batching |

---

## Exercise 1 — `training_memory_bytes`: the 16 bytes/parameter rule

### Why it exists

Before launching training you must know whether it fits. The weights are only a fraction of the memory.

| item | bytes / param | why it exists |
|---|---|---|
| bf16 weights | 2 | forward/backward |
| bf16 gradients | 2 | result of backward |
| fp32 master weights | 4 | tiny updates vanish in bf16 (ex. 15), so the optimizer updates an fp32 copy |
| Adam m | 4 | running mean of gradients |
| Adam v | 4 | running mean of squared gradients |
| **total** | **16** | 7B → **112 GB** before activations |

The function: every weight stored at `weight_bytes`, plus **14 bytes per trainable** parameter.

| setup | result |
|---|---|
| full FT, 7B | 112 GB |
| LoRA, 7B, 20M trainable | 14.28 GB |
| QLoRA, 70B, 100M trainable, 4-bit | 36.4 GB |

### On the exam

*Weights + grads + Adam for full fine-tuning of 13B?* 13B × 16 ≈ **208 GB**. Wrong answers: 26 GB
(bf16 weights only), 52 GB, 1 TB. *Where does LoRA's saving come from?* Gradients and optimizer
states exist only for adapter parameters: **~14 bytes saved per frozen parameter**.

---

## Exercise 2 — `max_batch_size`: how many users fit on a GPU

### Why it exists

At inference, memory = weights + **KV cache per sequence** (Lab 01) × concurrent sequences + workspace.
Batch size decides throughput and cost per token, and it's usually limited by the KV cache, not by compute.

```
80 GB GPU, keep 10% for workspace → 72 GB; Llama-2-7B fp16 weights 13.5 GB → 58.5 GB left
KV per 4k-token sequence 2 GiB → floor(58.5e9 / 2.147e9) = 27 concurrent sequences
```

### On the exam

Every KV-cache technique raises that 27: **GQA** (fewer KV heads), **FP8 KV cache** (half the bytes:
*"8 sessions of 32k fit; how do I double that?"*), **paged KV** (no wasted reservations, ex. 12).
Weight quantization frees some memory but doesn't shrink the per-session cache.

---

## Exercise 3 — `quantize_int8`: fewer bytes per weight

### Why it exists

Fewer bits per weight means less memory **and** fewer bytes to read per decode step, which is
faster in memory-bound decode (ex. 10). The difficulty is **outliers**.

```
w = [[0.1, -0.5, 0.3], [40.0, -2.0, 1.0]]
per-tensor (one scale 40/127 = 0.315): row 0 → [0, -2, 1] → recovered [0.0, -0.63, 0.31]  (0.1 lost)
per-row scales (0.0039 and 0.315):     row 0 → [25, -127, 76] → [0.098, -0.50, 0.299] ✓
```

`scale = max|w| / 127`, `q = round(w / scale)`. One outlier forces a huge step size for everyone;
per-channel (or per-group of 128) scales isolate it. The test requires a 10× lower error.

### On the exam

| technique | what's quantized | helps |
|---|---|---|
| **weight-only INT4** (AWQ, GPTQ) | weights | memory-bound **decode** |
| **W8A8** (SmoothQuant, ex. 13) | weights + activations | compute-bound **prefill** (INT8 tensor cores) |
| **FP8** (Hopper/Ada, Transformer Engine) | weights + activations | training and inference |
| **KV-cache FP8/INT8** | cache | concurrency / context |
| **PTQ** | after training, calibration set | fast; fine at 8 bits |
| **QAT** | simulated quantization during training | when PTQ at 4-bit loses too much accuracy |

NVIDIA tooling: **TensorRT Model Optimizer** (PTQ, QAT, sparsity, distillation) → TensorRT-LLM engines.

---

## Exercise 4 — `DynamicLossScaler`: making FP16 training work

### Why it exists

FP16's smallest normal value is 6.1e-5 (ex. 15). Many gradients are smaller and **underflow to 0**:
`float16(1e-8) = 0`. **Loss scaling** multiplies the loss (so every gradient) by a large factor before
backward (`1e-8 × 1024 = 1.02e-5` survives), and unscales before the update. Too large a factor overflows
to `inf`, so the scale **adapts**:

| step | overflow? | action | scale after |
|---|---|---|---|
| 1 | yes | **skip** the update, halve | 512 |
| 2–3 | no | count good steps | 512 |
| 4 | no | 3 good steps → double | 1024 |
| 6 | yes | skip, halve, reset counter | 512 |

`update()` returns False, meaning **skip `optimizer.step()`**, because an `inf` gradient would destroy the weights.

### On the exam

*Repeated inf/NaN and a dropping scale:* each overflow step is **skipped and the scale halved**.
**BF16 has FP32's exponent range, so no loss scaling is needed**, the main reason A100/H100 training uses BF16.

---

## Exercise 5 — `accumulated_gradients`: a big batch that doesn't fit

### Why it exists

The recipe says global batch 512, but only 16 fit per GPU. Run several micro-batches and **add their
gradients** before one optimizer step.

```python
model.zero_grad()                         # once
for chunk in 4 chunks:
    (loss(chunk) / 4).backward()          # backward ADDS into .grad; ÷4 so the sum = full-batch mean
optimizer.step()
```

The test checks that this equals the full-batch gradient to floating-point precision.

### On the exam

**Global batch = micro-batch × accumulation steps × data-parallel GPUs**: 16 × 4 × 8 = 512.
It saves **memory, not time**, and it isn't equivalent to scaling the learning rate.

---

## Exercise 6 — `distillation_loss`: a small student copies a big teacher

### Why it exists

A small model trained on hard labels only learns "the answer is 7". A teacher's full distribution
says "7, but a bit like 1", and that **dark knowledge** trains better students. That's how compact
production models are made cheaply (NVIDIA **Minitron** = prune + distil).

```
L = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T)) + (1 − α) · CE(student, label)
teacher logits [3, 1, -1]:  T=1 → [0.867, 0.117, 0.016]   T=4 → [0.506, 0.307, 0.186]
```

- **T > 1** softens the teacher so the relationships between classes become visible (the same temperature as Lab 02).
- **T²** restores the gradient scale that softening shrinks (by about 1/T²).
- `F.kl_div` takes the student as **log-probs** and the teacher as **probs**.

### On the exam

*Why divide by T > 1 and multiply by T²?* Softening reveals the teacher's ranking of wrong classes,
and T² balances the two loss terms. Wrong: "deterministic outputs", "avoids overflow", "only matters at inference".

---

## Exercise 7 — `prune_2_4`: sparsity the GPU can use

### Why it exists

Zeroing random weights rarely speeds anything up, because GPUs can't skip irregular zeros.
**2:4 structured sparsity**, exactly 2 zeros in every 4 consecutive weights, is a pattern that
**sparse tensor cores** (Ampere and later) accelerate by up to 2× on matmuls.

```
[0.1, -0.9, 0.3, 0.05 | 2.0, -3.0, 0.0, 1.0]  →  [0, -0.9, 0.3, 0 | 2.0, -3.0, 0, 0]
```

Keep the 2 largest magnitudes per group. Accuracy drops, so **fine-tune after pruning**.

### On the exam

2:4 gives 50% sparsity with hardware support. Unstructured sparsity isn't accelerated automatically,
and 2:4 doesn't shrink the KV cache or remove layers (removing layers is depth pruning, as in Minitron).

---

## Exercise 8 — static vs in-flight batching

### Why it exists

LLM outputs vary wildly in length (10 to 2,000 tokens). With **static batching** every request
waits for the longest in its batch, so slots sit idle. **In-flight (continuous) batching** swaps
finished requests out and new ones in **at every decode step**.

```
lengths [10, 200, 12, 15, 180, 9, 11, 14], 4 slots (451 tokens of real work)
static:    [10,200,12,15] → 200 steps, [180,9,11,14] → 180 steps   total 380
in-flight: slot 1 finishes at 10 → the 180-token request starts … last finish at 200
```

A min-heap of slot-free times: each request takes the earliest free slot.

### On the exam

In-flight batching (TensorRT-LLM, vLLM, NIM) improves **throughput and GPU utilisation**, not the
latency of a single request. Bigger static batches or padding outputs to the maximum make the waste worse.

---

## Exercise 9 — `speculative_expected_tokens`: guessing ahead

### Why it exists

Decode is memory-bound (ex. 10): verifying 5 tokens costs about the same as generating 1. A cheap
**draft** model guesses γ tokens, and the big model checks them in **one** pass.

```
expected tokens per big-model pass = 1 + α + α² + … + α^γ = (1 − α^(γ+1)) / (1 − α)
```

| α \ γ | 2 | 4 | 8 |
|---|---|---|---|
| 0.5 | 1.75 | 1.94 | 2.00 |
| 0.8 | 2.44 | 3.36 | 4.33 |
| 0.9 | 2.71 | 4.10 | 6.13 |

### On the exam (Select TWO)

✅ the draft proposes and the target verifies in one pass; ✅ the output **distribution is identical** to
the target's. ❌ lower memory (both models are loaded); ❌ biggest gains at large batch (it's a
**low-batch latency** technique); ❌ retraining the target. Variants: Medusa heads, EAGLE.

---

## Exercise 10 — `decode_tokens_per_sec_bound`: the decode roofline

### Why it exists

Every decode step reads **every weight once** to produce one token per sequence. At batch 1 the GPU waits on memory:

```
tokens/s ≤ bandwidth / weight bytes × batch
```

| model | GPU | bound |
|---|---|---|
| 70B fp16 (140 GB) | H100 3.35 TB/s | ~24 tokens/s |
| 8B fp16 | L4 300 GB/s | ~19 tokens/s |
| 8B INT4 | L4 300 GB/s | ~75 tokens/s |

This one line explains batching (one read serves many sequences), weight quantization (fewer bytes)
and speculative decoding (more tokens per read).

### On the exam

*70B FP16 at batch 1 is far below peak FLOPs: bottleneck?* **Memory bandwidth.** Not tensor-core compute,
not PCIe, not the tokenizer.

---

## Exercise 11 — `checkpointed_activation_bytes`: trading compute for memory

### Why it exists

Backward needs the activations saved during forward. They grow with **batch × sequence × hidden × layers**
and often dominate memory at long sequences. **Activation (gradient) checkpointing** keeps only
some layer inputs and **recomputes** the rest during backward.

### How it works

Store the input of every segment (a "checkpoint"); in backward, recompute one segment at a time:

```
memory = (#segments × layer_input) + (segment_size × layer_activations)
64 layers, input and activations cost 1 unit each:
  segment 1 → 64+1 = 65    segment 8 → 8+8 = 16 (minimum: the classic √n rule)    no checkpointing → 64
realistic (a layer's activations ≈ 17× its input): segment 2 → 32 + 34 = 66 units vs 1,088 without, ~16× less
```

Cost: an **extra forward pass**, typically **+20–35% compute**. **Selective checkpointing**
recomputes only cheap-but-large parts (like attention scores) to cut the overhead.

### On the exam

*OOM from activations at 8k sequence length; weights fit:* **activation checkpointing** (costs extra
compute). Wrong: INT4 weights (activations unchanged), a higher learning rate, "disable the KV cache"
(an inference mechanism).

---

## Exercise 12 — `PagedKVCache`: virtual memory for the KV cache

### Why it exists

Naive servers **reserve a contiguous max-length buffer** per request: 2,048 slots for an answer that
ends after 20 tokens. Most of the KV memory is wasted or fragmented, so fewer requests fit.
**PagedAttention** (vLLM, TensorRT-LLM, NIM) allocates the cache in **small fixed-size blocks on
demand**, tracked by a per-sequence **block table**, like OS virtual memory pages.

### How it works

8 blocks of 16 slots; sequence "a" has 20 tokens, "b" has 5:

```
a → blocks [7, 6]   (32 slots, 20 used)     b → block [5]   (16 slots, 5 used)
wasted: 12 + 11 = 23 slots      vs contiguous 2,048-slot reservations: 4,071 slots wasted
free("a") → its 2 blocks return to the pool immediately
```

- A new block is taken only when the current one is full, and **any** free block works (no need for contiguity).
- An empty pool raises `MemoryError`; real schedulers then **preempt** or queue requests.
- Bonus: identical prefixes (a shared system prompt) can **share blocks**, which is prefix caching.

### On the exam

*What does PagedAttention solve?* **Fragmentation and over-reservation of contiguous KV buffers.**
Not attention FLOPs (that's FlashAttention's area, Lab 01), not FP16 softmax stability, not tokenisation.

---

## Exercise 13 — SmoothQuant: making activations quantizable

### Why it exists

W8A8 (INT8 weights **and** activations) speeds up compute-bound prefill on INT8 tensor cores. But LLM
**activations have huge outliers in a few channels**, so one INT8 scale can't fit both the outlier and
the normal values. Weights are smooth and easy to quantize. **SmoothQuant moves the difficulty from activations into weights.**

### How it works

```
s_j = max|X_j|^α / max|W_j|^(1−α)       (α = 0.5)
X' = X / s,  W' = s · W   →  X'·W' = X·W  exactly (the test checks it)
```

With one activation channel 60× larger than the rest: the channel max/median ratio drops from 54 to
10, and the **W8A8 error drops about 3×** (0.052 → 0.017). No retraining: it's a mathematically
equivalent rescaling, done offline.

### On the exam

*W8A8 accuracy is ruined by activation outliers:* **SmoothQuant.** AWQ is weight-only (activations
untouched), distillation doesn't fix outliers, and temperature is unrelated.

---

## Exercise 14 — `latency_breakdown`: TTFT vs time per output token

### Why it exists

Users feel two different delays, caused by two different phases:

| phase | what | bound by | metric |
|---|---|---|---|
| **prefill** | process the whole prompt in parallel, build the KV cache | **compute** | **TTFT** (time to first token) |
| **decode** | one token at a time | **memory bandwidth** | **TPOT / ITL** (time per output token) |

### How it works

A 12,000-token RAG prompt, 300 output tokens, prefill 20k tokens/s, decode 40 tokens/s:

```
TTFT = 12,000 / 20,000 = 0.60 s      TPOT = 1/40 = 25 ms      end-to-end = 0.6 + 299 × 0.025 = 8.08 s
with 10,000 prompt tokens already cached (prefix caching): TTFT = 0.10 s, TPOT unchanged
```

### On the exam

*Long RAG prompts, slow first token, fine inter-token latency (Select TWO):* **prefix/KV-cache reuse**
and **faster prefill** (chunked prefill, FP8, efficient attention). Weight-only INT4 and speculative
decoding target **decode**, and max_tokens doesn't affect TTFT. **Chunked prefill** splits long prompts
so they don't stall other users' decode steps, which smooths ITL spikes.

---

## Exercise 15 — `float_format`: range vs precision

### Why it exists

"Which precision?" questions come down to two properties: **range** (the largest and smallest
representable values, set by the exponent bits) and **precision** (the gap between neighbouring
values, set by the mantissa bits).

| format | exp / mantissa | max | smallest normal | epsilon |
|---|---|---|---|---|
| FP32 | 8 / 23 | 3.4e38 | 1.2e-38 | 1.2e-7 |
| **BF16** | **8** / 7 | **3.4e38** | 1.2e-38 | 7.8e-3 |
| **FP16** | 5 / **10** | **65,504** | 6.1e-5 | 9.8e-4 |
| FP8 E5M2 | 5 / 2 | 57,344 | 6.1e-5 | 0.25 |
| FP8 E4M3 | 4 / 3 | 448 * | 1.6e-2 | 0.125 |

\* E4M3 as used on Hopper (OCP FP8 spec) reuses the exponent codes an IEEE layout reserves for inf, so its
max is 448 rather than the IEEE-style 240 this formula gives.

- **BF16 = FP32's range with less precision**, so no overflow/underflow and no loss scaling.
- **FP16 = more precision, tiny range**, so it needs loss scaling (ex. 4).
- **FP8 on Hopper:** **E4M3 for forward** (weights, activations: more precision), **E5M2 for gradients**
  (more range), with per-tensor scaling managed by **Transformer Engine**.
- **TF32** (Ampere+): FP32 range with a 10-bit mantissa, used automatically for FP32 matmuls on tensor cores.

### On the exam

*Why prefer BF16 over FP16 on A100/H100?* **Same exponent range as FP32, so gradients rarely overflow
or underflow and no loss scaling is needed.** Wrong: "more mantissa bits" (it has fewer), "half the
memory of FP16" (both are 16 bits), "the only tensor-core format".

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise / section |
|---|---|
| Training memory, LoRA memory | 1 |
| BF16 vs FP16, loss scaling | 4, 15 |
| Gradient accumulation | 5 |
| Activation checkpointing | 11 |
| Decode bottleneck | 10 |
| In-flight batching | 8 |
| Paged KV cache | 12 |
| KV-cache quantization | 2, 3 |
| Quantization granularity, PTQ vs QAT | 3 |
| SmoothQuant / W8A8 | 13 |
| Speculative decoding | 9 |
| Knowledge distillation | 6 |
| 2:4 sparsity | 7 |
| Latency vs throughput, TTFT | 8, 14 |
| TensorRT-LLM, data loading | README §4–5 and the bottleneck table at the top |
