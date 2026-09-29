# Lab 06 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: one GPU runs out of **memory** (the model doesn't fit) or **time** (training takes
months). Distributed training splits the work, and every way of splitting it adds
**communication** between GPUs. The whole domain is about that trade-off.

```
split the DATA   → data parallel (DDP), ZeRO / FSDP           ex. 1, 6
split the LAYERS → pipeline parallel (PP)                      ex. 2
split each MATRIX → tensor parallel (TP)                       ex. 4, 5
and pay for it   → all-reduce, all-gather… over NVLink / network   ex. 3
measure it       → FLOPs, MFU, roofline                        ex. 7, 8
```

---

## Exercise 1 — `zero_bytes_per_gpu`: stop storing the same thing N times

Plain **DDP** keeps a full copy of everything on every GPU: 16 bytes per parameter (Lab 05).
Across 64 GPUs that is 64 identical copies of the optimizer state. **ZeRO** shards (splits) the
redundant parts across the data-parallel GPUs:

```
per parameter:  2 B params  +  2 B grads  +  12 B optimizer (fp32 master, m, v)
stage 0 (DDP):  2 + 2 + 12                   = 16 B
stage 1:        2 + 2 + 12/N                 ← optimizer states sharded
stage 2:        2 + (2 + 12)/N               ← + gradients sharded
stage 3 / FSDP: (2 + 2 + 12)/N               ← + parameters sharded
```

| stage | 7.5B model, 64 GPUs | 7B model, 8 GPUs |
|---|---|---|
| 0 (DDP) | 120.0 GB | 112.0 GB |
| 1 | 31.4 GB | 38.5 GB |
| 2 | 16.6 GB | 26.3 GB |
| 3 (FSDP FULL_SHARD) | 1.9 GB | 14.0 GB |

The cost is **communication**. Stage 3 must all-gather each layer's parameters just before using
it (and free them after). The code is the formula: pick which parts get divided by N.

**Exam link:** "high memory per GPU with DDP" → ZeRO/FSDP. FSDP `SHARD_GRAD_OP` ≈ ZeRO-2, `FULL_SHARD` ≈ ZeRO-3.

---

## Exercise 2 — `pipeline_bubble_fraction`: idle time in pipelines

Pipeline parallelism puts layers 1–8 on GPU 0, 9–16 on GPU 1, and so on. GPU 1 can't start until
GPU 0 finishes, so at the start and end of each step some GPUs wait: the **bubble**.

```
4 stages, 4 micro-batches (F = forward of micro-batch n):
GPU0  F1 F2 F3 F4 .  .  .
GPU1  .  F1 F2 F3 F4 .  .        "." = idle
GPU2  .  .  F1 F2 F3 F4 .
GPU3  .  .  .  F1 F2 F3 F4       each GPU is idle (p − 1) = 3 slots out of (m + p − 1) = 7
```

`bubble / (compute + bubble) = (p − 1) / (m + p − 1)`, in the code as `bubble / (micro_batches + bubble)`.

| micro-batches m | 4 stages | 8 stages |
|---|---|---|
| 4 | 42.9% idle | 63.6% |
| 8 | 27.3% | 46.7% |
| 16 | 15.8% | 30.4% |
| 32 | 8.6% | 17.9% |
| 64 | 4.5% | 9.9% |

**More micro-batches shrink the bubble. More stages make it worse.** Interleaved schedules give each
GPU `v` smaller chunks and divide the bubble by `v`: 4 stages, 4 micro-batches, v = 3 → 20%.

---

## Exercise 3 — `ring_allreduce_bytes` / `allreduce_seconds`: what communication costs

DDP must **all-reduce** gradients: every GPU ends with the sum (then the average) of all GPUs'
gradients. NCCL's **ring** algorithm passes chunks around a ring in two phases (reduce-scatter,
then all-gather). Each GPU sends:

```
2 · (N − 1) / N · size
```

For 14 GB of gradients (7B params in bf16):

| GPUs | sent per GPU | at 400 GB/s (NVLink-class) | at 25 GB/s (PCIe-class) |
|---|---|---|---|
| 2 | 14.0 GB | 0.035 s | 0.56 s |
| 8 | 24.5 GB | 0.061 s | 0.98 s |
| 64 | 27.6 GB | 0.069 s | 1.10 s |

The per-GPU traffic approaches `2 × size` and **barely grows with more GPUs**. That is why the
ring is bandwidth-optimal. **Link speed** is what matters: the same all-reduce is 16× slower over
PCIe than over NVLink. The GPU lab's `allreduce_bench.py` measures your real link (L4 and T4 are PCIe).

---

## Exercise 4 — `rank_coords` / `tensor_parallel_group`: who talks to whom

With 16 GPUs (2 nodes × 8), TP = 4 and PP = 2 give DP = 16 / (4·2) = 2. Every GPU (rank) gets a
coordinate. The Megatron order makes **TP the fastest-changing**:

```
rank = tp_rank + tp · (dp_rank + dp · pp_rank)
```

| rank | (tp, dp, pp) |
|---|---|
| 0 | (0, 0, 0) |
| 3 | (3, 0, 0) |
| 4 | (0, 1, 0) |
| 5 | (1, 1, 0) |
| 8 | (0, 0, 1) |
| 13 | (1, 1, 1) |

Decoding uses `%` and `//`: `tp = rank % tp`, `dp = (rank // tp) % dp`, `pp = rank // (tp·dp)`.

**Why TP innermost?** Ranks 4–7 form one TP group: **consecutive ranks sit on the same node**, and
TP communicates inside every layer, so it needs NVLink. PP only passes activations between stages
occasionally and can cross nodes. That's the exam rule: **TP within a node, PP across nodes, DP outermost.**
`tensor_parallel_group(6, 4)` rounds down to the group start: `[4, 5, 6, 7]`.

---

## Exercise 5 — tensor parallelism (Megatron MLP)

A transformer MLP is `relu(x · W1) · W2`. Split it over n GPUs **without** communicating in the middle:

1. **Column-parallel W1:** each GPU gets some *columns* of W1 and computes its slice of the hidden
   layer. No communication, because each output column depends only on its own column of W1.
2. **Apply the activation locally:** relu works element by element, so it runs on each slice independently.
3. **Row-parallel W2:** each GPU holds the matching *rows* of W2, multiplies its hidden slice, and
   gets a **partial sum** of the final output.
4. **One all-reduce** adds the partial sums. That is the only communication for the whole MLP.

Example, 2 GPUs: `x = [1, 2]`, `W1 = [[1, 0, -1, 2], [0, 1, -1, -1]]`, `W2 = 4 rows of 1`.

```
full:  x·W1 = [1, 2, -3, 0] → relu → [1, 2, 0, 0] → ·W2 = 3
GPU 0: columns 0–1 → [1, 2]  → relu [1, 2] → · rows 0–1 of W2 = 3
GPU 1: columns 2–3 → [-3, 0] → relu [0, 0] → · rows 2–3 of W2 = 0
all-reduce: 3 + 0 = 3 ✓
```

Why not split W1 by rows? Then each GPU would hold a *partial sum* of the hidden layer, and relu of
a partial sum is wrong (`relu(a) + relu(b) ≠ relu(a + b)`), so you'd need an all-reduce *before* the
activation too. Column-then-row is the trick that halves the communication.

---

## Exercise 6 — `average_gradients`: the heart of DDP

Each GPU (rank) computes gradients on **its own slice of the batch**. To stay identical, all
replicas must apply the **same** update, the average gradient:

```python
for p in model.parameters():
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)   # every rank now holds the SUM
    p.grad /= dist.get_world_size()                 # SUM / N = average
```

`ddp_check.py` runs this for real: `torchrun` starts 2 processes, each computes gradients on half
of 32 examples, calls your function, and rank 0 checks the result equals the single-process
gradient on all 32 examples. That works because the average of the half-batch mean gradients
equals the full-batch mean gradient when the halves are equal-sized. It uses **gloo** on CPU and
**NCCL** on GPUs, with no code change.

Real DDP does the same thing, but in **buckets** (~25 MB) **overlapped with backward**: while
later layers are still computing gradients, earlier buckets are already being all-reduced.

---

## Exercise 7 — `training_flops` / `mfu`: how long will training take?

A transformer spends about **2 FLOPs per parameter per token** in the forward pass (a multiply and
an add) and about **4** in backward (gradients with respect to activations and weights). So:

```
training FLOPs ≈ 6 · N (params) · D (tokens)
Llama-2-7B, 2T tokens: 6 × 7e9 × 2e12 = 8.4 × 10²² FLOPs
```

**MFU** (model FLOPs utilisation) is how much of the hardware's peak you actually use:

```
MFU = 6 · N · tokens_per_second / (n_gpus · peak_flops_per_gpu)
1024 A100s (312 TFLOPS) at 3.2M tokens/s on 7B: 1.34e17 / 3.19e17 = 42%
```

At 42% MFU, that 7B / 2T-token run takes about **7.2 days** on those 1024 GPUs. Good large runs
reach 35–55%. Low MFU points to communication, pipeline bubbles, input stalls or small kernels,
and **Nsight Systems** is how you find which.

---

## Exercise 8 — roofline: memory-bound or compute-bound?

A kernel needs data from memory and does math on it. **Arithmetic intensity** = FLOPs per byte moved.

```
GEMM C(m,n) = A(m,k)·B(k,n):   FLOPs = 2·m·n·k     bytes = 2·(m·k + k·n + m·n)   (bf16)
```

| m (tokens in the batch) | intensity with n = k = 8192 |
|---|---|
| 1 (decode, batch 1) | 1.0 FLOP/byte |
| 16 | 15.9 |
| 256 | 240.9 |
| 4096 (prefill) | 2048 |

The GPU's **ridge point** is `peak FLOPs / bandwidth`: H100 ≈ 295, L4 ≈ 403, T4 ≈ 203 FLOPs/byte.
Below it the GPU waits on memory (**memory-bound**); above it the math units are the limit
(**compute-bound**). Decode at batch 1 sits at 1, hundreds of times below the ridge, which is why
batching, quantization and speculative decoding (Lab 05) help it, and why prefill (thousands of
tokens at once) is compute-bound.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| high memory per GPU, multi-node, slow | ZeRO/FSDP sharding + TP within node + mixed precision |
| shard optimizer + grads, keep params | ZeRO-2 / FSDP SHARD_GRAD_OP |
| pipeline idle time | more micro-batches, interleaved schedule |
| TP across nodes / over PCIe | bad: TP needs NVLink |
| collectives: DDP / FSDP / MoE / PP | all-reduce / all-gather + reduce-scatter / all-to-all / send-recv |
| where does time go, gaps in the timeline | Nsight Systems (then Nsight Compute per kernel) |
| label phases in the profile | NVTX ranges |
| all processes use GPU 0 | missing `torch.cuda.set_device(LOCAL_RANK)` |
| poor multi-node scaling | NCCL fabric check (nccl-tests, NCCL_DEBUG), overlap, bigger batches |
| MFU / training time | 6·N·D; MFU = achieved / peak |
