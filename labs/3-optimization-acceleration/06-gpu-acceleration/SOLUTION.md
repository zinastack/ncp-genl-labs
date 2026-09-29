# Lab 06 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: one GPU runs out of **memory** (the model doesn't fit) or **time** (training takes
months). Distributed training splits the work, and **every way of splitting adds communication**.
The domain is about choosing the split that costs the least communication on your hardware, then
checking with a profiler that the GPUs are actually busy.

```
split the DATA     → DDP (6, 11), ZeRO / FSDP (1, 9)            all-reduce / reduce-scatter + all-gather
split the LAYERS   → pipeline parallel (2)                       point-to-point send/recv
split each MATRIX  → tensor parallel (4, 5)                      all-reduce inside every layer
split the SEQUENCE → context parallel (10)                       K/V passed around a ring
split the EXPERTS  → expert parallel (MoE)                       all-to-all (9)
measure it         → comm cost (3), MFU (7), roofline (8), timelines (12), scaling (13)
```

---

## First: which parallelism for which problem

| Problem | Technique | Communication | Where it belongs |
|---|---|---|---|
| Model fits; want more throughput | **DDP** | all-reduce of gradients, overlapped with backward | anywhere |
| Model + optimizer don't fit per GPU | **ZeRO-1/2/3, FSDP** | reduce-scatter grads; all-gather params (stage 3) | DP dimension |
| Single layers too big / too slow | **tensor parallel** | all-reduce per layer (on the critical path) | **inside a node (NVLink)**, TP ≤ 8 |
| Model too deep for one node | **pipeline parallel** | activations between stages | **across nodes** |
| Sequences too long (128k+) | **context parallel** (ring attention) | K/V blocks around a ring | with TP |
| LayerNorm/dropout activations with TP | **sequence parallel** | reduce-scatter/all-gather instead of all-reduce | with TP |
| MoE with many experts | **expert parallel** | **all-to-all** token routing | across GPUs |

Rule of thumb for 3D parallelism: **TP inside the node, PP across nodes, DP outermost.**

---

## Exercise 1 — `zero_bytes_per_gpu`: stop storing the same thing N times

### Why it exists

Plain DDP keeps **16 bytes per parameter on every GPU** (Lab 05): 64 GPUs hold 64 identical copies
of the optimizer state. **ZeRO** shards the redundant parts across the data-parallel GPUs:

```
per parameter: 2 B params + 2 B grads + 12 B optimizer (fp32 master, m, v)
stage 0 (DDP):  2 + 2 + 12          stage 1: 2 + 2 + 12/N
stage 2:        2 + (2 + 12)/N      stage 3 / FSDP FULL_SHARD: (2 + 2 + 12)/N
```

| stage | 7.5B, 64 GPUs | 7B, 8 GPUs |
|---|---|---|
| 0 | 120 GB | 112 GB |
| 1 | 31.4 GB | 38.5 GB |
| 2 | 16.6 GB | 26.3 GB |
| 3 | 1.9 GB | 14.0 GB |

The price is communication: stage 3 must **all-gather** each layer's parameters just before using
them and free them afterwards (exercise 9).

### On the exam

*Shard optimizer states **and** gradients, keep full parameters:* **ZeRO-2 / FSDP `SHARD_GRAD_OP`.**
`FULL_SHARD` = ZeRO-3. `HYBRID_SHARD` = shard within a node, replicate across nodes (less
inter-node traffic). **ZeRO-Offload / CPU offload** moves optimizer state to CPU RAM: it fits bigger models, but runs slower.

---

## Exercise 2 — `pipeline_bubble_fraction`: idle time in pipelines

### Why it exists

Pipeline parallelism puts groups of layers on different GPUs. GPU 1 can't start until GPU 0 has
finished a micro-batch, so at the start and end of every step some GPUs idle: the **bubble**.

```
idle fraction = (p − 1) / (m + p − 1)     p = stages, m = micro-batches per step
```

| m | 4 stages | 8 stages |
|---|---|---|
| 4 | 42.9% | 63.6% |
| 8 | 27.3% | 46.7% |
| 32 | 8.6% | 17.9% |
| 64 | 4.5% | 9.9% |

**More micro-batches shrink the bubble; more stages grow it.** Interleaved (virtual) stages divide it by v.
1F1B scheduling has the same bubble as GPipe but lower activation memory.

---

## Exercise 3 — ring all-reduce cost

### Why it exists

DDP must sum gradients across all GPUs every step. NCCL's **ring** all-reduce has each GPU send

```
2 · (N − 1) / N · size        (reduce-scatter pass + all-gather pass, exercise 9)
```

| GPUs | per GPU, 14 GB of gradients | at 400 GB/s | at 25 GB/s (PCIe-class) |
|---|---|---|---|
| 2 | 14.0 GB | 0.035 s | 0.56 s |
| 8 | 24.5 GB | 0.061 s | 0.98 s |
| 64 | 27.6 GB | 0.069 s | 1.10 s |

The per-GPU traffic barely grows with N (bandwidth-optimal); **link speed** is what matters.
`allreduce_bench.py` measures your real link in the GPU lab: L4 and T4 are PCIe-only.

---

## Exercise 4 — `rank_coords`: where each GPU sits in 3D parallelism

### Why it exists

With TP × PP × DP, each GPU (rank) needs a coordinate saying which pieces it holds and whom it talks
to. Megatron makes **TP the fastest-changing** dimension:

```
rank = tp_rank + tp · (dp_rank + dp · pp_rank)      16 GPUs, TP = 4, PP = 2 → DP = 2
rank 5 → (tp 1, dp 1, pp 0)     rank 13 → (1, 1, 1)     TP group of rank 6 → [4, 5, 6, 7]
```

**Why TP innermost:** consecutive ranks sit on the same node, so the chattiest groups talk over NVLink.

### On the exam

*175B on 32 nodes × 8 H100:* **TP = 8 inside each node, PP across nodes, DP over the rest.**
Wrong: TP = 32 across nodes (per-layer all-reduces over the network), PP inside and TP across (backwards),
DP only (a 175B replica can't fit on one GPU).

---

## Exercise 5 — tensor parallelism (Megatron MLP)

### Why it exists

When a single layer is too big or slow for one GPU, **split its matrices**. Megatron's trick needs
only **one all-reduce for the whole MLP**:

1. **Column-parallel W1:** each GPU computes its slice of the hidden layer. No communication.
2. The **activation runs locally** (element-wise).
3. **Row-parallel W2:** each GPU produces a **partial sum** of the output.
4. **All-reduce** adds the partials.

```
x = [1, 2], W1 = [[1, 0, -1, 2], [0, 1, -1, -1]], W2 = column of 1s
full: relu([1, 2, -3, 0]) · W2 = 3
GPU 0: relu([1, 2]) · W2[0:2] = 3     GPU 1: relu([-3, 0]) · W2[2:4] = 0     all-reduce → 3 ✓
```

Splitting W1 by rows instead would need an all-reduce *before* relu, because `relu(a) + relu(b) ≠ relu(a + b)`.

### On the exam

TP communicates **twice per transformer block** in forward (and again in backward), **on the critical path**,
so it needs **NVLink/NVSwitch**. Over PCIe, DP or PP usually scale better.

---

## Exercise 6 — `average_gradients`: the heart of DDP

### Why it exists

Each GPU computes gradients on **its own slice** of the batch (DistributedSampler). To keep
replicas identical, all must apply the **same** update: the average gradient.

```python
dist.all_reduce(p.grad, op=SUM); p.grad /= world_size
```

`ddp_check.py` runs this under real `torchrun` (gloo on CPU, NCCL on GPUs) and checks it equals the
single-process full-batch gradient.

### On the exam

*What does DDP synchronise every step?* **Gradients** (all-reduce during backward). Weights are
broadcast **once** at start-up; inputs differ per rank; activations stay local.
*All 4 processes allocate on GPU 0:* missing `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))`.

---

## Exercise 7 — `training_flops` / `mfu`

### Why it exists

To plan a run (and to know whether your cluster is being used well):

```
training FLOPs ≈ 6 · N · D       (2 forward + 4 backward per parameter per token)
Llama-2-7B, 2T tokens: 8.4e22 FLOPs
MFU = 6 · N · tokens/s ÷ (GPUs × peak):  1024 A100 at 3.2M tokens/s on 7B → 42%  (≈ 7.2 days)
```

Good large runs reach **35–55% MFU**. Lower MFU points to communication, bubbles, input stalls or small kernels, and exercise 12 shows how to find which.

### On the exam

*70B at 250k tokens/s on 512 H100 (989 TFLOPS):* 6 × 70e9 × 2.5e5 / (512 × 9.89e14) ≈ **21%**.

---

## Exercise 8 — roofline: memory-bound or compute-bound?

### Why it exists

A kernel is limited either by math throughput or by how fast data arrives. **Arithmetic intensity**
(FLOPs per byte moved) decides which:

| GEMM m (tokens), n = k = 8192, bf16 | intensity |
|---|---|
| 1 (decode) | 1.0 FLOP/byte |
| 256 | 241 |
| 4096 (prefill) | 2048 |

**Ridge point** = peak FLOPs ÷ bandwidth: H100 ≈ 295, L4 ≈ 403, T4 ≈ 203. Below it the kernel is
**memory-bound**; above it, **compute-bound**.

### On the exam (Select TWO: memory-bound)

✅ decode at small batch (GEMV); ✅ element-wise ops (activations, residual adds, norms), which is why
**kernel fusion** helps. ❌ large-batch prefill GEMMs, large square matmuls and FP8 GEMMs at scale are compute-bound.

---

## Exercise 9 — the collectives: all-reduce, reduce-scatter, all-gather, all-to-all

### Why it exists

Every parallelism strategy is defined by **which collective it uses**, and exam questions ask exactly that.

```
rank 0 holds [1, 2, 3, 4], rank 1 holds [10, 20, 30, 40]

all_reduce      → both hold  [11, 22, 33, 44]                        DDP gradients
reduce_scatter  → rank 0: [11, 22]   rank 1: [33, 44]                 ZeRO-2/3 / FSDP gradients
all_gather      → both hold the concatenation of everyone's pieces    FSDP parameters, TP outputs
all_to_all      → rank j receives what every rank addressed to j      MoE expert routing
```

**The key insight, which the test checks:** `all_gather(reduce_scatter(x)) == all_reduce(x)`.
That's how NCCL's ring all-reduce works internally (two passes, hence the `2·(N−1)/N` in exercise 3),
and why FSDP costs about the same communication as DDP: it just splits the two halves apart in time.

### On the exam

| technique | collective |
|---|---|
| DDP gradient sync | **all-reduce** |
| ZeRO / FSDP | **reduce-scatter** (grads) + **all-gather** (params) |
| Megatron TP | **all-reduce** (or RS + AG with sequence parallelism) |
| pipeline parallel | point-to-point **send/recv** |
| MoE expert parallel | **all-to-all** |
| broadcast | initial weights from rank 0 |

---

## Exercise 10 — `ring_attention_rank`: context parallelism

### Why it exists

At 128k–1M tokens, even one layer's attention activations don't fit on a GPU. **Context parallelism**
splits the **sequence** across GPUs: each rank owns a block of queries, and key/value blocks **travel
around a ring**, so every query eventually sees every key, but **no GPU ever holds the whole sequence's K and V**.

### How it works

Exactly FlashAttention's **online softmax** (Lab 01, exercise 14), with blocks that come from other
GPUs instead of from local memory:

```
for each (k, v) block arriving around the ring:
    s = q·kᵀ/√d;  m_new = max(m, max s);  c = exp(m − m_new)
    l = l·c + Σ exp(s − m_new);   acc = acc·c + exp(s − m_new)·v;   m = m_new
return acc / l
```

The test shows it matches full attention to ~1e-16 **in any arrival order**. Real implementations
overlap sending the next block with computing on the current one.

### On the exam

*128k-context training with TP = 8 still OOMs on attention activations:* **context parallelism
(ring attention)**. Sequence parallelism covers LayerNorm/dropout regions, while vocabulary size,
removing packing and switching Adam for SGD don't address long-sequence activations.

---

## Exercise 11 — `gradient_buckets`: how DDP hides communication

### Why it exists

Waiting until backward finishes and then all-reducing everything would leave the GPU idle during
communication. Backward computes gradients **from the last layer to the first**, so DDP groups them
in that order into **buckets** (default ~25 MB) and starts each bucket's all-reduce **as soon as it's
full**, while backward continues on earlier layers.

```
param sizes (MB) [10, 30, 5, 5, 20, 8], cap 25 MB, walked in reverse:
[5] (8)  → [4, 3] (20+5)  → [2] (5)  → [1] (30, oversized, alone)  → [0] (10)
```

### On the exam

- **Bigger buckets** mean fewer, more efficient messages, but communication starts later. **Smaller buckets**
  overlap earlier, with more per-message latency (`bucket_cap_mb` tunes this).
- *Poor multi-node scaling with exposed all-reduce (Select TWO):* **check the fabric** (InfiniBand/RoCE,
  GPUDirect RDMA, NCCL not falling back to TCP; `nccl-tests`, `NCCL_DEBUG=INFO`) and **raise compute per
  all-reduce** (bigger per-GPU batch or accumulation, bucketing and overlap).

---

## Exercise 12 — `timeline_stats`: reading a profile

### Why it exists

"Training is slow" is a symptom; **Nsight Systems** shows *where* the time goes on a timeline of CPU
threads, CUDA kernels, NCCL calls and your **NVTX** ranges. This exercise computes the numbers you'd read off it:

```
compute 0–10, NCCL 8–12, compute 12–20, NCCL 20–26, (idle 26–28), compute 28–30     span 30
gpu_busy     = 28/30 = 93%
compute      = 20/30 = 67%
exposed_comm =  8/30 = 27%   (10–12 and 20–26; 8–10 was hidden behind compute)
```

Merging overlapping intervals (`_union_length`) is the key operation: overlapped communication costs nothing.

### On the exam: which tool

| question | tool |
|---|---|
| Where is time going? Gaps? Data loading? NCCL overlap? | **Nsight Systems** (system timeline) |
| Why is *this kernel* slow (memory- vs compute-bound, occupancy, stalls)? | **Nsight Compute** (`ncu`) |
| Label forward/backward/optimizer on the timeline | **NVTX ranges** |
| Top ops by CUDA time from Python | PyTorch profiler |
| Fleet GPU health (util, memory, ECC, XID) for Prometheus | **DCGM / dcgm-exporter** |

Common timeline findings: gaps between kernels caused by the **data loader** (more workers, pinned
memory), **host syncs** (`.item()`, `.cpu()` inside the loop), **exposed NCCL** (fabric or overlap),
and tiny kernels (launch overhead → CUDA graphs, fusion).

---

## Exercise 13 — `scaling_efficiency` and `amdahl_speedup`

### Why it exists

Adding GPUs helps only as far as the work parallelises.

```
1 GPU 1,000 tokens/s, 8 GPUs 6,200 tokens/s → efficiency 6,200 / 8,000 = 77.5%
Amdahl, 95% parallel: 8 GPUs → 5.9×, 64 → 15.4×, 1024 → 19.6×   (ceiling 1/0.05 = 20×)
```

The non-parallel 5% (data loading, synchronisation, exposed communication) caps the gain no matter
how many GPUs you add. That's why exercise 12's exposed-communication number matters so much.

### On the exam

**Strong scaling** is a fixed problem on more GPUs (Amdahl limits it). **Weak scaling** grows the problem with
the GPUs, e.g. a constant per-GPU batch, which is how data parallelism usually scales. *8 → 64 GPUs gives only
3× more throughput:* profile first (Nsight Systems), then fix the fabric, overlap, or input pipeline.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise / section |
|---|---|
| Choosing a distributed configuration | "Which parallelism for which problem", 1, 4 |
| ZeRO stages, FSDP | 1, 9 |
| Where to put TP and PP, interconnect | 4, 5 |
| Pipeline bubble | 2 |
| NCCL collectives | 3, 9 |
| DDP semantics, torchrun | 6, 11 |
| Scaling efficiency | 11, 13 |
| MFU | 7 |
| Roofline | 8 |
| Sequence / context parallelism | 10 |
| Profiling workflow, Nsight Compute, NVTX, GPU monitoring | 12 |
| Mixed precision on Hopper | Lab 05, exercise 15 |
