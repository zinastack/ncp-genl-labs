# Lab 06 — GPU Acceleration & Distributed Training (14% of exam)

> Blueprint: *multi-GPU setups, distributed training, parallelism techniques, performance profiling.*

You will write the calculators that size a distributed job (ZeRO memory, pipeline bubble,
all-reduce traffic, FLOPs and MFU, roofline), simulate Megatron tensor parallelism in NumPy,
and run a **real multi-process DDP gradient all-reduce with `torchrun`** (gloo on CPU, NCCL on
GPUs). On a 2-GPU Brev instance, `make gpu-06` runs `ddp_check.py`, `allreduce_bench.py` and
`dist_train.py` (DDP scaling, DDP vs FSDP memory) with NCCL, and `make s3-profile-06` captures an
Nsight Systems timeline.

```
make test-06          # YOUR exercises (run from the repo root)
make solutions-06     # reference solutions
make quiz-06          # exam-style questions
make gpu-06           # 2+ GPUs: NCCL check, all-reduce bandwidth, DDP scaling, DDP vs FSDP
make s3-profile-06    # Nsight Systems + PyTorch profiler traces
```

---

## 1. Parallelism strategies

| Strategy | What is split | Communication | Use when |
|---|---|---|---|
| **Data parallel (DDP)** | the batch; a full model copy per GPU | **all-reduce** of gradients each step (overlapped with backward in buckets) | the model fits on one GPU |
| **ZeRO-1 / 2 / 3** (DeepSpeed), **FSDP** (PyTorch) | optimizer states (1), + gradients (2), + parameters (3) sharded across DP ranks | reduce-scatter grads; all-gather params (stage 3 / FSDP) just in time | the model + optimizer don't fit; same code as DP |
| **Tensor parallel (TP)** (Megatron) | individual weight matrices (column/row split) | all-reduce **inside every layer** (2 per block fwd): needs NVLink | huge layers; **keep TP within a node** (TP ≤ 8) |
| **Sequence parallel (SP)** | LayerNorm/dropout activations along the sequence, alongside TP | reduce-scatter / all-gather instead of all-reduce | cuts activation memory with TP |
| **Context parallel (CP)** / ring attention | the sequence across GPUs for attention | K/V passed around a ring | very long contexts (128k+) |
| **Pipeline parallel (PP)** | groups of layers (stages) | point-to-point activations between stages | the model is too deep for one node; **works across nodes** |
| **Expert parallel (EP)** | MoE experts across GPUs | **all-to-all** token routing | MoE models |

**3D parallelism** = TP × PP × DP (e.g. Megatron-LM / NeMo: `tensor_model_parallel_size`,
`pipeline_model_parallel_size`; DP = world / (TP·PP)). The rule of thumb: **TP inside the node
(NVLink), PP across nodes, DP outermost**.

### ZeRO memory per GPU (Ψ params, N GPUs, mixed-precision Adam = 2+2+12 bytes)

| Stage | Bytes per GPU |
|---|---|
| 0 (DDP) | `16Ψ` |
| 1 (optimizer) | `4Ψ + 12Ψ/N` |
| 2 (+ gradients) | `2Ψ + 14Ψ/N` |
| 3 (+ parameters) = FSDP full shard | `16Ψ/N` |

### Pipeline bubble

With `p` stages and `m` micro-batches (GPipe / 1F1B), idle fraction = `(p−1) / (m + p − 1)`.
**More micro-batches → smaller bubble**; interleaved (virtual) stages divide the bubble by `v`.

## 2. Hardware and communication

- **NVLink / NVSwitch**: GPU↔GPU inside a node (H100 NVLink 4: 900 GB/s per GPU). **PCIe** is
  much slower (~64 GB/s Gen5 x16). **InfiniBand / RoCE** between nodes (200–400 Gb/s per NIC)
  with **GPUDirect RDMA** (NIC ↔ GPU memory without the CPU).
- **NCCL** implements collectives: **all-reduce** (DDP grads), **all-gather** (FSDP params, TP),
  **reduce-scatter** (ZeRO grads), **broadcast**, **all-to-all** (MoE). Ring all-reduce sends
  `2(N−1)/N × size` bytes per GPU, which is bandwidth-optimal and nearly independent of N.
- **Overlap** communication with compute (DDP gradient buckets, FSDP prefetch). Bigger buckets
  mean fewer, larger messages.
- Debugging: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=ALL`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`,
  `NCCL_P2P_DISABLE`; test fabrics with **nccl-tests** (`all_reduce_perf`). Timeouts often mean
  one straggling or crashed rank.
- Launch: `torchrun --nproc_per_node=8 --nnodes=2 --rdzv_endpoint=host:29500 train.py`,
  or SLURM `srun`. Each process gets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.

## 3. Profiling toolbox

| Tool | Level | Answers |
|---|---|---|
| `nvidia-smi` / `nvidia-smi dmon` / **DCGM** | device | utilisation, memory, power, clocks, ECC errors, NVLink traffic |
| **Nsight Systems** (`nsys profile -t cuda,nvtx,osrt,cublas,cudnn -o out python train.py`) | **system timeline** | Where are the gaps? CPU-bound? Data loading? NCCL overlap? Kernel-launch overhead? |
| **Nsight Compute** (`ncu --set full`) | **single kernel** | Is this kernel memory- or compute-bound? Occupancy, warp stalls, roofline |
| **NVTX ranges** (`torch.cuda.nvtx.range_push/pop`) | annotations | label phases (forward/backward/optimizer) in the nsys timeline |
| **PyTorch profiler** (`torch.profiler`, TensorBoard trace) | framework ops | top ops by CUDA time, memory, stack traces |

**Workflow:** start with Nsight Systems to find *where* time goes, then use Nsight Compute on the
hot kernels. Low GPU utilisation with gaps in the timeline usually means the input pipeline,
Python overhead or synchronisation (`.item()`, `.cpu()`) is the problem, not the GPU.

**MFU (model FLOPs utilisation)** = achieved FLOPs / peak. Training FLOPs ≈ `6 × params × tokens`
(2 forward + 4 backward). Good large-scale runs reach 35–55% MFU.

**Roofline:** arithmetic intensity (FLOPs/byte) below the ridge point `peak_FLOPs / bandwidth`
(H100 SXM: ~989 TFLOPS bf16 / 3.35 TB/s ≈ 295) means **memory-bound**. GEMMs with big batch
are compute-bound; decode, element-wise ops and norms are memory-bound, which is why kernel fusion helps.

## 4. Exam traps

- High memory **per GPU** with DDP: switch to ZeRO/FSDP sharding. TP across nodes over Ethernet is a trap; TP needs NVLink.
- Pipeline bubbles fall as **micro-batches increase**, not as stages increase.
- FSDP `FULL_SHARD` ≈ ZeRO-3. `SHARD_GRAD_OP` ≈ ZeRO-2. `HYBRID_SHARD` shards within a node and replicates across nodes.
- DDP all-reduces **gradients**, not weights, and each rank sees a different data shard (DistributedSampler).
- More GPUs don't help if the job is **input-bound** or **communication-bound**. Profile first.
- `CUDA_VISIBLE_DEVICES` controls which GPUs a process sees. With torchrun use `LOCAL_RANK` for `torch.cuda.set_device`.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `zero_bytes_per_gpu` | ZeRO stages 0–3 / FSDP |
| 2 | `pipeline_bubble_fraction` | PP efficiency, micro-batches, interleaving |
| 3 | `ring_allreduce_bytes`, `allreduce_seconds` | NCCL communication cost |
| 4 | `rank_coords`, `tensor_parallel_group` | 3D-parallel rank layout (TP innermost) |
| 5 | `column_parallel`, `row_parallel`, `megatron_mlp` | tensor parallelism with one all-reduce |
| 6 | `average_gradients` | DDP all-reduce with `torch.distributed` (run under torchrun) |
| 7 | `training_flops`, `mfu` | 6·N·D and utilisation |
| 8 | `gemm_arithmetic_intensity`, `is_memory_bound` | roofline analysis |
