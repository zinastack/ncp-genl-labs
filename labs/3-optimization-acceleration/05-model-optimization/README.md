# Lab 05 — Model Optimization (17% of exam, the largest domain)

> Blueprint: *memory optimization, batch optimization, performance tuning, efficiency improvements.*

This domain is about knowing **where memory and time go** and which lever moves which number.
You will write the memory calculators, implement INT8 quantization (per-tensor vs per-channel),
a dynamic loss scaler, gradient accumulation, the distillation loss and 2:4 structured sparsity,
and simulate the batching and decoding tricks behind TensorRT-LLM and NIM.

```
make test-05          # YOUR exercises (run from the repo root)
make solutions-05     # reference solutions
make quiz-05          # exam-style questions
make gpu-05           # GPU part (gpu_lab.py) on the Brev/AWS instance
```

---

## 1. Where the memory goes

**Training (mixed precision + Adam), per parameter:**

| Item | Bytes | Notes |
|---|---|---|
| bf16/fp16 weights | 2 | used in forward/backward |
| bf16/fp16 gradients | 2 | |
| fp32 master weights | 4 | optimizer updates these |
| Adam momentum `m` | 4 | fp32 |
| Adam variance `v` | 4 | fp32 |
| **Total** | **16 B/param** | 7B model → **112 GB** before activations |

**Activations** scale with `batch × seq_len × hidden × layers`. Reduce them with **activation
(gradient) checkpointing**: store only some layer inputs and recompute the rest in the backward
pass, which costs about 30% more compute for large memory savings. Also use **FlashAttention**
(no n² score matrix) and a smaller micro-batch combined with **gradient accumulation**.

**LoRA** adds only 2 B/param for frozen weights plus 16 B per *trainable* param. **QLoRA** uses about 0.5 B/param for weights.

**Inference:** weights (`params × bytes`) + **KV cache** (Lab 01 formula) + activations/workspace.
KV cache usually limits batch size and context length.

## 2. Precision and quantization

| Format | Bits | Range / notes | Where |
|---|---|---|---|
| FP32 | 32 | full | master weights, optimizer |
| TF32 | 19 (in fp32 storage) | fp32 range, 10-bit mantissa; automatic on Ampere+ tensor cores | matmuls |
| FP16 | 16 | max **65,504**; small range, so it needs **loss scaling** | V100 era |
| **BF16** | 16 | **same exponent range as FP32**, less precision; no loss scaling needed | A100/H100 default |
| **FP8** (E4M3 / E5M2) | 8 | Hopper/Ada via **Transformer Engine**, with per-tensor scaling; E4M3 for forward, E5M2 for gradients | H100 training and inference |
| INT8 | 8 | W8A8 needs activation smoothing (**SmoothQuant**) because of outliers | inference |
| INT4 / FP4 (NVFP4) | 4 | weight-only **AWQ** / **GPTQ**; NVFP4 on Blackwell | memory-bound decode |

- **PTQ** (post-training quantization) uses a small calibration set and no retraining. It is fast,
  with some accuracy risk. **QAT** (quantization-aware training) simulates quantization during
  training and gives the best accuracy at low bit widths.
- **Per-channel** (or per-group, e.g. 128) scales beat per-tensor scales because one outlier
  channel no longer blows up everyone's step size.
- **Weight-only quantization** speeds up *memory-bound* decode (fewer bytes to read). It doesn't
  help compute-bound prefill much.
- **KV-cache quantization** (FP8/INT8 KV) doubles the tokens or batch you can fit.
- NVIDIA tools: **TensorRT Model Optimizer** (`modelopt`: PTQ, QAT, sparsity, distillation) and **TensorRT-LLM** engines.

## 3. Making models smaller

- **Knowledge distillation:** the student matches the teacher's softened distribution:
  `L = α·T²·KL(softmax(t/T) ‖ softmax(s/T)) + (1−α)·CE(s, y)`. The `T²` keeps gradient scale constant.
- **Pruning:** unstructured (sparse weights, hard to accelerate); **structured** (remove heads,
  channels, layers); **2:4 semi-structured** (2 of every 4 weights are zero), which
  Ampere+/Hopper **sparse tensor cores** accelerate by up to 2× on matmuls.
- **NVIDIA Minitron:** prune width and depth of a large model, then distil from it. This gives
  better small models at a fraction of the training cost.

## 4. Inference performance: the two phases

| Phase | What | Bound by | Key metric |
|---|---|---|---|
| **Prefill** | process the whole prompt in parallel and build the KV cache | **compute** | **TTFT** (time to first token) |
| **Decode** | one token at a time per sequence, reading all weights and the KV cache | **memory bandwidth** | **ITL / TPOT** (inter-token latency), tokens/s |

At batch 1, decode speed ≤ `memory_bandwidth / bytes_of_weights`. For a 70B fp16 model (140 GB)
on an H100 (3.35 TB/s), that is about 24 tokens/s. **Batching** amortises the weight reads over
many sequences, which is why throughput grows with batch size until you become compute-bound or
run out of KV-cache memory.

| Technique | What it fixes |
|---|---|
| **In-flight (continuous) batching** | Static batches wait for the longest sequence. In-flight batching evicts finished sequences and admits new ones every step, for much higher throughput (TensorRT-LLM, vLLM, NIM). |
| **Paged KV cache (PagedAttention)** | Allocates the KV cache in blocks instead of contiguous `max_len` buffers, removing fragmentation and allowing prefix sharing. |
| **Chunked prefill** | Splits long prompts into chunks interleaved with decode steps, which smooths ITL spikes. |
| **Speculative decoding** | A small draft model (or Medusa heads / EAGLE) proposes γ tokens and the big model verifies them in one pass. Output is identical; latency drops by the expected number of accepted tokens. |
| **KV-cache reuse / prefix caching** | Shared system prompts skip prefill. |
| **Kernel fusion, CUDA graphs** | Fewer kernel launches, which matters most at small batch. |
| **FlashAttention / XQA kernels** | Fast, memory-efficient attention. |

**Throughput vs latency:** larger batches raise throughput (tokens/s/GPU) and raise per-request
latency. Choose by SLO. Triton's `max_queue_delay_microseconds` makes the same trade-off.

## 5. Training efficiency checklist

bf16/FP8 mixed precision · FlashAttention · activation checkpointing (selective) · gradient
accumulation to reach the target global batch · fused optimizers (fused AdamW) · sequence packing ·
`torch.compile` / CUDA graphs · overlap of communication and compute (Lab 06) · data loader with
pinned memory and prefetch so the GPU never waits on input.

**Effective (global) batch = micro_batch × grad_accum_steps × data_parallel_size.**

## 6. Exam traps

- BF16 needs no loss scaling (fp32 range). FP16 does (**dynamic loss scaling**: halve on
  overflow and skip the step; grow after N clean steps).
- Gradient accumulation saves memory, **not time**. Time per sample stays the same or gets worse.
- Activation checkpointing trades **compute for memory**.
- Quantization cuts memory and bandwidth, which gives big decode gains. Prefill is compute-bound
  and needs FP8/INT8 *activations* (W8A8) to speed up.
- Speculative decoding doesn't change the output distribution. It helps latency most at **low batch**.
- Continuous batching improves **throughput and GPU utilisation**, not the latency of a single request.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `training_memory_bytes` | 16 B/param rule, LoRA, QLoRA |
| 2 | `max_batch_size` | KV-cache-limited inference batch |
| 3 | `quantize_int8`, `dequantize` | symmetric absmax, per-tensor vs per-channel |
| 4 | `DynamicLossScaler` | fp16 loss scaling |
| 5 | `accumulated_gradients` | gradient accumulation equals the large-batch gradient |
| 6 | `distillation_loss` | KD with temperature |
| 7 | `prune_2_4` | 2:4 semi-structured sparsity |
| 8 | `static_batching_steps`, `inflight_batching_steps` | why continuous batching wins |
| 9 | `speculative_expected_tokens` | draft/verify speed-up |
| 10 | `decode_tokens_per_sec_bound` | memory-bandwidth roofline for decode |
