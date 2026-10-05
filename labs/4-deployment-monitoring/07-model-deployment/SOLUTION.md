# Lab 07 — Solution walkthrough

For each task: **what you should see**, **why**, and **on the exam** (the question shape it prepares you
for, with why the wrong answers are wrong). Your exact numbers depend on the GPU, driver and image
versions; the *direction* of every effect below is what matters.

```
client ──HTTP/gRPC──► K8s Service ──► Pod ──► Triton ──► dynamic batcher ──► model instances ──► GPU
                       (K2–K6)                (1)          (2)                (3, 4, 5, 6, 7)
LLMs:  client ──OpenAI API──► trtllm-serve / NIM ──► in-flight batching + paged KV cache   (8, 9)
```

---

## Task 1 — Explore a running Triton

**What you see.** `/v2` lists the server version and extensions (`model_repository`, `statistics`,
`schedule_policy`, ...). `live` and `ready` both return `200`. `make models` shows `text_classifier`
`READY` and the other models `UNAVAILABLE` with reason *unloaded*: the server runs in **explicit
model-control mode** and only loaded what `--load-model` named. `make config` shows fields you never
wrote, for example `instance_group[0].gpus: [0]`, `instance_group[0].name`, `optimization`,
`default_model_filename`, `dynamic_batching.priority_levels`: Triton **auto-completes** the config.

**Why.**
- **live** = the process is up (restart it if not). **ready** = every loaded model can serve (send
  traffic only if so). That maps one-to-one onto Kubernetes liveness and readiness probes (K2).
- The request in step 4 carries 2 texts because `max_batch_size: 16` makes the first dimension a
  **batch** dimension: shape `[2, 1]` = 2 requests' worth of `dims: [1]`.
- Average batch size = `inference_count / execution_count` (inferences per model execution).
- Port **8002** serves Prometheus metrics; 8000 is HTTP and 8001 gRPC.

**On the exam.** *"Prometheus shows no Triton data."* → scrape **:8002/metrics**. Not 8000 (HTTP
inference), not 8001 (gRPC), and not `/v2/health/ready` (a probe, not metrics).

## Task 2 — Dynamic batching vs P95 latency

**What you see** (typical pattern on an L4):

| Run | avg batch | c=1 throughput · P95 | c=16 throughput · P95 |
|---|---|---|---|
| A no batching | 1.0 | baseline · lowest | flat: barely above c=1 · P95 grows with concurrency |
| B batching, no delay | 1 at c=1, several at c=16 | ≈ A | **2–4× A** · P95 well below A |
| C 2 ms delay | slightly larger | ≈ A, P95 +≈2 ms | ≈ B or a bit better |
| D 50 ms delay | large | **≈ 20/s** · P95 ≈ 50 ms + compute | ≈ C, P95 much higher |

**Why.**
- **A**: every request runs alone, so the GPU executes 16 tiny batches one after another; queue
  time grows linearly with concurrency. `max_batch_size` only *allows* batching; without
  `dynamic_batching` nobody combines requests.
- **B**: with no delay, the batcher takes **whatever is already waiting** when an instance is free.
  Under load the queue fills while the GPU works, so batches form for free.
- **C**: waiting up to 2 ms collects slightly fuller batches at moderate load and costs at most 2 ms.
- **D**: at concurrency 1 there is nobody to batch with, so every request waits the full 50 ms for
  companions who never come. **The queue delay adds straight to latency at low load.**
- For a **20 ms P95** SLO: C (or B). Budget `queue delay + compute(batch) + network ≤ SLO`.
  Calculation 1 reproduces exactly this: at 400 req/s, batch ≤ 8 with a 2 ms delay gives P95 ≈ 17 ms;
  a 20 ms delay alone breaks the SLO.

**On the exam.** *"P95 ≤ 20 ms, compute 7 ms at batch 1, 12 ms at batch 8: which config?"* →
`preferred_batch_size: [4, 8]`, `max_queue_delay_microseconds: 2000`.
- `max_queue_delay_microseconds: 50000` → the wait alone is 50 ms.
- `preferred_batch_size: [32]` → waits for batches that rarely fill, and 32 may exceed `max_batch_size`.
- Disabling batching → throughput collapses under load and the queue makes P95 worse, not better.

## Task 3 — Instance groups

**What you see.** 2 instances: throughput up noticeably at high concurrency, P95 down; 3 instances: a
smaller gain or none. Each instance adds a full copy of the weights (`nvidia-smi` memory grows per
instance: ~150–300 MB for DistilBERT in FP16 plus the CUDA context). The logs show
`text_classifier_0_0 on cuda:0`, `text_classifier_0_1 on cuda:0`. `KIND_CPU`: much slower (DistilBERT
on a few vCPUs), and only because `model.py` reads `args["model_instance_kind"]`; a Python model that
ignores it would silently keep using the GPU.

**Why.** One instance executes one batch at a time. While it tokenizes on the CPU, copies inputs or
splits outputs, the GPU idles. A second instance **overlaps** its CPU work and transfers with the first
instance's GPU compute. Once the GPU is busy all the time (utilisation near 100%), more instances add
only memory and contention.

**On the exam.** *"GPU utilisation is 40% at peak while requests queue."* → increase
`instance_group.count` (and/or enable dynamic batching). Adding replicas on more GPUs costs money
before the current GPU is saturated; raising `max_queue_delay` adds latency; a bigger GPU doesn't fix
an idle GPU.

## Task 4 — ONNX Runtime vs TensorRT

**What you see** (typical order, highest throughput first): **TRT fp16** > ORT + TensorRT EP fp16 ≈ TRT
fp16 > TRT fp32 > ORT fp32 > Python/PyTorch `text_classifier` (which also pays for tokenization in
Python). FP16 is often **2–3×** FP32 for a BERT-class model on an L4. The `MAX_BS=8` engine fails to
load: Triton reports that the config's `max_batch_size` (16) exceeds what the engine's optimization
profile supports (8). The ORT + TensorRT EP load takes much longer the first time (TensorRT builds the
engine inside Triton).

**Why.**
- **TensorRT** fuses layers (attention, GELU, layer norm), picks the fastest kernels for *this* GPU and
  runs FP16 on **tensor cores**. ONNX Runtime's CUDA provider executes the graph more op by op.
- **Optimization profile** = the min/opt/max input shapes the engine was tuned for. The engine is
  fastest near `opt`; a shape outside min/max is rejected. Triton's `max_batch_size` must fit in it.
- An engine is **specific to the GPU architecture and TensorRT version** it was built with: an L4
  (sm_89) engine won't run on a T4 (sm_75) or an H100 (sm_90). Build per target GPU (in CI, or let NIM
  and TensorRT-LLM pick a matching prebuilt engine).
- **INT8 needs calibration** (or a quantization-aware-trained model) to choose per-tensor scales. A
  `--int8` build without calibration data runs fast with placeholder scales, so its outputs can't be
  trusted. Check accuracy on a validation set before shipping any lower precision.
- The **TensorRT execution provider** for ONNX Runtime is the low-effort path: one config block, no
  engine management, but a slow first load and less control than a native TensorRT engine.

**On the exam.** *"Reduce latency of an ONNX model on Triton with minimal effort."* → enable the
TensorRT accelerator in `optimization { execution_accelerators }` (or convert to a TensorRT plan).
Retraining, adding CPU instances, or raising the queue delay don't make the model execute faster.

## Task 5 — Ensemble

**What you see.** The ensemble returns the same labels as `text_classifier`. With the TensorRT model in
the middle it usually beats the Python-backend model on throughput. `make stats` per step shows the
**tokenizer** (Python, CPU) taking a large share; raising it to 4 CPU instances often raises ensemble
throughput until the TensorRT step or CPU cores become the limit.

**Why.**
- `input_map` / `output_map` keys are **each model's own tensor names**; values are the **ensemble's
  internal tensor names** that wire step outputs to the next step's inputs. Tensors stay in Triton's
  memory: no client round-trip between steps.
- Tokenization is string processing: CPU work. Keep GPU instances for the model.
- An ensemble is a **static DAG**. Loops, conditions or calling different models depending on the input
  need **BLS** (a Python model that calls other models through `pb_utils.InferenceRequest`).

**On the exam.** *"Pre-processing in each client adds a network hop and inconsistent tokenization."* →
a Triton **ensemble** (or BLS). A separate microservice still adds a hop; doing it in the client is
the problem; a bigger GPU doesn't remove the hop.

## Task 6 — Versions, model control, warm-up

**What you see.** `all: {}` serves v1 and v2 side by side, each callable at `/v2/models/<m>/versions/<n>`.
`specific: [1]` unloads v2. `latest: {num_versions: 1}` serves only v2 (the highest number).
`first-request` without warm-up: the first request is noticeably slower than warm ones (lazy CUDA and
TensorRT initialisation, memory allocation). With `model_warmup`, the **load** takes longer and the first
request is about as fast as a warm one: Triton runs the warm-up requests **before** marking the model READY.
Unloading the ensemble unloads composing models that nothing else uses.

**Why / modes.**
- `none` (default): load everything at start, never change. `poll`: watch the repository and apply
  changes automatically (convenient, risky in production: a half-copied file gets loaded). `explicit`:
  load/unload only through the API, the controlled choice for production and Kubernetes.
- Warm-up trades a slower, blocked load (readiness stays false longer) for no cold first request.

**On the exam.** *"Latency spikes for the first requests after each deployment."* → `model_warmup` in
config.pbtxt (and a readiness probe so traffic waits). A larger `max_queue_delay` makes it worse;
more replicas each have the same cold start.

## Task 7 — Model Analyzer

**What you see.** It varies `max_batch_size`, `instance_group.count`, dynamic batching and client
concurrency, runs perf_analyzer on each, and writes a summary ranking configs that meet `p99 ≤ 30 ms`
by throughput, plus the winning `config.pbtxt` files. Its winner is usually at least as good as your
best manual `distilbert_onnx` run, and found without guessing.

**Why.** The best config depends on the model, GPU and traffic, and the search space multiplies.
**Filter by constraints first, then maximise the objective**; the unconstrained winner almost always
breaks the latency budget (calculation 3).

**On the exam.** *"Find the config with maximum throughput under a latency budget across batch sizes
and instance counts."* → **Model Analyzer**. perf_analyzer measures one config at a time; DCGM measures
the GPU, not configs; Nsight Systems profiles kernels.

## Task 8 — One LLM, four servers

**What you see.**
1. The plan for Qwen2.5-1.5B on an L4 at `FRACTION=0.9`: weights 2.88 GiB, KV cache ~16.4 GiB (vLLM) or
   ~16.8 GiB (TensorRT-LLM), ~610,000–630,000 tokens, ~240 sequences of 2,560 tokens. 64 users fit easily.
2. The server's own log reports a KV cache close to your number. Differences of a few percent come from the
   overhead guess (activations, CUDA graphs, the CUDA context), which the real server measures by running a
   dummy forward pass at start-up.
3. At concurrency 1, TTFT for a 512-token prompt is tens of ms (the prediction is ~26 ms plus HTTP and
   tokenization) and TPOT is ~15 ms. At 32, TPOT rises only modestly while output tokens/s rises by an
   order of magnitude, and TTFT p95 grows (prefills of new requests compete with the running decodes).
4. TensorRT-LLM with the same 0.9 allocates a slightly **larger** cache: its fraction applies to the memory
   left after the weights, vLLM's to the whole GPU.
5. Behind Triton the numbers are close to the direct servers; the extra cost is the Python backend and the
   generate-endpoint wrapping (a few ms per request), not the engine.
6. Knobs: `PROMPT=8192` cuts the sequences that fit about 3.5×; `FRACTION=0.3` leaves ~2.9 GiB = ~107,000 KV
   tokens, less than 64 requests × 2,512 tokens, so requests wait (`num_requests_waiting` > 0) and TTFT p95 jumps;
   `KVBITS=8` doubles the KV tokens; `CHUNK=256` gives `max_num_tokens` 512, so a 2,000-token prompt is spread
   over several iterations and TTFT multiplies; `USERS=4` caps running sequences at 4, so at concurrency 32 most of the latency is
   queueing; FP8 weights halve the weight memory and speed up decoding; Qwen2.5-7B (14.2 GiB) still fits
   but with far fewer sequences.

**Why.**
- **Server config vs request.** Sizes and scheduling (`max_seq_len`/`max_model_len`, `max_batch_size`/
  `max_num_seqs`, `max_num_tokens`/`max_num_batched_tokens`, KV memory, block size, chunked prefill) are
  fixed when the server starts. Sampling (`temperature`, `top_p`, `top_k`, penalties, `max_tokens`, stop) is
  per request, with defaults from `generation_config.json` (vLLM applies them; check what your server does).
- **TTFT** = waiting + prefill (compute-bound: ~2 × parameters FLOPs per prompt token). **TPOT** = one decode
  step (memory-bound: read all weights once per step, shared by the whole batch, plus every sequence's KV
  cache). That is why batching multiplies throughput almost for free, until the KV reads dominate or the
  cache is full.
- **Three limits on concurrency:** `max_batch_size` (configured), KV-cache capacity (memory), and the
  per-iteration token budget (`max_num_tokens`). The lowest one wins.
- **Triton wrapping.** For LLM backends Triton's own batcher is **off** (`max_batch_size: 0` in
  `triton_config`): the engine does in-flight batching. `decoupled: True` lets one request return many
  responses (token streaming). The vLLM backend uses `KIND_MODEL`: the model places itself on the GPU(s).

**On the exam.** *"Long prompts make TTFT spike for everyone while decoding is fine."* → enable chunked
prefill / tune the token budget (`max_num_tokens`). *"The server rejects requests with long prompts plus
`max_tokens`."* → `max_model_len`/`max_seq_len` too small (or lower `max_tokens`). *"Requests queue while
GPU memory shows free space reserved."* → the KV-cache fraction or `max_num_seqs` limits concurrency.
Raising temperature or `top_p` never changes throughput or memory.

## Task 9 — NIM

**What you see.** `list-model-profiles` lists profiles such as `tensorrt_llm-l4-fp16-tp1-throughput`,
`...-latency`, and `vllm-...` fallbacks, marking which are compatible with the detected GPU. At start the
NIM logs the profile it selected (a TensorRT-LLM profile matching the L4 if one exists, else vLLM). The API
is OpenAI-compatible; `/v1/metrics` exposes Prometheus metrics (TTFT, KV-cache usage, request counts).

**Why / when.**

| Option | Choose it when |
|---|---|
| **NIM** | you want a supported, optimised endpoint quickly; NVIDIA builds and validates the engines per GPU (needs NGC access / NVIDIA AI Enterprise in production) |
| **Triton + TensorRT-LLM backend** | many models and frameworks on one server, ensembles, Triton's batching and metrics everywhere |
| **`trtllm-serve`** | a single LLM with full control over TensorRT-LLM settings and an OpenAI API |

`NIM_MODEL_PROFILE` pins the engine profile instead of auto-selection (reproducible deployments). The model
cache (`/opt/nim/.cache`) is mounted so restarts and new pods don't download weights again. Pin the image
version: `:latest` changes under you.

**On the exam.** *"Deploy an optimised Llama endpoint quickly, OpenAI-compatible, on the team's GPUs."* →
**NIM**. Writing a FastAPI wrapper around transformers isn't optimised; building TensorRT-LLM engines by
hand is the slow path; a hosted API breaks "on our GPUs".

---

## K1 — A GPU node

`nvidia.com/gpu: 1` is an **extended resource** advertised by the **NVIDIA device plugin** (a DaemonSet);
without it Kubernetes knows nothing about GPUs. GPU feature discovery adds labels such as
`nvidia.com/gpu.product=NVIDIA-L4` and `nvidia.com/gpu.memory=23034`: use
`nodeSelector: {nvidia.com/gpu.product: NVIDIA-L4}` (or node affinity) to target a GPU type. The **GPU
Operator** installs and upgrades all of this, driver included, on every GPU node.

## K2 — Probes vs a slow load

Without a `startupProbe`, the liveness probe starts immediately. Triton only opens its HTTP port after
the initial model load, so liveness gets *connection refused* three times → the kubelet kills the
container → restart → same again → `CrashLoopBackOff`. A startup probe (`failureThreshold × periodSeconds`
longer than the worst load time) disables liveness and readiness until it succeeds once.

| Probe | Question it answers | On failure |
|---|---|---|
| startup | has it finished starting? | keep waiting, then restart |
| readiness | can it take traffic now? | removed from Service endpoints (no restart) |
| liveness | is it hung? | container restarted |

**On the exam.** *"LLM pods restart in a loop during the 6-minute model load."* → add a `startupProbe`
(or lengthen it). Raising memory limits, adding replicas or removing the readiness probe don't stop the
liveness kills.

## K3 — Scheduling and time-slicing

The second pod was `Pending` with `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`. GPUs are
whole, non-shareable units by default: a pod can't request `0.5` (integer only). Time-slicing makes the
device plugin advertise N "replicas" of each GPU; pods take turns on the GPU **with no memory limit and no
fault isolation** between them: one pod can exhaust memory for all. **MIG** (A100, H100, H200, B200)
partitions the GPU into isolated instances with their own memory and compute (`nvidia.com/mig-1g.10gb`).

**On the exam.** *"Several small inference models each use 10% of an A100; isolation required."* → MIG.
Time-slicing doesn't isolate; one pod per GPU wastes 90%; MPS shares compute but its isolation is limited.

## K4 — Autoscaling on queue time

CPU stays low (Triton's threads wait on the GPU) while queue time climbs. A CPU-based HPA never scales.
The HPA reads `triton_avg_queue_us` per pod through the custom-metrics API (prometheus-adapter →
Prometheus → Triton :8002), scales up within seconds, and scales down only after the stabilization window
(GPU pods are slow and expensive to start, so flapping is costly). `maxReplicas` is capped by the GPUs
the node can allocate (4 time-sliced here). Little's law (calculation 2) sizes the steady state.

**On the exam.** *"HPA on CPU never scales the Triton deployment while latency breaks the SLO."* → scale on
a custom metric: queue time or pending requests (via Prometheus Adapter or KEDA). GPU utilisation alone is
weaker: it saturates at 100% and doesn't say how many requests wait.

## K5 — A rolling update that hangs

`maxSurge: 1, maxUnavailable: 0` means "start the new pod first, never drop below 1 ready". The new pod
needs a GPU; the only one is used by the old pod → `Pending` forever (calculation 4: max pods = 2, but the
node can run 1). Fix A (`maxSurge: 0, maxUnavailable: 1`) kills the old pod first: the rollout completes with
**downtime** (0 ready during the swap). Fix B keeps capacity but needs a spare GPU. `kubectl rollout undo`
returns to the previous ReplicaSet.

**On the exam.** *"Rollout stuck, new pod Pending, Insufficient nvidia.com/gpu, must keep capacity."* →
provide spare GPU capacity for the surge (or roll out to a new node pool / blue-green). `Recreate` and
`maxUnavailable` > 0 drop capacity; deleting the old pod by hand is the same downtime with extra steps.

## K6 — Canary

Kubernetes Services balance across **ready endpoints**, so 3 v1 + 1 v2 pods ≈ 25% to v2. The canary check
compares both versions **over the same window** and returns `rollback` (v2 is ~40 ms slower per batch).
For an exact or header-based split (5%, internal users first) you need a service mesh or gateway (Istio,
Gateway API weights) or a router in front. Canary needs one extra pod; **blue-green** needs a full second
set of GPUs during the switch; **shadow** duplicates traffic to the new version and discards its answers.

---

## Calculations

### 1 — `simulate_dynamic_batching`: why the queue delay matters

This is the exam question *"P95 ≤ 20 ms; compute 7 ms at batch 1 and ~12 ms at batch 8: which dynamic
batching config?"*, answered numerically:

```
launch = max(GPU free, min(oldest arrival + max_delay, time the batch would be full))
```

At 400 requests/s, compute = 6.3 + 0.7 × batch ms:

| config | P50 | P95 | verdict |
|---|---|---|---|
| no batching | 905 ms | 1,713 ms | ✗ one request at a time manages 143/s < 400/s: the queue grows forever |
| batch ≤ 8, delay 2 ms | 13.3 | 17.4 | ✓ the exam's answer |
| batch ≤ 8, delay 20 ms | 20.6 | 29.4 | ✗ the wait alone uses up the SLO |

Batching makes the load possible (8 requests in 11.9 ms ≈ 670 requests/s of capacity), and the delay adds
straight to latency.

### 2 — `replicas_needed`: Little's law

```
in-flight requests = arrival rate × latency:   200 req/s × 0.5 s = 100
per replica 16 concurrent → 100/16 = 6.25 → round UP → 7 replicas   (8 with 20% headroom)
```

### 3 — `pick_best_config`: what Model Analyzer decides

```
bs8_i1_d2ms    640 req/s  P95 17.4 ms  3 GB
bs8_i2_d1ms    780 req/s  P95 19.1 ms  6 GB     ← best with P95 ≤ 20 ms and ≤ 8 GB
bs16_i2_d5ms   900 req/s  P95 24.0 ms  6 GB     ← best if the SLO were 40 ms
bs32_i4       1100 req/s  P95 31.0 ms  14 GB    ← too much memory
```

Filter first, then maximise.

### 4 — `rollout_bounds`: capacity during an update

```
max total pods = replicas + maxSurge          ← for GPU pods, the surge needs FREE GPUs
min ready pods = replicas − maxUnavailable    ← the capacity you keep serving with

3 replicas, maxSurge 1, maxUnavailable 0 → up to 4 pods (1 spare GPU), never below 3 ready
4 replicas, maxSurge 0, maxUnavailable 1 → no spare GPU needed, temporarily 3 ready
```

### 5 — `rerank_top_n`: precision for RAG

A fast **bi-encoder** retriever finds the right passages among many (**high recall**) but ranks them
loosely (**low precision**). A **cross-encoder reranker** reads each (query, passage) pair together: far
more accurate, too slow for the whole corpus. Retrieve many, rerank, keep the top few.

| *RAG service cites irrelevant passages; no re-ingestion allowed* | verdict |
|---|---|
| Add a cross-encoder reranking stage (e.g. a NeMo Retriever reranking NIM) over a larger candidate set | ✅ a query-time change |
| Re-chunk and re-embed the corpus | ❌ that is re-ingestion |
| Send top-50 passages to the LLM | ❌ lowers precision further |
| Raise the temperature | ❌ doesn't fix retrieval |

### Part B — serving one LLM (calculations 6–19)

Worked for the lab scenario: **Qwen2.5-1.5B-Instruct on one L4**, prompts ≤ 2,048, answers ≤ 512, 64 users.

**6 · `model_spec` — reading config.json.** `hidden_size 1536`, `num_attention_heads 12`,
`num_key_value_heads 2` (**GQA**: 6 query heads share each K/V head), `head_dim = 1536/12 = 128`, 28 layers,
`intermediate_size 8960`, `vocab_size 151936`, `tie_word_embeddings true`.

```
per layer: Q,O 2·1536·12·128 = 4.72 M   K,V 2·1536·2·128 = 0.79 M   MLP 3·1536·8960 = 41.29 M
28 layers = 1,310 M  + embeddings 151,936·1,536 = 233 M (tied: counted once)  = 1,543,569,408 ≈ 1.54 B
```

Qwen2.5-7B is untied (embeddings counted twice) → 7.62 B; Llama-3.1-8B → 8.03 B. All match the model cards.

**7, 8 · weights and KV per token.**

```
weights FP16: 1.5436e9 × 2 B = 2.875 GiB      INT4: 0.72 GiB
KV per token: 2 (K,V) × 28 layers × 2 KV heads × 128 × 2 B = 28,672 B = 28 KiB
   without GQA (12 KV heads): 168 KiB   ·   FP8 KV cache: 14 KiB   ·   Llama-3.1-8B: 128 KiB
```

**9 · `kv_cache_gib` — the same 0.9 means different things.**

| server | parameter | formula | Qwen 1.5B, 0.9 |
|---|---|---|---|
| TensorRT-LLM | `kv_cache_free_gpu_memory_fraction` | (22.5 − 2.875 − 1) × 0.9 | **16.76 GiB** |
| vLLM | `gpu_memory_utilization` | 22.5 × 0.9 − 2.875 − 1 | **16.38 GiB** |

At 0.1, TensorRT-LLM still gets 1.86 GiB of cache; vLLM gets nothing, because 10% of the GPU can't even hold the
weights. (vLLM's value is also why two vLLM servers can share a GPU at 0.45 each.)

**10 · `max_sequences` — paged KV cache.** Blocks of 32 tokens (TensorRT-LLM default; vLLM uses 16):
16.76 GiB ÷ (32 × 28 KiB) = 19,615 blocks; a 2,560-token sequence needs 80 blocks → **245 sequences**.
A 1,000-token sequence still occupies 32 whole blocks (1,024 tokens). Paging wastes at most one block per
sequence, instead of reserving `max_seq_len` for every request as a contiguous cache would.

**11 · `prefill_iterations` — the token budget and chunked prefill.** Each iteration processes at most
`max_num_tokens` tokens; 48 decoding sequences use 48 of them. With 2,048, a 4,001-token prompt needs 3
iterations with chunking. Without chunking it can't be scheduled at all, because the whole prompt must fit
one iteration. Chunking also keeps decode steps short, which is why it lowers TTFT and ITL spikes under
mixed traffic.

**12 · `estimate_latency` — prefill vs decode.**

```
prefill 512 tokens: 2 × 1.54e9 × 512 / (121e12 × 0.5) = 26 ms           ← compute-bound
decode, batch 1:   (3.09 GB weights + 18 MB KV) / (300 GB/s × 0.7) = 14.8 ms → 68 tok/s
decode, batch 32:  (3.09 GB + 587 MB KV)        / 210 GB/s        = 17.5 ms → 1,829 tok/s
```

32× the throughput for +18% step time: the weights are read once per step for the whole batch. At batch 64
with 2,560-token contexts the KV reads (4.7 GB) exceed the weights and the step grows to ~37 ms: long
contexts make decoding slower, not just bigger.

**13 · `plan_llm_server`.** `max_seq_len` 2,560; 245 sequences fit, so `max_batch_size = min(64, 245) = 64`;
`max_num_tokens = ⌈(64 + 2,048)/256⌉ × 256 = 2,304`. Llama-3.1-8B with 5k-token contexts: FP16 weights leave
room for only **9** sequences (warning: 32 users queue); INT8 → 20; INT4 → 25. Quantizing weights buys
concurrency through KV-cache memory.

**14–17 · rendering.** Same plan, four syntaxes:

| setting | trtllm-serve | vLLM | Triton + vLLM `model.json` | Triton + TRT-LLM `model.yaml` |
|---|---|---|---|---|
| context | `--max_seq_len` | `--max-model-len` | `max_model_len` | `max_seq_len` |
| running sequences | `--max_batch_size` | `--max-num-seqs` | `max_num_seqs` | `max_batch_size` |
| tokens per iteration | `--max_num_tokens` | `--max-num-batched-tokens` | `max_num_batched_tokens` | `max_num_tokens` |
| KV memory | `--kv_cache_free_gpu_memory_fraction` | `--gpu-memory-utilization` | `gpu_memory_utilization` | `kv_cache_config.free_gpu_memory_fraction` |
| chunked prefill | `enable_chunked_prefill` (extra YAML) | `--enable-chunked-prefill` | `enable_chunked_prefill` | `enable_chunked_prefill` |
| KV dtype | `kv_cache_config.dtype` | `--kv-cache-dtype` | `kv_cache_dtype` | `kv_cache_config.dtype` |

Triton adds its own layer: `config.pbtxt` (`backend: "vllm"` + `KIND_MODEL`, or the Python LLM-API model)
and, for TensorRT-LLM, `triton_config: {max_batch_size: 0, decoupled: true}`.

**18 · `effective_sampling`.** Qwen's `generation_config.json` says temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.1. A request that only sets `max_tokens` runs with **those**, not with 1.0/1.0. A request
with `temperature: 0` becomes greedy: top_p and top_k no longer matter, repetition_penalty still does.
This is why "the same prompt gives different answers on two servers" is often a defaults difference.

**19 · `generate_request`.** The same greedy 32-token request:

```
OpenAI (trtllm-serve, vLLM, NIM)  POST /v1/completions   {"model", "prompt", "max_tokens": 32, "temperature": 0, ...}
Triton + vLLM                     POST /v2/models/llm/generate_stream  {"text_input", "stream": true, "parameters": {"max_tokens": 32, "temperature": 0, "top_k": -1, ...}}
Triton + TensorRT-LLM (LLM API)   POST /v2/models/llm/generate_stream  {"text_input", "streaming": true, "sampling_param_max_tokens": 32, "sampling_param_top_k": 1}
```

And `prompt_tokens + max_tokens > max_model_len` is rejected by every server: size `max_seq_len` for the
longest prompt **plus** the longest answer.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Task / calculation |
|---|---|
| Triton dynamic batching | task 2, calculation 1 |
| Instance groups | task 3 |
| TensorRT engines (portability, optimization profiles, ORT accelerator) | task 4 |
| Model control | task 6 |
| TensorRT-LLM, vLLM, LLM serving parameters (KV cache, batching, prefill, context, sampling) | task 8, calculations 6–19 |
| Triton ports, Triton protocol | task 1 |
| Serving LLMs with Triton, streaming | tasks 8, 9, calculations 16, 17, 19 |
| LLM serving memory, KV cache sizing | calculations 6–10, 13 |
| Chunked prefill, LLM serving limits | calculations 11, 13, 19 |
| Sampling defaults | calculation 18 |
| Ensembles | task 5 |
| Model Analyzer | task 7, calculation 3 |
| Model warm-up, stateful models | task 6 (stateful models: sequence batcher, README exam traps) |
| NIM, NIM API | task 9 |
| Container images | tasks 1, 9 (pinned NGC tags) |
| Kubernetes GPU scheduling, GPU Operator, MIG | K1, K3 |
| Startup probes, health probes | K2, task 6 |
| GPU sharing (time-slicing vs MIG) | K3 |
| Autoscaling, capacity planning | K4, calculation 2 |
| Rolling updates, blue-green | K5, K6, calculation 4 |
| RAG relevance, reranking | calculation 5 |
