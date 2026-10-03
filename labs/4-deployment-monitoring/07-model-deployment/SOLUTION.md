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

## Task 8 — TensorRT-LLM

**What you see.**
1. The log reports the KV-cache capacity in tokens. With Qwen2.5-1.5B on an L4 and `KV=0.5`, expect
   on the order of a few hundred thousand tokens: the same arithmetic as `kv_cache_tokens` (28 layers ×
   2 KV heads × 128 × 2 (K,V) × 2 bytes = 28 KiB per token).
2. Concurrency 1 → 8 → 32: output tokens/s rises several-fold; **TPOT barely moves** at first; **TTFT
   rises** with concurrency.
3. `MBS=4` at concurrency 32: throughput capped near the 4-request level, TTFT explodes (requests wait
   for one of 4 slots).
4. `KV=0.05`: the cache holds only a few sequences of 512 + 128 tokens, so the scheduler (guaranteed
   no-evict) admits fewer requests at once: throughput drops and TTFT rises, like a small MBS.
5. `MNT=1024`: at most ~2 prompts of 512 tokens are prefilled per iteration, so under a burst
   prompts wait their turn: TTFT rises while TPOT stays similar.
6. `BACKEND=tensorrt`: an engine build step at start (minutes), then similar or better speed. INT8
   weight-only: the weights take about half the memory (more room for KV cache) and TPOT improves.

**Why.**
- **In-flight (continuous) batching** adds and removes requests *every iteration*, not per batch, so the
  GPU stays full. Decoding is **memory-bandwidth-bound**: reading the weights once serves the whole
  batch, so TPOT stays flat while throughput scales, until compute or KV cache runs out.
- **TTFT** = queueing + prefill. More concurrent users means more prefills compete per iteration.
- **max_batch_size** = concurrent sequences; **max_num_tokens** = tokens processed per iteration
  (prefill chunks + one per decoding sequence); **KV fraction** = cache size = how many tokens can be
  in flight. Each can be the bottleneck.
- On the **TensorRT backend**, `max_batch_size`, `max_num_tokens`, `max_seq_len` and quantization are
  **baked into the engine**: change them and you rebuild. The KV fraction is a runtime setting.
- **Weight-only INT8 (W8A16)**: decode time is dominated by reading weights from memory; half the bytes
  → faster decode and more free memory, with no calibration needed. FP8 (Ada/Hopper) quantizes
  activations too, which needs calibration.

**On the exam.** *"High TTFT at peak, GPU memory mostly used by weights."* → quantize weights (INT8/FP8)
to free memory for KV cache and raise the batch; consider chunked prefill. Increasing `max_batch_size`
alone fails if the KV cache can't hold the extra sequences; static batching makes TTFT worse; more
`max_tokens` per response doesn't change TTFT.

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

### 6 — `kv_cache_tokens`: how many tokens fit

```
bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes
Qwen2.5-1.5B, FP16: 2 × 28 × 2 × 128 × 2 = 28,672 B = 28 KiB
20 GiB free × 0.5 = 10 GiB → 10,737,418,240 / 28,672 = 374,491 tokens ≈ 182 sequences of 2,048 tokens
```

| change | tokens | why |
|---|---|---|
| fraction 0.5 → 0.9 | 674,084 | more memory for the cache |
| FP8 KV cache | 748,982 | half the bytes per element |
| no GQA (12 KV heads instead of 2) | 62,415 | 6× more K/V vectors per token |

This is why GQA, FP8 KV cache and weight quantization all raise the concurrency an LLM server can sustain.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Task / calculation |
|---|---|
| Triton dynamic batching | task 2, calculation 1 |
| Instance groups | task 3 |
| TensorRT engines (portability, optimization profiles, ORT accelerator) | task 4 |
| Model control | task 6 |
| TensorRT-LLM (engine settings, KV cache, weight-only quantization) | task 8, calculation 6 |
| Triton ports, Triton protocol | task 1 |
| Serving LLMs with Triton, streaming | tasks 8, 9 |
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
