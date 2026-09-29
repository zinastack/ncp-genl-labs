# Lab 07 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: a trained model is just files. **Serving** turns it into a reliable network service:
a server that batches requests on the GPU (Triton or NIM), packaged in a container, and scheduled and
scaled by Kubernetes.

```
client ──HTTP/gRPC──► K8s Service ──► Pod (container) ──► Triton / NIM ──► model on GPU
 (6, 9)                (4, 5, 10)                          (1, 2, 3, 7, 8)
                                          RAG: retrieve → rerank (11) → LLM
```

| Component | Role | API |
|---|---|---|
| **TensorRT-LLM** | compiles LLMs into optimised engines (Lab 05) | library |
| **Triton Inference Server** | multi-framework server: dynamic batching, instance groups, ensembles, versions | **KServe v2** (`/v2/models/<name>/infer`); metrics on **:8002** |
| **NIM** | pre-built, optimised model containers from NGC | **OpenAI-compatible** (`/v1/chat/completions`) |
| **Kubernetes + GPU Operator** | schedules GPU pods (`nvidia.com/gpu`), drivers, DCGM exporter | manifests |

---

## Exercise 1 — `triton_config`: telling Triton how to run a model

### Why it exists

Triton needs to know each model's inputs and outputs and **how to batch and replicate it**. One
`config.pbtxt` per model controls the two biggest performance knobs: dynamic batching and instance count.

```protobuf
name: "text_classifier"
backend: "python"
max_batch_size: 16
input  [ { name: "TEXT"  data_type: TYPE_STRING dims: [ 1 ] } ]
output [ { name: "LABEL" data_type: TYPE_STRING dims: [ 1 ] }, { name: "SCORE" data_type: TYPE_FP32 dims: [ 1 ] } ]
dynamic_batching { preferred_batch_size: [ 4, 8 ]  max_queue_delay_microseconds: 2000 }
instance_group [ { count: 2 kind: KIND_GPU } ]
```

| field | meaning |
|---|---|
| `max_batch_size: 16` | Triton may batch up to 16 requests. **0 disables batching** (and `dims` must then include the batch dimension) |
| `dims: [ 1 ]` | the shape of **one** request, without the batch dimension |
| `max_queue_delay_microseconds` | how long a request may wait for others to join its batch; **adds directly to latency** |
| `instance_group count: 2` | two copies of the model run concurrently, overlapping one's CPU work with the other's GPU work |

### On the exam

*A small model reaches only 30% GPU utilisation even at high concurrency, because execution alternates
with host-side pre/post-processing.* **Increase `instance_group` count.** Removing batching or changing the version policy doesn't add concurrency.

---

## Exercise 2 — `validate_model_repository`

### Why it exists

Triton refuses to load a model whose folder is wrong, usually with an unhelpful error. The rules:

```
model_repository/text_classifier/config.pbtxt       ← required; its name: must match the folder
model_repository/text_classifier/1/model.py         ← numeric version folders, not empty
```

`version_policy` chooses which versions are served (latest N, all, or specific ones). With
`--model-control-mode=explicit`, models can be loaded and unloaded through the API for hot swaps.

---

## Exercise 3 — `simulate_dynamic_batching`: why the queue delay matters

### Why it exists

This is your exam question: *"P95 ≤ 20 ms; compute 7 ms at batch 1 and ~12 ms at batch 8: which
dynamic batching config?"* The simulator answers it numerically.

```
launch = max(GPU free, min(oldest arrival + max_delay, time the batch would be full))
```

At 400 requests/s, compute = 6.3 + 0.7 × batch ms:

| config | P50 | P95 | verdict |
|---|---|---|---|
| no batching | 905 ms | 1,713 ms | ✗ one request at a time manages 143/s < 400/s: the queue grows forever |
| batch ≤ 8, delay 2 ms | 13.3 | 17.4 | ✓ the exam's answer |
| batch ≤ 8, delay 20 ms | 20.6 | 29.4 | ✗ the wait alone uses up the SLO |

1. **Batching makes the load possible:** 8 requests in 11.9 ms is about 670 requests/s of capacity.
2. **The delay adds straight to latency.** Budget it: `delay + compute(batch) + network ≤ SLO`.

### On the exam

A large `max_queue_delay_microseconds` wins throughput benchmarks and breaks latency SLOs.
Confirm the chosen config with **perf_analyzer** or **Model Analyzer** (exercise 8).

---

## Exercise 4 — `k8s_deployment` / `k8s_hpa`

### Why it exists

Kubernetes keeps N identical pods running, restarts failed ones and scales them. GPU inference needs a few specific settings:

| field | why |
|---|---|
| `resources.limits: {nvidia.com/gpu: 1}` | the only way to get a GPU: the NVIDIA device plugin advertises this resource |
| `readinessProbe /v2/health/ready` | route traffic only once the model is loaded |
| `livenessProbe /v2/health/live` | restart a hung server |
| `startupProbe` (in `k8s/deployment.yaml`) | give slow-loading models minutes before liveness checks start |
| `RollingUpdate maxUnavailable: 0, maxSurge: 1` | never drop capacity during a rollout (exercise 10) |

**HPA on CPU is the wrong signal** for GPU inference: the GPU saturates while CPU stays low. Scale on
**custom metrics** (Triton queue time, requests in flight, DCGM GPU utilisation) through Prometheus Adapter or KEDA.

### On the exam

*LLM pods killed in a restart loop while loading for 4 minutes:* **add a startupProbe**.
*HPA doesn't react to GPU saturation (Select TWO):* **queue time per request** and **in-flight requests / DCGM GPU utilisation**.

---

## Exercise 5 — `replicas_needed`: Little's law

```
in-flight requests = arrival rate × latency:   200 req/s × 0.5 s = 100
per replica 16 concurrent → 100/16 = 6.25 → round UP → 7 replicas   (8 with 20% headroom)
```

Round up, because 6 replicas would be overloaded. Real sizing adds headroom for spikes and a replica failing.

---

## Exercise 6 — `kserve_infer_request`: talking to Triton

`POST /v2/models/text_classifier/infer` with
`{"inputs": [{"name": "TEXT", "shape": [2, 1], "datatype": "BYTES", "data": [...]}], "outputs": [...]}`.
The shape is the batch plus the config's dims, and strings travel as `BYTES`.

---

## Exercise 7 — `triton_ensemble_config`: a pipeline inside one server

### Why it exists

A real request needs **pre-processing → model → post-processing** (tokenize, run a TensorRT
engine, map logits to a label). As three microservices, every step adds a network hop and
serialisation. A Triton **ensemble** chains the models **inside Triton**, passing tensors in memory.

### How it works

```protobuf
name: "classify_pipeline"
platform: "ensemble"                 ← no backend: Triton's scheduler runs the steps
ensemble_scheduling {
  step [
    { model_name: "tokenizer"   input_map { key: "TEXT"      value: "RAW_TEXT" } output_map { key: "INPUT_IDS" value: "ids" } },
    { model_name: "bert_trt"    input_map { key: "INPUT_IDS" value: "ids" }      output_map { key: "LOGITS"    value: "logits" } },
    { model_name: "postprocess" input_map { key: "LOGITS"    value: "logits" }   output_map { key: "LABEL"     value: "LABEL" } }
  ]
}
```

**Keys are each model's own tensor names; values are the ensemble's internal tensor names**, and a
shared value (`ids`, `logits`) wires one step's output to the next step's input. For control flow
(loops, conditions), use **BLS** (Business Logic Scripting) in the Python backend instead.

### On the exam

*Keep a CPU tokenizer → GPU model → CPU post-processing pipeline inside one server:* **ensemble (or BLS)**.
Sequence batching is for stateful models, version policies pick versions, and rate limiting controls priority.

---

## Exercise 8 — `pick_best_config`: what Model Analyzer decides

### Why it exists

The best batch size, instance count and queue delay depend on the model, the GPU and the traffic,
and guessing is slow. **Model Analyzer** sweeps configurations with **perf_analyzer** and picks the
best one **under your constraints**.

```
bs8_i1_d2ms    640 req/s  P95 17.4 ms  3 GB
bs8_i2_d1ms    780 req/s  P95 19.1 ms  6 GB     ← best with P95 ≤ 20 ms and ≤ 8 GB
bs16_i2_d5ms   900 req/s  P95 24.0 ms  6 GB     ← best if the SLO were 40 ms
bs32_i4       1100 req/s  P95 31.0 ms  14 GB    ← too much memory
```

Filter first, then maximise: the highest throughput overall usually breaks the SLO.

---

## Exercise 9 — `openai_chat_request` / `parse_sse_stream`: talking to NIM, streaming

### Why it exists

NIM exposes the **OpenAI-compatible** API, so any OpenAI client works by changing the base URL.
**Streaming** (`"stream": true`) sends tokens as they're generated, so users see text after the
**TTFT** instead of after the whole answer. Perceived latency drops dramatically.

### How it works

Streaming responses are **Server-Sent Events**:

```
data: {"choices":[{"delta":{"role":"assistant"}}]}      ← the first chunk often has only the role
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
→ "Hello world"
```

Skip non-`data:` lines (keep-alives, comments), stop at `[DONE]`, and treat a missing or `None` `content` as empty.

### On the exam

**NIM = prebuilt, optimised model containers with an OpenAI-compatible API** (they pick TensorRT-LLM
or another engine for your GPU; pulled from NGC with an API key). Triton's own protocol is KServe v2.

---

## Exercise 10 — `rollout_bounds`: capacity during an update

### Why it exists

A rolling update replaces pods one by one. Two settings bound what happens meanwhile:

```
max total pods = replicas + maxSurge          ← for GPU pods, the surge needs FREE GPUs
min ready pods = replicas − maxUnavailable    ← the capacity you keep serving with

3 replicas, maxSurge 1, maxUnavailable 0 → up to 4 pods (1 spare GPU), never below 3 ready
4 replicas, maxSurge 0, maxUnavailable 1 → no spare GPU needed, temporarily 3 ready
```

### On the exam

*Never drop below 3 ready replicas during a rollout:* **RollingUpdate, maxUnavailable 0, maxSurge 1**,
plus a readiness probe that passes only when the model is loaded. `Recreate` or delete-and-apply cause downtime.
**Blue-green** needs a full second set of GPUs; **canary** (Lab 08) shifts a small share of traffic first.

---

## Exercise 11 — `rerank_top_n`: precision for RAG

### Why it exists

A fast **bi-encoder** retriever (embedding similarity) finds the right passages among many (**high recall**)
but ranks them loosely, so the LLM gets noise and cites irrelevant passages (**low precision**).
A **cross-encoder reranker** reads each (query, passage) pair together, which is far more accurate
but too slow for the whole corpus. So: retrieve many, rerank, keep the top few.

```
retrieve top 50 (bi-encoder, fast) → rerank with a cross-encoder → pass the top 5 to the LLM
```

Lab 09's GPU lab measures this: the reranker reorders the same candidates and raises P@1, MRR and nDCG.

### On the exam

*A RAG Blueprint service cites irrelevant passages: high recall, low precision, no re-ingestion allowed.*

| option | verdict |
|---|---|
| Add a cross-encoder reranking stage (e.g. a NeMo Retriever reranking NIM) over a larger candidate set | ✅ a query-time change |
| Re-chunk and re-embed the corpus | ❌ that is re-ingestion |
| Send top-50 passages to the LLM | ❌ lowers precision further |
| Raise the temperature | ❌ doesn't fix retrieval |

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise |
|---|---|
| Triton dynamic batching | 1, 3 |
| Instance groups | 1 |
| Triton ports, Triton protocol | 6, table at the top |
| Kubernetes GPU scheduling, autoscaling, startup probes | 4 |
| Capacity planning | 5 |
| Ensembles | 7 |
| Model Analyzer | 8 |
| NIM | 9 |
| Rolling updates | 10 |
| RAG relevance | 11 |
| Containers | README §3 (pinned NGC tags, NVIDIA Container Toolkit) |
