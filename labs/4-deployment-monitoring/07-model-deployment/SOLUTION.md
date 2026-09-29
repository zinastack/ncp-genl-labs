# Lab 07 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: a trained model is just files. **Serving** turns it into a reliable network service:
a server that batches requests on the GPU (Triton), packaged in a container, scheduled and scaled
by Kubernetes.

```
client ──HTTP/gRPC──► Kubernetes Service ──► Pod (container) ──► Triton ──► model on GPU
         (ex. 6)          (ex. 4, 5)           (Dockerfile)       (ex. 1, 2, 3)
```

---

## Exercise 1 — `triton_config`: telling Triton how to run a model

Triton reads one `config.pbtxt` per model. The function builds this (the GPU lab ships exactly
this file):

```protobuf
name: "text_classifier"
backend: "python"
max_batch_size: 16
input [
  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }
]
output [
  { name: "LABEL" data_type: TYPE_STRING dims: [ 1 ] },
  { name: "SCORE" data_type: TYPE_FP32 dims: [ 1 ] }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 2000
}
instance_group [ { count: 2 kind: KIND_GPU } ]
```

| field | meaning |
|---|---|
| `max_batch_size: 16` | Triton may batch up to 16 requests. `0` disables batching, and then `dims` must include the batch dimension |
| `dims: [ 1 ]` | the shape of **one** request's tensor, *without* the batch dimension (Triton adds it) |
| `preferred_batch_size` | batch sizes the model runs best at; Triton tries to form these |
| `max_queue_delay_microseconds` | the longest a request waits for others to join its batch |
| `instance_group count: 2` | two copies of the model run concurrently, overlapping one's CPU work with the other's GPU work |

Code notes: the `dynamic_batching` block is only emitted if a field is given, and inside it only
the given fields. The helper `_tensors` formats every input and output the same way, so the text
stays consistent.

---

## Exercise 2 — `validate_model_repository`: the folder rules

```
model_repository/
└── text_classifier/          ← model name = directory name
    ├── config.pbtxt          ← required
    └── 1/                    ← numeric version directory, with files inside
        └── model.py
```

The validator walks each model directory and reports, in a fixed order: a missing config, no
numeric version directory (`latest/` doesn't count; `d.name.isdigit()`), empty version directories,
and a config `name:` that differs from the directory name (read with a regex). Triton refuses to
load a model in any of these cases, usually with a less readable error. The last test runs your
validator on the **real repository shipped with the lab**.

---

## Exercise 3 — `simulate_dynamic_batching`: why the queue delay matters

This is the scenario behind the exam question *"P95 ≤ 20 ms, compute 7 ms at batch 1 and ~12 ms at
batch 8, which dynamic-batching config?"*

### The rules (one model instance)

```
launch = max(free, min(oldest_arrival + max_delay, time the batch would be full))
```

The batch starts when the GPU is free **and** either the delay has passed or enough requests are
waiting. It includes every request that has arrived by then, up to `max_batch`.

### Tiny trace (compute always 10 ms, max batch 2, delay 5 ms)

```
arrivals 0.0, 0.5, 1.0
batch 1: oldest arrives 0.0, batch full at 0.5 (2nd arrival) → launch 0.5 → finish 10.5
         latencies 10.5 and 10.0
batch 2: oldest 1.0, can't fill → wait until 1.0 + 5 = 6.0, but the GPU is busy until 10.5
         → launch 10.5 → finish 20.5 → latency 19.5
result [10.5, 10.0, 19.5]
```

### The real question: 400 requests/s, compute = 6.3 + 0.7 × batch ms (7 ms at 1, 11.9 at 8)

| config | P50 | P95 | verdict |
|---|---|---|---|
| no batching (max 1) | 905 ms | 1713 ms | ✗ capacity 1/7 ms = 143 req/s < 400: the queue grows forever |
| batch ≤ 8, delay 0 | 13.1 | 17.2 | ✓ |
| batch ≤ 8, delay 0.5 ms | 12.7 | 16.8 | ✓ |
| batch ≤ 8, delay 2 ms | 13.3 | 17.4 | ✓ the exam's answer |
| batch ≤ 8, delay 20 ms | 20.6 | 29.4 | ✗ the wait alone uses up the SLO |

Two lessons:
1. **Batching is what makes the load possible at all**: 8 requests in 11.9 ms is about 670 req/s of capacity.
2. **The delay adds directly to latency.** Budget it: `delay + compute(batch) + network ≤ SLO`.
   At high load batches fill up anyway, so a small delay is enough. A large one mostly hurts at low load:
   two requests 50 ms apart with a 100 ms delay wait 107.7 ms and 57.7 ms for a 7.7 ms computation.

Code notes: arrivals are sorted, so an index `i` marks the oldest unserved request; `full_at` is the
arrival time of the `max_batch`-th waiting request (or infinity), and each loop serves one batch.

---

## Exercise 4 — `k8s_deployment` / `k8s_hpa`: running it on Kubernetes

A **Deployment** keeps N identical pods running. The important GPU-specific parts:

| field | why |
|---|---|
| `resources.limits: {nvidia.com/gpu: 1}` | the only way to get a GPU: the NVIDIA device plugin (GPU Operator) advertises this resource and the scheduler places the pod on a node with a free GPU |
| `selector.matchLabels` = template labels | how the Deployment finds its pods (must match exactly) |
| ports 8000 / 8001 / 8002 | HTTP, gRPC, Prometheus metrics |
| `readinessProbe /v2/health/ready` | send traffic only once the model is loaded |
| `livenessProbe /v2/health/live` | restart the container if the server hangs |
| `RollingUpdate maxUnavailable: 0, maxSurge: 1` | during an update, start a new pod before removing an old one, so capacity never drops |

The shipped `k8s/deployment.yaml` also adds a **startupProbe**, so liveness doesn't kill a pod that
is still loading a large model, plus a shared-memory volume the Python backend needs.

The **HPA** (Horizontal Pod Autoscaler) adds or removes pods based on a metric. For GPU inference
**CPU is the wrong signal**: the GPU saturates while CPU stays low. Scale on a **custom metric**
such as Triton queue time per request (`type: Pods`, `averageValue`), exposed through Prometheus
Adapter or KEDA. `averageValue` must be a string in Kubernetes, hence `str(target_average)`.

---

## Exercise 5 — `replicas_needed`: Little's law

In a stable system, **requests in flight = arrival rate × time each spends in the system**.

```
200 req/s × 0.5 s = 100 requests in flight
each replica handles 16 at once → 100 / 16 = 6.25 → round UP → 7 replicas
with 20% headroom: 120 / 16 = 7.5 → 8
```

Round **up** (`math.ceil`): 6 replicas would be overloaded. The `- 1e-9` stops float noise from
turning an exact 4.0 into 5. `max(1, …)` means you always keep at least one replica.

---

## Exercise 6 — `kserve_infer_request`: talking to Triton

Triton speaks the **KServe v2** protocol: `POST /v2/models/<name>/infer` with

```json
{"inputs":  [{"name": "TEXT", "shape": [2, 1], "datatype": "BYTES", "data": ["great GPU", "slow network"]}],
 "outputs": [{"name": "LABEL"}, {"name": "SCORE"}]}
```

- `shape [2, 1]` = batch of 2 × the config's `dims [1]`. Here the client sends the batch dimension.
- Strings are `BYTES` on the wire (`TYPE_STRING` in the config).
- NIM, by contrast, exposes an **OpenAI-compatible** API (`/v1/chat/completions`). Know which is which for the exam.

`client.py` in the GPU lab uses your function to call the real server and measure P50/P95/P99 under increasing concurrency.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| P95 SLO + dynamic batching config | small `max_queue_delay` (≈ ms) + preferred sizes you measured |
| low GPU use, host-side work | more `instance_group` count |
| which port for Prometheus | 8002 `/metrics` |
| schedule a pod on a GPU | `resources.limits nvidia.com/gpu` |
| autoscaling on CPU doesn't react | custom metrics: queue time, in-flight, DCGM GPU util |
| pods killed while loading | startupProbe |
| pipeline of pre/post-processing | Triton ensemble or BLS |
| find the best batch/instance config | Model Analyzer (uses perf_analyzer) |
| prebuilt optimised LLM container, OpenAI API | NIM |
| RAG cites irrelevant passages, high recall / low precision | add a reranker (query time, no re-ingestion) |
