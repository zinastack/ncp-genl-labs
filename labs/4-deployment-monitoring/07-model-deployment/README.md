# Lab 07 — Model Deployment (9% of exam)

> Blueprint: *containerized inference pipelines, orchestration (Kubernetes, NVIDIA Triton), serving.*

The exercises cover Triton model configuration, a **dynamic-batching simulator** (so you can
*see* why `max_queue_delay_microseconds` matters for a P95 SLO), model-repository validation,
Kubernetes manifests for GPU inference, and capacity planning. On a GPU instance you then run a
real **Triton Inference Server** in Docker with a Python-backend text classifier, hit it with
`perf_analyzer`, and (optionally) deploy the same thing to Kubernetes.

```
make test-07          # YOUR exercises (run from the repo root)
make solutions-07     # reference solutions
make quiz-07          # exam-style questions
make s4-triton-up     # build + start Triton, DCGM exporter, Prometheus, Grafana
make gpu-07           # client latency sweep + perf_analyzer
```

---

## 1. The NVIDIA serving stack

| Layer | What it does |
|---|---|
| **TensorRT-LLM** | Compiles LLMs into optimised engines (fused kernels, FP8/INT4, in-flight batching, paged KV). |
| **Triton Inference Server** | Multi-framework server (TensorRT, TensorRT-LLM, PyTorch, ONNX, Python, vLLM backends); HTTP `:8000`, gRPC `:8001`, Prometheus metrics `:8002`; dynamic batching; concurrent model instances; ensembles; model versioning. |
| **NIM** | Pre-built, optimised **containers** per model with an OpenAI-compatible API (`/v1/chat/completions`), which pick the best engine for the GPU. Pulled from NGC with an API key. |
| **Kubernetes + GPU Operator** | Schedules GPU pods (`nvidia.com/gpu` resource), installs drivers, container toolkit, DCGM exporter. **NIM Operator** / Helm charts manage NIM deployments. |

## 2. Triton model repository

```
model_repository/
└── text_classifier/
    ├── config.pbtxt
    ├── 1/               ← version directory (integer); model.py / model.plan / model.onnx ...
    └── 2/
```

```protobuf
name: "text_classifier"
backend: "python"                       # or platform: "tensorrt_plan", "onnxruntime_onnx", ...
max_batch_size: 16                      # >0 enables batching; dims below EXCLUDE the batch dim
input  [ { name: "TEXT"   data_type: TYPE_STRING dims: [ 1 ] } ]
output [ { name: "LABEL"  data_type: TYPE_STRING dims: [ 1 ] },
         { name: "SCORE"  data_type: TYPE_FP32   dims: [ 1 ] } ]
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 2000    # wait ≤2 ms to form a bigger batch
}
instance_group [ { count: 2 kind: KIND_GPU } ]   # 2 copies per GPU → overlap execution
version_policy: { latest: { num_versions: 1 } }
```

- **Dynamic batching** merges individual requests server-side. `max_queue_delay_microseconds`
  trades a little latency for throughput. The delay **adds directly to P95**, so budget it:
  `queue_delay + compute(batch) + network ≤ SLO`.
- **Instance groups** run several copies of the model concurrently (on one or more GPUs), overlapping compute with data transfer.
- **Sequence batching** is for stateful models (the same sequence goes to the same instance).
- **Ensembles / BLS** chain pre-processing → model → post-processing inside Triton, avoiding extra network hops.
- **Model control:** `--model-control-mode=explicit` plus the load/unload API allows hot swaps. `version_policy` selects which versions are served.
- **Tools:** `perf_analyzer` (latency/throughput vs concurrency), **Model Analyzer** (sweeps
  configs such as batch sizes and instance counts to meet constraints), **GenAI-Perf** (LLM
  metrics: TTFT, ITL, tokens/s).

## 3. Containers and Kubernetes

- Base images from **NGC** (`nvcr.io/nvidia/tritonserver:<yy.mm>-py3`, `nvcr.io/nim/...`). Run with
  `docker run --gpus all` (NVIDIA Container Toolkit). **Pin image tags**; never use `latest` in production.
- Pod spec: `resources.limits: {nvidia.com/gpu: 1}`. **Readiness probe** `/v2/health/ready`,
  **liveness** `/v2/health/live`. Mount the model repository from a PVC or object storage.
  Set `shm-size` (Python backend uses shared memory).
- **Autoscaling:** the HPA on CPU is useless for GPU inference. Scale on **custom metrics**
  (Triton queue time, requests in flight, GPU utilisation from DCGM) through Prometheus Adapter or KEDA.
- **Rollouts:** rolling update with `maxUnavailable: 0`, canary or blue-green (see Lab 08).
  Account for **model load time** (minutes for LLMs) in probes (`startupProbe`) and scaling.
- **MIG** (A100/H100) splits one GPU into isolated instances for small models. **Time-slicing** shares a GPU without isolation.
- **Capacity (Little's law):** concurrent requests = arrival rate × latency. Replicas needed =
  ⌈RPS × latency / concurrency-per-replica⌉.

## 4. Exam traps

- `max_batch_size: 0` disables Triton batching, and then dims *include* the batch dim.
- A large `max_queue_delay_microseconds` wins throughput benchmarks and breaks latency SLOs.
- CPU-based HPA doesn't reflect GPU saturation. Use queue or latency metrics.
- NIM exposes an **OpenAI-compatible** API. Triton exposes the **KServe v2** inference protocol (`/v2/models/<name>/infer`).
- Port 8002 is metrics. Scrape it with Prometheus.
- Model warm-up (`model_warmup` in config.pbtxt) avoids first-request latency spikes.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `triton_config` | write a config.pbtxt with dynamic batching and instance groups |
| 2 | `validate_model_repository` | repository layout rules |
| 3 | `simulate_dynamic_batching` | queue delay vs latency vs throughput |
| 4 | `k8s_deployment`, `k8s_hpa` | GPU pod spec, probes, custom-metric autoscaling |
| 5 | `replicas_needed` | Little's-law capacity planning |
| 6 | `kserve_infer_request` | Triton / KServe v2 HTTP payload |
| 7 | `triton_ensemble_config` | a pre-process → model → post-process pipeline inside Triton |
| 8 | `pick_best_config` | Model-Analyzer-style config selection under latency/memory limits |
| 9 | `openai_chat_request`, `parse_sse_stream` | NIM's OpenAI-compatible API and streaming (SSE) |
| 10 | `rollout_bounds` | rolling-update capacity: maxSurge, maxUnavailable, spare GPUs |
| 11 | `rerank_top_n` | two-stage RAG retrieval: recall first, then precision with a reranker |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).

GPU part: `triton/model_repository/text_classifier` (Python backend, HF DistilBERT on GPU),
`client.py`, `docker-compose.yml`, and `k8s/` manifests.
