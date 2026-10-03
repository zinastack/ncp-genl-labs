# Lab 07 — Model Deployment (9% of exam)

> Blueprint: *containerized inference pipelines, orchestration (Kubernetes, NVIDIA Triton), serving.*

Deployment questions on the exam are scenarios: *"P95 doubled after you enabled dynamic batching"*,
*"the second replica is stuck Pending"*, *"the rollout hangs on a single-GPU node"*. You answer
them best after you have **turned the knob yourself and watched what happened**. So this lab is
mostly hands-on **tasks** on a GPU instance:

| Part | Where | What |
|---|---|---|
| **Tasks 1–9** | GPU instance, Docker | Triton (batching, instances, backends, ensembles, versions, Model Analyzer), TensorRT, TensorRT-LLM, NIM |
| **Tasks K1–K6** | same instance, Kubernetes (k3s) | GPU scheduling, probes, time-slicing, autoscaling on queue time, rolling updates, canary |
| **Calculations** | laptop | `exercises.py`: the numbers behind the tasks (batching, Little's law, rollout bounds, KV cache) |

Every task follows the same loop: **scenario → change a real file or flag → run one `make` command →
record what you see → answer the questions**. [`SOLUTION.md`](SOLUTION.md) has the expected results and
the explanation for each task. `make check-07` tells you which tasks you have completed.

## Setup

GPU: **1× L4** (the S4 Brev Launchable, or `make brev-up S=4` from your laptop). All commands below run
from the section folder on the instance:

```bash
cd ~/ncp-genl-labs/labs/4-deployment-monitoring
make doctor          # GPU, driver, Docker
make triton-up       # build the Triton image (first time ~10 min) and start Triton + Prometheus + Grafana
make check-07        # progress
```

From the repo root, any target works as `make s4-<target>` (for example `make s4-perf M=text_classifier`).
Images are pinned (`TRITON_VERSION=25.08`, TensorRT-LLM `1.0.0`). With an NVIDIA driver ≥ 580 you can use
newer tags: `make triton-up TRITON_VERSION=26.09`.

---

## Concepts in five minutes

| Layer | What it does | API |
|---|---|---|
| **TensorRT** | compiles a network (usually from ONNX) into an **engine** for one GPU architecture and TensorRT version: kernel fusion, FP16/INT8/FP8 | library, `trtexec` |
| **TensorRT-LLM** | the same for LLMs: fused attention, paged KV cache, **in-flight batching**, FP8/INT4; served with `trtllm-serve` | OpenAI-compatible |
| **Triton Inference Server** | multi-framework server (TensorRT, ONNX Runtime, PyTorch, Python, TensorRT-LLM, vLLM backends): dynamic batching, instance groups, ensembles, versions | **KServe v2** on HTTP **:8000** / gRPC **:8001**; Prometheus metrics **:8002** |
| **NIM** | a pre-built container per model, from NGC, that picks the best engine for your GPU (a "profile") | **OpenAI-compatible** (`/v1/chat/completions`) |
| **Kubernetes + GPU Operator** | schedules GPU pods (`nvidia.com/gpu`); the operator installs driver, container toolkit, device plugin, GPU feature discovery, DCGM exporter, MIG manager | manifests, Helm |

A Triton **model repository** is a folder per model, a `config.pbtxt`, and numbered **version** folders:

```
model_repository/
├── text_classifier/      config.pbtxt + 1/model.py      Python backend (PyTorch inside)
├── distilbert_onnx/      config.pbtxt + 1/model.onnx    ONNX Runtime backend       (task 4)
├── distilbert_trt/       config.pbtxt + 1/model.plan    TensorRT engine            (task 4)
├── tokenizer/, postprocess/                             Python, KIND_CPU           (task 5)
└── sentiment_ensemble/   config.pbtxt + 1/              ensemble: tokenizer → TensorRT → postprocess
```

Triton runs here in **explicit model-control mode**: only `text_classifier` loads at start, and you load
the others (`make load M=...`). After editing a `config.pbtxt`, `make reload M=...` applies it without
restarting the server.

---

## Task 1 — Explore a running Triton

**Scenario.** You inherit a Triton deployment. You need to know which port Prometheus should scrape,
which endpoint a Kubernetes readiness probe should call, and what configuration Triton *actually*
applied (it fills in fields you didn't write).

1. Server and health:
   ```bash
   curl -s localhost:8000/v2 | python3 -m json.tool        # version + supported extensions
   curl -si localhost:8000/v2/health/live | head -1         # process is up
   curl -si localhost:8000/v2/health/ready | head -1        # every loaded model is ready to serve
   ```
2. `make models`: which models are `READY`, which are `UNAVAILABLE`, and why?
3. `make config M=text_classifier`: compare it with `triton/model_repository/text_classifier/config.pbtxt`.
   List two fields Triton added.
4. A raw KServe v2 request:
   ```bash
   curl -s localhost:8000/v2/models/text_classifier/infer -H 'Content-Type: application/json' \
     -d '{"inputs":[{"name":"TEXT","shape":[2,1],"datatype":"BYTES","data":["Great GPU","Slow network"]}]}'
   ```
5. `make gpu-07` (client latency sweep), then `make stats M=text_classifier` and
   `curl -s localhost:8002/metrics | grep -E '^nv_inference_(request_success|count|exec_count|queue_duration_us)'`.

**Answer.** What is the difference between *live* and *ready*? Which number tells you the average batch
size Triton formed? Why can one request in step 4 contain 2 texts?

## Task 2 — Dynamic batching vs P95 latency

**Scenario.** *"P95 must stay ≤ 20 ms, and the team wants more throughput from the same GPU. Which
`dynamic_batching` settings?"* You will measure four configurations of `text_classifier`.

Edit `triton/model_repository/text_classifier/config.pbtxt`. After each edit: `make reload M=text_classifier`,
then `make perf M=text_classifier LABEL="..."` (perf_analyzer at concurrency 1, 6, 11, 16; every run is
recorded with the live config):

| Run | `dynamic_batching` block | LABEL |
|---|---|---|
| A | delete the whole block | `A no batching` |
| B | `dynamic_batching { }` | `B batching, no delay` |
| C | `preferred_batch_size: [ 4, 8 ]` and `max_queue_delay_microseconds: 2000` (the original) | `C 2 ms delay` |
| D | `max_queue_delay_microseconds: 50000` | `D 50 ms delay` |

Then `make perf-report`.

**Record** for each run: average batch size, throughput and P95 at concurrency 1 and 16.

**Answer.** Which run has the best throughput at concurrency 16, and why? Why is D's P95 at
concurrency 1 roughly 50 ms worse than B's? Why is A's average batch exactly 1.0 even though
`max_batch_size` is 16? What would you pick for the 20 ms SLO? (Check your reasoning with
calculation 1, `simulate_dynamic_batching`.)

## Task 3 — Instance groups

**Scenario.** *"GPU utilisation is 40% at peak while requests queue. What do you change in
config.pbtxt?"*

Restore run C's batching, then change `instance_group`:

1. `count: 1` → `2` → `3` (`kind: KIND_GPU`). For each: `make reload`, `make perf LABEL="2 GPU instances"`,
   and `nvidia-smi --query-gpu=memory.used --format=csv` (memory per instance).
2. `make triton-logs`: find the lines `text_classifier_0_0 on cuda:0`, `text_classifier_0_1 ...`.
3. `count: 2 kind: KIND_CPU`: reload and perf. Read `1/model.py` `initialize()`: why does a Python model
   only run on the CPU if its code reads `model_instance_kind`?

**Answer.** Why can two instances on **one** GPU raise throughput (what overlaps)? What does each extra
instance cost? When does adding instances stop helping?

## Task 4 — ONNX Runtime vs TensorRT: backends, precision, optimization profiles

**Scenario.** *"Latency is too high on the PyTorch (Python backend) model. Without changing the model
architecture, what is the NVIDIA way to make it faster?"*

1. Export and serve with ONNX Runtime:
   `make export-onnx`, `make load M=distilbert_onnx`, `make perf M=distilbert_onnx LABEL="ORT fp32"`.
2. Build a TensorRT engine from the same ONNX file (read the `trtexec` command `make` prints):
   `make trt-build PREC=fp32`, `make load M=distilbert_trt`, `make perf M=distilbert_trt LABEL="TRT fp32"`.
3. FP16: `make trt-build PREC=fp16`, `make reload M=distilbert_trt`, `make perf M=distilbert_trt LABEL="TRT fp16"`.
4. Break it on purpose: `make trt-build PREC=fp16 MAX_BS=8`, then `make reload M=distilbert_trt` and read
   the error in `make triton-logs`. Fix it two ways: `max_batch_size: 8` in the config, or rebuild with `MAX_BS=16`.
5. Let ONNX Runtime use TensorRT for you: uncomment the `optimization { execution_accelerators ... }` block in
   `distilbert_onnx/config.pbtxt`, `make reload M=distilbert_onnx` (note how long the load takes), then
   `make perf M=distilbert_onnx LABEL="ORT + TensorRT EP fp16"`.

**Answer.** Rank the five runs by throughput. Why is FP16 faster on an L4? What does an **optimization
profile** (min/opt/max shapes) fix, and what happens to a request outside it? Could you copy
`model.plan` to a T4 or an H100 server? Why does a PREC=int8 build without calibration data give fast but
untrustworthy results?

## Task 5 — An ensemble: tokenizer → TensorRT → postprocess inside Triton

**Scenario.** *"Clients send raw text. Tokenization in every client is inconsistent and each extra
network hop adds latency. How do you serve text in, label out, with the TensorRT model?"*

1. `make load M=sentiment_ensemble` (it loads `tokenizer`, `distilbert_trt` and `postprocess` too: `make models`).
2. `make infer M=sentiment_ensemble TEXT="The rollout went smoothly"`.
3. `make gpu-07 M=sentiment_ensemble` and `make gpu-07 M=text_classifier`: compare latency and throughput.
4. `make perf M=sentiment_ensemble`, then `make stats M=tokenizer`, `M=distilbert_trt`, `M=postprocess`:
   where is the time spent?
5. Raise the tokenizer to `count: 4 kind: KIND_CPU`, reload `tokenizer` and perf the ensemble again.

**Answer.** What do `input_map` and `output_map` connect? Why is the tokenizer `KIND_CPU`? When would you
need **BLS** (business logic scripting) instead of an ensemble?

## Task 6 — Versions, model control and warm-up

**Scenario.** *"Ship model v2 next to v1 without restarting Triton, let one client pin v1, and stop the
latency spike on the first request after every deploy."*

1. `make version2`, then set `version_policy: { all: {} }` in `distilbert_onnx/config.pbtxt` and reload.
   `make models` shows v1 and v2. Call each explicitly:
   `curl -s localhost:8000/v2/models/distilbert_onnx/versions/1` and `.../versions/2`.
2. Switch to `version_policy: { specific: { versions: [ 1 ] } }`, reload: what happens to v2? Then
   `latest: { num_versions: 1 }`: which version serves?
3. `make first-request M=distilbert_trt`: note load time, first-request and warm latency. Uncomment the
   `model_warmup` block in `distilbert_trt/config.pbtxt`, run it again.
4. `make unload M=sentiment_ensemble`, `make models`: what happened to its composing models?

**Answer.** What do `none`, `poll` and `explicit` model-control modes do, and which one would you use in
Kubernetes? What does warm-up trade?

## Task 7 — Model Analyzer: let a tool search the configs

**Scenario.** *"Find the configuration with the highest throughput whose p99 latency stays under 30 ms,
without hand-editing config.pbtxt for a week."*

`make ma-07` (stops the Compose Triton, runs Model Analyzer in docker mode for ~20–40 minutes, starts
Triton again). Read `results/model_analyzer/reports/summaries/distilbert_onnx/` and the generated configs in
`results/model_analyzer/configs/`.

**Answer.** Which knobs did it vary? Which config won, and how does it compare with your best manual run
of `distilbert_onnx`? Why must the constraint be set **before** picking the winner? (Calculation 3,
`pick_best_config`, does the same selection.)

## Task 8 — TensorRT-LLM: in-flight batching, KV cache, engines, quantization

**Scenario.** *"A chat service on one L4: users complain about the wait before the first word (TTFT)
at peak, and finance wants more tokens per GPU. Which TensorRT-LLM settings?"*

`make triton-down` first (free the GPU memory). Server on `:8010`, model `Qwen/Qwen2.5-1.5B-Instruct`:

1. Baseline (PyTorch backend): `make trtllm-up`, then `make trtllm-logs` and find the KV-cache size
   (tokens) it allocated. Compare with calculation 6, `kv_cache_tokens`.
2. In-flight batching: `make trtllm-bench CONC=1`, `CONC=8`, `CONC=32`. How do TTFT, TPOT and output
   tokens/s move as concurrency grows?
3. `make trtllm-up MBS=4`, `make trtllm-bench CONC=32`: what happens to TTFT and throughput?
4. `make trtllm-up KV=0.05`, `make trtllm-bench CONC=32`: what limits the batch now?
5. `make trtllm-up MNT=1024`, `make trtllm-bench CONC=32`: what does `max_num_tokens` limit per iteration?
6. Engine path: `make trtllm-up BACKEND=tensorrt` (watch the engine build in `make trtllm-logs`), bench at `CONC=32`.
   Then `make trtllm-up BACKEND=tensorrt EXTRA=int8_weight_only.yml`, bench again, and compare
   `nvidia-smi` memory.
7. `make trtllm-report`, then `make trtllm-down`.

**Answer.** Why does TTFT grow with concurrency while TPOT stays nearly flat (until it doesn't)? Which
settings are baked into a TensorRT engine at build time and which can change at serve time? Why does INT8
weight-only quantization speed up decoding even though the math is still FP16?

## Task 9 — NIM: the packaged path

**Scenario.** *"The team wants an OpenAI-compatible Llama endpoint on its own GPU this week, with
NVIDIA-optimised engines, and no engine builds to maintain."*

You need a free NGC API key (ngc.nvidia.com → Setup → Generate API Key, and accept the model's terms in
the NGC catalog): `export NGC_API_KEY=...`. Stop other GPU servers first (`make trtllm-down`).

1. `make nim-profiles`: which profiles exist (backend, precision, tensor parallelism, throughput vs latency)
   and which are compatible with an L4?
2. `make nim-up`, then `make nim-logs`: which profile did it choose and why?
3. `make nim-chat`, then `curl -s localhost:8020/v1/metrics | grep -i -E "time_to_first|kv_cache" | head`.
4. `make nim-bench CONC=1` and `CONC=16` (GenAI-Perf): TTFT, inter-token latency, tokens/s.
5. Pin a profile: `make nim-up PROFILE=<id from step 1>`. Find the exact image version
   (`docker inspect --format '{{index .RepoDigests 0}}' $(docker ps -qf name=genl-nim | head -1)`) and say why
   production should pin it instead of `:latest`.

**Answer.** NIM vs Triton + TensorRT-LLM vs `trtllm-serve`: when would you choose each? What does
`NIM_MODEL_PROFILE` control? Where does NIM cache the downloaded model, and why mount it as a volume?

---

## Kubernetes tasks (same instance)

`make k8s-up` installs single-node **k3s** using the host's Docker (so your Triton image is directly
usable), makes `nvidia` Docker's default runtime, and installs the **NVIDIA device plugin** with **GPU
feature discovery** through Helm. On a real cluster the **GPU Operator** installs those, plus the driver,
container toolkit, DCGM exporter and MIG manager; see the comments in `k8s/install-k3s.sh`.
Restarting Docker stops the Compose stack: `make triton-up` again afterwards for Lab 08.

### K1 — A GPU node

`make k8s-up`, then `kubectl describe node | grep -E "nvidia.com/gpu|gpu.product|gpu.memory"`.

**Answer.** Where does `nvidia.com/gpu: 1` come from? Which label would a `nodeSelector` use to pin a pod
to L4 nodes?

### K2 — Deploy Triton; probes vs a slow model load

1. `make k8s-deploy`, then `kubectl get pods -w` (Ctrl+C when Ready), `kubectl describe pod -l app=triton`
   (probe settings and events), `make gpu-07 URL=http://localhost:30800`.
2. Drill: `make k8s-drill-probe` removes the `startupProbe` and makes the model take 120 s to load.
   Watch `kubectl get pods -w`: RESTARTS climbs and the pod ends in `CrashLoopBackOff`. Read the events.
3. Fix it properly: set `SLOW_LOAD_SECONDS` to `"120"` in `k8s/deployment.yaml` (startupProbe still in
   place) and `make k8s-deploy`. Ready after ~2.5 min, 0 restarts. Set it back to `"0"` and redeploy.

**Answer.** Why did the liveness probe kill a healthy-but-loading server? What does each probe protect?

### K3 — GPU scheduling and time-slicing

1. `kubectl scale deploy triton --replicas=2`, then `make k8s-status` and `kubectl describe pod <pending pod>`.
2. `make k8s-timeslice N=4`: the node now advertises 4 GPUs. What happens to the Pending pod?
   `nvidia-smi` shows two `tritonserver` processes on one GPU.

**Answer.** Why was the pod Pending? What does time-slicing share and what does it *not* isolate? When
would you use MIG instead (and on which GPUs)? Can a pod request `nvidia.com/gpu: 0.5`?

### K4 — Autoscale on queue time, not CPU

Needs `N=4` from K3.

1. `make k8s-monitoring` (Prometheus on :30900 + prometheus-adapter), `make k8s-hpa`.
2. `kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/triton_avg_queue_us" | python3 -m json.tool`
3. Terminal 1: `make k8s-load C=64 S=300`. Terminal 2: `kubectl get hpa triton -w`, and `kubectl top pods`.
4. Stop the load and watch the scale-down wait for the stabilization window.

**Answer.** What was the CPU usage while requests queued? Why scale on queue time (or pending requests)
rather than CPU or GPU utilisation? What caps `maxReplicas` here?

### K5 — A rolling update that hangs

1. Back to one GPU: `kubectl delete hpa triton; kubectl scale deploy triton --replicas=1; make k8s-timeslice N=1`.
2. `make k8s-rollout`: the new pod stays Pending and the rollout times out. Why? (calculation 4, `rollout_bounds`)
3. Fix A: `kubectl patch deploy triton -p '{"spec":{"strategy":{"rollingUpdate":{"maxSurge":0,"maxUnavailable":1}}}}'`
   and watch it finish. What did you give up? Fix B would be a spare GPU (`make k8s-timeslice N=2`).
4. `kubectl rollout history deploy/triton` and `kubectl rollout undo deploy/triton`.

### K6 — Canary and rollback

1. `make k8s-timeslice N=4`, `make k8s-canary` (3 × v1 + 1 × v2, where v2 adds 40 ms per batch).
2. `make k8s-load C=32 S=180`, then `make k8s-canary-check` (your `canary_verdict` from Lab 08 decides).
3. In Prometheus (`http://localhost:30900`, forward the port from Brev):
   `sum by (version) (rate(nv_inference_request_duration_us[1m])) / sum by (version) (rate(nv_inference_request_success[1m]))`.
4. `make k8s-rollback`. Finally `make k8s-down`.

**Answer.** How is traffic split between v1 and v2 here, and what would give you an exact 5% split?
Canary vs blue-green vs shadow: which needs twice the GPUs?

---

## Calculations (`exercises.py`, laptop)

`make test-07` from the repo root. Each one is the arithmetic behind a task:

| # | Function | Used in |
|---|---|---|
| 1 | `simulate_dynamic_batching` | task 2: why the queue delay adds to P95 |
| 2 | `replicas_needed` | K4: Little's-law capacity planning |
| 3 | `pick_best_config` | task 7: Model Analyzer's selection rule |
| 4 | `rollout_bounds` | K5: maxSurge / maxUnavailable with scarce GPUs |
| 5 | `rerank_top_n` | RAG serving: recall first, then a reranker for precision |
| 6 | `kv_cache_tokens` | task 8: how many tokens the KV cache holds |

## Exam traps

- `max_batch_size: 0` disables Triton batching, and then dims *include* the batch dimension.
- A large `max_queue_delay_microseconds` wins throughput benchmarks and breaks latency SLOs.
- More model instances need more GPU memory and only help while the GPU still has idle time.
- Stateful models (a stream or conversation must stay on one instance) use the **sequence batcher**, not the dynamic batcher.
- A TensorRT engine is tied to the GPU architecture and the TensorRT version it was built with, and to its optimization profiles.
- NIM and `trtllm-serve` expose the **OpenAI** API; Triton exposes **KServe v2** (`/v2/models/<name>/infer`); metrics on **:8002**.
- CPU-based HPA doesn't see GPU saturation. Scale on queue time or pending requests.
- A rolling update with `maxUnavailable: 0` needs a **free GPU** for the surge pod.
- `startupProbe` protects slow model loads from the liveness probe.
- Time-slicing shares a GPU without memory or fault isolation; MIG (A100/H100/B200) isolates.
- Pin image tags (`tritonserver:25.08-py3`, NIM versions); never deploy `:latest`.

## Further reading

Chosen to explain the concepts behind this lab; read the *Start here* items first.

**Start here**
- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (Anyscale): static vs dynamic vs in-flight (continuous) batching, and why it matters for LLMs.
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (NVIDIA): where Triton, TensorRT-LLM and batching fit together.

**NVIDIA docs**
- Triton: [model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) (instance groups, max_batch_size, ensembles) and [batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) (dynamic batching, preferred sizes, queue delay).
- [NIM for LLMs](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html): the OpenAI-compatible container, profiles and deployment.
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/): engine build, in-flight batching, paged KV cache and quantization.
