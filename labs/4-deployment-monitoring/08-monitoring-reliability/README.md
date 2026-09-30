# Lab 08 — Production Monitoring & Reliability (7% of exam)

> Blueprint: *monitoring dashboards, anomaly detection, automated retraining, version management.*

You will turn raw telemetry into decisions: latency percentiles, Triton's Prometheus metrics,
anomaly detection, **data drift** (PSI, KS), **SLO burn-rate** alerts, a retraining trigger, a
model registry with promotion and rollback, and canary analysis. On the GPU instance the section
stack runs **Triton + Prometheus + Grafana + DCGM exporter**, so you can watch your Lab 07 model
under load.

```
make test-08          # YOUR exercises (run from the repo root)
make solutions-08     # reference solutions
make quiz-08          # exam-style questions
make s4-triton-up     # same stack as Lab 07 (Grafana :3000, Prometheus :9090)
make gpu-08           # traffic phases → watch dashboards and alerts
```

---

## 1. What to monitor for an LLM / inference service

| Layer | Signals | Source |
|---|---|---|
| **Service (golden signals)** | latency P50/P95/P99, traffic (RPS), errors (5xx, timeouts), saturation (queue time, in-flight) | Triton `:8002/metrics`, NIM metrics, API gateway |
| **LLM-specific** | **TTFT**, **ITL/TPOT**, tokens/s, input/output token counts, KV-cache utilisation, request queue depth | NIM / TensorRT-LLM / vLLM metrics |
| **GPU** | utilisation (SM activity), memory used, power, temperature, ECC / XID errors, NVLink | **DCGM exporter** (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, ...) |
| **Model quality** | drift of inputs and outputs, refusal rate, guardrail triggers, user feedback (thumbs), LLM-judge scores on samples, hallucination / groundedness checks | app logs, evaluation jobs |
| **Cost** | GPU-hours, tokens per dollar | billing plus usage metrics |

Useful Triton metrics: `nv_inference_request_success`, `nv_inference_request_failure`,
`nv_inference_count`, `nv_inference_exec_count` (inference_count / exec_count = **average
batch size**), `nv_inference_queue_duration_us`, `nv_inference_compute_infer_duration_us`,
`nv_gpu_utilization`, `nv_gpu_memory_used_bytes`. They are counters, so compute rates in PromQL:

```promql
# average queue time per request over 1m (µs)
rate(nv_inference_queue_duration_us[1m]) / rate(nv_inference_request_success[1m])
# average dynamic batch size
rate(nv_inference_count[1m]) / rate(nv_inference_exec_count[1m])
```

**Percentiles, not averages:** tail latency (P95/P99) is what users feel and what SLOs specify.
Averages hide the tail.

## 2. Anomalies and drift

- **Point anomalies:** a rolling z-score (value vs trailing mean/std), EWMA control charts, or
  seasonal baselines (daily and weekly traffic patterns).
- **Data drift:** compare the production distribution with the training/reference distribution.
  - **PSI** (population stability index) = Σ (a−e)·ln(a/e) over bins. **< 0.1 stable,
    0.1–0.25 moderate, > 0.25 significant drift.**
  - **KS test** (max CDF gap) for continuous features. For text, compare **embedding** distributions,
    prompt lengths, topic or intent mix, language mix.
- **Concept drift:** the input→label relationship changes. It needs labels (delayed feedback) to detect.
- **Prediction drift:** the output distribution shifts (for example the refusal rate doubles). This is an early warning without labels.

## 3. SLOs, error budgets and alerting

- **SLO:** for example 99.9% of requests succeed with P95 < 500 ms over 30 days. The **error budget** is 0.1%.
- **Burn rate** = observed error rate / (1 − SLO). A burn rate of 1 uses the budget exactly
  over the window. Alert on **fast burn** (for example 14.4× over 1 h) and **slow burn** (for example 6× over 6 h).
  Multi-window alerts avoid paging on blips.
- Alert on symptoms (latency, errors) rather than causes. Every alert needs a runbook.

## 4. Lifecycle: versions, rollout, retraining

- **Model registry** (MLflow, W&B, NGC private registry): immutable versions with lineage
  (data, code, hyperparameters, eval results), stages (`staging` → `production` → `archived`),
  and a **one-step rollback** to the previous production version.
- **Rollout patterns:**
  - **Shadow:** new model gets mirrored traffic and its responses are discarded; zero user risk.
  - **Canary:** a small % of live traffic; compare latency, errors and quality; then promote or roll back.
  - **Blue-green:** two full environments with an instant switch; easy rollback, 2× resources during the cut-over.
  - **A/B test:** a statistically powered comparison of business metrics.
- **Automated retraining triggers:** significant drift (PSI > 0.25), a quality drop on
  monitored evaluations, enough new labelled data, or a schedule. Retraining must pass the **same
  evaluation gates** before promotion. Never auto-deploy without gates.
- **Reliability:** redundancy (N+1 replicas across zones), health checks, timeouts and retries
  with back-off, circuit breakers, graceful degradation (fall back to a smaller model or cached
  answers), and rate limiting.

## 5. Exam traps

- CPU metrics are poor autoscaling and alerting signals for GPU inference. Use queue time and GPU metrics.
- Averages of percentiles are wrong. Aggregate histograms, then compute percentiles.
- Drift alone doesn't mean accuracy dropped. Use it as a trigger for investigation or evaluation, not blind retraining.
- Canary compares **the same time window**. Comparing today's canary with last week's baseline confounds traffic changes.
- Version everything together: model weights **and** tokenizer, prompt templates, guardrail configs and engine build settings.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function / class | Concept |
|---|---|---|
| 1 | `latency_summary` | nearest-rank percentiles |
| 2 | `parse_prometheus`, `triton_avg_queue_ms`, `triton_avg_batch_size` | reading Triton metrics |
| 3 | `rolling_zscore_anomalies` | anomaly detection |
| 4 | `psi`, `ks_statistic` | data drift |
| 5 | `burn_rate`, `should_page` | SLO burn-rate alerting |
| 6 | `retraining_decision` | automated retraining triggers |
| 7 | `ModelRegistry` | versioning, promotion, rollback |
| 8 | `canary_route`, `canary_verdict` | deterministic traffic split and promote/rollback |
| 9 | `merge_histograms`, `histogram_quantile` | correct latency percentiles across replicas (Prometheus histograms) |
| 10 | `CircuitBreaker` | fail fast when a dependency is down, then probe for recovery |
| 11 | `backoff_delays` | retries with exponential backoff and jitter |
| 12 | `embedding_drift` | drift detection for text inputs via embeddings |
| 13 | `gpu_node_action` | acting on DCGM health signals (XID, ECC, temperature) |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).

## Further reading

Chosen to explain the concepts behind this lab; read the *Start here* items first.

**Start here**
- [Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) (Google SRE book): the four golden signals, and why you alert on symptoms rather than causes.
- [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) (Google SRE workbook): error budgets and multi-window burn-rate alerts.
- [Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) (Chip Huyen): covariate, label and concept drift, and how to detect each.

**Go deeper**
- [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) (Prometheus): why P95 comes from `histogram_quantile` over buckets.
- [Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html) (Martin Fowler): the closed → open → half-open pattern.

**NVIDIA docs**
- [Triton metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html): queue time vs compute time, and what each counter means.
- [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter): GPU utilisation, memory, temperature, XID and ECC metrics for Prometheus.
