# Lab 08 — Production Monitoring & Reliability (7% of exam)

> Blueprint: *monitoring dashboards, anomaly detection, automated retraining, version management.*

Monitoring questions are scenarios too: *"latency alerts fire but GPU utilisation is low"*, *"the P95 on
the dashboard looks wrong after we scaled to 4 replicas"*, *"inputs changed, should we retrain?"*. In this
lab you monitor the Triton you deployed in Lab 07 with **Prometheus, Grafana and the DCGM exporter**, make
real alerts fire, break things on purpose, and measure drift:

| Part | Where | What |
|---|---|---|
| **Tasks 1–7** | GPU instance (same stack as Lab 07) | metrics, PromQL, dashboards, alerts, failure drills, drift, canary |
| **Calculations** | laptop | `exercises.py`: percentiles, Triton counters, anomalies, PSI/KS, burn rate, canary verdict, histogram quantiles, backoff, embedding drift |

The scripts that drive the tasks (`loadgen.py`, `drift.py`, the Kubernetes canary check) call the
calculation functions: the reference solutions by default, **yours** with `USE_EXERCISES=1`.
[`SOLUTION.md`](SOLUTION.md) has the expected results. `make check-08` asks Prometheus which of your alerts
have fired.

## Setup

```bash
cd ~/ncp-genl-labs/labs/4-deployment-monitoring
make triton-up       # Triton :8000/:8002 · Prometheus :9090 · Grafana :3000 · DCGM exporter :9400
make check-08
```

Open Grafana (dashboard *Triton inference (GENL Lab 08)*) and Prometheus through the Launchable's secure
links (`grafana`, `prometheus`), or forward them from your laptop: `make brev-forward S=4 PORT=3000` and
`PORT=9090`. The configs live in `prometheus/` (scrape config, recording rules, alerts) and `grafana/`.

---

## Task 1 — From raw counters to the numbers on a dashboard

**Scenario.** *"Grafana shows an average queue time of 12 ms. Where does that number come from, and is it
the last minute or since the server started?"*

1. `make metrics-08`: 20 s of traffic, then the averages computed by `parse_prometheus`,
   `triton_avg_queue_ms` and `triton_avg_batch_size` (yours with `USE_EXERCISES=1`), and the raw lines.
2. `make stats M=text_classifier` (Triton's own statistics API): same numbers?
3. In Prometheus (Graph tab), compare `nv_inference_queue_duration_us` (a raw counter) with
   `rate(nv_inference_queue_duration_us[1m])` and the recording rule `triton:avg_queue_ms`
   (`prometheus/alerts.yml`).

**Answer.** Why is a counter divided by a counter an average *since start*, and how does `rate()` fix that?
What does `nv_inference_count / nv_inference_exec_count` measure?

## Task 2 — PromQL and percentiles

**Scenario.** *"After scaling to 4 replicas the team averages each pod's P95 on the dashboard. Is that
the service's P95?"*

Run `make gpu-08` in another terminal for traffic, then query:

```promql
sum by (model) (rate(nv_inference_request_success[1m]))                 # requests/s
triton:avg_queue_ms                                                       # recording rule
nv_inference_request_summary_us{quantile="0.95"} / 1000                   # P95 per pod (summary)
sum by (model) (nv_inference_pending_request_count)                       # requests waiting
DCGM_FI_DEV_GPU_UTIL                                                      # GPU busy %
```

Compare the summary P95 with the P95 that `loadgen.py` prints for the same phase.

**Answer.** Why can't you average or sum summary quantiles across pods? What would you need instead
(calculation 7, `merge_histograms` + `histogram_quantile`)? Why is the server-side P95 lower than the
client-side one?

## Task 3 — Load phases on the dashboard

**Scenario.** *"Match symptoms to causes: which panels move first when traffic bursts, and which when
requests start failing?"*

`make gpu-08` runs **steady** (2 concurrent, 60 s) → **burst** (64, 90 s) → **bad requests** (8, 150 s,
20% malformed). Watch Grafana and write down, per phase: requests/s, avg queue ms, avg batch size, error
ratio, P95, pending requests, GPU utilisation and memory.

**Answer.** In the burst, which rose more: compute time or queue time? What happened to the average batch
size, and why is that good? Did GPU utilisation reach 100%? Which golden signal does each panel represent?

## Task 4 — Make alerts fire, then fix the cause

**Scenario.** *"Page only for user-facing problems, but never miss a fast error-budget burn."*

Read `prometheus/alerts.yml` first (`expr`, `for`, labels).

1. Queue time: `make load-08 PHASES="overload:96:180"` until
   `TritonQueueTimeHigh` goes **pending** then **firing** (Prometheus → Alerts, or `make alerts`).
2. Fix the cause, not the alert: give `text_classifier` 2 instances and batching (Lab 07 tasks 2–3),
   `make reload M=text_classifier`, and run the same overload. Does it still fire?
3. Error budget: the *bad requests* phase of `make gpu-08` fires `TritonErrorBudgetFastBurn`.
   Compute the burn rate by hand (calculation 5).

**Answer.** What does `for: 1m` prevent? Why does the error-budget alert compare the error ratio with
14.4 × 0.1%? Which of these alerts should page a human at night, and which should open a ticket?

## Task 5 — Failure drills

**Scenario.** *"Know what each failure looks like on the dashboards before it happens in production."*

1. **Server down:** `docker compose stop triton`, wait 45 s, `make alerts` (`TritonDown`),
   then `docker compose start triton`. What did the Triton panels show meanwhile?
2. **Model unloaded mid-traffic:** start `make gpu-08`, then `make unload M=text_classifier`; watch the
   error ratio; `make load M=text_classifier`. Is the server *live*? *ready*?
3. **GPU memory pressure:** `make gpu-hog GB=21` (holds 21 GB for 5 minutes). Watch *GPU memory used*;
   `GPUMemoryNearlyFull` fires after 1 minute. While it holds, `make reload M=text_classifier`: what happens?
4. **GPU hardware errors (read only):** look up XID 48 (double-bit ECC), 79 (GPU fallen off the bus) and
   the `GPUXidErrors` rule. What should automation do with the node?

**Answer.** Which alerts are symptoms (users feel them) and which are causes? Why does an outage make
latency panels go *empty* rather than red?

## Task 6 — Drift

**Scenario.** *"A new client started sending support tickets to the sentiment endpoint. Nothing errors.
How do you notice, and should you retrain?"*

`make drift-08` (or `USE_EXERCISES=1 make drift-08`): 400 reference requests (short reviews) and 400 current
requests (long log-filled tickets) through the live model; PSI and KS on input length and model
confidence; label mix.

**Answer.** Which PSI band is each signal in? Which is *input* drift and which *prediction* drift? Why is
retraining on these inputs the wrong reaction here?

## Task 7 — Canary with real metrics

Lab 07 task **K6**: `make k8s-canary`, traffic, `make k8s-canary-check` (your `canary_verdict`), rollback.

**Answer.** Why must stable and canary be compared over the same time window? Which metrics belong in the
verdict besides latency and errors for an LLM (quality, refusals, cost)?

---

## Calculations (`exercises.py`, laptop)

`make test-08` from the repo root.

| # | Function | Used in |
|---|---|---|
| 1 | `latency_summary` | loadgen phases: nearest-rank P50/P95/P99 |
| 2 | `parse_prometheus`, `triton_avg_queue_ms`, `triton_avg_batch_size` | task 1 |
| 3 | `rolling_zscore_anomalies` | spotting spikes in a metric series |
| 4 | `psi`, `ks_statistic` | task 6 |
| 5 | `burn_rate`, `should_page` | task 4 |
| 6 | `canary_route`, `canary_verdict` | task 7 (K6) |
| 7 | `merge_histograms`, `histogram_quantile` | task 2: percentiles across replicas |
| 8 | `backoff_delays` | retries without a thundering herd |
| 9 | `embedding_drift` | drift for text inputs |

## Concepts

### 1. What to monitor for an LLM / inference service

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

### 2. Anomalies and drift

- **Point anomalies:** a rolling z-score (value vs trailing mean/std), EWMA control charts, or
  seasonal baselines (daily and weekly traffic patterns).
- **Data drift:** compare the production distribution with the training/reference distribution.
  - **PSI** (population stability index) = Σ (a−e)·ln(a/e) over bins. **< 0.1 stable,
    0.1–0.25 moderate, > 0.25 significant drift.**
  - **KS test** (max CDF gap) for continuous features. For text, compare **embedding** distributions,
    prompt lengths, topic or intent mix, language mix.
- **Concept drift:** the input→label relationship changes. It needs labels (delayed feedback) to detect.
- **Prediction drift:** the output distribution shifts (for example the refusal rate doubles). This is an early warning without labels.

### 3. SLOs, error budgets and alerting

- **SLO:** for example 99.9% of requests succeed with P95 < 500 ms over 30 days. The **error budget** is 0.1%.
- **Burn rate** = observed error rate / (1 − SLO). A burn rate of 1 uses the budget exactly
  over the window. Alert on **fast burn** (for example 14.4× over 1 h) and **slow burn** (for example 6× over 6 h).
  Multi-window alerts avoid paging on blips.
- Alert on symptoms (latency, errors) rather than causes. Every alert needs a runbook.

### 4. Lifecycle: versions, rollout, retraining

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

### 5. Exam traps

- CPU metrics are poor autoscaling and alerting signals for GPU inference. Use queue time and GPU metrics.
- Averages of percentiles are wrong. Aggregate histograms, then compute percentiles.
- Drift alone doesn't mean accuracy dropped. Use it as a trigger for investigation or evaluation, not blind retraining.
- Canary compares **the same time window**. Comparing today's canary with last week's baseline confounds traffic changes.
- Version everything together: model weights **and** tokenizer, prompt templates, guardrail configs and engine build settings.

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
