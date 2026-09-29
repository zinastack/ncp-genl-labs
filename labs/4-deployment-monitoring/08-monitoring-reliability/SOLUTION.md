# Lab 08 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: once a model is live you must know three things: **is it fast, is it working, is it
still accurate?** You also need to **change versions safely** and **survive failures**. Each exercise
turns raw telemetry into a decision.

```
latency (1, 9) & errors (5) ─┐
Triton metrics (2)           ├─► dashboards + alerts ─► on-call
anomalies (3), GPU health (13)┘
input drift (4, 12) ─► retrain? (6) ─► registry (7) ─► canary (8) ─► promote / rollback
failing dependencies ─► timeouts + backoff (11) + circuit breaker (10)
```

| Layer | Signals | Source |
|---|---|---|
| Service ("golden signals") | latency P50/P95/P99, traffic, errors, saturation (queue time) | Triton `:8002/metrics`, NIM, gateway |
| LLM-specific | **TTFT, ITL/TPOT**, tokens/s, token counts, KV-cache use | NIM / TensorRT-LLM / vLLM metrics |
| GPU | utilisation, memory, power, temperature, **XID / ECC errors** | **DCGM exporter** |
| Quality | input/output drift, refusal rate, guardrail triggers, user feedback, judge scores | logs, evaluation jobs |

---

## Exercise 1 — `latency_summary`: percentiles, not averages

### Why it exists

Ten requests `[12, 15, 11, 14, 13, 200, 12, 16, 13, 14]` ms: the **mean is 32 ms**, yet no request
took anywhere near 32 ms. **P50 = 13 ms** is the typical user and **P95 = 200 ms** is the unlucky tail.
SLOs are written on percentiles because users feel the tail.

Nearest-rank percentile: sort, then take position `ceil(p/100 · n) − 1`. For 1…100 ms: P50 = 50, P95 = 95, P99 = 99.

### On the exam

*The dashboard shows 180 ms average, well under the 500 ms target, but users complain:* track **tail
percentiles (P95/P99) from histograms**, plus **TTFT and ITL** for streaming. Averaging per-replica P99s is statistically wrong (exercise 9).

---

## Exercise 2 — `parse_prometheus` + Triton metrics

### Why it exists

Triton's metrics are **counters** (running totals). Meaning comes from **ratios**:

| question | formula | example |
|---|---|---|
| average queue wait | `nv_inference_queue_duration_us / nv_inference_request_success` | 2,500,000 / 1,000 = **2.5 ms** |
| average dynamic batch size | `nv_inference_count / nv_inference_exec_count` | 1,000 / 200 = **5** |

The key `(name, frozenset(labels))` is hashable and independent of label order. In PromQL, wrap both
sides in `rate(...[1m])` to get recent values; `alerts.yml` defines these as recording rules.

### On the exam

*inference_count +12,000 and exec_count +1,500 in a minute:* **dynamic batching forms batches of about 8.**
Failures are in `nv_inference_request_failure` and GPU utilisation is in `nv_gpu_utilization` / DCGM.

---

## Exercise 3 — `rolling_zscore_anomalies`: spotting spikes

Compare each value with the **previous** window: `[20, 21, 19, 20, 22, 20, 45, 21]`, window 5 → at
index 6, mean 20.4 and std 1.02, so z = 24, an anomaly. Excluding the current point matters, because a spike
included in its own baseline inflates the std and hides itself. Production adds seasonality (weekday vs weekend baselines).

---

## Exercise 4 — `psi` and `ks_statistic`: has the input changed?

### Why it exists

A model trained on one input distribution degrades when production inputs **drift**, for example users
paste longer documents or ask about new topics. Compare today's distribution with the training reference.

| production (reference mean 100, sd 20) | PSI | KS |
|---|---|---|
| same | 0.002 | 0.009 |
| mean 105 | 0.057 | 0.101 |
| mean 115, sd 22 | 0.472 | 0.276 |
| mean 130, sd 25 | 1.486 | 0.502 |

**PSI rule of thumb: < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.** PSI uses equal-count bins
of the reference, with outer edges opened to ±∞, and fractions clipped to avoid `ln(0)`. KS is the largest gap between the two cumulative distributions.

### On the exam

*PSI = 0.34 on prompt length and topic features:* **significant drift → evaluate on recent labelled data,
then retrain through the evaluation gates** if quality dropped. Drift is a trigger to investigate, not proof of degradation.

---

## Exercise 5 — `burn_rate` / `should_page`: alerting on SLOs

### Why it exists

A 99.9% SLO gives an **error budget** of 0.1%. **Burn rate = observed error rate ÷ allowed error rate.**
At 1× the budget lasts exactly the SLO period; at 14.4× a 30-day budget is gone in about 2 days.

| window | errors / total | burn |
|---|---|---|
| 5 min | 200 / 1,000 | 200× |
| 1 h | 1,600 / 100,000 | 16× → **page** (both windows above 14.4) |
| 1 h, blip only | 500 / 100,000 | 5× → no page |

**Page only if both windows burn fast**: the short one proves it's happening now, and the long one proves it isn't a blip.

---

## Exercise 6 — `retraining_decision`

Three triggers, returned as **reasons** for auditability: `drift` (PSI > 0.25), `quality_drop`
(accuracy fell more than 3 points), `new_data` (≥ 5,000 new labels). A retrained model still passes the **same
evaluation gates** before promotion. Never auto-deploy, and don't retrain on a fixed schedule regardless of evidence.

---

## Exercise 7 — `ModelRegistry`: versions, gates, rollback

```
register v1, v2, v3 (eval failed) → promote v1 → promote v2 (v1 archived)
promote v3 → refused (evaluation gate)  → rollback → v1 production, v2 archived
```

- The **evaluation gate** sits inside `promote`: a failing model can't reach production.
- **One-step rollback** restores the previous production version.
- Version everything that changes behaviour: **weights, tokenizer, prompt templates, generation parameters, guardrail config**.
  *"The weights are fine but prod quality dropped; the template differs from evaluation"* is exactly this gap.

---

## Exercise 8 — `canary_route` / `canary_verdict`

**Sticky routing:** hash the user id mod 100 and compare with the canary %. The same user always sees
the same version (measured: 9.5% of 5,000 users routed to a 10% canary).
**Verdict** from the **same time window**: roll back if P95 regresses more than 10% or errors rise more than 0.5 points.

| pattern | how | risk / cost |
|---|---|---|
| **shadow** | mirror traffic to the new model, discard its answers | zero user impact; doubles inference cost |
| **canary** | small % of real users, compare, then promote or roll back | limited blast radius |
| **blue-green** | two full environments, instant switch | 2× resources during the switch |
| **A/B test** | statistically powered comparison of business metrics | needs traffic and time |

*"Validate on real traffic with zero user impact"* → shadow. *"Canary: +35% P95, 4× errors"* → automatic rollback.

---

## Exercise 9 — `histogram_quantile`: aggregating latency correctly

### Why it exists

With 10 replicas, each reporting its own P95, what is the **service's** P95? **Not the average of the
P95s.** Prometheus solves this with **histograms**: each replica counts requests per latency bucket,
counts **add up** across replicas, and the percentile is computed from the merged counts.

### How it works

Buckets (seconds) `[0.05, 0.1, 0.25, 0.5, 1.0, +Inf]`, cumulative counts:

```
replica A (900 fast requests): [800, 880, 895, 900, 900, 900]   P95 = 0.084 s
replica B (100 slow requests): [  0,  10,  40,  80, 100, 100]   P95 = 0.875 s
average of the two P95s: 0.480 s   ← wrong: gives the small slow replica half the weight
merged  [800, 890, 935, 980, 1000, 1000] → true P95 = 0.333 s
```

Interpolation: the 950th request falls in bucket (0.25, 0.5], which holds requests 936–980, so
`0.25 + 0.25 × (950 − 935) / (980 − 935) = 0.333`. If the rank lands in the +Inf bucket there's
nothing to interpolate toward, so the last finite bound is returned (as Prometheus does). That's a reason to make the top bucket generous.

### On the exam

PromQL: `histogram_quantile(0.95, sum by (le) (rate(latency_bucket[5m])))`, which **sums the buckets
first, then takes the quantile**. Averaging percentiles is a classic wrong answer.

---

## Exercise 10 — `CircuitBreaker`: stop calling a failing dependency

### Why it exists

When a downstream LLM endpoint becomes slow or broken, callers keep waiting and retrying. Threads
pile up and **your** service goes down too (cascading failure). A **circuit breaker** notices repeated
failures and **fails fast** for a while (serve a fallback: a smaller model, a cached answer, a
polite error), then **probes** to see whether the dependency has recovered.

```
closed ──(3 failures)──► open ──(30 s cool-down)──► half_open ──success──► closed
                          ▲                             │
                          └───────── failure ───────────┘
```

The test walks every transition with a fake clock: three failures open it, calls are rejected while
open, one trial is allowed after 30 s, a failed trial re-opens it immediately, and a successful trial closes it.

---

## Exercise 11 — `backoff_delays`: retry without a stampede

### Why it exists

Retrying immediately hammers a struggling service. Retrying on a fixed schedule makes **thousands of
clients retry at the same moment** (a thundering herd). **Exponential backoff with full jitter** spreads retries out:

```
bounds (base 0.1 s, cap 2 s): 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0
each actual wait = random between 0 and the bound
```

Always pair retries with **timeouts** and a **retry limit**, and retry only **idempotent** requests.

### On the exam (Select TWO)

*A slow LLM dependency takes the whole API down:* ✅ **timeouts with bounded retries and exponential
backoff with jitter**; ✅ **a circuit breaker with a fallback**. ❌ unlimited retries, ❌ no timeouts,
❌ scaling the API tier 10× (it doesn't fix the dependency and can overload it further).

---

## Exercise 12 — `embedding_drift`: drift for text

### Why it exists

PSI and KS need numeric features. For LLM inputs the question is *"are users asking about different
things?"*. Embed a sample of prompts (Lab 01, exercise 10) and compare with the reference period:

```
drift = 1 − cosine(mean embedding of reference, mean embedding of current)
same topic: < 0.05          users moved to a different topic: > 0.8
```

Centroid distance is the simplest signal. Richer ones cluster the embeddings and compare the topic
mix, or compute PSI on cluster assignments. Also track prompt length, language mix, refusal rate and
guardrail triggers. Output drift is an early warning that needs no labels.

---

## Exercise 13 — `gpu_node_action`: acting on GPU health

### Why it exists

GPUs fail in ways that don't crash immediately: a GPU with **uncorrectable memory errors** can return
**silently corrupted results**. **DCGM** (through dcgm-exporter into Prometheus) reports health, and
some signals demand removing the node.

| DCGM signal | meaning | action |
|---|---|---|
| `DCGM_FI_DEV_XID_ERRORS > 0` | driver-reported GPU fault | **drain** |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0` | uncorrectable (double-bit) ECC errors | **drain** |
| `DCGM_FI_DEV_GPU_TEMP ≥ 85 °C` | overheating: throttling, instability | alert, check cooling |

### On the exam

*One node returns corrupted outputs and later crashes; DCGM shows XID and rising ECC errors:*
**cordon and drain the node, run DCGM diagnostics (`dcgmi diag`), and let the scheduler move workloads.**
"Increase the batch size", "ECC errors are always corrected" (single-bit ones are; double-bit ones aren't)
and "restart hourly" are wrong.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise |
|---|---|
| Latency metrics | 1, 9 |
| Triton metrics | 2 |
| Data drift | 4, 12 |
| SLO burn rate | 5 |
| Automated retraining | 6 |
| Version management | 7 |
| Rollout strategies, canary analysis | 8 |
| Reliability patterns | 10, 11 |
| GPU health | 13 |
