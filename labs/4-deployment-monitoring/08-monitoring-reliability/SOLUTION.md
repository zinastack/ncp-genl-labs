# Lab 08 — Solution walkthrough

For each task: **what you should see**, **why**, and **on the exam**. Then the calculations, and the
reliability concepts that have no exercise any more (they are drills or design questions).

```
Triton :8002 ─┐                          ┌─► Grafana dashboards (tasks 1–3)
DCGM   :9400 ─┼─► Prometheus (rules) ────┼─► alerts: queue, error budget, down, GPU (tasks 4–5)
loadgen/drift ┘                          └─► drift (6), canary verdict (7)
```

| Layer | Signals | Source |
|

---

## Task 1 — From raw counters to dashboard numbers

**What you see.** The script's averages match `make stats`. The raw counters only ever grow;
`rate(counter[1m])` is the per-second increase over the last minute; `triton:avg_queue_ms` divides two
rates.

**Why.** `nv_inference_queue_duration_us / nv_inference_request_success` on raw counters is the average
**since the server started**: a 5-minute spike after a week of traffic barely moves it. Dividing
**rates** gives the average over the window, which is what a dashboard and an alert need.
`nv_inference_count / nv_inference_exec_count` = inferences per model execution = **average batch size**
formed by the dynamic batcher.

**On the exam.** *"The queue-time panel never reacts to incidents."* → it plots counter ratios since
start: use `rate()` over a window. Lowering the scrape interval or adding replicas doesn't change that.

## Task 2 — PromQL and percentiles

**What you see.** The summary P95 per pod is close to, and lower than, the client P95: the server
measures from request arrival to response inside Triton; the client adds HTTP, JSON and network time.

**Why.** A **summary** computes quantiles inside each process. Quantiles don't add: the average of two
pods' P95 is not the P95 of their combined traffic (a slow pod with little traffic gets the same weight).
A **histogram** exports cumulative bucket counts; buckets from all pods can be summed, and
`histogram_quantile(0.95, sum by (le) (rate(..._bucket[5m])))` gives the true fleet P95 (calculation 7).
Triton's histograms currently cover time to first response; for request latency it offers counters and
summaries, so across replicas you either query per pod (worst pod) or put a histogram at the gateway.

**On the exam.** *"Compute the P95 across 8 replicas correctly."* → aggregate **histogram buckets**, then
take the quantile. Averaging per-pod P95s, taking the max of averages, or averaging latency are wrong.

## Task 3 — Load phases on the dashboard

**What you see.**

| phase | requests/s | queue ms | avg batch | error ratio | GPU util |
|---|---|---|---|---|---|
| steady | low | ≈ 0 | ≈ 1 | 0 | low |
| burst | high, plateaus | **rises sharply** | **rises** (4–16) | 0 | high, often < 100% |
| bad requests | medium | low | small | **≈ 20%** | low–medium |

**Why.** In the burst the GPU's compute per request stays roughly constant; requests **wait** for an
instance: saturation shows up as **queue time** and **pending requests** first. Larger batches are the
dynamic batcher absorbing load, which raises throughput. GPU utilisation can stay below 100% because a
Python-backend model also spends time on the CPU (tokenization), which is why queue time, not GPU
utilisation, is the better saturation signal. Golden signals: latency (P95), traffic (req/s),
errors (error ratio), saturation (queue, pending, GPU memory).

## Task 4 — Alerts

**What you see.** `TritonQueueTimeHigh` goes pending, then firing after 1 minute of sustained overload.
With 2 instances and batching, the same overload produces a lower queue time, and the alert may not fire
at all. The bad-requests phase fires `TritonErrorBudgetFastBurn` after 2 minutes.

**Why.**
- `for: 1m` requires the condition to hold for a minute: no pages for a single slow scrape.
- Burn rate = error ratio / (1 − SLO). With a 99.9% SLO the budget is 0.1%; a 20% error ratio burns it
  **200×** faster than sustainable. 14.4× is the standard fast-burn threshold: at that rate a 30-day budget
  is gone in about 2 days (30 / 14.4), and 2% of it in one hour.
- Page on **symptoms users feel** (error-budget burn, latency SLO). Queue time and GPU memory are causes or
  early warnings: tickets or dashboards, unless they predict an imminent breach.

**On the exam.** *"Too many pages from brief latency blips."* → multi-window burn-rate alerts with a `for`
duration. Raising the threshold to an arbitrary value hides real incidents; removing alerts on latency
ignores the SLO.

## Task 5 — Failure drills

**What you see.**
1. Triton stopped: requests fail with connection errors; the Triton panels go **empty** (no new samples),
   not to zero; `up{job="triton"}` becomes 0 and `TritonDown` fires after 30 s.
2. Model unloaded: the server stays **live**, readiness becomes false, requests get HTTP 400 *unknown model*
   or *not ready*, the error ratio jumps; loading it again recovers.
3. GPU hog: memory used jumps to ~22 GB; `GPUMemoryNearlyFull` fires after 1 minute; reloading the model
   may fail with an out-of-memory error, and `make triton-logs` shows it.
4. XID 48 (double-bit ECC) and 79 (fallen off the bus) mean the GPU can't be trusted: cordon and drain the
   node, run `dcgmi diag`, replace or reset the GPU (see *GPU health* below).

**Why.** When a target is down there is nothing to scrape, so rate-based panels have no data: alert on
`up == 0` (or `absent()`), never only on thresholds of the missing metric. Errors and latency are symptoms;
GPU memory, XID and queue time are causes.

## Task 6 — Drift

**What you see.** Input length PSI far above 0.25 (**significant**, KS near 1.0: the lengths barely
overlap). Confidence PSI moderate or significant: the model is less sure on text it wasn't trained for.
The label mix shifts too.

**Why.** Input drift = the distribution of **inputs** moved (length here). Prediction drift = the
distribution of **outputs** moved (confidence, label mix): available immediately, no labels needed.
Concept drift (the input→label relation changes) needs ground truth to detect. Retraining a *sentiment*
model on log tickets would teach it nonsense: drift is a trigger to **investigate**. Here the fix is
routing those requests elsewhere or rejecting them, not retraining.

**On the exam.** *"PSI on prompt length is 0.4, no labels yet, accuracy unknown."* → investigate: sample
the new inputs, run evaluations on them, then decide. Immediate automatic retraining or ignoring a PSI above
0.25 are the wrong reactions.

## Task 7 — Canary with real metrics

See Lab 07 SOLUTION K6. The same time window removes traffic-mix and time-of-day effects. For LLMs the
verdict also needs quality signals (judge scores on sampled traffic, refusal rate, guardrail triggers,
thumbs-down rate) and cost (tokens per request), not just latency and errors.

---

## Calculations

### 1 — `latency_summary`: percentiles, not averages

#### Why it exists

Ten requests `[12, 15, 11, 14, 13, 200, 12, 16, 13, 14]` ms: the **mean is 32 ms**, yet no request
took anywhere near 32 ms. **P50 = 13 ms** is the typical user and **P95 = 200 ms** is the unlucky tail.
SLOs are written on percentiles because users feel the tail.

Nearest-rank percentile: sort, then take position `ceil(p/100 · n) − 1`. For 1…100 ms: P50 = 50, P95 = 95, P99 = 99.

#### On the exam

*The dashboard shows 180 ms average, well under the 500 ms target, but users complain:* track **tail
percentiles (P95/P99) from histograms**, plus **TTFT and ITL** for streaming. Averaging per-replica P99s is statistically wrong (calculation 7).

---

### 2 — `parse_prometheus` + Triton metrics

#### Why it exists

Triton's metrics are **counters** (running totals). Meaning comes from **ratios**:

| question | formula | example |
|---|---|---|
| average queue wait | `nv_inference_queue_duration_us / nv_inference_request_success` | 2,500,000 / 1,000 = **2.5 ms** |
| average dynamic batch size | `nv_inference_count / nv_inference_exec_count` | 1,000 / 200 = **5** |

The key `(name, frozenset(labels))` is hashable and independent of label order. In PromQL, wrap both
sides in `rate(...[1m])` to get recent values; `alerts.yml` defines these as recording rules.

#### On the exam

*inference_count +12,000 and exec_count +1,500 in a minute:* **dynamic batching forms batches of about 8.**
Failures are in `nv_inference_request_failure` and GPU utilisation is in `nv_gpu_utilization` / DCGM.

---

### 3 — `rolling_zscore_anomalies`: spotting spikes

Compare each value with the **previous** window: `[20, 21, 19, 20, 22, 20, 45, 21]`, window 5 → at
index 6, mean 20.4 and std 1.02, so z = 24, an anomaly. Excluding the current point matters, because a spike
included in its own baseline inflates the std and hides itself. Production adds seasonality (weekday vs weekend baselines).

---

### 4 — `psi` and `ks_statistic`: has the input changed?

#### Why it exists

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

#### On the exam

*PSI = 0.34 on prompt length and topic features:* **significant drift → evaluate on recent labelled data,
then retrain through the evaluation gates** if quality dropped. Drift is a trigger to investigate, not proof of degradation.

---

### 5 — `burn_rate` / `should_page`: alerting on SLOs

#### Why it exists

A 99.9% SLO gives an **error budget** of 0.1%. **Burn rate = observed error rate ÷ allowed error rate.**
At 1× the budget lasts exactly the SLO period; at 14.4× a 30-day budget is gone in about 2 days.

| window | errors / total | burn |
|---|---|---|
| 5 min | 200 / 1,000 | 200× |
| 1 h | 1,600 / 100,000 | 16× → **page** (both windows above 14.4) |
| 1 h, blip only | 500 / 100,000 | 5× → no page |

**Page only if both windows burn fast**: the short one proves it's happening now, and the long one proves it isn't a blip.

---

### 6 — `canary_route` / `canary_verdict`

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

### 7 — `histogram_quantile`: aggregating latency correctly

#### Why it exists

With 10 replicas, each reporting its own P95, what is the **service's** P95? **Not the average of the
P95s.** Prometheus solves this with **histograms**: each replica counts requests per latency bucket,
counts **add up** across replicas, and the percentile is computed from the merged counts.

#### How it works

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

#### On the exam

PromQL: `histogram_quantile(0.95, sum by (le) (rate(latency_bucket[5m])))`, which **sums the buckets
first, then takes the quantile**. Averaging percentiles is a classic wrong answer.

---

### 8 — `backoff_delays`: retry without a stampede

#### Why it exists

Retrying immediately hammers a struggling service. Retrying on a fixed schedule makes **thousands of
clients retry at the same moment** (a thundering herd). **Exponential backoff with full jitter** spreads retries out:

```
bounds (base 0.1 s, cap 2 s): 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0
each actual wait = random between 0 and the bound
```

Always pair retries with **timeouts** and a **retry limit**, and retry only **idempotent** requests.

#### On the exam (Select TWO)

*A slow LLM dependency takes the whole API down:* ✅ **timeouts with bounded retries and exponential
backoff with jitter**; ✅ **a circuit breaker with a fallback**. ❌ unlimited retries, ❌ no timeouts,
❌ scaling the API tier 10× (it doesn't fix the dependency and can overload it further).

---

### 9 — `embedding_drift`: drift for text

#### Why it exists

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

## Concepts without an exercise

These used to be exercises; they are now covered by the drills above and by design questions on the exam.

### Retraining decision

Three common triggers, logged as **reasons** for auditability: `drift` (PSI > 0.25), `quality_drop`
(accuracy fell more than 3 points), `new_data` (≥ 5,000 new labels). A retrained model still passes the **same
evaluation gates** before promotion. Never auto-deploy, and don't retrain on a fixed schedule regardless of evidence.

---

### Model registry: versions, gates, rollback

```
register v1, v2, v3 (eval failed) → promote v1 → promote v2 (v1 archived)
promote v3 → refused (evaluation gate)  → rollback → v1 production, v2 archived
```

- The **evaluation gate** sits inside `promote`: a failing model can't reach production.
- **One-step rollback** restores the previous production version.
- Version everything that changes behaviour: **weights, tokenizer, prompt templates, generation parameters, guardrail config**.
  *"The weights are fine but prod quality dropped; the template differs from evaluation"* is exactly this gap.

---

### Circuit breaker: stop calling a failing dependency

#### Why it exists

When a downstream LLM endpoint becomes slow or broken, callers keep waiting and retrying. Threads
pile up and **your** service goes down too (cascading failure). A **circuit breaker** notices repeated
failures and **fails fast** for a while (serve a fallback: a smaller model, a cached answer, a
polite error), then **probes** to see whether the dependency has recovered.

```
closed ──(3 failures)──► open ──(30 s cool-down)──► half_open ──success──► closed
                          ▲                             │
                          └───────── failure ───────────┘
```

Typical settings: open after 3 consecutive failures, stay open 30 s, then allow one trial request
(half-open); a failed trial re-opens it immediately, a successful one closes it. Libraries and service
meshes (Envoy/Istio outlier detection) implement this for you.

---

### GPU health: acting on DCGM signals

#### Why it exists

GPUs fail in ways that don't crash immediately: a GPU with **uncorrectable memory errors** can return
**silently corrupted results**. **DCGM** (through dcgm-exporter into Prometheus) reports health, and
some signals demand removing the node.

| DCGM signal | meaning | action |
|---|---|---|
| `DCGM_FI_DEV_XID_ERRORS > 0` | driver-reported GPU fault | **drain** |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0` | uncorrectable (double-bit) ECC errors | **drain** |
| `DCGM_FI_DEV_GPU_TEMP ≥ 85 °C` | overheating: throttling, instability | alert, check cooling |

#### On the exam

*One node returns corrupted outputs and later crashes; DCGM shows XID and rising ECC errors:*
**cordon and drain the node, run DCGM diagnostics (`dcgmi diag`), and let the scheduler move workloads.**
"Increase the batch size", "ECC errors are always corrected" (single-bit ones are; double-bit ones aren't)
and "restart hourly" are wrong.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Task / calculation / concept |
|---|---|
| Latency metrics | task 2, calculations 1 and 7 |
| Triton metrics | task 1, calculation 2 |
| Data drift | task 6, calculations 4 and 9 |
| SLO burn rate | task 4, calculation 5 |
| Automated retraining | task 6, concept *Retraining decision* |
| Version management | concept *Model registry* |
| Rollout strategies, canary analysis | task 7, calculation 6 |
| Reliability patterns | task 5, calculation 8, concept *Circuit breaker* |
| GPU health | task 5, concept *GPU health* |
| Alerting on absence | task 5 |
| Saturation signals | task 3 |
| Alerting practice | task 4 |
