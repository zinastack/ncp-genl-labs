# Lab 08 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: once a model is live, you need to know **is it fast, is it working, is it still
accurate**, and be able to **change versions safely**. Each exercise turns raw telemetry into a
decision.

```
latency & errors (1, 5) ─┐
Triton metrics (2)      ─┼─► dashboards + alerts ─► on-call
anomalies (3)           ─┘
drift (4) ─► retrain? (6) ─► registry (7) ─► canary (8) ─► promote / rollback
```

---

## Exercise 1 — `latency_summary`: percentiles, not averages

Ten requests: `[12, 15, 11, 14, 13, 200, 12, 16, 13, 14]` ms, with one slow outlier.

| metric | value | what it tells you |
|---|---|---|
| mean | 32 ms | misleading: no request actually took ~32 ms |
| P50 (median) | 13 ms | the typical user |
| P95 | 200 ms | the unlucky 5%: this is what an SLO targets |
| max | 200 ms | |

The average hides the tail, and users feel the tail. **Nearest-rank percentile:** sort, then take
the value at position `ceil(p/100 · n) − 1`. For 1…100 ms that gives P50 = 50, P95 = 95, P99 = 99.

> **Exam trap:** don't average percentiles across replicas (the average of P99s is not the P99).
> Aggregate histograms, then compute the percentile.

---

## Exercise 2 — `parse_prometheus` + Triton metrics

Triton exposes plain-text metrics on port 8002:

```
nv_inference_request_success{model="text_classifier",version="1"} 1000
nv_inference_count{model="text_classifier",version="1"} 1000
nv_inference_exec_count{model="text_classifier",version="1"} 200
nv_inference_queue_duration_us{model="text_classifier",version="1"} 2500000
```

**Parsing:** skip `#` comment lines and split each line into name, `{labels}` and value. The key is
`(name, frozenset(labels))`, because a frozenset is hashable, so it works as a dict key, and its
label order doesn't matter.

**These are counters** (running totals since start), so meaning comes from **ratios**:

| question | formula | here |
|---|---|---|
| average queue wait | `queue_duration_us / request_success` | 2,500,000 / 1000 = 2,500 µs = **2.5 ms** |
| average dynamic batch size | `inference_count / exec_count` | 1000 / 200 = **5** requests per GPU execution |

In Prometheus you'd write the same ratio with `rate(...[1m])` on top and bottom to get the value
over the last minute. The shipped `alerts.yml` defines these as recording rules.

---

## Exercise 3 — `rolling_zscore_anomalies`: spotting spikes

Compare each new value with the **previous window** (not including itself):

```
values [20, 21, 19, 20, 22, 20, 45, 21], window 5, threshold 3
at index 6: previous 5 = [21, 19, 20, 22, 20] → mean 20.4, std 1.02
z = (45 − 20.4) / 1.02 = 24 > 3 → anomaly at index 6
```

- The window *before* `i` matters: including the spike itself would inflate the std and hide it.
- `std == 0` (a perfectly flat series) is skipped to avoid dividing by zero.
- Production uses seasonal baselines (weekday vs weekend), but the idea is the same.

---

## Exercise 4 — `psi` and `ks_statistic`: has the input changed?

A model trained on one distribution degrades when production inputs **drift**, for example users
start pasting longer documents. Compare today's feature (prompt length here) with the training reference.

### PSI (population stability index)

1. Cut the **reference** into 10 bins with equal counts (deciles, from `np.quantile`), and open the
   outer edges to ±∞ so every new value lands somewhere.
2. Compute the fraction of each sample per bin, clipped to at least 1e-6 so `ln(0)` can't happen.
3. `PSI = Σ (actual − expected) · ln(actual / expected)`.

| production distribution (reference mean 100, sd 20) | PSI | KS |
|---|---|---|
| same (100, 20) | 0.002 | 0.009 |
| mean 105 | 0.057 | 0.101 |
| mean 115, sd 22 | 0.472 | 0.276 |
| mean 130, sd 25 | 1.486 | 0.502 |

Rule of thumb: **< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant drift.**

### KS statistic

The biggest vertical gap between the two **cumulative** distributions:
`[1,2,3]` vs `[4,5]` gives 1.0 (completely separate), and `[1,3,5]` vs `[2,4,6]` gives 0.33 (interleaved).
Code: sort both, evaluate each empirical CDF at every observed point with `searchsorted(side="right")`,
and take the max absolute difference.

> Drift is a **trigger to investigate** (evaluate on fresh labelled data), not proof the model got worse.

---

## Exercise 5 — `burn_rate` / `should_page`: alerting on SLOs

SLO 99.9% success → the **error budget** is 0.1% of requests.

```
burn rate = observed error rate / allowed error rate (1 − SLO)
```

| window | errors / total | error rate | burn rate |
|---|---|---|---|
| 5 min | 200 / 1,000 | 20% | **200×** |
| 1 hour | 1,600 / 100,000 | 1.6% | **16×** |
| 1 hour (blip only) | 500 / 100,000 | 0.5% | 5× |

A burn rate of 1 would use exactly the whole budget over the SLO period. At 14.4× a 30-day budget is
gone in about 2 days. **Page only if both windows exceed the threshold**: the short window proves
it's happening *now*, and the long window proves it's not a 30-second blip. The third row shows a
short spike with a calm hour, so no page.

---

## Exercise 6 — `retraining_decision`: retrain for a reason

Three independent triggers, reported in a fixed order:

| trigger | condition | example |
|---|---|---|
| `drift` | PSI > 0.25 | inputs changed significantly |
| `quality_drop` | baseline − current accuracy > 3 points | 0.92 → 0.88 |
| `new_data` | ≥ 5,000 new labelled examples | enough to learn something new |

Returning the **reasons** (not just True/False) makes the decision auditable. A retrained model
still has to pass the **same evaluation gates** before promotion, so never auto-deploy.

---

## Exercise 7 — `ModelRegistry`: versions, promotion, rollback

State: each version has a stage (`staging` → `production` → `archived`), plus a **history** list of
production versions in promotion order.

```
register v1, v2, v3(eval failed)       all staging
promote v1                              v1 production              history [v1]
promote v2                              v2 production, v1 archived history [v1, v2]
promote v3                              refused: eval_passed is False (the gate)
rollback                                v1 production, v2 archived history [v1]
rollback                                refused: nothing left to roll back to
```

- `promote` checks the **evaluation gate** first, so a failing model can never reach production.
- `rollback` pops the current version from history and restores the previous one: one step, no guessing.
- Real registries (MLflow, W&B, NGC) add lineage (data, code, metrics) and also version the
  tokenizer, prompt templates and guardrail config, because all of them change behaviour.

---

## Exercise 8 — `canary_route` / `canary_verdict`: safe rollouts

**Sticky routing:** hash the user id, take `% 100`, and send the user to the canary if the result is
below the percentage. The same user always lands on the same version (a consistent experience and
clean comparisons), and about 10% of 5,000 users went to the canary (measured: 9.5%).

**Verdict** (same time window for both, so traffic changes don't confuse the comparison):

| canary vs stable (P95 200 ms, errors 0.2%) | result |
|---|---|
| P95 210 ms, errors 0.3% | promote (+5% latency, +0.1 points of errors are within limits) |
| P95 260 ms | rollback (+30% latency) |
| errors 2% | rollback |

Other patterns to know: **shadow** (mirror traffic, discard the responses, zero user risk),
**blue-green** (two full environments, instant switch), **A/B** (statistical comparison of business metrics).

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| average looks fine, users complain | P95/P99 from histograms; TTFT/ITL for LLMs |
| inference_count / exec_count | average batch size |
| queue_duration / request_success | average queue time: a scaling signal |
| PSI 0.3 | significant drift → evaluate, then retrain through gates |
| page on real incidents without noise | multi-window burn-rate alerts |
| validate with zero user impact | shadow deployment |
| canary regressed | automatic rollback |
| weights fine but prod quality dropped | version prompts, templates and params with the model |
| XID / ECC errors | cordon and drain the node, run DCGM diagnostics |
