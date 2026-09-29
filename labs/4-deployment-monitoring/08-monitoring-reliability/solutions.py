"""Lab 08 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import hashlib
import math
import re

import numpy as np


def _nearest_rank(sorted_vals, p):
    # The value at position ceil(p% · n) − 1 of the sorted list (no interpolation).
    return sorted_vals[max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)]


def latency_summary(latencies_ms):
    s = sorted(latencies_ms)
    return {
        "p50": _nearest_rank(s, 50),
        "p95": _nearest_rank(s, 95),
        "p99": _nearest_rank(s, 99),
        "mean": sum(s) / len(s),  # reported for contrast: the tail is what users feel
        "max": s[-1],
    }


# name{labels} value, e.g. nv_inference_count{model="m",version="1"} 1200
_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            name, labels, value = m.groups()
            # frozenset: hashable (usable as a dict key) and independent of label order.
            out[(name, frozenset(_LABEL.findall(labels or "")))] = float(value)
    return out


def _model_metric(metrics, name, model):
    # Sum over all series of this metric for the model (e.g. across versions).
    return sum(v for (n, labels), v in metrics.items() if n == name and ("model", model) in labels)


def triton_avg_queue_ms(metrics, model):
    # Counters are running totals, so the ratio gives the average per request.
    return _model_metric(metrics, "nv_inference_queue_duration_us", model) / \
        _model_metric(metrics, "nv_inference_request_success", model) / 1000


def triton_avg_batch_size(metrics, model):
    # Inferences (batch items) per model execution = average dynamic batch size.
    return _model_metric(metrics, "nv_inference_count", model) / _model_metric(metrics, "nv_inference_exec_count", model)


def rolling_zscore_anomalies(values, window, threshold=3.0):
    out = []
    for i in range(window, len(values)):
        prev = np.asarray(values[i - window : i], dtype=float)  # history BEFORE i (the spike must not hide itself)
        std = prev.std()
        # std 0 = perfectly flat history: skip to avoid dividing by zero.
        if std > 0 and abs(values[i] - prev.mean()) > threshold * std:
            out.append(i)
    return out


def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))  # equal-count bins of the reference
    edges[0], edges[-1] = -np.inf, np.inf  # every new value lands in some bin
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)  # clip: no ln(0)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    xs = np.concatenate([a, b])
    # Empirical CDF of each sample at every observed point; KS = the largest vertical gap.
    cdf_a = np.searchsorted(a, xs, side="right") / len(a)
    cdf_b = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def burn_rate(errors, total, slo):
    return 0.0 if total == 0 else (errors / total) / (1 - slo)  # observed ÷ allowed error rate


def should_page(short_window, long_window, slo, threshold=14.4):
    # Short window: it's happening now. Long window: it's not a brief blip. Page only if both.
    return burn_rate(*short_window, slo) > threshold and burn_rate(*long_window, slo) > threshold


def retraining_decision(psi_value, accuracy, baseline_accuracy, new_labeled, max_drop=0.03, min_new_labeled=5000):
    reasons = []  # reasons make the decision auditable
    if psi_value > 0.25:
        reasons.append("drift")
    if baseline_accuracy - accuracy > max_drop:
        reasons.append("quality_drop")
    if new_labeled >= min_new_labeled:
        reasons.append("new_data")
    return bool(reasons), reasons


class ModelRegistry:
    def __init__(self):
        self._versions = {}   # version -> {"metrics": ..., "stage": ...}
        self._history = []    # production versions in promotion order

    def register(self, version, metrics):
        if version in self._versions:
            raise ValueError(f"version {version} already registered")
        self._versions[version] = {"metrics": dict(metrics), "stage": "staging"}

    def promote(self, version):
        # Evaluation gate first: a model that failed evaluation can never reach production.
        if not self._versions[version]["metrics"].get("eval_passed"):
            raise ValueError(f"version {version} has not passed evaluation")
        current = self.production()
        if current is not None:
            self._versions[current]["stage"] = "archived"
        self._versions[version]["stage"] = "production"
        self._history.append(version)

    def production(self):
        return next((v for v, info in self._versions.items() if info["stage"] == "production"), None)

    def rollback(self):
        if len(self._history) < 2:
            raise ValueError("nothing to roll back to")
        current = self._history.pop()  # undo the latest promotion
        previous = self._history[-1]
        self._versions[current]["stage"] = "archived"
        self._versions[previous]["stage"] = "production"
        return previous

    def stage(self, version):
        return self._versions[version]["stage"]


def canary_route(request_key, canary_percent):
    # Stable hash → the same user always gets the same version (sticky routing).
    bucket = int(hashlib.md5(request_key.encode()).hexdigest(), 16) % 100
    return "canary" if bucket < canary_percent else "stable"


def canary_verdict(stable, canary, max_p95_regression=0.10, max_error_increase=0.005):
    # Both dicts must come from the SAME time window, or traffic changes confuse the comparison.
    if canary["p95_ms"] > stable["p95_ms"] * (1 + max_p95_regression):
        return "rollback"
    if canary["error_rate"] - stable["error_rate"] > max_error_increase:
        return "rollback"
    return "promote"
