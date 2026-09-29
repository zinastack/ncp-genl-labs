"""Lab 08 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import hashlib
import math
import re
from collections.abc import Callable

import numpy as np


def _nearest_rank(sorted_vals: list[float], p: float) -> float:
    # The value at position ceil(p% · n) − 1 of the sorted list (no interpolation).
    return sorted_vals[max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)]


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
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


def parse_prometheus(text: str) -> dict[tuple[str, frozenset], float]:
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


def _model_metric(metrics: dict[tuple[str, frozenset], float], name: str, model: str) -> float:
    # Sum over all series of this metric for the model (e.g. across versions).
    return sum(v for (n, labels), v in metrics.items() if n == name and ("model", model) in labels)


def triton_avg_queue_ms(metrics: dict[tuple[str, frozenset], float], model: str) -> float:
    # Counters are running totals, so the ratio gives the average per request.
    return _model_metric(metrics, "nv_inference_queue_duration_us", model) / \
        _model_metric(metrics, "nv_inference_request_success", model) / 1000


def triton_avg_batch_size(metrics: dict[tuple[str, frozenset], float], model: str) -> float:
    # Inferences (batch items) per model execution = average dynamic batch size.
    return _model_metric(metrics, "nv_inference_count", model) / _model_metric(metrics, "nv_inference_exec_count", model)


def rolling_zscore_anomalies(values: list[float], window: int, threshold: float = 3.0) -> list[int]:
    out = []
    for i in range(window, len(values)):
        prev = np.asarray(values[i - window : i], dtype=float)  # history BEFORE i (the spike must not hide itself)
        std = prev.std()
        # std 0 = perfectly flat history: skip to avoid dividing by zero.
        if std > 0 and abs(values[i] - prev.mean()) > threshold * std:
            out.append(i)
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))  # equal-count bins of the reference
    edges[0], edges[-1] = -np.inf, np.inf  # every new value lands in some bin
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)  # clip: no ln(0)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    xs = np.concatenate([a, b])
    # Empirical CDF of each sample at every observed point; KS = the largest vertical gap.
    cdf_a = np.searchsorted(a, xs, side="right") / len(a)
    cdf_b = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def burn_rate(errors: int, total: int, slo: float) -> float:
    return 0.0 if total == 0 else (errors / total) / (1 - slo)  # observed ÷ allowed error rate


def should_page(
    short_window: tuple[int, int], long_window: tuple[int, int], slo: float,
    threshold: float = 14.4,
) -> bool:
    # Short window: it's happening now. Long window: it's not a brief blip. Page only if both.
    return burn_rate(*short_window, slo) > threshold and burn_rate(*long_window, slo) > threshold


def retraining_decision(
    psi_value: float, accuracy: float, baseline_accuracy: float, new_labeled: int,
    max_drop: float = 0.03, min_new_labeled: int = 5000,
) -> tuple[bool, list[str]]:
    reasons = []  # reasons make the decision auditable
    if psi_value > 0.25:
        reasons.append("drift")
    if baseline_accuracy - accuracy > max_drop:
        reasons.append("quality_drop")
    if new_labeled >= min_new_labeled:
        reasons.append("new_data")
    return bool(reasons), reasons


class ModelRegistry:
    def __init__(self) -> None:
        self._versions = {}   # version -> {"metrics": ..., "stage": ...}
        self._history = []    # production versions in promotion order

    def register(self, version: str, metrics: dict) -> None:
        if version in self._versions:
            raise ValueError(f"version {version} already registered")
        self._versions[version] = {"metrics": dict(metrics), "stage": "staging"}

    def promote(self, version: str) -> None:
        # Evaluation gate first: a model that failed evaluation can never reach production.
        if not self._versions[version]["metrics"].get("eval_passed"):
            raise ValueError(f"version {version} has not passed evaluation")
        current = self.production()
        if current is not None:
            self._versions[current]["stage"] = "archived"
        self._versions[version]["stage"] = "production"
        self._history.append(version)

    def production(self) -> str | None:
        return next((v for v, info in self._versions.items() if info["stage"] == "production"), None)

    def rollback(self) -> str:
        if len(self._history) < 2:
            raise ValueError("nothing to roll back to")
        current = self._history.pop()  # undo the latest promotion
        previous = self._history[-1]
        self._versions[current]["stage"] = "archived"
        self._versions[previous]["stage"] = "production"
        return previous

    def stage(self, version: str) -> str:
        return self._versions[version]["stage"]


def canary_route(request_key: str, canary_percent: float) -> str:
    # Stable hash → the same user always gets the same version (sticky routing).
    bucket = int(hashlib.md5(request_key.encode()).hexdigest(), 16) % 100
    return "canary" if bucket < canary_percent else "stable"


def canary_verdict(
    stable: dict[str, float], canary: dict[str, float], max_p95_regression: float = 0.10,
    max_error_increase: float = 0.005,
) -> str:
    # Both dicts must come from the SAME time window, or traffic changes confuse the comparison.
    if canary["p95_ms"] > stable["p95_ms"] * (1 + max_p95_regression):
        return "rollback"
    if canary["error_rate"] - stable["error_rate"] > max_error_increase:
        return "rollback"
    return "promote"


def merge_histograms(per_replica_counts: list[list[int]]) -> list[int]:
    # Cumulative bucket counts from several replicas simply ADD UP (same bucket bounds).
    return [sum(c) for c in zip(*per_replica_counts)]


def histogram_quantile(bounds: list[float], cumulative_counts: list[int], q: float) -> float:
    # Prometheus-style: find the bucket holding the q-th observation, then interpolate linearly
    # inside it (observations are assumed evenly spread within a bucket).
    rank = q * cumulative_counts[-1]
    prev_bound, prev_count = 0.0, 0
    for bound, count in zip(bounds, cumulative_counts):
        if count >= rank:
            if math.isinf(bound):  # can't interpolate into +Inf: report the last finite bound
                return prev_bound
            return prev_bound + (bound - prev_bound) * (rank - prev_count) / (count - prev_count)
        prev_bound, prev_count = bound, count
    return prev_bound


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout: float) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = "closed"  # closed = normal; open = fail fast; half_open = probing
        self.failures = 0
        self.opened_at = 0.0

    def allow(self, now: float) -> bool:
        if self.state == "open" and now - self.opened_at >= self.reset_timeout:
            self.state = "half_open"  # cool-down over: let a trial request through
        return self.state != "open"  # while open, fail fast (serve a fallback) without calling

    def record(self, success: bool, now: float) -> None:
        if success:
            self.state, self.failures = "closed", 0
            return
        self.failures += 1
        # A failed probe, or too many consecutive failures, opens the circuit.
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state, self.opened_at = "open", now


def backoff_delays(attempts: int, base: float, cap: float, jitter: Callable[[float], float]) -> list[float]:
    # Exponential backoff (base, 2·base, 4·base, … capped) with "full jitter": a random wait up to
    # that bound, so thousands of clients don't retry in synchronised waves.
    return [jitter(min(cap, base * 2**i)) for i in range(attempts)]


def embedding_drift(reference: np.ndarray, current: np.ndarray) -> float:
    # Text drift via embeddings: 1 − cosine similarity of the two centroids.
    # 0 = same typical meaning; larger = users are talking about different things.
    a, b = reference.mean(axis=0), current.mean(axis=0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def gpu_node_action(metrics: dict[str, float], max_temp_c: float = 85.0) -> tuple[str, list[str]]:
    # Turn DCGM health metrics into an operational decision.
    reasons = []
    if metrics.get("DCGM_FI_DEV_XID_ERRORS", 0) > 0:
        reasons.append("xid")  # driver-reported GPU faults
    if metrics.get("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", 0) > 0:
        reasons.append("ecc_dbe")  # uncorrectable (double-bit) memory errors: results can be corrupt
    if reasons:
        return "drain", reasons  # cordon + drain the node, run DCGM diagnostics
    if metrics.get("DCGM_FI_DEV_GPU_TEMP", 0) >= max_temp_c:
        return "alert", ["temperature"]  # thermal throttling risk: investigate cooling
    return "ok", []
