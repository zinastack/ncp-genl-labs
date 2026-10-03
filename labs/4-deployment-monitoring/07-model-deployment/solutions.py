"""Lab 07 — reference solutions for the calculations. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import math
from collections.abc import Callable


def simulate_dynamic_batching(
    arrivals_ms: list[float], max_batch: int, max_delay_ms: float,
    compute_ms: Callable[[int], float],
) -> list[float]:
    n = len(arrivals_ms)
    latencies = [0.0] * n
    i, free = 0, 0.0  # i = oldest unserved request; free = when the GPU becomes idle
    while i < n:
        # The batch could be "full" once max_batch requests are waiting.
        full_at = arrivals_ms[i + max_batch - 1] if i + max_batch - 1 < n else math.inf
        # Start when the GPU is free AND (the oldest has waited max_delay OR the batch is full).
        launch = max(free, min(arrivals_ms[i] + max_delay_ms, full_at))
        j = i
        while j < n and j - i < max_batch and arrivals_ms[j] <= launch:  # everyone who has arrived
            j += 1
        finish = launch + compute_ms(j - i)
        for k in range(i, j):
            latencies[k] = finish - arrivals_ms[k]  # queue wait + batching delay + compute
        free, i = finish, j
    return latencies


def replicas_needed(
    requests_per_sec: float, latency_sec: float, concurrency_per_replica: int,
    headroom: float = 0.0,
) -> int:
    in_flight = requests_per_sec * latency_sec * (1 + headroom)  # Little's law: L = λ · W
    # Round UP (fewer replicas would overload); -1e-9 keeps float noise from turning 4.0 into 5.
    return max(1, math.ceil(in_flight / concurrency_per_replica - 1e-9))


def pick_best_config(results: list[dict], max_p95_ms: float, max_gpu_mem_gb: float) -> dict | None:
    # What Model Analyzer does after sweeping configs with perf_analyzer: discard configurations that
    # break a constraint, then take the highest throughput among the rest.
    feasible = [r for r in results if r["p95_ms"] <= max_p95_ms and r["gpu_mem_gb"] <= max_gpu_mem_gb]
    return max(feasible, key=lambda r: r["throughput"], default=None)


def rollout_bounds(replicas: int, max_surge: int, max_unavailable: int) -> tuple[int, int]:
    # During a RollingUpdate Kubernetes keeps total pods ≤ replicas + maxSurge and ready pods ≥
    # replicas − maxUnavailable. For GPU pods, the surge is extra GPUs you must have free.
    return replicas + max_surge, replicas - max_unavailable


def rerank_top_n(query: str, candidates: list[str], score_fn: Callable[[str, str], float], top_n: int) -> list[str]:
    # Retrieve many (high recall), then re-score each (query, passage) PAIR with a slower, more
    # accurate cross-encoder and keep only the best few for the LLM (high precision).
    return sorted(candidates, key=lambda c: -score_fn(query, c))[:top_n]


def kv_cache_tokens(
    free_gpu_gib: float, kv_fraction: float, n_layers: int, n_kv_heads: int, head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    # One token's KV cache: a K and a V vector (×2) per layer, per KV head (GQA shrinks this), in
    # the cache dtype. TensorRT-LLM gives the cache kv_fraction of the memory left after the weights.
    per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem
    return int(free_gpu_gib * 2**30 * kv_fraction // per_token)
