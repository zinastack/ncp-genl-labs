"""Lab 06 — GPU Acceleration & Distributed Training. Fill in every TODO, then run:

    pytest labs/3-optimization-acceleration/06-gpu-acceleration
"""

import numpy as np
import torch
import torch.distributed as dist


# 1 ─────────────────────────────────────────────────────────────────────────────
def zero_bytes_per_gpu(n_params: float, n_gpus: int, stage: int) -> float:
    """Model-state bytes per GPU under ZeRO (mixed-precision Adam: 2 B params, 2 B grads,
    12 B optimizer states per parameter). stage 0 = plain DDP, 3 = FSDP full shard.
    """
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def pipeline_bubble_fraction(stages: int, micro_batches: int, virtual_stages: int = 1) -> float:
    """Fraction of the step each GPU is idle: bubble / (compute + bubble), where
    compute = micro_batches and bubble = (stages − 1) / virtual_stages (in micro-batch units).
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def ring_allreduce_bytes(size_bytes: float, n_gpus: int) -> float:
    """Bytes each GPU sends in a ring all-reduce (reduce-scatter + all-gather): 2(N−1)/N · size."""
    raise NotImplementedError


def allreduce_seconds(size_bytes: float, n_gpus: int, bus_bandwidth: float) -> float:
    """Time for the ring all-reduce if each GPU sends at bus_bandwidth bytes/s."""
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def rank_coords(rank: int, world_size: int, tp: int, pp: int) -> tuple[int, int, int]:
    """Megatron-style layout with TP fastest-varying, then DP, then PP:
        rank = tp_rank + tp * (dp_rank + dp * pp_rank),  dp = world_size // (tp * pp)
    Return (tp_rank, dp_rank, pp_rank). Raise ValueError if world_size % (tp*pp) != 0.
    """
    raise NotImplementedError


def tensor_parallel_group(rank: int, tp: int) -> list[int]:
    """All ranks that share `rank`'s TP group (consecutive ranks → same node → NVLink)."""
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def column_parallel(x: np.ndarray, w: np.ndarray, n: int) -> list[np.ndarray]:
    """Split w (d_in, d_out) into n column shards; each 'GPU' computes x @ shard.
    Return the list of partial outputs (each (batch, d_out/n)). No communication yet.
    """
    raise NotImplementedError


def row_parallel(x_shards: list[np.ndarray], w: np.ndarray, n: int) -> np.ndarray:
    """Split w (d_in, d_out) into n ROW shards; GPU i computes x_shards[i] @ row_shard_i.
    Sum the partial results (this is the all-reduce) and return (batch, d_out).
    """
    raise NotImplementedError


def megatron_mlp(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, n: int) -> np.ndarray:
    """relu(x @ w1) @ w2 with w1 column-parallel and w2 row-parallel: the nonlinearity is
    applied locally on each shard, so the whole MLP needs just ONE all-reduce (the sum).
    """
    raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def average_gradients(model: torch.nn.Module) -> None:
    """DDP's core step: all-reduce (SUM) every parameter's .grad across ranks, then divide by
    the world size, so every rank ends with the average gradient. Use torch.distributed.
    """
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def training_flops(n_params: float, n_tokens: float) -> float:
    """≈ 6 · N · D (2 for forward, 4 for backward)."""
    raise NotImplementedError


def mfu(tokens_per_sec: float, n_params: float, n_gpus: int, peak_flops_per_gpu: float) -> float:
    """Model FLOPs utilisation = achieved training FLOP/s ÷ aggregate peak FLOP/s."""
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def gemm_arithmetic_intensity(m: int, n: int, k: int, bytes_per_elem: float = 2) -> float:
    """FLOPs per byte for C(m,n) = A(m,k) @ B(k,n): 2mnk FLOPs over reading A and B and
    writing C once, i.e. bytes_per_elem · (mk + kn + mn).
    """
    raise NotImplementedError


def is_memory_bound(intensity: float, peak_flops: float, mem_bandwidth: float) -> bool:
    """True if intensity is below the roofline ridge point peak_flops / mem_bandwidth."""
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
# Simulated collectives. `per_rank[i]` is the 1-D array held by rank i; return what each rank holds after.
def all_reduce(per_rank: list[np.ndarray]) -> list[np.ndarray]:
    """Every rank receives the element-wise SUM of all ranks' arrays."""
    raise NotImplementedError


def reduce_scatter(per_rank: list[np.ndarray]) -> list[np.ndarray]:
    """Sum across ranks, then rank i keeps only slice i of the sum (np.array_split into N parts)."""
    raise NotImplementedError


def all_gather(shards: list[np.ndarray]) -> list[np.ndarray]:
    """Every rank receives the concatenation of all ranks' shards, in rank order."""
    raise NotImplementedError


def all_to_all(send: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """send[i][j] is what rank i sends to rank j. Return recv where recv[j][i] = send[i][j]."""
    raise NotImplementedError


# 10 ────────────────────────────────────────────────────────────────────────────
def ring_attention_rank(q: np.ndarray, kv_shards: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Context parallelism, seen from one rank. q (n_q, d) are this rank's queries; kv_shards is the
    list of (k_block, v_block) arriving from each rank around the ring. Combine them with the
    online softmax (running max m, denominator l, accumulator acc, as in Lab 01 exercise 14) and
    return exact (non-causal) attention of q over all keys: (n_q, d_v).
    """
    raise NotImplementedError


# 11 ────────────────────────────────────────────────────────────────────────────
def gradient_buckets(param_bytes: list[int], bucket_cap_bytes: int) -> list[list[int]]:
    """DDP gradient bucketing. Walk parameter indices in REVERSE order (the order backward produces
    gradients) and fill buckets: start a new bucket when adding the next parameter would exceed the
    cap (a single oversized parameter gets its own bucket). Return the buckets as lists of indices.
    """
    raise NotImplementedError


# 12 ────────────────────────────────────────────────────────────────────────────
def timeline_stats(events: list[tuple[float, float, str]]) -> dict[str, float]:
    """Read a GPU timeline like Nsight Systems. events: (start, end, kind), kind "compute" or "nccl";
    intervals may overlap. With span = last end − first start, return fractions of span:
      "gpu_busy"     = time covered by ANY event
      "compute"      = time covered by compute events
      "exposed_comm" = gpu_busy − compute   (NCCL time not hidden behind compute)
    """
    raise NotImplementedError


# 13 ────────────────────────────────────────────────────────────────────────────
def scaling_efficiency(throughput_1: float, throughput_n: float, n: int) -> float:
    """throughput_n / (n × throughput_1)."""
    raise NotImplementedError


def amdahl_speedup(parallel_fraction: float, n: int) -> float:
    """Amdahl's law: 1 / ((1 − p) + p / n)."""
    raise NotImplementedError
