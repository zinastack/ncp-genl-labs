"""Lab 06 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import numpy as np
import torch
import torch.distributed as dist


def zero_bytes_per_gpu(n_params: float, n_gpus: int, stage: int) -> float:
    p, g, o = 2 * n_params, 2 * n_params, 12 * n_params  # bf16 params, bf16 grads, fp32 Adam states
    if stage == 0:
        return p + g + o                # DDP: everything replicated on every GPU
    if stage == 1:
        return p + g + o / n_gpus       # optimizer states sharded
    if stage == 2:
        return p + (g + o) / n_gpus     # + gradients sharded
    if stage == 3:
        return (p + g + o) / n_gpus     # + parameters sharded (= FSDP FULL_SHARD)
    raise ValueError(stage)


def pipeline_bubble_fraction(stages: int, micro_batches: int, virtual_stages: int = 1) -> float:
    # Fill + drain: each GPU idles (p − 1) micro-batch slots per step. Interleaving divides that by v.
    bubble = (stages - 1) / virtual_stages
    return bubble / (micro_batches + bubble)


def ring_allreduce_bytes(size_bytes: float, n_gpus: int) -> float:
    # Reduce-scatter sends (N−1)/N of the data, all-gather sends it again: bandwidth-optimal.
    return 2 * (n_gpus - 1) / n_gpus * size_bytes


def allreduce_seconds(size_bytes: float, n_gpus: int, bus_bandwidth: float) -> float:
    return ring_allreduce_bytes(size_bytes, n_gpus) / bus_bandwidth


def rank_coords(rank: int, world_size: int, tp: int, pp: int) -> tuple[int, int, int]:
    if world_size % (tp * pp):
        raise ValueError("world_size must be divisible by tp * pp")
    dp = world_size // (tp * pp)
    # rank = tp_rank + tp·(dp_rank + dp·pp_rank): TP varies fastest, so a TP group is a block
    # of consecutive ranks, i.e. the same node, connected by NVLink.
    return rank % tp, (rank // tp) % dp, rank // (tp * dp)


def tensor_parallel_group(rank: int, tp: int) -> list[int]:
    start = rank - rank % tp  # round down to the first rank of this TP block
    return list(range(start, start + tp))


def column_parallel(x: np.ndarray, w: np.ndarray, n: int) -> list[np.ndarray]:
    # Each "GPU" owns some COLUMNS of w, so it computes a slice of the output. No communication.
    return [x @ shard for shard in np.split(w, n, axis=1)]


def row_parallel(x_shards: list[np.ndarray], w: np.ndarray, n: int) -> np.ndarray:
    # Each "GPU" owns some ROWS of w and multiplies its input slice → a partial sum of the output.
    # Adding the partials is the all-reduce.
    return sum(xs @ ws for xs, ws in zip(x_shards, np.split(w, n, axis=0)))


def megatron_mlp(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, n: int) -> np.ndarray:
    # Column-parallel W1 gives complete hidden values per slice, so the (element-wise) activation
    # runs locally. Row-parallel W2 then needs just ONE all-reduce for the whole MLP.
    hidden_shards = [np.maximum(h, 0) for h in column_parallel(x, w1, n)]
    return row_parallel(hidden_shards, w2, n)


def average_gradients(model: torch.nn.Module) -> None:
    world = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # in place: every rank now holds the sum
            p.grad /= world  # sum → average, so every replica applies the same update


def training_flops(n_params: float, n_tokens: float) -> float:
    return 6 * n_params * n_tokens  # ≈2 FLOPs/param/token forward + ≈4 backward


def mfu(tokens_per_sec: float, n_params: float, n_gpus: int, peak_flops_per_gpu: float) -> float:
    return 6 * n_params * tokens_per_sec / (n_gpus * peak_flops_per_gpu)  # achieved / peak


def gemm_arithmetic_intensity(m: int, n: int, k: int, bytes_per_elem: float = 2) -> float:
    # FLOPs (a multiply-add per m·n·k) over bytes moved (read A and B, write C once).
    return 2 * m * n * k / (bytes_per_elem * (m * k + k * n + m * n))


def is_memory_bound(intensity: float, peak_flops: float, mem_bandwidth: float) -> bool:
    return intensity < peak_flops / mem_bandwidth  # below the roofline's ridge point


# Simulated collectives: `per_rank[i]` is the tensor held by rank i.
def all_reduce(per_rank: list[np.ndarray]) -> list[np.ndarray]:
    total = sum(per_rank)  # every rank ends with the SUM of everyone's tensor (DDP gradients)
    return [total.copy() for _ in per_rank]


def reduce_scatter(per_rank: list[np.ndarray]) -> list[np.ndarray]:
    # Sum across ranks, but each rank keeps only ITS 1/N slice of the result (ZeRO-2/3 gradients).
    n = len(per_rank)
    return np.array_split(sum(per_rank), n)


def all_gather(shards: list[np.ndarray]) -> list[np.ndarray]:
    full = np.concatenate(shards)  # every rank ends with everyone's pieces (FSDP parameters, TP)
    return [full.copy() for _ in shards]


def all_to_all(send: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    # send[i][j] = what rank i sends to rank j; rank j receives [send[0][j], send[1][j], ...].
    # MoE expert parallelism: every rank routes tokens to the ranks holding their experts.
    n = len(send)
    return [[send[i][j] for i in range(n)] for j in range(n)]


def ring_attention_rank(q: np.ndarray, kv_shards: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    # Context parallelism: this rank owns a slice of the queries; key/value blocks from every
    # rank arrive one by one around the ring. Online softmax (Lab 01, ex. 14) merges them exactly.
    d = q.shape[1]
    m = np.full((q.shape[0], 1), -np.inf)
    l = np.zeros((q.shape[0], 1))
    acc = np.zeros((q.shape[0], kv_shards[0][1].shape[1]))
    for k, v in kv_shards:
        s = q @ k.T / np.sqrt(d)
        m_new = np.maximum(m, s.max(axis=1, keepdims=True))
        p = np.exp(s - m_new)
        c = np.exp(m - m_new)
        l = l * c + p.sum(axis=1, keepdims=True)
        acc = acc * c + p @ v
        m = m_new
    return acc / l  # no rank ever holds the full sequence's K and V


def gradient_buckets(param_bytes: list[int], bucket_cap_bytes: int) -> list[list[int]]:
    # DDP fills buckets in REVERSE parameter order, the order backward produces gradients, so the
    # first bucket's all-reduce can start while backward is still computing earlier layers.
    buckets, current, size = [], [], 0
    for idx in reversed(range(len(param_bytes))):
        if current and size + param_bytes[idx] > bucket_cap_bytes:
            buckets.append(current)
            current, size = [], 0
        current.append(idx)
        size += param_bytes[idx]
    if current:
        buckets.append(current)
    return buckets


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total


def timeline_stats(events: list[tuple[float, float, str]]) -> dict[str, float]:
    # events: (start, end, kind) with kind "compute" or "nccl", like rows in an Nsight Systems timeline.
    span = max(e for _, e, _ in events) - min(s for s, _, _ in events)
    compute = [(s, e) for s, e, k in events if k == "compute"]
    nccl = [(s, e) for s, e, k in events if k == "nccl"]
    busy = _union_length(compute + nccl)
    compute_time = _union_length(compute)
    # Exposed communication = time the GPU is busy with NCCL and NOT hiding it behind compute.
    exposed_comm = busy - compute_time
    return {"gpu_busy": busy / span, "compute": compute_time / span, "exposed_comm": exposed_comm / span}


def scaling_efficiency(throughput_1: float, throughput_n: float, n: int) -> float:
    return throughput_n / (n * throughput_1)  # 1.0 = perfect linear scaling


def amdahl_speedup(parallel_fraction: float, n: int) -> float:
    # The serial part (1 − p) never speeds up, so it caps the gain: max speed-up = 1 / (1 − p).
    return 1 / ((1 - parallel_fraction) + parallel_fraction / n)
