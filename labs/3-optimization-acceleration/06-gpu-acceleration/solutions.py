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
