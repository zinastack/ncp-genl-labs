"""Lab 06 — reference solutions."""

import numpy as np
import torch
import torch.distributed as dist


def zero_bytes_per_gpu(n_params, n_gpus, stage):
    p, g, o = 2 * n_params, 2 * n_params, 12 * n_params
    if stage == 0:
        return p + g + o
    if stage == 1:
        return p + g + o / n_gpus
    if stage == 2:
        return p + (g + o) / n_gpus
    if stage == 3:
        return (p + g + o) / n_gpus
    raise ValueError(stage)


def pipeline_bubble_fraction(stages, micro_batches, virtual_stages=1):
    bubble = (stages - 1) / virtual_stages
    return bubble / (micro_batches + bubble)


def ring_allreduce_bytes(size_bytes, n_gpus):
    return 2 * (n_gpus - 1) / n_gpus * size_bytes


def allreduce_seconds(size_bytes, n_gpus, bus_bandwidth):
    return ring_allreduce_bytes(size_bytes, n_gpus) / bus_bandwidth


def rank_coords(rank, world_size, tp, pp):
    if world_size % (tp * pp):
        raise ValueError("world_size must be divisible by tp * pp")
    dp = world_size // (tp * pp)
    return rank % tp, (rank // tp) % dp, rank // (tp * dp)


def tensor_parallel_group(rank, tp):
    start = rank - rank % tp
    return list(range(start, start + tp))


def column_parallel(x, w, n):
    return [x @ shard for shard in np.split(w, n, axis=1)]


def row_parallel(x_shards, w, n):
    return sum(xs @ ws for xs, ws in zip(x_shards, np.split(w, n, axis=0)))


def megatron_mlp(x, w1, w2, n):
    hidden_shards = [np.maximum(h, 0) for h in column_parallel(x, w1, n)]
    return row_parallel(hidden_shards, w2, n)


def average_gradients(model):
    world = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


def training_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens


def mfu(tokens_per_sec, n_params, n_gpus, peak_flops_per_gpu):
    return 6 * n_params * tokens_per_sec / (n_gpus * peak_flops_per_gpu)


def gemm_arithmetic_intensity(m, n, k, bytes_per_elem=2):
    return 2 * m * n * k / (bytes_per_elem * (m * k + k * n + m * n))


def is_memory_bound(intensity, peak_flops, mem_bandwidth):
    return intensity < peak_flops / mem_bandwidth
