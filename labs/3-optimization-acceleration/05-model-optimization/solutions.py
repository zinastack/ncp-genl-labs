"""Lab 05 — reference solutions."""

import heapq

import numpy as np
import torch
import torch.nn.functional as F


def training_memory_bytes(n_params, trainable_params=None, weight_bytes=2):
    trainable = n_params if trainable_params is None else trainable_params
    return n_params * weight_bytes + trainable * 14


def max_batch_size(gpu_bytes, weight_bytes, kv_bytes_per_seq, reserve_fraction=0.1):
    free = gpu_bytes * (1 - reserve_fraction) - weight_bytes
    return max(0, int(free // kv_bytes_per_seq))


def quantize_int8(w, per_channel=False):
    if per_channel:
        absmax = np.abs(w).max(axis=1, keepdims=True)
    else:
        absmax = np.abs(w).max()
    scale = np.maximum(absmax, 1e-12) / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, np.asarray(scale, dtype=np.float64)


def dequantize(q, scale):
    return q.astype(np.float64) * scale


class DynamicLossScaler:
    def __init__(self, init_scale=2.0**16, growth_interval=2000):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.good_steps = 0

    def update(self, found_inf):
        if found_inf:
            self.scale /= 2
            self.good_steps = 0
            return False
        self.good_steps += 1
        if self.good_steps == self.growth_interval:
            self.scale *= 2
            self.good_steps = 0
        return True


def accumulated_gradients(model, x, y, micro_batch):
    model.zero_grad()
    chunks = list(zip(x.split(micro_batch), y.split(micro_batch)))
    for xb, yb in chunks:
        loss = F.mse_loss(model(xb), yb) / len(chunks)
        loss.backward()
    return [p.grad.clone() for p in model.parameters()]


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    t = temperature
    kd = F.kl_div(F.log_softmax(student_logits / t, -1), F.softmax(teacher_logits / t, -1), reduction="batchmean")
    ce = F.cross_entropy(student_logits, labels)
    return alpha * t * t * kd + (1 - alpha) * ce


def prune_2_4(w):
    groups = w.reshape(-1, 4)
    keep = np.argsort(-np.abs(groups), axis=1)[:, :2]
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    return np.where(mask, groups, 0).reshape(w.shape)


def static_batching_steps(output_lengths, batch_size):
    return sum(max(output_lengths[i : i + batch_size]) for i in range(0, len(output_lengths), batch_size))


def inflight_batching_steps(output_lengths, batch_size):
    # Each slot is busy until a finish time; a new request starts when the earliest slot frees.
    slots = [0] * min(batch_size, len(output_lengths))
    heapq.heapify(slots)
    finish = 0
    for length in output_lengths:
        start = heapq.heappop(slots)
        end = start + length
        finish = max(finish, end)
        heapq.heappush(slots, end)
    return finish


def speculative_expected_tokens(acceptance_rate, draft_len):
    a = acceptance_rate
    if a == 1:
        return draft_len + 1
    return (1 - a ** (draft_len + 1)) / (1 - a)


def decode_tokens_per_sec_bound(n_params, bytes_per_param, mem_bandwidth, batch=1):
    return mem_bandwidth / (n_params * bytes_per_param) * batch
