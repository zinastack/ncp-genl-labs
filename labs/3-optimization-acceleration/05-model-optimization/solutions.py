"""Lab 05 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import heapq

import numpy as np
import torch
import torch.nn.functional as F


def training_memory_bytes(
    n_params: float, trainable_params: float | None = None, weight_bytes: float = 2,
) -> float:
    trainable = n_params if trainable_params is None else trainable_params
    # Every weight is stored once (2 B bf16, 0.5 B for 4-bit QLoRA). Each TRAINABLE one also
    # needs a 2 B grad + 4 B fp32 master copy + 4 B Adam m + 4 B Adam v = 14 B.
    return n_params * weight_bytes + trainable * 14


def max_batch_size(
    gpu_bytes: float, weight_bytes: float, kv_bytes_per_seq: float, reserve_fraction: float = 0.1,
) -> int:
    free = gpu_bytes * (1 - reserve_fraction) - weight_bytes  # what's left for KV caches
    return max(0, int(free // kv_bytes_per_seq))  # 0 if the weights alone don't fit


def quantize_int8(w: np.ndarray, per_channel: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if per_channel:
        absmax = np.abs(w).max(axis=1, keepdims=True)  # one range per output row → (rows, 1)
    else:
        absmax = np.abs(w).max()  # one range for everything: outliers set the step size
    scale = np.maximum(absmax, 1e-12) / 127.0  # 1e-12 guards all-zero rows against ÷0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, np.asarray(scale, dtype=np.float64)


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) * scale  # w ≈ q · scale (the rounding error is what's lost)


class DynamicLossScaler:
    def __init__(self, init_scale: float = 2.0**16, growth_interval: int = 2000) -> None:
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.good_steps = 0

    def update(self, found_inf: bool) -> bool:
        if found_inf:
            # Gradients overflowed: halve the scale and tell the caller to SKIP this optimizer
            # step (an inf gradient would destroy the weights).
            self.scale /= 2
            self.good_steps = 0
            return False
        self.good_steps += 1
        if self.good_steps == self.growth_interval:  # long clean run → probe a bigger scale
            self.scale *= 2
            self.good_steps = 0
        return True


def accumulated_gradients(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, micro_batch: int,
) -> list[torch.Tensor]:
    model.zero_grad()  # once: backward() ADDS into .grad across micro-batches
    chunks = list(zip(x.split(micro_batch), y.split(micro_batch)))
    for xb, yb in chunks:
        # ÷ number of chunks so the summed gradient equals the full-batch mean-loss gradient.
        loss = F.mse_loss(model(xb), yb) / len(chunks)
        loss.backward()
    return [p.grad.clone() for p in model.parameters()]


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
    temperature: float = 2.0, alpha: float = 0.5,
) -> torch.Tensor:
    t = temperature
    # kl_div expects the student as LOG-probabilities and the teacher as probabilities.
    # T > 1 softens both distributions so the student sees the teacher's "dark knowledge".
    kd = F.kl_div(F.log_softmax(student_logits / t, -1), F.softmax(teacher_logits / t, -1), reduction="batchmean")
    ce = F.cross_entropy(student_logits, labels)  # the ordinary hard-label loss
    return alpha * t * t * kd + (1 - alpha) * ce  # T² restores the gradient scale softening removed


def prune_2_4(w: np.ndarray) -> np.ndarray:
    groups = w.reshape(-1, 4)  # consecutive groups of 4 along the last axis
    keep = np.argsort(-np.abs(groups), axis=1)[:, :2]  # the 2 largest magnitudes per group
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    return np.where(mask, groups, 0).reshape(w.shape)  # exactly 50% zeros, hardware-friendly pattern


def static_batching_steps(output_lengths: list[int], batch_size: int) -> int:
    # Each fixed batch holds the GPU until its LONGEST request is done.
    return sum(max(output_lengths[i : i + batch_size]) for i in range(0, len(output_lengths), batch_size))


def inflight_batching_steps(output_lengths: list[int], batch_size: int) -> int:
    # Min-heap of "time this slot becomes free". Each request (in arrival order) takes the
    # earliest free slot, and a finished request frees its slot immediately.
    slots = [0] * min(batch_size, len(output_lengths))
    heapq.heapify(slots)
    finish = 0
    for length in output_lengths:
        start = heapq.heappop(slots)
        end = start + length
        finish = max(finish, end)
        heapq.heappush(slots, end)
    return finish


def speculative_expected_tokens(acceptance_rate: float, draft_len: int) -> float:
    a = acceptance_rate
    if a == 1:  # every guess accepted: γ drafts + 1 token from the target (formula would ÷0)
        return draft_len + 1
    return (1 - a ** (draft_len + 1)) / (1 - a)  # 1 + a + a² + … + a^γ (geometric series)


def decode_tokens_per_sec_bound(
    n_params: float, bytes_per_param: float, mem_bandwidth: float, batch: int = 1,
) -> float:
    # Each decode step reads every weight once and yields one token per sequence in the batch.
    return mem_bandwidth / (n_params * bytes_per_param) * batch
