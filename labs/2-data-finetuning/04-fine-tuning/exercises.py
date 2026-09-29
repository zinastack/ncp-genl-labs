"""Lab 04 — Fine-Tuning. Fill in every TODO, then run:

    pytest labs/2-data-finetuning/04-fine-tuning
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


# 1 ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update.

    y = base(x) + (dropout(x) @ Aᵀ @ Bᵀ) * (alpha / r)

    - Freeze base.weight and base.bias (requires_grad = False).
    - self.lora_A: nn.Parameter of shape (r, in_features), kaiming_uniform_(a=math.sqrt(5)).
    - self.lora_B: nn.Parameter of shape (out_features, r), zeros, so the initial output == base.
    - self.scaling = alpha / r
    """

    def __init__(
        self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Return a NEW plain nn.Linear whose weight = W + scaling · B @ A (bias copied)."""
        raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def apply_lora(
    model: nn.Module, target_modules: list[str], r: int = 8, alpha: float = 16,
) -> nn.Module:
    """Replace every nn.Linear whose *attribute name* is in target_modules with LoRALinear.
    Then freeze every parameter except the LoRA A/B matrices. Modify in place and return model.

    Hint: iterate over list(model.named_modules()); for each parent, look at its direct
    children with named_children() and use setattr(parent, child_name, LoRALinear(child, ...)).
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def build_sft_example(
    prompt_ids: list[int], response_ids: list[int], eos_id: int,
) -> tuple[list[int], list[int]]:
    """input_ids = prompt + response + [eos]; labels = same but IGNORE_INDEX on every prompt
    position (the model learns to produce the response and EOS, not the prompt).
    """
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy. logits (B, T, V), labels (B, T).
    Position t predicts label t+1: shift logits left and labels right. Ignore IGNORE_INDEX.
    """
    raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(label_{t+1} | ≤t) over non-ignored positions, per sequence → shape (B,)."""
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def dpo_loss(
    policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor, beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimisation loss (mean over the batch). Inputs are (B,) sequence log-probs.

    L = -log σ( β · [(π_c − ref_c) − (π_r − ref_r)] )     (use F.logsigmoid)
    """
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 200, lr: float = 1e-2,
) -> list[float]:
    """Full-batch classification training. AdamW over ONLY params with requires_grad=True,
    cross-entropy loss on model(x) vs y. Return the list of per-step loss values (floats).
    """
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
# The 16 NormalFloat-4 levels from the QLoRA paper (as used by bitsandbytes).
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(w: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise NF4 (QLoRA's 4-bit format). Reshape w to (-1, block_size). Per block:
    absmax = max |value| (clamp to ≥ 1e-12), normalise to [-1, 1], and replace each value by the
    INDEX (0–15) of the nearest NF4 level. Return (codes as uint8, absmax of shape (n_blocks, 1)).
    """
    raise NotImplementedError


def nf4_dequantize(codes: torch.Tensor, absmax: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Inverse: NF4_LEVELS[codes] × each block's absmax, reshaped to `shape`."""
    raise NotImplementedError


# 10 ────────────────────────────────────────────────────────────────────────────
class SoftPrompt(nn.Module):
    """Prompt tuning / p-tuning: n_virtual learned "virtual token" embeddings prepended to every
    input. self.prompt: nn.Parameter (n_virtual, d_model), init randn × 0.02.
    forward(input_embeds (B, T, d)) → (B, n_virtual + T, d).
    """

    def __init__(self, n_virtual: int, d_model: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 11 ────────────────────────────────────────────────────────────────────────────
def reward_model_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley–Terry loss for training an RLHF reward model on preference pairs:
    mean of −log σ(r_chosen − r_rejected). Use F.logsigmoid.
    """
    raise NotImplementedError


def preference_accuracy(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> float:
    """Fraction of pairs where the reward model scores the chosen answer strictly higher."""
    raise NotImplementedError


# 12 ────────────────────────────────────────────────────────────────────────────
class MultiLoRALinear(nn.Module):
    """Multi-LoRA serving: one frozen base layer, many adapters, one mixed batch.

    adapters: name → (A (r, in), B (out, r), scaling).
    forward(x (batch, in), adapter_names: one name per row, or None for the plain base model):
    out = base(x) for the whole batch, then for each row with an adapter add
    (x[i] @ Aᵀ @ Bᵀ) × scaling.
    """

    def __init__(self, base: nn.Linear, adapters: dict[str, tuple[torch.Tensor, torch.Tensor, float]]) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adapter_names: list[str | None]) -> torch.Tensor:
        raise NotImplementedError


# 13 ────────────────────────────────────────────────────────────────────────────
def lr_at_step(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """Warm-up + cosine schedule (steps count from 0).
    step < warmup_steps: max_lr × (step + 1) / warmup_steps
    afterwards: progress = min(1, (step − warmup) / max(1, total_steps − warmup));
                lr = min_lr + 0.5 × (max_lr − min_lr) × (1 + cos(π × progress))
    """
    raise NotImplementedError
