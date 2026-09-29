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

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Return a NEW plain nn.Linear whose weight = W + scaling · B @ A (bias copied)."""
        raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def apply_lora(model: nn.Module, target_modules: list[str], r: int = 8, alpha: float = 16) -> nn.Module:
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
def build_sft_example(prompt_ids: list[int], response_ids: list[int], eos_id: int) -> tuple[list[int], list[int]]:
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
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1) -> torch.Tensor:
    """Direct Preference Optimisation loss (mean over the batch). Inputs are (B,) sequence log-probs.

    L = -log σ( β · [(π_c − ref_c) − (π_r − ref_r)] )     (use F.logsigmoid)
    """
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 200, lr: float = 1e-2) -> list[float]:
    """Full-batch classification training. AdamW over ONLY params with requires_grad=True,
    cross-entropy loss on model(x) vs y. Return the list of per-step loss values (floats).
    """
    raise NotImplementedError
