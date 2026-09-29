"""Lab 04 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100  # PyTorch cross_entropy's default ignore_index


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # the pre-trained weight never changes
        self.r, self.scaling = r, alpha / r  # α/r keeps the update size stable when r changes
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        # B = 0 → B·A = 0 → at step 0 the layer is exactly the base layer. A is random so the
        # gradient w.r.t. B is non-zero and learning starts immediately.
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init nn.Linear uses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ Aᵀ @ Bᵀ = (B·A)·x without ever building the full d_out × d_in matrix.
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        # Fold the update into one dense layer: W' = W + (α/r)·B·A → no extra inference cost.
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scaling * self.lora_B @ self.lora_A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(
    model: nn.Module, target_modules: list[str], r: int = 8, alpha: float = 16,
) -> nn.Module:
    # list(...) snapshots the tree before we modify it.
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in target_modules and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))  # swap the layer in place
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name  # train ONLY the adapters (heads, norms stay frozen)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    params = list(model.parameters())
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)


def build_sft_example(
    prompt_ids: list[int], response_ids: list[int], eos_id: int,
) -> tuple[list[int], list[int]]:
    input_ids = list(prompt_ids) + list(response_ids) + [eos_id]
    # Loss only on the response (+ EOS, so the model learns to stop). Prompt positions are
    # ignored, otherwise the model also learns to write user prompts.
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids) + [eos_id]
    return input_ids, labels


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Position t predicts token t+1: drop the last prediction and the first label.
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)  # mean over kept tokens


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = logits[:, :-1].log_softmax(-1)  # same shift as the loss
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    # gather picks log p(actual next token). clamp turns -100 into a valid index; the mask
    # then zeroes those positions.
    token_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)  # sum of logs = log of the whole sequence's probability


def dpo_loss(
    policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor, beta: float = 0.1,
) -> torch.Tensor:
    # How much more the policy (vs the frozen reference) prefers chosen over rejected.
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    # -log σ(β·margin): ln 2 at the start (margin 0), → 0 as the chosen answer wins.
    # logsigmoid is the numerically stable form of log(sigmoid(z)).
    return -F.logsigmoid(beta * margin).mean()


def train(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 200, lr: float = 1e-2,
) -> list[float]:
    # Only trainable params go to the optimizer: Adam's extra state (m, v) exists only for them.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# The 16 NormalFloat-4 levels from the QLoRA paper (as used by bitsandbytes): quantiles of a
# normal distribution rescaled to [-1, 1], with an exact 0.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(w: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = w.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # one scale per block of 64
    normed = blocks / absmax  # now in [-1, 1]
    # Each weight becomes the index (0–15, i.e. 4 bits) of the nearest NF4 level.
    codes = (normed.unsqueeze(-1) - NF4_LEVELS).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax


def nf4_dequantize(codes: torch.Tensor, absmax: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return (NF4_LEVELS[codes.long()] * absmax).reshape(shape)  # level × block scale


class SoftPrompt(nn.Module):
    def __init__(self, n_virtual: int, d_model: int) -> None:
        super().__init__()
        # The ONLY trainable weights: n_virtual "virtual token" embeddings. The LLM stays frozen.
        self.prompt = nn.Parameter(torch.randn(n_virtual, d_model) * 0.02)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # Prepend the same learned vectors to every sequence in the batch: (B, n + T, d).
        batch = input_embeds.shape[0]
        return torch.cat([self.prompt.unsqueeze(0).expand(batch, -1, -1), input_embeds], dim=1)


def reward_model_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    # Bradley–Terry: P(chosen beats rejected) = σ(r_c − r_r). Maximise its log-likelihood.
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def preference_accuracy(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> float:
    return (reward_chosen > reward_rejected).float().mean().item()  # how often the RM agrees with humans


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, adapters: dict[str, tuple[torch.Tensor, torch.Tensor, float]]) -> None:
        super().__init__()
        self.base = base  # ONE copy of the big weight, shared by every customer
        self.adapters = adapters  # name → (A, B, scaling): megabytes each

    def forward(self, x: torch.Tensor, adapter_names: list[str | None]) -> torch.Tensor:
        out = self.base(x)  # the expensive part runs once for the whole mixed batch
        for i, name in enumerate(adapter_names):  # the cheap low-rank part is per request
            if name is not None:
                a, b, scaling = self.adapters[name]
                out[i] = out[i] + (x[i] @ a.T @ b.T) * scaling
        return out


def lr_at_step(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        # Linear warm-up: Adam's early moment estimates are noisy, so start with small steps.
        return max_lr * (step + 1) / warmup_steps
    # Cosine decay from max_lr to min_lr over the remaining steps: large steps early, fine steps at the end.
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
