"""Lab 04 — reference solutions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scaling * self.lora_B @ self.lora_A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(model, target_modules, r=8, alpha=16):
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in target_modules and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
    return model


def count_parameters(model):
    params = list(model.parameters())
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)


def build_sft_example(prompt_ids, response_ids, eos_id):
    input_ids = list(prompt_ids) + list(response_ids) + [eos_id]
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids) + [eos_id]
    return input_ids, labels


def causal_lm_loss(logits, labels):
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)


def sequence_logprob(logits, labels):
    logp = logits[:, :-1].log_softmax(-1)
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    token_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def train(model, x, y, steps=200, lr=1e-2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
