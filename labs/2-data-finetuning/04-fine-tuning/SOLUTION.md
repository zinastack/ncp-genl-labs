# Lab 04 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: a pre-trained LLM already knows language. **Fine-tuning** nudges its weights toward
a task, a style or a domain. The problem is size: updating all 7 billion weights needs gradients
and optimizer state for every one of them. **LoRA** trains a tiny add-on instead, and **DPO** teaches
preferences directly.

```
                 frozen W (billions of weights)
x ──► ──────────────────────────────► (+) ──► y          y = W·x + (α/r)·B·A·x
  └─► A (r × d_in) ─► B (d_out × r) ──┘                 only A and B are trained
```

---

## Exercise 1 — `LoRALinear`: the low-rank update

### The idea

Research found that the weight *change* needed for adaptation, `ΔW`, has **low rank**: it can be
written as the product of two thin matrices. Instead of learning a full `d_out × d_in` matrix,
learn `B (d_out × r)` and `A (r × d_in)` with a small rank `r` (8–64):

| matrix | full ΔW | LoRA (r) | trainable share |
|---|---|---|---|
| 4096 × 4096, r = 8 | 16,777,216 | 65,536 | 0.39% |
| 4096 × 4096, r = 16 | 16,777,216 | 131,072 | 0.78% |
| 4096 × 11008 (MLP), r = 16 | 45,088,768 | 241,664 | 0.54% |

LoRA params per matrix = `r · (d_in + d_out)`.

### Line by line

```python
for p in self.base.parameters():
    p.requires_grad = False                          # 1. freeze the original weight
self.lora_A = nn.Parameter(torch.empty(r, in))       # 2. random init (kaiming)
self.lora_B = nn.Parameter(torch.zeros(out, r))      # 3. ZERO init
self.scaling = alpha / r                             # 4. scale
forward: base(x) + (dropout(x) @ Aᵀ @ Bᵀ) * scaling
```

- **Why is B zero?** Then `B·A = 0`, so at step 0 the model is **exactly** the pre-trained model
  (the test checks `layer(x) == base(x)`). Training starts from a known-good point. A is random so
  that the gradient with respect to B is not zero: if both were zero, nothing would ever learn.
- **Why α / r?** It keeps the size of the update roughly constant when you change `r`, so you can
  try r = 8 vs 32 without re-tuning the learning rate. A common choice is α = 2r (scale 2) or α = r.
- `x @ A.T @ B.T` computes `B·A·x` without ever building the full `d_out × d_in` matrix, which is cheap.

### `merge()`

After training, fold the update in: `W' = W + scaling · B @ A`. The result is a plain `nn.Linear`
with **no extra latency**, and the test confirms merged output equals LoRA output.
The trade-off: a merged copy serves one adapter. Unmerged, one base model can serve **many adapters**
(multi-LoRA in NIM and TensorRT-LLM).

---

## Exercise 2 — `apply_lora`: inject adapters by name

Real models are nested modules. In Llama, attention projections are named `q_proj`, `k_proj`,
`v_proj`, `o_proj` and the MLP has `gate_proj`, `up_proj`, `down_proj`. PEFT's `target_modules`
picks layers **by attribute name**:

```python
for _, parent in list(model.named_modules()):          # every module in the tree…
    for name, child in list(parent.named_children()):  # …and its direct children
        if name in target_modules and isinstance(child, nn.Linear):
            setattr(parent, name, LoRALinear(child, r, alpha))   # swap it in place
for name, p in model.named_parameters():
    p.requires_grad = "lora_" in name                   # freeze everything else
```

- `list(...)` snapshots the modules first, because modifying a tree while iterating it is unsafe.
- `setattr(parent, name, …)` replaces the child, the same way `model.q_proj = …` would.
- The final loop freezes **everything** except LoRA matrices, including heads and norms. The test checks that the classification head stays frozen.

---

## Exercise 3 — `count_parameters`: what you actually train

```python
trainable = sum(p.numel() for p in params if p.requires_grad)
```

In the test model, LoRA on `q_proj`, `v_proj` (32→32) and `down_proj` (64→32) with r = 4 gives
`4·64·2 + 4·96 = 896` trainable parameters out of 8,388 in total. On a real 7B model, targeting q and v in all
32 layers with r = 16 is ~8.4M, **0.12%**. That's why LoRA checkpoints are megabytes and the
optimizer state is tiny (Lab 05 turns this into GB).

---

## Exercise 4 — `build_sft_example`: learn the answer, not the question

Supervised fine-tuning data is (prompt, response) pairs. We feed both, but compute the loss **only
on the response**:

```
input_ids = [5, 6, 7,    8, 9, 2]         prompt + response + EOS
labels    = [-100,-100,-100, 8, 9, 2]     -100 = "ignore this position"
```

- `-100` is PyTorch's default `ignore_index` for cross-entropy: those positions add nothing to the loss.
- **Without masking,** the model is also trained to *write user prompts*, and after SFT it may
  start generating the next user question. That's a classic exam scenario.
- **EOS is kept in the labels** so the model learns *when to stop*. Mask it and the model rambles.

---

## Exercise 5 — `causal_lm_loss`: the shift

A causal LM at position *t* predicts token *t+1*. So logits and labels are **offset by one**:

```python
shift_logits = logits[:, :-1]      # predictions made at positions 0..T-2
shift_labels = labels[:, 1:]       # the tokens that actually came next: 1..T-1
F.cross_entropy(flat logits, flat labels, ignore_index=-100)   # mean over non-ignored tokens
```

Worked example (vocabulary of 3). Logits favour the correct next token by 2 points each time:
`p(correct) = e² / (e² + 2) = 0.787`, so each position costs `−ln 0.787 = 0.2395`.

| labels | loss (mean) | what happened |
|---|---|---|
| `[-100, 1, 2, 0]` | 0.2395 | 3 positions, all predicted well |
| `[-100, -100, 2, 0]` | 0.2395 | first position masked: still the mean of the rest |
| `[-100, 2, 2, 0]` | 0.9062 | one label disagrees with the model: that position costs 2.24 |

Hugging Face models do this shift internally when you pass `labels=`. Writing it yourself shows
why the last position's logits are never used for the loss.

---

## Exercise 6 — `sequence_logprob`: how likely is a whole answer?

DPO (next) needs the model's log-probability of an **entire response**: the sum over its tokens.

```python
logp = logits[:, :-1].log_softmax(-1)                          # shifted, as before
token_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1))  # pick the prob of the real token
(token_logp * mask).sum(-1)                                     # sum only response tokens
```

- `gather` picks, at each position, the log-probability of the token that actually came next.
- `clamp(min=0)` turns `-100` into a valid index so `gather` doesn't crash; the mask then zeroes those positions.
- With the example above: 3 tokens × ln 0.787 = **−0.7186**; masking the first gives −0.4791.
  A sum of logs is the log of the product: the probability of the whole sequence.

---

## Exercise 7 — `dpo_loss`: learning preferences without RL

We have pairs: a **chosen** answer `y_w` and a **rejected** answer `y_l`. Classic RLHF trains a
reward model and then runs PPO. **DPO** skips both with one loss:

```
L = −log σ( β · [ (log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l)) ] )
                  └── how much MORE the policy likes y_w ──┘   └── … and y_l, vs the frozen reference ──┘
```

- `σ` is the sigmoid. `−log σ(z)` is small when `z` is large, meaning the policy has moved toward
  the chosen answer more than toward the rejected one, **relative to where it started** (the reference).
- At the start policy = reference, so the margin is 0 and the loss is `ln 2 = 0.693` (the test checks this).
- **β** controls how hard to push:

| margin | loss with β = 0.1 | loss with β = 0.5 |
|---|---|---|
| 0 | 0.693 | 0.693 |
| 2 | 0.598 | 0.313 |
| 4 | 0.513 | 0.127 |
| 8 | 0.371 | 0.018 |

  A small β needs a bigger margin to reduce the loss, which keeps the policy close to the reference (like PPO's KL penalty).
- `F.logsigmoid` is used instead of `log(sigmoid(z))` for numerical stability: sigmoid can round to 0, and `log(0) = -inf`.

---

## Exercise 8 — `train`: only the trainable parameters

```python
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
for _ in range(steps):
    opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
```

Passing **only** trainable parameters to the optimizer is where LoRA's memory saving comes from:
Adam keeps two extra numbers (m, v) per parameter it manages. The test pre-trains a tiny model on
task A, adds LoRA, learns task B (loss drops by more than half), and then checks every original
weight is **bit-for-bit unchanged**: the base model is intact, and only the adapter learned.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| why B = 0 at init | start exactly at the base model; A random so B gets gradients |
| trainable parameter count | r·(d_in + d_out) per adapted matrix |
| huge model on one GPU | QLoRA: 4-bit NF4 frozen base + LoRA |
| model generates user turns after SFT | prompt tokens not masked (-100) |
| many customer variants of one model | multi-LoRA serving, one adapter each |
| no latency overhead | merge adapters (W + (α/r)·B·A) |
| forgetting general skills | PEFT, replay general data, fewer epochs, lower lr |
| preference data, no reward model | DPO (β controls drift from the reference) |
| domain vocabulary and knowledge | continued pre-training (DAPT), then SFT |
