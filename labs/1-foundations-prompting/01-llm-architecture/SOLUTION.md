# Lab 01 — Solution walkthrough

A guided explanation of every exercise: **what** the function does, **why** each line exists,
a **small example with real numbers** (all computed by running `solutions.py`), what **breaks**
without it, and how it shows up **on the exam**. Read one exercise at a time after trying it yourself.

The big picture you are building, one token at a time:

```
token ids → embeddings (+ position info) → [ attention → FFN ] × N layers → scores for the next token
                                               ▲
                     exercises 1–4 build this; 5–6 add position; 7 normalises; 8–10 are the numbers
```

---

## Exercise 1 — `softmax`: turning scores into weights

### What it is for

Many places in an LLM produce **scores**: arbitrary real numbers saying "how much does this
matter?". Softmax turns a list of scores into **weights** that are all positive and sum to 1
(percentages), and it keeps the order: a higher score always gets more weight.

In attention the scores are `QKᵀ/√d_k` (exercise 3). At the output of the model they are the
**logits**, one per vocabulary word, and softmax gives next-token probabilities (Lab 02).

### The math, then the numbers

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Scores `[2.0, 1.0, 0.0]`:

| step | values |
|---|---|
| `exp` makes everything positive and widens gaps (+1 in score is ×2.7 in weight) | `[7.39, 2.72, 1.00]` |
| divide by the sum (11.11) so they add up to 1 | `[0.665, 0.245, 0.090]` |

### Why not simply pick the biggest score?

Because attention does not *select* a token. It **blends all value vectors** with these weights
(`weights @ V`, exercise 3). With `v1=[1,0], v2=[0,1], v3=[1,1]`:
`0.665·v1 + 0.245·v2 + 0.090·v3 = [0.755, 0.335]`: mostly token 1, some of token 2, a little of 3.
Picking only the max (`argmax`) has three problems:

1. **It loses information.** In "The animal didn't cross the street because *it* was too tired",
   *it* needs information from *animal* and from the context around it, not from just one token.
2. **It can't be trained.** Training nudges weights and watches the loss change (gradient descent).
   Nudging a score from 2.00 to 2.01 does not change which token wins, so `argmax` has a
   **zero gradient** and the model gets no learning signal. With softmax the weight moves from
   0.665 to about 0.667, a small, smooth change that gradients can follow. That is why it is called
   a *soft* max.
3. **It hides uncertainty.** Scores `[2.00, 1.99]` → argmax says 100% / 0%; softmax says ~50% / 50%.

> **Exam link: temperature.** `softmax(x / T)`: as `T → 0` the weights approach `[1, 0, 0]`,
> pure argmax, which is **greedy decoding** when choosing the next word. High `T` flattens the
> distribution, giving more random sampling. Inside attention you always want the smooth blend (T = 1).

### Line by line

```python
x_max = np.max(x, axis=axis, keepdims=True)
x_max = np.where(np.isfinite(x_max), x_max, 0.0)
e = np.exp(x - x_max)
total = np.sum(e, axis=axis, keepdims=True)
return e / np.where(total == 0, 1.0, total)
```

**Subtracting the max (numerical stability).** Computers have limits: in float64 `exp(709) ≈ 8·10³⁰⁷`
but `exp(710) = inf`, and `inf / inf = nan`. The fix uses a property of softmax: subtracting the
same number from every score changes nothing, because
`exp(x − c) = exp(x) / exp(c)` and the `exp(c)` appears in both numerator and denominator and cancels.
Check: `softmax([1000, 1001, 1002]) = softmax([0, 1, 2]) = [0.090, 0.245, 0.665]`.
Choosing `c = max`:
- the largest term becomes `exp(0) = 1`, all others are in `(0, 1]`, so nothing can overflow;
- at least one term is 1, so the sum is ≥ 1 and we never divide by zero on a normal row.

**`axis` and `keepdims`.** Attention scores have shape `(batch, queries, keys)`. Each **query**
needs its own weights over the **keys**, so we normalise along the last axis (each row sums to 1).
`keepdims=True` keeps the max as shape `(…, 1)` instead of `(…)`, so NumPy can broadcast it and
subtract each row's own max from that row.

**`-inf` and masking.** The causal mask (exercise 2) writes `-inf` into blocked positions.
`exp(-inf) = 0`, so blocked keys get exactly 0% weight: `softmax([0, -inf]) = [1, 0]`.
Finite scores never reach exactly 0: `exp(-5) = 0.0067`. That is why masking uses `-inf`
rather than a large negative number.

**The two `np.where` guards (edge case: a fully-masked row).** If *every* entry in a row is
`-inf` (a query allowed to see nothing), the max is `-inf` and `-inf − (-inf) = nan`. The first
guard uses 0 instead of a non-finite max. Then every `exp` is 0, the sum is 0, and `0/0 = nan`.
The second guard divides by 1 instead, so the row comes out as all zeros ("attend to nothing"),
not `nan`. This never happens with a causal mask, because a token can always see itself, but it
does happen with padding masks on empty sequences, and a single `nan` poisons the whole model.

---

## Exercise 2 — `causal_mask`: no peeking at the future

A decoder LLM is trained to predict token *t+1* from tokens `0…t`. If token 2 could attend to
token 3, it could simply *copy the answer*. The mask marks what each query may see:

```
         key 0  1  2  3
query 0 [ 1  0  0  0 ]      np.tril(np.ones((n, n), dtype=bool))
query 1 [ 1  1  0  0 ]      lower triangle incl. the diagonal
query 2 [ 1  1  1  0 ]      True = allowed, False = blocked (becomes -inf)
query 3 [ 1  1  1  1 ]
```

- The diagonal is `True`: a token always sees itself.
- **Encoders (BERT)** use no causal mask. They are bidirectional and only mask padding. That is
  why BERT is good at understanding and embeddings and cannot generate text left-to-right.

---

## Exercise 3 — `scaled_dot_product_attention`: the heart of the transformer

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  + mask ) · V
```

**What Q, K and V mean.** Every token is projected into three vectors:
- **Query** (Q): "what am I looking for?"
- **Key** (K): "what do I contain?"
- **Value** (V): "what information do I pass on if someone attends to me?"

The dot product `q · k` measures how well a query matches a key: large means relevant.

### Worked example (3 tokens, `d_k = 2`)

```
Q = K = [[1,0], [0,1], [1,1]]        V = [[10,0], [0,10], [5,5]]
```

| step | code | result (row = query) |
|---|---|---|
| scores | `q @ kᵀ` | `[[1,0,1], [0,1,1], [1,1,2]]` |
| scale | `/ √2` | `[[.707,0,.707], [0,.707,.707], [.707,.707,1.414]]` |
| mask | `np.where(mask, scores, -inf)` | `[[.707,-inf,-inf], [0,.707,-inf], [.707,.707,1.414]]` |
| softmax per row | `softmax(…)` | `[[1,0,0], [.33,.67,0], [.25,.25,.50]]` |
| blend values | `weights @ v` | `[[10,0], [3.3,6.7], [5,5]]` |

Read the last row: token 3 matches itself best (score 2 before scaling), so half its output comes
from its own value `[5,5]` and a quarter from each earlier token. Token 1 can only see itself, so
its output is exactly its own value.

### Why divide by √d_k?

A dot product sums `d_k` products. For random vectors its spread (standard deviation) grows like
`√d_k`. Measured: `d=4 → 2.0`, `d=64 → 8.0`, `d=512 → 22.4`; after dividing by `√d_k` all are ≈1.0.
Large scores push softmax to near one-hot: 8 random scores at `d=512` gave a top weight of 0.985
unscaled versus 0.308 scaled. Near one-hot means **tiny gradients**, so learning stalls. Scaling
keeps softmax in its useful, smooth range.

> **Exam trap:** the √d_k scaling is about **softmax saturation / gradient stability**,
> not speed and not normalising rows (softmax does that).

### Code notes

- `np.swapaxes(k, -1, -2)` transposes only the last two axes, so it works for any leading batch/head dimensions.
- The mask is applied **before** softmax (as `-inf`). Masking after softmax would leave rows that no longer sum to 1.
- Return both the output and the weights. The weights are what attention visualisations plot.

---

## Exercise 4 — `multi_head_attention`: several attentions in parallel

One attention head can learn one kind of relation (for example "which noun does this pronoun refer
to"). **Multi-head** attention runs `h` smaller heads in parallel, so different heads learn different relations
(syntax, coreference, position…), and then merges them.

Shape walkthrough with `seq=5, d_model=8, h=2 → head_dim = 4`:

```
x               (5, 8)
x @ w_q         (5, 8)          project (same for K, V)
reshape         (5, 2, 4)       split the 8 features into 2 heads × 4
transpose       (2, 5, 4)       heads first → attention runs per head (batched)
attention       (2, 5, 4)
transpose back  (5, 2, 4)
reshape         (5, 8)          concatenate heads
@ w_o           (5, 8)          output projection mixes information across heads
```

- The cost is about the same as one big head: the width is split, not multiplied.
- **MQA / GQA** (exam favourite): several query heads share the same K/V heads, which shrinks the
  **KV cache** (exercise 9). Llama-2-70B has 64 query heads but only 8 KV heads.
- The test checks **causality**: changing the last token must not change earlier outputs.

---

## Exercise 5 — `sinusoidal_positions`: telling the model where each token is

Attention by itself has no notion of order: "dog bites man" and "man bites dog" contain the same
tokens. The original Transformer **adds** a position vector to each token embedding:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

With `d=4`: the frequencies are `1` (dims 0–1) and `0.01` (dims 2–3).

```
pos 0: [ 0.000, 1.000, 0.000, 1.000 ]
pos 1: [ 0.841, 0.540, 0.010, 1.000 ]     sin(1), cos(1), sin(0.01), cos(0.01)
pos 2: [ 0.909,-0.416, 0.020, 1.000 ]
```

Low dimensions oscillate fast and distinguish neighbouring positions. High dimensions oscillate
slowly and encode coarse position, like the hands of a clock. Code trick: `np.arange(0, d, 2)`
gives the even indices `2i`, and slicing `pe[:, 0::2]` / `pe[:, 1::2]` fills even and odd columns.

Modern LLMs use RoPE instead (next exercise). GPT-2 and BERT use **learned** absolute positions,
so they cannot go beyond their trained length (1024 / 512).

---

## Exercise 6 — `apply_rope`: rotary position embeddings (Llama, Mistral, Nemotron)

Instead of *adding* a position vector, RoPE **rotates** each pair of features of Q and K by an
angle proportional to the position:

```
pair (x0, x1) at position p, frequency θ:
out0 = x0·cos(pθ) − x1·sin(pθ)
out1 = x0·sin(pθ) + x1·cos(pθ)        ← a standard 2-D rotation
```

Example: `x = [1, 0, 0, 1]` (two pairs), `d=4`, frequencies `[1, 0.01]`.
- position 1: pair 0 `(1,0)` rotates by 1 rad → `(0.540, 0.841)`; pair 1 `(0,1)` by 0.01 rad → `(-0.010, 1.000)`
- position 2: rotates twice as far → `(-0.416, 0.909)` and `(-0.020, 1.000)`

**The key property: attention depends only on relative distance.** Rotating `q` by `mθ` and `k` by
`nθ` gives a dot product that depends only on the difference `m − n`. Measured with the same q, k:

| q position | k position | q·k |
|---|---|---|
| 3 | 1 | 8.7036 |
| 10 | 8 | 8.7036 |
| 100 | 98 | 8.7036 |
| 10 | 1 | 10.0526 (different gap, different score) |

That is what the test checks. It is also why context can be **extended** by rescaling θ
(NTK/YaRN "rope scaling"), and why rotation preserves vector length and position 0 is the identity.

Code: `inv_freq = base ** (-np.arange(0, d, 2) / d)` holds one frequency per pair; `theta = positions[:, None] * inv_freq[None, :]`
builds an angle table (positions × pairs); even and odd features are the two halves of each pair.

---

## Exercise 7 — `layer_norm` and `rms_norm`: keeping activations in range

Deep stacks multiply many matrices, so values can drift huge or tiny, and training becomes
unstable. Normalisation rescales each token's vector.

`x = [1, 2, 3, 6]` → mean 3, variance 3.5, RMS 3.536.

| | formula | result |
|---|---|---|
| **LayerNorm** | `(x − mean) / √(var + ε) · γ + β` | `[-1.069, -0.535, 0, 1.604]` (mean 0, variance 1) |
| **RMSNorm** | `x / √(mean(x²) + ε) · γ` | `[0.283, 0.566, 0.849, 1.697]` (only rescaled, not centred) |

- RMSNorm skips the mean and the bias: slightly cheaper, and it works as well. Llama uses it.
- `γ` (and `β`) are **learned**, so the model can undo the normalisation where useful.
- `ε` avoids division by zero for an all-constant vector.
- `axis=-1, keepdims=True`: normalise each token's feature vector separately (not across tokens).
- **Pre-norm** (`x + f(Norm(x))`) vs post-norm is an architecture choice about *where* the norm goes (see the README).

---

## Exercise 8 — `gpt2_param_count`: where the 124 M parameters live

Count every weight matrix and bias. For GPT-2 small (`d=768, L=12, V=50257, n_ctx=1024`):

| part | formula | count |
|---|---|---|
| token embeddings | `V·d` | 38,597,376 |
| position embeddings | `n_ctx·d` | 786,432 |
| per layer: 2 LayerNorms | `2·(2d)` | 3,072 |
| per layer: fused QKV | `d·3d + 3d` | 1,771,776 |
| per layer: attention output | `d·d + d` | 590,592 |
| per layer: MLP up (d→4d) | `d·4d + 4d` | 2,362,368 |
| per layer: MLP down (4d→d) | `4d·d + d` | 2,360,064 |
| × 12 layers | | 85,054,464 |
| final LayerNorm | `2d` | 1,536 |
| LM head | **tied** to token embeddings | 0 |
| **total** | | **124,439,808** |

- Per layer ≈ `12d²`: attention `4d²` plus MLP `8d²`, so the **MLP holds about ⅔** (exam question).
- **Weight tying:** the output layer reuses the embedding matrix, saving another 38.6 M.
- The GPU lab confirms the number against the real Hugging Face model.

---

## Exercise 9 — `kv_cache_bytes`: the memory that limits serving

When generating, each new token attends to all previous tokens. Recomputing their K and V every step
would be wasteful, so they are **cached**:

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
```

- Llama-2-7B (32 layers, 32 KV heads, head_dim 128), 4k tokens, fp16: **exactly 2 GiB per sequence**.
- A 70B model with 80 layers: with 64 KV heads it would need 10 GiB per 4k sequence; with **GQA's
  8 KV heads, 1.25 GiB**. That 8× saving is why GQA exists.
- The GPU lab measures Qwen2.5-0.5B's real cache (2 KV heads) and matches this formula exactly:
  6 MiB at 512 tokens, 24 MiB at 2048.

> **Exam link:** long context and large batches are limited by **KV-cache memory**, not FLOPs.
> The fixes are GQA/MQA, FP8 KV cache, and paged KV cache (Lab 05).

---

## Exercise 10 — `masked_mean_pool`: one vector per sentence

Encoders output one vector per token (`last_hidden_state`, shape `(batch, seq, dim)`). For search
and similarity you need **one vector per sentence**. Averaging the tokens works well, but
**padding tokens must be excluded**:

```
hidden = [[1,2], [3,4], [100,100]]     mask = [1, 1, 0]   (third token is padding)
masked mean = ([1,2] + [3,4]) / 2 = [2, 3]
naive mean  = [34.7, 35.3]            ← padding garbage dominates
```

Code: `mask[..., None]` turns the mask into shape `(batch, seq, 1)` so it broadcasts over features.
Multiply to zero out padding, sum over tokens, and divide by the number of *real* tokens (clipped
to avoid ÷0).

> **Exam link:** with Hugging Face BERT, `outputs.pooler_output` is "the pooled output": the
> [CLS] vector passed through dense + tanh. Masked mean pooling of `last_hidden_state` is what
> Sentence-Transformers use and is usually better for similarity.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| "why scale by √d_k" | softmax saturation → vanishing gradients |
| "reduce KV cache", "long context memory" | GQA/MQA, FP8 KV, paged attention |
| "FlashAttention" | exact attention, IO-aware tiling; not an approximation |
| "relative position", "extend context" | RoPE (+ rope scaling) |
| "sentence embedding from BERT" | `pooler_output` or masked mean pooling |
| "which architecture for generation / embeddings / translation" | decoder-only / encoder-only / encoder-decoder |
| "temperature → 0" | softmax becomes argmax = greedy decoding |
