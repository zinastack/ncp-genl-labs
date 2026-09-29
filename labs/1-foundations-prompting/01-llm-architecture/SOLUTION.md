# Lab 01 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves, its benefits, and the alternatives. This is what the exam tests.
- **How it works:** the idea in plain words, with a small example using real numbers (all computed by running `solutions.py`).
- **The code:** why each line is there and what breaks without it.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The exam never asks you to write these formulas. It asks *which technique solves which problem*
and *which statement is true*. The code is how you build the intuition, and the "why" and "on the
exam" parts turn it into answers.

The big picture, one token at a time:

```
token ids ─► embedding (11) ─► + position (5, 6) ─► [ attention (1–4, 14) → FFN / MoE (15) ] × N (13) ─► LM head (11) ─► next token
                                                    norms (7) and residuals (13) around every sub-layer
          encoder vs decoder vs encoder-decoder = which mask and which attention (2, 12)
          numbers to size models and servers: parameters (8), KV cache (9); sentence vectors (10)
```

---

## Exercise 1 — `softmax`: turning scores into weights

### Why it exists

Attention produces **scores** (how relevant is each token?), and the output layer produces
**logits** (how likely is each next word?). Both are arbitrary real numbers. Softmax turns them into
**weights** that are positive and sum to 1, keeps their order, and, crucially, is **smooth**, so the
model can be trained with gradients.

### How it works

`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. Scores `[2.0, 1.0, 0.0]`:

| step | values |
|---|---|
| `exp` makes everything positive and widens gaps (+1 in score is ×2.7 in weight) | `[7.39, 2.72, 1.00]` |
| divide by the sum (11.11) | `[0.665, 0.245, 0.090]` |

**Why not simply pick the biggest score?** Attention doesn't *select* a token; it **blends all
value vectors** with these weights (`weights @ V`, exercise 3). With `v1=[1,0], v2=[0,1], v3=[1,1]`:
`0.665·v1 + 0.245·v2 + 0.090·v3 = [0.755, 0.335]`. Picking only the max (`argmax`) would:

1. **Lose information:** in "The animal didn't cross the street because *it* was too tired", *it* needs *animal* **and** context.
2. **Stop learning:** nudging a score from 2.00 to 2.01 doesn't change which token wins, so argmax has a **zero gradient**. With softmax the weight moves from 0.665 to 0.667, a smooth signal that gradients can follow.
3. **Hide uncertainty:** `[2.00, 1.99]` gives 100%/0% with argmax and about 50%/50% with softmax.

### The code

```python
x_max = np.max(x, axis=axis, keepdims=True)
x_max = np.where(np.isfinite(x_max), x_max, 0.0)
e = np.exp(x - x_max)
total = np.sum(e, axis=axis, keepdims=True)
return e / np.where(total == 0, 1.0, total)
```

- **Subtract the max:** in float64, `exp(710) = inf` and `inf/inf = nan`. Subtracting the same number from every score changes nothing (`exp(x−c) = exp(x)/exp(c)`, and `exp(c)` cancels), so `softmax([1000, 1001, 1002]) = softmax([0, 1, 2]) = [0.090, 0.245, 0.665]`. With `c = max` the biggest term is `exp(0) = 1`: no overflow, and the sum is at least 1.
- **`axis=-1, keepdims=True`:** each query's row sums to 1, and `(…, 1)` broadcasts so each row subtracts its own max.
- **`-inf` means masked:** `exp(-inf) = 0`, so blocked keys get exactly 0%. Finite scores never reach exactly 0 (`exp(-5) = 0.0067`).
- **The two `np.where` guards:** a row that is entirely `-inf` (a query allowed to see nothing, which happens with padding) would compute `-inf − (-inf) = nan` and then `0/0 = nan`. The guards return all zeros instead. A single `nan` poisons the whole model.

### On the exam

> **Temperature** is softmax with a knob: `softmax(x / T)`. As `T → 0` the weights approach
> `[1, 0, 0]`, which is argmax = **greedy decoding**. A high `T` flattens the distribution, so sampling is more random.
> Low temperature gives **consistency, not correctness**. It doesn't reduce hallucination. (Lab 02 goes deeper.)

---

## Exercise 2 — `causal_mask`: no peeking at the future

### Why it exists

A decoder LLM is trained to predict token *t+1* from tokens 0…t, **for every position of a sentence
in a single pass** (that parallelism is what makes training fast). Without a mask, position 1 could
attend to the word it's supposed to predict and simply copy it. It would score perfectly in training
and fail at generation, when the future doesn't exist yet.

### How it works

```
                keys →   The   cat   sat   down
queries ↓  The         [  ✓     ✗     ✗     ✗  ]
           cat         [  ✓     ✓     ✗     ✗  ]      np.tril(np.ones((n, n), dtype=bool))
           sat         [  ✓     ✓     ✓     ✗  ]      lower triangle incl. diagonal
           down        [  ✓     ✓     ✓     ✓  ]
```

`False` becomes `-inf` before softmax, so future tokens get exactly 0% weight. The diagonal is
`True`: a token always sees itself, so a causal row is never fully masked.

### On the exam

**The mask defines the architecture family** (exercise 12 builds the other two):

| family | attention mask | trained to | best for | examples |
|---|---|---|---|---|
| **decoder-only** | **causal** (past only) | predict the next token | generation, chat, in-context learning | GPT, Llama, Mistral, Nemotron |
| **encoder-only** | **padding only** (sees both directions) | fill in masked words (MLM) | classification, NER, **embeddings**, reranking | BERT, RoBERTa, DeBERTa |
| **encoder-decoder** | encoder bidirectional + decoder causal + **cross-attention** | map an input sequence to an output sequence | translation, summarisation, speech-to-text | T5, BART, Whisper |

*"Why can't BERT generate open-ended text like GPT?"* It was trained **bidirectionally** (no
causal mask) to fill in blanks, not to predict the next word from the past only.

---

## Exercise 3 — `scaled_dot_product_attention`: the heart of the transformer

### Why it exists

Before transformers, RNNs read text one word at a time, so information from word 1 had to survive
through every step to reach word 500, and training couldn't be parallelised. **Attention lets every
token look directly at every other token in one step**, and all positions are computed in parallel
on a GPU. That's why transformers scaled to today's LLMs.

### How it works

`Attention(Q, K, V) = softmax(QKᵀ / √d_k + mask) · V`, where each token is projected into:
- **Query:** "what am I looking for?"
- **Key:** "what do I contain?"
- **Value:** "what do I pass on?"

Worked example (3 tokens, `d_k = 2`, causal): `Q = K = [[1,0],[0,1],[1,1]]`, `V = [[10,0],[0,10],[5,5]]`

| step | code | result (row = query) |
|---|---|---|
| scores | `q @ kᵀ` | `[[1,0,1], [0,1,1], [1,1,2]]` |
| scale | `/ √2` | `[[.707,0,.707], [0,.707,.707], [.707,.707,1.414]]` |
| mask | `where(mask, …, -inf)` | `[[.707,-inf,-inf], [0,.707,-inf], [.707,.707,1.414]]` |
| softmax | per row | `[[1,0,0], [.33,.67,0], [.25,.25,.50]]` |
| blend | `weights @ v` | `[[10,0], [3.3,6.7], [5,5]]` |

### Why divide by √d_k

A dot product sums `d_k` products, so its spread grows like `√d_k`. Measured on random vectors:
`d=4 → 2.0`, `d=64 → 8.0`, `d=512 → 22.4`, and after dividing by `√d_k` all are ≈1.0. Big scores make
softmax nearly one-hot (8 random scores at d=512: top weight 0.985 unscaled vs 0.308 scaled), and
nearly one-hot means **vanishing gradients**.

### The code

`np.swapaxes(k, -1, -2)` transposes only the last two axes, so batch and head dimensions pass
through. Mask **before** softmax (as `-inf`), so rows still sum to 1. Return the weights too; they're
what attention visualisations plot.

### On the exam

*Why does scaled dot-product attention divide QKᵀ by √d_k?*

| option | verdict |
|---|---|
| To keep dot products from growing with dimension, which would push softmax into saturated regions with vanishing gradients | ✅ |
| To normalise attention weights so each row sums to one | ❌ softmax already does that |
| To reduce the FLOPs of the attention matmul | ❌ dividing costs extra; nothing is saved |
| To make attention invariant to token order | ❌ attention is *already* order-blind, which is the problem exercise 5 solves |

---

## Exercise 4 — `multi_head_attention`: several attentions in parallel

### Why it exists

One attention head produces one weighting, so it can focus on one kind of relation at a time. Text
has many simultaneous relations: grammar (subject↔verb), reference (*it*↔*animal*), position
(neighbouring words). **Multi-head** attention runs `h` smaller heads in parallel, each free to
specialise, then merges them, at about the cost of one big head.

### How it works

`seq=5, d_model=8, h=2 → head_dim 4`:

```
x (5,8) → x @ w_q (5,8) → reshape (5,2,4) → transpose (2,5,4)   heads first
→ attention per head (2,5,4) → transpose back (5,2,4) → reshape (5,8) → @ w_o (5,8)
```

The width is **split**, not multiplied, so compute and parameters match single-head attention of
the same `d_model`. `w_o` mixes what the heads found.

### On the exam: MHA, MQA and GQA

| variant | K/V heads | KV-cache size | used by |
|---|---|---|---|
| multi-head (MHA) | one per query head | largest | GPT-2, Llama-2-7B |
| multi-query (MQA) | 1 shared by all | smallest, slight quality loss | PaLM, Falcon |
| **grouped-query (GQA)** | a few groups (e.g. 8 for 64 query heads) | ÷8 here, near-MHA quality | Llama-2-70B, Llama-3, Mistral, Qwen |

*"Which architectural feature most directly reduces the KV-cache footprint per token?"*
GQA/MQA (fewer KV heads). Wrong answers you'll see: SwiGLU (the FFN activation), RMSNorm (the
norm type) and weight tying (embeddings). None of them touch the cache. Exercise 9 puts numbers on it.

---

## Exercise 5 — `sinusoidal_positions`: telling the model where each token is

### Why it exists

**Attention is order-blind.** Measured with your `multi_head_attention` (no causal mask, no
positions): shuffling the input tokens simply shuffles the outputs the same way. The model computes
exactly the same thing for "dog bites man" and "man bites dog". Adding position vectors breaks this,
and with sinusoidal positions added, shuffled input no longer gives shuffled output. **Word order
only exists for the model if you inject it.**

### How it works

The original Transformer **adds** a fixed vector to each token embedding:
`PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(same)`. With `d = 4`:

```
pos 0: [0.000,  1.000, 0.000, 1.000]
pos 1: [0.841,  0.540, 0.010, 1.000]
pos 2: [0.909, -0.416, 0.020, 1.000]
```

Low dimensions oscillate fast (fine position) and high dimensions slowly (coarse position), like
the hands of a clock. There's nothing to learn: 0 parameters.

### The four ways to encode position (the exam's favourite comparison)

| method | how | extra params | relative distance? | longer than training? | used by |
|---|---|---|---|---|---|
| **learned absolute** | a trainable vector per position, **added** | `n_ctx × d` (GPT-2: 786k) | no | **no**: there's no vector for position 1025 | GPT-2, BERT |
| **sinusoidal** | fixed sin/cos vector, **added** | 0 | indirectly | in theory; poorly in practice | original Transformer |
| **RoPE** | **rotate** Q and K by position | 0 | **yes, exactly** | yes, with rope scaling (NTK, YaRN) | Llama, Mistral, Qwen, Nemotron |
| **ALiBi** | subtract a distance penalty from scores | 0 | yes | good extrapolation | BLOOM, MPT |

Code: `np.arange(0, d, 2)` gives the even indices `2i`; `pe[:, 0::2]` and `pe[:, 1::2]` fill even and odd columns.

---

## Exercise 6 — `apply_rope`: rotary position embeddings

### Why it exists (and why it won)

What matters in language is mostly **relative** position: an adjective modifies the noun *right after it*,
wherever the phrase sits in the document. Absolute schemes make the model learn that "position 5 vs 6"
and "position 505 vs 506" are the same relation separately. RoPE builds relativity into the maths:

- **Relative by construction:** the query·key score depends only on the **distance** between tokens.
- **No parameters:** nothing to learn, and no table limited to a maximum length.
- **Context extension:** because position is an angle, rescaling the angles ("rope scaling", NTK, YaRN)
  stretches a model trained at 4k tokens to 32k or 128k with little fine-tuning. That is how most long-context models are made.
- **Applied to Q and K only**, inside attention, so values and the residual stream carry no position
  noise, and cached keys (KV cache) are simply stored already rotated.

### How it works

Split each Q/K vector into pairs and **rotate** each pair by `position × θ_i` (a different speed
θ per pair):

```
out0 = x0·cos(pθ) − x1·sin(pθ)
out1 = x0·sin(pθ) + x1·cos(pθ)        standard 2-D rotation
```

`x = [1, 0, 0, 1]`, speeds `[1, 0.01]`: position 1 gives `(0.540, 0.841, -0.010, 1.000)`, and
position 2 rotates twice as far, `(-0.416, 0.909, -0.020, 1.000)`.

**The relative property, measured** (same q, k vectors):

| q position | k position | q·k |
|---|---|---|
| 3 | 1 | 8.7036 |
| 10 | 8 | 8.7036 |
| 100 | 98 | 8.7036 |
| 10 | 1 | 10.0526 (different distance, different score) |

Why: rotating q by angle `mθ` and k by `nθ` leaves the angle **between** them as `(m−n)θ`, and a dot
product only depends on that angle. Rotation also keeps vector length, and position 0 is the identity.

### On the exam

*Which property of Rotary Position Embeddings (RoPE) makes it popular in modern LLMs?*

| option | verdict |
|---|---|
| The query–key dot product depends only on the relative distance between positions | ✅ the table above |
| It adds a learned vector per absolute position to the token embedding | ❌ that describes **learned absolute** embeddings (GPT-2, BERT): absolute, added and learned. RoPE is relative, a rotation, and has no parameters |
| It removes the need for a causal mask | ❌ position encoding says *where* tokens are; the mask says *which* tokens are visible. Llama uses both |
| It makes attention linear in sequence length | ❌ RoPE doesn't change the n×n score matrix; cost stays O(n²). Linear attention (Performer, Linformer) is a different, approximate technique |

Other RoPE facts that get asked: extending context means rope scaling; it's applied to Q and K,
not V; and it adds no parameters.

---

## Exercise 7 — `layer_norm` and `rms_norm`: keeping activations in range

### Why it exists

A deep network multiplies through dozens of layers. Without control, activations drift very large
(overflow, `nan`) or very small (vanishing signal), and the "right" learning rate differs layer by
layer, so training becomes unstable. **Normalisation rescales each token's vector to a standard
size at every layer**, keeping everything in a range where gradients behave. Learned `γ` (and `β`)
let the model undo it where useful.

### How it works

`x = [1, 2, 3, 6]` → mean 3, variance 3.5, RMS 3.536

| | formula | result |
|---|---|---|
| **LayerNorm** | `(x − mean)/√(var + ε) · γ + β` | `[-1.069, -0.535, 0, 1.604]` (mean 0, variance 1) |
| **RMSNorm** | `x/√(mean(x²) + ε) · γ` | `[0.283, 0.566, 0.849, 1.697]` (rescaled, not centred) |

- `axis=-1, keepdims=True`: normalise **each token's features** separately. That's why it's *layer* norm, not *batch* norm: it doesn't depend on other examples, so it works with any batch size, including 1 at inference.
- `ε` avoids dividing by zero for a constant vector.

### On the exam

- **RMSNorm** (Llama, Mistral, Nemotron) drops the mean subtraction and the bias. It's cheaper, works as well, and is **not** related to the KV cache or sequence length.
- **Pre-norm vs post-norm** is about *where* the norm sits in the block (exercise 13), a different question from *which* norm.

---

## Exercise 8 — `gpt2_param_count`: where the 124 M parameters live

### Why it matters

Parameter counts drive everything you'll size later: weights memory (Lab 05), training FLOPs
≈ 6·N·tokens (Lab 06), and which GPU a model fits on. You need to reason about *where* parameters
sit, because the exam asks which part to target (LoRA, MoE, vocabulary).

### How it works (GPT-2 small: d = 768, L = 12, V = 50,257, n_ctx = 1,024)

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
| **total** | | **124,439,808** (the GPU lab confirms it against the real model) |

### On the exam

*With a 4× FFN expansion, what fraction of a block's parameters is in the FFN?* **About two thirds.**
Attention has 4 matrices of d×d = `4d²`; the FFN has d×4d + 4d×d = `8d²`; 8/12 ≈ 67%.
That's why **MoE replaces the FFN** (exercise 15) and why LoRA often targets the MLP matrices too.

---

## Exercise 9 — `kv_cache_bytes`: the memory that limits serving

### Why it exists

Generation produces one token at a time, and each new token attends to all previous ones. Without
a cache, step 1000 would recompute K and V for 999 old tokens, making generation quadratic.
**Caching K and V** makes each step compute only the new token, at the price of **memory that grows
with every token and every concurrent user**.

### How it works

`bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value`

- Llama-2-7B (32 layers, 32 KV heads, 128), 4k tokens, fp16: **exactly 2 GiB per sequence**.
- A 70B-class model (80 layers) at 4k tokens: **10 GiB** per sequence with 64 KV heads, **1.25 GiB** with GQA's 8.
- The GPU lab measured Qwen2.5-0.5B's real cache (2 KV heads): 6 MiB at 512 tokens and 24 MiB at 2048, exactly this formula.

### On the exam

Long context and large batches are limited by **KV-cache memory**, not FLOPs. The fixes, and where
they're covered: **GQA/MQA** (fewer KV heads, exercise 4), **FP8 KV cache** (fewer bytes, Lab 05)
and **PagedAttention** (no wasted reserved memory, Lab 05). Compute-side tricks such as
FlashAttention don't shrink the cache.

---

## Exercise 10 — `masked_mean_pool`: one vector per sentence

### Why it exists

Semantic search, RAG retrieval, clustering and deduplication all compare **whole texts**, so each text
needs **one vector**. Encoders output one vector *per token*; pooling collapses them.

### How it works

```
hidden = [[1,2], [3,4], [100,100]]   mask = [1, 1, 0]   (third token is padding)
masked mean = ([1,2] + [3,4]) / 2 = [2, 3]
naive mean  = [34.7, 35.3]           ← padding garbage dominates
```

`mask[..., None]` gives shape `(batch, seq, 1)`, which broadcasts over features. Zero out padding,
sum, and divide by the real token count (clipped to avoid ÷0).

### On the exam

*With Hugging Face BERT, which retrieves the pooled sentence embedding?*

| option | verdict |
|---|---|
| `outputs.pooler_output` | ✅ the [CLS] vector passed through dense + tanh: "the pooled output" |
| `outputs.logits[:, 0, :]` | ❌ a bare encoder (`AutoModel`) has no logits |
| `model.generate(...).sequences` | ❌ generate is for decoders/seq2seq, and returns token ids, not embeddings |
| `outputs.hidden_states[0]` | ❌ that's the embedding-layer output (layer 0), per token, and only with `output_hidden_states=True` |

In practice, **masked mean pooling** (this exercise) is usually the better similarity embedding. It's what Sentence-Transformers use.

---

## Exercise 11 — `embed` and `lm_head`: words in, words out

### Why it exists

Neural networks work on numbers, not words. A tokenizer turns text into **ids**, and the
**embedding matrix** turns each id into a learned vector where similar meanings end up close
together. At the other end, the **LM head** turns the final hidden state back into one score per
vocabulary word. **Weight tying** uses the *same* matrix for both. The intuition: "a word's meaning
vector" and "what a hidden state that predicts this word looks like" should agree. It also saves a whole `V × d` matrix.

### How it works

```
embedding (vocab 3, d 2) = [[1.0, 0.0],   ← token 0
                            [0.0, 1.0],   ← token 1
                            [0.7, 0.7]]   ← token 2
embed([[2, 0]])       → [[[0.7, 0.7], [1.0, 0.0]]]          just a row lookup
lm_head([0.9, 0.1])   → [0.9, 0.1, 0.7]  → token 0 scores highest (most similar direction)
```

- `embedding[token_ids]` is fancy indexing: the same result as `one_hot(ids) @ embedding` without building the one-hot matrix.
- `hidden @ embedding.T` gives one dot product per vocabulary word. Softmax (exercise 1) then turns these logits into next-token probabilities.

### On the exam: vocabulary size is a trade-off

Going from a 32k to a 128k vocabulary at d = 4096:
- **Benefit:** fewer tokens per text (better for code and non-English languages), which means shorter sequences, lower attention cost, smaller KV cache and more text per context window.
- **Cost:** embedding parameters grow from 131 M (32,000 × 4,096) to 525 M (128,256 × 4,096), per matrix. Untied (input + output), that's about **0.8 B extra**, and the final softmax over the vocabulary gets **more** expensive, not less.
- It doesn't change layer count or head size (a common distractor).

---

## Exercise 12 — `padding_mask` and `cross_attention`: the other two families

### Why they exist

With exercise 2 you built the **decoder's** causal mask. The other two families differ in exactly two ideas:

- **Encoder (BERT):** understanding tasks can read the *whole* input at once, so there's no causal mask. The only thing to hide is **padding**, the filler tokens that make batch rows equal length.
- **Encoder-decoder (T5, BART, Whisper):** the decoder generates the output (causal), and at every layer it also looks at the **encoded input** through **cross-attention**. Translation needs this, because each output word may depend on any source word.

### How it works

**Padding mask:** `[[1, 1, 1, 0]]` becomes shape `(1, 1, 4)`: `True` for real keys, one row that broadcasts over all queries.
- Token 0 attends to token 2, which comes *after* it (bidirectional; the test checks `w[0,0,2] > 0`).
- Every query gives the padding key weight 0, and changing the padding token changes no real output.

**Cross-attention:** queries from the decoder, keys and values from the encoder:

```python
q, k, v = x_dec @ w_q, enc_out @ w_k, enc_out @ w_v      # (n_dec, d), (n_enc, d), (n_enc, d)
out = attention(q, k, v, enc_mask)                        # (n_dec, d): one row per output token
```

- Changing decoder token 2 changes only output row 2 (each query is independent).
- Changing any source token can change **every** decoder row (every output word can consult the whole source).
- No causal mask here: the source is fully known before decoding starts. Only source *padding* is masked.

### On the exam

*A team needs (1) embeddings for semantic search and (2) open-ended chat.* **Encoder-only for (1),
decoder-only for (2).** Encoder-decoder suits transduction (translation, summarisation) but isn't
the default for either. See the family table in exercise 2.

---

## Exercise 13 — `transformer_block`: residuals and where the norm goes

### Why it exists

A transformer is the **same block stacked N times** (GPT-2 small: 12, Llama-70B: 80). Two design
choices make deep stacks trainable:

- **Residual connections** (`x + f(x)`): each sub-layer only adds a *correction* to a running
  "residual stream". The input has a direct highway to the output, so gradients reach the first
  layers undiminished, and a useless layer can learn `f ≈ 0` and get out of the way.
- **Where the norm sits:**
  - **Post-norm** (original Transformer, BERT): `x = norm(x + f(x))`, which normalises *the residual stream itself* after every sub-layer.
  - **Pre-norm** (GPT-2 onwards, Llama, Nemotron): `x = x + f(norm(x))`, which normalises only the sub-layer's *input* and leaves the highway untouched.

### How it works

The test shows the difference in one line each. With empty sub-layers (`f = 0`):
- pre-norm: `x + 0 + 0 = x`, a **clean identity path**;
- post-norm: `norm(norm(x))`, so the stream is rescaled at every layer even when the layer does nothing.

In practice that means post-norm deep models are **unstable early in training** and need careful
learning-rate **warm-up**. Pre-norm trains stably at great depth, which is why every modern LLM uses it.

### On the exam

*Why do most modern decoder LLMs use pre-norm?*

| option | verdict |
|---|---|
| Pre-norm keeps a clean residual path, giving more stable gradients in very deep stacks | ✅ |
| Pre-norm removes the need for residual connections | ❌ the residuals are exactly what pre-norm protects |
| Pre-norm halves the parameters per block | ❌ same norms, just moved |
| Pre-norm is required for the causal mask to work | ❌ masking is unrelated to norm placement |

---

## Exercise 14 — `flash_attention`: same answer, far less memory traffic

### Why it exists

Standard attention builds the full n × n score matrix and writes it to GPU memory (HBM), then reads
it back for softmax, then again to multiply by V. At 8,192 tokens that's **128 MiB per head per
sequence** in fp16. The GPU spends most of its time **moving** that matrix, not doing math, and long
contexts run out of memory. **FlashAttention computes exactly the same result without ever storing
the full matrix.** It works on small tiles that fit in fast on-chip SRAM (a 128×128 fp16 tile is 32 KiB).

### How it works: the online softmax

Softmax needs the max and the sum of a **whole row**, which you don't have if you only see one
block of keys at a time. The trick is to keep running values and **correct them** when a bigger max shows up:

```
for each block of keys:
    s = q @ k_blockᵀ / √d                    scores for this block only
    m_new = max(m, max(s))                   the running max may grow
    p = exp(s − m_new)
    c = exp(m − m_new)                       earlier sums used the old max: rescale them
    l   = l·c   + sum(p)                     running softmax denominator
    acc = acc·c + p @ v_block                running weighted sum of values
    m = m_new
return acc / l
```

The test runs block sizes 1, 3, 4 and 10, with and without the causal mask, and requires the result
to equal your exercise 3 to ~1e-15. **It's the same number, computed in a different order.**
The first block always contains key 0, which every query may see, so `m` becomes finite straight
away and no `nan` can appear.

### On the exam

*Which statement about FlashAttention is correct?*

| option | verdict |
|---|---|
| It computes **exact** attention but tiles the work in on-chip SRAM, never materialising the n×n matrix in HBM | ✅ **IO-aware**: less memory traffic, less activation memory |
| It approximates attention with low-rank projections for linear complexity | ❌ that's Linformer/Performer (approximate). FlashAttention is exact and still O(n²) FLOPs |
| It only speeds up inference, not training | ❌ it speeds up both and saves activation memory in training |
| It requires changing the architecture and retraining | ❌ it's a drop-in kernel for the same maths |

The GPU lab measures this: the math kernel's memory grows ~n², flash stays small. FlashAttention-2 needs sm_80+ (A100, L4, H100), so a T4 shows it as unsupported.
Don't confuse it with **PagedAttention** (Lab 05), which is about *storing the KV cache* without wasted memory, not computing attention.

---

## Exercise 15 — `moe_layer`: more parameters, same compute

### Why it exists

Bigger models know more, but every parameter costs compute on every token. **Mixture of Experts**
replaces the FFN (where ⅔ of the parameters live, exercise 8) with *several* FFNs ("experts") and
a small **router** that sends each token to only the **top-k**. The model *has* many parameters but
*uses* few per token: much more capacity at roughly the compute of a small model.

### How it works

```
router logits = x @ router_w                 (tokens, n_experts): how suitable each expert is
chosen = top-k experts per token             e.g. 2 of 8
gates  = softmax over the chosen k logits    renormalised to sum to 1
output = Σ gate × expert(x)                  only k experts run for this token
```

The test checks one token by hand. With identical experts (each doubling its input) and top_k = 3,
the output is exactly `2·x`: the gates always sum to 1.

**Mixtral-8x7B-shaped numbers** (`moe_param_counts`, SwiGLU experts, 32 layers):
**45.1 B expert parameters in total, 11.3 B active per token** (2 of 8), exactly a quarter.

### On the exam

*8 experts, top-2 routing, compared with a dense model of the same total size?*

| option | verdict |
|---|---|
| Fewer FLOPs per token, but **all** experts must stay in GPU memory | ✅ any token may choose any expert |
| Lower memory, because only 2 experts are loaded | ❌ the router's choice changes token by token; all must be resident |
| Same FLOPs, better quality through ensembling | ❌ only k of E experts compute |
| Smaller KV cache because experts share attention | ❌ MoE replaces the FFN; attention and the cache are unchanged |

Serving many experts across GPUs is **expert parallelism**, and routing tokens to them uses the **all-to-all** collective (Lab 06).

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise(s) | Also in |
|---|---|---|
| Encoder embeddings (pooler_output) | 10 | GPU lab (real BERT) |
| Attention scaling (√d_k) | 3 | |
| Architecture families | 2, 12 | |
| GQA / KV cache | 4, 9 | GPU lab (measured cache) |
| Positional encoding (RoPE) | 5, 6 | |
| FlashAttention | 14 | GPU lab (kernel benchmark) |
| Parameter distribution | 8 | GPU lab (real GPT-2 count) |
| Mixture of Experts | 15 | |
| Normalisation (pre-norm) | 7, 13 | |
| Vocabulary size | 11 | |
