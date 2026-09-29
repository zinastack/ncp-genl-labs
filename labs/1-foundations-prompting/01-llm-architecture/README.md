# Lab 01 — LLM Architecture (6% of exam)

> Blueprint: *understanding and applying foundational LLM structures and mechanisms.*

You will build the core of a transformer from scratch in NumPy (embeddings, attention, masking
for all three architecture families, multi-head attention, positional encodings, normalisation,
a full block, FlashAttention's tiling and a Mixture-of-Experts layer) and write the
calculators the exam expects you to do in your head: parameter counts and KV-cache size.

```
make test-01          # YOUR exercises (run from the repo root)
make solutions-01     # reference solutions
make quiz-01          # exam-style questions
make gpu-01           # GPU part (gpu_lab.py) on the Brev/AWS instance
```

---

## 1. The decoder-only transformer, end to end

```
token ids ─► embedding (V×d) ─► + position info ─► N × block ─► final norm ─► LM head (d×V) ─► logits
                                                     │
                     block (pre-norm, used by GPT-2/Llama/Nemotron):
                     x = x + Attention(Norm(x))
                     x = x + FFN(Norm(x))
```

| Piece | What to remember |
|---|---|
| **Self-attention** | `softmax(QKᵀ / √d_k + mask) · V`. The `√d_k` keeps dot products from growing with dimension and saturating softmax (tiny gradients). |
| **Causal mask** | Decoder-only models add `-inf` above the diagonal so token *i* only sees tokens ≤ *i*. Encoders (BERT) use no causal mask, only a padding mask. |
| **Multi-head attention** | Split `d_model` into `h` heads of size `d_model/h`; each head learns a different relation; concat then project with `W_o`. |
| **MQA / GQA** | Multi-Query: all heads share 1 K/V head. Grouped-Query: `g` K/V groups (Llama-2-70B, Llama-3). Both shrink the **KV cache** and speed decoding with little quality loss. |
| **FFN** | Two linear layers, hidden ≈ 4·d (GELU) or ≈ 8/3·d with **SwiGLU** (Llama). Holds ~2/3 of the parameters. |
| **Norm** | LayerNorm (mean+variance) or **RMSNorm** (variance only, no centring, cheaper; Llama). **Pre-norm** trains more stably than post-norm at depth. |
| **Residuals** | Let gradients flow through deep stacks; every sub-layer is `x + f(x)`. |
| **LM head** | Often **weight-tied** to the input embedding (GPT-2), saving V·d parameters. |
| **MoE** | Replace the FFN with *E* experts + a router picking top-k (Mixtral: 8 experts, top-2). Total params ≫ active params per token. |

## 2. Three architecture families

| Family | Examples | Pre-training objective | Best for |
|---|---|---|---|
| Encoder-only | BERT, RoBERTa, DeBERTa | Masked LM (bidirectional) | Classification, NER, **embeddings**, reranking |
| Decoder-only | GPT, Llama, Mistral, Nemotron | Causal LM (next token) | Generation, chat, in-context learning |
| Encoder-decoder | T5, BART, (Whisper) | Span corruption / denoising, seq2seq | Translation, summarisation, structured transduction |

## 3. Position information

- **Learned absolute** (GPT-2, BERT): one vector per position, so the model can't go beyond `n_ctx`.
- **Sinusoidal** (original Transformer): `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(...)`.
- **RoPE** (Llama, Mistral, Nemotron): rotate each (even, odd) pair of Q and K by angle `pos·θ_i`.
  The dot product then depends only on the **relative** offset. Context can be extended by
  scaling θ (NTK / YaRN / "rope scaling").
- **ALiBi** (BLOOM, MPT): add a linear distance penalty to attention scores; extrapolates well.

## 4. Numbers you should be able to derive

**Parameters (GPT-style, per layer)** ≈ `12·d²` (attention `4d²` + FFN `8d²`), so the total is
≈ `12·L·d² + V·d`. GPT-2 small (L=12, d=768, V=50257) has **124,439,808** parameters including
biases, LayerNorms and position embeddings. You compute this exactly in the exercises.

**KV cache** = `2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes`.
Llama-2-7B, 4k context, fp16, batch 1: `2·32·32·128·4096·2 B = 2 GiB`. GQA with 8 KV heads
cuts that by 4×. This is why long context and large batches are limited by **memory**, not FLOPs.

**Attention cost** scales as O(n²·d) in sequence length. **FlashAttention** is *exact*
attention, made faster by tiling in SRAM so the n×n matrix is never written to HBM
(IO-aware). It doesn't approximate anything.

## 5. Getting embeddings out of an encoder (a favourite exam question)

With Hugging Face `outputs = model(**inputs)` on a BERT model:

- `outputs.last_hidden_state` has shape `(batch, seq, hidden)`: one vector per token.
- `outputs.pooler_output` has shape `(batch, hidden)`: the `[CLS]` vector passed through
  dense + tanh (trained for next-sentence prediction). **This is "the pooled output".**
- `outputs.last_hidden_state[:, 0]` is the raw `[CLS]` token.
- **Mean pooling with the attention mask** is usually the best sentence embedding for
  similarity (Sentence-Transformers). Padding tokens must be excluded from the mean.

## 6. Exam traps

- "Scaling by √d_k" is about **softmax saturation / gradient stability**, not speed.
- FlashAttention = exact + IO-aware. Sparse or linear attention = approximate.
- GQA/MQA reduce **KV-cache memory and bandwidth** at inference. They don't cut training FLOPs much.
- Encoder models are not autoregressive. Don't pick BERT for open-ended generation.
- A bigger vocabulary means shorter sequences but a larger embedding + LM head (`V·d` each if untied).
- Temperature, top-k and top-p are **decoding** choices, not architecture (covered in Lab 02).

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `softmax` | numerically stable softmax |
| 2 | `causal_mask` | autoregressive masking |
| 3 | `scaled_dot_product_attention` | the attention equation |
| 4 | `multi_head_attention` | head split / merge, output projection |
| 5 | `sinusoidal_positions` | absolute positional encoding |
| 6 | `apply_rope` | rotary embeddings and the relative-position property |
| 7 | `layer_norm`, `rms_norm` | normalisation variants |
| 8 | `gpt2_param_count` | exact parameter accounting |
| 9 | `kv_cache_bytes` | inference memory planning (MHA vs GQA) |
| 10 | `masked_mean_pool` | sentence embeddings from an encoder |
| 11 | `embed`, `lm_head` | token embeddings, weight tying, vocabulary-size trade-off |
| 12 | `padding_mask`, `cross_attention` | encoder (bidirectional) and encoder-decoder attention |
| 13 | `transformer_block` | residual connections, pre-norm vs post-norm |
| 14 | `flash_attention` | tiled "online softmax": exact attention without the n×n matrix |
| 15 | `moe_layer`, `moe_param_counts` | Mixture of Experts: top-k routing, total vs active parameters |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).

GPU part (`gpu_lab.py`): BERT pooling on a real model, GPT-2's real parameter count vs yours, a measured
GQA KV cache vs your formula, and math vs memory-efficient vs FlashAttention kernels.
