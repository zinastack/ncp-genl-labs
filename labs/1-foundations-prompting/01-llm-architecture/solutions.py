"""Lab 01 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

from collections.abc import Callable

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # softmax(x) == softmax(x - c) for any constant c (exp(c) cancels top and bottom).
    # Subtracting the row max makes the largest exponent exp(0) = 1, so exp() can't overflow.
    x_max = np.max(x, axis=axis, keepdims=True)  # keepdims → shape (..., 1) broadcasts per row
    # A fully-masked row (all -inf) has max -inf, and -inf - (-inf) = nan. Use 0 instead.
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)  # masked entries: exp(-inf) = 0 → exactly 0% weight
    total = np.sum(e, axis=axis, keepdims=True)
    # A fully-masked row sums to 0; return zeros ("attend to nothing") instead of 0/0 = nan.
    return e / np.where(total == 0, 1.0, total)


def causal_mask(n: int) -> np.ndarray:
    # Lower triangle incl. diagonal: query i may see keys 0..i, never the future.
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = q.shape[-1]
    # 1. Scores: how well each query matches each key. swapaxes transposes only the last two
    #    axes, so leading batch/head dimensions pass through unchanged.
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)  # √d_k keeps softmax out of saturation
    # 2. Mask BEFORE softmax: blocked positions become -inf → weight exactly 0.
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    # 3. Scores → weights (each query's row sums to 1).
    weights = softmax(scores, axis=-1)
    # 4. Blend the value vectors with those weights.
    return weights @ v, weights


def multi_head_attention(
    x: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray, w_o: np.ndarray, n_heads: int,
    causal: bool = True,
) -> np.ndarray:
    seq, d_model = x.shape
    head_dim = d_model // n_heads  # the width is split across heads, not multiplied

    def split(t):  # (seq, d_model) -> (seq, heads, head_dim) -> (heads, seq, head_dim)
        return t.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)

    q, k, v = split(x @ w_q), split(x @ w_k), split(x @ w_v)
    mask = causal_mask(seq) if causal else None
    out, _ = scaled_dot_product_attention(q, k, v, mask)  # runs for all heads at once (batched)
    merged = out.transpose(1, 0, 2).reshape(seq, d_model)  # concatenate the heads back
    return merged @ w_o  # output projection mixes information across heads


def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]  # (seq, 1)
    i = np.arange(0, d_model, 2)[None, :]  # even dims 2i → (1, d/2)
    angles = pos / np.power(10000.0, i / d_model)  # (seq, d/2): fast waves in low dims, slow in high
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)  # even columns
    pe[:, 1::2] = np.cos(angles)  # odd columns
    return pe


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    d = x.shape[-1]
    inv_freq = base ** (-np.arange(0, d, 2) / d)  # one rotation speed per feature pair
    theta = positions[:, None] * inv_freq[None, :]  # angle table: (seq, d/2)
    cos, sin = np.cos(theta), np.sin(theta)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]  # the two coordinates of each pair
    out = np.empty_like(x, dtype=float)
    # Standard 2-D rotation of every pair. Because rotations compose, q·k after RoPE depends
    # only on the position DIFFERENCE between query and key.
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # Per token (last axis): centre to mean 0, scale to variance 1, then learned scale/shift.
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Like LayerNorm without centring or bias: divide by the root-mean-square (Llama).
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * gamma


def gpt2_param_count(vocab: int, n_ctx: int, d_model: int, n_layers: int) -> int:
    d = d_model
    embeddings = vocab * d + n_ctx * d  # token + learned position embeddings
    per_layer = (
        2 * (2 * d)                 # ln_1, ln_2 (weight + bias)
        + (d * 3 * d + 3 * d)       # fused QKV projection
        + (d * d + d)               # attention output projection
        + (d * 4 * d + 4 * d)       # MLP up (d → 4d)
        + (4 * d * d + d)           # MLP down (4d → d)
    )                               # ≈ 12·d²: attention 4d², MLP 8d²
    final_ln = 2 * d
    return embeddings + n_layers * per_layer + final_ln  # LM head is tied: adds nothing


def kv_cache_bytes(
    n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int,
    bytes_per_elem: int = 2,
) -> int:
    # 2 = one K and one V tensor per layer. GQA shrinks this by using fewer KV heads.
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def masked_mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    m = attention_mask[..., None].astype(hidden.dtype)  # (batch, seq, 1) broadcasts over features
    # Zero out padding, sum the real tokens, divide by how many real tokens there are.
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def embed(token_ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    # An embedding layer is just a row lookup: token id i → row i of the (vocab, d) matrix.
    # (Mathematically the same as one_hot(ids) @ embedding, without building the one-hot.)
    return embedding[token_ids]


def lm_head(hidden: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    # Tied output layer: score every vocabulary token by the dot product of the hidden state
    # with that token's embedding. Reusing the input matrix saves vocab × d parameters.
    return hidden @ embedding.T


def padding_mask(attention_mask: np.ndarray) -> np.ndarray:
    # Encoder (BERT-style) mask: every query may see every REAL token, before or after it.
    # (batch, seq) → (batch, 1, seq): the middle axis broadcasts over all queries.
    return attention_mask.astype(bool)[:, None, :]


def cross_attention(
    x_dec: np.ndarray, enc_out: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray,
    enc_mask: np.ndarray | None = None,
) -> np.ndarray:
    # Encoder-decoder (T5, BART): queries come from the decoder, keys and values from the
    # encoder's output. No causal mask: the whole source sentence is already known.
    q, k, v = x_dec @ w_q, enc_out @ w_k, enc_out @ w_v
    out, _ = scaled_dot_product_attention(q, k, v, enc_mask)  # enc_mask hides source padding
    return out


def transformer_block(
    x: np.ndarray, attn_fn: Callable[[np.ndarray], np.ndarray], ffn_fn: Callable[[np.ndarray], np.ndarray],
    norm_fn: Callable[[np.ndarray], np.ndarray], pre_norm: bool = True,
) -> np.ndarray:
    if pre_norm:
        # Pre-norm (GPT-2 onwards, Llama): normalise the sub-layer INPUT; the residual path
        # x → x + … is never touched, so signal and gradients flow straight through deep stacks.
        x = x + attn_fn(norm_fn(x))
        return x + ffn_fn(norm_fn(x))
    # Post-norm (original Transformer, BERT): normalise AFTER adding, so every layer rescales
    # the residual stream itself.
    x = norm_fn(x + attn_fn(x))
    return norm_fn(x + ffn_fn(x))


def flash_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int, causal: bool = False) -> np.ndarray:
    n_q, d = q.shape
    # Running statistics per query row: max score so far (m), softmax denominator so far (l),
    # and the un-normalised weighted sum of values so far (acc).
    m = np.full((n_q, 1), -np.inf)
    l = np.zeros((n_q, 1))
    acc = np.zeros((n_q, v.shape[1]))
    rows = np.arange(n_q)[:, None]
    for start in range(0, k.shape[0], block_size):
        kb, vb = k[start : start + block_size], v[start : start + block_size]
        s = q @ kb.T / np.sqrt(d)  # scores for this block only: (n_q, block), never (n_q, n_k)
        if causal:
            cols = np.arange(start, start + kb.shape[0])[None, :]
            s = np.where(cols <= rows, s, -np.inf)
        m_new = np.maximum(m, s.max(axis=1, keepdims=True))
        p = np.exp(s - m_new)  # block weights relative to the NEW running max
        # "Online softmax": earlier sums were computed against the old max; rescale them.
        correction = np.exp(m - m_new)
        l = l * correction + p.sum(axis=1, keepdims=True)
        acc = acc * correction + p @ vb
        m = m_new
    return acc / l  # identical to softmax(QKᵀ/√d)·V: exact, not an approximation


def moe_layer(x: np.ndarray, router_w: np.ndarray, expert_w: np.ndarray, top_k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    logits = x @ router_w  # (tokens, n_experts): how well each expert suits each token
    chosen = np.argsort(-logits, axis=1)[:, :top_k]  # the top-k experts per token
    top_logits = np.take_along_axis(logits, chosen, axis=1)
    gates = softmax(top_logits, axis=1)  # renormalise over the chosen experts only
    out = np.zeros_like(x)
    for t in range(x.shape[0]):  # only top_k experts run per token: that's the compute saving
        for slot in range(top_k):
            e = chosen[t, slot]
            out[t] += gates[t, slot] * (x[t] @ expert_w[e])
    return out, chosen


def moe_param_counts(d_model: int, d_ff: int, n_experts: int, top_k: int, n_layers: int) -> tuple[int, int]:
    per_expert = 3 * d_model * d_ff  # SwiGLU FFN: gate, up and down matrices (no biases)
    # All experts must sit in memory (any token may pick any expert), but each token only
    # computes with top_k of them.
    return n_layers * n_experts * per_expert, n_layers * top_k * per_expert
