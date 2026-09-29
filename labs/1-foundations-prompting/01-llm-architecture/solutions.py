"""Lab 01 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import numpy as np


def softmax(x, axis=-1):
    # softmax(x) == softmax(x - c) for any constant c (exp(c) cancels top and bottom).
    # Subtracting the row max makes the largest exponent exp(0) = 1, so exp() can't overflow.
    x_max = np.max(x, axis=axis, keepdims=True)  # keepdims → shape (..., 1) broadcasts per row
    # A fully-masked row (all -inf) has max -inf, and -inf - (-inf) = nan. Use 0 instead.
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)  # masked entries: exp(-inf) = 0 → exactly 0% weight
    total = np.sum(e, axis=axis, keepdims=True)
    # A fully-masked row sums to 0; return zeros ("attend to nothing") instead of 0/0 = nan.
    return e / np.where(total == 0, 1.0, total)


def causal_mask(n):
    # Lower triangle incl. diagonal: query i may see keys 0..i, never the future.
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(q, k, v, mask=None):
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


def multi_head_attention(x, w_q, w_k, w_v, w_o, n_heads, causal=True):
    seq, d_model = x.shape
    head_dim = d_model // n_heads  # the width is split across heads, not multiplied

    def split(t):  # (seq, d_model) -> (seq, heads, head_dim) -> (heads, seq, head_dim)
        return t.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)

    q, k, v = split(x @ w_q), split(x @ w_k), split(x @ w_v)
    mask = causal_mask(seq) if causal else None
    out, _ = scaled_dot_product_attention(q, k, v, mask)  # runs for all heads at once (batched)
    merged = out.transpose(1, 0, 2).reshape(seq, d_model)  # concatenate the heads back
    return merged @ w_o  # output projection mixes information across heads


def sinusoidal_positions(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]  # (seq, 1)
    i = np.arange(0, d_model, 2)[None, :]  # even dims 2i → (1, d/2)
    angles = pos / np.power(10000.0, i / d_model)  # (seq, d/2): fast waves in low dims, slow in high
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)  # even columns
    pe[:, 1::2] = np.cos(angles)  # odd columns
    return pe


def apply_rope(x, positions, base=10000.0):
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


def layer_norm(x, gamma, beta, eps=1e-5):
    # Per token (last axis): centre to mean 0, scale to variance 1, then learned scale/shift.
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x, gamma, eps=1e-6):
    # Like LayerNorm without centring or bias: divide by the root-mean-square (Llama).
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * gamma


def gpt2_param_count(vocab, n_ctx, d_model, n_layers):
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


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    # 2 = one K and one V tensor per layer. GQA shrinks this by using fewer KV heads.
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def masked_mean_pool(hidden, attention_mask):
    m = attention_mask[..., None].astype(hidden.dtype)  # (batch, seq, 1) broadcasts over features
    # Zero out padding, sum the real tokens, divide by how many real tokens there are.
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
