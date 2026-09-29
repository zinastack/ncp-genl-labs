"""Lab 01 — reference solutions. Try exercises.py first."""

import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)  # a row that is all -inf stays finite
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def multi_head_attention(x, w_q, w_k, w_v, w_o, n_heads, causal=True):
    seq, d_model = x.shape
    head_dim = d_model // n_heads

    def split(t):  # (seq, d_model) -> (heads, seq, head_dim)
        return t.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)

    q, k, v = split(x @ w_q), split(x @ w_k), split(x @ w_v)
    mask = causal_mask(seq) if causal else None
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    merged = out.transpose(1, 0, 2).reshape(seq, d_model)
    return merged @ w_o


def sinusoidal_positions(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / np.power(10000.0, i / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def apply_rope(x, positions, base=10000.0):
    d = x.shape[-1]
    inv_freq = base ** (-np.arange(0, d, 2) / d)          # (d/2,)
    theta = positions[:, None] * inv_freq[None, :]         # (seq, d/2)
    cos, sin = np.cos(theta), np.sin(theta)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=float)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x, gamma, eps=1e-6):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * gamma


def gpt2_param_count(vocab, n_ctx, d_model, n_layers):
    d = d_model
    embeddings = vocab * d + n_ctx * d
    per_layer = (
        2 * (2 * d)                 # ln_1, ln_2 (weight + bias)
        + (d * 3 * d + 3 * d)       # fused QKV
        + (d * d + d)               # attention output projection
        + (d * 4 * d + 4 * d)       # MLP up
        + (4 * d * d + d)           # MLP down
    )
    final_ln = 2 * d
    return embeddings + n_layers * per_layer + final_ln


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def masked_mean_pool(hidden, attention_mask):
    m = attention_mask[..., None].astype(hidden.dtype)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
