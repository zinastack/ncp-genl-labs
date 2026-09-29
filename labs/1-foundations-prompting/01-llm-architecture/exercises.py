"""Lab 01 — LLM Architecture. Fill in every TODO, then run:

    pytest labs/1-foundations-prompting/01-llm-architecture

All arrays are NumPy. Shapes are given in each docstring. Don't use torch here:
building it by hand is what makes the exam questions easy.
"""

import numpy as np


# 1 ─────────────────────────────────────────────────────────────────────────────
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    Hint: subtract the max along `axis` before exponentiating. -inf entries must become 0.
    """
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def causal_mask(n: int) -> np.ndarray:
    """Boolean (n, n) mask, True where query i MAY attend to key j (i.e. j <= i)."""
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V

    q: (..., n_q, d_k)   k: (..., n_k, d_k)   v: (..., n_k, d_v)
    mask: optional boolean array broadcastable to (..., n_q, n_k); False = blocked.
    Returns (output (..., n_q, d_v), weights (..., n_q, n_k)).
    """
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def multi_head_attention(
    x: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray, w_o: np.ndarray, n_heads: int,
    causal: bool = True,
) -> np.ndarray:
    """Self-attention with `n_heads` heads.

    x: (seq, d_model). Each weight is (d_model, d_model); project with x @ w.
    Steps: project → split into heads (n_heads, seq, head_dim) → attention per head
    → merge back to (seq, d_model) → output projection with w_o.
    """
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    """(seq_len, d_model) with PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(same)."""
    raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotary position embedding.

    x: (seq, d) with d even. positions: (seq,) integer positions.
    For each pair (x[2i], x[2i+1]) rotate by angle  positions * base^(-2i/d):
        out[2i]   = x[2i] cos θ - x[2i+1] sin θ
        out[2i+1] = x[2i] sin θ + x[2i+1] cos θ
    """
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalise over the last axis: (x - mean) / sqrt(var + eps) * gamma + beta."""
    raise NotImplementedError


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """x / sqrt(mean(x²) + eps) * gamma. No mean subtraction, no bias (Llama-style)."""
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def gpt2_param_count(vocab: int, n_ctx: int, d_model: int, n_layers: int) -> int:
    """Exact parameter count of a GPT-2-style model with a tied LM head.

    Include: token embeddings, learned position embeddings, and per layer:
    two LayerNorms (weight+bias), fused QKV projection (+bias), attention output
    projection (+bias), MLP up (d→4d, +bias) and down (4d→d, +bias).
    Plus a final LayerNorm. The LM head is tied, so it adds nothing.
    GPT-2 small (50257, 1024, 768, 12) should give 124,439,808.
    """
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
def kv_cache_bytes(
    n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int,
    bytes_per_elem: int = 2,
) -> int:
    """Bytes needed to cache K and V for every layer, token and sequence in the batch."""
    raise NotImplementedError


# 10 ────────────────────────────────────────────────────────────────────────────
def masked_mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Sentence embeddings by averaging token vectors, ignoring padding.

    hidden: (batch, seq, dim)   attention_mask: (batch, seq) of 0/1   → (batch, dim)
    """
    raise NotImplementedError
