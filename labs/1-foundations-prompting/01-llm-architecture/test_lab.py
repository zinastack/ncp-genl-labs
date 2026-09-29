import numpy as np
import pytest
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)


def test_1_softmax(lab):
    x = rng.normal(size=(3, 5))
    out = lab.softmax(x)
    np.testing.assert_allclose(out, torch.softmax(torch.tensor(x), -1).numpy(), atol=1e-12)
    big = np.array([1000.0, 1001.0, 1002.0])
    assert np.all(np.isfinite(lab.softmax(big))), "overflow: subtract the max first"
    np.testing.assert_allclose(lab.softmax(np.array([0.0, -np.inf])), [1.0, 0.0])


def test_2_causal_mask(lab):
    m = lab.causal_mask(4)
    assert m.dtype == bool and m.shape == (4, 4)
    assert m[3, 0] and m[2, 2] and not m[0, 1], "True means 'may attend': lower triangle incl. diagonal"


def test_3_attention_matches_torch(lab):
    q, k, v = (rng.normal(size=(2, 6, 8)) for _ in range(3))
    mask = lab.causal_mask(6) if hasattr(lab, "causal_mask") else np.tril(np.ones((6, 6), bool))
    out, w = lab.scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(*(torch.tensor(t) for t in (q, k, v)), is_causal=True)
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-10)
    np.testing.assert_allclose(w.sum(-1), 1.0)
    assert np.all(w[..., 0, 1:] == 0), "first token must only see itself"


def test_4_multi_head_attention(lab):
    seq, d, h = 5, 8, 2
    x = rng.normal(size=(seq, d))
    ws = [rng.normal(size=(d, d)) for _ in range(4)]
    out = lab.multi_head_attention(x, *ws, n_heads=h, causal=True)
    assert out.shape == (seq, d)

    # Reference via torch: project, reshape to heads, SDPA, merge, project.
    xt, (wq, wk, wv, wo) = torch.tensor(x), (torch.tensor(w) for w in ws)
    split = lambda t: t.view(seq, h, d // h).transpose(0, 1)
    o = F.scaled_dot_product_attention(split(xt @ wq), split(xt @ wk), split(xt @ wv), is_causal=True)
    ref = o.transpose(0, 1).reshape(seq, d) @ wo
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-10)

    # Causality: changing the last token must not change earlier outputs.
    x2 = x.copy()
    x2[-1] += 10
    out2 = lab.multi_head_attention(x2, *ws, n_heads=h, causal=True)
    np.testing.assert_allclose(out[:-1], out2[:-1], atol=1e-10)


def test_5_sinusoidal_positions(lab):
    pe = lab.sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16)
    np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-12)
    np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-12)
    np.testing.assert_allclose(pe[7, 2], np.sin(7 / 10000 ** (2 / 16)))


def test_6_rope_relative_property(lab):
    d = 16
    q, k = rng.normal(size=(1, d)), rng.normal(size=(1, d))
    rot = lambda v, p: lab.apply_rope(v, np.array([p]))
    # <RoPE(q, m), RoPE(k, n)> depends only on m - n.
    s1 = (rot(q, 10) @ rot(k, 7).T).item()
    s2 = (rot(q, 103) @ rot(k, 100).T).item()
    np.testing.assert_allclose(s1, s2, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(rot(q, 42)), np.linalg.norm(q)), "rotation preserves norm"
    np.testing.assert_allclose(rot(q, 0), q), "position 0 is the identity"


def test_7_norms(lab):
    x = rng.normal(size=(4, 10)) * 3 + 2
    g, b = rng.normal(size=10), rng.normal(size=10)
    ref = F.layer_norm(torch.tensor(x), (10,), torch.tensor(g), torch.tensor(b)).numpy()
    np.testing.assert_allclose(lab.layer_norm(x, g, b), ref, atol=1e-6)
    ref_rms = F.rms_norm(torch.tensor(x), (10,), torch.tensor(g), eps=1e-6).numpy()
    np.testing.assert_allclose(lab.rms_norm(x, g), ref_rms, atol=1e-6)


def test_8_gpt2_params(lab):
    assert lab.gpt2_param_count(50257, 1024, 768, 12) == 124_439_808
    # GPT-2 XL (1.5B)
    assert lab.gpt2_param_count(50257, 1024, 1600, 48) == 1_557_611_200


def test_9_kv_cache(lab):
    llama2_7b = lab.kv_cache_bytes(32, 32, 128, 4096, 1, 2)
    assert llama2_7b == 2 * 1024**3, "Llama-2-7B, 4k ctx, fp16 → exactly 2 GiB"
    gqa = lab.kv_cache_bytes(32, 8, 128, 4096, 1, 2)
    assert llama2_7b / gqa == 4, "8 KV heads instead of 32 → 4x smaller cache"


def test_10_masked_mean_pool(lab):
    hidden = rng.normal(size=(2, 4, 3))
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    out = lab.masked_mean_pool(hidden, mask)
    np.testing.assert_allclose(out[0], hidden[0, :2].mean(0))
    np.testing.assert_allclose(out[1], hidden[1].mean(0))
