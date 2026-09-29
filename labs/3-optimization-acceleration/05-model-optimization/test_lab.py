import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

GB = 1e9


def test_1_training_memory(lab):
    assert lab.training_memory_bytes(7e9) / GB == pytest.approx(112), "7B full FT = 16 B/param"
    lora = lab.training_memory_bytes(7e9, trainable_params=20e6)
    assert lora / GB == pytest.approx(14.28), "frozen bf16 weights + 14 B per LoRA param"
    qlora = lab.training_memory_bytes(70e9, trainable_params=100e6, weight_bytes=0.5)
    assert qlora / GB == pytest.approx(36.4)


def test_2_max_batch(lab):
    # Llama-2-7B fp16 on an 80 GB GPU with 4k context: 13.5 GB weights, 2 GiB KV per sequence
    kv = 2 * 32 * 32 * 128 * 4096 * 2
    assert lab.max_batch_size(80e9, 13.5e9, kv) == 27
    assert lab.max_batch_size(16e9, 14e9, kv) == 0


def test_3_quantization(lab):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 64))
    w[0] *= 50  # one outlier output channel
    q, s = lab.quantize_int8(w)
    assert q.dtype == np.int8 and np.abs(q).max() == 127 and np.ndim(s) == 0
    qc, sc = lab.quantize_int8(w, per_channel=True)
    assert sc.shape == (8, 1)
    err_tensor = np.abs(lab.dequantize(q, s) - w)[1:].mean()
    err_channel = np.abs(lab.dequantize(qc, sc) - w)[1:].mean()
    assert err_channel < err_tensor / 10, "per-channel scales isolate the outlier channel"


def test_4_loss_scaler(lab):
    s = lab.DynamicLossScaler(init_scale=1024, growth_interval=3)
    assert s.update(found_inf=True) is False and s.scale == 512
    assert all(s.update(False) for _ in range(3)) and s.scale == 1024
    s.update(False)
    s.update(False)
    s.update(True)
    assert s.scale == 512
    s.update(False)
    s.update(False)
    assert s.scale == 512, "the good-step counter resets after an overflow"


def test_5_grad_accumulation(lab):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 2))
    x, y = torch.randn(32, 6), torch.randn(32, 2)
    accumulated = lab.accumulated_gradients(model, x, y, micro_batch=8)
    model.zero_grad()
    F.mse_loss(model(x), y).backward()
    for a, p in zip(accumulated, model.parameters()):
        torch.testing.assert_close(a, p.grad)


def test_6_distillation(lab):
    torch.manual_seed(0)
    s, t, y = torch.randn(4, 10), torch.randn(4, 10), torch.randint(0, 10, (4,))
    T = 3.0
    kl = (F.softmax(t / T, -1) * (F.log_softmax(t / T, -1) - F.log_softmax(s / T, -1))).sum(-1).mean()
    expected = 0.7 * T * T * kl + 0.3 * F.cross_entropy(s, y)
    torch.testing.assert_close(lab.distillation_loss(s, t, y, T, 0.7), expected)
    assert lab.distillation_loss(t, t, y, T, 1.0).abs() < 1e-6, "identical logits → zero KD loss"


def test_7_prune_2_4(lab):
    w = np.array([[0.1, -0.9, 0.3, 0.05, 2.0, -3.0, 0.0, 1.0]])
    np.testing.assert_array_equal(lab.prune_2_4(w), [[0, -0.9, 0.3, 0, 2.0, -3.0, 0, 0]])
    big = np.random.default_rng(1).normal(size=(16, 32))
    assert (lab.prune_2_4(big) == 0).mean() == 0.5


def test_8_batching(lab):
    lengths = [10, 200, 12, 15, 180, 9, 11, 14]
    assert lab.static_batching_steps(lengths, 4) == 200 + 180
    assert lab.inflight_batching_steps(lengths, 4) == 200
    assert lab.inflight_batching_steps([5, 5, 5], 1) == 15


def test_9_speculative(lab):
    assert lab.speculative_expected_tokens(0.0, 4) == pytest.approx(1.0), "all rejected → still 1 token (the target's own)"
    assert lab.speculative_expected_tokens(1.0, 4) == 5
    assert lab.speculative_expected_tokens(0.8, 4) == pytest.approx(3.3616)


def test_10_decode_roofline(lab):
    h100_bw = 3.35e12
    assert lab.decode_tokens_per_sec_bound(70e9, 2, h100_bw) == pytest.approx(23.9, abs=0.1)
    int4 = lab.decode_tokens_per_sec_bound(70e9, 0.5, h100_bw)
    assert int4 == pytest.approx(4 * 23.93, rel=0.01), "4-bit weights → ~4x faster memory-bound decode"
    assert lab.decode_tokens_per_sec_bound(70e9, 2, h100_bw, batch=16) == pytest.approx(16 * 23.93, rel=0.01)


def test_11_activation_checkpointing(lab):
    # Equal per-layer input and activation cost: the classic √n rule.
    costs = {seg: lab.checkpointed_activation_bytes(64, 1, 1, seg) for seg in (1, 2, 4, 8, 16, 32, 64)}
    assert min(costs, key=costs.get) == 8 and costs[8] == 16, "√64 = 8 segments of 8 layers"
    # Realistic: a layer's internal activations ≈ 17× its input.
    assert lab.checkpointed_activation_bytes(64, 17, 1, 2) == 66
    assert 64 * 17 / lab.checkpointed_activation_bytes(64, 17, 1, 2) > 16, "~16× less activation memory"
    assert lab.checkpointed_activation_bytes(10, 5, 1, 4) == 3 * 1 + 4 * 5, "ceil for a partial last segment"


def test_12_paged_kv_cache(lab):
    c = lab.PagedKVCache(num_blocks=8, block_size=16)
    for _ in range(20):
        c.append_token("a")
    for _ in range(5):
        c.append_token("b")
    assert len(c.block_tables["a"]) == 2 and len(c.block_tables["b"]) == 1
    assert len(c.free_blocks) == 5
    assert c.wasted_slots() == (32 - 20) + (16 - 5), "only the tail of each last block is wasted"
    c.free("a")
    assert len(c.free_blocks) == 7 and "a" not in c.block_tables
    small = lab.PagedKVCache(num_blocks=1, block_size=4)
    for _ in range(4):
        small.append_token("x")
    with pytest.raises(MemoryError):
        small.append_token("x")


def test_13_smoothquant(lab):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 16))
    x[:, 3] *= 60  # one outlier activation channel (typical of LLMs)
    w = rng.normal(size=(16, 8)) * 0.05
    s = lab.smoothquant_scales(np.abs(x).max(0), np.abs(w).max(1))
    xs, ws = lab.smooth(x, w, s)
    np.testing.assert_allclose(xs @ ws, x @ w, atol=1e-10)  # mathematically identical

    def q8(t, axis=None):
        scale = np.abs(t).max(axis=axis, keepdims=axis is not None) / 127
        return np.round(t / scale) * scale

    naive = np.abs(q8(x) @ q8(w, axis=0) - x @ w).mean()
    smoothed = np.abs(q8(xs) @ q8(ws, axis=0) - x @ w).mean()
    assert smoothed < naive / 2, "W8A8 error drops once the activation outlier is smoothed"


def test_14_latency_breakdown(lab):
    r = lab.latency_breakdown(12_000, 300, prefill_tokens_per_s=20_000, decode_tokens_per_s=40)
    assert r["ttft"] == pytest.approx(0.6) and r["tpot"] == pytest.approx(0.025)
    assert r["e2e"] == pytest.approx(0.6 + 299 * 0.025)
    cached = lab.latency_breakdown(12_000, 300, 20_000, 40, cached_prefix_tokens=10_000)
    assert cached["ttft"] == pytest.approx(0.1), "prefix caching cuts TTFT, not TPOT"
    assert cached["tpot"] == r["tpot"]


def test_15_float_formats(lab):
    fp16, bf16 = lab.float_format(5, 10), lab.float_format(8, 7)
    assert fp16["max"] == 65504.0 and fp16["min_normal"] == pytest.approx(6.1035e-05, rel=1e-4)
    assert bf16["max"] == pytest.approx(3.39e38, rel=1e-2), "bf16 has FP32's range"
    assert bf16["epsilon"] > fp16["epsilon"], "…but less precision than fp16"
    assert lab.float_format(5, 2)["max"] == 57344.0, "FP8 E5M2"
    assert fp16["max"] == float(np.finfo(np.float16).max)
