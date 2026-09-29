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
