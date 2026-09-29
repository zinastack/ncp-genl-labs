import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class TinyBlock(nn.Module):
    """Attention-shaped names so target_modules work like they do for Llama."""

    def __init__(self, d=32, n_classes=4):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential()
        self.mlp.up_proj = nn.Linear(d, 2 * d)
        self.mlp.down_proj = nn.Linear(2 * d, d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = torch.tanh(self.q_proj(x)) + torch.tanh(self.k_proj(x)) * torch.tanh(self.v_proj(x))
        h = h + self.mlp.down_proj(F.gelu(self.mlp.up_proj(h)))
        return self.head(h)


def test_1_lora_linear(lab):
    base = nn.Linear(16, 8)
    layer = lab.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    assert layer.lora_A.shape == (4, 16) and layer.lora_B.shape == (8, 4)
    assert layer.scaling == 2.0
    torch.testing.assert_close(layer(x), base(x)), "B is zero-initialised: step 0 == base model"
    assert not base.weight.requires_grad and layer.lora_A.requires_grad

    with torch.no_grad():
        layer.lora_B.normal_()
    merged = layer.merge()
    assert type(merged) is nn.Linear
    torch.testing.assert_close(merged(x), layer(x), atol=1e-5, rtol=1e-5)


def test_2_3_apply_lora_and_count(lab):
    model = TinyBlock()
    total_before = sum(p.numel() for p in model.parameters())
    lab.apply_lora(model, ["q_proj", "v_proj", "down_proj"], r=4)
    assert isinstance(model.q_proj, lab.LoRALinear) and isinstance(model.mlp.down_proj, lab.LoRALinear)
    assert isinstance(model.k_proj, nn.Linear) and not isinstance(model.k_proj, lab.LoRALinear)
    trainable, total = lab.count_parameters(model)
    expected = 4 * (32 + 32) * 2 + 4 * (64 + 32)
    assert trainable == expected
    assert total == total_before + expected
    assert not model.head.weight.requires_grad, "everything except LoRA must be frozen"


def test_4_sft_example(lab):
    ids, labels = lab.build_sft_example([5, 6, 7], [8, 9], eos_id=2)
    assert ids == [5, 6, 7, 8, 9, 2]
    assert labels == [-100, -100, -100, 8, 9, 2]


def test_5_causal_lm_loss(lab):
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    labels[0, :3] = -100
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(lab.causal_lm_loss(logits, labels), expected)


def test_6_sequence_logprob(lab):
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 0]])
    lp = logits.log_softmax(-1)
    expected = torch.stack([lp[0, 0, 1] + lp[0, 1, 2] + lp[0, 2, 3], lp[1, 1, 4] + lp[1, 2, 0]])
    torch.testing.assert_close(lab.sequence_logprob(logits, labels), expected)


def test_7_dpo(lab):
    z = torch.zeros(3)
    torch.testing.assert_close(lab.dpo_loss(z, z, z, z), torch.tensor(torch.log(torch.tensor(2.0)).item()))
    better = lab.dpo_loss(torch.tensor([-1.0]), torch.tensor([-5.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
    worse = lab.dpo_loss(torch.tensor([-5.0]), torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
    assert better < 0.693 < worse
    torch.testing.assert_close(better, -F.logsigmoid(torch.tensor(0.5 * 4.0)))


def test_8_lora_finetune_keeps_base_frozen(lab):
    torch.manual_seed(1)
    x = torch.randn(256, 32)
    y_a = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()  # "pre-training" task
    y_b = (x[:, 2] > 0).long() + 2 * (x[:, 3] > 0).long()  # new downstream task

    model = TinyBlock()
    lab.train(model, x, y_a, steps=150, lr=1e-2)
    frozen = copy.deepcopy(model.state_dict())

    lab.apply_lora(model, ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "head"], r=8)
    losses = lab.train(model, x, y_b, steps=200, lr=1e-2)
    assert losses[-1] < 0.5 * losses[0], "LoRA should learn the new task"

    after = model.state_dict()
    for name, value in frozen.items():
        key = name.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
        torch.testing.assert_close(after.get(key, after.get(name)), value)


def test_9_nf4(lab):
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    codes, absmax = lab.nf4_quantize(w)
    assert codes.dtype == torch.uint8 and codes.max() <= 15 and absmax.shape == (256 * 256 // 64, 1)
    w_hat = lab.nf4_dequantize(codes, absmax, w.shape)
    assert w_hat.shape == w.shape
    err_nf4 = (w_hat - w).abs().mean()
    # Uniform 4-bit on the same blocks: NF4 levels suit bell-shaped weights better.
    b = w.reshape(-1, 64)
    am = b.abs().amax(1, keepdim=True)
    levels = torch.linspace(-1, 1, 16)
    uniform = (levels[((b / am).unsqueeze(-1) - levels).abs().argmin(-1)] * am).reshape(w.shape)
    assert err_nf4 < (uniform - w).abs().mean(), "NF4 beats uniform 4-bit on normally distributed weights"
    assert torch.all(w_hat.reshape(-1, 64).abs().amax(1, keepdim=True) == absmax), "each block's max is exact"


def test_10_soft_prompt(lab):
    sp = lab.SoftPrompt(n_virtual=5, d_model=16)
    x = torch.randn(3, 7, 16)
    out = sp(x)
    assert out.shape == (3, 12, 16)
    torch.testing.assert_close(out[:, 5:], x)
    torch.testing.assert_close(out[0, :5], out[2, :5])  # the same prompt for every sequence
    assert sum(p.numel() for p in sp.parameters()) == 5 * 16, "only n_virtual × d parameters"
    frozen = nn.Linear(16, 2)
    for p in frozen.parameters():
        p.requires_grad = False
    frozen(sp(x)).sum().backward()
    assert sp.prompt.grad is not None and frozen.weight.grad is None, "gradients flow into the soft prompt only"


def test_11_reward_model(lab):
    rc, rr = torch.tensor([2.0, 0.5, 1.0]), torch.tensor([0.0, 1.0, 1.0])
    expected = -(F.logsigmoid(torch.tensor(2.0)) + F.logsigmoid(torch.tensor(-0.5)) + F.logsigmoid(torch.tensor(0.0))) / 3
    torch.testing.assert_close(lab.reward_model_loss(rc, rr), expected)
    assert lab.preference_accuracy(rc, rr) == pytest.approx(1 / 3), "ties don't count as correct"


def test_12_multi_lora(lab):
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    adapters = {name: (torch.randn(2, 8), torch.randn(4, 2), 0.5) for name in ("acme", "globex")}
    layer = lab.MultiLoRALinear(base, adapters)
    x = torch.randn(3, 8)
    with torch.no_grad():
        out = layer(x, ["acme", None, "globex"])
    for i, name in enumerate(["acme", None, "globex"]):
        single = lab.LoRALinear(base, r=2, alpha=1.0)
        if name:
            with torch.no_grad():
                single.lora_A.copy_(adapters[name][0])
                single.lora_B.copy_(adapters[name][1])
        else:
            single.lora_B.data.zero_()
        with torch.no_grad():
            torch.testing.assert_close(out[i], single(x[i : i + 1])[0])


def test_13_lr_schedule(lab):
    lr = lambda t: lab.lr_at_step(t, 2e-4, 100, 1000, 2e-5)
    assert lr(0) == pytest.approx(2e-6) and lr(49) == pytest.approx(1e-4) and lr(99) == pytest.approx(2e-4)
    assert lr(100) == pytest.approx(2e-4), "peak right after warm-up"
    assert lr(550) == pytest.approx(1.1e-4), "halfway through the decay"
    assert lr(1000) == pytest.approx(2e-5) and lr(5000) == pytest.approx(2e-5), "floor at min_lr"
    assert all(lr(t) >= lr(t + 1) for t in range(100, 999)), "monotone decay after warm-up"
