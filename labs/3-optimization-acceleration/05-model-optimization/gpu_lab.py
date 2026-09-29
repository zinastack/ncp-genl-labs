# %% [markdown]
# # Lab 05 — GPU: measure every optimisation lever
# Section 3 instance (1× L4 recommended; T4 works, but it has no bf16 and no FlashAttention).
#
# 1. Tensor-core throughput: FP32 vs TF32 vs FP16/BF16 matmul TFLOPS
# 2. Your INT8 quantizer and 2:4 pruning on REAL model weights
# 3. Decode is memory-bound: tokens/s vs batch size, compared with your roofline bound
# 4. Activation checkpointing: memory vs time
# 5. Weight quantization (bitsandbytes INT8 / NF4): memory, speed, quality
# 6. Speculative (assisted) decoding: same output, less latency

# %%
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE, HALF = dev.device(), dev.half_dtype()
SMALL, LARGE = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
CUDA = torch.cuda.is_available()
print(dev.summary())

# Datasheet numbers for the roofline (dense tensor-core half precision, HBM/GDDR bandwidth)
SPECS = {"L4": (121e12, 300e9), "T4": (65e12, 320e9), "A10": (125e12, 600e9), "L40S": (362e12, 864e9),
         "A100": (312e12, 2.0e12), "H100": (989e12, 3.35e12)}
gpu_name = torch.cuda.get_device_name(0) if CUDA else "cpu"
peak_flops, bandwidth = next((v for k, v in SPECS.items() if k in gpu_name), (None, None))


# %% 1. Matmul throughput by precision
def tflops(dtype, n=4096, iters=20, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a, b = torch.randn(n, n, device=DEVICE, dtype=dtype), torch.randn(n, n, device=DEVICE, dtype=dtype)
    a @ b
    dev.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    dev.sync()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


if CUDA:
    rows = [("FP32", tflops(torch.float32)), ("TF32", tflops(torch.float32, tf32=True)), (str(HALF), tflops(HALF))]
    for name, v in rows:
        print(f"{name:>15}: {v:6.1f} TFLOPS")
    if peak_flops:
        print(f"datasheet dense half-precision peak for {gpu_name}: {peak_flops / 1e12:.0f} TFLOPS")
    print("Tensor cores need FP16/BF16 (or TF32/FP8). That's why mixed precision is step one.")
else:
    print("[skipped] matmul benchmark needs CUDA")

# %% 2. Your quantizer and 2:4 pruning on a real weight matrix
small = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=torch.float32)
w = small.model.layers[10].mlp.down_proj.weight.detach().numpy().astype(np.float64)
x = np.random.default_rng(0).normal(size=(16, w.shape[1]))
ref = x @ w.T
for per_channel in (False, True):
    q, s = lab.quantize_int8(w, per_channel=per_channel)
    err = np.abs(x @ lab.dequantize(q, s).T - ref).mean() / np.abs(ref).mean()
    print(f"INT8 {'per-channel' if per_channel else 'per-tensor '}: relative output error {err:.4%}")
pruned = lab.prune_2_4(w)
print(f"2:4 pruning: sparsity {np.mean(pruned == 0):.0%}, relative output error "
      f"{np.abs(x @ pruned.T - ref).mean() / np.abs(ref).mean():.1%} (needs fine-tuning to recover)")
del small

# %% 3. Decode throughput vs batch size (memory-bound regime)
tok = AutoTokenizer.from_pretrained(SMALL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=HALF).to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
NEW = 64
for bs in ((1, 4, 16, 64) if CUDA else (1, 4)):
    batch = tok(["Write a short story about a GPU."] * bs, return_tensors="pt").to(DEVICE)
    model.generate(**batch, max_new_tokens=4, min_new_tokens=4, do_sample=False)  # warm-up
    dev.sync()
    t0 = time.perf_counter()
    model.generate(**batch, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False)
    dev.sync()
    tps = bs * NEW / (time.perf_counter() - t0)
    bound = lab.decode_tokens_per_sec_bound(n_params, 2, bandwidth, bs) if bandwidth else float("nan")
    print(f"batch {bs:3d}: {tps:8.0f} tokens/s   (memory-bandwidth bound ≈ {bound:,.0f})")
print("Throughput climbs almost linearly with batch: each weight read is shared by more sequences.\n"
      "A 0.5B model at batch 1 sits far below the bound because Python/kernel-launch overhead dominates.\n"
      "That is what CUDA graphs and TensorRT-LLM remove.")

# %% 4. Activation checkpointing
if CUDA:
    ids = torch.randint(0, model.config.vocab_size, (8, 512), device=DEVICE)
    model.train()
    for ckpt in (False, True):
        if ckpt:
            model.gradient_checkpointing_enable()
        model.zero_grad()
        dev.reset_peak()
        t0 = time.perf_counter()
        model(ids, labels=ids).loss.backward()
        dev.sync()
        print(f"checkpointing={ckpt!s:5}: peak {dev.peak_gb():5.2f} GB, fwd+bwd {time.perf_counter() - t0:5.2f} s")
    model.gradient_checkpointing_disable()
    model.eval()
    print("Less memory for more time: activations are recomputed in the backward pass.")
del model
dev.free()

# %% 5. Weight quantization with bitsandbytes (CUDA only)
TEXT = ("Tensor parallelism splits individual weight matrices across GPUs, while pipeline parallelism "
        "assigns groups of layers to different devices and passes activations between them.")
tok_l = AutoTokenizer.from_pretrained(LARGE)


def evaluate(m):
    enc = tok_l(TEXT, return_tensors="pt").to(m.device)
    with torch.no_grad():
        loss = m(**enc, labels=enc.input_ids).loss.item()
    prompt = tok_l("Explain the KV cache.", return_tensors="pt").to(m.device)
    m.generate(**prompt, max_new_tokens=4)
    dev.sync()
    t0 = time.perf_counter()
    m.generate(**prompt, max_new_tokens=64, min_new_tokens=64, do_sample=False)
    dev.sync()
    return m.get_memory_footprint() / 1e9, 64 / (time.perf_counter() - t0), float(np.exp(loss))


if CUDA:
    try:
        from transformers import BitsAndBytesConfig

        configs = {
            str(HALF): dict(torch_dtype=HALF),
            "INT8 (LLM.int8)": dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True)),
            "NF4 (4-bit)": dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                      bnb_4bit_compute_dtype=HALF)),
        }
        print(f"{'weights':<18}{'GB':>7}{'tok/s':>9}{'PPL':>8}")
        for name, kwargs in configs.items():
            m = AutoModelForCausalLM.from_pretrained(LARGE, device_map={"": 0}, **kwargs).eval()
            gb, tps, ppl = evaluate(m)
            print(f"{name:<18}{gb:>7.2f}{tps:>9.1f}{ppl:>8.2f}")
            del m
            dev.free()
        print("Memory drops 2–4×. bitsandbytes kernels favour memory over speed; TensorRT-LLM "
              "INT4-AWQ/FP8 kernels also get the speed-up.")
    except ImportError:
        print("[skipped] pip install bitsandbytes")

# %% 6. Speculative decoding: 0.5B draft proposes, 1.5B target verifies
if CUDA:
    target = AutoModelForCausalLM.from_pretrained(LARGE, torch_dtype=HALF).to(DEVICE).eval()
    draft = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=HALF).to(DEVICE).eval()
    prompt = tok_l("List five facts about GPUs:\n1.", return_tensors="pt").to(DEVICE)
    timings, outputs = {}, {}
    for name, kwargs in [("target only", {}), ("assisted (draft)", {"assistant_model": draft})]:
        target.generate(**prompt, max_new_tokens=8, **kwargs)
        dev.sync()
        t0 = time.perf_counter()
        outputs[name] = target.generate(**prompt, max_new_tokens=128, do_sample=False, **kwargs)
        dev.sync()
        timings[name] = time.perf_counter() - t0
        print(f"{name:<17}: {timings[name]:.2f} s")
    same = torch.equal(outputs["target only"], outputs["assisted (draft)"])
    print(f"identical greedy output: {same}; speed-up {timings['target only'] / timings['assisted (draft)']:.2f}×")
    for alpha in (0.6, 0.8, 0.9):
        print(f"  theory: acceptance {alpha:.0%}, γ=5 → {lab.speculative_expected_tokens(alpha, 5):.2f} tokens per target pass")
