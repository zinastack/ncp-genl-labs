# %% [markdown]
# # Lab 01 — GPU: architecture on real models
# Runs on the Section 1 instance (1× L4 or T4). Also runs on a Mac/CPU with the CUDA-only
# cells skipped. Execute as a script (`make gpu-01`) or cell-by-cell (`# %%`) in VS Code/Jupyter.
#
# 1. BERT outputs: pooler_output vs [CLS] vs masked mean pooling
# 2. GPT-2: your exact parameter count vs the real model; weight tying
# 3. KV cache: measure a real GQA model's cache and check your kv_cache_bytes formula
# 4. Attention kernels: math vs memory-efficient vs FlashAttention (time + memory vs seq len)

# %%
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE, HALF = dev.device(), dev.half_dtype()
print(dev.summary())

# %% 1. BERT: three ways to get a sentence vector
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
batch = tok(["GPUs accelerate training.", "Distributed data parallel replicates the model on every GPU."],
            padding=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = bert(**batch)

cls = out.last_hidden_state[:, 0]
mean = torch.from_numpy(lab.masked_mean_pool(out.last_hidden_state.cpu().numpy(), batch["attention_mask"].cpu().numpy()))
print("last_hidden_state", tuple(out.last_hidden_state.shape), "| pooler_output", tuple(out.pooler_output.shape))
print("pooler_output == tanh(dense([CLS]))?",
      torch.allclose(out.pooler_output, torch.tanh(bert.pooler.dense(cls)), atol=1e-4))
print("cosine(sentence1, sentence2): pooler=%.3f  mean-pool=%.3f" % (
    F.cosine_similarity(out.pooler_output[0], out.pooler_output[1], dim=0),
    F.cosine_similarity(mean[0], mean[1], dim=0)))

# %% 2. GPT-2: parameter count and weight tying
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
real = sum(p.numel() for p in gpt2.parameters())  # .parameters() de-duplicates tied weights
c = gpt2.config
print(f"real: {real:,}   yours: {lab.gpt2_param_count(c.vocab_size, c.n_positions, c.n_embd, c.n_layer):,}")
print("LM head tied to token embeddings:", gpt2.lm_head.weight.data_ptr() == gpt2.transformer.wte.weight.data_ptr())
del gpt2


# %% 3. KV cache of a GQA model (Qwen2.5-0.5B: 24 layers, 14 query heads but only 2 KV heads)
def cache_bytes(cache) -> int:
    if hasattr(cache, "layers"):                       # transformers >= 4.56
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache"):
        pairs = zip(cache.key_cache, cache.value_cache)
    else:                                              # legacy tuple of tuples
        pairs = cache
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pairs)


name = "Qwen/Qwen2.5-0.5B-Instruct"
qtok = AutoTokenizer.from_pretrained(name)
qwen = AutoModelForCausalLM.from_pretrained(name, torch_dtype=HALF).to(DEVICE).eval()
cfg = qwen.config
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(f"{name}: layers={cfg.num_hidden_layers} q_heads={cfg.num_attention_heads} "
      f"kv_heads={cfg.num_key_value_heads} head_dim={head_dim}")

for seq_len in (512, 2048):
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=DEVICE)
    with torch.no_grad():
        cache = qwen(ids, use_cache=True).past_key_values
    measured = cache_bytes(cache)
    predicted = lab.kv_cache_bytes(cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim, seq_len, 1,
                                   torch.finfo(HALF).bits // 8)
    mha = lab.kv_cache_bytes(cfg.num_hidden_layers, cfg.num_attention_heads, head_dim, seq_len, 1,
                             torch.finfo(HALF).bits // 8)
    print(f"seq={seq_len:5d}  measured={measured / 2**20:6.1f} MiB  yours={predicted / 2**20:6.1f} MiB  "
          f"(without GQA it would be {mha / 2**20:6.1f} MiB)")
del qwen, cache

# %% 4. Attention kernels (CUDA only): math vs memory-efficient vs FlashAttention
if torch.cuda.is_available():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {"math": SDPBackend.MATH, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION}
    print(f"\n{'seq':>6} " + " ".join(f"{b:>22}" for b in backends))
    for seq in (1024, 2048, 4096, 8192):
        q, k, v = (torch.randn(4, 16, seq, 64, device="cuda", dtype=HALF) for _ in range(3))
        row = []
        for label, backend in backends.items():
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)  # warm-up
                    dev.reset_peak()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    dev.sync()
                row.append(f"{(time.perf_counter() - t0) * 100:7.2f} ms {dev.peak_gb():6.2f} GB")
            except RuntimeError as e:
                row.append(f"{'n/a (' + ('OOM' if 'memory' in str(e) else 'unsupported') + ')':>22}")
                torch.cuda.empty_cache()
        print(f"{seq:>6} " + " ".join(f"{r:>22}" for r in row))
    print("math materialises the n×n score matrix (memory grows ~n²); flash/mem-efficient tile it in SRAM.\n"
          "FlashAttention needs sm_80+ (A100/L4/H100); on a T4 it shows as unsupported.")
else:
    print("\n[skipped] attention kernel benchmark needs an NVIDIA GPU")
