# %% [markdown]
# # Lab 04 — GPU: full fine-tuning vs LoRA vs QLoRA on a real model
# Section 2 instance (1× L4 24 GB recommended; T4 16 GB works with the defaults).
#
# 1. Measure peak memory of one training step: full FT vs LoRA vs QLoRA (4-bit, bitsandbytes)
#    and compare with your Lab 05 memory calculator's prediction
# 2. LoRA SFT with prompt-masked labels (your build_sft_example) using PEFT
# 3. Check the style was learned, then merge_and_unload() for zero-overhead serving

# %%
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE, HALF = dev.device(), dev.half_dtype()
MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(MODEL)
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
print(dev.summary(), "\nmodel:", MODEL)

# %% 1. One optimizer step, three ways
batch = tok(["Explain what the KV cache stores during decoding. " * 20] * 4, return_tensors="pt",
            truncation=True, max_length=256)
batch = {k: v.to(DEVICE) for k, v in batch.items()}


def step_memory(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    dev.reset_peak()
    t0 = time.perf_counter()
    model(**batch, labels=batch["input_ids"]).loss.backward()
    opt.step()
    dev.sync()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dev.peak_gb(), time.perf_counter() - t0, trainable


results = {}
full = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEVICE)  # fp32 master weights
n_params = sum(p.numel() for p in full.parameters())
results["full FT (fp32 Adam)"] = step_memory(full)
del full
dev.free()

lora = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=HALF).to(DEVICE), LORA)
results["LoRA (half base)"] = step_memory(lora)
del lora
dev.free()

if torch.cuda.is_available():
    try:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=HALF)
        q = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
        q = get_peft_model(prepare_model_for_kbit_training(q, use_gradient_checkpointing=False), LORA)
        results["QLoRA (4-bit NF4 base)"] = step_memory(q)
        del q
    except ImportError:
        print("[skipped QLoRA] pip install bitsandbytes")

print(f"\n{'method':<24}{'peak GB':>9}{'step s':>9}{'trainable':>14}")
for name, (gb, secs, trainable) in results.items():
    print(f"{name:<24}{gb:>9.2f}{secs:>9.2f}{trainable:>14,}")
mem = load(HERE.parents[1] / "3-optimization-acceleration" / "05-model-optimization")
lora_trainable = results["LoRA (half base)"][2]
print("\npredictions (weights + grads + Adam states, activations excluded, so measured is higher):")
print(f"  full FT, fp32 Adam ≈ {n_params * 16 / 1e9:.2f} GB (4 B weights + 4 B grads + 8 B Adam)")
print(f"  LoRA (Lab 05 calc) ≈ {mem.training_memory_bytes(n_params, lora_trainable) / 1e9:.2f} GB")

# %% 2. LoRA SFT on a toy "house style" with prompt-masked labels
DATA = [
    ("What does NCCL do?", "ANSWER: Collective communication (all-reduce, all-gather) between GPUs."),
    ("What is Triton dynamic batching?", "ANSWER: Server-side grouping of requests into batches within a queue delay."),
    ("What is tensor parallelism?", "ANSWER: Splitting each weight matrix across GPUs, usually within a node."),
    ("What is the KV cache?", "ANSWER: Stored keys/values of past tokens so decoding doesn't recompute them."),
    ("What is LoRA?", "ANSWER: A frozen weight plus a trainable low-rank update B·A scaled by alpha/r."),
    ("What is FSDP?", "ANSWER: Sharding parameters, gradients and optimizer states across data-parallel ranks."),
]


def encode(q, a):
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(a, add_special_tokens=False).input_ids
    eos = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id
    ids, labels = lab.build_sft_example(p_ids, r_ids, eos)          # YOUR collator
    return torch.tensor([ids], device=DEVICE), torch.tensor([labels], device=DEVICE)


model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=HALF).to(DEVICE), LORA)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
for epoch in range(4):
    total = 0.0
    for q, a in DATA:
        ids, labels = encode(q, a)
        loss = lab.causal_lm_loss(model(input_ids=ids).logits.float(), labels)   # YOUR shifted loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        total += loss.item()
    print(f"epoch {epoch}: mean loss {total / len(DATA):.3f}")


# %% 3. Did it learn the style on an unseen question? Then merge.
@torch.no_grad()
def ask(m, question):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = m.generate(**ids, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


model.eval()
question = "What is pipeline parallelism?"
print("LoRA   :", ask(model, question))
with model.disable_adapter():
    print("base   :", ask(model, question))
merged = model.merge_and_unload()
print("merged :", ask(merged, question), "  ← identical to LoRA output, with no adapter overhead")
