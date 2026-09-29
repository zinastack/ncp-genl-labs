# %% [markdown]
# # Lab 09 — GPU: evaluate real models with YOUR metrics
# Section 5 instance (1× L4/T4). Benchmarks with lm-evaluation-harness run separately: `make lm-eval`.
#
# 1. Perplexity: model size and in-domain vs scrambled text
# 2. RAG retrieval: bi-encoder vs cross-encoder reranking (recall@k, MRR, nDCG)
# 3. Model A vs B on short-answer QA: EM / F1 + paired bootstrap CI
# 4. LLM-as-judge: agreement with human labels (Cohen's κ) and position bias

# %%
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE, HALF = dev.device(), dev.half_dtype()
MODELS = {"A (0.5B)": "Qwen/Qwen2.5-0.5B-Instruct", "B (1.5B)": "Qwen/Qwen2.5-1.5B-Instruct"}
print(dev.summary())
loaded = {k: (AutoTokenizer.from_pretrained(v), AutoModelForCausalLM.from_pretrained(v, torch_dtype=HALF).to(DEVICE).eval())
          for k, v in MODELS.items()}

# %% 1. Perplexity from token log-probs
TEXT = ("Data parallelism replicates the model on every GPU and averages gradients with an all-reduce, "
        "while tensor parallelism splits each weight matrix across GPUs connected by NVLink.")
words = TEXT.split()
random.seed(0)
SCRAMBLED = " ".join(random.sample(words, len(words)))


@torch.no_grad()
def token_logprobs(tok, model, text):
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    logp = model(ids).logits[0, :-1].float().log_softmax(-1)
    return logp.gather(-1, ids[0, 1:, None]).squeeze(-1).tolist()


for name, (tok, model) in loaded.items():
    print(f"{name}: PPL real={lab.perplexity(token_logprobs(tok, model, TEXT)):6.1f}  "
          f"scrambled={lab.perplexity(token_logprobs(tok, model, SCRAMBLED)):8.1f}")
print("Bigger model → lower PPL. Same tokenizer, so the comparison is valid.")

# %% 2. Retrieval: bi-encoder recall vs cross-encoder reranking precision
PASSAGES = {
    "p1": "Triton Inference Server exposes Prometheus metrics on port 8002.",
    "p2": "Dynamic batching in Triton groups requests within max_queue_delay_microseconds.",
    "p3": "NCCL implements all-reduce, all-gather and reduce-scatter collectives for GPUs.",
    "p4": "FSDP shards parameters, gradients and optimizer states across data-parallel ranks.",
    "p5": "LoRA freezes the base weights and trains a low-rank update B times A.",
    "p6": "QLoRA quantizes the frozen base model to 4-bit NF4 and trains LoRA adapters.",
    "p7": "The KV cache stores keys and values of previous tokens to avoid recomputation.",
    "p8": "PagedAttention allocates the KV cache in fixed-size blocks to reduce fragmentation.",
    "p9": "NeMo Guardrails adds input, dialog, retrieval, execution and output rails.",
    "p10": "Speculative decoding uses a draft model whose tokens the target model verifies.",
    "p11": "Tensor parallelism should stay within a node because it communicates every layer.",
    "p12": "Pipeline parallelism splits layers into stages and suffers from a pipeline bubble.",
    "p13": "Triton model repositories contain a config.pbtxt and numbered version directories.",
    "p14": "DCGM exporter publishes GPU utilization and memory metrics for Prometheus.",
    "p15": "BF16 has the same exponent range as FP32 so loss scaling is not required.",
    "p16": "Gradient accumulation sums gradients over micro-batches before the optimizer step.",
}
QUERIES = [
    ("Which port should Prometheus scrape on Triton?", {"p1"}),
    ("How does Triton decide how long to wait to form a batch?", {"p2"}),
    ("What does FSDP shard?", {"p4"}),
    ("How does QLoRA reduce memory for fine-tuning?", {"p6", "p5"}),
    ("Why does paged attention help serving throughput?", {"p8", "p7"}),
    ("Where should tensor parallel groups be placed?", {"p11"}),
    ("Do I need loss scaling with bf16?", {"p15"}),
    ("What monitoring exports GPU memory usage?", {"p14"}),
]
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    ids, texts = list(PASSAGES), list(PASSAGES.values())
    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=str(DEVICE))
    p_emb = bi.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    runs = {"bi-encoder": [], "+ reranker": []}
    for q, _ in QUERIES:
        scores = (bi.encode(q, convert_to_tensor=True, normalize_embeddings=True) @ p_emb.T).tolist()
        top10 = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[:10]]
        ce_scores = ce.predict([(q, PASSAGES[p]) for p in top10])
        runs["bi-encoder"].append(top10)
        runs["+ reranker"].append([p for _, p in sorted(zip(ce_scores, top10), key=lambda t: -t[0])])
    gold = [g for _, g in QUERIES]
    for name, ranked in runs.items():
        p1 = sum(lab.precision_at_k(r, g, 1) for r, g in zip(ranked, gold)) / len(gold)
        r5 = sum(lab.recall_at_k(r, g, 5) for r, g in zip(ranked, gold)) / len(gold)
        ndcg = sum(lab.ndcg_at_k(r, {d: 1 for d in g}, 5) for r, g in zip(ranked, gold)) / len(gold)
        print(f"{name:<11} P@1={p1:.2f}  recall@5={r5:.2f}  MRR={lab.mrr(ranked, gold):.2f}  nDCG@5={ndcg:.2f}")
    print("Reranking reorders the same candidates, so recall@10 is unchanged and precision at the top improves.")
except ImportError:
    print("[skipped] pip install sentence-transformers")

# %% 3. A vs B on short-answer QA with EM/F1 and a paired bootstrap CI
QA = [
    ("What does GPU stand for?", "graphics processing unit"),
    ("What is the default HTTP port of Triton Inference Server?", "8000"),
    ("Which company makes CUDA?", "NVIDIA"),
    ("What does LoRA stand for?", "low-rank adaptation"),
    ("How many bits are in a byte?", "8"),
    ("What collective does DDP use to synchronize gradients?", "all-reduce"),
    ("What is the capital of France?", "Paris"),
    ("What precision format has 8 exponent bits and 7 mantissa bits?", "bfloat16"),
    ("What does RAG stand for?", "retrieval-augmented generation"),
    ("Which optimizer keeps first and second moment estimates?", "Adam"),
    ("What is 12 times 12?", "144"),
    ("What does KV stand for in KV cache?", "key value"),
    ("Which activation does Llama use in its MLP?", "SwiGLU"),
    ("What does MoE stand for?", "mixture of experts"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which NVIDIA library provides multi-GPU collectives?", "NCCL"),
]


@torch.no_grad()
def answer(tok, model, q):
    prompt = tok.apply_chat_template([{"role": "user", "content": q + " Answer with only a few words."}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**enc, max_new_tokens=12, do_sample=False)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


f1 = {}
for name, (tok, model) in loaded.items():
    preds = [answer(tok, model, q) for q, _ in QA]
    em = [lab.exact_match(p, g) for p, (_, g) in zip(preds, QA)]
    f1[name] = [lab.token_f1(p, g) for p, (_, g) in zip(preds, QA)]
    print(f"{name}: EM={sum(em) / len(em):.2f}  F1={sum(f1[name]) / len(QA):.2f}")
diff, lo, hi = lab.paired_bootstrap(*f1.values())
print(f"F1(B) − F1(A) = {diff:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] → "
      f"{'significant' if lo > 0 or hi < 0 else 'NOT significant with only ' + str(len(QA)) + ' questions'}")

# %% 4. LLM-as-judge vs human labels, and position bias
HUMAN = [  # (question, candidate answer, human verdict)
    ("What does DDP synchronize?", "Gradients, averaged with an all-reduce.", "correct"),
    ("What does DDP synchronize?", "The input data batches across GPUs.", "incorrect"),
    ("Why use BF16?", "It keeps FP32's exponent range so training is stable without loss scaling.", "correct"),
    ("Why use BF16?", "Because it is more precise than FP32.", "incorrect"),
    ("What is the KV cache?", "Saved keys and values of earlier tokens reused during decoding.", "correct"),
    ("What is the KV cache?", "A cache of the model weights on the CPU.", "incorrect"),
    ("What is LoRA?", "Training a low-rank update while freezing base weights.", "correct"),
    ("What is LoRA?", "A long-range radio protocol used for GPUs.", "incorrect"),
    ("What does Triton expose on port 8002?", "Prometheus metrics.", "correct"),
    ("What does Triton expose on port 8002?", "The gRPC inference endpoint.", "incorrect"),
]
judge_tok, judge = loaded["B (1.5B)"]


@torch.no_grad()
def judge_says(prompt, choices):
    enc = judge_tok(judge_tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True), return_tensors="pt").to(DEVICE)
    logits = judge(**enc).logits[0, -1]
    first = [judge_tok.encode(c, add_special_tokens=False)[0] for c in choices]
    return choices[int(torch.stack([logits[i] for i in first]).argmax())]


RUBRIC = ("You are grading answers about GPU and LLM engineering. A correct answer is factually accurate and "
          "addresses the question. Reply with exactly one word: correct or incorrect.")
verdicts = [judge_says(f"{RUBRIC}\n\nQuestion: {q}\nAnswer: {a}", ["correct", "incorrect"]) for q, a, _ in HUMAN]
human = [h for _, _, h in HUMAN]
print(f"judge vs human: raw agreement {sum(v == h for v, h in zip(verdicts, human)) / len(human):.0%}, "
      f"Cohen's κ = {lab.cohens_kappa(verdicts, human):.2f}")

consistent = 0
pairs = [(HUMAN[i], HUMAN[i + 1]) for i in range(0, len(HUMAN), 2)]
for (q, good, _), (_, bad, _) in pairs:
    ask = lambda a1, a2: judge_says(f"Question: {q}\nAnswer 1: {a1}\nAnswer 2: {a2}\n"
                                    "Which answer is better? Reply with 1 or 2.", ["1", "2"])
    consistent += (ask(good, bad) == "1") and (ask(bad, good) == "2")
print(f"pairwise judgements consistent after swapping positions: {consistent}/{len(pairs)}")
