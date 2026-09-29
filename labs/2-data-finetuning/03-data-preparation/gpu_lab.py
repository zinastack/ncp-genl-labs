# %% [markdown]
# # Lab 03 — GPU: curate a real corpus with YOUR pipeline
# Section 2 instance (1× L4/T4). Loads AG News through `datasets` (falls back to a built-in
# corpus offline), injects known near-duplicates, then runs:
#
# 1. normalise → exact dedup → MinHash + LSH banding (your functions, candidate pairs only)
# 2. heuristic quality filters and PII redaction statistics
# 3. semantic dedup with GPU sentence embeddings (the SemDedup idea)
# 4. tokenizer fertility across tokenizers, domains and languages
# 5. train a domain BPE tokenizer with HF `tokenizers` and compare
# 6. padding vs bucketing vs packing efficiency on real token lengths

# %%
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE = dev.device()
N_DOCS = int(os.environ.get("N_DOCS", 3000))
random.seed(0)
print(dev.summary())


# %% Load a corpus
def load_corpus(n):
    try:
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news", split=f"train[:{n}]")
        return [row["text"] for row in ds]
    except Exception as e:  # offline or `datasets` unavailable
        print(f"[datasets unavailable: {type(e).__name__}] using the built-in corpus")
        topics = ["GPU", "stock market", "football", "election", "vaccine", "satellite", "chip factory", "bank"]
        verbs = ["announces", "reports", "delays", "expands", "cuts", "launches", "wins", "loses"]
        return [f"{random.choice(topics).title()} company {random.choice(verbs)} plans for "
                f"{random.choice(topics)} after quarter {i % 4 + 1}; analysts expect "
                f"{random.randint(1, 99)}% growth in {2020 + i % 6}. " * 3 for i in range(n)]


docs = load_corpus(N_DOCS)
# Inject known duplicates so we can measure recall: 3% exact copies with whitespace noise,
# 3% near-duplicates with a few words changed.
exact_copies = [d.replace(" ", "  ", 3) for d in random.sample(docs, len(docs) * 3 // 100)]
near_copies = []
for d in random.sample(docs, len(docs) * 3 // 100):
    w = d.split()
    for _ in range(2):
        w[random.randrange(len(w))] = "UPDATED"
    near_copies.append(" ".join(w))
corpus = docs + exact_copies + near_copies
random.shuffle(corpus)
print(f"{len(corpus)} documents ({len(exact_copies)} exact + {len(near_copies)} near duplicates injected)")

# %% 1. Exact dedup, then MinHash + LSH banding
t0 = time.perf_counter()
normalized = [lab.normalize_text(d) for d in corpus]
unique = lab.exact_dedup(corpus)
print(f"exact dedup: {len(corpus)} → {len(unique)} ({len(corpus) - len(unique)} removed) in {time.perf_counter() - t0:.1f}s")

NUM_PERM, BANDS = 64, 16          # 16 bands × 4 rows: pairs with Jaccard ≳ 0.5 collide with high probability
ROWS = NUM_PERM // BANDS
t0 = time.perf_counter()
sigs = [lab.minhash_signature(lab.shingles(lab.normalize_text(d), 3), NUM_PERM) for d in unique]
buckets = defaultdict(list)
for i, s in enumerate(sigs):
    for b in range(BANDS):
        buckets[(b, s[b * ROWS:(b + 1) * ROWS].tobytes())].append(i)
candidates = {(a, c) for ids in buckets.values() if len(ids) > 1 for a in ids for c in ids if a < c}
dups = {(a, c) for a, c in candidates if lab.estimate_jaccard(sigs[a], sigs[c]) >= 0.7}
drop = {c for _, c in dups}
print(f"MinHash+LSH: {len(candidates)} candidate pairs (vs {len(unique) * (len(unique) - 1) // 2:,} all-pairs), "
      f"{len(dups)} near-duplicate pairs, {len(drop)} docs dropped in {time.perf_counter() - t0:.1f}s")
curated = [d for i, d in enumerate(unique) if i not in drop]

# %% 2. Quality filters and PII
issues = Counter(issue for d in curated for issue in lab.quality_issues(d))
print("quality rule failures:", dict(issues) or "none")
print("(AG News items are short, so 'word_count' (< 50 words) dominates. Thresholds are corpus-specific!)")
pii_hits = sum(lab.redact_pii(d) != d for d in curated)
print(f"documents containing PII patterns: {pii_hits}")

# %% 3. Semantic dedup on the GPU: embed, normalise, cosine similarity, threshold
try:
    from sentence_transformers import SentenceTransformer

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
    t0 = time.perf_counter()
    emb = emb_model.encode(curated, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
    sims = emb @ emb.T
    sims.fill_diagonal_(0)
    i, j = torch.where(torch.triu(sims) > 0.95)
    print(f"semantic dedup: {len(i)} pairs with cosine > 0.95 among {len(curated)} docs "
          f"({time.perf_counter() - t0:.1f}s on {DEVICE})")
    for a, b in list(zip(i.tolist(), j.tolist()))[:2]:
        print("  ·", curated[a][:90], "\n  ·", curated[b][:90])
except ImportError:
    print("[skipped] pip install sentence-transformers")

# %% 4. Tokenizer fertility (tokens per word) across tokenizers and text types
SAMPLES = {
    "English news": "The central bank kept interest rates unchanged while inflation eased for a third month.",
    "Python code": "def all_reduce(tensor, group=None):\n    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)\n    return tensor / world_size",
    "Medical": "Patient presents with paroxysmal supraventricular tachycardia and hypokalemia post-thyroidectomy.",
    "French": "La banque centrale a maintenu ses taux directeurs inchangés alors que l'inflation ralentissait.",
    "Korean": "중앙은행은 인플레이션이 3개월 연속 완화되자 기준금리를 동결했다.",
}
tokenizers_ = {name: AutoTokenizer.from_pretrained(name) for name in
               ["gpt2", "bert-base-uncased", "Qwen/Qwen2.5-0.5B-Instruct"]}
print(f"\n{'text':<14}" + "".join(f"{n.split('/')[-1][:18]:>20}" for n in tokenizers_))
for label, text in SAMPLES.items():
    words = len(text.split())
    print(f"{label:<14}" + "".join(f"{len(t.encode(text, add_special_tokens=False)) / words:>20.2f}"
                                   for t in tokenizers_.values()))
print("Higher = more fragmentation = longer sequences and more cost. Note Korean and medical jargon.")

# %% 5. Train a domain BPE tokenizer (HF tokenizers, Rust) and compare with GPT-2 on held-out domain text
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402

bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.train_from_iterator(curated[:-200], trainers.BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "<eos>"]))
held_out = curated[-200:]
ours = sum(len(bpe.encode(d).ids) for d in held_out)
gpt2 = sum(len(tokenizers_["gpt2"].encode(d)) for d in held_out)
print(f"held-out tokens: domain BPE (8k vocab) = {ours:,} vs GPT-2 (50k vocab) = {gpt2:,}")
print("A small in-domain vocabulary can rival a much larger general one on its own domain. "
      "Your from-scratch train_bpe learns the same kind of merges, just slower.")
print("first learned merges from your implementation:", lab.train_bpe(Counter(" ".join(curated[:300]).split()), 8))

# %% 6. Padding vs bucketing vs packing with real token lengths
qtok = tokenizers_["Qwen/Qwen2.5-0.5B-Instruct"]
token_lists = [qtok.encode(d, add_special_tokens=False) for d in curated]
lengths = [len(t) for t in token_lists]
print(f"\nlengths: mean {np.mean(lengths):.0f}, max {max(lengths)} tokens")
print(f"padding, random order, batch 32   : {lab.padding_efficiency(lengths, 32):.1%} real tokens")
print(f"padding, length-bucketed, batch 32: {lab.padding_efficiency(sorted(lengths), 32):.1%}")
packs = lab.pack_sequences(token_lists, 2048, qtok.eos_token_id)
fill = sum(map(len, packs)) / (len(packs) * 2048)
print(f"packing into 2048-token rows      : {fill:.1%} ({len(packs)} rows instead of {len(lengths)} sequences)")
