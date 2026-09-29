# Lab 03 — Data Preparation (9% of exam)

> Blueprint: *dataset cleaning/curation, tokenization, vocabulary management, data organization.*

You will build a miniature version of the **NeMo Curator** pipeline (normalise, dedup
exact and fuzzy, filter for quality, redact PII), train a BPE tokenizer from scratch, and handle
the data-organisation details that decide whether fine-tuning succeeds: packing, padding, splits
without leakage, and new special tokens.

```
make test-03          # YOUR exercises (run from the repo root)
make solutions-03     # reference solutions
make quiz-03          # exam-style questions
make gpu-03           # GPU part (gpu_lab.py) on the Brev/AWS instance
```

---

## 1. The curation pipeline

```
raw text ─► language ID ─► unicode/whitespace normalise ─► exact dedup (hash) ─► fuzzy dedup (MinHash+LSH)
        ─► heuristic quality filters ─► model-based quality classifier ─► PII redaction ─► toxicity / safety filter
        ─► decontamination (remove benchmark n-grams) ─► blend / upsample domains ─► tokenise & pack ─► shards
```

| Stage | Why | NVIDIA tooling |
|---|---|---|
| **Exact dedup** | Hash each normalised doc (MD5/xxhash) and drop repeats. Duplicates waste compute and increase memorisation. | NeMo Curator `ExactDuplicates` (GPU, RAPIDS) |
| **Fuzzy dedup** | Near-duplicates (templated pages, boilerplate). MinHash signatures estimate Jaccard similarity; LSH buckets find candidate pairs without an O(n²) comparison. | `FuzzyDuplicates` |
| **Semantic dedup** | Embed docs, cluster, and drop items too close in embedding space. | `SemDedup` |
| **Heuristic filters** | Gopher/C4-style rules: word count, mean word length, symbol ratio, repeated lines, "lorem ipsum", ellipsis lines. | `ScoreFilter` + heuristic filters |
| **Classifier filters** | fastText or DeBERTa quality/domain classifiers. | Quality / domain classifiers |
| **PII redaction** | Emails, phones, IDs, names. Needed for compliance (GDPR/HIPAA). | `PiiModifier` (Presidio-based) |
| **Decontamination** | Remove training docs that overlap evaluation benchmarks (n-gram match), or your scores are inflated. | `TaskDecontamination` |
| **Synthetic data** | Generate or augment data with an LLM, then filter with a reward model. | NeMo Curator SDG, Nemotron-4-340B-Reward |

**Garbage in, garbage out:** quality and diversity beat raw volume. A smaller clean dataset often
out-trains a larger noisy one, especially for fine-tuning (LIMA: ~1k curated examples).

## 2. Tokenization

| Algorithm | Used by | How it builds the vocabulary |
|---|---|---|
| **BPE** | GPT-2/3/4 (byte-level), Llama-3 (tiktoken), Nemotron | Start from characters or bytes; repeatedly **merge the most frequent adjacent pair**. |
| **WordPiece** | BERT | Like BPE but merges the pair that most increases likelihood; `##` marks continuations. |
| **Unigram LM** | T5, ALBERT, via **SentencePiece** | Start big and prune tokens that least reduce corpus likelihood. |
| **SentencePiece** | Llama-1/2, T5, Mistral | Library that treats input as raw Unicode including spaces (`▁`), so it is language-agnostic with no pre-tokenisation. Runs BPE or Unigram. |

- **Byte-level BPE** never produces `<unk>`: any string is representable as bytes.
- **Fertility** (tokens per word) measures tokenizer fit to a domain or language. High fertility
  means longer sequences, higher cost and less effective context. Domain jargon, code and
  non-Latin scripts often fragment.
- **Vocabulary management:** adding domain tokens or special tokens (`<|tool_call|>`, chat role
  markers) requires `tokenizer.add_tokens` / `add_special_tokens` **and**
  `model.resize_token_embeddings(len(tokenizer))`. Initialise new rows with the mean of the
  existing embeddings, not random noise. New embeddings must be *trained*: include
  `embed_tokens`/`lm_head` in the trainable modules even with LoRA.
- **Always use the tokenizer the model was trained with.** A mismatch silently produces garbage.
- Special tokens: BOS/EOS, PAD (often set `pad_token = eos_token` for decoder models), and
  `padding_side="left"` for **batched generation** with decoder-only models.

## 3. Organising data for training

- **Formats:** JSONL, one record per line. For SFT the record is `{"messages": [{"role": ..., "content": ...}]}`
  or `{"input": ..., "output": ...}`. Pre-training corpora are tokenised into binary
  `.bin/.idx` indexed datasets (Megatron / NeMo).
- **Sequence packing:** concatenate short examples (separated by EOS) into full `max_len` rows,
  so no compute is spent on padding. Use attention masks or position resets so packed examples
  don't attend to each other (NeMo `packed_sequence`).
- **Length bucketing:** batch similar lengths together to reduce padding when you don't pack.
- **Splits without leakage:** split by *group* (document, patient, user, conversation), dedupe
  *before* splitting, and keep a held-out test set you never tune on.
- **Data blending:** mix domains with explicit weights and upsample scarce, high-value data.
  Keep some general data to reduce **catastrophic forgetting** during domain adaptation.
- **Class balance** for classification SFT: stratified splits, re-weighting or resampling.

## 4. Exam traps

- MinHash estimates **Jaccard similarity** of shingle sets. LSH is what makes the search sub-quadratic.
- Dedup *before* the train/test split. Otherwise near-duplicates leak across the split.
- Adding tokens without resizing embeddings leads to index errors or silently untrained vectors.
- Right-padding with a decoder-only model during batched generation leads to wrong continuations. Use left padding.
- Lowercasing or stripping punctuation for LLM training data is usually **harmful**. Normalise Unicode and whitespace, not case.
- Packing without cross-document masking can leak context between examples. It is acceptable for pre-training, but be careful in SFT.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `normalize_text` | Unicode NFKC, control chars, whitespace |
| 2 | `exact_dedup` | hash-based exact deduplication |
| 3 | `shingles`, `minhash_signature`, `estimate_jaccard` | fuzzy dedup |
| 4 | `quality_issues` | Gopher-style heuristic filters |
| 5 | `redact_pii` | regex PII redaction |
| 6 | `train_bpe`, `bpe_encode` | byte-pair encoding from scratch |
| 7 | `pack_sequences`, `padding_efficiency` | packing vs padding |
| 8 | `split_by_group` | leakage-free deterministic splits |
| 9 | `resize_embeddings` | adding tokens to a vocabulary |
