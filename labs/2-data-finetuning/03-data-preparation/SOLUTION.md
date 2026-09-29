# Lab 03 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: **the data decides the model.** Before a single GPU-hour of training, a pipeline
cleans the raw text, removes duplicates, drops junk and private data, checks that no test answers
leaked in, then turns text into token ids arranged efficiently for the GPU.

```
raw text → normalise (1) → exact dedup (2) → fuzzy dedup (3, 10) → quality filter (4) → PII (5)
         → decontaminate (12) → blend sources (14) → tokenise (6, 11) → pack / pad (7, 13)
         → split without leakage (8) → train                        (+ new tokens: 9)
```

This is the **NeMo Curator** pipeline in miniature. It does these steps on GPUs at billions of documents.

---

## Exercise 1 — `normalize_text`: make equal text look equal

### Why it exists

The same sentence arrives in many byte-level forms: full-width letters (`Ｈｅｌｌｏ`), ligatures
(`ﬁ`), invisible control characters, random spacing. Without normalisation, hashing (ex. 2) treats
them as different documents and duplicates slip through.

### How it works

`"  Ｈｅｌｌｏ\x07   world\t!\n\n\n\n  next   line  "` → `"Hello world !\n\nnext line"`

| step | code | why |
|---|---|---|
| Unicode NFKC | `unicodedata.normalize("NFKC", …)` | folds compatibility forms: `Ｈ→H`, `ﬁ→fi` |
| drop control chars | keep `\n`, `\t`; drop category `Cc` | invisible junk |
| collapse spaces/tabs | `re.sub(r"[ \t]+", " ", …)` | `"world\t!"` → `"world !"` |
| strip lines, ≤ 2 newlines | | keep paragraphs, drop empty runs |

Tab is itself a control character, so it must be kept in step 2 or it would be deleted instead of becoming a space.

### On the exam

**Don't lowercase or strip punctuation** for LLM training data: "US" and "us" differ, and models
must learn to write proper case and punctuation. Normalise *encoding and whitespace*, not *content*.

---

## Exercise 2 — `exact_dedup`: drop identical copies cheaply

### Why it exists

Web crawls are full of copies. Duplicates **waste compute**, make the model **memorise** (and
regurgitate) text, and if copies span train and test they **inflate evaluation scores**. Exact
dedup is the cheapest first pass: hash each normalised document and keep the first of each hash.

### How it works

```
["Hello  world", "hello world", "Hello world", "Other doc"]
 → keeps "Hello  world" (its normalised form equals "Hello world", so the third is dropped)
 → keeps "hello world" (case differs, so it's a different document)
```

A hash turns a document into a fixed-size number, and a set lookup is O(1), so dedup is linear in corpus size.
`hash64` uses MD5, **not Python's `hash()`**, which is randomly salted per process: the same text
would hash differently tomorrow, and parallel workers would disagree.

### On the exam

Exact dedup misses **near-duplicates** (the same article with a new timestamp). That's the next exercise.

---

## Exercise 3 — MinHash: estimating similarity with short signatures

### Why it exists

We want **Jaccard similarity** of word shingles (overlapping 3-word pieces), but comparing every pair
of 2 billion documents directly is impossible (about 10¹⁸ pairs). **MinHash** compresses each
document into a short signature whose agreement rate *estimates* Jaccard.

### How it works

```
"the cat sat on the mat" → {the cat sat, cat sat on, sat on the, on the mat}
"the cat sat on a mat"   → {the cat sat, cat sat on, sat on a,   on a mat}
shared 2, union 6 → J = 0.333
```

**The key fact:** for a random hash function h, `P(min h(A) = min h(B)) = J(A, B)`. The smallest
hash in `A ∪ B` is equally likely to be any element of the union, and the two minima agree exactly
when that element is in both. Use many hash functions and count agreements:

| num_perm | estimate (true J = 0.333) |
|---|---|
| 16 | 0.438 |
| 64 | 0.406 |
| 256 | 0.336 |

Code: `h_i(x) = (a_i·hash64(x) + b_i) mod P` with the prime `P = 2⁶¹ − 1` gives cheap, different
hash functions. The product exceeds 64 bits, so compute with **Python ints** (NumPy int64 would overflow silently).

### On the exam

MinHash estimates **Jaccard similarity of shingle sets**. It's the signature step. Making the search
**sub-quadratic** is LSH's job (exercise 10).

---

## Exercise 4 — `quality_issues`: cheap rules that remove junk

### Why it exists

A large share of crawled text is menus, spam, tables of numbers and "Read more…" teasers. Training
on it wastes compute and teaches bad habits. **Gopher/C4-style heuristics** remove it without any model:

| rule | fails when | catches |
|---|---|---|
| `word_count` | < 50 or > 100,000 words | fragments, giant dumps |
| `mean_word_length` | outside 3–10 characters | code tables, broken tokenisation |
| `symbol_ratio` | (`#` + ellipses) per word > 0.1 | hashtag spam, teasers |
| `alpha_fraction` | < 80% of words contain a letter | number tables, logs |
| `ellipsis_lines` | > 30% of lines end with `...` | "Click to read more..." pages |

Return **reasons**, not just keep/drop, so removals are explainable. Thresholds are corpus-specific:
AG News blurbs in the GPU lab nearly all fail `word_count`.

### On the exam

Heuristics come **first** (cheap, high-yield). **Model-based quality classifiers** (fastText,
NeMo Curator's DeBERTa classifiers) come after, on what's left. Human review doesn't scale, and
"remove everything with numbers" throws away technical content.

---

## Exercise 5 — `redact_pii`: remove personal data before training

### Why it exists

LLMs can **memorise** training text and repeat it, including someone's phone number. Regulations
(GDPR, HIPAA) require protecting personal data, and **redaction after training can't remove what was memorised.**

### How it works

```
"Call 555-123-4567 or mail jo@x.com, SSN 123-45-6789, host 10.0.0.1"
→ "Call [PHONE] or mail [EMAIL], SSN [SSN], host [IP]"
```

**Order matters:** an SSN and an IP are digits plus separators that a loose phone pattern could
partially match, so redact the more specific patterns first. Regexes are the cheap layer; production uses
NER-based tools (Presidio, NeMo Curator `PiiModifier`) for names and addresses.

### On the exam

For regulated data: **redact or pseudonymise before training** and **keep an auditable record of
sources, consent and the redaction process** (data lineage). More epochs increase memorisation risk;
removing punctuation anonymises nothing.

---

## Exercise 6 — `train_bpe` / `bpe_encode`: how tokenizers are built

### Why it exists

Models need a **fixed vocabulary**. Whole words give a huge vocabulary with unknown words everywhere;
single characters give very long sequences. **Subword tokenization** is the compromise: frequent
words become one token, and rare words split into known pieces.

### How it works

BPE starts from characters and repeatedly **merges the most frequent adjacent pair**
(counts `low:5, lower:2, newest:6, widest:3`):

| merge | pair | why it wins |
|---|---|---|
| 1 | `e`+`s` | 6 + 3 = 9 (ties with `s`+`t`; alphabetical order breaks the tie) |
| 2 | `es`+`t` | 9 |
| 3 | `l`+`o` | 5 + 2 = 7 |
| 4 | `lo`+`w` | 7 |

After 4 merges: `lowest → [low, est]`, `newer → [n, e, w, e, r]` (no merge applies). Pair counts are
weighted by word frequency; `min(pairs, key=(-count, pair))` makes it deterministic; encoding replays merges **in learned order**.

### On the exam: the four tokenizer families

| algorithm | how it builds the vocabulary | marker | used by |
|---|---|---|---|
| **BPE** | merge the most frequent pair | | GPT-2, Llama-3 (byte-level, ex. 11) |
| **WordPiece** | merge the pair that most increases likelihood | `##` on continuations | BERT |
| **Unigram LM** | start big, prune tokens that least reduce likelihood | | T5, ALBERT |
| **SentencePiece** | a *library* treating text as raw Unicode including spaces (`▁`), so no pre-tokenisation is needed; runs BPE or Unigram | `▁` | Llama-1/2, T5, Mistral |

Always use **the tokenizer the model was trained with**. Another tokenizer's ids mean nothing to the model.

---

## Exercise 7 — `pack_sequences` / `padding_efficiency`: stop computing on padding

### Why it exists

GPUs process rectangular batches, and short sequences are padded to the longest. **Padding tokens cost
the same compute as real ones.** With 180-token SFT examples padded to 4,096, over 95% of the GPU's
work is wasted.

### How it works

```
lengths [5, 100, 6, 98, 4, 101, 7, 99], batch 2
arrival order: (5,100)→200 (6,98)→196 (4,101)→202 (7,99)→198   real 420 / 796 = 52.8% useful
sorted (length bucketing): 420 / 424 = 99.1% useful
```

**Packing** concatenates examples (separated by EOS) into full rows:
`max_len 7: [1,2,3]+EOS and [4,5]+EOS → [1,2,3,0,4,5,0]`. Truncate oversized items, and start a new row
when the next item doesn't fit.

### On the exam

*Slow SFT with short examples and low GPU use* points to **sequence packing** (a multi-fold
speed-up). In SFT, packed examples must not attend to each other (attention masks or position
resets). Padding everything to the maximum makes it worse.

---

## Exercise 8 — `split_by_group`: validation you can trust

### Why it exists

If the same patient, customer or document appears in both train and validation, the model is
tested on things it has effectively seen: great scores, disappointing production. **Split by group, not by row.**

### How it works

```python
bucket = hash64(str(record[group_key])) % 10_000   # a stable number per group
val if bucket < val_fraction * 10_000 else train    # the same group always lands on the same side
```

Hash-based rather than `random`: **deterministic** (the same split everywhere) and it works in
streaming or distributed jobs.

### On the exam

*96% validation, 71% in production, with many near-identical tickets.* That's **leakage**:
deduplicate *before* splitting, and split by group. A smaller validation set, a different learning rate or a new tokenizer don't fix it.

---

## Exercise 9 — `resize_embeddings`: adding new tokens

### Why it exists

New special tokens (`<|tool_call|>`, chat roles) or domain words get new ids. The embedding matrix
needs **one new row per token**, or the model crashes with an index error.

### How it works

Append rows initialised to the **mean** of the existing embeddings, a neutral start. Random rows
can have unusual norms and produce wild logits. Hugging Face's `resize_token_embeddings` does the same.

### On the exam

*Index-out-of-range after adding 200 domain tokens before LoRA fine-tuning (Select TWO):*
**resize the embeddings** (`model.resize_token_embeddings(len(tokenizer))`) and **make the embedding
and LM head trainable** (`modules_to_save`), because LoRA freezes them and the new rows would never learn.

---

## Exercise 10 — `lsh_candidate_pairs`: fuzzy dedup at web scale

### Why it exists

MinHash (ex. 3) gives each document a signature, but comparing every *pair* of signatures is still
O(n²). **Locality-sensitive hashing** finds likely duplicates **without comparing all pairs**: split
each signature into bands, and only documents that match an entire band land in the same bucket.

### How it works

64-value signature = 16 bands × 4 rows. Two documents become candidates if **any** band matches exactly:

```
P(one band matches) = J^rows          every row in the band must agree
P(candidate)        = 1 − (1 − J^rows)^bands
```

| Jaccard | 16 bands × 4 rows | 8 bands × 8 rows |
|---|---|---|
| 0.3 | 0.122 | 0.001 |
| 0.5 | 0.644 | 0.031 |
| 0.7 | **0.988** | 0.378 |
| 0.9 | 1.000 | 0.989 |

An S-shaped curve: similar pairs almost always collide and dissimilar ones rarely do. **More rows per
band makes the threshold stricter.** In the test, two near-identical sentences (J ≈ 0.81) collide and the unrelated one doesn't.

Code: `sig[b*rows:(b+1)*rows].tobytes()` turns a band into a dictionary key; documents sharing a key share a bucket; pairs are generated only within buckets.

### On the exam

*Remove near-duplicates from 2 billion documents:* **MinHash signatures + LSH**, then verify the candidates.
Exact hashing misses near-duplicates, all-pairs TF-IDF is quadratic, and a length filter isn't dedup.
**Semantic dedup** (NeMo Curator SemDedup) goes further: embed documents and remove those too close in meaning (the GPU lab does this on the GPU).

---

## Exercise 11 — `fertility` and `utf8_byte_tokens`: how well does a tokenizer fit?

### Why it exists

A tokenizer trained mostly on English splits other languages, code and jargon into many small
pieces. **Fertility** (tokens per word) measures it. High fertility means **longer sequences**:
more compute (attention is quadratic), less text per context window, higher cost, and often worse quality.

### How it works

```
"The central bank kept interest rates unchanged"   7 words, 46 UTF-8 bytes  → byte fertility 6.6
"중앙은행은 기준금리를 동결했다"                        3 words, 44 bytes         → byte fertility 14.7
```

`utf8_byte_tokens` shows the **byte-level** idea: every text is a sequence of values 0–255, so a
byte-level BPE tokenizer (GPT-2, Llama-3) can encode **anything**, with no `<unk>` ever. Korean
syllables take 3 bytes each, which is why they fragment before merges help. The GPU lab measures real
tokenizers (GPT-2, BERT, Qwen) on English, code, medical text, French and Korean.

### On the exam

*3.5 tokens per word on a Korean medical corpus vs 1.3 for English: consequences (Select TWO)?*
**Fewer words fit in the context window and cost rises**, and **fragmented pieces carry less meaning,
which can hurt quality.** Wrong: "the model will refuse Korean", "attention becomes linear".
Remedies: a tokenizer with better coverage, vocabulary extension plus continued pre-training, or a multilingual model.

---

## Exercise 12 — `contaminated_items`: did the test leak into training?

### Why it exists

Benchmarks live on the internet, and the internet is your training data. If test questions (or near
copies) were trained on, the model can **recall** answers instead of reasoning, and scores are inflated.
**Decontamination** removes training documents that overlap evaluation sets. NeMo Curator's task decontamination does this.

### How it works

Compare **long word n-grams** (13-grams, as popularised by GPT-3), after lowercasing and dropping punctuation:

```
train: "Question: What is the capital of France? Answer: Paris is the capital of France."
test:  "what is the capital of france answer paris"      → shares 5-grams → contaminated
test:  "How many legs does a spider have?"               → clean
```

Long n-grams rarely match by chance, but copied text shares them. Normalising case and punctuation
stops trivial reformatting from hiding a leak. With n = 13 the example isn't flagged: the test item has only 8 words.

### On the exam

*A new model scores unusually high on a public benchmark: which check tests contamination most directly?*
**Search the training data for long n-gram overlaps with the test items.** Higher temperature,
parameter counts or evaluating on the train split don't test for overlap. Decontaminate **before** training.

---

## Exercise 13 — `pad_batch`: which side to pad on

### Why it exists

Batches must be rectangular, so shorter sequences are padded, and the **side** matters:
- **Right padding** (tokens first): standard for training and for encoders.
- **Left padding** (padding first): required for **batched generation with decoder-only models**.
  Generation continues from the **last position** of every row, and with left padding the last column is
  every row's real last token. With right padding, short prompts would "continue" after pad tokens.

### How it works

```
[[5,6,7], [8]]   right → ids [[5,6,7], [8,0,0]]   mask [[1,1,1], [1,0,0]]
                 left  → ids [[5,6,7], [0,0,8]]   mask [[1,1,1], [0,0,1]]   last column = [7, 8] ✓
```

The **attention mask** (1 = real) tells attention to ignore padding in either case.

### On the exam

*Batched generation is good for the longest prompt, degraded for shorter ones* means right padding; set
`tokenizer.padding_side = "left"`. Passing the attention mask is correct, and `pad_token = eos_token`
is a common, valid choice for decoder models without a pad token.

---

## Exercise 14 — `blend_plan`: mixing data sources

### Why it exists

Training data comes from sources of very different sizes and value: a huge web crawl, a small
curated legal corpus, some instruction data. Sampling in proportion to size would drown the valuable
sources. **Blending** sets explicit mixture weights, and this tells you how often each source is repeated.

### How it works

20 B training tokens, weights web 7 : legal 2 : instructions 1:

| source | size | tokens drawn | epochs |
|---|---|---|---|
| web | 1,000 B | 14 B | 0.014 (a small slice) |
| legal | 2 B | 4 B | **2.0** (seen twice: upsampled) |
| instructions | 0.5 B | 2 B | **4.0** |

Upsampling scarce, high-value data is normal, but **many repeats cause memorisation and
overfitting**, so keep epochs of small sources modest.

### On the exam

*After fine-tuning only on legal documents, general reasoning drops sharply:* this is **catastrophic
forgetting**. **Blend in general-domain data** (replay), which is exactly what this plan does.
More legal-only epochs make it worse. PEFT and lower learning rates also help (Lab 04).

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise |
|---|---|
| Fuzzy deduplication | 3, 10 |
| Dedup and splits (leakage) | 2, 8 |
| Adding tokens | 9 |
| Tokenizer fertility, byte-level BPE | 11 |
| Tokenization algorithms | 6 |
| Sequence packing | 7 |
| Curation pipeline order | the pipeline diagram at the top, 1–5, 12 |
| Benchmark contamination | 12 |
| Padding side | 13 |
| PII | 5 |
| Heuristic quality filters | 4 |
| Data blending, forgetting | 14 |
