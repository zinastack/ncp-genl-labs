# Lab 03 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: **the data decides the model.** Before a single GPU-hour of training, a pipeline
cleans the raw text, removes duplicates, drops junk and private data, then turns text into
token ids arranged efficiently for the GPU.

```
raw text → normalise (1) → exact dedup (2) → fuzzy dedup (3) → quality filter (4) → PII redaction (5)
         → tokenizer (6) → pack / batch (7) → split without leakage (8) → train   (+ new tokens: 9)
```

---

## Exercise 1 — `normalize_text`: make equal text look equal

The same sentence can arrive in many byte-level forms: full-width letters (`Ｈｅｌｌｏ`),
ligatures (`ﬁ`), invisible control characters, random spacing. Without normalisation, hashing (ex. 2)
treats them as different documents.

```
"  Ｈｅｌｌｏ\x07   world\t!\n\n\n\n  next   line  "   →   "Hello world !\n\nnext line"
```

| step | code | why |
|---|---|---|
| Unicode NFKC | `unicodedata.normalize("NFKC", text)` | folds compatibility forms: `Ｈ→H`, `ﬁ→fi` |
| drop control chars | keep `\n` and `\t`, drop category `Cc` (like `\x07`, the bell) | invisible junk |
| collapse spaces/tabs | `re.sub(r"[ \t]+", " ", …)` | `"world\t!"` → `"world !"` |
| strip each line | `"\n".join(l.strip() …)` | trailing spaces |
| ≤ 2 newlines | `re.sub(r"\n{3,}", "\n\n", …)` | keep paragraphs, drop empty runs |

Watch the order: tab is itself a control character, so it must be kept in step 2 or it would be
deleted instead of becoming a space. **Don't lowercase:** "US" and "us" mean different things,
and LLMs need case.

---

## Exercise 2 — `exact_dedup`: drop identical copies cheaply

Hash the **normalised** text and keep the first document per hash:

```
["Hello  world", "hello world", "Hello world", "Other doc"]
 → keeps "Hello  world" (its normalised form equals "Hello world" → the third is dropped)
 → keeps "hello world" (case differs → different document)
```

- A hash turns a document into one fixed-size number. Comparing numbers in a `set` is O(1), so
  dedup is linear in corpus size.
- `hash64` uses MD5, not Python's `hash()`, which is **randomly salted per process**. The same text
  would hash differently tomorrow, and parallel workers would disagree.
- Return the **original** documents. Normalisation is only for comparison.

**Why dedup at all?** Duplicates waste compute, make the model **memorise** (and regurgitate)
text, and duplicates spanning train and test inflate your evaluation scores.

---

## Exercise 3 — MinHash: finding *near* duplicates

Exact hashing misses "the same article with a different timestamp". We want **Jaccard similarity**:

```
J(A, B) = |A ∩ B| / |A ∪ B|        (shared pieces / all pieces)
```

**Shingles** are the pieces: overlapping word 3-grams.

```
"the cat sat on the mat" → {the cat sat, cat sat on, sat on the, on the mat}
"the cat sat on a mat"   → {the cat sat, cat sat on, sat on a,   on a mat}
shared 2, union 6 → J = 0.333
```

Computing J for every pair of 2 billion documents is impossible (~10¹⁸ pairs). **MinHash**
compresses each document into a short signature so that:

```
P( min h(A) == min h(B) ) = J(A, B)      for a random hash function h
```

Why? Take the union `A ∪ B` and ask which element has the smallest hash. It is equally likely to
be any element of the union, and the two minima agree exactly when that element is in **both**
sets, so the probability is |A∩B| / |A∪B|.

So we use many hash functions (`num_perm`), take the min under each, and the **fraction of positions
that agree** estimates J. With the example above (true J = 0.333):

| num_perm | estimate |
|---|---|
| 16 | 0.438 |
| 64 | 0.406 |
| 256 | 0.336 |

More permutations give a better estimate but cost more. Code notes:
- `h_i(x) = (a_i · hash64(x) + b_i) mod P` with `P = 2⁶¹ − 1` (a prime) gives cheap, different hash functions.
- `int(ai) * h`: the product exceeds 64 bits, so compute with **Python ints** (unbounded), not NumPy int64 (it would overflow silently).

**LSH (used in the GPU lab):** split the 64-value signature into 16 bands of 4. Documents that
match on *any whole band* become candidates. Probability of becoming a candidate at similarity J
= `1 − (1 − J⁴)¹⁶`: J=0.3 → 0.12, 0.5 → 0.64, 0.7 → 0.99, 0.9 → 1.00. Similar pairs almost always
collide and dissimilar ones rarely do, so you only compare candidates. **That is what makes it
sub-quadratic** (exam keyword).

---

## Exercise 4 — `quality_issues`: cheap rules that remove junk

Web text contains menus, hashtag spam, number tables and "Read more…" teasers. Gopher/C4-style
rules catch them without any model:

| rule | fails when | catches |
|---|---|---|
| `word_count` | < 50 or > 100,000 words | fragments, giant dumps |
| `mean_word_length` | outside 3–10 characters | tables of codes, broken tokenisation |
| `symbol_ratio` | (`#` + `...`/`…`) per word > 0.1 | hashtag spam, teasers |
| `alpha_fraction` | < 80% of words contain a letter | number tables, logs |
| `ellipsis_lines` | > 30% of lines end with `...` | "Click to read more..." pages |

- The rules return **reasons**, not just keep or drop, so you can report *why* data was removed
  (NeMo Curator logs this too).
- Thresholds are corpus-specific. In the GPU lab, AG News items are short news blurbs, so nearly
  all fail `word_count`. That is a lesson: tune thresholds to your data.

---

## Exercise 5 — `redact_pii`: remove personal data before training

LLMs can **memorise** training text and repeat it, including someone's phone number.

```
"Call 555-123-4567 or mail jo@x.com, SSN 123-45-6789, host 10.0.0.1"
→ "Call [PHONE] or mail [EMAIL], SSN [SSN], host [IP]"
```

**Order matters:** an SSN `123-45-6789` and an IP `10.0.0.1` are made of digits and separators
that a loose phone regex could partially match. Redacting the more specific patterns first means
the phone rule never sees them. Regexes are the cheap first layer; production uses NER-based
tools (Presidio, NeMo Curator `PiiModifier`) for names and addresses.

---

## Exercise 6 — `train_bpe` / `bpe_encode`: how tokenizers are built

**Byte-Pair Encoding:** start from characters and repeatedly **merge the most frequent adjacent pair**
into a new token. Classic example (word counts `low:5, lower:2, newest:6, widest:3`):

| merge | pair | why it wins |
|---|---|---|
| 1 | `e`+`s` → `es` | appears 6 (newest) + 3 (widest) = 9 times; ties with `s`+`t` (also 9), and `('e','s')` sorts first |
| 2 | `es`+`t` → `est` | now 9 |
| 3 | `l`+`o` → `lo` | 5 + 2 = 7 |
| 4 | `lo`+`w` → `low` | 7 |
| 5 | `e`+`w` → `ew` | 6 (ties broken alphabetically) |
| 6 | `ew`+`est` → `ewest` | 6 |

With the first 4 merges: `lowest → [low, est]`, while `newer → [n, e, w, e, r]` because no
learned merge applies. Frequent words become single tokens and rare words break into pieces,
which is why **any word can still be encoded**.

Code notes:
- Pair counts are **weighted by word frequency** (`pairs[pair] += freq`).
- `min(pairs, key=lambda p: (-pairs[p], p))` means highest count first and alphabetical on ties, so the result is deterministic.
- `bpe_encode` replays merges **in learned order**. That order is the tokenizer's priority list.
- Real tokenizers (GPT-2, Llama-3) run BPE on **bytes**, so there is never an unknown token.

**Exam link:** high **fertility** (tokens per word) for a language or domain means longer
sequences, higher cost and less effective context. The GPU lab measures it for Korean and medical text.

---

## Exercise 7 — `pack_sequences` and `padding_efficiency`: stop computing on padding

GPUs process rectangular batches. Short sequences get **padded** to the longest in the batch, and
padding tokens cost the same compute as real ones.

```
lengths [5, 100, 6, 98, 4, 101, 7, 99], batch size 2, in arrival order:
(5,100)→200  (6,98)→196  (4,101)→202  (7,99)→198   real 420 / total 796 = 52.8% useful
sorted (bucketing): (4,5)→10 (6,7)→14 (98,99)→198 (100,101)→202   420 / 424 = 99.1%
```

**Packing** goes further: concatenate examples (separated by EOS) into full rows.

```
max_len 7: [1,2,3]+EOS and [4,5]+EOS fit in one row → [1,2,3,0,4,5,0]
           [6..12]+EOS is 8 tokens → truncated to 7 → [6,7,8,9,10,11,12]
           [13]+EOS → [13,0]
```

Rules in the code: append EOS, truncate anything longer than a row, and start a new row when the
next item doesn't fit. In SFT the packed examples must not attend to each other (attention masks
or position resets).

---

## Exercise 8 — `split_by_group`: validation you can trust

If the same patient (or customer, or document) appears in both train and validation, the model is
tested on things it has effectively seen, so the scores are inflated and production disappoints.
**Split by group**, not by row:

```python
bucket = hash64(str(record[group_key])) % 10_000       # stable number 0..9999 per group
val if bucket < val_fraction * 10_000 else train        # same group → same side, always
```

- Hash-based rather than `random`: **deterministic** (the same split every run and on every
  machine) and it works in streaming or distributed jobs, with no global shuffle needed.
- Dedup **before** splitting, or near-duplicates leak across the boundary anyway.

---

## Exercise 9 — `resize_embeddings`: adding new tokens

Adding special tokens (`<|tool_call|>`) or domain words gives them new ids. The embedding matrix
needs **one new row per new token**, or the model crashes with an index error.

```python
mean = embeddings.mean(axis=0, keepdims=True)          # the "average token"
np.vstack([embeddings, np.repeat(mean, n_new, axis=0)])
```

Why the mean and not random values? Random rows can have unusual norms and produce wild logits at
first. The mean is a neutral starting point (Hugging Face's `resize_token_embeddings` does this).
The new rows still need **training**: with LoRA, add `embed_tokens` and `lm_head` to the trainable modules.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| near-duplicates at web scale | MinHash + LSH (NeMo Curator fuzzy dedup) |
| great validation, poor production | leakage: dedup, then split by group |
| index error after adding tokens | `resize_token_embeddings` + train the new rows |
| many tokens per word in a language or domain | tokenizer fertility, so longer sequences and cost |
| slow SFT with short examples | sequence packing / length bucketing |
| batched generation broken for short prompts | pad on the **left** for decoder-only models |
| inflated benchmark scores | contamination: n-gram overlap with the test set |
| PII / compliance | redact before training, and document lineage |
