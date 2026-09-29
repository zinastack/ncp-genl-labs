# Lab 09 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link.

The picture: "is model B better than model A?" needs three things: the **right metric** for the
task, **enough evidence** that the difference is real, and a look at **where** it fails.

```
language modelling → perplexity (1)       extractive QA → EM / F1 (2)
translation → BLEU (3)                    summarisation → ROUGE-L (4)
retrieval / RAG → P@k, recall@k, MRR, nDCG (5)       code → pass@k (6)
is it real? → bootstrap CI (7)   can I trust the grader? → Cohen's κ (8)   where does it fail? → slices (9)
```

---

## Exercise 1 — `perplexity`: how surprised is the model?

```
PPL = exp( − mean over tokens of ln p(token) )
```

- If the model gives every correct token probability 0.25, PPL = **4**: it's as uncertain as a
  fair choice among 4 tokens. **PPL = 1** means perfect prediction.
- Token probabilities 0.9, 0.5, 0.1 → PPL ≈ 2.81. The one bad guess (0.1) dominates, because it's a log.
- The GPU lab measures real text vs the same words scrambled: the bigger model gets lower PPL, and scrambled text gets a much higher PPL.

> **Exam trap:** PPL is **per token**, so it's only comparable with the **same tokenizer and dataset**.
> A 128k-vocabulary model and a 32k one split text differently. PPL also measures language
> modelling, not task quality.

---

## Exercise 2 — `exact_match` / `token_f1`: extractive QA (SQuAD)

First **normalise** both strings so trivial differences don't count:
`"The  Eiffel Tower!"` → lowercase, strip punctuation, drop articles (a/an/the), collapse spaces → `"eiffel tower"`.

- **Exact match:** 1 if the normalised strings are identical, else 0. Strict.
- **Token F1:** partial credit from overlapping words.

```
prediction "Paris, France"  → [paris, france]      reference "Paris" → [paris]
common 1 → precision 1/2, recall 1/1 → F1 = 2·P·R/(P+R) = 0.667

prediction "the capital is Paris" → [capital, is, paris]   reference "Paris France" → [paris, france]
common 1 → P 1/3, R 1/2 → F1 = 0.4
```

`Counter(p) & Counter(r)` is the **multiset intersection**: a word counts as often as it appears
in *both*. Edge cases: both empty → 1.0 (nothing asked, nothing answered); exactly one empty → 0.

---

## Exercise 3 — `bleu`: n-gram **precision** for translation

"What fraction of the candidate's word sequences appear in the reference?" for n = 1..4, combined
with a geometric mean, times a **brevity penalty**:

```
reference "the cat is on the mat"
candidate                         BLEU-4   BLEU-1
"the cat is on the mat today"     0.809    0.857    (one extra word lowers precision)
"the cat is on"                   0.607    0.607    (every n-gram matches but it's too short → BP = e^(1−6/4))
"the the the the the the"         0         0.333   (clipping!)
```

- **Clipping:** a candidate word counts at most as many times as it appears in the reference
  (`min(cnt, ref[g])`). "the" is in the reference twice, so 6 × "the" gets 2/6, not 6/6.
  Without clipping, repeating one word would score perfectly.
- **Brevity penalty:** precision alone rewards saying very little ("the" is 100% precise), so short outputs are penalised: `BP = exp(1 − r/c)` if c ≤ r.
- If any n-gram precision is 0, BLEU = 0 (no smoothing). Sentence-level BLEU is harsh; corpus BLEU and smoothing exist for that reason.

---

## Exercise 4 — `rouge_l`: longest common subsequence for summaries

ROUGE is **recall-oriented**: "how much of the reference did the summary cover?" ROUGE-L uses the
**longest common subsequence** (in order, gaps allowed):

```
reference "police killed the gunman"
"police kill the gunman" → LCS = police, the, gunman = 3 → P 3/4, R 3/4 → F = 0.75
"the gunman kill police" → LCS = the, gunman = 2       → P 2/4, R 2/4 → F = 0.50
```

Same words, different order, lower score: LCS respects word order. The LCS uses classic dynamic
programming with a single rolling row (`dp`), so memory is O(len(b)).

> BLEU (precision) → translation; ROUGE (recall) → summarisation. **Neither checks facts.** A
> fluent hallucination can score well, which is why summaries also get faithfulness checks.

---

## Exercise 5 — retrieval metrics for RAG

Retrieved ranking `[d3, d1, d7, d2, d9]`, relevant documents `{d1, d2, d5}`:

| metric | question | value |
|---|---|---|
| precision@5 | how much of what I retrieved is relevant? | 2/5 = 0.40 |
| recall@5 | how much of what's relevant did I retrieve? | 2/3 = 0.67 (d5 missed) |
| MRR | how high is the **first** relevant hit? | first hit at rank 2 → 1/2 = 0.50 |
| nDCG@5 (graded d1=3, d2=2, d5=1) | are the **most** relevant documents near the top? | 0.578 |

**nDCG:** each hit is worth `relevance / log2(rank + 1)`, so later ranks count less. DCG =
3/log2(3) + 2/log2(5) = 2.75. The ideal order (3, 2, 1 at ranks 1–3) gives 4.76. nDCG = 2.75 / 4.76 = 0.578.

**Why this matters (exam RAG question):** high recall with low precision means the right passage
*is* retrieved but buried among noise, so **add a reranker**. The GPU lab shows it: the
cross-encoder reorders the same 10 candidates, P@1, MRR and nDCG rise, and recall@10 can't change.

---

## Exercise 6 — `pass_at_k`: code generation

Generate n = 20 samples per problem, and c = 3 pass the unit tests. "If I let the model try k
times, what's the chance at least one passes?"

```
pass@k = 1 − C(n−c, k) / C(n, k)        C(n−c, k)/C(n, k) = chance that k random picks are ALL failures
pass@1 = 0.15    pass@5 = 0.60    pass@10 = 0.89
```

Why not `1 − (1 − 0.15)^5 = 0.556`? That formula assumes sampling **with** replacement from an
infinite pool. The combinatorial estimator is **unbiased** for the actual n samples. When n − c < k
every pick of k must include a pass, so the result is 1.0 (`math.comb` would otherwise get invalid arguments).

---

## Exercise 7 — `paired_bootstrap`: is the improvement real?

Model B scores 72.25% vs A's 72.75% on 400 questions. Is that a difference or noise?
**Resample the questions** 2,000 times with replacement, and each time compute
`mean(B − A)` on the resampled set. The spread of those means is your uncertainty:

```
A 0.7275, B 0.7225 → difference −0.005, 95% CI [−0.015, +0.005]
the interval contains 0 → no evidence either model is better
```

- **Paired:** both models are scored on the *same* questions, and we resample question indices
  (`diff[idx]`), which removes question-difficulty noise.
- `rng.integers(0, n, size=(n_resamples, n))` builds all resamples at once (vectorised), then `.mean(axis=1)`.
- The 2.5th and 97.5th percentiles of the resampled means form the 95% confidence interval.

---

## Exercise 8 — `cohens_kappa`: agreement beyond chance

Checking an LLM judge against human labels? Raw agreement lies when one label dominates:

```
human: 18 "good", 2 "bad"      judge: says "good" 20 times
raw agreement 18/20 = 90%, yet the judge never detected a single "bad" → κ = 0
```

```
κ = (p_o − p_e) / (1 − p_e)
p_o = observed agreement
p_e = agreement expected by chance = Σ over labels of P(rater A says it) · P(rater B says it)
```

Here p_e = 0.9·1.0 + 0.1·0 = 0.9, so κ = (0.9 − 0.9)/(1 − 0.9) = 0. Another example (6 items, 4
agree, balanced labels): p_o = 0.667, p_e = 0.5, κ = 0.333. Rough scale: < 0.2 poor, 0.4–0.6
moderate, > 0.8 strong. The `p_e == 1` guard handles the case where both raters always give the same single label.

---

## Exercise 9 — `macro_f1` and `slice_accuracy`: where does it fail?

**Macro-F1** gives each class equal weight (per-class F1 = 2·TP / (2·TP + FP + FN), then an
unweighted mean):

```
true  [pos, pos, neg, neg, neu, neu]      pred [pos, neg, neg, neg, pos, neu]
F1: pos 0.50, neg 0.80, neu 0.67 → macro 0.656      (accuracy 0.667)
```

With imbalanced classes, accuracy can look great while a rare class is always wrong. Macro-F1 exposes that.

**Slices:** group results by an attribute and sort **worst first**:

```
language  accuracy  count
sw        0.25      4      ← look here first
fr        0.75      4
en        0.90      10     overall 0.72 hides that Swahili users get 25%
```

Then **error analysis**: sample failures from the worst slice, label *why* each failed (retrieval
miss, hallucination, format, tokenisation…), and fix the biggest bucket.

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| summarisation metric | ROUGE-L + faithfulness check |
| translation metric | BLEU (or chrF/COMET) |
| code benchmark | pass@k (HumanEval) |
| compare PPL across tokenizers | invalid; use the same tokenizer or bits-per-byte |
| B beats A by 0.5 points | bootstrap CI / paired test before claiming it |
| LLM judge reliability | rubric, position swap, κ against humans |
| RAG wrong answers | evaluate retrieval (recall/MRR) and generation (faithfulness) separately |
| aggregate fine, a group complains | slice metrics + error analysis |
| scalable evaluation pipeline | Triton/NIM batch inference, fixed prompts and params, tracked in W&B/MLflow |
| suspiciously high benchmark | contamination (n-gram overlap) |
