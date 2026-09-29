# Lab 09 — Solution walkthrough

Every exercise is explained in four parts:

- **Why it exists:** the problem it solves. This is what the exam tests.
- **How it works:** the idea, with a small example using real numbers (computed by running `solutions.py`).
- **The code:** why each line is there.
- **On the exam:** a question in the exam's style, with **why each wrong answer is wrong**.

The picture: "is model B better than model A?" needs the **right metric** for the task, **enough
evidence** that the difference is real, a **trustworthy grader** for open-ended output, and a look at
**where** it fails. Then it has to be **automated**, so every change is checked.

| Task | Metric | Exercise |
|---|---|---|
| language modelling | perplexity | 1 |
| extractive QA | exact match, token F1 | 2 |
| translation | BLEU (precision) | 3 |
| summarisation | ROUGE-L (recall) + faithfulness | 4, 11 |
| retrieval / reranking | precision@k, recall@k, MRR, nDCG | 5 |
| code | pass@k | 6 |
| multiple-choice benchmarks (MMLU, ARC, HellaSwag) | log-likelihood accuracy (acc / acc_norm) | 12 |
| open-ended quality | LLM-as-judge (debiased) + human agreement (κ) | 8, 10 |
| is the difference real? | paired bootstrap CI | 7 |
| where does it fail? | macro-F1, slices, error buckets | 9 |
| every change | regression gate in CI | 13 |

---

## Exercise 1 — `perplexity`

### Why it exists

Measures how well a language model predicts text: `exp(− mean ln p(token))`. If every correct token gets
probability 0.25, PPL = **4** ("as unsure as a fair 4-way choice"); 1 is perfect. It tracks
pre-training and domain adaptation progress without labels.

### On the exam

PPL is **per token**, so compare only with the **same tokenizer and data**; otherwise use bits per byte.
It measures language modelling, **not task quality**. The GPU lab shows the bigger model gets lower PPL, and scrambled text much higher.

---

## Exercise 2 — `exact_match` / `token_f1`

Normalise first (lowercase, strip punctuation and articles): `"The  Eiffel Tower!"` → `"eiffel tower"`.
**EM** is all or nothing. **Token F1** gives partial credit: `"Paris, France"` vs `"Paris"` → P 1/2, R 1/1, F1 0.667.
`Counter & Counter` counts each shared word as often as it appears in both. Used for extractive QA (SQuAD); both miss paraphrases.

---

## Exercise 3 — `bleu`

### Why it exists

For translation: **what fraction of the candidate's n-grams appear in the reference** (precision), n = 1..4,
geometric mean, times a **brevity penalty**.

```
ref "the cat is on the mat"
"the cat is on the mat today" → 0.809      "the cat is on" → 0.607 (too short: BP)
"the the the the the the"     → BLEU-1 0.333 thanks to CLIPPING (not 1.0)
```

**Clipping** caps each word at its count in the reference, so repetition can't game the score.

---

## Exercise 4 — `rouge_l`

For summaries: **longest common subsequence** (in order, gaps allowed), recall-oriented.
`"police kill the gunman"` vs `"police killed the gunman"` → LCS 3 → F = 0.75; the same words reordered drop to 0.50.

### On the exam

**BLEU → translation, ROUGE → summarisation, and neither checks facts.** A fluent hallucination can
score well, which is why summaries also get **faithfulness** (exercise 11). Code uses **pass@k**; classification uses accuracy/F1.

---

## Exercise 5 — retrieval metrics

Retrieved `[d3, d1, d7, d2, d9]`, relevant `{d1, d2, d5}`:

| metric | asks | value |
|---|---|---|
| precision@5 | how much of what I returned is relevant? | 0.40 |
| recall@5 | how much of the relevant set did I find? | 0.67 |
| MRR | how high is the first relevant hit? | 0.50 |
| nDCG@5 (graded 3/2/1) | are the most relevant documents near the top? | 0.578 |

A reranker changes **order**, so it moves MRR, nDCG and P@1, not recall over the candidate set (the GPU lab shows this).

---

## Exercise 6 — `pass_at_k`

`1 − C(n−c, k) / C(n, k)`: 20 samples, 3 pass → pass@1 = 0.15, pass@5 = 0.60, pass@10 = 0.89. It's the
**unbiased** estimate for the n samples you have. The naive `1 − (1−0.15)^5 = 0.556` assumes
sampling with replacement from an infinite pool. Code is judged by **executing tests**, because many correct programs differ textually.

---

## Exercise 7 — `paired_bootstrap`: is the improvement real?

### Why it exists

B scores 72.25% vs A's 72.75% on 400 questions. Is that real? Resample the **questions** with
replacement 2,000 times and recompute `mean(B − A)` each time:

```
difference −0.005, 95% CI [−0.015, +0.005] → contains 0 → no evidence either model is better
```

It's **paired** because both models are scored on the same items, which removes question-difficulty noise.

### On the exam

*B beats A by 0.5 points on 400 questions:* **compute a paired bootstrap CI (or paired test) first.**
Re-running B at a higher temperature or reporting only B's score is wrong.

---

## Exercise 8 — `cohens_kappa`: can we trust the grader?

18 "good", 2 "bad" by humans; a judge that always says "good" gets **90% raw agreement but κ = 0**:
`κ = (p_o − p_e)/(1 − p_e)` removes the agreement expected by chance (here p_e = 0.9). Check your LLM
judge's κ against human labels on a sample **before** trusting it at scale.

---

## Exercise 9 — `macro_f1` and `slice_accuracy`

Macro-F1 weights every class equally, so rare classes can't hide (0.656 vs accuracy 0.667 in the example).
Slices sort groups **worst first**: `sw 0.25, fr 0.75, en 0.90`. An overall 0.72 hides that Swahili users get 25%.
Then do **error analysis**: sample failures from the worst slice, label *why* each failed, and fix the biggest bucket.

---

## Exercise 10 — `pairwise_judge`: debiasing LLM-as-a-judge

### Why it exists

LLM judges scale qualitative evaluation, but they have **known biases**:

| bias | what happens | mitigation |
|---|---|---|
| **position** | prefers whichever answer is shown first (or second) | **evaluate both orders; count only consistent verdicts** |
| **verbosity** | prefers longer answers | rubric that penalises padding; length-controlled comparisons |
| **self-preference** | prefers outputs from its own model family | a judge from a different family |
| **leniency / scale drift** | scores cluster or shift | anchored rubric with examples; calibrate against humans (κ) |

### How it works

```python
first  = judge(q, A, B)   # A shown first
second = judge(q, B, A)   # B shown first
"1" then "2" → A wins      "2" then "1" → B wins      anything else → tie (position-driven)
```

A judge that always answers "1" gets **tie** every time: the swap exposes that its verdict followed
the slot, not the content. The GPU lab runs this with a real model and reports how many pairs stay consistent.

### On the exam (Select TWO)

✅ **swap positions and keep consistent verdicts**; ✅ **an anchored rubric plus agreement checks with humans**.
❌ the same model family as judge and candidate; ❌ "prefer the longer answer when unsure"; ❌ hiding the rubric.

---

## Exercise 11 — `faithfulness`: is the answer supported by the context?

### Why it exists

In RAG, an answer can be fluent, relevant and **wrong**, because it states things the retrieved
context doesn't support. **Faithfulness (groundedness)** = the share of the answer's claims that the
context supports. RAGAS-style evaluation splits the answer into claims and checks each with an NLI
model or an LLM judge (`supports`).

```
context: "ZeRO-3 shards parameters, gradients and optimizer states."
claims:  "ZeRO-3 shards parameters" ✓     "ZeRO-3 was invented in 1999" ✗     → 0.5
```

### On the exam

*RAG gives wrong answers on 20% of questions: how to find the failing part?* Evaluate **retrieval**
(recall@k, MRR against gold passages) **and generation** (faithfulness to the retrieved context, answer
correctness) separately. Retrieval never found the passage? Fix chunking, embeddings or the reranker. Found it but the
answer contradicts it? Fix prompting or the model.

---

## Exercise 12 — `mcq_predictions`: how benchmarks score multiple choice

### Why it exists

MMLU, ARC and HellaSwag are multiple choice. Tools like **lm-evaluation-harness** don't parse a
generated letter. They compute the log-likelihood of **each option's text** after the question and pick the most likely:

```
option:              0            1            2
total log-prob:     -6.0         -9.0         -5.5      → acc picks 2
tokens:              2            6            1
per token:          -3.0         -1.5         -5.5      → acc_norm picks 1
```

A long option accumulates more negative log-probability just by being long, so **acc_norm** normalises by
length. The harness reports both; HellaSwag's headline number is typically acc_norm.

### On the exam

Fix the prompt format, few-shot count and scoring method when comparing models. Changing them changes the
score. Benchmarks to recognise: **MMLU** (knowledge), **HellaSwag / ARC** (commonsense and science),
**GSM8K** (multi-step maths), **HumanEval** (code, pass@k), **TruthfulQA** (misconceptions),
**MT-Bench / Arena** (chat, judge or human preference). NVIDIA **NeMo Evaluator** runs these plus LLM-as-judge as a service.

---

## Exercise 13 — `regression_gate`: evaluation in CI

### Why it exists

Prompts, templates and model versions change every week, and each change can quietly break something.
Treat evaluation like software tests: a **versioned regression suite** (golden set, previously broken
cases, safety prompts) runs on every change, and a gate blocks the release if any metric falls more than its tolerance.

```
baseline  faithfulness 0.91  rougeL 0.41  safety_pass 0.99
candidate               0.90         0.35              0.99
allowed drop            0.02         0.03              0.00   → blocked: ["rougeL"]
```

Safety metrics usually get **zero tolerance**. Track every run's config (model, prompt, parameters,
data version) and results in an experiment tracker (W&B, MLflow) for comparison over time.

### On the exam

*A scalable, trustworthy evaluation pipeline with Triton and W&B (Select TWO):* **identical versioned
prompts, parameters and held-out data for every model, served through Triton, with every run logged**,
and **automatic metrics plus a calibrated judge and human review, with per-example logging for error
analysis.** Wrong: tuning prompts per model, ranking by training loss, manual notebooks.

---

## Quiz topic → where you learn it in this lab

| Quiz topic | Exercise |
|---|---|
| Choosing metrics | 1–6, table at the top |
| Perplexity | 1 |
| Retrieval metrics | 5 |
| pass@k | 6 |
| Statistical significance | 7 |
| Inter-rater agreement | 8 |
| Error analysis | 9 |
| LLM-as-judge bias | 10 |
| RAG evaluation | 5, 11 |
| Benchmark selection | 12 |
| Continuous evaluation, scalable pipeline | 13 |
