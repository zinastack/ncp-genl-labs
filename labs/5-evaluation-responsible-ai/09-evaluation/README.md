# Lab 09 — Evaluation (7% of exam)

> Blueprint: *quantitative/qualitative metrics, benchmarking, error analysis, scalable assessment.*

You will implement the metrics the exam names (perplexity, EM/F1, BLEU, ROUGE-L, retrieval
metrics, pass@k), the statistics that make comparisons trustworthy (paired bootstrap confidence
intervals, Cohen's κ for judge agreement), and slice-based error analysis. On the GPU instance you
benchmark a small model with **lm-evaluation-harness** and run an **LLM-as-judge** pass against a
local model server.

```
make test-09          # YOUR exercises (run from the repo root)
make solutions-09     # reference solutions
make quiz-09          # exam-style questions
make gpu-09           # GPU part (gpu_lab.py) on the Brev/AWS instance
make s5-lm-eval       # lm-evaluation-harness on a small model
```

---

## 1. Metric cheat sheet

| Metric | Measures | Good for | Blind spots |
|---|---|---|---|
| **Perplexity** `exp(−mean log p)` | how well the LM predicts text | pre-training and domain adaptation progress; same-tokenizer comparisons | not task quality; not comparable across tokenizers |
| **Accuracy / F1 (macro)** | classification | classification, multiple-choice benchmarks (MMLU) | class imbalance (use macro-F1) |
| **Exact match / token F1** | answer string overlap after normalisation | extractive QA (SQuAD) | paraphrases |
| **BLEU** | n-gram *precision* + brevity penalty | machine translation | meaning; single-reference bias |
| **ROUGE-1/2/L** | n-gram / LCS *recall*-oriented overlap | summarisation | faithfulness; abstractive paraphrase |
| **BERTScore** | embedding similarity of tokens | paraphrase-robust similarity | factuality |
| **pass@k** | P(at least one of k samples passes the unit tests) | code (HumanEval, MBPP) | needs executable tests |
| **Precision@k / Recall@k / MRR / nDCG** | retrieval ranking | RAG retrievers and rerankers | end-answer quality |
| **Faithfulness / groundedness, answer relevance, context precision/recall** | RAG answer quality (RAGAS-style, often LLM-judged) | RAG | judge bias |
| **LLM-as-a-judge** | rubric-based quality (helpfulness, correctness, safety) | open-ended generation at scale | position, verbosity and self-preference bias; calibrate against humans |
| **Human evaluation** | gold-standard qualitative judgement | final decisions, preference data | cost, rater disagreement (measure with Cohen's κ) |

**Benchmarks to recognise:** MMLU (knowledge, multiple choice), HellaSwag and ARC
(commonsense and science reasoning), GSM8K (grade-school maths), HumanEval (code), TruthfulQA
(truthfulness), MT-Bench and Arena (chat quality, LLM judge / human preference), HELM (holistic).
Tools: **lm-evaluation-harness**, **NeMo Evaluator** (a microservice for academic benchmarks,
LLM-as-judge and custom datasets), RAGAS, and W&B / MLflow for tracking.

## 2. Doing evaluation properly

1. **Held-out, deduplicated, decontaminated** test sets. Never tune on the test set.
2. Fix decoding for reproducibility (temperature 0 or a fixed seed), and fix the prompt template and few-shot count.
3. Report **uncertainty**: bootstrap confidence intervals or paired tests. A 0.5-point gain on 200 examples is noise.
4. **Slice** results by input type, length, language, topic and user group, and look at the *worst*
   slices. Aggregate metrics hide failures.
5. **Error analysis:** sample the failures, categorise them (hallucination, format error, refusal,
   retrieval miss, reasoning error), count them, and fix the biggest bucket first.
6. Combine **quantitative** (automatic metrics) with **qualitative** (human or judge review of
   samples) evaluation. Track everything (model version, data version, prompt, parameters) in an
   experiment tracker.
7. **Scalable assessment:** batch inference through Triton or NIM, parallel judge calls, cached
   generations, and a CI-style regression suite that runs on every model or prompt change.

**LLM-judge hygiene:** a clear rubric with anchored scores, reasoning before the score, swapping
answer positions for pairwise comparisons (position bias), controlling for length (verbosity
bias), a judge from a different model family than the one evaluated (self-preference), and
periodic **agreement checks with humans** (κ).

## 3. Exam traps

- BLEU is precision-oriented (translation). ROUGE is recall-oriented (summarisation). Neither measures factual correctness.
- Perplexity comparisons need the **same tokenizer and dataset**.
- Suspiciously high benchmark scores point to **contamination**. Check n-gram overlap.
- The unbiased pass@k uses `1 − C(n−c, k)/C(n, k)` with n ≥ k samples, not `1 − (1−p)^k` computed from a single greedy run.
- For RAG, evaluate retrieval (recall@k, MRR) **and** generation (faithfulness) separately, so you know which component failed.
- κ corrects agreement for chance. 90% raw agreement can still be a poor κ when one label dominates.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function | Concept |
|---|---|---|
| 1 | `perplexity` | LM quality |
| 2 | `normalize_answer`, `exact_match`, `token_f1` | extractive QA metrics |
| 3 | `bleu` | n-gram precision + brevity penalty |
| 4 | `rouge_l` | LCS-based F-measure |
| 5 | `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k` | retrieval / reranking |
| 6 | `pass_at_k` | unbiased code-generation metric |
| 7 | `paired_bootstrap` | confidence interval for a model difference |
| 8 | `cohens_kappa` | judge–human agreement |
| 9 | `macro_f1`, `slice_accuracy` | error analysis |
| 10 | `pairwise_judge` | position-debiased LLM-as-a-judge |
| 11 | `faithfulness` | RAG groundedness: share of claims supported by the context |
| 12 | `mcq_predictions` | how lm-evaluation-harness scores multiple choice (acc vs acc_norm) |
| 13 | `regression_gate` | continuous evaluation: blocking releases on metric regressions |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).

## Further reading

Chosen to explain the concepts behind this lab; read the *Start here* items first.

**Start here**
- [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) (Eugene Yan): LLM-as-judge biases (position, verbosity, self-preference) and how to measure agreement.
- [The LLM Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) (Hugging Face): automatic benchmarks, human evaluation, judges and common pitfalls.
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) (Hugging Face): the sliding-window perplexity you compute on the GPU.

**Go deeper**
- [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/) (Eugene Yan): evals, RAG, guardrails and feedback loops in production.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): the harness behind most leaderboards.
- [HELM](https://crfm.stanford.edu/helm/) (Stanford CRFM): holistic, multi-metric evaluation.

**Papers**
- [Judging LLM-as-a-Judge (MT-Bench)](https://arxiv.org/abs/2306.05685) · [RAGAS](https://arxiv.org/abs/2309.15217) · [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)
