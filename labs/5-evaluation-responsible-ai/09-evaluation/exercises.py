"""Lab 09 — Evaluation. Fill in every TODO, then run:

    pytest labs/5-evaluation-responsible-ai/09-evaluation
"""

import math
import re
import string
from collections import Counter
from collections.abc import Callable

import numpy as np


# 1 ─────────────────────────────────────────────────────────────────────────────
def perplexity(token_logprobs: list[float]) -> float:
    """exp(− mean natural-log probability of each token)."""
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, remove punctuation, remove articles (a, an, the),
    collapse whitespace.
    """
    raise NotImplementedError


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalised strings are equal else 0.0."""
    raise NotImplementedError


def token_f1(prediction: str, reference: str) -> float:
    """F1 over normalised tokens using multiset overlap (Counter &). 0 if no overlap.
    If both are empty return 1.0; if exactly one is empty return 0.0.
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU (single reference, whitespace tokens, no smoothing):
      p_n = clipped n-gram matches / candidate n-grams, for n = 1..max_n
      BLEU = BP · exp(mean(log p_n));  0.0 if any p_n == 0
      BP = 1 if c > r else exp(1 − r/c)   (c, r = candidate / reference lengths)
    """
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 on whitespace tokens: LCS length → P = lcs/len(cand), R = lcs/len(ref),
    F = 2PR/(P+R) (0 if lcs == 0).
    """
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError


def mrr(retrieved_lists: list[list[str]], relevant_sets: list[set[str]]) -> float:
    """Mean over queries of 1 / rank of the first relevant item (0 if none)."""
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """DCG = Σ_{i=1..k} rel_i / log2(i + 1); nDCG = DCG / ideal DCG (0 if ideal is 0).
    `relevance` maps doc id → graded relevance (missing = 0).
    """
    raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c pass: 1 − C(n−c, k) / C(n, k).
    Return 1.0 when n − c < k.
    """
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def paired_bootstrap(
    scores_a: list[float], scores_b: list[float], n_resamples: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    """Per-example scores of two systems on the SAME examples. Resample example indices with
    replacement (rng = np.random.default_rng(seed); rng.integers(0, n, size=(n_resamples, n))),
    compute mean(b − a) for each resample. Return (observed mean diff, 2.5th pct, 97.5th pct).
    """
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def cohens_kappa(rater_a: list, rater_b: list) -> float:
    """κ = (p_o − p_e) / (1 − p_e); p_e from each rater's label marginals. Return 1.0 if p_e == 1."""
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
def macro_f1(y_true: list, y_pred: list) -> float:
    """Unweighted mean of per-class F1 over the classes present in y_true ∪ y_pred."""
    raise NotImplementedError


def slice_accuracy(records: list[dict], key: str) -> list[tuple[str, float, int]]:
    """Group records (each has `key` and a boolean "correct") by record[key].
    Return [(group, accuracy, count)] sorted by accuracy ASCENDING (worst slice first),
    ties broken by group name.
    """
    raise NotImplementedError


# 10 ────────────────────────────────────────────────────────────────────────────
def pairwise_judge(judge: Callable[[str, str, str], str], question: str, answer_a: str, answer_b: str) -> str:
    """Position-debiased pairwise LLM-as-judge. judge(question, first, second) returns "1" or "2"
    (which shown answer is better). Call it twice: (a, b) and (b, a).
    Consistent preference for a → "A"; for b → "B"; anything else → "tie".
    """
    raise NotImplementedError


# 11 ────────────────────────────────────────────────────────────────────────────
def faithfulness(claims: list[str], context: str, supports: Callable[[str, str], bool]) -> float:
    """RAG faithfulness: fraction of the answer's claims for which supports(claim, context) is True.
    No claims → 1.0.
    """
    raise NotImplementedError


# 12 ────────────────────────────────────────────────────────────────────────────
def mcq_predictions(choice_logprobs: list[list[float]], choice_lengths: list[list[int]], normalize: bool) -> list[int]:
    """Log-likelihood multiple-choice scoring (lm-evaluation-harness). For each question pick the
    index of the choice with the highest score: its total log-probability (acc), or with
    normalize=True that total divided by the choice's length (acc_norm).
    """
    raise NotImplementedError


# 13 ────────────────────────────────────────────────────────────────────────────
def regression_gate(baseline: dict[str, float], candidate: dict[str, float], max_drop: dict[str, float]) -> tuple[bool, list[str]]:
    """Release gate. For each metric in max_drop (in that order) the candidate fails it if
    baseline − candidate > the allowed drop. Return (passed, failed metric names).
    """
    raise NotImplementedError
