"""Lab 09 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import math
import re
import string
from collections import Counter, defaultdict

import numpy as np


def perplexity(token_logprobs: list[float]) -> float:
    # exp(average surprise): 4.0 means "as unsure as a fair pick among 4 tokens"; 1.0 is perfect.
    return math.exp(-sum(token_logprobs) / len(token_logprobs))


def normalize_answer(s: str) -> str:
    # SQuAD rules: case, punctuation, articles and spacing shouldn't decide correctness.
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)  # both empty → 1, one empty → 0
    common = sum((Counter(p) & Counter(r)).values())  # multiset intersection: min count per word
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    c, r = candidate.split(), reference.split()
    log_ps = []
    for n in range(1, max_n + 1):
        cand, ref = _ngrams(c, n), _ngrams(r, n)
        total = sum(cand.values())
        # Clipping: a candidate n-gram counts at most as often as it appears in the reference.
        matches = sum(min(cnt, ref[g]) for g, cnt in cand.items())
        if total == 0 or matches == 0:
            return 0.0  # geometric mean with a zero term (no smoothing)
        log_ps.append(math.log(matches / total))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))  # brevity penalty: precision alone rewards short outputs
    return bp * math.exp(sum(log_ps) / max_n)


def _lcs(a: list[str], b: list[str]) -> int:
    # Longest common subsequence (in order, gaps allowed); DP with one rolling row.
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(candidate: str, reference: str) -> float:
    c, r = candidate.split(), reference.split()
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in retrieved[:k]) / k  # of what I returned, how much is relevant


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in retrieved[:k]) / len(relevant)  # of what's relevant, how much I found


def mrr(retrieved_lists: list[list[str]], relevant_sets: list[set[str]]) -> float:
    total = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        total += next((1 / i for i, d in enumerate(retrieved, 1) if d in relevant), 0.0)  # 1/rank of first hit
    return total / len(retrieved_lists)


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    # Each hit is worth relevance / log2(rank + 1), so lower ranks count less.
    dcg = sum(relevance.get(d, 0) / math.log2(i + 1) for i, d in enumerate(retrieved[:k], 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]  # the best possible ordering
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0  # any k picks must include a passing sample
    # 1 − P(all k picks, without replacement, are failures): unbiased for the n samples we have.
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def paired_bootstrap(
    scores_a: list[float], scores_b: list[float], n_resamples: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    # PAIRED: both systems on the same examples, so resample example indices once for both.
    diff = np.asarray(scores_b, float) - np.asarray(scores_a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)  # one mean difference per resample (vectorised)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_kappa(rater_a: list, rater_b: list) -> float:
    n = len(rater_a)
    p_o = sum(a == b for a, b in zip(rater_a, rater_b)) / n  # observed agreement
    ca, cb = Counter(rater_a), Counter(rater_b)
    p_e = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)  # agreement expected by chance
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def macro_f1(y_true: list, y_pred: list) -> float:
    f1s = []
    for c in sorted(set(y_true) | set(y_pred), key=str):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))  # per-class F1
    # Unweighted mean: a rare class counts as much as a frequent one.
    return sum(f1s) / len(f1s)


def slice_accuracy(records: list[dict], key: str) -> list[tuple[str, float, int]]:
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(bool(r["correct"]))
    rows = [(g, sum(v) / len(v), len(v)) for g, v in groups.items()]
    return sorted(rows, key=lambda row: (row[1], str(row[0])))  # worst slice first
