import math

import numpy as np
import pytest


def test_1_perplexity(lab):
    assert lab.perplexity([math.log(0.25)] * 10) == pytest.approx(4.0), "uniform over 4 tokens → PPL 4"
    assert lab.perplexity([0.0, 0.0]) == pytest.approx(1.0)


def test_2_em_f1(lab):
    assert lab.normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert lab.exact_match("the Eiffel tower.", "Eiffel Tower") == 1.0
    assert lab.exact_match("Paris, France", "Paris") == 0.0
    assert lab.token_f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert lab.token_f1("London", "Paris") == 0.0
    assert lab.token_f1("", "") == 1.0 and lab.token_f1("a", "Paris") == 0.0


def test_3_bleu(lab):
    ref = "the cat is on the mat"
    assert lab.bleu(ref, ref) == pytest.approx(1.0)
    assert lab.bleu("the cat sat on a mat", ref) == 0.0, "no 4-gram match → 0 without smoothing"
    cand = "the cat is on the mat today"
    # p1=6/7, p2=5/6, p3=4/5, p4=3/4, candidate longer → BP=1
    expected = math.exp((math.log(6 / 7) + math.log(5 / 6) + math.log(4 / 5) + math.log(3 / 4)) / 4)
    assert lab.bleu(cand, ref) == pytest.approx(expected)
    short = "the cat is on"
    assert lab.bleu(short, ref) == pytest.approx(math.exp(1 - 6 / 4) * 1.0), "brevity penalty"


def test_4_rouge_l(lab):
    ref = "police killed the gunman"
    assert lab.rouge_l("police kill the gunman", ref) == pytest.approx(0.75)
    assert lab.rouge_l("the gunman kill police", ref) == pytest.approx(0.5)
    assert lab.rouge_l("xyz", ref) == 0.0


def test_5_retrieval(lab):
    retrieved = ["d3", "d1", "d7", "d2", "d9"]
    relevant = {"d1", "d2", "d5"}
    assert lab.precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert lab.recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert lab.mrr([retrieved, ["d5"], ["x"]], [relevant, relevant, relevant]) == pytest.approx((1 / 2 + 1 + 0) / 3)
    rel = {"d1": 3, "d2": 2, "d5": 1}
    dcg = 3 / math.log2(3) + 2 / math.log2(5)
    idcg = 3 / 1 + 2 / math.log2(3) + 1 / 2
    assert lab.ndcg_at_k(retrieved, rel, 5) == pytest.approx(dcg / idcg)
    assert lab.ndcg_at_k(["d1", "d2", "d5"], rel, 3) == pytest.approx(1.0)


def test_6_pass_at_k(lab):
    assert lab.pass_at_k(10, 1, 1) == pytest.approx(0.1)
    assert lab.pass_at_k(10, 1, 10) == 1.0
    assert lab.pass_at_k(20, 5, 5) == pytest.approx(1 - math.comb(15, 5) / math.comb(20, 5))
    assert lab.pass_at_k(10, 0, 5) == 0.0


def test_7_bootstrap(lab):
    rng = np.random.default_rng(1)
    a = (rng.random(1000) < 0.70).astype(float)
    b_better = np.where(rng.random(1000) < 0.15, 1.0, a)     # b fixes ~15% of a's items
    diff, lo, hi = lab.paired_bootstrap(a, b_better)
    assert lo > 0 and lo < diff < hi, "a real improvement: CI excludes 0"
    noise = np.where(rng.random(1000) < 0.02, 1 - a, a)       # random flips
    _, lo2, hi2 = lab.paired_bootstrap(a, noise)
    assert lo2 < 0 < hi2, "no real difference: CI includes 0"


def test_8_kappa(lab):
    assert lab.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    a = ["good"] * 18 + ["bad"] * 2
    b = ["good"] * 20
    assert lab.cohens_kappa(a, b) == pytest.approx(0.0), "90% raw agreement, zero beyond chance"
    x = ["y", "y", "n", "n", "y", "n"]
    y = ["y", "n", "n", "n", "y", "y"]
    assert lab.cohens_kappa(x, y) == pytest.approx(1 / 3)


def test_9_error_analysis(lab):
    y_true = ["pos", "pos", "neg", "neg", "neu", "neu"]
    y_pred = ["pos", "neg", "neg", "neg", "pos", "neu"]
    # pos F1=0.5, neg F1=0.8, neu F1=2/3
    assert lab.macro_f1(y_true, y_pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    records = [{"lang": "en", "correct": True}] * 9 + [{"lang": "en", "correct": False}] \
        + [{"lang": "sw", "correct": False}] * 3 + [{"lang": "sw", "correct": True}] \
        + [{"lang": "fr", "correct": True}] * 3 + [{"lang": "fr", "correct": False}]
    assert lab.slice_accuracy(records, "lang") == [("sw", 0.25, 4), ("fr", 0.75, 4), ("en", 0.9, 10)]


def test_10_pairwise_judge(lab):
    prefers_longer = lambda q, first, second: "1" if len(first) > len(second) else "2"
    always_first = lambda q, first, second: "1"
    assert lab.pairwise_judge(prefers_longer, "q", "a detailed answer", "short") == "A"
    assert lab.pairwise_judge(prefers_longer, "q", "short", "a detailed answer") == "B"
    assert lab.pairwise_judge(always_first, "q", "x", "y") == "tie", "position bias is caught by swapping"


def test_11_faithfulness(lab):
    supports = lambda claim, ctx: claim.lower() in ctx.lower()
    ctx = "ZeRO-3 shards parameters, gradients and optimizer states."
    assert lab.faithfulness(["zero-3 shards parameters", "zero-3 was invented in 1999"], ctx, supports) == 0.5
    assert lab.faithfulness([], ctx, supports) == 1.0


def test_12_mcq(lab):
    logps = [[-6.0, -9.0, -5.5], [-2.0, -4.0, -3.0]]
    lengths = [[2, 6, 1], [1, 4, 3]]
    assert lab.mcq_predictions(logps, lengths, normalize=False) == [2, 0], "raw likelihood favours short options"
    assert lab.mcq_predictions(logps, lengths, normalize=True) == [1, 1], "per-token: -1.5 beats -3 and -5.5"


def test_13_regression_gate(lab):
    base = {"faithfulness": 0.91, "rougeL": 0.41, "safety_pass": 0.99}
    ok, failed = lab.regression_gate(base, {"faithfulness": 0.90, "rougeL": 0.35, "safety_pass": 0.99},
                                     {"faithfulness": 0.02, "rougeL": 0.03, "safety_pass": 0.0})
    assert (ok, failed) == (False, ["rougeL"])
    assert lab.regression_gate(base, dict(base, safety_pass=0.98), {"safety_pass": 0.0}) == (False, ["safety_pass"])
    assert lab.regression_gate(base, base, {"rougeL": 0.0}) == (True, [])
