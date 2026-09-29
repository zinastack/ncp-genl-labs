import numpy as np
import pytest


def test_1_normalize(lab):
    raw = "  Ｈｅｌｌｏ\u0007   world\t!\n\n\n\n  next   line  "
    assert lab.normalize_text(raw) == "Hello world !\n\nnext line"
    assert lab.normalize_text("ﬁne") == "fine", "NFKC folds ligatures"
    assert lab.normalize_text("GPU") == "GPU", "don't lowercase"


def test_2_exact_dedup(lab):
    docs = ["Hello  world", "hello world", "Hello world", "Other doc"]
    assert lab.exact_dedup(docs) == ["Hello  world", "hello world", "Other doc"]


def test_3_minhash(lab):
    base = ("nvidia nemo curator removes near duplicate documents from large web crawls "
            "using minhash signatures and locality sensitive hashing on gpus quickly").split()
    near = base.copy()
    near[5] = "almost"
    other = "the quick brown fox jumps over the lazy dog while the cat sleeps in the warm sun".split()

    a, b, c = (lab.shingles(" ".join(w)) for w in (base, near, other))
    true_j = len(a & b) / len(a | b)
    sa, sb, sc = (lab.minhash_signature(s, num_perm=256) for s in (a, b, c))
    assert lab.estimate_jaccard(sa, sa) == 1.0
    assert abs(lab.estimate_jaccard(sa, sb) - true_j) < 0.1
    assert lab.estimate_jaccard(sa, sc) < 0.1
    assert lab.shingles("two words", 3) == {"two words"}


GOOD = " ".join(["Tensor parallelism splits individual weight matrices across GPUs within a node."] * 8)


def test_4_quality(lab):
    assert lab.quality_issues(GOOD) == []
    assert lab.quality_issues("too short") == ["word_count"]
    hashtags = " ".join(["#gpu #ai #llm #deal"] * 20)
    assert "symbol_ratio" in lab.quality_issues(hashtags)
    numbers = " ".join(["123 456 789 000 and"] * 20)
    assert "alpha_fraction" in lab.quality_issues(numbers)
    teaser = "\n".join(["Click here to read the full story about the new release..."] * 10)
    assert "ellipsis_lines" in lab.quality_issues(teaser)


def test_5_pii(lab):
    text = ("Contact jane.doe+ml@example.co.uk or 555-123-4567 / (555) 987-6543 / +1 555 222 3333. "
            "SSN 123-45-6789, server 10.0.0.12, version 3.14.")
    assert lab.redact_pii(text) == (
        "Contact [EMAIL] or [PHONE] / [PHONE] / [PHONE]. SSN [SSN], server [IP], version 3.14."
    )


WORDS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


def test_6_bpe(lab):
    merges = lab.train_bpe(WORDS, 4)
    assert merges[:3] == [("e", "s"), ("es", "t"), ("l", "o")]
    assert len(merges) == 4
    assert lab.bpe_encode("lowest", merges) == ["low", "est"]
    assert lab.bpe_encode("xyz", merges) == ["x", "y", "z"]
    assert len(lab.train_bpe({"ab": 1}, 10)) == 1, "stop when no pairs remain"


def test_7_packing(lab):
    packs = lab.pack_sequences([[1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11, 12], [13]], max_len=7, eos_id=0)
    assert packs == [[1, 2, 3, 0, 4, 5, 0], [6, 7, 8, 9, 10, 11, 12], [13, 0]]
    lengths = [5, 100, 6, 98, 4, 101, 7, 99]
    assert lab.padding_efficiency(lengths, 2) == pytest.approx(420 / 796)
    assert lab.padding_efficiency(sorted(lengths), 2) > 0.95, "length bucketing removes most padding"


def test_8_split_by_group(lab):
    records = [{"patient": p, "note": i} for p in range(200) for i in range(3)]
    train, val = lab.split_by_group(records, "patient", 0.2)
    tr, va = {r["patient"] for r in train}, {r["patient"] for r in val}
    assert not tr & va, "a patient must not appear in both splits"
    assert 0.1 < len(va) / 200 < 0.3
    assert lab.split_by_group(records, "patient", 0.2) == (train, val), "must be deterministic"


def test_9_resize_embeddings(lab):
    e = np.arange(12, dtype=float).reshape(4, 3)
    out = lab.resize_embeddings(e, 2)
    assert out.shape == (6, 3)
    np.testing.assert_array_equal(out[:4], e)
    np.testing.assert_allclose(out[4], e.mean(0))


def test_10_lsh(lab):
    docs = ["the quick brown fox jumps over the lazy dog near the river bank today",
            "the quick brown fox jumps over the lazy dog near the river bank now",
            "completely different text about gpus and tensor cores in data centers"]
    sigs = [lab.minhash_signature(lab.shingles(d), 64) for d in docs]
    assert lab.lsh_candidate_pairs(sigs, bands=16) == {(0, 1)}
    same = np.arange(8, dtype=np.uint64)
    assert lab.lsh_candidate_pairs([same, same.copy(), same + 100], bands=4) == {(0, 1)}
    assert lab.lsh_candidate_probability(0.7, 16, 4) == pytest.approx(0.988, abs=1e-3)
    assert lab.lsh_candidate_probability(0.3, 16, 4) == pytest.approx(0.122, abs=1e-3)
    assert lab.lsh_candidate_probability(0.7, 8, 8) < lab.lsh_candidate_probability(0.7, 16, 4), \
        "more rows per band = stricter threshold"


def test_11_fertility_and_bytes(lab):
    assert lab.utf8_byte_tokens("GPU") == [71, 80, 85]
    assert len(lab.utf8_byte_tokens("중앙")) == 6, "each Hangul syllable is 3 UTF-8 bytes"
    assert all(0 <= b < 256 for b in lab.utf8_byte_tokens("naïve 中文 🚀")), "any text fits in 256 base tokens"
    en = ["The central bank kept interest rates unchanged"]
    ko = ["중앙은행은 기준금리를 동결했다"]
    assert lab.fertility(en, str.split) == 1.0
    assert lab.fertility(en, lab.utf8_byte_tokens) == pytest.approx(46 / 7)
    assert lab.fertility(ko, lab.utf8_byte_tokens) > 2 * lab.fertility(en, lab.utf8_byte_tokens)


def test_12_contamination(lab):
    train = ["Question: What is the capital of France? Answer: Paris is the capital of France."]
    test = ["what is the capital of france answer paris", "How many legs does a spider have?"]
    assert lab.contaminated_items(train, test, n=5) == [0], "case and punctuation don't hide a leak"
    assert lab.contaminated_items(train, test, n=13) == [], "the test item is shorter than 13 words"


def test_13_pad_batch(lab):
    ids, mask = lab.pad_batch([[5, 6, 7], [8]], pad_id=0, side="right")
    np.testing.assert_array_equal(ids, [[5, 6, 7], [8, 0, 0]])
    np.testing.assert_array_equal(mask, [[1, 1, 1], [1, 0, 0]])
    ids, mask = lab.pad_batch([[5, 6, 7], [8]], pad_id=0, side="left")
    np.testing.assert_array_equal(ids, [[5, 6, 7], [0, 0, 8]])
    np.testing.assert_array_equal(mask, [[1, 1, 1], [0, 0, 1]])
    assert list(ids[:, -1]) == [7, 8], "left padding: the last column holds every row's real last token"


def test_14_blend_plan(lab):
    plan = lab.blend_plan({"web": 1e12, "legal": 2e9, "instructions": 5e8},
                          {"web": 7, "legal": 2, "instructions": 1}, total_tokens=2e10)
    assert plan["web"] == pytest.approx((1.4e10, 0.014))
    assert plan["legal"] == pytest.approx((4e9, 2.0)), "legal data is seen twice (upsampled)"
    assert plan["instructions"] == pytest.approx((2e9, 4.0))
