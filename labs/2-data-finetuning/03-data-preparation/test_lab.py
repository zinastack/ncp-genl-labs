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
