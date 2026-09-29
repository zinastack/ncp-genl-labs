"""Lab 03 — reference solutions."""

import hashlib
import re
import unicodedata
from collections import Counter

import numpy as np

MERSENNE_P = (1 << 61) - 1


def hash64(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def exact_dedup(docs):
    seen, out = set(), []
    for doc in docs:
        h = hash64(normalize_text(doc))
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def shingles(text, n=3):
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def minhash_signature(shingle_set, num_perm=128, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_P, num_perm)
    b = rng.integers(0, MERSENNE_P, num_perm)
    hashes = [hash64(s) for s in shingle_set]
    sig = [min((int(ai) * h + int(bi)) % MERSENNE_P for h in hashes) for ai, bi in zip(a, b)]
    return np.array(sig, dtype=np.uint64)


def estimate_jaccard(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))


def quality_issues(text):
    words = text.split()
    n = len(words)
    issues = []
    if not 50 <= n <= 100_000:
        issues.append("word_count")
    if n == 0 or not 3 <= sum(map(len, words)) / n <= 10:
        issues.append("mean_word_length")
    symbols = text.count("#") + text.count("...") + text.count("…")
    if n == 0 or symbols / n > 0.1:
        issues.append("symbol_ratio")
    if n == 0 or sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:
        issues.append("alpha_fraction")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and sum(l.endswith(("...", "…")) for l in lines) / len(lines) > 0.3:
        issues.append("ellipsis_lines")
    return issues


_PII = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("[PHONE]", re.compile(r"(?:\+1[\s.-]?)?(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b")),
]


def redact_pii(text):
    for placeholder, pattern in _PII:
        text = pattern.sub(placeholder, text)
    return text


def _merge_word(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(word_freqs, num_merges):
    vocab = {word: list(word) for word in word_freqs}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in word_freqs.items():
            syms = vocab[word]
            for pair in zip(syms, syms[1:]):
                pairs[pair] += freq
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        vocab = {w: _merge_word(s, best) for w, s in vocab.items()}
    return merges


def bpe_encode(word, merges):
    symbols = list(word)
    for pair in merges:
        symbols = _merge_word(symbols, pair)
    return symbols


def pack_sequences(seqs, max_len, eos_id):
    packs, current = [], []
    for seq in seqs:
        item = (list(seq) + [eos_id])[:max_len]
        if current and len(current) + len(item) > max_len:
            packs.append(current)
            current = []
        current = current + item
    if current:
        packs.append(current)
    return packs


def padding_efficiency(lengths, batch_size):
    real = total = 0
    for i in range(0, len(lengths), batch_size):
        batch = lengths[i : i + batch_size]
        real += sum(batch)
        total += max(batch) * len(batch)
    return real / total


def split_by_group(records, group_key, val_fraction):
    train, val = [], []
    for r in records:
        bucket = hash64(str(r[group_key])) % 10_000
        (val if bucket < val_fraction * 10_000 else train).append(r)
    return train, val


def resize_embeddings(embeddings, n_new):
    mean = embeddings.mean(axis=0, keepdims=True)
    return np.vstack([embeddings, np.repeat(mean, n_new, axis=0)])
