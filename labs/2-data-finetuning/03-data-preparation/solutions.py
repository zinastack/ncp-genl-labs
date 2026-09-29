"""Lab 03 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable

import numpy as np

MERSENNE_P = (1 << 61) - 1  # a large prime: modular hashing mod a prime spreads values evenly


def hash64(s: str) -> int:
    # Stable across processes and machines (Python's hash() is randomly salted per process).
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # Ｈ→H, ﬁ→fi: one canonical form per character
    # Drop invisible control characters, but keep \n and \t (tab is category Cc too, and we want
    # it collapsed to a space in the next step, not deleted).
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)  # keep paragraph breaks, drop empty runs
    return text.strip()  # no lowercasing: case carries meaning


def exact_dedup(docs: list[str]) -> list[str]:
    seen, out = set(), []
    for doc in docs:
        h = hash64(normalize_text(doc))  # compare normalised text…
        if h not in seen:
            seen.add(h)
            out.append(doc)  # …but keep the original document
    return out


def shingles(text: str, n: int = 3) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}  # overlapping word n-grams


def minhash_signature(shingle_set: set[str], num_perm: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)  # same seed → same hash family for every document
    a = rng.integers(1, MERSENNE_P, num_perm)
    b = rng.integers(0, MERSENNE_P, num_perm)
    hashes = [hash64(s) for s in shingle_set]
    # For each hash function h_i(x) = (a_i·x + b_i) mod P, keep the MIN over the set.
    # P(min h_i(A) == min h_i(B)) = Jaccard(A, B). Python ints avoid int64 overflow.
    sig = [min((int(ai) * h + int(bi)) % MERSENNE_P for h in hashes) for ai, bi in zip(a, b)]
    return np.array(sig, dtype=np.uint64)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))  # fraction of agreeing minima ≈ Jaccard


def quality_issues(text: str) -> list[str]:
    words = text.split()
    n = len(words)
    issues = []  # return reasons, not just keep/drop, so filtering is explainable
    if not 50 <= n <= 100_000:
        issues.append("word_count")
    if n == 0 or not 3 <= sum(map(len, words)) / n <= 10:
        issues.append("mean_word_length")
    symbols = text.count("#") + text.count("...") + text.count("…")
    if n == 0 or symbols / n > 0.1:
        issues.append("symbol_ratio")  # hashtag spam, teaser text
    if n == 0 or sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:
        issues.append("alpha_fraction")  # number tables, logs
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and sum(l.endswith(("...", "…")) for l in lines) / len(lines) > 0.3:
        issues.append("ellipsis_lines")  # "read more..." pages
    return issues


# Most specific patterns first: SSNs and IPs are digits + separators that a loose phone
# pattern could otherwise partially match.
_PII = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("[PHONE]", re.compile(r"(?:\+1[\s.-]?)?(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b")),
]


def redact_pii(text: str) -> str:
    for placeholder, pattern in _PII:
        text = pattern.sub(placeholder, text)
    return text


def _merge_word(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    # Replace every adjacent occurrence of `pair` in the symbol list with the merged symbol.
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(word_freqs: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    vocab = {word: list(word) for word in word_freqs}  # every word starts as characters
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in word_freqs.items():
            syms = vocab[word]
            for pair in zip(syms, syms[1:]):
                pairs[pair] += freq  # a pair in a frequent word counts that many times
        if not pairs:
            break  # every word is already a single symbol
        best = min(pairs, key=lambda p: (-pairs[p], p))  # most frequent; ties → alphabetical
        merges.append(best)
        vocab = {w: _merge_word(s, best) for w, s in vocab.items()}
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word)
    for pair in merges:  # learned order = priority order
        symbols = _merge_word(symbols, pair)
    return symbols


def pack_sequences(seqs: list[list[int]], max_len: int, eos_id: int) -> list[list[int]]:
    packs, current = [], []
    for seq in seqs:
        item = (list(seq) + [eos_id])[:max_len]  # EOS separates examples; truncate oversize ones
        if current and len(current) + len(item) > max_len:  # doesn't fit → close this row
            packs.append(current)
            current = []
        current = current + item
    if current:
        packs.append(current)
    return packs


def padding_efficiency(lengths: list[int], batch_size: int) -> float:
    real = total = 0
    for i in range(0, len(lengths), batch_size):
        batch = lengths[i : i + batch_size]
        real += sum(batch)
        total += max(batch) * len(batch)  # every row is padded to the batch's longest
    return real / total


def split_by_group(
    records: list[dict], group_key: str, val_fraction: float,
) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for r in records:
        # A stable hash of the GROUP decides the side, so all records of a group stay together,
        # and the split is identical on every run and machine.
        bucket = hash64(str(r[group_key])) % 10_000
        (val if bucket < val_fraction * 10_000 else train).append(r)
    return train, val


def resize_embeddings(embeddings: np.ndarray, n_new: int) -> np.ndarray:
    # New token rows start at the mean embedding: a neutral start (random rows can produce
    # extreme logits). They still need training.
    mean = embeddings.mean(axis=0, keepdims=True)
    return np.vstack([embeddings, np.repeat(mean, n_new, axis=0)])


def lsh_candidate_pairs(signatures: list[np.ndarray], bands: int) -> set[tuple[int, int]]:
    # Cut every signature into `bands` equal slices. Two documents become candidates if ANY
    # slice matches exactly, so we only compare documents that share a bucket, never all pairs.
    rows = len(signatures[0]) // bands
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for doc, sig in enumerate(signatures):
        for b in range(bands):
            key = (b, sig[b * rows : (b + 1) * rows].tobytes())  # band index + band contents
            buckets.setdefault(key, []).append(doc)
    return {(a, c) for docs in buckets.values() for i, a in enumerate(docs) for c in docs[i + 1 :]}


def lsh_candidate_probability(jaccard: float, bands: int, rows: int) -> float:
    # One band matches with probability J^rows (every row must agree); a pair is a candidate
    # unless ALL bands miss: 1 − (1 − J^r)^b. This S-curve is the dedup threshold.
    return 1 - (1 - jaccard**rows) ** bands


def fertility(texts: list[str], tokenize: Callable[[str], list]) -> float:
    # Tokens per whitespace word, over the whole set: ~1.3 is typical for English with a good
    # tokenizer; 3+ means text is fragmented (longer sequences, higher cost, less context).
    words = sum(len(t.split()) for t in texts)
    return sum(len(tokenize(t)) for t in texts) / words


def utf8_byte_tokens(text: str) -> list[int]:
    # Byte-level tokenizers start from these 256 possible values, so ANY string is representable
    # and there is never an <unk> token. Non-Latin characters take 2–4 bytes each.
    return list(text.encode("utf-8"))


def contaminated_items(train_docs: list[str], test_items: list[str], n: int = 13) -> list[int]:
    def grams(text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"\w+", text.lower())  # compare words, ignoring case and punctuation
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    seen: set[tuple[str, ...]] = set()
    for doc in train_docs:
        seen |= grams(doc)
    # A test item that shares any long n-gram with the training data may have been memorised.
    return [i for i, item in enumerate(test_items) if grams(item) & seen]


def pad_batch(seqs: list[list[int]], pad_id: int, side: str = "right") -> tuple[np.ndarray, np.ndarray]:
    width = max(map(len, seqs))
    ids = np.full((len(seqs), width), pad_id)
    mask = np.zeros((len(seqs), width), dtype=int)
    for row, seq in enumerate(seqs):
        # Right padding (training, encoders): tokens first. Left padding (batched generation with
        # decoders): tokens last, so every row's final position is its real last token.
        cols = slice(0, len(seq)) if side == "right" else slice(width - len(seq), width)
        ids[row, cols] = seq
        mask[row, cols] = 1
    return ids, mask


def blend_plan(dataset_tokens: dict[str, float], weights: dict[str, float], total_tokens: float) -> dict[str, tuple[float, float]]:
    # For each source: how many tokens to draw (weight × budget) and how many passes (epochs)
    # over it that means. Epochs > 1 = upsampling a scarce source; < 1 = using a slice of a big one.
    total_weight = sum(weights.values())
    plan = {}
    for name, size in dataset_tokens.items():
        take = total_tokens * weights[name] / total_weight
        plan[name] = (take, take / size)
    return plan
