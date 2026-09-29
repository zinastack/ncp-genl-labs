"""Lab 03 — Data Preparation. Fill in every TODO, then run:

    pytest labs/2-data-finetuning/03-data-preparation
"""

import hashlib
import re
import unicodedata
from collections import Counter

import numpy as np

MERSENNE_P = (1 << 61) - 1


def hash64(s: str) -> int:
    """Stable 64-bit hash (Python's built-in hash() is salted per process — never use it for dedup)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


# 1 ─────────────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Apply Unicode NFKC, drop control characters (category 'Cc') except newline and tab,
    collapse runs of spaces/tabs to one space, strip each line, collapse 3+ newlines to 2,
    and strip the whole string. Do NOT lowercase.
    """
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def exact_dedup(docs: list[str]) -> list[str]:
    """Keep the first occurrence of each document, comparing *normalised* text by hash.
    Return the original (un-normalised) documents that survive, in order.
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def shingles(text: str, n: int = 3) -> set[str]:
    """Set of lowercase word n-grams joined by single spaces. Fewer than n words → {whole text}."""
    raise NotImplementedError


def minhash_signature(shingle_set: set[str], num_perm: int = 128, seed: int = 0) -> np.ndarray:
    """MinHash signature: for i in range(num_perm) with hash functions
    h_i(x) = (a_i * hash64(x) + b_i) mod MERSENNE_P, take min over the set.

    Draw a (from 1..P-1) and b (from 0..P-1) with np.random.default_rng(seed).integers.
    Do the arithmetic with Python ints (it overflows int64). Return a uint64 array.
    """
    raise NotImplementedError


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of positions where the two signatures agree ≈ Jaccard(A, B)."""
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def quality_issues(text: str) -> list[str]:
    """Gopher-style heuristic filters. Return the names of the rules that FAIL (empty = keep):

    "word_count"        — number of whitespace words not in [50, 100_000]
    "mean_word_length"  — mean word length not in [3, 10]
    "symbol_ratio"      — (count of '#' + count of '...' or '…') / words > 0.1
    "alpha_fraction"    — fraction of words containing an alphabetic char < 0.8
    "ellipsis_lines"    — fraction of non-empty lines ending in '...' or '…' > 0.3
    Return names in the order listed.
    """
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def redact_pii(text: str) -> str:
    """Replace PII with placeholders: [EMAIL], [SSN], [IP], [PHONE].

    - emails:  name@domain.tld
    - SSN:     123-45-6789
    - IPv4:    four dot-separated 1–3 digit numbers
    - phones:  US numbers like 555-123-4567, (555) 123-4567, 555.123.4567, +1 555 123 4567
    Order matters: redact SSNs and IPs before phones.
    """
    raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def train_bpe(word_freqs: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Learn BPE merges. Each word starts as a list of characters. Repeat num_merges times:
    count adjacent symbol pairs weighted by word frequency, pick the most frequent pair
    (ties → lexicographically smallest pair), merge it everywhere. Stop early if no pairs remain.
    Return the merges in the order learned.
    """
    raise NotImplementedError


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenise a word by applying the learned merges in order (priority = learn order)."""
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def pack_sequences(seqs: list[list[int]], max_len: int, eos_id: int) -> list[list[int]]:
    """Greedy in-order packing. Append eos_id to each sequence, truncating seq+[eos] to
    max_len if longer. Add it to the current pack if it fits within max_len, else start a new
    pack. Return the packs (no padding).
    """
    raise NotImplementedError


def padding_efficiency(lengths: list[int], batch_size: int) -> float:
    """Batch `lengths` in the given order into batches of `batch_size`, pad each batch to its
    longest sequence, and return real_tokens / total_tokens_including_padding.
    """
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def split_by_group(
    records: list[dict], group_key: str, val_fraction: float,
) -> tuple[list[dict], list[dict]]:
    """Deterministic split with no group in both sets. A record goes to validation when
    hash64(str(record[group_key])) % 10_000 < val_fraction * 10_000. Return (train, val).
    """
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
def resize_embeddings(embeddings: np.ndarray, n_new: int) -> np.ndarray:
    """Append n_new rows for new tokens, each initialised to the mean of existing rows."""
    raise NotImplementedError
