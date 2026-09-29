"""Lab 02 — reference solutions."""

import json
import re
from collections import Counter

import numpy as np


def build_prompt(instruction, examples, query, input_label="Input", output_label="Output"):
    blocks = [instruction]
    blocks += [f"{input_label}: {x}\n{output_label}: {y}" for x, y in examples]
    blocks.append(f"{input_label}: {query}\n{output_label}:")
    return "\n\n".join(blocks)


def to_chatml(messages, add_generation_prompt=True):
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits, temperature):
    if temperature == 0:
        probs = np.zeros_like(logits, dtype=float)
        probs[np.argmax(logits)] = 1.0
        return probs
    return _softmax(logits / temperature)


def top_k_filter(logits, k):
    out = np.full_like(logits, -np.inf, dtype=float)
    keep = np.argsort(logits)[-k:]
    out[keep] = logits[keep]
    return out


def top_p_filter(logits, p):
    probs = _softmax(logits)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    n_keep = int(np.searchsorted(cumulative, p) + 1)  # first index where cum >= p, inclusive
    out = np.full_like(logits, -np.inf, dtype=float)
    keep = order[:n_keep]
    out[keep] = logits[keep]
    return out


def apply_repetition_penalty(logits, generated_ids, penalty):
    out = logits.astype(float).copy()
    for t in set(generated_ids):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


def constrain_to_choices(logits, allowed_ids):
    out = np.full_like(logits, -np.inf, dtype=float)
    out[allowed_ids] = logits[allowed_ids]
    return out


# Capture lazily up to a sentence-ending period (". " or "." at end) or the end of the line,
# so "The answer is 3.5." gives "3.5".
_ANSWER = re.compile(r"(?:the answer is|answer:)[ \t]*(.*?)[ \t]*(?:\.(?=\s|$)|$)", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(text):
    matches = [m.group(1) for m in _ANSWER.finditer(text) if m.group(1)]
    return matches[-1] if matches else None


def self_consistency(completions):
    answers = [a for a in map(extract_final_answer, completions) if a is not None]
    if not answers:
        return None, 0.0
    counts = Counter(answers)  # Counter preserves first-seen order, so ties go to the earliest
    winner, votes = counts.most_common(1)[0]
    return winner, votes / len(answers)


def select_examples(query_vec, example_vecs, k):
    q = query_vec / np.linalg.norm(query_vec)
    e = example_vecs / np.linalg.norm(example_vecs, axis=1, keepdims=True)
    sims = e @ q
    return [int(i) for i in np.argsort(-sims, kind="stable")[:k]]


def parse_json_output(text, required_keys=()):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            missing = [k for k in required_keys if k not in obj]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return obj
    raise ValueError("no JSON object found")
