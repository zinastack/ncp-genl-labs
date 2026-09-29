"""Lab 10 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import re
from collections import defaultdict

REQUIRED_CARD_SECTIONS = (
    "intended_use", "out_of_scope_use", "training_data", "evaluation",
    "bias_and_fairness", "limitations", "license",
)

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"you are now (in )?(dan|developer mode)",
    r"reveal (your|the) (system prompt|hidden instructions)",
    r"pretend (that )?you have no (rules|restrictions|guidelines)",
]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")


def selection_rates(y_pred, groups):
    counts = defaultdict(lambda: [0, 0])  # group → [approved, total]
    for p, g in zip(y_pred, groups):
        counts[g][0] += p
        counts[g][1] += 1
    return {g: pos / n for g, (pos, n) in counts.items()}  # P(ŷ = 1 | group)


def demographic_parity_difference(y_pred, groups):
    rates = selection_rates(y_pred, groups).values()
    return max(rates) - min(rates)  # 0 = every group selected at the same rate


def disparate_impact_ratio(y_pred, groups):
    rates = selection_rates(y_pred, groups).values()
    # < 0.8 fails the four-fifths rule. Nobody selected anywhere → no disparity to measure.
    return 1.0 if max(rates) == 0 else min(rates) / max(rates)


def equalized_odds_difference(y_true, y_pred, groups):
    tpr, fpr = {}, {}
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        pos = [i for i in idx if y_true[i] == 1]
        neg = [i for i in idx if y_true[i] == 0]
        # Skip a rate when the group has no positives (or negatives): it would be 0/0.
        if pos:
            tpr[g] = sum(y_pred[i] for i in pos) / len(pos)
        if neg:
            fpr[g] = sum(y_pred[i] for i in neg) / len(neg)
    gap = lambda d: max(d.values()) - min(d.values()) if d else 0.0
    return max(gap(tpr), gap(fpr))  # are the ERRORS equally distributed across groups?


def counterfactual_prompts(template, attribute_values):
    # Identical prompts except for the attribute, so any output difference is caused by it.
    return {v: template.format(group=v) for v in attribute_values}


def counterfactual_gap(scores):
    hi = max(scores, key=scores.get)
    lo = min(scores, key=scores.get)
    return scores[hi] - scores[lo], hi, lo


def detect_prompt_injection(text):
    # Cheap first layer only: paraphrases and encodings evade regexes (use classifier rails too).
    return [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]


def _mask_pii(text):
    return _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", text))


class GuardrailedLLM:
    REFUSAL = "I can't help with that request."

    def __init__(self, llm, blocked_topics=(), toxic_terms=()):
        self.llm = llm
        self.blocked_topics = [t.lower() for t in blocked_topics]
        self.toxic_terms = [t.lower() for t in toxic_terms]

    def __call__(self, user_text):
        triggered = []  # audit trail: only rails that actually changed or blocked something
        # INPUT RAILS: blocking ones return early, so the LLM is never called.
        if detect_prompt_injection(user_text):
            return self.REFUSAL, ["injection"]
        if any(t in user_text.lower() for t in self.blocked_topics):
            return self.REFUSAL, ["blocked_topic"]

        masked = _mask_pii(user_text)  # the user's PII never leaves the system
        if masked != user_text:
            triggered.append("pii_input")
        response = self.llm(masked)

        # OUTPUT RAILS: catch what the input rails can't (the model's own output).
        clean = _mask_pii(response)
        if clean != response:
            triggered.append("pii_output")
        if any(t in clean.lower() for t in self.toxic_terms):
            triggered.append("toxic_output")
            return self.REFUSAL, triggered
        return clean, triggered


def attack_success_rate(results):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(bool(r["succeeded"]))
    out = {c: sum(v) / len(v) for c, v in by_cat.items()}
    out["overall"] = sum(bool(r["succeeded"]) for r in results) / len(results)
    return out


def model_card_gaps(card):
    # Missing OR blank (whitespace-only) sections both count as undocumented.
    return [s for s in REQUIRED_CARD_SECTIONS if not str(card.get(s, "")).strip()]
