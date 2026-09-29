"""Lab 10 — reference solutions."""

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
    counts = defaultdict(lambda: [0, 0])
    for p, g in zip(y_pred, groups):
        counts[g][0] += p
        counts[g][1] += 1
    return {g: pos / n for g, (pos, n) in counts.items()}


def demographic_parity_difference(y_pred, groups):
    rates = selection_rates(y_pred, groups).values()
    return max(rates) - min(rates)


def disparate_impact_ratio(y_pred, groups):
    rates = selection_rates(y_pred, groups).values()
    return 1.0 if max(rates) == 0 else min(rates) / max(rates)


def equalized_odds_difference(y_true, y_pred, groups):
    tpr, fpr = {}, {}
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        pos = [i for i in idx if y_true[i] == 1]
        neg = [i for i in idx if y_true[i] == 0]
        if pos:
            tpr[g] = sum(y_pred[i] for i in pos) / len(pos)
        if neg:
            fpr[g] = sum(y_pred[i] for i in neg) / len(neg)
    gap = lambda d: max(d.values()) - min(d.values()) if d else 0.0
    return max(gap(tpr), gap(fpr))


def counterfactual_prompts(template, attribute_values):
    return {v: template.format(group=v) for v in attribute_values}


def counterfactual_gap(scores):
    hi = max(scores, key=scores.get)
    lo = min(scores, key=scores.get)
    return scores[hi] - scores[lo], hi, lo


def detect_prompt_injection(text):
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
        triggered = []
        if detect_prompt_injection(user_text):
            return self.REFUSAL, ["injection"]
        if any(t in user_text.lower() for t in self.blocked_topics):
            return self.REFUSAL, ["blocked_topic"]

        masked = _mask_pii(user_text)
        if masked != user_text:
            triggered.append("pii_input")
        response = self.llm(masked)

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
    return [s for s in REQUIRED_CARD_SECTIONS if not str(card.get(s, "")).strip()]
