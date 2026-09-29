"""Lab 10 — Safety, Ethics & Compliance. Fill in every TODO, then run:

    pytest labs/5-evaluation-responsible-ai/10-safety-ethics
"""

import re
from collections import defaultdict
from collections.abc import Callable, Sequence

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


# 1 ─────────────────────────────────────────────────────────────────────────────
def selection_rates(y_pred: list[int], groups: list[str]) -> dict[str, float]:
    """P(ŷ = 1 | group) for each group."""
    raise NotImplementedError


def demographic_parity_difference(y_pred: list[int], groups: list[str]) -> float:
    """max selection rate − min selection rate across groups."""
    raise NotImplementedError


def disparate_impact_ratio(y_pred: list[int], groups: list[str]) -> float:
    """min selection rate / max selection rate (1.0 if max is 0). < 0.8 fails the four-fifths rule."""
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def equalized_odds_difference(y_true: list[int], y_pred: list[int], groups: list[str]) -> float:
    """max(TPR gap, FPR gap) across groups, where gap = max − min over groups.
    A group with no positives (or no negatives) is skipped for that rate.
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def counterfactual_prompts(template: str, attribute_values: list[str]) -> dict[str, str]:
    """Fill the {group} placeholder in `template` with each value → {value: prompt}."""
    raise NotImplementedError


def counterfactual_gap(scores: dict[str, float]) -> tuple[float, str, str]:
    """Given a score per attribute value (e.g. sentiment of the LLM's answer, or P(hire)),
    return (max − min, value_with_max, value_with_min).
    """
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def detect_prompt_injection(text: str) -> list[str]:
    """Return the INJECTION_PATTERNS that match `text` (case-insensitive), in list order."""
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
class GuardrailedLLM:
    """Wrap any `llm(prompt) -> str` with rails.

    __call__(user_text) returns (response, triggered) where `triggered` lists rail names in order:
      input rails (stop at the first that blocks):
        "injection"     → detect_prompt_injection finds a pattern → response = REFUSAL
        "blocked_topic" → any of `blocked_topics` appears (case-insensitive) → response = REFUSAL
      if not blocked:
        "pii_input"     → mask PII in the user text before sending it to the LLM (does NOT block)
        call the LLM
      output rails:
        "pii_output"    → mask PII in the LLM response
        "toxic_output"  → any of `toxic_terms` in the response (case-insensitive) → response = REFUSAL
    PII masking: emails → [EMAIL], US phone numbers like 555-123-4567 → [PHONE].
    A rail is listed only if it actually changed/blocked something.
    """

    REFUSAL = "I can't help with that request."

    def __init__(
        self, llm: Callable[[str], str], blocked_topics: Sequence[str] = (),
        toxic_terms: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError

    def __call__(self, user_text: str) -> tuple[str, list[str]]:
        raise NotImplementedError


# 6 ─────────────────────────────────────────────────────────────────────────────
def attack_success_rate(results: list[dict]) -> dict[str, float]:
    """results: [{"category": "jailbreak", "succeeded": True}, ...].
    Return {category: success rate} plus "overall".
    """
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def model_card_gaps(card: dict) -> list[str]:
    """REQUIRED_CARD_SECTIONS missing from `card` or present but empty/whitespace, in order."""
    raise NotImplementedError
