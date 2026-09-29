import numpy as np
import pytest


def test_1_build_prompt_few_shot(lab):
    p = lab.build_prompt(
        "Classify the sentiment of the financial headline as positive, negative or neutral.",
        [("Shares soar after record earnings", "positive"), ("Company misses revenue guidance", "negative")],
        "Fed holds rates steady",
    )
    assert p == (
        "Classify the sentiment of the financial headline as positive, negative or neutral.\n\n"
        "Input: Shares soar after record earnings\nOutput: positive\n\n"
        "Input: Company misses revenue guidance\nOutput: negative\n\n"
        "Input: Fed holds rates steady\nOutput:"
    )


def test_1_build_prompt_zero_shot(lab):
    p = lab.build_prompt("Translate to French.", [], "Good morning", "English", "French")
    assert p == "Translate to French.\n\nEnglish: Good morning\nFrench:"


def test_2_chatml(lab):
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    assert lab.to_chatml(msgs) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert lab.to_chatml(msgs, add_generation_prompt=False).endswith("Hi<|im_end|>\n")


LOGITS = np.array([2.0, 1.0, 0.5, -1.0, 3.0])


def test_3_temperature(lab):
    p1 = lab.apply_temperature(LOGITS, 1.0)
    np.testing.assert_allclose(p1.sum(), 1.0)
    cold, hot = lab.apply_temperature(LOGITS, 0.2), lab.apply_temperature(LOGITS, 5.0)
    assert cold.max() > p1.max() > hot.max(), "low T sharpens, high T flattens"
    np.testing.assert_array_equal(lab.apply_temperature(LOGITS, 0), [0, 0, 0, 0, 1])


def test_3_top_k(lab):
    out = lab.top_k_filter(LOGITS, 2)
    assert np.isfinite(out).sum() == 2 and out[4] == 3.0 and out[0] == 2.0


def test_3_top_p(lab):
    probs = np.exp(LOGITS) / np.exp(LOGITS).sum()  # ≈ [.23 .08 .05 .01 .63]
    assert probs[4] < 0.7 < probs[4] + probs[0]
    out = lab.top_p_filter(LOGITS, 0.7)
    assert set(np.flatnonzero(np.isfinite(out))) == {0, 4}
    assert np.isfinite(lab.top_p_filter(LOGITS, 0.01)).sum() == 1, "always keep the top token"
    assert np.isfinite(lab.top_p_filter(LOGITS, 1.0)).sum() == 5


def test_4_repetition_penalty(lab):
    out = lab.apply_repetition_penalty(LOGITS, [0, 3, 0], 2.0)
    np.testing.assert_allclose(out, [1.0, 1.0, 0.5, -2.0, 3.0])
    assert LOGITS[0] == 2.0, "don't mutate the input"


def test_5_constrain(lab):
    out = lab.constrain_to_choices(LOGITS, [1, 2])
    assert np.argmax(out) == 1 and np.isfinite(out).sum() == 2


def test_6_extract_final_answer(lab):
    assert lab.extract_final_answer("3 + 4 = 7, then 7 * 2 = 14. The answer is 14.") == "14"
    assert lab.extract_final_answer("Reasoning...\nAnswer: negative") == "negative"
    assert lab.extract_final_answer("First I guessed the answer is 3. Recheck: Answer: 5.") == "5"
    assert lab.extract_final_answer("So the answer is 3.5.") == "3.5"
    assert lab.extract_final_answer("I am not sure.") is None


def test_6_self_consistency(lab):
    samples = [
        "12 apples minus 5 is 7. The answer is 7.",
        "12 - 5 = 8? no, 7. Answer: 7",
        "The answer is 8.",
        "hmm",
        "The answer is 7",
    ]
    ans, agreement = lab.self_consistency(samples)
    assert ans == "7" and agreement == pytest.approx(3 / 4)
    assert lab.self_consistency(["no idea"]) == (None, 0.0)


def test_7_select_examples(lab):
    q = np.array([1.0, 0.0])
    ex = np.array([[0.0, 1.0], [10.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    assert lab.select_examples(q, ex, 2) == [1, 2], "cosine, not dot product: magnitude must not matter"


def test_8_parse_json(lab):
    reply = 'Sure! Here you go:\n```json\n{"label": "negative", "confidence": 0.82, "why": "rate {hike}"}\n```'
    assert lab.parse_json_output(reply, ("label",)) == {"label": "negative", "confidence": 0.82, "why": "rate {hike}"}
    with pytest.raises(ValueError):
        lab.parse_json_output('{"confidence": 0.5}', ("label",))
    with pytest.raises(ValueError):
        lab.parse_json_output("I cannot answer that {sorry")
