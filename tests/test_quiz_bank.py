"""Sanity checks on every labs/*/*/quiz.toml so the quiz runner never breaks."""

import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (Ubuntu 22.04 GPU images)
    import tomli as tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
QUIZ_FILES = sorted(ROOT.glob("labs/*/*/quiz.toml"))
# Options are shuffled at runtime, so explanations must not point at letters.
LETTER_REF = re.compile(r"\b(option|answer|choice)\s+[A-F]\b|\([A-F]\)|\b[A-F]\)", re.IGNORECASE)


def test_all_ten_domains_present():
    codes = sorted(tomllib.loads(p.read_text())["code"] for p in QUIZ_FILES)
    assert codes == [f"{i:02d}" for i in range(1, 11)]


def test_weights_match_blueprint():
    assert sum(tomllib.loads(p.read_text())["weight"] for p in QUIZ_FILES) == 100


@pytest.mark.parametrize("path", QUIZ_FILES, ids=lambda p: p.parent.name)
def test_quiz_file(path):
    data = tomllib.loads(path.read_text())
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in data["questions"]:
        assert q["id"].startswith(data["code"] + "-"), q["id"]
        assert 3 <= len(q["options"]) <= 6, q["id"]
        assert len(set(q["options"])) == len(q["options"]), f"{q['id']}: duplicate options"
        valid = set("ABCDEF"[: len(q["options"])])
        assert q["answer"] and set(q["answer"]) <= valid, q["id"]
        if len(q["answer"]) > 1:
            assert "select" in q["question"].lower(), f"{q['id']}: multi-answer question must say 'Select N'"
        assert len(q["explanation"].strip()) > 40, f"{q['id']}: explanation too short"
        assert not LETTER_REF.search(q["explanation"]), f"{q['id']}: explanation references an option letter"
