"""exercises.py and solutions.py must expose identical, fully type-annotated signatures.

This checks the source (ast), so it works on untouched stubs and on finished exercises alike.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LABS = sorted(p.parent for p in ROOT.glob("labs/*/*/exercises.py"))


def public_signatures(path: Path) -> dict[str, ast.FunctionDef]:
    """Top-level functions and class methods, keyed 'name' or 'Class.method'. Private helpers skipped."""
    tree = ast.parse(path.read_text())
    sigs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            sigs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    sigs[f"{node.name}.{item.name}"] = item
    return sigs


def render(fn: ast.FunctionDef) -> str:
    return f"({ast.unparse(fn.args)}) -> {ast.unparse(fn.returns) if fn.returns else '?'}"


@pytest.mark.parametrize("lab", LABS, ids=lambda p: p.name)
def test_signatures_match_and_are_typed(lab):
    exercises = public_signatures(lab / "exercises.py")
    solutions = public_signatures(lab / "solutions.py")
    assert exercises.keys() == solutions.keys(), "same functions in both files"
    for name, fn in exercises.items():
        assert render(fn) == render(solutions[name]), f"{name}: signatures differ"
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        untyped = [a.arg for a in args if a.annotation is None and a.arg != "self"]
        assert not untyped, f"{name}: missing type hints for {untyped}"
        assert fn.returns is not None, f"{name}: missing return type"
