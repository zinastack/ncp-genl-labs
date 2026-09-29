"""Shared pytest wiring for every lab.

Each lab folder has `exercises.py` (yours to fill in) and `solutions.py`
(reference). Tests receive whichever one you ask for through the `lab` fixture:

    pytest labs/1-foundations-prompting/01-llm-architecture          # your code
    pytest labs/1-foundations-prompting/01-llm-architecture --solutions
"""

import importlib.util
import os
import pathlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--solutions",
        action="store_true",
        help="Run the tests against solutions.py instead of exercises.py",
    )


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"{path.parent.name}.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lab(request):
    use_solutions = request.config.getoption("--solutions") or os.environ.get("LAB_SOLUTIONS") == "1"
    name = "solutions.py" if use_solutions else "exercises.py"
    return _load(pathlib.Path(request.module.__file__).with_name(name))
