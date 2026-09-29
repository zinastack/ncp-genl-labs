"""Load a lab's implementation for the GPU scripts.

By default gpu_lab.py uses the reference solutions so it always runs. Once your exercises pass,
run with USE_EXERCISES=1 to drive the real models with YOUR code:

    USE_EXERCISES=1 python labs/1-foundations-prompting/02-prompt-engineering/gpu_lab.py
"""

import importlib.util
import os
from pathlib import Path


def load(lab_dir: str | Path):
    name = "exercises.py" if os.environ.get("USE_EXERCISES") == "1" else "solutions.py"
    path = Path(lab_dir) / name
    spec = importlib.util.spec_from_file_location(f"{Path(lab_dir).name}.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"[using {path.relative_to(Path(lab_dir).parents[1])}]")
    return module
