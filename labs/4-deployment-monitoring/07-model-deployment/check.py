"""Which Lab 07 tasks have you done? Checks the LIVE state (Triton, Kubernetes) and recorded results.

    make s4-check-07

A task passes when you have actually run it, not when a file exists: perf runs must be recorded
with different configs, models must be loaded, and so on. Nothing here changes any state.
"""

import csv
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

LAB = Path(__file__).parent
RESULTS = LAB / "results"
URL = "http://localhost:8000"


def get(path: str, method: str = "GET") -> object:
    req = urllib.request.Request(URL + path, data=b"" if method == "POST" else None, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def runs(model: str) -> dict[str, list[dict]]:
    """Recorded perf runs for a model, grouped by config summary."""
    out: dict[str, list[dict]] = {}
    if (RESULTS / "perf_history.csv").exists():
        with (RESULTS / "perf_history.csv").open() as f:
            for r in csv.DictReader(f):
                if r["model"] == model:
                    out.setdefault(r["config"], []).append(r)
    return out


def index() -> dict[tuple[str, str], str]:
    try:
        return {(m["name"], m.get("version", "")): m.get("state", "") for m in get("/v2/repository/index", "POST")}
    except OSError:
        return {}


def ready(models: dict[tuple[str, str], str], name: str) -> bool:
    return any(n == name and state == "READY" for (n, _), state in models.items())


def kubectl(*args: str) -> str:
    if not shutil.which("kubectl"):
        return ""
    p = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=20)
    return p.stdout.strip() if p.returncode == 0 else ""


def main() -> None:
    models = index()
    tc_configs = runs("text_classifier")
    checks: list[tuple[str, bool, str]] = []

    checks.append(("1 Explore a running Triton", ready(models, "text_classifier"),
                   "make s4-triton-up, then work through task 1"))
    dyn = {c.split(" inst=")[0] for c in tc_configs}
    checks.append(("2 Dynamic batching", len(dyn) >= 3 and any("dyn=off" in c for c in dyn),
                   f"{len(dyn)} batching settings recorded; need ≥ 3 including dyn=off (make s4-perf LABEL=...)"))
    inst = {c.split(" inst=")[1] for c in tc_configs if " inst=" in c}
    checks.append(("3 Instance groups", len(inst) >= 2, f"{len(inst)} instance settings recorded; need ≥ 2"))
    onnx, trt = runs("distilbert_onnx"), runs("distilbert_trt")
    checks.append(("4 ONNX Runtime vs TensorRT", bool(onnx) and bool(trt),
                   f"perf runs: distilbert_onnx {len(onnx)}, distilbert_trt {len(trt)}; need both"))
    checks.append(("5 Ensemble", ready(models, "sentiment_ensemble"), "make s4-load M=sentiment_ensemble"))
    versions = {v for (n, v), s in models.items() if n == "distilbert_onnx" and s == "READY"}
    checks.append(("6 Versions + model control", len(versions) >= 2,
                   f"distilbert_onnx versions READY: {sorted(versions) or 'none'}; need 2 (version_policy all)"))
    checks.append(("7 Model Analyzer", (RESULTS / "model_analyzer" / "reports").exists(), "make s4-ma-07"))
    engines: dict[str, set[str]] = {}
    if (RESULTS / "llm" / "bench.csv").exists():
        with (RESULTS / "llm" / "bench.csv").open() as f:
            for r in csv.DictReader(f):
                engines.setdefault(r["engine"], set()).add(r["settings"])
    settings = sum(len(v) for v in engines.values())
    checks.append(("8 One LLM, four servers", len(engines) >= 3 and settings >= 5,
                   f"benchmarked {len(engines)} engines, {settings} configs; need ≥ 3 engines and ≥ 5 configs (make s4-llm-bench)"))
    nim = list((RESULTS / "nim").glob("*/")) if (RESULTS / "nim").exists() else []
    checks.append(("9 NIM", bool(nim), "make s4-nim-up, then make s4-nim-bench"))

    gpus = kubectl("get", "node", "-o", "jsonpath={.items[0].status.allocatable.nvidia\\.com/gpu}")
    checks.append(("K1 Cluster with a schedulable GPU", gpus not in ("", "0"), "make s4-k8s-up"))
    checks.append(("K2 Triton Deployment ready", kubectl("get", "deploy", "triton", "-o",
                   "jsonpath={.status.readyReplicas}") not in ("", "0"), "make s4-k8s-deploy"))
    checks.append(("K3 Time-slicing", gpus.isdigit() and int(gpus) > 1, f"allocatable nvidia.com/gpu = {gpus or '?'}"))
    hpa = kubectl("get", "hpa", "triton", "-o", "jsonpath={.status.currentMetrics[0].pods.current.averageValue}")
    checks.append(("K4 HPA on queue time", bool(hpa), "HPA with a live custom metric (make s4-k8s-monitoring s4-k8s-hpa)"))
    revisions = kubectl("rollout", "history", "deploy/triton").count("\n") - 1
    checks.append(("K5 Rolling update", revisions >= 2, f"{max(revisions, 0)} revisions; need ≥ 2 (make s4-k8s-rollout)"))
    checks.append(("K6 Canary", bool(kubectl("get", "deploy", "triton-v2", "-o", "name")) or revisions >= 3,
                   "make s4-k8s-canary, s4-k8s-canary-check, s4-k8s-rollback"))

    for name, ok, hint in checks:
        print(f"  {'✔' if ok else '·'} {name:<36} {'' if ok else hint}")
    print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} done")


if __name__ == "__main__":
    main()
