"""Lab 07 — reference solutions. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import math
import re
from collections.abc import Callable
from pathlib import Path


def _tensors(block: str, tensors: list[tuple[str, str, list[int]]]) -> str:
    # One line per tensor: name, type, and the per-request shape (Triton adds the batch dim).
    items = ",\n".join(
        f'  {{ name: "{n}" data_type: {t} dims: [ {", ".join(map(str, d))} ] }}' for n, t, d in tensors
    )
    return f"{block} [\n{items}\n]"


def triton_config(
    name: str, backend: str, max_batch_size: int, inputs: list[tuple[str, str, list[int]]],
    outputs: list[tuple[str, str, list[int]]], preferred_batch_sizes: list[int] | None = None,
    max_queue_delay_us: int | None = None, instance_count: int = 1, kind: str = "KIND_GPU",
) -> str:
    parts = [
        f'name: "{name}"',  # must equal the model's directory name
        f'backend: "{backend}"',
        f"max_batch_size: {max_batch_size}",  # 0 = no batching (dims then include the batch dim)
        _tensors("input", inputs),
        _tensors("output", outputs),
    ]
    if preferred_batch_sizes or max_queue_delay_us is not None:  # only emit what was asked for
        fields = []
        if preferred_batch_sizes:
            fields.append(f"  preferred_batch_size: [ {', '.join(map(str, preferred_batch_sizes))} ]")
        if max_queue_delay_us is not None:
            fields.append(f"  max_queue_delay_microseconds: {max_queue_delay_us}")  # adds to latency!
        parts.append("dynamic_batching {\n" + "\n".join(fields) + "\n}")
    parts.append(f"instance_group [ {{ count: {instance_count} kind: {kind} }} ]")  # concurrent copies
    return "\n".join(parts) + "\n"


def validate_model_repository(repo: str | Path) -> list[str]:
    problems = []
    for model in sorted(p for p in Path(repo).iterdir() if p.is_dir()):
        config = model / "config.pbtxt"
        versions = [d for d in model.iterdir() if d.is_dir() and d.name.isdigit()]  # "1", "2", …
        if not config.exists():
            problems.append(f"{model.name}: missing config.pbtxt")
        if not versions:
            problems.append(f"{model.name}: no version directory")
        for v in sorted(versions, key=lambda d: int(d.name)):
            if not any(v.iterdir()):
                problems.append(f"{model.name}/{v.name}: empty version directory")
        if config.exists():
            m = re.search(r'^\s*name:\s*"([^"]+)"', config.read_text(), re.MULTILINE)
            if m and m.group(1) != model.name:
                problems.append(f"{model.name}: config name mismatch")
    return problems


def simulate_dynamic_batching(
    arrivals_ms: list[float], max_batch: int, max_delay_ms: float,
    compute_ms: Callable[[int], float],
) -> list[float]:
    n = len(arrivals_ms)
    latencies = [0.0] * n
    i, free = 0, 0.0  # i = oldest unserved request; free = when the GPU becomes idle
    while i < n:
        # The batch could be "full" once max_batch requests are waiting.
        full_at = arrivals_ms[i + max_batch - 1] if i + max_batch - 1 < n else math.inf
        # Start when the GPU is free AND (the oldest has waited max_delay OR the batch is full).
        launch = max(free, min(arrivals_ms[i] + max_delay_ms, full_at))
        j = i
        while j < n and j - i < max_batch and arrivals_ms[j] <= launch:  # everyone who has arrived
            j += 1
        finish = launch + compute_ms(j - i)
        for k in range(i, j):
            latencies[k] = finish - arrivals_ms[k]  # queue wait + batching delay + compute
        free, i = finish, j
    return latencies


def k8s_deployment(name: str, image: str, gpus: int = 1, replicas: int = 1) -> dict:
    labels = {"app": name}  # selector and pod labels must match, or the Deployment finds no pods
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
            # Start a new pod before removing an old one: capacity never drops during a rollout.
            "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [{
                        "name": "triton",
                        "image": image,
                        "args": ["tritonserver", "--model-repository=/models"],
                        "ports": [
                            {"containerPort": 8000, "name": "http"},
                            {"containerPort": 8001, "name": "grpc"},
                            {"containerPort": 8002, "name": "metrics"},  # Prometheus scrapes this
                        ],
                        # The GPU request: the NVIDIA device plugin makes this resource schedulable.
                        "resources": {"limits": {"nvidia.com/gpu": gpus}},
                        "readinessProbe": {"httpGet": {"path": "/v2/health/ready", "port": 8000}},  # traffic gate
                        "livenessProbe": {"httpGet": {"path": "/v2/health/live", "port": 8000}},  # restart if hung
                    }]
                },
            },
        },
    }


def k8s_hpa(
    name: str, metric: str, target_average: float, min_replicas: int = 1, max_replicas: int = 8,
) -> dict:
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": name},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            # A per-pod custom metric (e.g. queue time) instead of CPU, which stays low while the GPU saturates.
            "metrics": [{
                "type": "Pods",
                "pods": {"metric": {"name": metric},
                         "target": {"type": "AverageValue", "averageValue": str(target_average)}},  # quantity string
            }],
        },
    }


def replicas_needed(
    requests_per_sec: float, latency_sec: float, concurrency_per_replica: int,
    headroom: float = 0.0,
) -> int:
    in_flight = requests_per_sec * latency_sec * (1 + headroom)  # Little's law: L = λ · W
    # Round UP (fewer replicas would overload); -1e-9 keeps float noise from turning 4.0 into 5.
    return max(1, math.ceil(in_flight / concurrency_per_replica - 1e-9))


def kserve_infer_request(
    texts: list[str], input_name: str = "TEXT", output_names: tuple[str, ...] = ("LABEL", "SCORE"),
) -> dict:
    return {
        # shape = [batch, *config dims]; strings travel as BYTES.
        "inputs": [{"name": input_name, "shape": [len(texts), 1], "datatype": "BYTES", "data": list(texts)}],
        "outputs": [{"name": n} for n in output_names],
    }
