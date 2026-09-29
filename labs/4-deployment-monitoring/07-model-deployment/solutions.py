"""Lab 07 — reference solutions."""

import math
import re
from pathlib import Path


def _tensors(block, tensors):
    items = ",\n".join(
        f'  {{ name: "{n}" data_type: {t} dims: [ {", ".join(map(str, d))} ] }}' for n, t, d in tensors
    )
    return f"{block} [\n{items}\n]"


def triton_config(name, backend, max_batch_size, inputs, outputs, preferred_batch_sizes=None,
                  max_queue_delay_us=None, instance_count=1, kind="KIND_GPU"):
    parts = [
        f'name: "{name}"',
        f'backend: "{backend}"',
        f"max_batch_size: {max_batch_size}",
        _tensors("input", inputs),
        _tensors("output", outputs),
    ]
    if preferred_batch_sizes or max_queue_delay_us is not None:
        fields = []
        if preferred_batch_sizes:
            fields.append(f"  preferred_batch_size: [ {', '.join(map(str, preferred_batch_sizes))} ]")
        if max_queue_delay_us is not None:
            fields.append(f"  max_queue_delay_microseconds: {max_queue_delay_us}")
        parts.append("dynamic_batching {\n" + "\n".join(fields) + "\n}")
    parts.append(f"instance_group [ {{ count: {instance_count} kind: {kind} }} ]")
    return "\n".join(parts) + "\n"


def validate_model_repository(repo):
    problems = []
    for model in sorted(p for p in Path(repo).iterdir() if p.is_dir()):
        config = model / "config.pbtxt"
        versions = [d for d in model.iterdir() if d.is_dir() and d.name.isdigit()]
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


def simulate_dynamic_batching(arrivals_ms, max_batch, max_delay_ms, compute_ms):
    n = len(arrivals_ms)
    latencies = [0.0] * n
    i, free = 0, 0.0
    while i < n:
        full_at = arrivals_ms[i + max_batch - 1] if i + max_batch - 1 < n else math.inf
        launch = max(free, min(arrivals_ms[i] + max_delay_ms, full_at))
        j = i
        while j < n and j - i < max_batch and arrivals_ms[j] <= launch:
            j += 1
        finish = launch + compute_ms(j - i)
        for k in range(i, j):
            latencies[k] = finish - arrivals_ms[k]
        free, i = finish, j
    return latencies


def k8s_deployment(name, image, gpus=1, replicas=1):
    labels = {"app": name}
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
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
                            {"containerPort": 8002, "name": "metrics"},
                        ],
                        "resources": {"limits": {"nvidia.com/gpu": gpus}},
                        "readinessProbe": {"httpGet": {"path": "/v2/health/ready", "port": 8000}},
                        "livenessProbe": {"httpGet": {"path": "/v2/health/live", "port": 8000}},
                    }]
                },
            },
        },
    }


def k8s_hpa(name, metric, target_average, min_replicas=1, max_replicas=8):
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": name},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [{
                "type": "Pods",
                "pods": {"metric": {"name": metric},
                         "target": {"type": "AverageValue", "averageValue": str(target_average)}},
            }],
        },
    }


def replicas_needed(requests_per_sec, latency_sec, concurrency_per_replica, headroom=0.0):
    in_flight = requests_per_sec * latency_sec * (1 + headroom)
    return max(1, math.ceil(in_flight / concurrency_per_replica - 1e-9))


def kserve_infer_request(texts, input_name="TEXT", output_names=("LABEL", "SCORE")):
    return {
        "inputs": [{"name": input_name, "shape": [len(texts), 1], "datatype": "BYTES", "data": list(texts)}],
        "outputs": [{"name": n} for n in output_names],
    }
