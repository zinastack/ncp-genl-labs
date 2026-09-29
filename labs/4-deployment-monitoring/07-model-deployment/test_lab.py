import re

import numpy as np
import pytest


def test_1_triton_config(lab):
    cfg = lab.triton_config(
        "text_classifier", "python", 16,
        inputs=[("TEXT", "TYPE_STRING", [1])],
        outputs=[("LABEL", "TYPE_STRING", [1]), ("SCORE", "TYPE_FP32", [1])],
        preferred_batch_sizes=[4, 8], max_queue_delay_us=2000, instance_count=2,
    )
    assert re.search(r'name:\s*"text_classifier"', cfg)
    assert re.search(r'backend:\s*"python"', cfg)
    assert re.search(r"max_batch_size:\s*16", cfg)
    assert re.search(r'input\s*\[\s*\{\s*name:\s*"TEXT"\s+data_type:\s*TYPE_STRING\s+dims:\s*\[\s*1\s*\]', cfg)
    assert re.search(r'name:\s*"SCORE"\s+data_type:\s*TYPE_FP32', cfg)
    assert re.search(r"dynamic_batching\s*\{[^}]*preferred_batch_size:\s*\[\s*4,\s*8\s*\]", cfg)
    assert re.search(r"max_queue_delay_microseconds:\s*2000", cfg)
    assert re.search(r"instance_group\s*\[\s*\{\s*count:\s*2\s+kind:\s*KIND_GPU", cfg)
    plain = lab.triton_config("m", "onnxruntime", 0, [("X", "TYPE_FP32", [-1, 3])], [("Y", "TYPE_FP32", [-1])])
    assert "dynamic_batching" not in plain and re.search(r"dims:\s*\[\s*-1,\s*3\s*\]", plain)


def test_2_validate_repository(lab, tmp_path):
    good = tmp_path / "good"
    (good / "1").mkdir(parents=True)
    (good / "1" / "model.py").write_text("")
    (good / "config.pbtxt").write_text('name: "good"\nbackend: "python"\n')

    bad = tmp_path / "bad"
    (bad / "2").mkdir(parents=True)
    (bad / "config.pbtxt").write_text('name: "other"\n')

    noversion = tmp_path / "noversion"
    (noversion / "latest").mkdir(parents=True)

    assert lab.validate_model_repository(tmp_path) == [
        "bad/2: empty version directory",
        "bad: config name mismatch",
        "noversion: missing config.pbtxt",
        "noversion: no version directory",
    ]


def compute(bs):  # ~7 ms at batch 1, ~12 ms at batch 8 (numbers from the exam question)
    return 6.3 + 0.7 * bs


def test_3_dynamic_batching(lab):
    arrivals = [i * 2.5 for i in range(400)]  # 400 req/s

    no_batching = lab.simulate_dynamic_batching(arrivals, 1, 0, compute)
    assert no_batching[-1] > 1000, "one-at-a-time capacity is ~140 req/s: the queue grows without bound"

    batched = lab.simulate_dynamic_batching(arrivals, 8, 2.0, compute)
    assert np.percentile(batched, 95) <= 20, "batch 8 + 2 ms delay meets a 20 ms P95 SLO"

    too_patient = lab.simulate_dynamic_batching(arrivals, 8, 20.0, compute)
    assert np.percentile(too_patient, 95) > 20, "a 20 ms queue delay alone blows the 20 ms SLO"

    huge_delay = lab.simulate_dynamic_batching([0.0, 50.0], 8, 100.0, compute)
    assert huge_delay[0] > 100, "a long queue delay adds straight to latency at low load"

    assert lab.simulate_dynamic_batching([0.0, 0.5, 1.0], 2, 5.0, lambda b: 10.0) == [10.5, 10.0, 19.5]


def test_4_k8s(lab):
    d = lab.k8s_deployment("triton", "nvcr.io/nvidia/tritonserver:24.08-py3", gpus=1, replicas=2)
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert d["apiVersion"] == "apps/v1" and d["kind"] == "Deployment" and d["spec"]["replicas"] == 2
    assert d["spec"]["selector"]["matchLabels"] == d["spec"]["template"]["metadata"]["labels"] == {"app": "triton"}
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {p["containerPort"] for p in c["ports"]} == {8000, 8001, 8002}
    assert c["readinessProbe"]["httpGet"]["path"] == "/v2/health/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/v2/health/live"
    assert d["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0

    h = lab.k8s_hpa("triton", "nv_inference_queue_duration_us", 5000, 1, 4)
    assert h["apiVersion"] == "autoscaling/v2" and h["spec"]["scaleTargetRef"]["name"] == "triton"
    m = h["spec"]["metrics"][0]
    assert m["type"] == "Pods" and m["pods"]["target"]["averageValue"] == "5000"


def test_5_capacity(lab):
    assert lab.replicas_needed(200, 0.5, 16) == 7        # 100 in flight / 16
    assert lab.replicas_needed(200, 0.5, 16, headroom=0.2) == 8
    assert lab.replicas_needed(1, 0.01, 8) == 1
    assert lab.replicas_needed(64, 0.25, 16) == 1        # exactly 16 in flight


def test_6_kserve_payload(lab):
    body = lab.kserve_infer_request(["great GPU", "slow network"])
    assert body["inputs"][0] == {"name": "TEXT", "shape": [2, 1], "datatype": "BYTES", "data": ["great GPU", "slow network"]}
    assert body["outputs"] == [{"name": "LABEL"}, {"name": "SCORE"}]


def test_shipped_triton_repository_is_valid(lab):
    import pathlib
    repo = pathlib.Path(__file__).parent / "triton" / "model_repository"
    assert lab.validate_model_repository(repo) == []
