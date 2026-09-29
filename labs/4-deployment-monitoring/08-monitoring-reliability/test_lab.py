import numpy as np
import pytest

TRITON_METRICS = """
# HELP nv_inference_request_success Number of successful inference requests
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="text_classifier",version="1"} 1000
nv_inference_request_success{model="other",version="1"} 5
nv_inference_count{model="text_classifier",version="1"} 1000
nv_inference_exec_count{model="text_classifier",version="1"} 200
nv_inference_queue_duration_us{model="text_classifier",version="1"} 2500000
nv_gpu_utilization{gpu_uuid="GPU-abc"} 0.73
process_uptime_seconds 42.5
"""


def test_1_latency_summary(lab):
    lat = list(range(1, 101))  # 1..100 ms
    s = lab.latency_summary(lat)
    assert (s["p50"], s["p95"], s["p99"], s["max"]) == (50, 95, 99, 100)
    assert s["mean"] == 50.5
    assert lab.latency_summary([5, 1, 9])["p95"] == 9


def test_2_prometheus(lab):
    m = lab.parse_prometheus(TRITON_METRICS)
    assert m[("nv_gpu_utilization", frozenset({("gpu_uuid", "GPU-abc")}))] == 0.73
    assert m[("process_uptime_seconds", frozenset())] == 42.5
    assert lab.triton_avg_queue_ms(m, "text_classifier") == pytest.approx(2.5)
    assert lab.triton_avg_batch_size(m, "text_classifier") == pytest.approx(5.0)


def test_3_anomalies(lab):
    rng = np.random.default_rng(0)
    series = list(20 + rng.normal(0, 1, 200))
    series[120] = 45     # latency spike
    series[170] = -5
    assert lab.rolling_zscore_anomalies(series, window=30, threshold=4) == [120, 170]
    assert lab.rolling_zscore_anomalies([1.0] * 50, 10) == []


def test_4_drift(lab):
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 20, 20000)          # e.g. prompt length at training time
    same = rng.normal(100, 20, 20000)
    shifted = rng.normal(130, 25, 20000)      # users now paste longer documents
    assert lab.psi(ref, same) < 0.02
    assert lab.psi(ref, shifted) > 0.25
    assert lab.ks_statistic(ref, same) < 0.03
    assert lab.ks_statistic(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])) == 1.0
    assert lab.ks_statistic(ref, shifted) > 0.4


def test_5_burn_rate(lab):
    assert lab.burn_rate(10, 1000, 0.999) == pytest.approx(10)
    assert lab.burn_rate(0, 0, 0.999) == 0
    assert lab.should_page((200, 1000), (1600, 100000), 0.999)          # 200x and 16x
    assert not lab.should_page((200, 1000), (500, 100000), 0.999), "short blip only: don't page"


def test_6_retraining(lab):
    assert lab.retraining_decision(0.05, 0.91, 0.92, 100) == (False, [])
    assert lab.retraining_decision(0.31, 0.85, 0.92, 6000) == (True, ["drift", "quality_drop", "new_data"])
    assert lab.retraining_decision(0.12, 0.88, 0.92, 0) == (True, ["quality_drop"])


def test_7_registry(lab):
    r = lab.ModelRegistry()
    r.register("v1", {"eval_passed": True})
    r.register("v2", {"eval_passed": True})
    r.register("v3", {"eval_passed": False})
    with pytest.raises(ValueError):
        r.register("v1", {})
    assert r.production() is None and r.stage("v1") == "staging"
    r.promote("v1")
    r.promote("v2")
    assert r.production() == "v2" and r.stage("v1") == "archived"
    with pytest.raises(ValueError):
        r.promote("v3")  # failed evaluation gate
    assert r.rollback() == "v1"
    assert r.production() == "v1" and r.stage("v2") == "archived"
    with pytest.raises(ValueError):
        r.rollback()


def test_8_canary(lab):
    routes = [lab.canary_route(f"user-{i}", 10) for i in range(5000)]
    assert 0.08 < routes.count("canary") / 5000 < 0.12
    assert lab.canary_route("user-42", 10) == lab.canary_route("user-42", 10), "sticky per user"
    stable = {"p95_ms": 200, "error_rate": 0.002}
    assert lab.canary_verdict(stable, {"p95_ms": 210, "error_rate": 0.003}) == "promote"
    assert lab.canary_verdict(stable, {"p95_ms": 260, "error_rate": 0.002}) == "rollback"
    assert lab.canary_verdict(stable, {"p95_ms": 190, "error_rate": 0.02}) == "rollback"
