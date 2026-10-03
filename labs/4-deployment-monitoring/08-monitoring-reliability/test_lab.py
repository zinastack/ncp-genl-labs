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


def test_6_canary(lab):
    routes = [lab.canary_route(f"user-{i}", 10) for i in range(5000)]
    assert 0.08 < routes.count("canary") / 5000 < 0.12
    assert lab.canary_route("user-42", 10) == lab.canary_route("user-42", 10), "sticky per user"
    stable = {"p95_ms": 200, "error_rate": 0.002}
    assert lab.canary_verdict(stable, {"p95_ms": 210, "error_rate": 0.003}) == "promote"
    assert lab.canary_verdict(stable, {"p95_ms": 260, "error_rate": 0.002}) == "rollback"
    assert lab.canary_verdict(stable, {"p95_ms": 190, "error_rate": 0.02}) == "rollback"


BOUNDS = [0.05, 0.1, 0.25, 0.5, 1.0, float("inf")]  # latency buckets in seconds (Prometheus "le")
FAST = [800, 880, 895, 900, 900, 900]   # replica A: 900 requests, mostly fast
SLOW = [0, 10, 40, 80, 100, 100]        # replica B: 100 requests, slow


def test_7_histograms(lab):
    assert lab.merge_histograms([FAST, SLOW]) == [800, 890, 935, 980, 1000, 1000]
    p95_a = lab.histogram_quantile(BOUNDS, FAST, 0.95)
    p95_b = lab.histogram_quantile(BOUNDS, SLOW, 0.95)
    assert p95_a == pytest.approx(0.084375) and p95_b == pytest.approx(0.875)
    true_p95 = lab.histogram_quantile(BOUNDS, lab.merge_histograms([FAST, SLOW]), 0.95)
    assert true_p95 == pytest.approx(1 / 3)
    assert (p95_a + p95_b) / 2 == pytest.approx(0.4797, abs=1e-4), "averaging per-replica P95s is wrong"
    assert lab.histogram_quantile(BOUNDS, [0, 0, 0, 0, 5, 10], 0.99) == 1.0, "+Inf bucket → last finite bound"


def test_8_backoff(lab):
    no_jitter = lab.backoff_delays(7, base=0.1, cap=2.0, jitter=lambda upper: upper)
    assert no_jitter == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0])
    rng = np.random.default_rng(0)
    delays = lab.backoff_delays(7, 0.1, 2.0, jitter=lambda upper: rng.uniform(0, upper))
    assert all(0 <= d <= u for d, u in zip(delays, no_jitter)), "full jitter stays under the bound"


def test_9_embedding_drift(lab):
    rng = np.random.default_rng(0)
    topic_a, topic_b = np.eye(16)[0] * 5, np.eye(16)[1] * 5
    ref = topic_a + rng.normal(size=(500, 16))
    same = topic_a + rng.normal(size=(500, 16))
    shifted = topic_b + rng.normal(size=(500, 16))
    assert lab.embedding_drift(ref, same) < 0.05
    assert lab.embedding_drift(ref, shifted) > 0.8, "users now ask about a different topic"
