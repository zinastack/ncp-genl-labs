import numpy as np


def compute(bs):  # ~7 ms at batch 1, ~12 ms at batch 8 (numbers from the exam question)
    return 6.3 + 0.7 * bs


def test_1_dynamic_batching(lab):
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


def test_2_capacity(lab):
    assert lab.replicas_needed(200, 0.5, 16) == 7        # 100 in flight / 16
    assert lab.replicas_needed(200, 0.5, 16, headroom=0.2) == 8
    assert lab.replicas_needed(1, 0.01, 8) == 1
    assert lab.replicas_needed(64, 0.25, 16) == 1        # exactly 16 in flight


def test_3_pick_best_config(lab):
    results = [
        {"name": "bs8_i1_d2ms", "throughput": 640, "p95_ms": 17.4, "gpu_mem_gb": 3},
        {"name": "bs16_i2_d5ms", "throughput": 900, "p95_ms": 24.0, "gpu_mem_gb": 6},
        {"name": "bs8_i2_d1ms", "throughput": 780, "p95_ms": 19.1, "gpu_mem_gb": 6},
        {"name": "bs32_i4", "throughput": 1100, "p95_ms": 31.0, "gpu_mem_gb": 14},
    ]
    assert lab.pick_best_config(results, 20, 8)["name"] == "bs8_i2_d1ms"
    assert lab.pick_best_config(results, 40, 8)["name"] == "bs16_i2_d5ms", "memory limit excludes bs32"
    assert lab.pick_best_config(results, 5, 8) is None


def test_4_rollout_bounds(lab):
    assert lab.rollout_bounds(3, 1, 0) == (4, 3), "never below 3 ready; needs 1 spare GPU"
    assert lab.rollout_bounds(4, 0, 1) == (4, 3), "no spare GPU; temporarily 3 ready"


def test_5_rerank(lab):
    passages = ["NCCL all-reduce basics", "Triton exposes metrics on port 8002", "Grafana dashboards"]
    overlap = lambda q, p: len(set(q.lower().split()) & set(p.lower().split()))
    assert lab.rerank_top_n("which port for triton metrics", passages, overlap, 1) == [passages[1]]
    assert len(lab.rerank_top_n("q", passages, overlap, 2)) == 2


def test_6_kv_cache_tokens(lab):
    # Qwen2.5-1.5B: 28 layers, 2 KV heads (GQA), head_dim 128, FP16 → 28 KiB per token
    assert lab.kv_cache_tokens(20, 0.5, 28, 2, 128) == 374_491, "10 GiB / 28672 B"
    assert lab.kv_cache_tokens(20, 0.5, 28, 2, 128) // 2048 == 182, "≈ 182 full 2k-token sequences in flight"
    assert lab.kv_cache_tokens(20, 0.9, 28, 2, 128) == 674_084, "more fraction → more concurrent sequences"
    assert lab.kv_cache_tokens(20, 0.5, 28, 2, 128, bytes_per_elem=1) == 748_982, "FP8 KV cache doubles capacity"
    assert lab.kv_cache_tokens(20, 0.5, 28, 12, 128) == 62_415, "without GQA (12 KV heads) 6× fewer tokens"
