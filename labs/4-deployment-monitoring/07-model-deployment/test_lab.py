import json
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).parent / "llm" / "models"


def load(name: str, file: str = "config.json") -> dict:
    return json.loads((MODELS / name / file).read_text())


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


# ── Part B · the LLM serving scenario ───────────────────────────────────────────
QWEN = "Qwen2.5-1.5B-Instruct"
WORKLOAD = {"engine": "trtllm", "max_prompt": 2048, "max_output": 512, "users": 64,
            "weight_bits": 16, "kv_bits": 16, "fraction": 0.9}


def qwen_spec(lab):
    return lab.model_spec(load(QWEN), QWEN)


def test_6_model_spec(lab):
    qwen = qwen_spec(lab)
    assert (qwen.layers, qwen.hidden, qwen.heads, qwen.kv_heads, qwen.head_dim) == (28, 1536, 12, 2, 128)
    assert qwen.tied_embeddings and qwen.max_context == 32768
    assert qwen.params == 1_543_569_408, "≈ 1.54 B, the size on the model card"
    assert lab.model_spec(load("Qwen2.5-7B-Instruct")).params == 7_615_283_200, "untied: output head counted"
    llama = lab.model_spec(load("Llama-3.1-8B-Instruct"))
    assert (llama.kv_heads, llama.head_dim, llama.params) == (8, 128, 8_029_995_008)
    mha = lab.model_spec({"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 1,
                          "intermediate_size": 128, "vocab_size": 10, "max_position_embeddings": 32})
    assert mha.kv_heads == 4 and mha.head_dim == 16 and not mha.tied_embeddings


def test_7_8_weights_and_kv(lab):
    qwen = qwen_spec(lab)
    assert lab.weight_gib(qwen, 16) == pytest.approx(2.875, abs=1e-3)
    assert lab.weight_gib(qwen, 4) == pytest.approx(0.719, abs=1e-3)
    assert lab.kv_bytes_per_token(qwen, 16) == 28_672, "2 × 28 layers × 2 KV heads × 128 × 2 bytes"
    assert lab.kv_bytes_per_token(qwen, 8) == 14_336, "FP8 KV cache halves it"
    assert lab.kv_bytes_per_token(lab.model_spec(load("Llama-3.1-8B-Instruct")), 16) == 131_072


def test_9_kv_cache_gib(lab):
    assert lab.kv_cache_gib(22.5, 2.875, "trtllm", 0.9) == pytest.approx(16.7625)
    assert lab.kv_cache_gib(22.5, 2.875, "vllm", 0.9) == pytest.approx(16.375), "same 0.9, smaller cache"
    assert lab.kv_cache_gib(22.5, 2.875, "vllm", 0.1) == 0.0, "0.1 of the GPU can't even hold the weights"
    assert lab.kv_cache_gib(22.5, 2.875, "trtllm", 0.1) == pytest.approx(1.8625), "0.1 of the FREE memory still works"
    with pytest.raises(ValueError):
        lab.kv_cache_gib(22.5, 2.875, "sglang", 0.9)


def test_10_max_sequences(lab):
    qwen = qwen_spec(lab)
    assert lab.max_sequences(16.7625, qwen, 16, 2560, 32) == 245
    assert lab.max_sequences(10.0, qwen, 16, 1000, 32) == lab.max_sequences(10.0, qwen, 16, 1024, 32) == 365, \
        "1,000 tokens still occupy 32 whole blocks"
    assert lab.max_sequences(16.7625, qwen, 8, 2560, 32) == 490, "FP8 KV cache: twice the sequences"


def test_11_prefill_iterations(lab):
    assert lab.prefill_iterations(4001, 2048, 48, chunked=True) == 3, "budget 2000 per iteration"
    assert lab.prefill_iterations(1000, 2048, 48, chunked=False) == 1
    with pytest.raises(ValueError):
        lab.prefill_iterations(4000, 2048, 48, chunked=False)
    with pytest.raises(ValueError):
        lab.prefill_iterations(10, 64, 64, chunked=True)


def test_12_estimate_latency(lab):
    qwen = qwen_spec(lab)
    one = lab.estimate_latency(qwen, lab.L4, 16, 512, 1, 640)
    assert one["prefill_ms"] == pytest.approx(26.13, abs=0.01)
    assert one["decode_step_ms"] == pytest.approx(14.79, abs=0.01)
    many = lab.estimate_latency(qwen, lab.L4, 16, 512, 32, 640)
    assert many["decode_step_ms"] == pytest.approx(17.50, abs=0.01), "32× the batch, +18% step time"
    assert many["tokens_per_s"] == pytest.approx(1828.9, abs=0.1)
    int4 = lab.estimate_latency(qwen, lab.L4, 4, 512, 1, 640)
    assert int4["decode_step_ms"] < one["decode_step_ms"] / 3, "decode speed follows weight bytes"


def test_13_plan(lab):
    qwen = qwen_spec(lab)
    p = lab.plan_llm_server(qwen, lab.L4, WORKLOAD)
    assert (p["max_seq_len"], p["max_batch_size"], p["max_num_tokens"]) == (2560, 64, 2304)
    assert (p["weights_gib"], p["kv_cache_gib"], p["kv_tokens"], p["max_sequences"]) == (2.88, 16.76, 627_737, 245)
    assert p["tokens_per_block"] == 32 and p["chunked_prefill"] is True and p["warnings"] == []
    assert p["predicted"] == {"prefill_ms": 104.5, "decode_step_ms": 37.1, "tokens_per_s": 1726.5}
    assert lab.plan_llm_server(qwen, lab.L4, {**WORKLOAD, "engine": "vllm"})["tokens_per_block"] == 16

    llama = lab.model_spec(load("Llama-3.1-8B-Instruct"), "llama")
    big = {**WORKLOAD, "users": 32, "max_prompt": 4096, "max_output": 1024}
    fp16 = lab.plan_llm_server(llama, lab.L4, big)
    assert fp16["max_batch_size"] == 9 and "9" in fp16["warnings"][0] and "32" in fp16["warnings"][0]
    assert lab.plan_llm_server(llama, lab.L4, {**big, "weight_bits": 4})["max_batch_size"] == 25, \
        "INT4 weights free memory for KV cache"
    with pytest.raises(ValueError):
        lab.plan_llm_server(qwen, lab.L4, {**WORKLOAD, "max_prompt": 32768})
    with pytest.raises(ValueError):
        lab.plan_llm_server(lab.model_spec(load("Qwen2.5-7B-Instruct")), lab.L4,
                            {**WORKLOAD, "engine": "vllm", "fraction": 0.6})


def test_14_15_render_servers(lab):
    qwen = qwen_spec(lab)
    p = lab.plan_llm_server(qwen, lab.L4, WORKLOAD)
    args, extra = lab.render_trtllm_serve(p)
    assert args == ["--backend", "pytorch", "--max_batch_size", "64", "--max_num_tokens", "2304",
                    "--max_seq_len", "2560", "--kv_cache_free_gpu_memory_fraction", "0.9"]
    assert extra == {"enable_chunked_prefill": True}
    assert lab.render_trtllm_serve({**p, "kv_bits": 8})[1]["kv_cache_config"] == {"dtype": "fp8"}
    with pytest.raises(ValueError):
        lab.render_vllm_args(p)

    v = lab.plan_llm_server(qwen, lab.L4, {**WORKLOAD, "engine": "vllm"})
    assert lab.render_vllm_args(v) == [
        "--max-model-len", "2560", "--max-num-seqs", "64", "--max-num-batched-tokens", "2304",
        "--gpu-memory-utilization", "0.9", "--kv-cache-dtype", "auto", "--block-size", "16",
        "--enable-chunked-prefill"]
    with pytest.raises(ValueError):
        lab.render_trtllm_serve(v)


def test_16_17_render_triton(lab):
    qwen = qwen_spec(lab)
    v = lab.plan_llm_server(qwen, lab.L4, {**WORKLOAD, "engine": "vllm", "kv_bits": 8})
    config, model_json = lab.render_triton_vllm(v, "Qwen/Qwen2.5-1.5B-Instruct")
    assert 'backend: "vllm"' in config and "KIND_MODEL" in config
    assert model_json == {"model": "Qwen/Qwen2.5-1.5B-Instruct", "gpu_memory_utilization": 0.9,
                          "max_model_len": 2560, "max_num_seqs": 64, "max_num_batched_tokens": 2304,
                          "enable_chunked_prefill": True, "block_size": 16, "kv_cache_dtype": "fp8",
                          "disable_log_stats": False}

    t = lab.plan_llm_server(qwen, lab.L4, WORKLOAD)
    y = lab.render_triton_trtllm(t, "Qwen/Qwen2.5-1.5B-Instruct")
    assert y["max_batch_size"] == 64 and y["triton_config"] == {"max_batch_size": 0, "decoupled": True}, \
        "the engine batches; Triton's batcher stays off"
    assert y["kv_cache_config"] == {"free_gpu_memory_fraction": 0.9} and y["backend"] == "pytorch"
    assert (y["max_num_tokens"], y["max_seq_len"], y["enable_chunked_prefill"]) == (2304, 2560, True)


def test_18_effective_sampling(lab):
    gc = load(QWEN, "generation_config.json")
    s = lab.effective_sampling(gc, {"max_tokens": 128})
    assert s == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1,
                 "max_tokens": 128, "greedy": False}, "the model's defaults apply when the request is silent"
    g = lab.effective_sampling(gc, {"temperature": 0, "top_p": None, "max_tokens": 64})
    assert g["greedy"] and (g["top_p"], g["top_k"]) == (1.0, 0) and g["repetition_penalty"] == 1.1
    assert lab.effective_sampling({"do_sample": False}, {})["greedy"]
    assert lab.effective_sampling({}, {})["max_tokens"] == 16


def test_19_generate_request(lab):
    s = lab.effective_sampling(load(QWEN, "generation_config.json"), {"max_tokens": 64})
    path, body = lab.generate_request("openai", "Qwen/Qwen2.5-1.5B-Instruct", "Hi", s, 10, 2560)
    assert path == "/v1/completions" and body["top_k"] == 20 and body["repetition_penalty"] == 1.1
    assert body["stream"] is True and body["model"] == "Qwen/Qwen2.5-1.5B-Instruct"

    path, body = lab.generate_request("triton-vllm", "llm", "Hi", s, 10, 2560, stream=False)
    assert path == "/v2/models/llm/generate"
    assert body["parameters"] == {"max_tokens": 64, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                  "repetition_penalty": 1.1}
    assert body["exclude_input_in_output"] is True

    greedy = lab.effective_sampling({}, {"temperature": 0, "max_tokens": 32})
    assert lab.generate_request("triton-vllm", "llm", "Hi", greedy, 10, 2560)[1]["parameters"]["top_k"] == -1
    path, body = lab.generate_request("triton-trtllm", "llm", "Hi", greedy, 10, 2560)
    assert path == "/v2/models/llm/generate_stream"
    assert body == {"text_input": "Hi", "streaming": True, "sampling_param_max_tokens": 32,
                    "sampling_param_exclude_input_from_output": True, "sampling_param_top_k": 1}
    with pytest.raises(ValueError):
        lab.generate_request("openai", "m", "Hi", {**s, "max_tokens": 600}, 2000, 2560)
    with pytest.raises(ValueError):
        lab.generate_request("grpc", "m", "Hi", s, 10, 2560)
