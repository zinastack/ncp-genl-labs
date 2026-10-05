"""Lab 07 — reference solutions for the calculations. Try exercises.py first.

Every function is explained step by step, with worked numeric examples, in SOLUTION.md.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


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


def replicas_needed(
    requests_per_sec: float, latency_sec: float, concurrency_per_replica: int,
    headroom: float = 0.0,
) -> int:
    in_flight = requests_per_sec * latency_sec * (1 + headroom)  # Little's law: L = λ · W
    # Round UP (fewer replicas would overload); -1e-9 keeps float noise from turning 4.0 into 5.
    return max(1, math.ceil(in_flight / concurrency_per_replica - 1e-9))


def pick_best_config(results: list[dict], max_p95_ms: float, max_gpu_mem_gb: float) -> dict | None:
    # What Model Analyzer does after sweeping configs with perf_analyzer: discard configurations that
    # break a constraint, then take the highest throughput among the rest.
    feasible = [r for r in results if r["p95_ms"] <= max_p95_ms and r["gpu_mem_gb"] <= max_gpu_mem_gb]
    return max(feasible, key=lambda r: r["throughput"], default=None)


def rollout_bounds(replicas: int, max_surge: int, max_unavailable: int) -> tuple[int, int]:
    # During a RollingUpdate Kubernetes keeps total pods ≤ replicas + maxSurge and ready pods ≥
    # replicas − maxUnavailable. For GPU pods, the surge is extra GPUs you must have free.
    return replicas + max_surge, replicas - max_unavailable


def rerank_top_n(query: str, candidates: list[str], score_fn: Callable[[str, str], float], top_n: int) -> list[str]:
    # Retrieve many (high recall), then re-score each (query, passage) PAIR with a slower, more
    # accurate cross-encoder and keep only the best few for the LLM (high precision).
    return sorted(candidates, key=lambda c: -score_fn(query, c))[:top_n]


# ── Part B · Serving one LLM: from config.json to four server configs ──────────
@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    max_context: int
    tied_embeddings: bool
    params: int


L4 = {"name": "L4", "memory_gib": 22.5, "tflops_fp16": 121.0, "bandwidth_gbs": 300.0}


def model_spec(hf_config: dict, name: str = "") -> ModelSpec:
    hidden, heads = hf_config["hidden_size"], hf_config["num_attention_heads"]
    kv_heads = hf_config.get("num_key_value_heads", heads)  # absent = plain multi-head attention
    head_dim = hf_config.get("head_dim") or hidden // heads
    layers, inter = hf_config["num_hidden_layers"], hf_config["intermediate_size"]
    vocab, tied = hf_config["vocab_size"], hf_config.get("tie_word_embeddings", False)
    # Per layer: Q and O projections (hidden × heads·head_dim), K and V (hidden × kv_heads·head_dim:
    # GQA makes these small), and a gated MLP with three hidden × intermediate matrices.
    attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * inter
    # Embedding table, plus a separate output head unless the two share weights.
    embed = vocab * hidden * (1 if tied else 2)
    return ModelSpec(name, layers, hidden, heads, kv_heads, head_dim, inter, vocab,
                     hf_config["max_position_embeddings"], tied, layers * (attn + mlp) + embed)


def weight_gib(spec: ModelSpec, bits: int) -> float:
    return spec.params * bits / 8 / 2**30


def kv_bytes_per_token(spec: ModelSpec, kv_bits: int) -> int:
    # One K and one V vector per layer per KV head. The KV cache dtype, not the weight dtype, decides.
    return 2 * spec.layers * spec.kv_heads * spec.head_dim * kv_bits // 8


def kv_cache_gib(gpu_gib: float, weights_gib: float, engine: str, fraction: float, overhead_gib: float = 1.0) -> float:
    if engine == "trtllm":   # kv_cache_free_gpu_memory_fraction: share of memory FREE after weights
        kv = (gpu_gib - weights_gib - overhead_gib) * fraction
    elif engine == "vllm":   # gpu_memory_utilization: share of ALL memory for weights + activations + KV
        kv = gpu_gib * fraction - weights_gib - overhead_gib
    else:
        raise ValueError(f"unknown engine {engine!r}")
    return max(kv, 0.0)


def max_sequences(kv_gib: float, spec: ModelSpec, kv_bits: int, seq_len: int, tokens_per_block: int) -> int:
    # Paged KV cache: memory is handed out in fixed blocks, so a sequence occupies whole blocks.
    blocks_per_seq = math.ceil(seq_len / tokens_per_block)
    total_blocks = int(kv_gib * 2**30 // (tokens_per_block * kv_bytes_per_token(spec, kv_bits)))
    return total_blocks // blocks_per_seq


def prefill_iterations(prompt_tokens: int, max_num_tokens: int, decode_seqs: int, chunked: bool) -> int:
    # Every iteration processes at most max_num_tokens tokens; each running sequence decodes one.
    budget = max_num_tokens - decode_seqs
    if budget <= 0:
        raise ValueError("decoding sequences use the whole token budget: no room for prefill")
    if chunked:  # chunked prefill splits the prompt over several iterations
        return math.ceil(prompt_tokens / budget)
    if prompt_tokens > budget:  # without chunking the whole prompt must fit one iteration
        raise ValueError("prompt longer than the token budget: enable chunked prefill or raise max_num_tokens")
    return 1


def estimate_latency(
    spec: ModelSpec, gpu: dict, weight_bits: int, prompt_tokens: int, batch: int, context_tokens: int,
    kv_bits: int = 16, mfu: float = 0.5, bandwidth_efficiency: float = 0.7,
) -> dict[str, float]:
    # Prefill is compute-bound: ~2 FLOPs per parameter per prompt token.
    prefill_ms = 2 * spec.params * prompt_tokens / (gpu["tflops_fp16"] * 1e12 * mfu) * 1000
    # Decode is memory-bound: each step reads every weight once (shared by the batch) plus every
    # sequence's KV cache.
    step_bytes = spec.params * weight_bits / 8 + batch * context_tokens * kv_bytes_per_token(spec, kv_bits)
    step_ms = step_bytes / (gpu["bandwidth_gbs"] * 1e9 * bandwidth_efficiency) * 1000
    return {"prefill_ms": prefill_ms, "decode_step_ms": step_ms, "tokens_per_s": batch / step_ms * 1000}


def plan_llm_server(spec: ModelSpec, gpu: dict, workload: dict) -> dict:
    engine = workload["engine"]
    tpb = workload.get("tokens_per_block", 32 if engine == "trtllm" else 16)
    max_seq_len = workload["max_prompt"] + workload["max_output"]
    if max_seq_len > spec.max_context:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds the model's context {spec.max_context}")
    w = weight_gib(spec, workload["weight_bits"])
    kv = kv_cache_gib(gpu["memory_gib"], w, engine, workload["fraction"])
    seqs = max_sequences(kv, spec, workload["kv_bits"], max_seq_len, tpb)
    if seqs < 1:
        raise ValueError(f"{spec.name} at {workload['weight_bits']}-bit leaves no room for even one sequence")
    batch = min(workload["users"], seqs)
    chunk = workload.get("prefill_chunk", 2048)
    plan = {
        "model": spec.name, "engine": engine, "max_seq_len": max_seq_len, "max_batch_size": batch,
        # Room for every running sequence's decode token plus one prefill chunk, rounded up to 256.
        "max_num_tokens": math.ceil((batch + chunk) / 256) * 256,
        "fraction": workload["fraction"], "weight_bits": workload["weight_bits"], "kv_bits": workload["kv_bits"],
        "tokens_per_block": tpb, "chunked_prefill": True,
        "weights_gib": round(w, 2), "kv_cache_gib": round(kv, 2),
        "kv_tokens": int(kv * 2**30 // kv_bytes_per_token(spec, workload["kv_bits"])), "max_sequences": seqs,
        "warnings": [],
    }
    if seqs < workload["users"]:
        plan["warnings"].append(f"KV cache holds {seqs} full-length sequences for {workload['users']} users: "
                                "the rest queue, so TTFT rises")
    est = estimate_latency(spec, gpu, workload["weight_bits"], workload["max_prompt"], batch, max_seq_len,
                           workload["kv_bits"])
    plan["predicted"] = {k: round(v, 1) for k, v in est.items()}
    return plan


def render_trtllm_serve(plan: dict) -> tuple[list[str], dict]:
    if plan["engine"] != "trtllm":
        raise ValueError("plan was sized with vLLM memory semantics")
    args = ["--backend", "pytorch", "--max_batch_size", str(plan["max_batch_size"]),
            "--max_num_tokens", str(plan["max_num_tokens"]), "--max_seq_len", str(plan["max_seq_len"]),
            "--kv_cache_free_gpu_memory_fraction", str(plan["fraction"])]
    # Everything without a CLI flag goes into the YAML passed with --extra_llm_api_options.
    extra: dict = {"enable_chunked_prefill": plan["chunked_prefill"]}
    if plan["kv_bits"] == 8:
        extra["kv_cache_config"] = {"dtype": "fp8"}
    return args, extra


def render_vllm_args(plan: dict) -> list[str]:
    if plan["engine"] != "vllm":
        raise ValueError("plan was sized with TensorRT-LLM memory semantics")
    args = ["--max-model-len", str(plan["max_seq_len"]), "--max-num-seqs", str(plan["max_batch_size"]),
            "--max-num-batched-tokens", str(plan["max_num_tokens"]),
            "--gpu-memory-utilization", str(plan["fraction"]),
            "--kv-cache-dtype", "fp8" if plan["kv_bits"] == 8 else "auto",
            "--block-size", str(plan["tokens_per_block"])]
    if plan["chunked_prefill"]:
        args.append("--enable-chunked-prefill")
    return args


def render_triton_vllm(plan: dict, model_id: str) -> tuple[str, dict]:
    # The vLLM backend owns its GPU placement (KIND_MODEL) and batches internally.
    config = 'backend: "vllm"\ninstance_group [ { count: 1 kind: KIND_MODEL } ]\n'
    # model.json = vLLM engine arguments, the same names as the CLI flags with underscores.
    model_json = {
        "model": model_id, "gpu_memory_utilization": plan["fraction"], "max_model_len": plan["max_seq_len"],
        "max_num_seqs": plan["max_batch_size"], "max_num_batched_tokens": plan["max_num_tokens"],
        "enable_chunked_prefill": plan["chunked_prefill"], "block_size": plan["tokens_per_block"],
        "kv_cache_dtype": "fp8" if plan["kv_bits"] == 8 else "auto", "disable_log_stats": False,
    }
    return config, model_json


def render_triton_trtllm(plan: dict, model_id: str) -> dict:
    if plan["engine"] != "trtllm":
        raise ValueError("plan was sized with vLLM memory semantics")
    kv: dict = {"free_gpu_memory_fraction": plan["fraction"]}
    if plan["kv_bits"] == 8:
        kv["dtype"] = "fp8"
    return {
        "model": model_id, "backend": "pytorch", "tensor_parallel_size": 1,
        "max_batch_size": plan["max_batch_size"], "max_num_tokens": plan["max_num_tokens"],
        "max_seq_len": plan["max_seq_len"], "enable_chunked_prefill": plan["chunked_prefill"],
        "kv_cache_config": kv,
        # Triton's own batcher stays OFF (0): TensorRT-LLM does in-flight batching itself.
        # decoupled = a request may return many responses: required for token streaming.
        "triton_config": {"max_batch_size": 0, "decoupled": True},
    }


def effective_sampling(generation_config: dict, request: dict) -> dict:
    out = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_tokens": 16}
    # 1) the model's generation_config.json supplies defaults (vLLM does this with --generation-config auto)
    for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if k in generation_config:
            out[k] = generation_config[k]
    if generation_config.get("do_sample") is False:
        out["temperature"] = 0.0
    # 2) the request overrides anything it sets
    out.update({k: v for k, v in request.items() if v is not None})
    # 3) temperature 0 means greedy: the filters can no longer change the result
    out["greedy"] = out["temperature"] == 0
    if out["greedy"]:
        out["top_p"], out["top_k"] = 1.0, 0
    return out


def generate_request(
    protocol: str, model_name: str, prompt: str, sampling: dict, prompt_tokens: int, max_model_len: int,
    stream: bool = True,
) -> tuple[str, dict]:
    if prompt_tokens + sampling["max_tokens"] > max_model_len:  # the server would reject it
        raise ValueError(f"{prompt_tokens} prompt + {sampling['max_tokens']} new tokens > max_model_len {max_model_len}")
    if protocol == "openai":  # trtllm-serve, vLLM, NIM
        body = {"model": model_name, "prompt": prompt, "max_tokens": sampling["max_tokens"],
                "temperature": sampling["temperature"], "top_p": sampling["top_p"], "stream": stream}
        if sampling["top_k"] > 0:
            body["top_k"] = sampling["top_k"]  # an extension: not in OpenAI's own API
        if sampling["repetition_penalty"] != 1.0:
            body["repetition_penalty"] = sampling["repetition_penalty"]
        return "/v1/completions", body
    path = f"/v2/models/{model_name}/generate_stream" if stream else f"/v2/models/{model_name}/generate"
    if protocol == "triton-vllm":  # Triton generate extension; vLLM SamplingParams go in "parameters"
        params = {"max_tokens": sampling["max_tokens"], "temperature": sampling["temperature"],
                  "top_p": sampling["top_p"], "top_k": sampling["top_k"] or -1,  # vLLM: -1 = disabled
                  "repetition_penalty": sampling["repetition_penalty"]}
        return path, {"text_input": prompt, "stream": stream, "exclude_input_in_output": True, "parameters": params}
    if protocol == "triton-trtllm":  # LLM API backend: one input tensor per sampling parameter
        body = {"text_input": prompt, "streaming": stream, "sampling_param_max_tokens": sampling["max_tokens"],
                "sampling_param_exclude_input_from_output": True}
        if sampling["greedy"]:
            body["sampling_param_top_k"] = 1  # greedy = keep only the most likely token
        else:
            body["sampling_param_temperature"] = sampling["temperature"]
            body["sampling_param_top_p"] = sampling["top_p"]
            if sampling["top_k"] > 0:
                body["sampling_param_top_k"] = sampling["top_k"]
        return path, body
    raise ValueError(f"unknown protocol {protocol!r}")
