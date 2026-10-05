"""Lab 07 — the calculations and configs behind the deployment tasks. Fill in every TODO, then run:

    make test-07

Part A (1–5): Triton and Kubernetes arithmetic used in tasks 2, 7, K4 and K5.
Part B (6–19): ONE scenario, end to end: serve an LLM on one L4. You read the model's real config.json
(llm/models/), size weights and KV cache, choose batching / prefill / context settings, render the
configs for four servers (trtllm-serve, vLLM, Triton + vLLM backend, Triton + TensorRT-LLM backend),
and build generate requests for each. Task 8 on the GPU deploys YOUR plan (USE_EXERCISES=1) and
measures it against your predictions.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


# Part A · Triton and Kubernetes arithmetic ────────────────────────────────────

# 1 ─────────────────────────────────────────────────────────────────────────────
def simulate_dynamic_batching(
    arrivals_ms: list[float], max_batch: int, max_delay_ms: float,
    compute_ms: Callable[[int], float],
) -> list[float]:
    """Single model instance with a dynamic batcher. Return each request's latency (finish − arrival).

    Loop until all requests are served (arrivals are sorted):
      oldest = first unserved request, free = time the instance becomes idle (starts at 0)
      full_at = arrival of the (max_batch)-th unserved request (inf if there aren't that many)
      launch = max(free, min(arrival[oldest] + max_delay_ms, full_at))
      batch = unserved requests with arrival <= launch, at most max_batch of them
      finish = launch + compute_ms(len(batch));  free = finish
    `compute_ms` is a function batch_size -> milliseconds.
    """
    raise NotImplementedError


# 2 ─────────────────────────────────────────────────────────────────────────────
def replicas_needed(
    requests_per_sec: float, latency_sec: float, concurrency_per_replica: int,
    headroom: float = 0.0,
) -> int:
    """Little's law: in-flight = RPS × latency. Add `headroom` (e.g. 0.2 = 20% spare), then
    divide by per-replica concurrency and round UP. Minimum 1.
    """
    raise NotImplementedError


# 3 ─────────────────────────────────────────────────────────────────────────────
def pick_best_config(results: list[dict], max_p95_ms: float, max_gpu_mem_gb: float) -> dict | None:
    """Model-Analyzer-style selection. results: dicts with "name", "throughput", "p95_ms",
    "gpu_mem_gb". Return the highest-throughput result meeting BOTH limits, or None.
    """
    raise NotImplementedError


# 4 ─────────────────────────────────────────────────────────────────────────────
def rollout_bounds(replicas: int, max_surge: int, max_unavailable: int) -> tuple[int, int]:
    """Kubernetes RollingUpdate limits: (max total pods, min ready pods) =
    (replicas + max_surge, replicas − max_unavailable).
    """
    raise NotImplementedError


# 5 ─────────────────────────────────────────────────────────────────────────────
def rerank_top_n(query: str, candidates: list[str], score_fn: Callable[[str, str], float], top_n: int) -> list[str]:
    """Second-stage reranking: score every (query, candidate) pair with score_fn (a cross-encoder)
    and return the top_n candidates, best first.
    """
    raise NotImplementedError


# Part B · Serving one LLM: from config.json to four server configs ────────────
# Scenario: a chat assistant on ONE NVIDIA L4 (24 GB). Prompts up to 2,048 tokens, answers up to 512,
# 64 concurrent users. Default model Qwen2.5-1.5B-Instruct; Qwen2.5-7B and Llama-3.1-8B for
# comparison. Configs: llm/models/<model>/config.json and generation_config.json.

@dataclass
class ModelSpec:
    """Given (no TODO): what serving needs to know about a model."""
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


# Given: the GPU. 23,034 MiB usable, FP16 tensor-core peak (dense), memory bandwidth.
L4 = {"name": "L4", "memory_gib": 22.5, "tflops_fp16": 121.0, "bandwidth_gbs": 300.0}


# 6 ─────────────────────────────────────────────────────────────────────────────
def model_spec(hf_config: dict, name: str = "") -> ModelSpec:
    """Read a Hugging Face config.json dict into a ModelSpec.

    Keys: hidden_size, num_attention_heads, num_key_value_heads (missing → = heads), head_dim
    (missing → hidden_size // heads), num_hidden_layers, intermediate_size, vocab_size,
    max_position_embeddings, tie_word_embeddings (missing → False).
    params (ignore biases and norms):
      per layer  = Q and O: 2 · hidden · heads · head_dim  +  K and V: 2 · hidden · kv_heads · head_dim
                   + gated MLP: 3 · hidden · intermediate
      embeddings = vocab · hidden, twice if the output head is not tied
      params     = layers · per layer + embeddings
    """
    raise NotImplementedError


# 7 ─────────────────────────────────────────────────────────────────────────────
def weight_gib(spec: ModelSpec, bits: int) -> float:
    """GiB of weights at `bits` per parameter (16 = FP16/BF16, 8 = INT8/FP8, 4 = INT4): params · bits / 8 / 2**30."""
    raise NotImplementedError


# 8 ─────────────────────────────────────────────────────────────────────────────
def kv_bytes_per_token(spec: ModelSpec, kv_bits: int) -> int:
    """KV-cache bytes per token: 2 (K and V) · layers · kv_heads · head_dim · kv_bits / 8 (integer)."""
    raise NotImplementedError


# 9 ─────────────────────────────────────────────────────────────────────────────
def kv_cache_gib(gpu_gib: float, weights_gib: float, engine: str, fraction: float, overhead_gib: float = 1.0) -> float:
    """KV-cache size the server allocates. The SAME fraction means different things:
      "trtllm" (kv_cache_free_gpu_memory_fraction): (gpu − weights − overhead) · fraction
      "vllm"   (gpu_memory_utilization):            gpu · fraction − weights − overhead
    Never below 0. Any other engine → ValueError.
    """
    raise NotImplementedError


# 10 ────────────────────────────────────────────────────────────────────────────
def max_sequences(kv_gib: float, spec: ModelSpec, kv_bits: int, seq_len: int, tokens_per_block: int) -> int:
    """Full-length sequences that fit in a PAGED KV cache.
    blocks per sequence = ceil(seq_len / tokens_per_block)
    total blocks        = floor(kv_gib · 2**30 / (tokens_per_block · kv_bytes_per_token))
    return total blocks // blocks per sequence
    """
    raise NotImplementedError


# 11 ────────────────────────────────────────────────────────────────────────────
def prefill_iterations(prompt_tokens: int, max_num_tokens: int, decode_seqs: int, chunked: bool) -> int:
    """Scheduler iterations needed to prefill one prompt while `decode_seqs` sequences are decoding.
    budget = max_num_tokens − decode_seqs (each decoding sequence uses one token per iteration);
    budget ≤ 0 → ValueError.
    chunked: ceil(prompt_tokens / budget). Not chunked: 1 if the prompt fits the budget, else ValueError.
    """
    raise NotImplementedError


# 12 ────────────────────────────────────────────────────────────────────────────
def estimate_latency(
    spec: ModelSpec, gpu: dict, weight_bits: int, prompt_tokens: int, batch: int, context_tokens: int,
    kv_bits: int = 16, mfu: float = 0.5, bandwidth_efficiency: float = 0.7,
) -> dict[str, float]:
    """Roofline estimate. Return {"prefill_ms", "decode_step_ms", "tokens_per_s"}.
    prefill (compute-bound): 2 · params · prompt_tokens / (tflops_fp16 · 1e12 · mfu), in ms
    decode step (memory-bound): bytes = params · weight_bits / 8 + batch · context_tokens · kv_bytes_per_token
                                ms = bytes / (bandwidth_gbs · 1e9 · bandwidth_efficiency) · 1000
    tokens_per_s = batch / decode_step_ms · 1000
    """
    raise NotImplementedError


# 13 ────────────────────────────────────────────────────────────────────────────
def plan_llm_server(spec: ModelSpec, gpu: dict, workload: dict) -> dict:
    """Size the server. workload keys: engine ("trtllm" | "vllm"), max_prompt, max_output, users,
    weight_bits, kv_bits, fraction, optional tokens_per_block (default 32 trtllm, 16 vllm) and
    prefill_chunk (default 2048).

    max_seq_len    = max_prompt + max_output   (> spec.max_context → ValueError)
    weights, kv    = weight_gib, kv_cache_gib(gpu["memory_gib"], weights, engine, fraction)
    max_sequences  = max_sequences(kv, spec, kv_bits, max_seq_len, tokens_per_block)  (< 1 → ValueError)
    max_batch_size = min(users, max_sequences)
    max_num_tokens = ceil((max_batch_size + prefill_chunk) / 256) · 256

    Return a dict with keys: model (spec.name), engine, max_seq_len, max_batch_size, max_num_tokens,
    fraction, weight_bits, kv_bits, tokens_per_block, chunked_prefill (True), weights_gib and
    kv_cache_gib (rounded to 2 decimals), kv_tokens (floor of kv bytes / bytes per token),
    max_sequences, warnings (list: add one string mentioning both numbers when max_sequences < users),
    predicted = estimate_latency(spec, gpu, weight_bits, max_prompt, max_batch_size, max_seq_len,
    kv_bits) with every value rounded to 1 decimal.
    """
    raise NotImplementedError


# 14 ────────────────────────────────────────────────────────────────────────────
def render_trtllm_serve(plan: dict) -> tuple[list[str], dict]:
    """trtllm-serve flags + the --extra_llm_api_options YAML (as a dict). Plan engine must be
    "trtllm" (else ValueError).
    flags: --backend pytorch --max_batch_size N --max_num_tokens N --max_seq_len N
           --kv_cache_free_gpu_memory_fraction F       (values as strings, in this order)
    extra: {"enable_chunked_prefill": plan's value}, plus {"kv_cache_config": {"dtype": "fp8"}} if kv_bits == 8
    """
    raise NotImplementedError


# 15 ────────────────────────────────────────────────────────────────────────────
def render_vllm_args(plan: dict) -> list[str]:
    """vLLM engine flags. Plan engine must be "vllm" (else ValueError). In this order:
    --max-model-len, --max-num-seqs, --max-num-batched-tokens, --gpu-memory-utilization,
    --kv-cache-dtype (fp8 if kv_bits == 8 else auto), --block-size, then --enable-chunked-prefill if set.
    """
    raise NotImplementedError


# 16 ────────────────────────────────────────────────────────────────────────────
def render_triton_vllm(plan: dict, model_id: str) -> tuple[str, dict]:
    """Triton + vLLM backend: (config.pbtxt text, model.json dict).
    config.pbtxt: backend "vllm" and instance_group [ { count: 1 kind: KIND_MODEL } ]
    model.json:   vLLM engine args with underscores: model, gpu_memory_utilization, max_model_len,
                  max_num_seqs, max_num_batched_tokens, enable_chunked_prefill, block_size,
                  kv_cache_dtype ("fp8"/"auto"), disable_log_stats False (metrics on)
    """
    raise NotImplementedError


# 17 ────────────────────────────────────────────────────────────────────────────
def render_triton_trtllm(plan: dict, model_id: str) -> dict:
    """Triton + TensorRT-LLM LLM-API backend: the model.yaml dict. Plan engine must be "trtllm".
    Keys: model, backend "pytorch", tensor_parallel_size 1, max_batch_size, max_num_tokens, max_seq_len,
    enable_chunked_prefill, kv_cache_config {"free_gpu_memory_fraction": F (+ "dtype": "fp8" if kv_bits 8)},
    triton_config {"max_batch_size": 0, "decoupled": True}.
    """
    raise NotImplementedError


# 18 ────────────────────────────────────────────────────────────────────────────
def effective_sampling(generation_config: dict, request: dict) -> dict:
    """The sampling settings a request really runs with.
    1. start: temperature 1.0, top_p 1.0, top_k 0 (off), repetition_penalty 1.0, max_tokens 16
    2. generation_config.json overrides temperature/top_p/top_k/repetition_penalty if present;
       do_sample False → temperature 0.0
    3. request keys whose value is not None override everything
    4. add "greedy": temperature == 0; when greedy set top_p 1.0 and top_k 0
    """
    raise NotImplementedError


# 19 ────────────────────────────────────────────────────────────────────────────
def generate_request(
    protocol: str, model_name: str, prompt: str, sampling: dict, prompt_tokens: int, max_model_len: int,
    stream: bool = True,
) -> tuple[str, dict]:
    """(URL path, JSON body) for one request. prompt_tokens + max_tokens > max_model_len → ValueError.
    "openai" (trtllm-serve, vLLM, NIM): "/v1/completions", {model, prompt, max_tokens, temperature, top_p,
        stream} + top_k if > 0 + repetition_penalty if != 1.0
    "triton-vllm": "/v2/models/<model_name>/generate_stream" ("/generate" if not stream),
        {text_input, stream, exclude_input_in_output: True, parameters: {max_tokens, temperature, top_p,
        top_k (−1 when 0), repetition_penalty}}
    "triton-trtllm": same paths, {text_input, streaming, sampling_param_max_tokens,
        sampling_param_exclude_input_from_output: True} + greedy → sampling_param_top_k 1, otherwise
        sampling_param_temperature, sampling_param_top_p (+ sampling_param_top_k if > 0)
    anything else → ValueError
    """
    raise NotImplementedError
