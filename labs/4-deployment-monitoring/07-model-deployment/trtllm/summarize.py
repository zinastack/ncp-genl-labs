"""Summarise every TensorRT-LLM benchmark run (results/trtllm/*.json) as one table.

Each run was started by `make trtllm-bench`, which stores the server settings as metadata, so you
can compare TTFT / TPOT / throughput across max_batch_size, max_num_tokens, KV-cache fraction,
backend and quantization.
"""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results" / "trtllm"

if not RUNS.exists() or not any(RUNS.glob("*.json")):
    raise SystemExit("no runs yet: make trtllm-up, then make trtllm-bench")

print(f"{'server settings':<52} {'conc':>4} {'req/s':>6} {'out tok/s':>9} {'TTFT p50':>9} {'TTFT p95':>9} "
      f"{'TPOT ms':>8} {'ITL p95':>8}")
for path in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime):
    r = json.loads(path.read_text())
    meta = " ".join(f"{k}={r[k]}" for k in ("engine", "mbs", "mnt", "kv", "extra") if k in r)
    g = lambda k: float(r.get(k) or 0)
    print(f"{meta:<52} {r.get('max_concurrency', '?'):>4} {g('request_throughput'):>6.1f} {g('output_throughput'):>9.0f} "
          f"{g('median_ttft_ms'):>9.1f} {g('p95_ttft_ms'):>9.1f} {g('mean_tpot_ms'):>8.2f} {g('p95_itl_ms'):>8.2f}")
print("\nTTFT = prefill + queueing (what a chat user waits before the first word); "
      "TPOT/ITL = decode speed per token.")
