"""Call the Triton text_classifier over HTTP (KServe v2) and measure client-side latency.

    python client.py                         # one request, then a concurrency sweep
    python client.py --url http://host:8000 --concurrency 1 4 16 --requests 400

Uses your kserve_infer_request from solutions.py (swap to exercises.py once you've done it).
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solutions import kserve_infer_request  # noqa: E402

SAMPLES = [
    "The new GPU cluster cut our training time in half.",
    "Deployment failed again and the on-call engineer is furious.",
    "Latency is within the SLO after enabling dynamic batching.",
    "The model keeps timing out under load.",
]


def infer(url, texts):
    body = json.dumps(kserve_infer_request(texts)).encode()
    req = urllib.request.Request(f"{url}/v2/models/text_classifier/infer", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def sweep(url, concurrency, n_requests):
    def one(i):
        t0 = time.perf_counter()
        infer(url, [SAMPLES[i % len(SAMPLES)]])
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        lat = list(pool.map(one, range(n_requests)))
    elapsed = time.perf_counter() - t0
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"concurrency={concurrency:<3} throughput={n_requests / elapsed:7.1f} req/s  "
          f"p50={p50:6.1f} ms  p95={p95:6.1f} ms  p99={p99:6.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()

    out = infer(args.url, SAMPLES)
    labels = {o["name"]: o["data"] for o in out["outputs"]}
    for text, label, score in zip(SAMPLES, labels["LABEL"], labels["SCORE"]):
        print(f"{label:>8} {score:.3f}  {text}")
    print()
    for c in args.concurrency:
        sweep(args.url, c, args.requests)
    print("\nNow compare with Triton's server-side view: curl -s localhost:8002/metrics | grep nv_inference_")


if __name__ == "__main__":
    main()
