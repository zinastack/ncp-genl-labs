"""Drive the Section 4 stack through traffic phases and watch the monitoring react.

    make -C labs/4-deployment-monitoring triton-up
    python labs/4-deployment-monitoring/08-monitoring-reliability/loadgen.py

Phases: steady (low concurrency) → burst (high concurrency, queue time grows) → bad requests
(failures, error-budget burn). After each phase it reads Triton's /metrics with YOUR
parse_prometheus/triton_avg_* functions and prints the Prometheus alerts that are firing.
Keep Grafana (http://localhost:3000) open while it runs.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from solutions import latency_summary, parse_prometheus, triton_avg_batch_size, triton_avg_queue_ms  # noqa: E402

TRITON, PROM = "http://localhost:8000", "http://localhost:9090"
GOOD = json.dumps({"inputs": [{"name": "TEXT", "shape": [1, 1], "datatype": "BYTES",
                               "data": ["The rollout went smoothly and latency dropped."]}]}).encode()
BAD = json.dumps({"inputs": [{"name": "WRONG", "shape": [1, 1], "datatype": "BYTES", "data": ["x"]}]}).encode()


def call(body):
    req = urllib.request.Request(f"{TRITON}/v2/models/text_classifier/infer", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(req, timeout=30).read()
        ok = True
    except urllib.error.HTTPError:
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def phase(name, concurrency, seconds, bad_fraction=0.0):
    deadline = time.time() + seconds
    results = []

    def worker(i):
        out = []
        n = 0
        while time.time() < deadline:
            n += 1
            out.append(call(BAD if bad_fraction and (n % round(1 / bad_fraction) == 0) else GOOD))
        return out

    with ThreadPoolExecutor(concurrency) as pool:
        for r in pool.map(worker, range(concurrency)):
            results += r
    lat = [ms for ms, ok in results if ok]
    errors = sum(not ok for _, ok in results)
    s = latency_summary(lat)
    print(f"\n== {name}: concurrency={concurrency} requests={len(results)} errors={errors} "
          f"rps={len(results) / seconds:.0f} p50={s['p50']:.1f}ms p95={s['p95']:.1f}ms p99={s['p99']:.1f}ms")

    metrics = parse_prometheus(urllib.request.urlopen("http://localhost:8002/metrics").read().decode())
    print(f"   triton (cumulative): avg queue {triton_avg_queue_ms(metrics, 'text_classifier'):.2f} ms, "
          f"avg batch size {triton_avg_batch_size(metrics, 'text_classifier'):.2f}")
    try:
        alerts = json.load(urllib.request.urlopen(f"{PROM}/api/v1/alerts"))["data"]["alerts"]
        for a in alerts:
            print(f"   ALERT [{a['state']}] {a['labels']['alertname']}: {a['annotations'].get('summary', '')}")
        if not alerts:
            print("   no alerts pending/firing")
    except OSError:
        print("   (Prometheus not reachable on :9090)")


if __name__ == "__main__":
    phase("steady", concurrency=2, seconds=60)
    phase("burst", concurrency=64, seconds=90)
    phase("bad requests", concurrency=8, seconds=150, bad_fraction=0.2)
    print("\nOpen Grafana → 'Triton inference (GENL Lab 08)' and match each phase to the panels.")
