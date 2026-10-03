"""Drive Triton through traffic phases and watch the monitoring react.

    make s4-gpu-08                                   # steady → burst → bad requests (Compose stack)
    python loadgen.py --phases "burst:64:120"        # your own phases: name:concurrency:seconds[:bad_fraction]
    python loadgen.py --url http://localhost:30800 --metrics-url http://localhost:30802/metrics --no-alerts

After each phase it reads Triton's /metrics with parse_prometheus / triton_avg_* (the reference
solutions, or YOUR exercises with USE_EXERCISES=1) and prints the Prometheus alerts that are firing.
Keep Grafana (http://localhost:3000) open while it runs.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
from common.labimpl import load  # noqa: E402

lab = load(HERE)
PROM = "http://localhost:9090"
GOOD = json.dumps({"inputs": [{"name": "TEXT", "shape": [1, 1], "datatype": "BYTES",
                               "data": ["The rollout went smoothly and latency dropped."]}]}).encode()
BAD = json.dumps({"inputs": [{"name": "WRONG", "shape": [1, 1], "datatype": "BYTES", "data": ["x"]}]}).encode()
DEFAULT_PHASES = "steady:2:60,burst:64:90,bad requests:8:150:0.2"


def call(url: str, body: bytes) -> tuple[float, bool]:
    req = urllib.request.Request(f"{url}/v2/models/text_classifier/infer", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(req, timeout=30).read()
        ok = True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError):
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def phase(args: argparse.Namespace, name: str, concurrency: int, seconds: float, bad_fraction: float = 0.0) -> None:
    deadline = time.time() + seconds
    every = round(1 / bad_fraction) if bad_fraction else 0

    def worker(_: int) -> list[tuple[float, bool]]:
        out, n = [], 0
        while time.time() < deadline:
            n += 1
            out.append(call(args.url, BAD if every and n % every == 0 else GOOD))
        return out

    results: list[tuple[float, bool]] = []
    with ThreadPoolExecutor(concurrency) as pool:
        for r in pool.map(worker, range(concurrency)):
            results += r
    lat = [ms for ms, ok in results if ok]
    errors = sum(not ok for _, ok in results)
    line = f"\n== {name}: concurrency={concurrency} requests={len(results)} errors={errors} rps={len(results) / seconds:.0f}"
    if lat:
        s = lab.latency_summary(lat)
        line += f" p50={s['p50']:.1f}ms p95={s['p95']:.1f}ms p99={s['p99']:.1f}ms"
    print(line)
    report(args)


def report(args: argparse.Namespace) -> None:
    try:
        metrics = lab.parse_prometheus(urllib.request.urlopen(args.metrics_url, timeout=10).read().decode())
        print(f"   triton since start (cumulative counters): avg queue {lab.triton_avg_queue_ms(metrics, 'text_classifier'):.2f} ms, "
              f"avg batch size {lab.triton_avg_batch_size(metrics, 'text_classifier'):.2f}")
    except OSError as e:
        print(f"   (metrics not reachable at {args.metrics_url}: {e})")
    if args.no_alerts:
        return
    try:
        alerts = json.load(urllib.request.urlopen(f"{PROM}/api/v1/alerts", timeout=10))["data"]["alerts"]
        for a in alerts:
            print(f"   ALERT [{a['state']}] {a['labels']['alertname']}: {a['annotations'].get('summary', '')}")
        if not alerts:
            print("   no alerts pending/firing")
    except OSError:
        print("   (Prometheus not reachable on :9090)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--metrics-url", default="http://localhost:8002/metrics")
    ap.add_argument("--phases", default=DEFAULT_PHASES, help="comma-separated name:concurrency:seconds[:bad_fraction]")
    ap.add_argument("--no-alerts", action="store_true", help="skip the Prometheus alert check (Kubernetes)")
    ap.add_argument("--metrics-only", action="store_true", help="send the phases, then print the raw metric lines too")
    args = ap.parse_args()
    for spec in args.phases.split(","):
        name, conc, secs, *bad = spec.split(":")
        phase(args, name, int(conc), float(secs), float(bad[0]) if bad else 0.0)
    if args.metrics_only:
        text = urllib.request.urlopen(args.metrics_url, timeout=10).read().decode()
        print("\nRaw counters behind those averages:")
        for line in text.splitlines():
            if line.startswith(("nv_inference_request_success", "nv_inference_count", "nv_inference_exec_count",
                                "nv_inference_queue_duration_us", "nv_inference_compute_infer_duration_us",
                                "nv_inference_request_summary_us")) and 'model="text_classifier"' in line:
                print("  ", line)
    else:
        print("\nOpen Grafana → 'Triton inference (GENL Lab 08)' and match each phase to the panels.")


if __name__ == "__main__":
    main()
