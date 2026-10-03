"""Which Lab 08 tasks have you done? Asks Prometheus which alerts have fired, and checks results.

    make s4-check-08
    make s4-alerts          # alerts pending/firing now + every alert that fired in the last 3 hours
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PROM = "http://localhost:9090"


def query(promql: str) -> list[dict]:
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    return json.load(urllib.request.urlopen(url, timeout=10))["data"]["result"]


def fired(hours: int = 3) -> set[str]:
    # ALERTS is a synthetic series Prometheus writes while an alert is pending or firing.
    return {r["metric"]["alertname"] for r in query(f'max_over_time(ALERTS{{alertstate="firing"}}[{hours}h])')}


def main() -> None:
    try:
        ever = fired()
    except OSError:
        sys.exit("Prometheus is not reachable on :9090: make s4-triton-up")
    if "--alerts" in sys.argv:
        now = json.load(urllib.request.urlopen(f"{PROM}/api/v1/alerts", timeout=10))["data"]["alerts"]
        print("now:", ", ".join(f"{a['labels']['alertname']} [{a['state']}]" for a in now) or "none")
        print("fired in the last 3 h:", ", ".join(sorted(ever)) or "none")
        return
    targets = {r["metric"]["job"]: r["value"][1] for r in query("up")}
    burst = query('max_over_time(triton:avg_queue_ms[3h])')
    checks = [
        ("1 Raw metrics → averages", targets.get("triton") == "1", "Prometheus must be scraping Triton (make s4-metrics-08)"),
        ("2 PromQL + percentiles", bool(query("nv_inference_request_summary_us")), "summary latencies enabled and scraped"),
        ("3 Load phases", bool(burst) and float(burst[0]["value"][1]) > 1, "run make s4-gpu-08 (queue time must rise in the burst)"),
        ("4a Queue-time alert fired", "TritonQueueTimeHigh" in ever, "overload Triton until TritonQueueTimeHigh fires"),
        ("4b Error-budget alert fired", "TritonErrorBudgetFastBurn" in ever, "the bad-requests phase of make s4-gpu-08"),
        ("5a Outage detected", "TritonDown" in ever, "docker compose stop triton for > 30 s, then start it"),
        ("5b GPU memory alert fired", "GPUMemoryNearlyFull" in ever, "make s4-gpu-hog"),
        ("6 Drift measured", (HERE / "results" / "drift.json").exists(), "make s4-drift-08"),
    ]
    for name, ok, hint in checks:
        print(f"  {'✔' if ok else '·'} {name:<30} {'' if ok else hint}")
    print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} done")


if __name__ == "__main__":
    main()
