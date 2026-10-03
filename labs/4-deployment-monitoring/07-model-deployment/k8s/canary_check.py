"""Task K6: canary analysis from Prometheus, decided by YOUR canary_verdict (Lab 08).

Reads, for the last 2 minutes, p95 latency and error rate per version (v1 = stable, v2 = canary)
from the in-cluster Prometheus (NodePort 30900), then prints promote / rollback.
Both versions are measured over the SAME window: that is what makes the comparison fair.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from common.labimpl import load  # noqa: E402

PROM = "http://localhost:30900"
lab08 = load(ROOT / "labs/4-deployment-monitoring/08-monitoring-reliability")


def query(promql: str) -> dict[str, float]:
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    data = json.load(urllib.request.urlopen(url, timeout=10))["data"]["result"]
    return {r["metric"].get("version", "?"): float(r["value"][1]) for r in data}


# Summaries can't be merged across pods (quantiles don't add up), so take the WORST pod per version.
p95 = query('max by (version) (nv_inference_request_summary_us{model="text_classifier",quantile="0.95"}) / 1000')
ok = query('sum by (version) (rate(nv_inference_request_success{model="text_classifier"}[2m]))')
bad = query('sum by (version) (rate(nv_inference_request_failure{model="text_classifier"}[2m]))')

if not {"v1", "v2"} <= p95.keys():
    sys.exit(f"need traffic on both versions first (make k8s-load); got p95 for {sorted(p95)}")
metrics = {v: {"p95_ms": p95[v], "error_rate": bad.get(v, 0) / max(ok.get(v, 0) + bad.get(v, 0), 1e-9)}
           for v in ("v1", "v2")}
share = ok.get("v2", 0) / max(ok.get("v1", 0) + ok.get("v2", 0), 1e-9)
for v, m in metrics.items():
    print(f"{v}: p95 {m['p95_ms']:.1f} ms, error rate {m['error_rate']:.2%}")
print(f"canary traffic share: {share:.0%}")
print("verdict:", lab08.canary_verdict(metrics["v1"], metrics["v2"]))
