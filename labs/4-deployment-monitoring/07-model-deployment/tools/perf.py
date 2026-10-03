"""Run perf_analyzer against one model and RECORD the result next to the config that produced it.

    python tools/perf.py text_classifier --label "no dynamic batching"
    python tools/perf.py distilbert_trt --concurrency 1:16:4
    python tools/perf.py --report                    # every recorded run, side by side

Each run: snapshot Triton's model stats → perf_analyzer (Triton SDK container) → stats again, so the
recorded average batch size is what the dynamic batcher really formed during THIS run. Results are
appended to results/perf_history.csv (git-ignored) with a one-line summary of the live config.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
HISTORY = RESULTS / "perf_history.csv"
URL = "http://localhost:8000"
FIELDS = ["time", "model", "label", "config", "concurrency", "throughput", "p50_ms", "p95_ms", "p99_ms",
          "queue_ms", "compute_ms", "avg_batch"]


def get(path: str) -> dict:
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.load(r)


def config_summary(cfg: dict) -> str:
    """One line describing the knobs this lab turns: batching, instances, backend, accelerators."""
    parts = [cfg.get("platform") or cfg.get("backend", "?"), f"mbs={cfg.get('max_batch_size', 0)}"]
    db = cfg.get("dynamic_batching")
    if db is None:
        parts.append("dyn=off")
    else:
        pref = db.get("preferred_batch_size") or []
        parts.append(f"dyn=on delay={int(db.get('max_queue_delay_microseconds', 0))}us"
                     + (f" pref={','.join(str(p) for p in pref)}" if pref else ""))
    groups = cfg.get("instance_group") or []
    parts.append("inst=" + "+".join(f"{g.get('count', 1)}x{g.get('kind', 'KIND_AUTO').removeprefix('KIND_')}" for g in groups))
    accel = (cfg.get("optimization") or {}).get("execution_accelerators", {}).get("gpu_execution_accelerator")
    if accel:
        parts.append("accel=" + ",".join(a["name"] for a in accel))
    return " ".join(parts)


def counts(model: str) -> tuple[int, int]:
    stats = get(f"/v2/models/{model}/stats")["model_stats"]
    return sum(int(s["inference_count"]) for s in stats), sum(int(s["execution_count"]) for s in stats)


def input_flags(cfg: dict) -> list[str]:
    if any(t["data_type"] == "TYPE_STRING" for t in cfg["input"]):
        return ["--input-data", "/work/triton/perf_inputs.json"]
    # Token-id models: zeros are valid ids ([PAD]) and every request has the same shape (128).
    flags = ["--input-data", "zero"]
    for t in cfg["input"]:
        flags += ["--shape", f"{t['name']}:128"]
    return flags


def run(args: argparse.Namespace) -> None:
    cfg = get(f"/v2/models/{args.model}/config")
    summary = config_summary(cfg)
    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / "perf_last.csv"
    cmd = ["docker", "run", "--rm", "--net=host", "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{LAB}:/work", args.sdk_image, "perf_analyzer", "-m", args.model, "-i", "grpc",
           "-u", "localhost:8001", "--concurrency-range", args.concurrency, "--percentile=95",
           "--measurement-interval", "4000", "-f", "/work/results/perf_last.csv", *input_flags(cfg)]
    if args.batch > 1:
        cmd += ["-b", str(args.batch)]
    print(f"config: {summary}\n$ {' '.join(cmd)}\n", flush=True)
    before = counts(args.model)
    subprocess.run(cmd, check=True)
    after = counts(args.model)
    avg_batch = (after[0] - before[0]) / max(after[1] - before[1], 1)

    rows = []
    with out_csv.open() as f:
        for r in csv.DictReader(f):
            us = lambda k: float(r.get(k) or 0) / 1000  # perf_analyzer reports microseconds
            rows.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model, "label": args.label,
                "config": summary, "concurrency": int(r["Concurrency"]),
                "throughput": round(float(r["Inferences/Second"]), 1),
                "p50_ms": round(us("p50 latency"), 2), "p95_ms": round(us("p95 latency"), 2),
                "p99_ms": round(us("p99 latency"), 2), "queue_ms": round(us("Server Queue"), 2),
                "compute_ms": round(us("Server Compute Infer"), 2), "avg_batch": round(avg_batch, 2),
            })
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print_table(rows)
    print(f"\navg batch formed by Triton during this run: {avg_batch:.2f}   (recorded in {HISTORY.relative_to(LAB)})")


def print_table(rows: list[dict]) -> None:
    print(f"\n{'conc':>4} {'infer/s':>9} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'queue ms':>9} {'compute ms':>11}")
    for r in rows:
        print(f"{r['concurrency']:>4} {r['throughput']:>9} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8} "
              f"{r['queue_ms']:>9} {r['compute_ms']:>11}")


def report() -> None:
    if not HISTORY.exists():
        sys.exit("no runs recorded yet: make s4-perf M=text_classifier")
    runs: dict[tuple, list[dict]] = {}
    with HISTORY.open() as f:
        for r in csv.DictReader(f):
            runs.setdefault((r["time"], r["model"], r["label"], r["config"], r["avg_batch"]), []).append(r)
    for (t, model, label, config, avg_batch), rows in runs.items():
        best = max(rows, key=lambda r: float(r["throughput"]))
        cells = "  ".join(f"c{r['concurrency']}: {float(r['throughput']):.0f}/s p95 {float(r['p95_ms']):.1f}ms" for r in rows)
        print(f"{t}  {model}  [{label or '-'}]\n    {config} · avg batch {avg_batch} · best {float(best['throughput']):.0f}/s\n    {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--label", default="", help="what you changed, e.g. 'delay 100us'")
    ap.add_argument("--concurrency", default="1:16:5", help="start:end:step for perf_analyzer")
    ap.add_argument("--batch", type=int, default=1, help="client-side batch size per request (-b)")
    ap.add_argument("--sdk-image", default=f"nvcr.io/nvidia/tritonserver:{os.environ.get('TRITON_VERSION', '25.08')}-py3-sdk")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print the live config summary of MODEL and exit")
    args = ap.parse_args()
    if args.report:
        report()
    elif args.summary and args.model:
        print("live config:", config_summary(get(f"/v2/models/{args.model}/config")))
    elif args.model:
        run(args)
    else:
        ap.error("give a model name or --report")


if __name__ == "__main__":
    main()
