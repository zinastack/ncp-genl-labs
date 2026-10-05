"""Task 8: benchmark the running LLM server with requests built by YOUR generate_request.

    python tools/llm_bench.py --engine vllm --concurrency 1 8 32
    python tools/llm_bench.py --report          # every run: measured vs predicted, all engines

Streams every response, so it measures what a chat user feels:
    TTFT  time to first token (queueing + prefill)
    TPOT  time per output token after the first (decode step time)
and compares them with estimate_latency() for the same prompt length, batch and context.
Results go to results/llm/bench.csv.
"""

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parents[2]))
from common.labimpl import load  # noqa: E402

URL = "http://localhost:8010"
BENCH = LAB / "results" / "llm" / "bench.csv"
PROTOCOL = {"trtllm": "openai", "vllm": "openai", "triton-vllm": "triton-vllm", "triton-trtllm": "triton-trtllm"}
FIELDS = ["time", "engine", "settings", "concurrency", "prompt_tokens", "requests", "ttft_p50_ms", "ttft_p95_ms",
          "tpot_ms", "out_tokens_per_s", "pred_prefill_ms", "pred_step_ms"]
WORDS = "the model serves requests on one GPU while the scheduler batches tokens every step".split()

lab = load(LAB)


def prompt_of(n: int, i: int) -> str:
    # Common lowercase words are ~1 token each; the request number keeps prompts distinct (no cache hits).
    words = [WORDS[(i + k) % len(WORDS)] for k in range(n - 8)]
    return f"Request {i}. Summarise this text: " + " ".join(words)


def one_request(engine: str, model_name: str, max_model_len: int, prompt_tokens: int, max_tokens: int, i: int) -> dict:
    sampling = lab.effective_sampling({}, {"temperature": 0, "max_tokens": max_tokens})
    path, body = lab.generate_request(PROTOCOL[engine], model_name, prompt_of(prompt_tokens, i), sampling,
                                      prompt_tokens, max_model_len, stream=True)
    # Benchmark-only: keep generating to max_tokens so every request decodes the same length.
    if PROTOCOL[engine] == "openai":
        body["ignore_eos"] = True
    elif engine == "triton-vllm":
        body["parameters"]["ignore_eos"] = True
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, chunks = None, 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            data = json.loads(line[5:])
            text = data["choices"][0].get("text", "") if "choices" in data else data.get("text_output", "")
            if text:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    if first is None:
        raise RuntimeError(f"no tokens streamed back from {path}")
    return {"ttft_ms": (first - t0) * 1000, "tpot_ms": (end - first) * 1000 / max(chunks - 1, 1), "tokens": chunks,
            "seconds": end - t0}


def run(args: argparse.Namespace) -> None:
    plan_path = LAB / "results" / "llm" / args.engine / "plan.json"
    if not plan_path.exists():
        sys.exit(f"no plan for {args.engine}: make llm-plan ENGINE={args.engine}")
    plan = json.loads(plan_path.read_text())
    name = plan["model_id"].split("/")[-1]
    spec = lab.model_spec(json.loads((LAB / "llm" / "models" / name / "config.json").read_text()), name)
    model_name = "llm" if args.engine.startswith("triton") else plan["model_id"]
    settings = (f"mbs={plan['max_batch_size']} mnt={plan['max_num_tokens']} len={plan['max_seq_len']} "
                f"frac={plan['fraction']} kv{plan['kv_bits']}")
    rows = []
    for conc in args.concurrency:
        n = max(conc * args.rounds, conc)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as pool:
            res = list(pool.map(lambda i: one_request(args.engine, model_name, plan["max_seq_len"], args.prompt_tokens,
                                                      args.max_tokens, i), range(n)))
        wall = time.perf_counter() - t0
        ttft = sorted(r["ttft_ms"] for r in res)
        pred = lab.estimate_latency(spec, lab.L4, plan["weight_bits"], args.prompt_tokens, conc,
                                    args.prompt_tokens + args.max_tokens, plan["kv_bits"])
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": args.engine, "settings": settings,
               "concurrency": conc, "prompt_tokens": args.prompt_tokens, "requests": n,
               "ttft_p50_ms": round(statistics.median(ttft), 1), "ttft_p95_ms": round(ttft[int(0.95 * (len(ttft) - 1))], 1),
               "tpot_ms": round(statistics.mean(r["tpot_ms"] for r in res), 2),
               "out_tokens_per_s": round(sum(r["tokens"] for r in res) / wall, 1),
               "pred_prefill_ms": round(pred["prefill_ms"], 1), "pred_step_ms": round(pred["decode_step_ms"], 2)}
        rows.append(row)
        print(f"{args.engine:<14} c={conc:<3} TTFT p50 {row['ttft_p50_ms']:>7} ms (pred. prefill {row['pred_prefill_ms']}) "
              f"p95 {row['ttft_p95_ms']:>7} · TPOT {row['tpot_ms']:>6} ms (pred. {row['pred_step_ms']}) · "
              f"{row['out_tokens_per_s']:>7} tok/s", flush=True)
    new = not BENCH.exists()
    BENCH.parent.mkdir(parents=True, exist_ok=True)
    with BENCH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def report() -> None:
    if not BENCH.exists():
        sys.exit("no runs yet: make llm-bench ENGINE=...")
    print(f"{'engine':<14} {'settings':<44} {'conc':>4} {'TTFT p50':>9} {'pred':>6} {'TTFT p95':>9} "
          f"{'TPOT':>6} {'pred':>6} {'tok/s':>7}")
    with BENCH.open() as f:
        for r in csv.DictReader(f):
            print(f"{r['engine']:<14} {r['settings']:<44} {r['concurrency']:>4} {r['ttft_p50_ms']:>9} "
                  f"{r['pred_prefill_ms']:>6} {r['ttft_p95_ms']:>9} {r['tpot_ms']:>6} {r['pred_step_ms']:>6} "
                  f"{r['out_tokens_per_s']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=PROTOCOL)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3, help="requests per concurrency level = rounds × concurrency")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    elif args.engine:
        run(args)
    else:
        ap.error("give --engine or --report")


if __name__ == "__main__":
    main()
