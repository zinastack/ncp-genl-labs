"""Task 6: detect input and prediction drift on the live model, with YOUR psi / ks_statistic.

Sends two batches of traffic to text_classifier:
  reference  short product-review sentences (what a sentiment model is trained for)
  current    long operational log lines and support tickets (a new client started sending these)
and compares the distributions of input length (input drift) and model confidence and label mix
(prediction drift: no ground-truth labels needed). Writes results/drift.json.

    make s4-drift-08                      # reference solutions
    USE_EXERCISES=1 make s4-drift-08      # your exercises
"""

import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
from common.labimpl import load  # noqa: E402

lab = load(HERE)
URL = "http://localhost:8000/v2/models/text_classifier/infer"
rng = random.Random(0)

ADJ = ["great", "terrible", "fast", "slow", "reliable", "broken", "excellent", "disappointing", "solid", "awful"]
NOUN = ["battery", "screen", "delivery", "support team", "keyboard", "price", "camera", "app", "update", "fan"]
LOG = ["GPU 0 XID 79 fallen off the bus", "pod triton-7c9 restarted (OOMKilled)", "p95 latency 412 ms over SLO",
       "NCCL timeout on rank 3 after 1800 s", "disk usage 91% on /var/lib/docker", "TLS handshake failed for client",
       "queue depth 128 exceeds threshold", "checkpoint saved to s3://models/run-42/step-9000"]


def reference_text() -> str:
    return f"The {rng.choice(NOUN)} is {rng.choice(ADJ)}."


def current_text() -> str:
    lines = rng.sample(LOG, k=rng.randint(3, 6))
    return f"Ticket #{rng.randint(1000, 9999)}: " + "; ".join(lines) + ". Please investigate."


def classify(texts: list[str]) -> tuple[list[str], list[float]]:
    labels, scores = [], []
    for i in range(0, len(texts), 16):  # max_batch_size is 16
        chunk = texts[i:i + 16]
        body = json.dumps({"inputs": [{"name": "TEXT", "shape": [len(chunk), 1], "datatype": "BYTES", "data": chunk}]})
        req = urllib.request.Request(URL, data=body.encode(), headers={"Content-Type": "application/json"})
        out = {o["name"]: o["data"] for o in json.load(urllib.request.urlopen(req, timeout=60))["outputs"]}
        labels += out["LABEL"]
        scores += out["SCORE"]
    return labels, scores


def psi_band(value: float) -> str:
    return "stable" if value < 0.1 else "moderate drift" if value < 0.25 else "SIGNIFICANT drift"


def main() -> None:
    ref = [reference_text() for _ in range(400)]
    cur = [current_text() for _ in range(400)]
    ref_labels, ref_scores = classify(ref)
    cur_labels, cur_scores = classify(cur)

    ref_len = np.array([len(t.split()) for t in ref], dtype=float)
    cur_len = np.array([len(t.split()) for t in cur], dtype=float)
    # Bin edges come from the reference distribution, so lengths far outside it land in the end bins.
    results = {
        "input_length": {"psi": lab.psi(ref_len, cur_len), "ks": lab.ks_statistic(ref_len, cur_len),
                         "mean_ref": ref_len.mean(), "mean_cur": cur_len.mean()},
        "confidence": {"psi": lab.psi(np.array(ref_scores), np.array(cur_scores)),
                       "ks": lab.ks_statistic(np.array(ref_scores), np.array(cur_scores)),
                       "mean_ref": float(np.mean(ref_scores)), "mean_cur": float(np.mean(cur_scores))},
        "positive_rate": {"ref": ref_labels.count("POSITIVE") / len(ref_labels),
                          "cur": cur_labels.count("POSITIVE") / len(cur_labels)},
    }
    for name in ("input_length", "confidence"):
        r = results[name]
        print(f"{name:<13} PSI {r['psi']:.3f} ({psi_band(r['psi'])})  KS {r['ks']:.3f}  "
              f"mean {r['mean_ref']:.2f} → {r['mean_cur']:.2f}")
    pr = results["positive_rate"]
    print(f"label mix     POSITIVE {pr['ref']:.0%} → {pr['cur']:.0%}")
    print("\nDrift is a trigger to investigate (sample the new inputs, run evaluations), not proof that\n"
          "accuracy dropped. Here the inputs aren't sentiment at all: the fix is routing or a different\n"
          "model, not retraining on them blindly.")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "drift.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
