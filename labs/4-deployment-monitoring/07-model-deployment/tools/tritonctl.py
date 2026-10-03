"""Tiny Triton control CLI over the HTTP API (KServe v2 + Triton's repository extension).

    python tools/tritonctl.py index                    # every model in the repository and its state
    python tools/tritonctl.py load distilbert_onnx     # load, or reload after editing config.pbtxt
    python tools/tritonctl.py unload distilbert_onnx
    python tools/tritonctl.py config text_classifier   # the config Triton is actually using
    python tools/tritonctl.py stats text_classifier    # cumulative counts, queue and compute time
    python tools/tritonctl.py infer sentiment_ensemble "The rollout went smoothly"
    python tools/tritonctl.py first-request distilbert_trt   # cold load + first request vs warm request
"""

import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://localhost:8000"


def call(method: str, path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} → HTTP {e.code}: {e.read().decode()}")
    return json.loads(raw) if raw else {}


def infer_body(model: str, text: str, batch: int = 1) -> dict:
    """A KServe v2 request: strings for text models, zero token ids (length 128) for id models."""
    cfg = call("GET", f"/v2/models/{model}/config")
    inputs = []
    for t in cfg["input"]:
        if t["data_type"] == "TYPE_STRING":
            inputs.append({"name": t["name"], "shape": [batch, 1], "datatype": "BYTES", "data": [text] * batch})
        else:
            inputs.append({"name": t["name"], "shape": [batch, 128], "datatype": "INT32", "data": [0] * (batch * 128)})
    return {"inputs": inputs}


def timed_infer(model: str, text: str) -> tuple[float, dict]:
    body = infer_body(model, text)
    t0 = time.perf_counter()
    out = call("POST", f"/v2/models/{model}/infer", body)
    return (time.perf_counter() - t0) * 1000, out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "index":
        for m in call("POST", "/v2/repository/index"):
            print(f"{m['name']:<22} v{m.get('version', '-'):<3} {m.get('state', 'UNAVAILABLE'):<12} {m.get('reason', '')}")
    elif cmd in ("load", "unload"):
        t0 = time.perf_counter()
        call("POST", f"/v2/repository/models/{args[0]}/{cmd}")
        print(f"{cmd}ed {args[0]} in {time.perf_counter() - t0:.1f} s")
    elif cmd == "config":
        print(json.dumps(call("GET", f"/v2/models/{args[0]}/config"), indent=2))
    elif cmd == "stats":
        for s in call("GET", f"/v2/models/{args[0]}/stats")["model_stats"]:
            st = s["inference_stats"]
            n = max(int(st["success"]["count"]), 1)
            print(f"{s['name']} v{s['version']}: requests={st['success']['count']} failures={st['fail']['count']} "
                  f"inferences={s['inference_count']} executions={s['execution_count']} "
                  f"avg_batch={int(s['inference_count']) / max(int(s['execution_count']), 1):.2f} "
                  f"avg_queue={int(st['queue']['ns']) / n / 1e6:.2f} ms "
                  f"avg_compute={int(st['compute_infer']['ns']) / n / 1e6:.2f} ms")
    elif cmd == "infer":
        ms, out = timed_infer(args[0], args[1] if len(args) > 1 else "The new GPU cluster cut our training time in half.")
        print(json.dumps({o["name"]: o["data"] for o in out["outputs"]}), f"({ms:.1f} ms)")
    elif cmd == "first-request":
        model = args[0]
        call("POST", f"/v2/repository/models/{model}/unload")
        time.sleep(2)
        t0 = time.perf_counter()
        call("POST", f"/v2/repository/models/{model}/load")
        load_s = time.perf_counter() - t0
        first, _ = timed_infer(model, "first")
        warm = sorted(timed_infer(model, "warm")[0] for _ in range(5))[2]
        print(f"{model}: load {load_s:.1f} s · first request {first:.1f} ms · warm request (median of 5) {warm:.1f} ms")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
