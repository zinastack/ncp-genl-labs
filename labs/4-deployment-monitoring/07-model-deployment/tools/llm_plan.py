"""Task 8: turn the Part B functions into real server configs.

    python tools/llm_plan.py --engine vllm --users 64 --prompt 2048 --output 512
    USE_EXERCISES=1 python tools/llm_plan.py ...      # YOUR functions size and render the configs

Reads llm/models/<model>/config.json, calls model_spec → plan_llm_server → render_* and writes
results/llm/<engine>/:
    plan.json                           the sizing, the predictions and the rendered configs
    trtllm:         serve.args, extra.yml          (trtllm-serve flags, --extra_llm_api_options)
    vllm:           vllm.args                      (vLLM engine flags)
    triton-vllm:    repo/llm/config.pbtxt, repo/llm/1/model.json
    triton-trtllm:  repo/llm/config.pbtxt, repo/llm/1/model.yaml (+ the backend's model.py)
`make llm-serve ENGINE=<engine>` starts the server from exactly these files.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parents[2]))
from common.labimpl import load  # noqa: E402

ENGINES = {"trtllm": "trtllm", "triton-trtllm": "trtllm", "vllm": "vllm", "triton-vllm": "vllm"}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  wrote {path.relative_to(LAB)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=ENGINES, default="vllm")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face id the server loads")
    ap.add_argument("--users", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=2048, help="max prompt tokens")
    ap.add_argument("--output", type=int, default=512, help="max output tokens")
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--kv-bits", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.9, help="KV fraction (trtllm) / GPU utilization (vllm)")
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    args = ap.parse_args()

    lab = load(LAB)
    name = args.model_id.split("/")[-1]
    cfg_path = LAB / "llm" / "models" / name / "config.json"
    if not cfg_path.exists():
        sys.exit(f"no {cfg_path.relative_to(LAB)}: add the model's config.json there first")
    spec = lab.model_spec(json.loads(cfg_path.read_text()), name)
    workload = {"engine": ENGINES[args.engine], "max_prompt": args.prompt, "max_output": args.output,
                "users": args.users, "weight_bits": args.weight_bits, "kv_bits": args.kv_bits,
                "fraction": args.fraction, "prefill_chunk": args.prefill_chunk}
    plan = lab.plan_llm_server(spec, lab.L4, workload)
    plan.update({"server": args.engine, "model_id": args.model_id, "params": spec.params})

    out = LAB / "results" / "llm" / args.engine
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"{spec.name}: {spec.params / 1e9:.2f} B params, {spec.layers} layers, {spec.kv_heads} KV heads × "
          f"{spec.head_dim}, context {spec.max_context}")
    print(f"weights {plan['weights_gib']} GiB · KV cache {plan['kv_cache_gib']} GiB = {plan['kv_tokens']:,} tokens = "
          f"{plan['max_sequences']} sequences of {plan['max_seq_len']}")
    print(f"max_batch_size {plan['max_batch_size']} · max_num_tokens {plan['max_num_tokens']} · "
          f"block {plan['tokens_per_block']} tokens · predicted TTFT (full prompt) {plan['predicted']['prefill_ms']} ms, "
          f"decode step at full batch {plan['predicted']['decode_step_ms']} ms")
    for w in plan["warnings"]:
        print(f"WARNING: {w}")

    if args.engine == "trtllm":
        flags, extra = lab.render_trtllm_serve(plan)
        write(out / "serve.args", " ".join(flags) + "\n")
        write(out / "extra.yml", "# --extra_llm_api_options (JSON is valid YAML)\n" + json.dumps(extra, indent=2) + "\n")
        plan["rendered"] = {"flags": flags, "extra_llm_api_options": extra}
    elif args.engine == "vllm":
        flags = lab.render_vllm_args(plan)
        write(out / "vllm.args", " ".join(flags) + "\n")
        plan["rendered"] = {"flags": flags}
    elif args.engine == "triton-vllm":
        config, model_json = lab.render_triton_vllm(plan, args.model_id)
        write(out / "repo" / "llm" / "config.pbtxt", config)
        write(out / "repo" / "llm" / "1" / "model.json", json.dumps(model_json, indent=2) + "\n")
        plan["rendered"] = {"config.pbtxt": config, "model.json": model_json}
    else:
        model_yaml = lab.render_triton_trtllm(plan, args.model_id)
        template = LAB / "llm" / "triton-trtllm-template"
        repo = out / "repo" / "llm"
        shutil.copytree(template, repo)
        # The template is named tensorrt_llm; Triton requires name == directory name.
        cfg = (repo / "config.pbtxt").read_text().replace('name: "tensorrt_llm"', 'name: "llm"', 1)
        write(repo / "config.pbtxt", cfg)
        write(repo / "1" / "model.yaml", "# LLM API arguments + triton_config (JSON is valid YAML)\n"
              + json.dumps(model_yaml, indent=2) + "\n")
        plan["rendered"] = {"model.yaml": model_yaml}

    write(out / "plan.json", json.dumps(plan, indent=2) + "\n")
    print(f"\nnext: make llm-serve ENGINE={args.engine}")


if __name__ == "__main__":
    main()
