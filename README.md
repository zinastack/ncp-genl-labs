# NCP-GENL Labs

Hands-on labs for the **NVIDIA-Certified Professional: Generative AI LLMs (NCP-GENL)** exam
([blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)).
Every lab combines three things:

1. **Review notes** (`README.md`): the concepts, numbers and exam traps for one blueprint domain.
2. **Exercises you implement** (`exercises.py`), checked by tests. They run on a laptop CPU in seconds.
3. **Exam-style questions** (`quiz.toml`): 138 scenario questions with explanations, including a timed mock exam.

Each lab also has a **GPU part** (`gpu_lab.py` plus section `make` targets) that runs your code on
real models on a cheap cloud GPU (L4/T4 via Brev or AWS). See [brev/README.md](brev/README.md).

## Sections and blueprint coverage

| Section | Lab | Blueprint domain | Weight | GPU part |
|---|---|---|---|---|
| **1 · Foundations & Prompting** | [01](labs/1-foundations-prompting/01-llm-architecture) | LLM Architecture | 6% | real KV cache vs your formula; FlashAttention vs math attention |
| | [02](labs/1-foundations-prompting/02-prompt-engineering) | Prompt Engineering | 13% | your sampler/constrained decoder on Qwen2.5; zero vs few-shot; self-consistency |
| **2 · Data & Fine-Tuning** | [03](labs/2-data-finetuning/03-data-preparation) | Data Preparation | 9% | curate a corpus: MinHash+LSH, GPU semantic dedup, tokenizer fertility, packing |
| | [04](labs/2-data-finetuning/04-fine-tuning) | Fine-Tuning | 13% | full FT vs LoRA vs QLoRA memory; LoRA SFT with your collator and loss |
| **3 · Optimization & Acceleration** | [05](labs/3-optimization-acceleration/05-model-optimization) | Model Optimization | 17% | tensor-core TFLOPS, decode roofline, checkpointing, INT8/NF4, speculative decoding |
| | [06](labs/3-optimization-acceleration/06-gpu-acceleration) | GPU Acceleration & Optimization | 14% | NCCL all-reduce bandwidth, DDP scaling, DDP vs FSDP, Nsight Systems |
| **4 · Deployment & Monitoring** | [07](labs/4-deployment-monitoring/07-model-deployment) | Model Deployment | 9% | Triton (Python backend) + dynamic batching + perf_analyzer; K8s manifests |
| | [08](labs/4-deployment-monitoring/08-monitoring-reliability) | Production Monitoring & Reliability | 7% | Prometheus + Grafana + DCGM on your Triton, load phases, alerts |
| **5 · Evaluation & Responsible AI** | [09](labs/5-evaluation-responsible-ai/09-evaluation) | Evaluation | 7% | perplexity, retrieval vs reranking, bootstrap CIs, LLM-judge κ, lm-eval-harness |
| | [10](labs/5-evaluation-responsible-ai/10-safety-ethics) | Safety, Ethics & Compliance | 5% | counterfactual bias probe, red-team ASR, NeMo Guardrails |

## Quick start (laptop, no GPU needed)

```bash
make setup              # .venv + numpy/torch/pytest (Python ≥ 3.10)
make help               # everything you can do from the root
make test-01            # run Lab 01's tests against YOUR exercises (all fail until you implement them)
make solutions-01       # the same tests against the reference solutions
make quiz-01            # exam questions for domain 01
```

The loop for each lab:

1. Read `labs/<section>/<lab>/README.md`.
2. Implement `exercises.py` until `make test-NN` is green. Check `solutions.py` only when stuck.
3. `make quiz-NN` until you are consistently above 80%.
4. On a GPU instance: `make gpu-NN` (or `USE_EXERCISES=1 make gpu-NN` to drive real models with your code).

Then `make mock` (65 questions, 120 minutes, weighted like the exam), `make review` (only the
questions you missed) and `make stats` (accuracy per domain).

## Makefiles

The root `Makefile` is the entry point. Each section has its own `labs/<section>/Makefile`
(shared rules in `mk/common.mk`). From the root:

| Command | What it does |
|---|---|
| `make test-NN` / `solutions-NN` / `quiz-NN` / `gpu-NN` | per-lab shortcuts (NN = 01…10); the root finds the right section |
| `make s<N>` | list Section N's targets |
| `make s<N>-<target>` | run any Section N target, e.g. `make s3-profile-06`, `make s4-triton-up`, `make s5-lm-eval` |
| `make setup-gpu SECTION=3` | install one section's GPU packages (`all` by default) |
| `make brev-setup SECTION=3` | full instance setup (what the Brev Launchable runs) |

Or work inside a section: `cd labs/3-optimization-acceleration && make help`.

## Layout

```
Makefile                 root entry point → delegates to labs/<section>/Makefile
mk/common.mk             rules shared by the section Makefiles
quiz.py                  quiz runner (practice, --mock, --review, --stats); stdlib only
conftest.py, pytest.ini  `lab` fixture switches between exercises.py and solutions.py (--solutions)
common/                  device/precision helpers, OpenAI-compatible LLM client, impl loader
brev/                    setup.sh (Brev Launchable / AWS) and the instance guide
tests/                   quiz-bank validation (ids, answers, "Select N" wording)
labs/<N>-<section>/
  Makefile               section targets (test, quiz, gpu-NN, section-specific GPU targets)
  requirements-gpu.txt   GPU extras for that section
  <NN>-<lab>/
    README.md            review notes
    exercises.py         ← you implement
    solutions.py         reference implementation
    test_lab.py          checks for both
    quiz.toml            exam-style questions
    gpu_lab.py           GPU part (# %% cells: run as a script, in VS Code, or as a generated notebook)
```

## Exam facts

60–70 questions · 120 minutes · $200 · valid 2 years · recommended experience: 2–3 years with
LLMs, transformers, prompt engineering, distributed parallelism and PEFT.
