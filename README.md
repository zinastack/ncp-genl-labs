# NCP-GENL Labs

Hands-on labs for the **NVIDIA-Certified Professional: Generative AI LLMs (NCP-GENL)** exam
([blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)).
Every lab combines three things:

1. **Review notes** (`README.md`): the concepts, numbers and exam traps for one blueprint domain.
2. **Exercises you implement** (`exercises.py`), checked by tests. They run on a laptop CPU in seconds.
   **`SOLUTION.md`** then explains every solution step by step, with worked numbers.
3. **Exam-style questions** (`quiz.toml`): 326 scenario questions with explanations, including a timed mock exam.

Each lab also has a **GPU part** (`gpu_lab.py` plus section `make` targets) that runs your code on
real models on a cheap cloud GPU (L4/T4 via Brev or AWS). See [brev/README.md](brev/README.md).

## Launched from Brev? Start here

Each section has its own Brev Launchable with a cheap GPU (1× L4; 2× L4 for Section 3).
The setup script has already cloned this repo to `/home/ubuntu/ncp-genl-labs`, created `.venv`,
installed the section's GPU packages and generated a Jupyter notebook next to every `gpu_lab.py`.
Setup takes a few minutes after the instance starts.

Open **Jupyter** (or a terminal) from the instance page, then:

```bash
cd ~/ncp-genl-labs
make doctor             # GPU, driver, PyTorch, Docker
make s1                 # list the section's targets (s1…s5)
make test-01            # your exercises vs the tests (CPU is fine)
make gpu-01             # the lab's GPU part; or open labs/…/01-llm-architecture/gpu_lab.ipynb
make quiz-01            # exam-style questions for the lab
```

| Launchable | Labs | GPU | Extra links |
|---|---|---|---|
| S1 Foundations & Prompting | 01, 02 | 1× L4 | |
| S2 Data & Fine-Tuning | 03, 04 | 1× L4 | |
| S3 Optimization & Acceleration | 05, 06 | 2× L4 | |
| S4 Deployment & Monitoring | 07, 08 | 1× L4 | `grafana` (3000), `prometheus` (9090) |
| S5 Evaluation & Responsible AI | 09, 10 | 1× L4 | |

All models are public. You can optionally set `HF_TOKEN` when you deploy for faster Hugging Face downloads.
**Stop the instance when you're done** because billing is per hour.

## Sections and blueprint coverage

| Section | Lab | Blueprint domain | Weight | GPU part |
|---|---|---|---|---|
| **1 · Foundations & Prompting** | [01](labs/1-foundations-prompting/01-llm-architecture) | LLM Architecture | 6% | real KV cache vs your formula; FlashAttention vs math attention |
| | [02](labs/1-foundations-prompting/02-prompt-engineering) | Prompt Engineering | 13% | your sampler/constrained decoder on Qwen2.5; zero vs few-shot; self-consistency |
| **2 · Data & Fine-Tuning** | [03](labs/2-data-finetuning/03-data-preparation) | Data Preparation | 9% | curate a corpus: MinHash+LSH, GPU semantic dedup, tokenizer fertility, packing |
| | [04](labs/2-data-finetuning/04-fine-tuning) | Fine-Tuning | 13% | full FT vs LoRA vs QLoRA memory; LoRA SFT with your collator and loss |
| **3 · Optimization & Acceleration** | [05](labs/3-optimization-acceleration/05-model-optimization) | Model Optimization | 17% | tensor-core TFLOPS, decode roofline, checkpointing, INT8/NF4, speculative decoding |
| | [06](labs/3-optimization-acceleration/06-gpu-acceleration) | GPU Acceleration & Optimization | 14% | NCCL all-reduce bandwidth, DDP scaling, DDP vs FSDP, Nsight Systems |
| **4 · Deployment & Monitoring** | [07](labs/4-deployment-monitoring/07-model-deployment) | Model Deployment | 9% | hands-on tasks: Triton batching/instances/ensembles, TensorRT, TensorRT-LLM, NIM, Kubernetes (k3s) GPU scheduling, HPA, rollouts |
| | [08](labs/4-deployment-monitoring/08-monitoring-reliability) | Production Monitoring & Reliability | 7% | hands-on tasks: Prometheus + Grafana + DCGM, alerts that fire, failure drills, drift, canary |
| **5 · Evaluation & Responsible AI** | [09](labs/5-evaluation-responsible-ai/09-evaluation) | Evaluation | 7% | perplexity, retrieval vs reranking, bootstrap CIs, LLM-judge κ, lm-eval-harness |
| | [10](labs/5-evaluation-responsible-ai/10-safety-ethics) | Safety, Ethics & Compliance | 5% | counterfactual bias probe, red-team ASR, NeMo Guardrails |

## Quick start (laptop, no GPU needed)

```bash
git clone https://github.com/zinastack/ncp-genl-labs.git && cd ncp-genl-labs
make setup              # .venv + numpy/torch/pytest (Python ≥ 3.10)
make help               # everything you can do from the root
make test-01            # run Lab 01's tests against YOUR exercises (all fail until you implement them)
make solutions-01       # the same tests against the reference solutions
make quiz-01            # exam questions for domain 01
```

The loop for each lab:

1. Read `labs/<section>/<lab>/README.md`.
2. Implement `exercises.py` until `make test-NN` is green. When stuck, or after finishing each
   exercise, read its section in `SOLUTION.md` (why each line exists, worked numbers, exam link).
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
| `make brev-up S=3` / `brev-shell` / `brev-stop` | private Brev GPU instance from your laptop via the Brev CLI (see [brev/README.md](brev/README.md)) |

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
    SOLUTION.md          step-by-step walkthrough of every exercise, with worked numbers
    solutions.py         reference implementation (commented)
    test_lab.py          checks for both
    quiz.toml            exam-style questions
    gpu_lab.py           GPU part (# %% cells: run as a script, in VS Code, or as a generated notebook)
```

## Study resources

Every lab README ends with a **Further reading** list for its domain. These resources cover several domains:

- [NCP-GENL exam page and blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/) (NVIDIA)
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face): training at scale (Labs 05, 06)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google DeepMind): memory, FLOPs and rooflines (Labs 05, 06)
- [Machine Learning Engineering](https://github.com/stas00/ml-engineering) (Stas Bekman): GPUs, networking and debugging (Labs 06, 08)
- [Lilian Weng's blog](https://lilianweng.github.io/): long surveys on transformers, prompting, inference optimization and adversarial attacks
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Sebastian Raschka, book)

## Exam facts

60–70 questions · 120 minutes · $200 · valid 2 years · recommended experience: 2–3 years with
LLMs, transformers, prompt engineering, distributed parallelism and PEFT.

## License

[MIT](LICENSE). This is an independent study resource, not affiliated with or endorsed by NVIDIA.
The practice questions are original and are not taken from the certification exam.
