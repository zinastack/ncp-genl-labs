# Lab 02 — Prompt Engineering (13% of exam)

> Blueprint: *domain adaptation, chain-of-thought techniques, zero/one/few-shot learning, output control.*

The exercises build the machinery behind prompting: few-shot prompt assembly, chat templates,
the sampling knobs (temperature, top-k, top-p, penalties), constrained decoding,
self-consistency voting, dynamic example selection and robust JSON parsing. `gpu_lab.py` then
drives Qwen2.5 on the GPU with *your* sampling and constrained-decoding code.

```
make test-02          # YOUR exercises (run from the repo root)
make solutions-02     # reference solutions
make quiz-02          # exam-style questions
make gpu-02           # GPU part (gpu_lab.py) on the Brev/AWS instance
make s1-llm-up        # optional: local vLLM server used by the last cell of gpu-02
```

---

## 1. The prompting toolbox

| Technique | What it is | When it wins |
|---|---|---|
| **Zero-shot** | Instruction only, no examples. Works because of instruction tuning / RLHF. | Common tasks the model already "knows" (sentiment, summarise, translate). |
| **One-shot** | One demonstration. Mainly shows the *format*. | Output format matters more than reasoning. |
| **Few-shot (in-context learning)** | k demonstrations (typically 3–8). No weight updates. | Domain-specific labels, style, edge cases; little or no labelled data. |
| **Chain-of-Thought (CoT)** | Ask for intermediate reasoning. Few-shot CoT uses worked examples; zero-shot CoT uses "Let's think step by step". | Arithmetic, multi-step logic, explainable decisions. Helps large models most. |
| **Self-consistency** | Sample N CoT paths at temperature > 0, then **majority-vote** the final answers. | Reasoning accuracy when you can afford N× compute. |
| **Tree-of-Thought** | Search over branching partial reasoning, with evaluation and backtracking. | Planning and puzzles. Expensive. |
| **ReAct** | Interleave *Thought → Action (tool call) → Observation*. | Agents and tool use; grounds reasoning in retrieved facts. |
| **Least-to-most / decomposition** | Break the problem into sub-questions solved in order. | Compositional generalisation. |
| **Role / system prompt** | Persona, rules, tone, safety policy. | Consistency across a product. |
| **RAG** | Put retrieved passages in the prompt and require citations. | Fresh or proprietary knowledge; reduces hallucination. |

**Prompt anatomy that scores well:** role, then task instruction, then constraints and policy,
then examples, then the input inside **delimiters** (`"""`, XML tags), then the output format
spec. Put the most important instruction first or last, never buried in the middle.

## 2. Adaptation spectrum (cheapest first)

```
prompt engineering ─► few-shot / dynamic examples ─► RAG ─► prompt/p-tuning ─► LoRA/PEFT ─► full SFT ─► continued pre-training
   no weight updates ───────────────────────────────────┘      (soft prompts)   └────── weight updates ───────────────┘
```

- **No parameter updates allowed?** Use system prompts, few-shot with curated domain examples,
  CoT with a structured rationale, RAG, and output constraints.
- **Prompt tuning / p-tuning** learns *soft prompt embeddings* while the model stays frozen. It is
  PEFT, not prompt engineering, because it needs training.
- Domain adaptation through prompts: a glossary of domain terms, a persona ("You are a
  board-certified radiologist…"), domain-specific few-shot examples, a structured output
  template, and retrieval of domain documents.

## 3. Output control: the decoding knobs

Given logits `z` for the next token:

| Knob | Effect | Typical |
|---|---|---|
| `temperature` T | `softmax(z / T)`. T→0 approaches greedy (deterministic); T>1 flattens (more random). | 0–0.3 for extraction and classification, 0.7–1.0 for creative work |
| `top_k` | Keep only the k highest logits. | 40–50 |
| `top_p` (nucleus) | Keep the smallest set whose cumulative probability ≥ p. Adapts to the distribution's shape. | 0.9–0.95 |
| `repetition_penalty` | Divides positive logits (multiplies negative ones) for tokens already generated. | 1.1–1.2 |
| `frequency / presence penalty` | Subtracts `count·α` / `β` from logits of seen tokens (OpenAI-style). | 0–1 |
| `max_tokens`, `stop` | Hard length limit and stop sequences. | always set these |
| `seed` | Reproducible sampling. | tests and evaluation |
| **Beam search** | Keeps the top-B partial sequences. Good for translation and summarisation, bland for chat. | B = 4 |

**Structured output:** ask for JSON with an explicit schema, show an example, set temperature
low, *validate and retry*. Better still, use **guided/constrained decoding**: NIM and vLLM accept
`guided_json`, `guided_choice` and `guided_regex`, and TensorRT-LLM/Triton support logits
processors. These mask invalid tokens at every step, so the output is valid by construction.

## 4. Exam traps

- "No parameter updates" rules out LoRA, p-tuning, SFT and RLHF. Choose few-shot, CoT, RAG and system prompts.
- Zero-shot works *because of instruction tuning*. The model generalises from the instruction.
- CoT helps **large** models on **multi-step** problems. For simple lookups it adds latency and cost for no gain.
- Self-consistency needs **sampling** (T > 0). Majority voting over identical greedy outputs is pointless.
- Low temperature gives consistency, not correctness. It doesn't reduce hallucination about facts; RAG does.
- Few-shot examples should be **diverse, balanced across labels, and similar to the query**
  (dynamic selection by embedding similarity). Label order and recency bias are real.
- Long prompts cost latency (prefill) and money. Few-shot trades tokens for accuracy.

## Exercises (`exercises.py`)

| # | Function | Concept |
|---|---|---|
| 1 | `build_prompt` | zero / one / few-shot prompt assembly |
| 2 | `to_chatml` | chat templates and the generation prompt |
| 3 | `apply_temperature`, `top_k_filter`, `top_p_filter` | sampling controls |
| 4 | `apply_repetition_penalty` | CTRL-style penalty |
| 5 | `constrain_to_choices` | constrained decoding (guided_choice) |
| 6 | `extract_final_answer`, `self_consistency` | CoT answer parsing and majority vote |
| 7 | `select_examples` | dynamic few-shot by cosine similarity |
| 8 | `parse_json_output` | robust structured-output parsing |
