# Lab 10 — Safety, Ethics & Compliance (5% of exam)

> Blueprint: *bias auditing, fairness assessment, guardrails, responsible AI practices.*

You will compute the fairness metrics used in bias audits, generate **counterfactual** probes
for an LLM, build a small **guardrails pipeline** (input rails, output rails, PII redaction,
injection detection) around any LLM function, and check a model card for completeness. On the GPU
instance you run **NVIDIA NeMo Guardrails** (`guardrails/`) in front of a local model.

```
make test-10          # YOUR exercises (run from the repo root)
make solutions-10     # reference solutions
make quiz-10          # exam-style questions
make gpu-10           # GPU part (gpu_lab.py) on the Brev/AWS instance
make s5-llm-up && make s5-guardrails   # NeMo Guardrails in front of a local vLLM model
```

---

## 1. Bias and fairness

**Why:** the primary goal of bias detection and mitigation across the LLM lifecycle is to ensure
the model's outputs are **fair and non-discriminatory across demographic groups** and don't
reinforce harmful stereotypes, which protects users and keeps the system compliant and
trustworthy.

| Metric | Definition | Fair when |
|---|---|---|
| **Demographic parity difference** | max − min over groups of P(ŷ=1 \| group) | ≈ 0 |
| **Disparate impact ratio** | min rate / max rate | ≥ 0.8 (the **four-fifths rule**) |
| **Equal opportunity** | TPR gap between groups | ≈ 0 |
| **Equalized odds difference** | max(TPR gap, FPR gap) | ≈ 0 |
| **Counterfactual fairness (LLMs)** | change only the protected attribute in the prompt ("he" → "she", names, nationality) and compare outputs, sentiment, scores or refusal rates | outputs invariant |

Bias sources: skewed or under-representative training data, historical bias in labels,
annotation bias, evaluation sets that miss subgroups, and deployment feedback loops.
Mitigations by lifecycle stage: **data** (rebalancing, augmentation, filtering toxic or
stereotyped content), **training** (debiasing objectives, preference alignment with fairness
criteria), **inference** (guardrails, prompt instructions, output filtering), **monitoring**
(per-group metrics in production). Benchmarks: BBQ, WinoBias, StereoSet, CrowS-Pairs, BOLD,
RealToxicityPrompts.

## 2. Guardrails

**NeMo Guardrails** adds programmable **rails** around an LLM app:

| Rail type | Runs on | Examples |
|---|---|---|
| **Input rails** | user message before the LLM | jailbreak / prompt-injection detection, PII masking, content safety check (e.g. Llama Nemotron Safety Guard / NemoGuard ContentSafety NIM), topic control |
| **Dialog rails** | conversation flow (Colang) | canonical forms, e.g. "if user asks about politics → refuse politely" |
| **Retrieval rails** | retrieved chunks in RAG | drop chunks with PII or untrusted instructions |
| **Execution rails** | tool/action calls | validate tool inputs and outputs |
| **Output rails** | LLM response before the user | self-check output, fact-checking / hallucination detection against context, PII redaction, toxicity |

Configuration lives in a folder: `config.yml` (models, enabled rails, prompts) plus `*.co` Colang
flows. The **NemoGuard NIMs** (content safety, topic control, jailbreak detection) are the
production-grade classifiers. Other tools: Llama Guard, Presidio (PII), and toxicity classifiers.

**Defence in depth:** system prompt policy, input rails, least-privilege tools, output rails,
logging and human review. Jailbreaks and **indirect prompt injection** (instructions hidden in
retrieved documents or web pages) are the main attack classes. **Red-team** before launch and
continuously after; track the **attack success rate**.

## 3. Responsible AI practices and compliance

- **Transparency:** model cards (intended use, out-of-scope uses, training data, evaluation
  including per-group results, limitations, ethical considerations, licence). NVIDIA publishes
  **Model Card++** with bias, explainability, privacy and safety subcards.
- **Privacy:** data minimisation, PII redaction before training (Lab 03), consent and licensing,
  retention policies, no training on customer data without agreement. GDPR (right to erasure,
  lawful basis), HIPAA for health data.
- **Regulation:** the **EU AI Act** is risk-based: prohibited, high-risk (conformity assessment,
  risk management, logging, human oversight), limited-risk (transparency such as disclosing AI
  interaction and labelling synthetic content), plus general-purpose AI obligations. Also the
  NIST AI RMF (Govern, Map, Measure, Manage) and ISO/IEC 42001.
- **Human oversight and accountability:** escalation paths, the ability to override, audit logs.
- **Content provenance:** watermarking and C2PA metadata for generated media.
- **Trustworthy AI (NVIDIA):** privacy, safety and security, transparency, non-discrimination.

## 4. Exam traps

- Guardrails **complement** alignment and don't replace it. Evaluation must measure both.
- Removing the protected attribute from the input doesn't remove bias (proxies like zip code or names remain). Audit outcomes per group.
- Fairness metrics can conflict (demographic parity vs equalized odds). Choose based on the use case and document why.
- Output rails catch what input rails miss, such as a hallucinated phone number. Use both.
- Indirect prompt injection arrives through **retrieved content or tools**, not the user message. Retrieval and execution rails matter.

## Exercises (`exercises.py`)

Stuck, or done with an exercise? [`SOLUTION.md`](SOLUTION.md) walks through every one step by step.

| # | Function / class | Concept |
|---|---|---|
| 1 | `selection_rates`, `demographic_parity_difference`, `disparate_impact_ratio` | group fairness |
| 2 | `equalized_odds_difference` | error-rate fairness |
| 3 | `counterfactual_prompts`, `counterfactual_gap` | LLM bias probing |
| 4 | `detect_prompt_injection` | input rail heuristics |
| 5 | `GuardrailedLLM` | input rails, output rails and PII masking around any LLM |
| 6 | `attack_success_rate` | red-team metric |
| 7 | `model_card_gaps` | transparency and documentation |
| 8 | `retrieval_rail` | retrieval rail against indirect prompt injection (+ PII masking) |
| 9 | `topic_rail` | embedding-based topical rail: decline off-topic requests |
| 10 | `memorization_leaks` | canary-based privacy / memorisation testing |
| 11 | `ai_act_risk_tier` | EU AI Act risk tiers and what each implies |

Every quiz topic maps to an exercise: see the table at the end of [`SOLUTION.md`](SOLUTION.md).

## Further reading

Chosen to explain the concepts behind this lab; read the *Start here* items first.

**Start here**
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/): prompt injection, sensitive-information disclosure, excessive agency and the rest of the list.
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) (Lilian Weng): jailbreaks, token-level attacks and red-teaming.
- [Prompt injection series](https://simonwillison.net/series/prompt-injection/) (Simon Willison): why direct and indirect (retrieved) injection is so hard to fix.

**Go deeper**
- [Fairness and Machine Learning](https://fairmlbook.org/) (Barocas, Hardt, Narayanan; free book): demographic parity, equalized odds, and why you can't satisfy them all.
- [EU AI Act explorer](https://artificialintelligenceact.eu/): the risk tiers and the obligations for each.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework): Govern, Map, Measure and Manage.

**NVIDIA docs**
- [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview): input, output, retrieval, dialog and topical rails, configured with Colang.

**Papers**
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) · [Extracting Training Data from LLMs](https://arxiv.org/abs/2012.07805)
