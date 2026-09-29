SECTION_NAME := Section 5 · Evaluation & Responsible AI
LABS         := 09-evaluation 10-safety-ethics
CODES        := 09 10

include ../../mk/common.mk

EVAL_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
EVAL_TASKS ?= arc_easy,hellaswag
EVAL_LIMIT ?= 200
.PHONY: lm-eval guardrails

# GPU: 1× L4 or T4.
lm-eval: ## Benchmark a model with lm-evaluation-harness (subset: EVAL_LIMIT examples per task)
	$(PY) -m lm_eval --model hf --model_args pretrained=$(EVAL_MODEL),dtype=float16 \
		--tasks $(EVAL_TASKS) --limit $(EVAL_LIMIT) --batch_size 16 --device cuda:0 \
		--output_path 09-evaluation/results

guardrails: ## NeMo Guardrails in front of the local model (run `make llm-up` first)
	$(PY) 10-safety-ethics/run_guardrails.py
