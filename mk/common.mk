# Shared targets for every section Makefile.
# A section Makefile sets SECTION_NAME, LABS (lab folders) and CODES (domain codes), then includes this.

ROOT        := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
VENV        ?= $(ROOT)/.venv
PY          := $(VENV)/bin/python
PYTEST      := $(PY) -m pytest -c $(ROOT)/pytest.ini --rootdir=$(ROOT)
MASTER_PORT ?= 29512
# Explicit IPv4 rendezvous (instead of --standalone) also works on laptops without reverse DNS.
TORCHRUN    := $(PY) -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port=$(MASTER_PORT)
# Number of NVIDIA GPUs; falls back to 2 CPU processes (gloo) on a laptop.
NGPU        ?= $(shell n=$$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); [ "$$n" -gt 0 ] && echo $$n || echo 2)

# Local OpenAI-compatible LLM server (vLLM) used by the prompting / evaluation / guardrails labs.
LLM_MODEL   ?= Qwen/Qwen2.5-1.5B-Instruct
LLM_PORT    ?= 8008
VLLM_IMAGE  ?= vllm/vllm-openai:v0.8.5

lab_dir = $(or $(wildcard $(1)-*),$(error No lab $(1) in $(SECTION_NAME). Labs here: $(CODES)))

.DEFAULT_GOAL := help
.PHONY: help test solutions quiz setup-gpu gpu notebooks doctor llm-up llm-down

help: ## Show this section's targets
	@echo "$(SECTION_NAME) (labs $(CODES))"
	@echo
	@grep -hE '^[a-zA-Z0-9_%-]+:.*## ' $(MAKEFILE_LIST) | sed 's/%/NN/' | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

test: ## Test YOUR exercises for every lab in this section
	$(PYTEST) $(LABS)

solutions: ## Test the reference solutions
	$(PYTEST) $(LABS) --solutions

test-%: ## Test one lab, e.g. make test-01
	$(PYTEST) $(call lab_dir,$*)

solutions-%: ## Test one lab's reference solution
	$(PYTEST) $(call lab_dir,$*) --solutions

quiz: ## Exam-style questions for this section
	$(PY) $(ROOT)/quiz.py $(CODES)

quiz-%: ## Questions for one domain, e.g. make quiz-02
	$(PY) $(ROOT)/quiz.py $*

setup-gpu: ## Install this section's GPU Python packages
	$(PY) -m pip install -r requirements-gpu.txt

gpu: $(addprefix gpu-,$(CODES)) ## Run every GPU lab in this section

gpu-%: ## Run one lab's GPU part (gpu_lab.py)
	$(PY) $(call lab_dir,$*)/gpu_lab.py

notebooks: ## Convert gpu_lab.py files to Jupyter notebooks
	$(if $(wildcard */gpu_lab.py),$(PY) -m jupytext --to ipynb $(wildcard */gpu_lab.py),@echo "$(SECTION_NAME): no gpu_lab.py (GPU work runs through make targets)")

doctor: ## Check GPU, driver, PyTorch and Docker
	@nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no NVIDIA GPU visible"
	@$(PY) -c "import sys; sys.path.insert(0, '$(ROOT)'); from common import device; print(device.summary())"
	@docker info --format 'docker {{.ServerVersion}} runtimes: {{range $$k, $$v := .Runtimes}}{{$$k}} {{end}}' 2>/dev/null || echo "docker not available"

llm-up: ## Start a vLLM OpenAI-compatible server on the GPU (port 8008)
	docker run -d --rm --name genl-llm --gpus all --ipc=host -p $(LLM_PORT):8000 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface $(VLLM_IMAGE) \
		--model $(LLM_MODEL) --dtype half --max-model-len 4096 --gpu-memory-utilization 0.45
	@echo "waiting for $(LLM_MODEL) ..."; \
		until curl -sf localhost:$(LLM_PORT)/v1/models >/dev/null; do sleep 5; done; echo "ready on :$(LLM_PORT)"

llm-down: ## Stop the vLLM server
	-docker stop genl-llm
