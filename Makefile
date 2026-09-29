# NCP-GENL labs: single entry point. Every section has its own Makefile under labs/<section>/;
# this one sets up the environment and delegates.
#
#   make setup            create .venv with the core (CPU) requirements
#   make test-05          test your exercises for lab 05 (any lab code 01–10)
#   make s3-gpu           run a section target: s<N>-<target>  (see `make s3` for the list)
#   make quiz / mock      practise exam questions

SECTIONS := 1-foundations-prompting 2-data-finetuning 3-optimization-acceleration \
            4-deployment-monitoring 5-evaluation-responsible-ai
PYTHON   ?= python3
VENV     := .venv
PY       := $(VENV)/bin/python
SECTION  ?= all

section_dir = $(or $(firstword $(foreach s,$(SECTIONS),$(if $(wildcard labs/$(s)/$(1)-*),labs/$(s)))),$(error No lab with code $(1); use 01–10))
selected    = $(if $(filter all,$(SECTION)),$(SECTIONS),$(filter $(SECTION)-%,$(SECTIONS)))

.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu test solutions quiz mock review stats gpu notebooks doctor brev-setup clean \
        s1 s2 s3 s4 s5

help: ## Show this help
	@echo "NCP-GENL labs: make <target>"
	@echo
	@grep -hE '^[a-zA-Z0-9_%-]+:.*## ' $(MAKEFILE_LIST) | sed 's/%/NN/' | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Sections:"
	@$(foreach s,$(SECTIONS),echo "  s$(firstword $(subst -, ,$(s)))  labs/$(s)";)
	@echo
	@echo "Section targets: make s<N>-<target>, e.g. make s1-test, make s3-gpu-06, make s4-triton-up"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip

setup: $(VENV)/bin/activate ## Create .venv and install the core requirements (CPU is enough)
	$(PY) -m pip install -r requirements.txt
	@$(PY) -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

setup-gpu: setup ## Install GPU packages: SECTION=all|1|2|3|4|5 (default all)
	@$(foreach s,$(selected),$(MAKE) --no-print-directory -C labs/$(s) setup-gpu &&) true

test: ## Test YOUR exercises in every lab (+ quiz bank checks)
	$(PY) -m pytest

solutions: ## Test all reference solutions
	$(PY) -m pytest --solutions

test-%: ## Test one lab's exercises, e.g. make test-05
	@$(MAKE) --no-print-directory -C $(call section_dir,$*) test-$*

solutions-%: ## Test one lab's reference solution
	@$(MAKE) --no-print-directory -C $(call section_dir,$*) solutions-$*

gpu-%: ## Run one lab's GPU part on the GPU instance, e.g. make gpu-04
	@$(MAKE) --no-print-directory -C $(call section_dir,$*) gpu-$*

gpu: ## Run every GPU lab of SECTION=… (default all)
	@$(foreach s,$(selected),$(MAKE) --no-print-directory -C labs/$(s) gpu &&) true

quiz: ## Practise questions (all domains, pick interactively)
	$(PY) quiz.py

quiz-%: ## Practise one domain, e.g. make quiz-06
	$(PY) quiz.py $*

mock: ## 65-question timed mock exam weighted like the blueprint
	$(PY) quiz.py --mock

review: ## Re-ask only the questions you got wrong
	$(PY) quiz.py --review

stats: ## Accuracy per domain from your quiz history
	$(PY) quiz.py --stats

notebooks: ## Generate .ipynb versions of every gpu_lab.py (for Jupyter on Brev)
	@$(foreach s,$(SECTIONS),$(MAKE) --no-print-directory -C labs/$(s) notebooks &&) true

doctor: ## Check GPU, driver, PyTorch and Docker
	@$(MAKE) --no-print-directory -C labs/$(firstword $(SECTIONS)) doctor

brev-setup: ## Full instance setup (what the Brev Launchable runs): SECTION=…
	LAB_SECTION=$(SECTION) bash brev/setup.sh

s1 s2 s3 s4 s5: ## List a section's targets, e.g. make s3
	@$(MAKE) --no-print-directory -C labs/$(filter $(subst s,,$@)-%,$(SECTIONS)) help

s1-%: ; @$(MAKE) --no-print-directory -C labs/$(word 1,$(SECTIONS)) $*
s2-%: ; @$(MAKE) --no-print-directory -C labs/$(word 2,$(SECTIONS)) $*
s3-%: ; @$(MAKE) --no-print-directory -C labs/$(word 3,$(SECTIONS)) $*
s4-%: ; @$(MAKE) --no-print-directory -C labs/$(word 4,$(SECTIONS)) $*
s5-%: ; @$(MAKE) --no-print-directory -C labs/$(word 5,$(SECTIONS)) $*

include brev/brev.mk

clean: ## Remove caches and generated notebooks/results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache labs/*/*/gpu_lab.ipynb labs/*/*/results labs/*/*/traces
