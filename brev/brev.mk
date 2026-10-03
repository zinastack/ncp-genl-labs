# Brev CLI targets: private GPU instances in your own Brev org, created from your laptop.
# The local repo is copied to the instance (no GitHub needed), then brev/setup.sh runs there.
#
#   make brev-plan S=3        preview the instance types that would be tried (free)
#   make brev-up S=3          create the instance, copy the repo, run setup   (starts billing)
#   make brev-shell S=3       open a shell on it
#   make brev-stop S=3        stop it (disk kept, compute billing stops)
#   make brev-delete S=3      delete it
#
# S = section 1..5. Override the GPU with TYPE=<brev type(s)>, the name with INSTANCE=<name>.

BREV      ?= brev
BREV_RUN  := $(BREV) --no-check-latest
S         ?=
INSTANCE  ?= genl-s$(S)

# Cheapest-first fallback chains (from `brev search gpu`). x86 only: T4g (ARM) won't run the images.
BREV_TYPES_1 := g2-standard-4:nvidia-l4:1,g6.xlarge,g2-standard-8:nvidia-l4:1
BREV_TYPES_2 := $(BREV_TYPES_1)
BREV_TYPES_3 := g2-standard-24:nvidia-l4:2,scaleway_L4x2,n1-highmem-2:nvidia-tesla-t4:2
BREV_TYPES_4 := g2-standard-8:nvidia-l4:1,g6.2xlarge,g2-standard-4:nvidia-l4:1   # L4: TensorRT-LLM INT8/FP8, NIM; 8 vCPUs for k3s
BREV_TYPES_5 := $(BREV_TYPES_1)
TYPE      ?= $(BREV_TYPES_$(S))

BREV_TARBALL := $(CURDIR)/.brev-genl-labs.tgz
need_section = $(if $(filter 1 2 3 4 5,$(S)),,$(error Set S=1..5, e.g. make $@ S=3))

.PHONY: brev-ls brev-plan brev-up brev-sync brev-setup-remote brev-hf-token brev-shell brev-open \
        brev-exec brev-forward brev-get brev-stop brev-start brev-delete brev-launch

brev-ls: ## List your Brev instances
	$(BREV) ls

brev-plan: ## Preview the GPU types brev-up would try for section S (no cost)
	$(call need_section)
	$(BREV_RUN) create $(INSTANCE) --type $(TYPE) --dry-run < /dev/null

brev-up: ## Create a private GPU instance for section S, copy the repo, run setup (billing starts)
	$(call need_section)
	$(BREV_RUN) create $(INSTANCE) --type $(TYPE) < /dev/null
	@$(MAKE) --no-print-directory brev-sync S=$(S) INSTANCE=$(INSTANCE)
	@$(MAKE) --no-print-directory brev-hf-token S=$(S) INSTANCE=$(INSTANCE)
	@$(MAKE) --no-print-directory brev-setup-remote S=$(S) INSTANCE=$(INSTANCE)
	@echo "Ready: make brev-shell S=$(S)   (and make brev-stop S=$(S) when you're done)"

brev-sync: ## Copy the current repo (tracked + untracked, not ignored) to the instance
	$(call need_section)
	git ls-files -co --exclude-standard -z | tar -czf $(BREV_TARBALL) --null -T -
	$(BREV_RUN) copy --host $(BREV_TARBALL) $(INSTANCE):ncp-genl-labs.tgz < /dev/null
	$(BREV_RUN) exec $(INSTANCE) --host "mkdir -p ~/ncp-genl-labs && tar -xzf ~/ncp-genl-labs.tgz -C ~/ncp-genl-labs && rm ~/ncp-genl-labs.tgz" < /dev/null
	@rm -f $(BREV_TARBALL)

brev-hf-token: ## Copy your Hugging Face token (HF_TOKEN or HUGGING_FACE) to the instance, if set
	$(call need_section)
	@TOKEN="$${HF_TOKEN:-$$HUGGING_FACE}"; \
	if [ -z "$$TOKEN" ]; then echo "No HF_TOKEN/HUGGING_FACE set: skipping (all lab models are public)."; exit 0; fi; \
	TMP=$$(mktemp) && chmod 600 $$TMP && printf '%s' "$$TOKEN" > $$TMP && \
	$(BREV_RUN) copy --host $$TMP $(INSTANCE):.hf_token < /dev/null && rm -f $$TMP && \
	$(BREV_RUN) exec $(INSTANCE) --host "mkdir -p ~/.cache/huggingface && mv ~/.hf_token ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token" < /dev/null

brev-setup-remote: ## Run brev/setup.sh for section S on the instance
	$(call need_section)
	$(BREV_RUN) exec $(INSTANCE) --host "cd ~/ncp-genl-labs && LAB_SECTION=$(S) bash brev/setup.sh" < /dev/null

brev-shell: ## Open a shell on the section-S instance
	$(call need_section)
	$(BREV) shell $(INSTANCE)

brev-open: ## Open the section-S instance in VS Code
	$(call need_section)
	$(BREV) open $(INSTANCE)

brev-exec: ## Run a make target remotely, e.g. make brev-exec S=5 CMD="gpu-09"
	$(call need_section)
	$(BREV_RUN) exec $(INSTANCE) --host "cd ~/ncp-genl-labs && make $(CMD)" < /dev/null

brev-forward: ## Forward a port to localhost, e.g. make brev-forward S=4 PORT=3000 (Grafana) or PORT=9090
	$(call need_section)
	$(if $(PORT),,$(error Set PORT=..., e.g. 3000 for Grafana, 9090 for Prometheus, 8888 for Jupyter))
	$(BREV) port-forward $(INSTANCE) -p $(PORT):$(PORT)

brev-get: ## Copy a file back, e.g. make brev-get S=3 FILE=labs/3-optimization-acceleration/06-gpu-acceleration/ddp_profile.nsys-rep
	$(call need_section)
	$(BREV_RUN) copy --host $(INSTANCE):ncp-genl-labs/$(FILE) . < /dev/null

brev-stop: ## Stop the section-S instance (compute billing stops; disk kept)
	$(call need_section)
	$(BREV_RUN) stop $(INSTANCE) < /dev/null

brev-start: ## Start a stopped section-S instance
	$(call need_section)
	$(BREV_RUN) start $(INSTANCE) < /dev/null

brev-delete: ## Delete the section-S instance
	$(call need_section)
	$(BREV_RUN) delete $(INSTANCE) < /dev/null

brev-launch: ## Deploy a Launchable you created in the console: make brev-launch S=3 LAUNCHABLE=env-...
	$(call need_section)
	$(if $(LAUNCHABLE),,$(error Set LAUNCHABLE=env-... (the Launchable ID from the console)))
	$(BREV_RUN) create $(INSTANCE) --launchable $(LAUNCHABLE) --param LAB_SECTION=$(S) < /dev/null
