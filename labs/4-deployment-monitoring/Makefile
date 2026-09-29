SECTION_NAME := Section 4 · Deployment & Monitoring
LABS         := 07-model-deployment 08-monitoring-reliability
CODES        := 07 08

include ../../mk/common.mk

TRITON_VERSION ?= 24.12
export TRITON_VERSION
COMPOSE := docker compose
.PHONY: triton-up triton-down triton-logs perf k8s-validate

# GPU: 1× T4 or L4. Everything runs in Docker (Triton, DCGM exporter, Prometheus, Grafana).
triton-up: ## Build and start Triton + DCGM + Prometheus + Grafana
	$(COMPOSE) up -d --build
	@echo "waiting for Triton to load the model ..."; \
		until curl -sf localhost:8000/v2/health/ready; do sleep 5; done
	@echo "Triton :8000/:8001/:8002 · Prometheus :9090 · Grafana :3000"

triton-down: ## Stop the stack
	$(COMPOSE) down

triton-logs: ## Follow Triton logs
	$(COMPOSE) logs -f triton

gpu-07: perf ## Client latency sweep + perf_analyzer (needs `make triton-up`)
	$(PY) 07-model-deployment/client.py

perf: ## perf_analyzer from the Triton SDK container: latency/throughput vs concurrency
	docker run --rm --net=host -v $(CURDIR)/07-model-deployment/triton:/work \
		nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3-sdk \
		perf_analyzer -m text_classifier -i grpc -u localhost:8001 \
		--input-data /work/perf_inputs.json --concurrency-range 1:17:4 --percentile=95

gpu-08: ## Traffic phases against the stack; watch Grafana and alerts (needs `make triton-up`)
	$(PY) 08-monitoring-reliability/loadgen.py

k8s-validate: ## Validate the Kubernetes manifests (kubectl client-side dry run)
	kubectl apply --dry-run=client -f 07-model-deployment/k8s/
