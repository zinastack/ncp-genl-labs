SECTION_NAME := Section 4 · Deployment & Monitoring
LABS         := 07-model-deployment 08-monitoring-reliability
CODES        := 07 08

include ../../mk/common.mk

# GPU: 1× L4 (TensorRT-LLM FP8/INT8 and most NIMs need Ada or newer). Disk: ~150 GB of images.
# Pinned NGC releases (CUDA 12.9, drivers ≥ 535 via forward compatibility). With a driver ≥ 580
# you can move to newer tags, e.g. make triton-up TRITON_VERSION=26.09.
TRITON_VERSION ?= 25.08
TRTLLM_IMAGE   ?= nvcr.io/nvidia/tensorrt-llm/release:1.0.0
NIM_IMAGE      ?= nvcr.io/nim/meta/llama-3.2-1b-instruct:latest
export TRITON_VERSION

L7      := 07-model-deployment
L8      := 08-monitoring-reliability
REPO7   := $(CURDIR)/$(L7)/triton/model_repository
IMAGE   := genl-labs/triton-textclf:$(TRITON_VERSION)
SDK     := nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3-sdk
TRT     := nvcr.io/nvidia/tensorrt:$(TRITON_VERSION)-py3
USER_ID := $(shell id -u):$(shell id -g)
CTL     := $(PY) $(L7)/tools/tritonctl.py
COMPOSE := docker compose
M       ?= text_classifier

.PHONY: triton-up triton-down triton-logs models load unload reload config stats infer perf perf-report \
        export-onnx trt-build version2 first-request ma-07 trtllm-up trtllm-logs trtllm-bench trtllm-report \
        trtllm-down ngc-login nim-profiles nim-up nim-logs nim-chat nim-bench nim-down check-07 \
        k8s-up k8s-deploy k8s-status k8s-timeslice k8s-drill-probe k8s-monitoring k8s-hpa k8s-load \
        k8s-rollout k8s-canary k8s-canary-check k8s-rollback k8s-down k8s-validate \
        load-08 metrics-08 drift-08 gpu-hog alerts check-08

# ── Lab 07 · Triton in Docker Compose ─────────────────────────────────────────
triton-up: ## Build and start Triton + DCGM + Prometheus + Grafana (explicit model control)
	$(COMPOSE) up -d --build
	@echo "waiting for Triton to load text_classifier ..."; \
		until curl -sf localhost:8000/v2/health/ready; do sleep 5; done
	@echo "Triton :8000 HTTP /:8001 gRPC /:8002 metrics · Prometheus :9090 · Grafana :3000"

triton-down: ## Stop the stack
	$(COMPOSE) down

triton-logs: ## Follow Triton logs (model loads, errors)
	$(COMPOSE) logs -f triton

models: ## List every model in the repository and its state
	@$(CTL) index

load: ## Load a model: make load M=distilbert_onnx
	@$(CTL) load $(M)

unload: ## Unload a model: make unload M=distilbert_onnx
	@$(CTL) unload $(M)

reload: ## Apply an edited config.pbtxt or a rebuilt engine (unload + load, no server restart): make reload M=...
	@-$(CTL) unload $(M)
	@$(CTL) load $(M) && $(PY) $(L7)/tools/perf.py --summary $(M)

config: ## The config Triton is actually using: make config M=...
	@$(CTL) config $(M)

stats: ## Cumulative stats: requests, avg batch, queue vs compute time: make stats M=...
	@$(CTL) stats $(M)

infer: ## One request: make infer M=sentiment_ensemble TEXT="..."
	@$(CTL) infer $(M) "$(or $(TEXT),The new GPU cluster cut our training time in half.)"

PCONC ?= 1:16:5
perf: ## perf_analyzer sweep, recorded with the live config: make perf M=... LABEL="what you changed"
	$(PY) $(L7)/tools/perf.py $(M) --label "$(LABEL)" --concurrency $(PCONC) --sdk-image $(SDK)

perf-report: ## Every recorded perf run side by side
	@$(PY) $(L7)/tools/perf.py --report

gpu-07: ## Client latency sweep over HTTP (what a real caller sees): make gpu-07 M=sentiment_ensemble [URL=...]
	$(PY) $(L7)/client.py --model $(M) $(if $(URL),--url $(URL))

export-onnx: ## Task 4: export DistilBERT to model_repository/distilbert_onnx/1/model.onnx
	docker run --rm --user $(USER_ID) -e HOME=/tmp -v $(REPO7):/repo -v $(CURDIR)/$(L7)/tools:/tools \
		$(IMAGE) python /tools/export_onnx.py /repo/distilbert_onnx/1/model.onnx

PREC   ?= fp16
MIN_BS ?= 1
OPT_BS ?= 8
MAX_BS ?= 16
SEQ    ?= 128
trt-build: ## Task 4: ONNX → TensorRT engine with trtexec: make trt-build PREC=fp32|fp16|int8 MAX_BS=16
	docker run --rm --gpus all --user $(USER_ID) -v $(REPO7):/repo $(TRT) /usr/src/tensorrt/bin/trtexec \
		--onnx=/repo/distilbert_onnx/1/model.onnx --saveEngine=/repo/distilbert_trt/1/model.plan \
		--minShapes=input_ids:$(MIN_BS)x$(SEQ),attention_mask:$(MIN_BS)x$(SEQ) \
		--optShapes=input_ids:$(OPT_BS)x$(SEQ),attention_mask:$(OPT_BS)x$(SEQ) \
		--maxShapes=input_ids:$(MAX_BS)x$(SEQ),attention_mask:$(MAX_BS)x$(SEQ) \
		$(if $(filter fp16,$(PREC)),--fp16) $(if $(filter int8,$(PREC)),--int8 --fp16) --skipInference
	@echo "engine built ($(PREC), batch $(MIN_BS)/$(OPT_BS)/$(MAX_BS)). Next: make load M=distilbert_trt"

version2: ## Task 6: create version 2 of distilbert_onnx (a copy of version 1)
	mkdir -p $(REPO7)/distilbert_onnx/2 && cp $(REPO7)/distilbert_onnx/1/model.onnx $(REPO7)/distilbert_onnx/2/
	@echo "now edit version_policy in distilbert_onnx/config.pbtxt, then make reload M=distilbert_onnx"

first-request: ## Task 6: cold load + first request vs warm: make first-request M=distilbert_trt
	@$(CTL) first-request $(M)

ma-07: ## Task 7: Model Analyzer config search for distilbert_onnx (stops Compose Triton, ~20-40 min)
	$(COMPOSE) stop triton
	mkdir -p $(L7)/results/model_analyzer
	docker run --rm --gpus all --net=host -v /var/run/docker.sock:/var/run/docker.sock \
		-v $(CURDIR)/$(L7):$(CURDIR)/$(L7) -w $(CURDIR)/$(L7) $(SDK) model-analyzer profile \
		-f $(CURDIR)/$(L7)/model_analyzer.yaml --model-repository $(REPO7) \
		--triton-docker-image nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3 \
		--output-model-repository-path $(CURDIR)/$(L7)/results/model_analyzer/configs \
		--export-path $(CURDIR)/$(L7)/results/model_analyzer
	$(COMPOSE) start triton
	@echo "summary: $(L7)/results/model_analyzer/reports/summaries/distilbert_onnx/"

# ── Lab 07 · TensorRT-LLM (OpenAI-compatible trtllm-serve on :8010) ───────────
BACKEND ?= pytorch
MBS     ?= 64
MNT     ?= 8192
MSL     ?= 2048
KV      ?= 0.5
EXTRA   ?=
trtllm-up: ## Task 8: serve LLM_MODEL with TensorRT-LLM: BACKEND=pytorch|tensorrt MBS= MNT= KV= EXTRA=file.yml
	-docker rm -f genl-trtllm >/dev/null 2>&1
	docker run -d --name genl-trtllm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
		-p 8010:8000 -e HF_TOKEN -v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		-v $(CURDIR)/$(L7):/work $(TRTLLM_IMAGE) trtllm-serve $(LLM_MODEL) --host 0.0.0.0 --port 8000 \
		--backend $(BACKEND) --max_batch_size $(MBS) --max_num_tokens $(MNT) --max_seq_len $(MSL) \
		--kv_cache_free_gpu_memory_fraction $(KV) $(if $(EXTRA),--extra_llm_api_options /work/trtllm/$(EXTRA))
	@echo "loading $(LLM_MODEL) ($(BACKEND) backend$(if $(filter tensorrt,$(BACKEND)), — building an engine first: several minutes)) ..."; \
		until curl -sf localhost:8010/health >/dev/null; do \
			docker ps -q -f name=genl-trtllm | grep -q . || { docker logs --tail 30 genl-trtllm; exit 1; }; sleep 5; done
	@echo "ready on :8010 (OpenAI API). Watch memory: nvidia-smi"

trtllm-logs: ## Follow TensorRT-LLM server logs (engine build, KV-cache size)
	docker logs -f genl-trtllm

CONC ?= 8
ISL  ?= 512
OSL  ?= 128
trtllm-bench: ## Task 8: TTFT/TPOT/throughput at concurrency CONC: make trtllm-bench CONC=1|8|32
	mkdir -p $(L7)/results/trtllm
	docker exec genl-trtllm python -m tensorrt_llm.serve.scripts.benchmark_serving --model $(LLM_MODEL) \
		--backend openai --dataset-name random --random-ids --random-input-len $(ISL) --random-output-len $(OSL) \
		--num-prompts $$(( $(CONC) * 8 )) --max-concurrency $(CONC) --ignore-eos --save-result \
		--result-dir /work/results/trtllm --result-filename run-$$(date +%s).json \
		--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
		--metadata engine=$(BACKEND) mbs=$(MBS) mnt=$(MNT) kv=$(KV) extra=$(or $(EXTRA),none)
	@$(PY) $(L7)/trtllm/summarize.py

trtllm-report: ## Every TensorRT-LLM benchmark run side by side
	@$(PY) $(L7)/trtllm/summarize.py

trtllm-down: ## Stop the TensorRT-LLM server
	-docker rm -f genl-trtllm

# ── Lab 07 · NIM (needs a free NGC API key: export NGC_API_KEY=...) ───────────
ngc-login: ## Log Docker in to nvcr.io with your NGC_API_KEY
	@test -n "$$NGC_API_KEY" || { echo "export NGC_API_KEY=... first (ngc.nvidia.com → Setup → API key)"; exit 1; }
	@echo "$$NGC_API_KEY" | docker login nvcr.io -u '$$oauthtoken' --password-stdin

nim-profiles: ngc-login ## Task 9: which engines/profiles this NIM offers for this GPU
	docker run --rm --gpus all -e NGC_API_KEY $(NIM_IMAGE) list-model-profiles

nim-up: ngc-login ## Task 9: start the NIM on :8020 (PROFILE=<id> to pin a profile)
	-docker rm -f genl-nim >/dev/null 2>&1
	mkdir -p $(HOME)/.cache/nim
	docker run -d --name genl-nim --gpus all --shm-size=16g -u $(USER_ID) -e NGC_API_KEY \
		$(if $(PROFILE),-e NIM_MODEL_PROFILE=$(PROFILE)) -e NIM_MAX_MODEL_LEN=4096 \
		-v $(HOME)/.cache/nim:/opt/nim/.cache -p 8020:8000 $(NIM_IMAGE)
	@echo "downloading + starting the NIM (first time: several minutes) ..."; \
		until curl -sf localhost:8020/v1/health/ready >/dev/null; do \
			docker ps -q -f name=genl-nim | grep -q . || { docker logs --tail 30 genl-nim; exit 1; }; sleep 10; done
	curl -s localhost:8020/v1/models; echo

nim-logs: ## Follow NIM logs (selected profile, backend)
	docker logs -f genl-nim

nim-chat: ## One OpenAI-style chat request to the NIM
	@model=$$(curl -s localhost:8020/v1/models | $(PY) -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"); \
	curl -s localhost:8020/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\": \"$$model\", \
		\"messages\": [{\"role\": \"user\", \"content\": \"In one sentence, what does Triton dynamic batching do?\"}], \
		\"max_tokens\": 80}" | $(PY) -m json.tool

nim-bench: ## Task 9: GenAI-Perf against the NIM (TTFT, ITL, tokens/s): make nim-bench CONC=8
	@model=$$(curl -s localhost:8020/v1/models | $(PY) -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"); \
	mkdir -p $(L7)/results/nim && docker run --rm --net=host --user $(USER_ID) -e HOME=/tmp -e HF_TOKEN \
		-v $(CURDIR)/$(L7)/results/nim:/out $(SDK) genai-perf profile -m $$model --endpoint-type chat --streaming \
		-u localhost:8020 --concurrency $(CONC) --synthetic-input-tokens-mean $(ISL) --output-tokens-mean $(OSL) \
		--tokenizer hf-internal-testing/llama-tokenizer --artifact-dir /out/c$(CONC)

nim-down: ## Stop the NIM
	-docker rm -f genl-nim

check-07: ## Check which Lab 07 tasks you have completed (live state + recorded results)
	@$(PY) $(L7)/check.py

# ── Lab 07 · Kubernetes on this instance (k3s) ────────────────────────────────
K8S := $(L7)/k8s
N   ?= 4
k8s-up: ## K1: single-node k3s + NVIDIA device plugin + GPU feature discovery
	REPLICAS=1 bash $(K8S)/install-k3s.sh

k8s-deploy: ## K2: deploy Triton (Deployment + NodePort Service :30800) and wait until Ready
	docker image inspect $(IMAGE) >/dev/null 2>&1 || $(COMPOSE) build triton
	sed 's/__TAG__/$(TRITON_VERSION)/' $(K8S)/deployment.yaml | kubectl apply -f -
	kubectl rollout status deploy/triton --timeout=15m
	curl -s localhost:30800/v2/health/ready -o /dev/null -w "Service ready: HTTP %{http_code}\n"

k8s-status: ## Pods, GPU allocation, HPA and recent events
	kubectl get deploy,pods,hpa -o wide
	@kubectl describe node | grep -A8 "Allocated resources" | grep -E "Resource|nvidia.com/gpu"
	@kubectl get events --sort-by=.lastTimestamp | tail -8

k8s-timeslice: ## K3: share the GPU between N pods with time-slicing: make k8s-timeslice N=4
	sed "s/__REPLICAS__/$(N)/" $(K8S)/device-plugin-config.yaml > /tmp/genl-dp-config.yaml
	helm upgrade nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin --reuse-values \
		--set-file config.map.config=/tmp/genl-dp-config.yaml --wait
	@echo "waiting for allocatable nvidia.com/gpu = $(N) ..."; \
		until [ "$$(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')" = "$(N)" ]; do sleep 3; done
	kubectl describe node | grep -E "nvidia.com/gpu" | sort -u

k8s-drill-probe: ## K2 drill: 120 s model load WITHOUT a startupProbe → watch liveness kill it
	kubectl patch deploy triton --type=json -p='[{"op":"remove","path":"/spec/template/spec/containers/0/startupProbe"},{"op":"replace","path":"/spec/template/spec/containers/0/env/0/value","value":"120"}]'
	@echo "watch: kubectl get pods -w   (RESTARTS climbs, then CrashLoopBackOff). Fix: make k8s-deploy"

k8s-monitoring: ## K4: in-cluster Prometheus (:30900) + prometheus-adapter (custom metrics API)
	helm upgrade -i prometheus prometheus-community/prometheus -n monitoring --create-namespace \
		-f $(K8S)/prometheus-values.yaml --wait
	helm upgrade -i prometheus-adapter prometheus-community/prometheus-adapter -n monitoring \
		-f $(K8S)/adapter-values.yaml --wait
	@echo "custom metrics appear after ~1 min: kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | python3 -m json.tool"

k8s-hpa: ## K4: autoscale Triton on queue time (needs k8s-monitoring and k8s-timeslice N=4)
	kubectl apply -f $(K8S)/hpa.yaml
	kubectl get hpa triton

k8s-load: ## K4/K6: load through the Service: make k8s-load C=48 S=180 (concurrency, seconds)
	$(PY) $(L8)/loadgen.py --url http://localhost:30800 --metrics-url http://localhost:30802/metrics --no-alerts \
		--phases "k8s:$(or $(C),48):$(or $(S),180)"

k8s-rollout: ## K5: rolling update (new pod-template annotation) and watch it: make k8s-rollout
	kubectl patch deploy triton -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"genl/rollout\":\"$$(date +%s)\"}}}}}"
	-kubectl rollout status deploy/triton --timeout=90s
	kubectl get pods -l app=triton -o wide

k8s-canary: ## K6: add a slower v2 Deployment behind the same Service (scale stable to 3 first)
	kubectl delete hpa triton --ignore-not-found
	kubectl scale deploy triton --replicas=3
	sed 's/__TAG__/$(TRITON_VERSION)/' $(K8S)/canary-v2.yaml | kubectl apply -f -
	kubectl rollout status deploy/triton-v2 --timeout=15m
	kubectl get pods -l app=triton -L version

k8s-canary-check: ## K6: compare v1 vs v2 from Prometheus and decide promote/rollback (your canary_verdict)
	$(PY) $(K8S)/canary_check.py

k8s-rollback: ## K6: roll the canary back (remove v2)
	kubectl delete deploy triton-v2 --ignore-not-found
	kubectl get pods -l app=triton -L version

k8s-down: ## Remove the Triton objects (keeps k3s). Uninstall k3s entirely: /usr/local/bin/k3s-uninstall.sh
	-kubectl delete hpa triton --ignore-not-found
	-kubectl delete deploy triton triton-v2 --ignore-not-found
	-kubectl delete svc triton --ignore-not-found

k8s-validate: ## Validate the Kubernetes manifests (client-side, no cluster needed)
	@for f in deployment.yaml canary-v2.yaml hpa.yaml; do \
		sed 's/__TAG__/$(TRITON_VERSION)/' $(K8S)/$$f | kubectl create --dry-run=client --validate=false -o name -f - || exit 1; done

# ── Lab 08 · Monitoring the stack ──────────────────────────────────────────────
gpu-08: ## Traffic phases (steady → burst → bad requests); watch Grafana :3000 and alerts
	$(PY) $(L8)/loadgen.py

load-08: ## Custom traffic phases: make load-08 PHASES="overload:96:180" (name:concurrency:seconds[:bad])
	$(PY) $(L8)/loadgen.py --phases "$(or $(PHASES),overload:96:180)"

metrics-08: ## Task 1: raw Triton metrics → averages computed by YOUR parse_prometheus/triton_avg_*
	$(PY) $(L8)/loadgen.py --phases "warmup:4:20" --metrics-only

drift-08: ## Task 6: reference vs shifted traffic → PSI/KS on input length and model score
	$(PY) $(L8)/drift.py

GB ?= 21
gpu-hog: ## Task 5 drill: hold GB of GPU memory for 5 minutes (GPUMemoryNearlyFull fires)
	docker run --rm --gpus all $(IMAGE) python -c "import torch, time; \
		x = torch.empty(int($(GB) * 2**30), dtype=torch.uint8, device='cuda'); \
		print('holding $(GB) GB, Ctrl+C to release'); time.sleep(300)"

alerts: ## Alerts pending/firing now, and every alert that fired in the last 3 hours
	@$(PY) $(L8)/check.py --alerts

check-08: ## Check which Lab 08 tasks you have completed
	@$(PY) $(L8)/check.py
