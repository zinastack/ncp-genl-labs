SECTION_NAME := Section 3 · Optimization & GPU Acceleration
LABS         := 05-model-optimization 06-gpu-acceleration
CODES        := 05 06

include ../../mk/common.mk

DIST       := 06-gpu-acceleration
DIST_SIZE  ?= base
.PHONY: ddp-check allreduce scaling ddp-vs-fsdp profile-06

# GPU: 2× L4 or 2× T4 for the distributed parts (lab 05 needs only one GPU).
gpu-06: ddp-check allreduce scaling ddp-vs-fsdp ## Multi-GPU: NCCL check, all-reduce bandwidth, scaling, DDP vs FSDP

ddp-check: ## Your average_gradients under torchrun (NCCL on GPUs)
	-nvidia-smi topo -m
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) ddp_check.py $(if $(USE_EXERCISES),,--solutions)

allreduce: ## NCCL all-reduce bandwidth vs message size
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) allreduce_bench.py

scaling: ## DDP throughput on 1 GPU vs all GPUs → scaling efficiency
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=1 dist_train.py --strategy ddp --size $(DIST_SIZE)
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) dist_train.py --strategy ddp --size $(DIST_SIZE)

ddp-vs-fsdp: ## Per-GPU memory of DDP vs FSDP on a ~0.9B model (DDP may OOM, which is the point)
	-cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) dist_train.py --strategy ddp --size large --batch 4
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) dist_train.py --strategy fsdp --size large --batch 4

profile-06: ## Nsight Systems timeline + PyTorch profiler trace of DDP training
	cd $(DIST) && nsys profile -t cuda,nvtx,osrt --force-overwrite true -o ddp_profile \
		$(TORCHRUN) --nproc_per_node=$(NGPU) dist_train.py --nvtx --steps 12
	cd $(DIST) && $(TORCHRUN) --nproc_per_node=$(NGPU) dist_train.py --steps 12 --torch-profile traces
	@echo "Open $(DIST)/ddp_profile.nsys-rep in Nsight Systems; traces/ in TensorBoard or https://ui.perfetto.dev"
