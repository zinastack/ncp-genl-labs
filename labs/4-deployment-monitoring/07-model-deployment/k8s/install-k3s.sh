#!/usr/bin/env bash
# Single-node Kubernetes (k3s) on this GPU instance, using the host's Docker as container runtime,
# so the Triton image you built for Docker Compose is directly usable (no registry, no copy).
#
# What a real GPU cluster gets from the NVIDIA GPU Operator, and what we do here instead:
#   driver               → already installed on the VM
#   container toolkit    → already installed; we make "nvidia" Docker's default runtime
#   device plugin        → helm chart nvidia-device-plugin (with time-slicing config)
#   GPU feature discovery→ enabled through the same chart (node labels nvidia.com/gpu.product, ...)
#   DCGM exporter        → already running in Docker Compose (Lab 08)
set -euo pipefail

NAMESPACE_DP=nvidia-device-plugin
DP_VERSION=${DP_VERSION:-0.20.1}
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1/4 Docker: make nvidia the default runtime (k3s + cri-dockerd can't pass --gpus)"
if ! docker info --format '{{.DefaultRuntime}}' | grep -q nvidia; then
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker
  echo "   Docker restarted: start the Compose stack again later with make triton-up"
fi

echo "== 2/4 k3s (single node, Docker runtime, no Traefik)"
if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker --disable traefik --write-kubeconfig-mode 644" sh -
fi
mkdir -p "$HOME/.kube"
cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do sleep 3; done
kubectl get nodes

echo "== 3/4 Helm"
command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo update >/dev/null

echo "== 4/4 NVIDIA device plugin + GPU feature discovery (time-slicing replicas: ${REPLICAS:-1})"
sed "s/__REPLICAS__/${REPLICAS:-1}/" "$HERE/device-plugin-config.yaml" > /tmp/genl-dp-config.yaml
helm upgrade -i nvdp nvdp/nvidia-device-plugin --namespace "$NAMESPACE_DP" --create-namespace \
  --version "$DP_VERSION" --set gfd.enabled=true --set-file config.map.config=/tmp/genl-dp-config.yaml --wait
echo "waiting for the node to advertise nvidia.com/gpu ..."
until kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' | grep -q '[1-9]'; do sleep 3; done
kubectl describe node | grep -E "nvidia.com/gpu|gpu.product|gpu.replicas" | sort -u
