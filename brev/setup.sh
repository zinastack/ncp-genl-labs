#!/usr/bin/env bash
# Instance setup for the NCP-GENL labs.
#
# Works as a Brev Launchable setup script (VM mode) and on any Ubuntu GPU VM with the NVIDIA driver,
# Docker and the NVIDIA Container Toolkit (Brev VMs, AWS "Deep Learning Base OSS Nvidia Driver GPU AMI").
#
#   LAB_SECTION=all|1|2|3|4|5   which section's GPU dependencies to install (Brev launch parameter)
#   REPO_URL=<git url>          only used when the script runs outside a clone of the repo
#
#   bash brev/setup.sh                    # from a clone
#   LAB_SECTION=3 bash brev/setup.sh
set -euo pipefail

LAB_SECTION="${LAB_SECTION:-all}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_GITHUB_USER/genl-labs.git}"
SUDO=$([ "$(id -u)" -eq 0 ] && echo "" || echo "sudo")

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# ── locate (or clone) the repository ────────────────────────────────────────────
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$here/../Makefile" ] && [ -d "$here/../labs" ]; then
  REPO="$(cd "$here/.." && pwd)"
elif [ -d /home/ubuntu/genl-labs/labs ]; then
  REPO=/home/ubuntu/genl-labs                      # where Brev clones the Launchable's repo
else
  REPO="${HOME}/genl-labs"
  git clone "$REPO_URL" "$REPO"
fi
cd "$REPO"
OWNER="$(stat -c %U "$REPO" 2>/dev/null || id -un)"
as_owner() { if [ "$(id -un)" = "$OWNER" ]; then "$@"; else $SUDO -u "$OWNER" -H "$@"; fi; }
log "Repository: $REPO (owner $OWNER) · section: $LAB_SECTION"

# ── system packages ─────────────────────────────────────────────────────────────
log "System packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y -qq
$SUDO apt-get install -y -qq python3-venv python3-pip make git curl >/dev/null

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || warn "no NVIDIA GPU visible"

# ── Python environment ─────────────────────────────────────────────────────────
log "Python venv + core requirements"
as_owner make setup PYTHON=python3

log "GPU Python packages for section: $LAB_SECTION"
as_owner make setup-gpu SECTION="$LAB_SECTION"

# ── section-specific tools ─────────────────────────────────────────────────────
if [[ "$LAB_SECTION" == all || "$LAB_SECTION" == 3 ]]; then
  log "Nsight Systems CLI (profiling, Lab 06)"
  if ! command -v nsys >/dev/null; then
    distro="ubuntu$(. /etc/os-release; echo "${VERSION_ID//./}")"
    arch="$(dpkg --print-architecture)"
    curl -fsSL "https://developer.download.nvidia.com/devtools/repos/${distro}/${arch}/nvidia.pub" \
      | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/${distro}/${arch}/ /" \
      | $SUDO tee /etc/apt/sources.list.d/nvidia-devtools.list >/dev/null
    { $SUDO apt-get update -y -qq && $SUDO apt-get install -y -qq nsight-systems-cli >/dev/null; } \
      || warn "could not install nsight-systems-cli; make profile-06 will still write PyTorch profiler traces"
  fi
fi

if [[ "$LAB_SECTION" == all || "$LAB_SECTION" == 4 ]]; then
  log "Building the Triton image and pulling monitoring images (several GB, done once)"
  (cd labs/4-deployment-monitoring && docker compose build triton && docker compose pull dcgm-exporter prometheus grafana) \
    || warn "docker compose failed; check that Docker and the NVIDIA Container Toolkit are installed"
fi

if [[ "$LAB_SECTION" == all || "$LAB_SECTION" == 1 || "$LAB_SECTION" == 5 ]]; then
  log "Pre-pulling the vLLM image used by 'make llm-up'"
  docker pull -q "$(grep -E '^VLLM_IMAGE' mk/common.mk | awk '{print $3}')" || warn "docker pull failed"
fi

# ── notebooks + final check ────────────────────────────────────────────────────
log "Jupyter notebooks from gpu_lab.py files"
as_owner make notebooks || warn "notebook generation failed"

log "Environment check"
as_owner make doctor || true

log "Done. Next: cd $REPO && make help"
