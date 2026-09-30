#!/bin/bash
# Paste this into the "Setup script" field of each NCP-GENL Labs Launchable.
# It only bootstraps: the real setup lives in the repo (brev/setup.sh), so fixes there apply to
# every Launchable without editing them. LAB_SECTION (1–5 or all) comes from the launch parameter.
set -euo pipefail

REPO_URL="https://github.com/zinastack/ncp-genl-labs.git"
REPO_DIR="/home/ubuntu/ncp-genl-labs"   # where Brev clones the Launchable's repository

if [ ! -d "$REPO_DIR/.git" ]; then
  if id ubuntu >/dev/null 2>&1; then
    sudo -u ubuntu git clone "$REPO_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

LAB_SECTION="${LAB_SECTION:-all}" bash "$REPO_DIR/brev/setup.sh"
