#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/gcp-lib.sh"
load_gcp_env
require_cmd gcloud
require_gcp_project

MODEL_NAME="${MODEL_NAME:-decoder-foundation-model}"
LOCAL_MODEL_DIR="$ROOT/models/$MODEL_NAME"
REMOTE_MODEL_DIR="/mnt/tfm/models/$MODEL_NAME"
WEIGHTS_FILE="$LOCAL_MODEL_DIR/model-00001-of-00001.safetensors"

if [[ ! -d "$LOCAL_MODEL_DIR" ]]; then
  echo "Local model directory not found: $LOCAL_MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS_FILE" ]]; then
  echo "Missing model weights: $WEIGHTS_FILE" >&2
  exit 1
fi

if head -n 1 "$WEIGHTS_FILE" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  echo "Model weights are still a Git LFS pointer: $WEIGHTS_FILE" >&2
  echo "Run locally: git lfs install && git lfs pull --include='models/$MODEL_NAME/**' --exclude=''" >&2
  exit 1
fi

wait_for_gcp_ssh 12 5

echo "Preparing remote model directory at $REMOTE_MODEL_DIR..."
gcloud_compute_ssh --command "sudo mkdir -p '$REMOTE_MODEL_DIR' && sudo chown -R \$(id -u):\$(id -g) '$REMOTE_MODEL_DIR'"

echo "Syncing $LOCAL_MODEL_DIR to $GCP_INSTANCE:$REMOTE_MODEL_DIR ..."
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  -czf - -C "$LOCAL_MODEL_DIR" . \
  | gcloud_compute_ssh --command "tar --no-same-owner --no-same-permissions --touch -xzf - -C '$REMOTE_MODEL_DIR'"

echo "Remote model files:"
gcloud_compute_ssh --command "find '$REMOTE_MODEL_DIR' -maxdepth 1 -type f -printf '%f %s bytes\n' | sort"
