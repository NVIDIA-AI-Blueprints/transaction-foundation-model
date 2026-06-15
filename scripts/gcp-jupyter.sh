#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/gcp-lib.sh"
load_gcp_env
require_cmd gcloud
require_gcp_project

REMOTE_PORT="${REMOTE_JUPYTER_PORT:-8888}"
LOCAL_PORT="$JUPYTER_PORT"
wait_for_gcp_ssh 12 5
wait_for_vm_bootstrap 72 5

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is not installed yet. Wait for startup-script bootstrap to finish.' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi is not available yet. Wait for GPU driver bootstrap or inspect /var/log/syslog.' >&2
  exit 1
fi
if ! sudo docker run --gpus all --rm "$GPU_TEST_IMAGE" nvidia-smi >/dev/null; then
  echo 'Docker cannot access the GPU. Restart Docker or inspect the NVIDIA Container Toolkit setup.' >&2
  exit 1
fi
mkdir -p /mnt/tfm/workspace /mnt/tfm/data /mnt/tfm/models /mnt/tfm/artifacts
cd "$REMOTE_WORKSPACE"

CHECKPOINT_DIR="/mnt/tfm/models/decoder-foundation-model"
CHECKPOINT_WEIGHTS="\$CHECKPOINT_DIR/model-00001-of-00001.safetensors"
if [[ ! -f "\$CHECKPOINT_DIR/config.json" || ! -f "\$CHECKPOINT_WEIGHTS" ]]; then
  echo "Warning: pretrained checkpoint is missing from \$CHECKPOINT_DIR. Notebooks 04/05 need: scripts/gcp-sync-models.sh" >&2
elif head -n 1 "\$CHECKPOINT_WEIGHTS" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  echo "Warning: checkpoint weights are still a Git LFS pointer in \$CHECKPOINT_DIR. Run scripts/gcp-sync-models.sh after pulling LFS locally." >&2
elif [[ "\$(wc -c < "\$CHECKPOINT_WEIGHTS")" -lt 1000000 ]]; then
  echo "Warning: checkpoint weights look too small in \$CHECKPOINT_DIR. Run scripts/gcp-sync-models.sh." >&2
fi

sudo docker rm -f tfm-jupyter >/dev/null 2>&1 || true
sudo docker pull "$NOTEBOOK_IMAGE"

NVIDIA_DEVICE_ARGS=()
for dev in /dev/nvidia* /dev/nvidia-caps/*; do
  if [[ -e "\$dev" && ! -d "\$dev" ]]; then
    NVIDIA_DEVICE_ARGS+=(--device "\$dev")
  fi
done

sudo docker run --gpus all --rm -d \
  --name tfm-jupyter \
  "\${NVIDIA_DEVICE_ARGS[@]}" \
  --shm-size=8g \
  --ulimit memlock=-1 \
  -p 127.0.0.1:${REMOTE_PORT}:8888 \
  -v "$REMOTE_WORKSPACE":/workspace \
  -v /mnt/tfm/data:/workspace/data \
  -v /mnt/tfm/models:/workspace/models \
  -v /mnt/tfm/artifacts:/workspace/artifacts \
  -w /workspace \
  -e GOOGLE_CLOUD_PROJECT="$GCP_PROJECT" \
  "$NOTEBOOK_IMAGE" \
  bash -lc 'git config --global --add safe.directory /workspace || true; jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="$JUPYTER_TOKEN"'

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if sudo docker exec tfm-jupyter nvidia-smi >/dev/null 2>&1; then
    exit 0
  fi
  sleep 2
done

echo 'Jupyter container started, but it cannot initialize NVML. Restart the container with scripts/gcp-jupyter.sh.' >&2
exit 1
EOF
)

gcloud_compute_ssh --command "$REMOTE_CMD"

echo
echo "Jupyter container started on $GCP_INSTANCE."
echo "Open: http://localhost:${LOCAL_PORT}/?token=${JUPYTER_TOKEN}"
echo "Keeping SSH port forwarding open. Press Ctrl-C to close the tunnel; the remote container will keep running."

gcloud_compute_ssh -- -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}"
