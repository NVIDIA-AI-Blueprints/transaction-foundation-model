#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/gcp-lib.sh"
load_gcp_env
require_cmd gcloud
require_gcp_project

SSH_BASE="$(gcloud_ssh_base)"
wait_for_gcp_ssh 12 5

echo "Preparing remote workspace at $REMOTE_WORKSPACE..."
eval "$SSH_BASE --command 'mkdir -p \"$REMOTE_WORKSPACE\" /mnt/tfm/data /mnt/tfm/models /mnt/tfm/artifacts && sudo chown -R \"\$(id -u):\$(id -g)\" \"$REMOTE_WORKSPACE\"'"

echo "Syncing workspace to $GCP_INSTANCE:$REMOTE_WORKSPACE ..."
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='.git' \
  --exclude='.context' \
  --exclude='.env.gcp' \
  --exclude='data/TabFormer' \
  --exclude='data/decoder_corpus' \
  --exclude='data/embeddings' \
  --exclude='data/outputs' \
  --exclude='models/decoder-demo' \
  --exclude='tmp_nb_rerun_results' \
  --exclude='tmp_nb_test_results' \
  -czf - -C "$ROOT" . \
  | eval "$SSH_BASE --command 'tar --no-same-owner --no-same-permissions --touch -xzf - -C \"$REMOTE_WORKSPACE\"'"

echo "Workspace synced."

if ! "$ROOT/scripts/gcp-sync-models.sh"; then
  echo "Warning: workspace sync completed, but model sync was skipped or failed." >&2
  echo "Notebooks 04/05 require: git lfs pull --include='models/decoder-foundation-model/**' --exclude='' && scripts/gcp-sync-models.sh" >&2
fi
