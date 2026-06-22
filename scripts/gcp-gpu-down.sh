#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/gcp-lib.sh"
load_gcp_env
require_cmd gcloud
require_gcp_project

gcloud compute instances stop "$GCP_INSTANCE" \
  --zone "$GCP_ZONE" \
  --project "$GCP_PROJECT"

echo "Stopped $GCP_INSTANCE. Persistent disk and GCS bucket were not deleted."

