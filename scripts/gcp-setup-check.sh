#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/gcp-lib.sh"
load_gcp_env

require_cmd gcloud
require_cmd terraform
require_cmd jq
require_gcp_project

echo "GCP project: $GCP_PROJECT"
echo "GCP zone: $GCP_ZONE"
echo "GPU VM: $GCP_INSTANCE ($GCP_MACHINE_TYPE)"
echo "Notebook image: $NOTEBOOK_IMAGE"

check_gcloud_user_auth
check_gcloud_adc_auth

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' >/tmp/tfm-gcloud-account 2>/tmp/tfm-gcloud-auth.err; then
  cat /tmp/tfm-gcloud-auth.err >&2
  echo "Run: gcloud auth login" >&2
  exit 1
fi

ACTIVE_ACCOUNT="$(cat /tmp/tfm-gcloud-account)"
echo "Active gcloud account: ${ACTIVE_ACCOUNT:-unknown}"

if ! gcloud services list --enabled --project "$GCP_PROJECT" --format='value(config.name)' >/dev/null 2>&1; then
  echo "gcloud is installed but needs reauthentication before provisioning." >&2
  echo "Run: gcloud auth login && gcloud auth application-default login" >&2
  exit 1
fi

echo "Local setup check passed. Use scripts/gcp-gpu-up.sh to provision/start the VM."
