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
check_gcloud_user_auth
check_gcloud_adc_auth
preflight_gpu_quota

TF_DIR="$ROOT/infra/gcp-notebook"
TF_VARS="$TF_DIR/terraform.auto.tfvars.json"

cat > "$TF_VARS" <<JSON
{
  "project_id": "$GCP_PROJECT",
  "region": "$GCP_REGION",
  "zone": "$GCP_ZONE",
  "name": "$GCP_INSTANCE",
  "machine_type": "$GCP_MACHINE_TYPE",
  "bucket_name": "$GCP_BUCKET",
  "notebook_image": "$NOTEBOOK_IMAGE"
}
JSON

terraform -chdir="$TF_DIR" init
terraform -chdir="$TF_DIR" apply -auto-approve

VM_STATUS="$(gcloud compute instances describe "$GCP_INSTANCE" \
  --zone "$GCP_ZONE" \
  --project "$GCP_PROJECT" \
  --format='value(status)')"

if [[ "$VM_STATUS" == "TERMINATED" ]]; then
  echo
  echo "Starting stopped VM $GCP_INSTANCE in $GCP_ZONE..."
  gcloud compute instances start "$GCP_INSTANCE" \
    --zone "$GCP_ZONE" \
    --project "$GCP_PROJECT"
  wait_for_gcp_ssh 36 5
elif [[ "$VM_STATUS" == "RUNNING" ]]; then
  wait_for_gcp_ssh 12 5
elif [[ "$VM_STATUS" != "RUNNING" ]]; then
  echo
  echo "VM $GCP_INSTANCE is currently $VM_STATUS. Wait for it to become RUNNING before starting Jupyter."
fi

echo
echo "VM requested. First boot can take several minutes while drivers and Docker are installed."
echo "Check bootstrap logs with:"
echo "  gcloud compute ssh $GCP_INSTANCE --zone $GCP_ZONE --project $GCP_PROJECT --command 'sudo tail -n 120 /var/log/syslog'"
