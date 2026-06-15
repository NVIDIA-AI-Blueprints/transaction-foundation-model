#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}

load_gcp_env() {
  local root override_gcp_project override_gcp_region override_gcp_zone override_gcp_instance
  local override_gcp_machine_type override_gcp_bucket override_notebook_image
  local override_gpu_test_image override_marimo_port override_jupyter_port override_jupyter_token
  local override_remote_workspace override_gcp_tunnel_through_iap
  local effective_jupyter_port

  root="$(repo_root)"
  override_gcp_project="${GCP_PROJECT-}"
  override_gcp_region="${GCP_REGION-}"
  override_gcp_zone="${GCP_ZONE-}"
  override_gcp_instance="${GCP_INSTANCE-}"
  override_gcp_machine_type="${GCP_MACHINE_TYPE-}"
  override_gcp_bucket="${GCP_BUCKET-}"
  override_notebook_image="${NOTEBOOK_IMAGE-}"
  override_gpu_test_image="${GPU_TEST_IMAGE-}"
  override_marimo_port="${MARIMO_PORT-}"
  override_jupyter_port="${JUPYTER_PORT-}"
  override_jupyter_token="${JUPYTER_TOKEN-}"
  override_remote_workspace="${REMOTE_WORKSPACE-}"
  override_gcp_tunnel_through_iap="${GCP_TUNNEL_THROUGH_IAP-}"

  if [[ -f "$root/.env.gcp" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$root/.env.gcp"
    set +a
  fi

  GCP_PROJECT="${override_gcp_project:-${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}}"
  GCP_REGION="${override_gcp_region:-${GCP_REGION:-us-central1}}"
  GCP_ZONE="${override_gcp_zone:-${GCP_ZONE:-us-central1-f}}"
  GCP_INSTANCE="${override_gcp_instance:-${GCP_INSTANCE:-tfm-gpu-notebook}}"
  GCP_MACHINE_TYPE="${override_gcp_machine_type:-${GCP_MACHINE_TYPE:-a2-highgpu-1g}}"
  GCP_BUCKET="${override_gcp_bucket:-${GCP_BUCKET:-${GCP_PROJECT}-tfm-gpu-artifacts}}"
  NOTEBOOK_IMAGE="${override_notebook_image:-${NOTEBOOK_IMAGE:-nvcr.io/nvidia/nemo:25.09.01}}"
  GPU_TEST_IMAGE="${override_gpu_test_image:-${GPU_TEST_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}}"
  effective_jupyter_port="${override_jupyter_port:-${JUPYTER_PORT:-}}"
  MARIMO_PORT="${override_marimo_port:-${MARIMO_PORT:-${effective_jupyter_port:-8080}}}"
  JUPYTER_PORT="${effective_jupyter_port:-$MARIMO_PORT}"
  JUPYTER_TOKEN="${override_jupyter_token:-${JUPYTER_TOKEN:-tfm-dev}}"
  REMOTE_WORKSPACE="${override_remote_workspace:-${REMOTE_WORKSPACE:-/mnt/tfm/workspace}}"
  GCP_TUNNEL_THROUGH_IAP="${override_gcp_tunnel_through_iap:-${GCP_TUNNEL_THROUGH_IAP:-0}}"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_gcp_project() {
  if [[ -z "${GCP_PROJECT:-}" || "$GCP_PROJECT" == "(unset)" ]]; then
    echo "GCP_PROJECT is not set. Copy .env.gcp.example to .env.gcp or run: gcloud config set project <project>" >&2
    exit 1
  fi
}

check_gcloud_user_auth() {
  local err
  err="$(mktemp)"
  if ! gcloud auth print-access-token >/dev/null 2>"$err"; then
    cat "$err" >&2
    rm -f "$err"
    echo "gcloud user credentials need reauthentication." >&2
    echo "Run: gcloud auth login" >&2
    exit 1
  fi
  rm -f "$err"
}

check_gcloud_adc_auth() {
  local err
  err="$(mktemp)"
  if ! gcloud auth application-default print-access-token >/dev/null 2>"$err"; then
    cat "$err" >&2
    rm -f "$err"
    echo "Terraform uses Application Default Credentials and they need reauthentication." >&2
    echo "Run: gcloud auth application-default login" >&2
    exit 1
  fi
  rm -f "$err"
}

gcloud_ssh_base() {
  local args=(gcloud compute ssh "$GCP_INSTANCE" --zone "$GCP_ZONE" --project "$GCP_PROJECT")
  if [[ "${GCP_TUNNEL_THROUGH_IAP:-0}" == "1" ]]; then
    args+=(--tunnel-through-iap)
  fi
  printf '%q ' "${args[@]}"
}

gcloud_compute_ssh() {
  local args=(gcloud compute ssh "$GCP_INSTANCE" --zone "$GCP_ZONE" --project "$GCP_PROJECT")
  if [[ "${GCP_TUNNEL_THROUGH_IAP:-0}" == "1" ]]; then
    args+=(--tunnel-through-iap)
  fi
  "${args[@]}" "$@"
}

wait_for_gcp_ssh() {
  local attempts="${1:-24}" delay="${2:-5}" err
  err="$(mktemp)"

  echo "Waiting for SSH on $GCP_INSTANCE..."
  for ((i = 1; i <= attempts; i++)); do
    : > "$err"
    if gcloud_compute_ssh --command 'true' -- -o ConnectTimeout=10 >/dev/null 2>"$err"; then
      rm -f "$err"
      echo "SSH is ready."
      return 0
    fi
    if ((i == attempts)); then
      cat "$err" >&2
      rm -f "$err"
      echo "SSH did not become ready on $GCP_INSTANCE after $((attempts * delay)) seconds." >&2
      return 1
    fi
    sleep "$delay"
  done
}

wait_for_vm_bootstrap() {
  local attempts="${1:-72}" delay="${2:-5}" err
  err="$(mktemp)"

  echo "Waiting for VM bootstrap marker..."
  for ((i = 1; i <= attempts; i++)); do
    : > "$err"
    if gcloud_compute_ssh --command 'test -f /opt/tfm/bootstrap.done' -- -o ConnectTimeout=10 >/dev/null 2>"$err"; then
      rm -f "$err"
      echo "VM bootstrap is complete."
      return 0
    fi
    if ((i == attempts)); then
      cat "$err" >&2
      rm -f "$err"
      echo "VM bootstrap did not complete on $GCP_INSTANCE after $((attempts * delay)) seconds." >&2
      echo "Check logs with: gcloud compute ssh $GCP_INSTANCE --zone $GCP_ZONE --project $GCP_PROJECT --command 'sudo tail -n 120 /var/log/syslog'" >&2
      return 1
    fi
    sleep "$delay"
  done
}

gpu_quota_requirement() {
  case "$GCP_MACHINE_TYPE" in
    a2-ultragpu-1g) echo "NVIDIA_A100_80GB_GPUS 1" ;;
    a2-ultragpu-2g) echo "NVIDIA_A100_80GB_GPUS 2" ;;
    a2-ultragpu-4g) echo "NVIDIA_A100_80GB_GPUS 4" ;;
    a2-ultragpu-8g) echo "NVIDIA_A100_80GB_GPUS 8" ;;
    a2-highgpu-1g) echo "NVIDIA_A100_GPUS 1" ;;
    a2-highgpu-2g) echo "NVIDIA_A100_GPUS 2" ;;
    a2-highgpu-4g) echo "NVIDIA_A100_GPUS 4" ;;
    a2-highgpu-8g) echo "NVIDIA_A100_GPUS 8" ;;
    a2-highgpu-16g) echo "NVIDIA_A100_GPUS 16" ;;
    g2-standard-24) echo "NVIDIA_L4_GPUS 2" ;;
    *) echo "" ;;
  esac
}

preflight_gpu_quota() {
  local region requirement metric needed quota_line limit usage available machine_type_err
  region="${GCP_ZONE%-*}"

  machine_type_err="$(mktemp)"
  if ! gcloud compute machine-types describe "$GCP_MACHINE_TYPE" \
    --zone "$GCP_ZONE" \
    --project "$GCP_PROJECT" >/dev/null 2>"$machine_type_err"; then
    cat "$machine_type_err" >&2
    rm -f "$machine_type_err"
    echo "Machine type $GCP_MACHINE_TYPE is not available in $GCP_ZONE." >&2
    exit 1
  fi
  rm -f "$machine_type_err"

  requirement="$(gpu_quota_requirement)"
  if [[ -z "$requirement" ]]; then
    echo "No GPU quota preflight mapping for $GCP_MACHINE_TYPE; continuing to Terraform." >&2
    return 0
  fi

  read -r metric needed <<<"$requirement"
  quota_line="$(gcloud compute regions describe "$region" \
    --project "$GCP_PROJECT" \
    --format=json \
    | jq -r --arg metric "$metric" '.quotas[] | select(.metric == $metric) | [.limit, .usage] | @tsv')"

  if [[ -z "$quota_line" ]]; then
    echo "No quota row for $metric in $region." >&2
    exit 1
  fi

  read -r limit usage <<<"$quota_line"
  available="$(awk -v limit="$limit" -v usage="$usage" 'BEGIN { print limit - usage }')"
  if awk -v available="$available" -v needed="$needed" 'BEGIN { exit !(available < needed) }'; then
    echo "Insufficient quota for $GCP_MACHINE_TYPE in $region: need $needed $metric, available $available." >&2
    exit 1
  fi

  echo "GPU quota preflight passed: $region has $available available $metric; $GCP_MACHINE_TYPE needs $needed."
}
